/* GEMM kernel —— AME（RISC-V 矩阵扩展）实现。
 *
 * 计算 C[M,N] = A[M,K] · B[N,K]ᵀ，这正是 AME 的原生模式，
 * 且 B 的 [N,K] 就是 PyTorch 权重的 [out_features, in_features] ——
 * 无需任何离线 repack 或运行时转置。
 *
 * ── tile 参数（实测确认，见 tests/ame_smoke.c）────────────────
 *   M ≤ 128    由 xtlenb/xtrlenb = 8192/64 得行数 128
 *   N ≤ 128    累加器 xalenb = 65536 = 128×128×4B
 *   K ≤ 32     BF16 下 K = TRLEN/16 = 512/16
 *   本模型所有 GEMM 的 K(1024/2048/3072) 是 32 的整数倍，
 *   N(1024/2048/3072/151936) 是 128 的整数倍，故 K/N 维不会出现部分 tile。
 *   M 在 decode 路径恒为 1（AME 支持 M<128 的部分 tile）。
 *
 * ── 一条必须遵守的纪律 ───────────────────────────────────────
 * mfmacc 的操作数顺序是 (md, B, A)：第二个放右矩阵，第三个放左矩阵。
 * 写反会得到转置的结果 —— tests/ame_smoke.c 用 M≠N 的小矩阵守住这一点。
 *
 * ── 历史：mzero 曾经不可用 ───────────────────────────────────
 * 早期 QEMU 的 mzero 在 M=128 时只清累加器前 64 行，残留会跨 GEMM 累加。
 * 该缺陷极隐蔽：单次调用完全正确，只有连续调用才暴露 —— 而一次推理要连续
 * 做上百次 GEMM。当时用 mlce32 从内存装零规避（多一次 64 KB 访存）。
 * 2026-08-06 的 QEMU 已修复，故改回原生 mzero。
 * tests/ame_smoke.c 的用例 6 是回归哨兵，若该缺陷复现会立刻失败。
 *
 * ── 精度 ────────────────────────────────────────────────────
 * mfmacc.s.bf16 要求两个源操作数都是 BF16，而激活是 FP32，
 * 故每次调用需把 A 转成 BF16。这不是实现取巧 —— 真实硬件上就是这样算的，
 * AME 只吃 BF16。因此本 kernel 的数值结果会与 kernels_scalar.c（全 FP32）
 * 有系统性差异，token 序列预计比标量版更早分叉，属预期而非缺陷。
 */
#include "bsp/board.h"      /* BOARD_DCACHE_* / BOARD_FENCE：AME 与 L1D 的一致性 */
#include "kernels.h"
#include "qwen3.h"

/* A 的 BF16 暂存。上界 = MAX_SEQ × INTERMEDIATE_SIZE
 * （最大的一次是 down_proj: K=3072；prefill 批量化后 M 最大 MAX_SEQ）。
 * decode 路径 M=1，实际只用到 K 个元素。 */
/* 本 kernel 只做 BF16。INT8 下 K 上限是 64（TRLEN/8），届时另开一份实现。 */
#define AME_TILE_K AME_TILE_K_BF16

/* A 的 BF16 暂存，上界 = MAX_SEQ × INTERMEDIATE_SIZE（1.5 MB BSS）。
 * 当前 forward 逐 token（M=1）只用到 K 个元素，但按 prefill 批量化的上界
 * 预留 —— 否则 M>1 时会静默返回，表现为输出"未写入"。 */
#define AME_A_MAX ((size_t)QWEN3_MAX_SEQ * QWEN3_INTERMEDIATE_SIZE)
static uint16_t g_abuf[AME_A_MAX] __attribute__((aligned(64)));

/* 清零累加器。作用范围由当前的 mtilem × mtilen 决定，故须先 set_tile。 */
static inline void acc_clear(void) {
    __asm__ volatile("mzero acc0");
}

/* mtilem/mtilen/mtilek 是 CSR 0x803/0x804/0x805。
 * 这里直接 csrw（rd=x0）而非 intrinsic 的 csrrw —— 我们不需要旧值。
 * 注意 __riscv_msettilemi 那组立即数版本有 _Static_assert(imm<=0x1f)，
 * 而我们要设 128，只能走寄存器版本。 */
static inline void set_tile(long m, long n, long k) {
    __asm__ volatile("csrw 0x803, %0" :: "r"(m));
    __asm__ volatile("csrw 0x804, %0" :: "r"(n));
    __asm__ volatile("csrw 0x805, %0" :: "r"(k));
}

void qwen3_gemm(float *c, const float *a, const uint16_t *b,
                int M, int K, int N) {
    /* 激活 FP32 -> BF16。AME 的两个源操作数都必须是 BF16。 */
    const size_t n_a = (size_t)M * (size_t)K;
    if (n_a > AME_A_MAX) return;            /* 调用方违约；上界见 AME_A_MAX 注释 */
    for (size_t i = 0; i < n_a; i++) g_abuf[i] = f32_to_bf16(a[i]);

    /* ═══════ AME 进入前的缓存同步 ═══════
     *
     * AME 的访存不经 L1D，直接读写 DDR，硬件不维护一致性。
     * 全部同步集中在这一个函数的边界上 —— AME 只在这里出现，
     * 所以 qwen3.c 一行不用改，197 次 GEMM 调用自动全覆盖。
     *
     * 两件事：
     *   1. g_abuf 是刚用标量/向量写的，还在 L1D 里，必须写回 DDR
     *   2. c 里可能残留 CPU 之前写的脏行，**必须在 AME 动手之前清掉**。
     *      ★ 这一步不能挪到 GEMM 之后：那时 c 的脏行一旦被写回，
     *        正好覆盖掉 AME 刚算出来的结果。顺序错了结果就是错的，
     *        而且错得间歇、难查。
     *      用 flush（写回+失效）而非 inval：此刻 AME 还没写，写回无害，
     *      且首尾不完整的 cache line 不会误伤相邻数据。
     * 权重 b 不用管：只读，CPU 从未写过，DDR 里本来就是对的。 */
    BOARD_DCACHE_CLEAN(g_abuf, n_a * sizeof g_abuf[0]);
    BOARD_DCACHE_FLUSH(c, (size_t)M * (size_t)N * sizeof *c);
    BOARD_FENCE();

    const long a_stride = (long)K * 2;      /* BF16 行字节 stride */
    const long b_stride = (long)K * 2;
    const long c_stride = (long)N * 4;      /* FP32 */

    for (int m0 = 0; m0 < M; m0 += AME_TILE_M) {
        const int mm = (M - m0 < AME_TILE_M) ? (M - m0) : AME_TILE_M;

        for (int n0 = 0; n0 < N; n0 += AME_TILE_N) {
            const int nn = (N - n0 < AME_TILE_N) ? (N - n0) : AME_TILE_N;

            set_tile(mm, nn, AME_TILE_K);
            acc_clear();

            for (int k0 = 0; k0 < K; k0 += AME_TILE_K) {
                const int kk = (K - k0 < AME_TILE_K) ? (K - k0) : AME_TILE_K;
                set_tile(mm, nn, kk);

                const uint16_t *ap = g_abuf + (size_t)m0 * K + k0;
                const uint16_t *bp = b      + (size_t)n0 * K + k0;

                /* A: mtilem×mtilek 左矩阵     B: mtilen×mtilek 右矩阵 */
                __asm__ volatile("mlae16 tr0,(%0),%1"
                                 :: "r"(ap), "r"(a_stride) : "memory");
                __asm__ volatile("mlbe16 tr1,(%0),%1"
                                 :: "r"(bp), "r"(b_stride) : "memory");
                /* acc0 += A · Bᵀ   —— 操作数顺序 (md, B, A) */
                __asm__ volatile("mfmacc.s.bf16 acc0,tr1,tr0");
            }

            /* 写出 mtilem×mtilen 的 FP32 结果。K 维已累加完，此处 k 值无关。 */
            set_tile(mm, nn, AME_TILE_K);
            float *cp = c + (size_t)m0 * N + n0;
            __asm__ volatile("msce32 acc0,(%0),%1"
                             :: "r"(cp), "r"(c_stride) : "memory");
        }
    }

    __asm__ volatile("mrelease");

    /* ═══════ AME 退出后的缓存同步 ═══════
     *
     * fence 先让 AME 的写落到 DDR（硬件团队确认 fence 覆盖 AME 的访存缓冲，
     * fence 退休即代表已落盘），再失效 L1D 里 c 的旧拷贝，
     * 否则后面的 RVV/标量读会命中旧值。
     *
     * 这里用 inval 而不是 flush 是安全的：进入前已经 flush 过 c，
     * AME 计算期间 CPU 没有写过它，所以此刻 c 在 L1D 里只可能有
     * 预取/推测带进来的**干净**拷贝，丢弃不会损失任何数据。
     * 反过来若这里用 flush，就会把旧值写回、覆盖 AME 的结果。 */
    BOARD_FENCE();
    BOARD_DCACHE_INVAL(c, (size_t)M * (size_t)N * sizeof *c);
    BOARD_FENCE();
}

const char *qwen3_kernel_name(void) { return "ame"; }
