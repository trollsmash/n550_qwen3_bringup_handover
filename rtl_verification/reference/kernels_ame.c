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

/* ═══════ AME 指令封装 ═══════
 *
 * 矩阵寄存器名（tr0..tr3 / acc0..acc3）在编码里是立即数字段，只能是
 * 字面量，所以走宏而不是函数参数。
 *
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║  HW-WAR-001   硬件缺陷规避 —— RTL 修复后应当整体回退              ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * 【缺陷】AME 指令与它前面那条**非 AME 且无数据相关**的指令之间会重叠
 *   执行，导致配置来不及生效。2026-09-01 真机实测：mtilem 没配进去，
 *   随后第一条 mlae16 报 mcause=2（mtval=0x04a7040b，build 0x09011236）。
 *
 * 【规避】把 mv 与 AME 指令封进**同一个 asm 块**。两个作用缺一不可：
 *   - mv 写的正是 AME 指令读的寄存器，构成 RAW 依赖，硬件必须等；
 *   - 同块内编译器不能插指令，也不会把 AME 指令变成 分支或跳转 的落点
 *     （扫描确认有几处 AME 指令是跳转目标，那种位置无论怎样分配寄存器
 *      都不可能产生数据相关）。
 *
 * 【覆盖范围】set_tile 的三条 csrw、全部矩阵 load/store。
 *   **mzero / mfmacc 不读任何通用寄存器，无法用此法规避**，只能等 RTL。
 *
 * 【回退】编译时加 -DAME_NO_HW_WAR_001，各处 #else 分支即原始写法。
 *   回退后请重新跑一遍 AME 指令邻接扫描确认无残留风险点。
 *
 * ★ 因此地址与 stride 钉死在 a0 / a1。
 *
 *   2026-09-01 真机实测（build 0x09011236）：放开约束后编译器分配到
 *   (rs1=a4, rs2=a0)，主核报 **mcause=2 非法指令**，mtval=0x04a7040b；
 *   而钉死 (a0,a1) 的 L2/L3 里同一条 mlae16(0x04b5040b) 跑得好好的。
 *   两份编码只差寄存器号，opcode/funct 完全一致。
 *
 *   限制在**主核**而非 AMU：DEC HLD 写明 rs1/rs2 不从指令字提取，
 *   而是主核读出寄存器**值**后透传给 AMU，所以 AMU 根本不看编号。
 *   主核那部分 RTL 不在我们手里，只能规避。
 *
 *   曾经一度以为这条约束多余（当时把故障归给 mrelease 破坏 I-cache）——
 *   那是**另一个独立的 bug**，两个都真实存在。别再解开第二次。
 *
 *   代价只是每条 load/store 前多一两条 mv，相对 8 KB 的访存可以忽略；
 *   三路并行照常工作，编译器负责把各自的地址搬进 a0。 */
#ifndef AME_NO_HW_WAR_001
/* ── HW-WAR-001 ── mv 与矩阵指令同块：制造 RAW 依赖 + 阻止编译器
 * 在中间插指令、也阻止矩阵指令成为 分支或跳转 的落点。 */
#define AME_LOAD_A(tr, ptr, stride)                                          \
    __asm__ volatile("mv a0, %0\n\t"                                         \
                     "mv a1, %1\n\t"                                         \
                     "mlae16 " tr ",(a0),a1"                                 \
                     :: "r"(ptr), "r"((long)(stride))                        \
                      : "a0", "a1", "memory")
#define AME_LOAD_B(tr, ptr, stride)                                          \
    __asm__ volatile("mv a0, %0\n\t"                                         \
                     "mv a1, %1\n\t"                                         \
                     "mlbe16 " tr ",(a0),a1"                                 \
                     :: "r"(ptr), "r"((long)(stride))                        \
                      : "a0", "a1", "memory")
#else
#define AME_LOAD_A(tr, ptr, stride)                                          \
    do { register const void *_p __asm__("a0") = (const void *)(ptr);        \
         register long _s __asm__("a1") = (long)(stride);                    \
         __asm__ volatile("mlae16 " tr ",(%0),%1"                            \
                          :: "r"(_p), "r"(_s) : "memory"); } while (0)
#define AME_LOAD_B(tr, ptr, stride)                                          \
    do { register const void *_p __asm__("a0") = (const void *)(ptr);        \
         register long _s __asm__("a1") = (long)(stride);                    \
         __asm__ volatile("mlbe16 " tr ",(%0),%1"                            \
                          :: "r"(_p), "r"(_s) : "memory"); } while (0)
#endif
/* acc += B · Aᵀ —— 操作数顺序是 (md, B, A)，写反会得到转置结果。 */
#define AME_MFMACC(acc, trb, tra)                                            \
    __asm__ volatile("mfmacc.s.bf16 " acc "," trb "," tra)
#ifndef AME_NO_HW_WAR_001
#define AME_STORE_C(acc, ptr, stride)                                        \
    __asm__ volatile("mv a0, %0\n\t"                                         \
                     "mv a1, %1\n\t"                                         \
                     "msce32 " acc ",(a0),a1"                                \
                     :: "r"(ptr), "r"((long)(stride))                        \
                      : "a0", "a1", "memory")
#else
#define AME_STORE_C(acc, ptr, stride)                                        \
    do { register void *_p __asm__("a0") = (void *)(ptr);                    \
         register long _s __asm__("a1") = (long)(stride);                    \
         __asm__ volatile("msce32 " acc ",(%0),%1"                           \
                          :: "r"(_p), "r"(_s) : "memory"); } while (0)
#endif

/* 清零累加器。作用范围由当前的 mtilem × mtilen 决定，故须先 set_tile。 */
static inline void acc_clear(void) {
    __asm__ volatile("mzero acc0");
}
/* 三路并行用前三个累加器。第四个 acc3 用不上 —— mfmacc 的两个源都必须
 * 在 tile 寄存器里，而 A 常驻一个、只剩三个给 B，路数就被卡在 3。
 * 硬要凑第四路就得让某个 B 复用已占用的 tr，那要等对应的 mfmacc 读完，
 * 引入依赖反而串行化。 */
static inline void acc_clear3(void) {
    __asm__ volatile("mzero acc0");
    __asm__ volatile("mzero acc1");
    __asm__ volatile("mzero acc2");
}

/* mtilem/mtilen/mtilek 是 CSR 0x803/0x804/0x805。
 * 这里直接 csrw（rd=x0）而非 intrinsic 的 csrrw —— 我们不需要旧值。
 * 注意 __riscv_msettilemi 那组立即数版本有 _Static_assert(imm<=0x1f)，
 * 而我们要设 128，只能走寄存器版本。 */
static inline void set_tile(long m, long n, long k) {
#ifndef AME_NO_HW_WAR_001
    /* ── HW-WAR-001 ── 见文件顶部说明。RTL 修好后 -DAME_NO_HW_WAR_001 回退。
     * 每条 csrw 前挂一条写 t0 的 mv，且三组封在同一个 asm 块里：
     * mv 写 t0、csrw 读 t0 构成 RAW 依赖，硬件不能把两者重叠；
     * 同块也让编译器无法在中间插入调度出来的标量指令。 */
    __asm__ volatile("mv   t0, %0\n\t"
                     "csrw 0x803, t0\n\t"
                     "mv   t0, %1\n\t"
                     "csrw 0x804, t0\n\t"
                     "mv   t0, %2\n\t"
                     "csrw 0x805, t0"
                     :: "r"(m), "r"(n), "r"(k) : "t0");
#else
    __asm__ volatile("csrw 0x803, %0" :: "r"(m));
    __asm__ volatile("csrw 0x804, %0" :: "r"(n));
    __asm__ volatile("csrw 0x805, %0" :: "r"(k));
#endif
}

/* 权重是否为 tile-major。由 qwen3_init 解析 header 后设定。 */
static int g_b_tiled;

/* 覆盖 qwen3.c 里的弱符号：本 kernel 两种布局都认。 */
int qwen3_set_weight_layout(int layout, int tile_n, int tile_k) {
    if (layout == QW3M_LAYOUT_ROW) { g_b_tiled = 0; return 0; }
    if (layout != QW3M_LAYOUT_TILE) return -1;
    /* tile 尺寸必须与本 kernel 编译时用的一致，否则地址算出来是错位的。 */
    if (tile_n != AME_TILE_N || tile_k != AME_TILE_K) return -1;
    /* tile-major 的地址计算假设 N、K 都能整除 tile 尺寸 —— 导出脚本已保证，
     * 这里把本模型出现的所有 GEMM 维度一次性复核，省得每次调用都查。
     * 不整除时最后一块 tile 不满，块的起始偏移就会全部错开。 */
    const int ns[] = { QWEN3_Q_DIM, QWEN3_KV_DIM,
                       QWEN3_HIDDEN_SIZE, QWEN3_INTERMEDIATE_SIZE };
    const int ks[] = { QWEN3_HIDDEN_SIZE, QWEN3_Q_DIM, QWEN3_INTERMEDIATE_SIZE };
    for (unsigned i = 0; i < sizeof ns / sizeof ns[0]; i++)
        if (ns[i] % AME_TILE_N) return -1;
    for (unsigned i = 0; i < sizeof ks / sizeof ks[0]; i++)
        if (ks[i] % AME_TILE_K) return -1;
    g_b_tiled = 1;
    return 0;
}

/* tile-major 下 (n0, k0) 号 tile 的元素偏移。
 * tile 按 n0 在外、k0 在内排列 —— 与下面的循环顺序一致，
 * 使相邻的 k0 迭代读到相邻的 tile，DDR 侧也就有了行局部性。 */
static inline size_t b_tile_off(int n0, int k0, int K) {
    return ((size_t)(n0 / AME_TILE_N) * ((size_t)K / AME_TILE_K)
            + (size_t)(k0 / AME_TILE_K)) * (AME_TILE_N * AME_TILE_K);
}

static void gemm_impl(int tiled, float *c, const float *a, const uint16_t *b,
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
    /* 行优先要跨过整行 K 个元素才到 tile 的下一行（读 64 B 跳 2 KB）；
     * tile-major 下一行紧接着上一行，跨度就是 tile 的行宽 64 B，
     * 整个 tile 是连续的 8 KB —— AXI 侧这才有合并成长 burst 的可能。 */
    const long b_stride = tiled ? (long)AME_TILE_K * 2 : (long)K * 2;
    const long c_stride = (long)N * 4;      /* FP32 */

    for (int m0 = 0; m0 < M; m0 += AME_TILE_M) {
        const int mm = (M - m0 < AME_TILE_M) ? (M - m0) : AME_TILE_M;

        /* A 的地址只跟 k0 走，与 n0 无关 —— 三路并行正是拿这一点做文章。
         * B 的按布局取，n 参数化以便三路各取各的 tile。 */
        #define AP(kk0)       (g_abuf + (size_t)m0 * K + (kk0))
        #define BPn(n_, kk0)  (tiled ? b + b_tile_off((n_), (kk0), K)      \
                                     : b + (size_t)(n_) * K + (kk0))

        int n0 = 0;

#ifndef AME_NO_PIPELINE
        /* ═══════ n 维三路并行 ═══════
         *
         * 一份 A 喂三个不同的 B tile。三者算的是输出矩阵**互不重叠的列区间**，
         * 各自累加各自的 k，最后写到 C 的三个相邻位置 —— n 不是归约维，
         * 所以累加器之间不需要相加。
         *
         * 收益不在数据复用：decode 时 M=1，每个权重元素只与唯一的激活值
         * 相乘一次，复用度恒为 1，软件层面无解（要提高只能增大 M）。
         * 省下的是 A 的重复加载（N/128 次降到 1/3），而 mtilem=1 时
         * A 每次才 64 字节，那点量占总访存 0.5%。
         *
         * 真正的收益是**把两个 load queue 填满**：原来每个 k 块只发
         * 一大一小两笔，小的那笔几十周期就回来了，第二个 queue 大半时间
         * 空着；现在三笔大的排队，通道持续满载。
         *
         * 四笔 load 先连续发出、再统一消费 —— 它们之间没有数据依赖，
         * 这样排能让硬件的乱序窗口一次看到全部四笔，不必指望它跨过
         * 中间的 mfmacc 去发现后面的 load。
         *
         * 只在 K 整除 32 时启用：否则最后一块的 mtilek 不同，而 mtilek 是
         * CSR、对所有 tile 寄存器全局生效，中途改会毁掉已经装好的那几个。 */
        if (K % AME_TILE_K == 0 && N >= 3 * AME_TILE_N) {
            /* ★ CSR 提到组循环之外。所有三路组的形状都是 (mm,128,32)，
             *   而硬件上改写 mtilem/mtilen/mtilek 会**强制指令排序**、
             *   切断乱序窗口 —— 每组重设一次等于每 384 列插一道屏障，
             *   正好卡在「上一组算完、下一组访存该启动」的接缝上。
             *   整个 GEMM 设一次就够。 */
            set_tile(mm, AME_TILE_N, AME_TILE_K);

            for (; n0 + 3 * AME_TILE_N <= N; n0 += 3 * AME_TILE_N) {
                acc_clear3();

                for (int k0 = 0; k0 < K; k0 += AME_TILE_K) {
                    /* 乘加夹在权重加载之间：操作数一就绪就开算，
                     * 不必等三块 B 全部到齐。
                     *
                     * ★ 别把四笔 load 堆在最前面。只有两个 load queue，
                     *   第三四笔会卡在发射阶段，**连带堵住排在它们后面的
                     *   mfmacc** —— 哪怕那些 mfmacc 的操作数早已就绪。
                     *   那是结构冒险而非数据依赖，乱序执行救不了已经堵在
                     *   发射端的指令。
                     *   交错之后稳态下只有两笔 load 在飞，正好一个 queue
                     *   一笔，中间的 mfmacc 不占 load queue。 */
                    AME_LOAD_A("tr0", AP(k0), a_stride);
                    AME_LOAD_B("tr1", BPn(n0,                    k0), b_stride);
                    AME_MFMACC("acc0", "tr1", "tr0");
                    AME_LOAD_B("tr2", BPn(n0 +     AME_TILE_N,   k0), b_stride);
                    AME_MFMACC("acc1", "tr2", "tr0");
                    AME_LOAD_B("tr3", BPn(n0 + 2 * AME_TILE_N,   k0), b_stride);
                    AME_MFMACC("acc2", "tr3", "tr0");
                }

                /* 不重设 CSR：k 循环内一条 csrw 都没有，msce32 要的
                 * mtilem/mtilen 仍是进入这一组时的值。多设一次不改变
                 * 任何行为，只白插三道屏障。 */
                float *cp = c + (size_t)m0 * N + n0;
                /* 三个 msce32 写到互不重叠的列区间，可以连续发出。 */
                AME_STORE_C("acc0", cp,                      c_stride);
                AME_STORE_C("acc1", cp +     AME_TILE_N,     c_stride);
                AME_STORE_C("acc2", cp + 2 * AME_TILE_N,     c_stride);
            }
        }
#endif

        /* 余数：不足三路的尾巴，以及 K 不整除 32 时的全部。
         * 走单路 + k 维双缓冲 —— 这条路径已过 QEMU 回归，拿它兜底
         * 比为边界另写一份三路变体稳妥。 */
        for (; n0 < N; n0 += AME_TILE_N) {
            const int nn = (N - n0 < AME_TILE_N) ? (N - n0) : AME_TILE_N;

            set_tile(mm, nn, AME_TILE_K);
            acc_clear();

            #define BP(kk0) BPn(n0, (kk0))

            const int nk = K / AME_TILE_K;      /* 整块数 */

#ifndef AME_NO_PIPELINE
            /* ═══════ k 维双缓冲 ═══════
             *
             * 串行写法是「载 A、载 B、算」，算的时候访存通道整个空着。
             * 这里用两组 tile 寄存器交替：算 tr0/tr1 的同时把下一块
             * 预取进 tr2/tr3，通道不再有空窗。
             *
             * 收益上限是「计算期间通道空闲的比例」—— 它不减少一个字节的
             * 权重访存，只是把洞填上。burst 之前访存慢到计算完全被淹没，
             * 这么写没有意义；burst 之后计算的占比浮上来才值得。
             *
             * 只在 K 能被 32 整除且至少两块时启用：否则最后一块的 mtilek
             * 不同，而 mtilek 是 CSR、对当前所有 tile 寄存器全局生效，
             * 预取下一块就会改掉当前块的形状。整除时 kk 恒为 32，
             * set_tile 一次即可，连带省下每次迭代的三条 CSR 写。 */
            if (nk >= 2 && K % AME_TILE_K == 0) {
                AME_LOAD_A("tr0", AP(0), a_stride);
                AME_LOAD_B("tr1", BP(0), b_stride);

                for (int i = 0; i < nk; i += 2) {
                    const int k1 = (i + 1) * AME_TILE_K;
                    const int k2 = (i + 2) * AME_TILE_K;
                    /* 偶数块在 tr0/tr1：先把奇数块预取进 tr2/tr3 再算 */
                    if (i + 1 < nk) {
                        AME_LOAD_A("tr2", AP(k1), a_stride);
                        AME_LOAD_B("tr3", BP(k1), b_stride);
                    }
                    AME_MFMACC("acc0", "tr1", "tr0");

                    if (i + 1 < nk) {
                        if (i + 2 < nk) {
                            AME_LOAD_A("tr0", AP(k2), a_stride);
                            AME_LOAD_B("tr1", BP(k2), b_stride);
                        }
                        AME_MFMACC("acc0", "tr3", "tr2");
                    }
                }
            } else
#endif
            {
                /* 串行路径。K 不整除 32 时的正确性靠它兜底；
                 * 真机上双缓冲若出问题，-DAME_NO_PIPELINE 一键退回这里。 */
                for (int k0 = 0; k0 < K; k0 += AME_TILE_K) {
                    const int kk = (K - k0 < AME_TILE_K) ? (K - k0) : AME_TILE_K;
                    set_tile(mm, nn, kk);
                    AME_LOAD_A("tr0", AP(k0), a_stride);
                    AME_LOAD_B("tr1", BP(k0), b_stride);
                    AME_MFMACC("acc0", "tr1", "tr0");
                }
            }
            #undef BP

            /* 写出 mtilem×mtilen 的 FP32 结果。
             * 同样不重设 CSR —— 双缓冲那条路径 k 循环内没动过；
             * 串行 fallback 那条最后一次 set_tile 的 mtilek 可能是尾块的
             * kk，但 msce32 只看 mtilem/mtilen，与 k 无关。 */
            float *cp = c + (size_t)m0 * N + n0;
            AME_STORE_C("acc0", cp, c_stride);
        }
        #undef AP
        #undef BPn
    }

    ame_release();   /* 原为 mrelease，见 board.h 的说明 */

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

/* 绝大多数 GEMM 走这里：权重按什么布局存放，由 qwen3_set_weight_layout 决定。 */
void qwen3_gemm(float *c, const float *a, const uint16_t *b,
                int M, int K, int N) {
    gemm_impl(g_b_tiled, c, a, b, M, K, N);
}

/* 权重确定是行优先的那几处走这里 —— 目前只有 lm_head，它复用 embed_tokens，
 * 而 embed_tokens 从不重排（要按 token id 整行查表）。
 * 这类调用不能受全局布局影响，否则每个数都取自错误的位置。 */
void qwen3_gemm_row(float *c, const float *a, const uint16_t *b,
                    int M, int K, int N) {
    gemm_impl(0, c, a, b, M, K, N);
}

const char *qwen3_kernel_name(void) { return "ame"; }
