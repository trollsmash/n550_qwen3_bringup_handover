/* GEMM kernel —— 纯 RVV 实现，但**读 tile-major 权重**。
 *
 * 存在的唯一目的：与 kernels_ame.c 做严格的 A/B 对比。
 *
 * ── 为什么不直接用 kernels_rvvbf16.c ──────────────────────────
 * 那一份按 row-major 寻址。拿它跟 AME 比，测出来的差距里混着两个变量：
 * 计算单元不同（RVV vs AME）、权重布局不同（row vs tile）。
 * 而布局本身就值好几倍 —— tile-major 能命中 MLSU 的行合并，
 * row-major 拿不到。两个变量搅在一起，结论说不清楚。
 *
 * 本文件读**同一份 tile-major 权重**，于是唯一的变量就只剩计算单元。
 *
 * ── tile-major 对 RVV 是不利的，这是有意保留的 ────────────────
 * tile 的形状（128 行 × 32 列）是照着 AME 的 TR 寄存器定的：
 * 一条 mfmacc 吞下整块，8 KB 一次读完。
 *
 * RVV 没有这种"整块吃"的指令，只能把一个 tile 拆成 128 个长度 32 的点积，
 * 每个都要单独归约。K=32 时 e16m1（vlmax=64）也只填一半，
 * 而且每 32 个乘加就要付一次 vfredusum 的代价。
 *
 * 这个劣势不是实现没写好，是**布局与指令形态不匹配的必然结果** ——
 * 恰恰是对比想要量出来的东西，所以不绕开它。
 *
 * 用 KERNEL=rvvtile 选择本实现。
 */
#include <riscv_vector.h>

#include "kernels.h"
#include "qwen3.h"
#include "bsp/board.h"

/* 与 kernels_ame.c 保持一致的 tile 形状 —— 权重就是按这个尺寸排的 */
#define TILE_N 128
#define TILE_K 32

/* 行优先路径每算这么多个输出，就用一条 vse32 刷回 c。
 * 分块是因为 N 可以很大（lm_head 是 151936），整行放不下栈。
 * 取 128 与 TILE_N 一致，e32m8 下（vlmax=256）一次存完。 */
#define ROW_CHUNK 128

/* A 的 BF16 暂存。上界与 AME 版一致：MAX_BATCH × INTERMEDIATE_SIZE。
 * CLP 下放窗口内（只有 RVV 碰，不必换视图；AME 不参与本 kernel）。 */
#define RVT_A_MAX ((size_t)QWEN3_MAX_BATCH * QWEN3_INTERMEDIATE_SIZE)
#ifdef BOARD_CLP
static uint16_t *const g_abuf = (uint16_t *)BOARD_CLP_ABUF_ADDR;
#else
static uint16_t g_abuf[RVT_A_MAX] __attribute__((aligned(64)));
#endif

/* tile-major 下 (n0,k0) 号 tile 的元素偏移 —— 与 kernels_ame.c 的
 * b_tile_off 逐字相同。两边必须一致，否则读到的是别的 tile。 */
static inline size_t tile_off(int n0, int k0, int K) {
    return ((size_t)(n0 / TILE_N) * ((size_t)K / TILE_K)
            + (size_t)(k0 / TILE_K)) * (TILE_N * TILE_K);
}

/* 向 qwen3_init 认领 tile-major —— 默认的弱符号实现只接受行优先，
 * 不覆盖它的话加载权重会直接被判为 QWEN3_ERR_LAYOUT。
 *
 * 检查项与 kernels_ame.c 完全一致：tile 尺寸要对得上，且本模型用到的
 * 所有 N、K 都得整除 tile 尺寸 —— 不整除时最后一块 tile 不满，
 * 后续所有块的起始偏移全部错开，而 GEMM 照样算得出结果，
 * 只是每个数都取自错误的位置。 */
int qwen3_set_weight_layout(int layout, int tile_n, int tile_k) {
    /* 本 kernel 的 qwen3_gemm 只会按 tile 寻址，喂行优先的权重必然算错，
     * 所以这里直接拒绝 —— 让它在加载阶段失败，好过输出一堆垃圾。 */
    if (layout != QW3M_LAYOUT_TILE) return -1;
    if (tile_n != TILE_N || tile_k != TILE_K) return -1;

    const int ns[] = { QWEN3_Q_DIM, QWEN3_KV_DIM,
                       QWEN3_HIDDEN_SIZE, QWEN3_INTERMEDIATE_SIZE };
    const int ks[] = { QWEN3_HIDDEN_SIZE, QWEN3_Q_DIM, QWEN3_INTERMEDIATE_SIZE };
    for (unsigned i = 0; i < sizeof ns / sizeof ns[0]; i++)
        if (ns[i] % TILE_N) return -1;
    for (unsigned i = 0; i < sizeof ks / sizeof ks[0]; i++)
        if (ks[i] % TILE_K) return -1;
    return 0;
}

void qwen3_gemm(float *c, const float *a, const uint16_t *b,
                int M, int K, int N) {
    const size_t n_a = (size_t)M * (size_t)K;
    if (n_a > RVT_A_MAX) return;                 /* 调用方违约 */

    /* 激活 FP32 -> BF16。与 AME 版同样必须转 —— vfwmaccbf16 的两个源
     * 都是 BF16。用 RVV 写而不是标量循环：CLP 下 g_abuf 在只有 RVV
     * 能访存的窗口里，标量碰它会触发 PMA 异常。 */
    {
        size_t rest = n_a;
        const float *src = a;
        uint16_t *dst = g_abuf;
        for (size_t vl; rest > 0; rest -= vl, src += vl, dst += vl) {
            vl = __riscv_vsetvl_e32m8(rest);
            __riscv_vse16_v_u16m4(dst,
                __riscv_vreinterpret_v_bf16m4_u16m4(
                    __riscv_vfncvtbf16_f_f_w_bf16m4(
                        __riscv_vle32_v_f32m8(src, vl), vl)), vl);
        }
    }
    BOARD_FENCE();

    const size_t vlmax1 = __riscv_vsetvlmax_e32m1();

    for (int m = 0; m < M; m++) {
        const uint16_t *arow = g_abuf + (size_t)m * K;
        float *crow = c + (size_t)m * N;

        /* 按 tile 遍历，顺序与 AME 版一致：n0 在外、k0 在内。
         * 这样相邻的 k0 迭代读到相邻的 tile，DDR 侧有行局部性 ——
         * 这一点对 RVV 同样有效，是布局带来的、与指令无关的好处。 */
        for (int n0 = 0; n0 < N; n0 += TILE_N) {

            /* ★ 累加值放栈上，不放 c。
             *
             * CLP 下 c 在只有 RVV 能访存的窗口里。写成 crow[n0+i] += ... 会
             * 编译成 flw/fadd/fsw —— 两条标量访存，上板立刻 PMA 异常
             * （实测 mcause=5、mepc 指向那条 flw、mtval 落在窗口内）。
             * 栈在普通 DDR 视图，标量读写合法。
             *
             * 顺带删掉了原先对 crow 的清零：累加不再落在 c 上，那一步多余了。
             * 改完的结构正好与 AME 版对齐 —— mzero、在 AR 里累加、msce32 存一次。 */
            float acc128[TILE_N];
            for (int i = 0; i < TILE_N; i++) acc128[i] = 0.0f;

            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                const uint16_t *tile = b + tile_off(n0, k0, K);
                const uint16_t *aseg = arow + k0;        /* 本 tile 对应的 32 个 A */

                /* ★ 这里是 RVV 相对 AME 最吃亏的地方。
                 * AME 一条 mfmacc 把整块 128×32 吞下去；
                 * RVV 只能逐行做长度 32 的点积，每行付一次归约。
                 * e16m1 的 vlmax 是 64，K=32 时只填一半 —— 布局是按
                 * AME 的 TR 定的，对向量单元并不合身。 */
                for (int i = 0; i < TILE_N; i++) {
                    const uint16_t *brow = tile + (size_t)i * TILE_K;

                    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, TILE_K);
                    vbfloat16m1_t va =
                        __riscv_vle16_v_bf16m1((const __bf16 *)aseg, TILE_K);
                    vbfloat16m1_t vb =
                        __riscv_vle16_v_bf16m1((const __bf16 *)brow, TILE_K);
                    acc = __riscv_vfwmaccbf16_vv_f32m2(acc, va, vb, TILE_K);

                    vfloat32m1_t zero =
                        __riscv_vfmv_v_f_f32m1(0.0f, vlmax1);
                    vfloat32m1_t s =
                        __riscv_vfredusum_vs_f32m2_f32m1(acc, zero, TILE_K);
                    acc128[i] += __riscv_vfmv_f_s_f32m1_f32(s);
                }
            }

            /* 一次写回窗口。vse32 是向量访存，窗口允许。
             * 长度按剩余量夹一下：布局检查保证 N 是 TILE_N 的整数倍，
             * 但真越界的话是覆盖 c 之后的内存，不会报错 —— 不值得省这个分支。 */
            {
                size_t rest = (size_t)((N - n0 < TILE_N) ? (N - n0) : TILE_N);
                const float *p = acc128;
                float *q = crow + n0;
                for (size_t vl; rest > 0; rest -= vl, p += vl, q += vl) {
                    vl = __riscv_vsetvl_e32m8(rest);
                    __riscv_vse32_v_f32m8(q, __riscv_vle32_v_f32m8(p, vl), vl);
                }
            }
        }
    }
    BOARD_FENCE();
}

/* lm_head 用的 embedding 表是行优先的（它同时要被按行查表），
 * 不走 tile 布局 —— 与 kernels_ame.c 的 qwen3_gemm_row 同理。 */
void qwen3_gemm_row(float *c, const float *a, const uint16_t *b,
                    int M, int K, int N) {
    const size_t n_a = (size_t)M * (size_t)K;
    if (n_a > RVT_A_MAX) return;

    {
        size_t rest = n_a;
        const float *src = a;
        uint16_t *dst = g_abuf;
        for (size_t vl; rest > 0; rest -= vl, src += vl, dst += vl) {
            vl = __riscv_vsetvl_e32m8(rest);
            __riscv_vse16_v_u16m4(dst,
                __riscv_vreinterpret_v_bf16m4_u16m4(
                    __riscv_vfncvtbf16_f_f_w_bf16m4(
                        __riscv_vle32_v_f32m8(src, vl), vl)), vl);
        }
    }
    BOARD_FENCE();

    const size_t vlmax8 = __riscv_vsetvlmax_e32m8();
    const size_t vlmax1 = __riscv_vsetvlmax_e32m1();

    for (int m = 0; m < M; m++) {
        const uint16_t *arow = g_abuf + (size_t)m * K;
        float *crow = c + (size_t)m * N;

        /* 行优先：整个 K 维连续，可以一路吃满向量宽度 ——
         * 与上面的 tile 版对照，正好说明布局对向量单元的影响。
         *
         * 与 qwen3_gemm 同样的理由：结果先落栈，再成批写回 —— 直接写
         * crow[n] 是一条标量 fsw，CLP 下碰窗口会触发 PMA 异常。 */
        for (int n0 = 0; n0 < N; n0 += ROW_CHUNK) {
            const int nn = (N - n0 < ROW_CHUNK) ? (N - n0) : ROW_CHUNK;
            float buf[ROW_CHUNK];

            for (int t = 0; t < nn; t++) {
                const uint16_t *brow = b + (size_t)(n0 + t) * K;
                vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax8);
                size_t rest = (size_t)K;
                const uint16_t *pa = arow, *pb = brow;
                for (size_t vl; rest > 0; rest -= vl, pa += vl, pb += vl) {
                    vl = __riscv_vsetvl_e32m8(rest);
                    acc = __riscv_vfwmaccbf16_vv_f32m8(acc,
                              __riscv_vle16_v_bf16m4((const __bf16 *)pa, vl),
                              __riscv_vle16_v_bf16m4((const __bf16 *)pb, vl), vl);
                }
                vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax1);
                buf[t] = __riscv_vfmv_f_s_f32m1_f32(
                             __riscv_vfredusum_vs_f32m8_f32m1(acc, zero, vlmax8));
            }

            {
                size_t left = (size_t)nn;
                const float *p = buf;
                float *q = crow + n0;
                for (size_t vl; left > 0; left -= vl, p += vl, q += vl) {
                    vl = __riscv_vsetvl_e32m8(left);
                    __riscv_vse32_v_f32m8(q, __riscv_vle32_v_f32m8(p, vl), vl);
                }
            }
        }
    }
    BOARD_FENCE();
}

const char *qwen3_kernel_name(void) { return "rvvtile"; }
