/* 按 AME tile 尺寸分块的标量 GEMM。
 *
 * 目的不是提速，而是**先把 tiling 逻辑验证对**，把它与"AME 指令怎么用"
 * 这两个未知数分开。等换成真正的 intrinsics 时，出问题必定在指令用法上。
 *
 * 结构完全对应 AME 的数据流：
 *
 *   for 每个输出 tile (128×128)          <- 一个 AR 累加器
 *       AR = 0
 *       for 每个 K 分块 (32)             <- 每次装载一对 TR
 *           装载 A_tile[M≤128, 32] -> TR0
 *           装载 B_tile[N≤128, 32] -> TR1
 *           AR += A_tile · B_tileᵀ       <- 一条 MMA 指令
 *       AR -> C
 *
 * 关键性质：累加在 AR（FP32）里跨 K 分块进行，与朴素实现的顺序累加不同，
 * 因此结果会有微小浮点差异 —— 这正是我们不追求 bit-exact 的原因。
 */
#include "kernels.h"
#include "qwen3.h"      /* bf16_to_f32 */

#define TM AME_TILE_M
#define TN AME_TILE_N
#define TK AME_TILE_K_BF16

void qwen3_gemm(float *c, const float *a, const uint16_t *b,
                int M, int K, int N) {
    /* AR 累加器：128×128 FP32 = 64KB，与硬件 AR 容量一致。 */
    static float ar[TM * TN];

    for (int m0 = 0; m0 < M; m0 += TM) {
        const int mm = (M - m0 < TM) ? (M - m0) : TM;   /* 部分 tile：AME 支持 */

        for (int n0 = 0; n0 < N; n0 += TN) {
            const int nn = (N - n0 < TN) ? (N - n0) : TN;

            for (int i = 0; i < mm * nn; i++) ar[i] = 0.0f;

            for (int k0 = 0; k0 < K; k0 += TK) {
                const int kk = (K - k0 < TK) ? (K - k0) : TK;

                /* 一次 MMA：AR[mm,nn] += A[mm,kk] · B[nn,kk]ᵀ */
                for (int i = 0; i < mm; i++) {
                    const float *arow = a + (size_t)(m0 + i) * (size_t)K + k0;
                    float *arrow = ar + (size_t)i * nn;
                    for (int j = 0; j < nn; j++) {
                        const uint16_t *brow =
                            b + (size_t)(n0 + j) * (size_t)K + k0;
                        float s = 0.0f;
                        for (int p = 0; p < kk; p++)
                            s += arow[p] * bf16_to_f32(brow[p]);
                        arrow[j] += s;
                    }
                }
            }

            /* AR -> C */
            for (int i = 0; i < mm; i++) {
                float *crow = c + (size_t)(m0 + i) * (size_t)N + n0;
                const float *arrow = ar + (size_t)i * nn;
                for (int j = 0; j < nn; j++) crow[j] = arrow[j];
            }
        }
    }
}

const char *qwen3_kernel_name(void) { return "tiled(128x32x128)"; }
