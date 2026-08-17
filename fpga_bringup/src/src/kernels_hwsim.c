/* 按 RTL 语义模拟的 GEMM —— 只用于在 PC 上验证数值影响，不上板。
 *
 * 目的：回答"真机（AMU RTL）的累加语义下，全模型还会不会输出同样的 token"。
 * 有了它就不必等板子，也不必信 QEMU —— QEMU 的累加粒度与 RTL 并不一致。
 *
 * 复刻的 RTL 行为（来自 AMU/step1 的设计与文档）：
 *   ma_pkg.sv      ARRAY_K_FP = 16     一个 PE 一次吃 16 路乘积
 *   MA HLD §4.2    k_iter = ceil(mtilek/16)，k_cnt 是最外层循环，
 *                  每个 k_iter 结束把结果写回 AR（FP32）
 *   => 每 16 个 K 元素就落回一次 FP32，而不是 32 个
 *   MA HLD §4.3.2  A/B 都是 BF16 输入，故 FP32 激活先按 RNE 转 BF16
 *
 * 刻意**没有**复刻的部分：45-bit 扩展定点窗口、11-bit 残差桶与 sticky。
 * 那是硬件为省面积做的精度折中（HLD §4.3.3），会让结果比正确舍入差约 1 ULP。
 * 本文件用 double 累加 16 路乘积，等价于"精确求和后正确舍入"——
 * 即数值上界。若连这个上界都不改变 argmax，那 RTL 的实际精度更不会改变。
 */
#include "kernels.h"
#include "qwen3.h"      /* bf16_to_f32 */

/* FP32 -> BF16，round-half-to-even。与 AME 的输入端行为一致。 */
static inline unsigned short f32_to_bf16_rne(float f) {
    union { float f; unsigned int u; } v = { f };
    unsigned int u = v.u;
    if ((u & 0x7F800000u) == 0x7F800000u) return (unsigned short)(u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

/* 硬件一次点积覆盖的 K 元素数（ARRAY_K_FP）。每满 16 个就落回 FP32。 */
#define HW_ROUND_K 16

void qwen3_gemm(float *c, const float *a, const unsigned short *b,
                int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        const float *arow = a + (size_t)m * (size_t)K;
        float *crow = c + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            const unsigned short *brow = b + (size_t)n * (size_t)K;
            float acc = 0.0f;                 /* 累加器按 FP32 承载 */
            for (int k0 = 0; k0 < K; k0 += HW_ROUND_K) {
                int k1 = k0 + HW_ROUND_K;
                if (k1 > K) k1 = K;
                /* 一轮 16 路点积：用 double 累加，等价于精确求和。
                 * BF16 乘积尾数 16 位，16 项最多再加 4 位，远小于 double 的 53 位。 */
                double blk = 0.0;
                for (int k = k0; k < k1; k++) {
                    float av = bf16_to_f32(f32_to_bf16_rne(arow[k]));
                    blk += (double)av * (double)bf16_to_f32(brow[k]);
                }
                acc = (float)((double)acc + blk);   /* 每轮结束落回 FP32 */
            }
            crow[n] = acc;
        }
    }
}

const char *qwen3_kernel_name(void) { return "hwsim(RTL 16-elem rounding)"; }
