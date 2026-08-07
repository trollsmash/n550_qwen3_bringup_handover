/* 标量 GEMM —— 正确性基准。
 *
 * 刻意写得最直白：所有其他实现（tiled / AME）都必须与它对齐。
 * 永远不要为了性能改动本文件。
 */
#include "kernels.h"
#include "qwen3.h"      /* bf16_to_f32 */

void qwen3_gemm(float *c, const float *a, const uint16_t *b,
                int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        const float *arow = a + (size_t)m * (size_t)K;
        float *crow = c + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            const uint16_t *brow = b + (size_t)n * (size_t)K;
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += arow[k] * bf16_to_f32(brow[k]);
            crow[n] = sum;
        }
    }
}

const char *qwen3_kernel_name(void) { return "scalar"; }
