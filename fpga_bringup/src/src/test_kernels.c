/* GEMM kernel 单元测试 —— 不依赖模型权重，秒级反馈。
 *
 * Phase 3 写 AME kernel 时，先让这个测试过，再去跑整模型。
 * 全模型跑一次要十几秒且失败信息笼统；这里失败会直接指出是哪个 (M,K,N)。
 *
 * 覆盖的形状取自 Qwen3-0.6B 的真实 GEMM，外加边界情况：
 *   - M=1            decode 路径（部分 tile，AME 支持）
 *   - M=4 / 127/128/129  prefill 路径与 tile 边界
 *   - K/N 取真实值（1024/2048/3072/151936 全是 128 的整数倍）
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels.h"
#include "qwen3.h"

/* 参考实现：与被测 kernel 独立，故意用最直白的写法。 */
static void gemm_ref(float *c, const float *a, const uint16_t *b,
                     int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double sum = 0.0;   /* 用 double 累加，排除参考实现自身的误差 */
            for (int k = 0; k < K; k++)
                sum += (double)a[(size_t)m * K + k]
                     * (double)bf16_to_f32(b[(size_t)n * K + k]);
            c[(size_t)m * N + n] = (float)sum;
        }
}

static uint32_t rng_state = 12345;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static int run_case(int M, int K, int N, float tol) {
    float    *a   = malloc((size_t)M * K * sizeof(float));
    uint16_t *b   = malloc((size_t)N * K * sizeof(uint16_t));
    float    *c   = malloc((size_t)M * N * sizeof(float));
    float    *ref = malloc((size_t)M * N * sizeof(float));
    if (!a || !b || !c || !ref) { printf("  OOM\n"); return 1; }

    for (size_t i = 0; i < (size_t)M * K; i++) a[i] = frand();
    /* 权重取 BF16 可精确表示的值，排除权重本身的舍入干扰 */
    for (size_t i = 0; i < (size_t)N * K; i++) b[i] = f32_to_bf16(frand());

    gemm_ref(ref, a, b, M, K, N);
    qwen3_gemm(c, a, b, M, K, N);

    float amax = 0.0f, emax = 0.0f;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        float r = fabsf(ref[i]);
        float e = fabsf(ref[i] - c[i]);
        if (r > amax) amax = r;
        if (e > emax) emax = e;
    }
    float rel = emax / (amax + 1e-30f);
    int ok = (rel < tol) && isfinite(rel);
    printf("  M=%-5d K=%-6d N=%-7d  max|ref|=%9.3g  rel_err=%9.3e  %s\n",
           M, K, N, (double)amax, (double)rel, ok ? "OK" : "FAIL");

    free(a); free(b); free(c); free(ref);
    return ok ? 0 : 1;
}

int main(void) {
    printf("GEMM kernel 测试 —— 实现: %s\n", qwen3_kernel_name());
    printf("  容差 1e-5（FP32 累加顺序差异应远小于此）\n\n");

    int fails = 0;
    const float tol = 1e-5f;

    printf("[decode 路径 / 部分 tile]\n");
    fails += run_case(1, 1024, 2048, tol);      /* q_proj  */
    fails += run_case(1, 1024, 1024, tol);      /* k_proj/v_proj */
    fails += run_case(1, 2048, 1024, tol);      /* o_proj  */
    fails += run_case(1, 1024, 3072, tol);      /* gate/up */
    fails += run_case(1, 3072, 1024, tol);      /* down    */

    printf("\n[prefill 路径 / tile 边界]\n");
    fails += run_case(4,   1024, 2048, tol);    /* 当前测试 prompt 长度 */
    fails += run_case(127, 1024, 1024, tol);    /* 差一行填满 */
    fails += run_case(128, 1024, 1024, tol);    /* 正好填满 */
    fails += run_case(129, 1024, 1024, tol);    /* 跨到第二个 M tile */

    printf("\n[K 维边界]\n");
    fails += run_case(8, 32,  128, tol);        /* 正好一个 K 分块 */
    fails += run_case(8, 64,  128, tol);        /* 两个 K 分块 */
    fails += run_case(8, 33,  128, tol);        /* K 非 32 整数倍（模型里不会出现）*/

    printf("\n%s (%d 个用例失败)\n",
           fails ? "[FAIL]" : "[PASS] 全部通过", fails);
    return fails ? 1 : 0;
}
