/* 非 GEMM 算子 —— 标量实现，正确性基准。
 *
 * 刻意写得最直白：ops_rvv.c 必须与它对齐。永远不要为了性能改动本文件。
 * 实现从 qwen3.c 原样搬出，行为逐位不变。
 */
#include <math.h>       /* sqrtf expf sinf cosf powf —— 全项目的 libm 依赖集中于此 */

#include "ops.h"
#include "qwen3.h"      /* bf16_to_f32 与模型常量 */

void qwen3_op_rmsnorm(float *out, const float *x, const uint16_t *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + QWEN3_RMS_NORM_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * bf16_to_f32(w[i]);
}

/* 频率 inv_freq[i] = theta^(-2i/head_dim)，theta = 1000000（不是 10000）。
 * 每次调用都重算 powf 是低效的，但这是正确性基准，不做优化；
 * RVV 版可以预计算成表。 */
void qwen3_op_rope(float *v, int n_heads, int pos) {
    const int half = QWEN3_HEAD_DIM / 2;
    for (int i = 0; i < half; i++) {
        float inv_freq = 1.0f / powf(QWEN3_ROPE_THETA,
                                     (float)(2 * i) / (float)QWEN3_HEAD_DIM);
        float ang = (float)pos * inv_freq;
        float c = cosf(ang), s = sinf(ang);
        for (int h = 0; h < n_heads; h++) {
            float *p = v + (size_t)h * QWEN3_HEAD_DIM;
            float a = p[i], b = p[i + half];
            p[i]        = a * c - b * s;
            p[i + half] = b * c + a * s;
        }
    }
}

void qwen3_op_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    /* 减最大值是数值稳定性所必需：不减的话 expf 会上溢成 inf。 */
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void qwen3_op_silu_mul(float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void qwen3_op_residual_add(float *x, const float *y, int n) {
    for (int i = 0; i < n; i++) x[i] += y[i];
}

int qwen3_op_argmax(const float *v, int n) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

void qwen3_op_bf16_to_f32(float *out, const uint16_t *in, int n) {
    for (int i = 0; i < n; i++) out[i] = bf16_to_f32(in[i]);
}

float qwen3_op_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void qwen3_op_axpy(float *y, float a, const float *x, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

const char *qwen3_ops_name(void) { return "scalar"; }
