/* 非 GEMM 算子的可替换接口。
 *
 * GEMM 占全模型 99% 的计算量，单独由 kernels.h 抽象；
 * 这里是剩下的 1%：归约、逐元素、三角函数那一类。
 *
 *   ops_scalar.c   直白的标量实现 —— 正确性基准，永不为性能改动
 *   ops_rvv.c      RVV 1.0 实现（VLEN=1024，LMUL=8 时单次 256 个 FP32）
 *
 * 两者必须通过同一套测试（tests/ops_test.c）。
 *
 * 把它们抽出来的另一个好处：math.h 的依赖（sqrtf/expf/sinf/cosf/powf）
 * 全部集中到 ops_*.c，qwen3.c 变成纯逻辑，不碰任何外部库。
 */
#ifndef QWEN3_OPS_H
#define QWEN3_OPS_H

#include <stdint.h>

/* RMSNorm：out = x / sqrt(mean(x²) + eps) * w。不减均值（与 LayerNorm 的关键区别）。
 * w 是 BF16 权重。out 与 x 可以是同一指针。 */
void qwen3_op_rmsnorm(float *out, const float *x, const uint16_t *w, int n);

/* RoPE，rotate_half 形式：配对的是 (i, i+head_dim/2)，不是 (2i, 2i+1)。
 * 就地修改 v 的 n_heads 个头。 */
void qwen3_op_rope(float *v, int n_heads, int pos);

/* softmax，就地。内部先减最大值，否则 expf 会上溢成 inf。 */
void qwen3_op_softmax(float *x, int n);

/* SwiGLU 的后半段：gate[i] = silu(gate[i]) * up[i]，就地写回 gate。 */
void qwen3_op_silu_mul(float *gate, const float *up, int n);

/* 残差：x[i] += y[i] */
void qwen3_op_residual_add(float *x, const float *y, int n);

/* 返回最大元素的下标。用于采样（151936 个候选）。 */
int qwen3_op_argmax(const float *v, int n);

/* BF16 -> FP32 批量转换。用于 embedding 查表。 */
void qwen3_op_bf16_to_f32(float *out, const uint16_t *in, int n);

/* 点积。attention 里算 q·k，n = head_dim = 128。 */
float qwen3_op_dot(const float *a, const float *b, int n);

/* y[i] += a * x[i]。attention 里按注意力权重累加 v。 */
void qwen3_op_axpy(float *y, float a, const float *x, int n);

/* 当前链接进来的实现名，用于日志与测试报告。 */
const char *qwen3_ops_name(void);

#endif /* QWEN3_OPS_H */
