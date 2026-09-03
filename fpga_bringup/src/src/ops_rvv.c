/* 非 GEMM 算子 —— RVV 1.0 实现。
 *
 * 目标硬件 VLEN=1024，LMUL=8 时一条指令处理 256 个 FP32；
 * hidden=1024 恰好 4 次循环，intermediate=3072 是 12 次。
 * 但代码不假设具体 VLEN —— 全部走 vsetvl 循环，换 VLEN 自动适配。
 *
 * ── 一点必须说明的取舍 ──────────────────────────────────────
 * exp 没有向量指令。本项目 ISA 不含超越函数扩展（xext 有，但 QEMU 未实现，
 * 已从 ISA 剔除）。因此 softmax 与 silu_mul 的 expf 部分保持标量，
 * 其余（求最大、求和、除法、乘法）向量化。
 *
 * BF16->FP32 用 zvfbfmin 的 vfwcvtbf16 专用指令（实测与位操作实现逐位相同）。
 *
 * 正确性由 tests/ops_test.c 保证：与 ops_scalar.c 逐算子比对。
 */
#include <math.h>               /* expf/sqrtf/powf/sinf/cosf —— 标量部分仍需 */
#include <riscv_vector.h>

#include "ops.h"
#include "qwen3.h"

/* BF16 -> FP32 widening 转换（zvfbfmin）。
 * 语义上就是"位模式左移 16 位"，也可以用 vzext+vsll+reinterpret 三条指令做，
 * 实测两者逐位相同；有专用指令时自然用专用的。
 * e16m4 与 e32m8 的元素数相同，故同一个 vl 可直接用于两者。 */
static inline vfloat32m8_t bf16_load_f32m8(const uint16_t *p, size_t vl) {
    vbfloat16m4_t h = __riscv_vle16_v_bf16m4((const __bf16 *)p, vl);
    return __riscv_vfwcvtbf16_f_f_v_f32m8(h, vl);
}

void qwen3_op_bf16_to_f32(float *out, const uint16_t *in, int n) {
    size_t rest = (size_t)n;
    for (size_t vl; rest > 0; rest -= vl, in += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        __riscv_vse32_v_f32m8(out, bf16_load_f32m8(in, vl), vl);
    }
}

void qwen3_op_rmsnorm(float *out, const float *x, const uint16_t *w, int n) {
    /* 第一遍：平方和。归约顺序与标量不同，只影响末位。 */
    float ss;
    {
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        size_t rest = (size_t)n; const float *p = x;
        for (size_t vl; rest > 0; rest -= vl, p += vl) {
            vl = __riscv_vsetvl_e32m8(rest);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(p, vl);
            acc = __riscv_vfredusum_vs_f32m8_f32m1(
                      __riscv_vfmul_vv_f32m8(v, v, vl), acc, vl);
        }
        ss = __riscv_vfmv_f_s_f32m1_f32(acc);
    }
    const float scale = 1.0f / sqrtf(ss / (float)n + QWEN3_RMS_NORM_EPS);

    /* 第二遍：out = x * scale * w */
    size_t rest = (size_t)n;
    for (size_t vl; rest > 0; rest -= vl, x += vl, w += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        vfloat32m8_t v  = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vw = bf16_load_f32m8(w, vl);
        v = __riscv_vfmul_vf_f32m8(v, scale, vl);
        v = __riscv_vfmul_vv_f32m8(v, vw, vl);
        __riscv_vse32_v_f32m8(out, v, vl);
    }
}

/* RoPE：外层 i（half=64 次）算 cos/sin 仍是标量（无向量三角函数），
 * 内层按 head 维向量化 —— 各 head 的同一位置相距 HEAD_DIM，用 strided 访存。
 * n_heads 最多 16，m1 (32 个 FP32) 足够。 */
void qwen3_op_rope(float *v, int n_heads, int pos) {
    const int half = QWEN3_HEAD_DIM / 2;
    const ptrdiff_t stride = (ptrdiff_t)QWEN3_HEAD_DIM * (ptrdiff_t)sizeof(float);
    const size_t vl = (size_t)n_heads;

    for (int i = 0; i < half; i++) {
        float inv_freq = 1.0f / powf(QWEN3_ROPE_THETA,
                                     (float)(2 * i) / (float)QWEN3_HEAD_DIM);
        float ang = (float)pos * inv_freq;
        float c = cosf(ang), s = sinf(ang);

        size_t avl = __riscv_vsetvl_e32m1(vl);
        vfloat32m1_t va = __riscv_vlse32_v_f32m1(v + i,        stride, avl);
        vfloat32m1_t vb = __riscv_vlse32_v_f32m1(v + i + half, stride, avl);
        /* na = a*c - b*s ;  nb = b*c + a*s */
        vfloat32m1_t na = __riscv_vfsub_vv_f32m1(
                              __riscv_vfmul_vf_f32m1(va, c, avl),
                              __riscv_vfmul_vf_f32m1(vb, s, avl), avl);
        vfloat32m1_t nb = __riscv_vfadd_vv_f32m1(
                              __riscv_vfmul_vf_f32m1(vb, c, avl),
                              __riscv_vfmul_vf_f32m1(va, s, avl), avl);
        __riscv_vsse32_v_f32m1(v + i,        stride, na, avl);
        __riscv_vsse32_v_f32m1(v + i + half, stride, nb, avl);
    }
}

/* softmax：求最大与求和向量化，exp 仍是标量（无向量 exp）。 */
void qwen3_op_softmax(float *x, int n) {
    float mx;
    {
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(x[0], __riscv_vsetvlmax_e32m1());
        size_t rest = (size_t)n; const float *p = x;
        for (size_t vl; rest > 0; rest -= vl, p += vl) {
            vl = __riscv_vsetvl_e32m8(rest);
            acc = __riscv_vfredmax_vs_f32m8_f32m1(
                      __riscv_vle32_v_f32m8(p, vl), acc, vl);
        }
        mx = __riscv_vfmv_f_s_f32m1_f32(acc);
    }

    /* 减最大值后取指数并累加 —— 标量。减最大值是数值稳定性所必需。 */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }

    const float inv = 1.0f / sum;
    size_t rest = (size_t)n; float *p = x;
    for (size_t vl; rest > 0; rest -= vl, p += vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        __riscv_vse32_v_f32m8(
            p, __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(p, vl), inv, vl), vl);
    }
}

/* silu_mul：expf 是主体开销且无向量版本，向量化收益有限，保持标量。
 * 留在这里而不是回退到 ops_scalar.c，是为了让"当前实现"的语义完整、可独立链接。 */
void qwen3_op_silu_mul(float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void qwen3_op_residual_add(float *x, const float *y, int n) {
    size_t rest = (size_t)n;
    for (size_t vl; rest > 0; rest -= vl, x += vl, y += vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        __riscv_vse32_v_f32m8(x,
            __riscv_vfadd_vv_f32m8(__riscv_vle32_v_f32m8(x, vl),
                                   __riscv_vle32_v_f32m8(y, vl), vl), vl);
    }
}

/* argmax：向量归约只给最大值不给下标，故两遍 ——
 * 先向量求最大值，再用向量比较 + vfirst 定位第一个相等元素。 */
int qwen3_op_argmax(const float *v, int n) {
    float mx;
    {
        /* ★ 初值用 -inf，不要用 v[0]。
         * v[0] 是**标量** load，而标量访存绕不开 L1 D-cache；logits 由 AME
         * 写（绕 cache），CLP 下又没有 INVAL 兜底，这一读可能拿到上一个
         * token 的残值。残值若大于真实最大值，mx 就是个不存在的数，
         * 下面的相等匹配全部落空、函数返回 0 —— 表现为随机吐出 token 0。
         * 用 -inf 起头既消掉这次标量访存，数值上也完全等价。 */
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(-__builtin_inff(),
                                                  __riscv_vsetvlmax_e32m1());
        size_t rest = (size_t)n; const float *p = v;
        for (size_t vl; rest > 0; rest -= vl, p += vl) {
            vl = __riscv_vsetvl_e32m8(rest);
            acc = __riscv_vfredmax_vs_f32m8_f32m1(
                      __riscv_vle32_v_f32m8(p, vl), acc, vl);
        }
        mx = __riscv_vfmv_f_s_f32m1_f32(acc);
    }

    size_t rest = (size_t)n; const float *p = v; int base = 0;
    for (size_t vl; rest > 0; rest -= vl, p += vl, base += (int)vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        vbool4_t eq = __riscv_vmfeq_vf_f32m8_b4(__riscv_vle32_v_f32m8(p, vl), mx, vl);
        long idx = __riscv_vfirst_m_b4(eq, vl);
        if (idx >= 0) return base + (int)idx;
    }
    return 0;   /* 不可达：mx 必然来自 v 中某个元素 */
}

float qwen3_op_dot(const float *a, const float *b, int n) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
    size_t rest = (size_t)n;
    for (size_t vl; rest > 0; rest -= vl, a += vl, b += vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        vfloat32m8_t prod = __riscv_vfmul_vv_f32m8(
                                __riscv_vle32_v_f32m8(a, vl),
                                __riscv_vle32_v_f32m8(b, vl), vl);
        acc = __riscv_vfredusum_vs_f32m8_f32m1(prod, acc, vl);
    }
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

void qwen3_op_axpy(float *y, float a, const float *x, int n) {
    size_t rest = (size_t)n;
    for (size_t vl; rest > 0; rest -= vl, y += vl, x += vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        /* vfmacc 是融合乘加，与标量 y[i] += a*x[i] 被收缩成 FMA 后逐位一致 */
        __riscv_vse32_v_f32m8(y,
            __riscv_vfmacc_vf_f32m8(__riscv_vle32_v_f32m8(y, vl), a,
                                    __riscv_vle32_v_f32m8(x, vl), vl), vl);
    }
}

const char *qwen3_ops_name(void) { return "rvv"; }
