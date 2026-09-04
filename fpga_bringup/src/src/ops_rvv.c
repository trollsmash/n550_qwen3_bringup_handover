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

/* softmax：全向量 —— 求最大、指数、求和、归一化都不含标量访存。 */
/* 向量 expf：exp(x) = 2^k · exp(r)，k = round(x·log2e)，r = x − k·ln2。
 * r 落在 ±ln2/2 内，泰勒 6 阶即可；2^k 直接把 k 塞进 float 指数位构造。
 *
 * 为什么必须有它：这颗核没有硬件超越函数，标量 expf 是软件实现，每次几十到
 * 上百周期。softmax 与 silu 每 token 合计要调三十万次以上，按 50 周期折算
 * 已与权重访存同量级 —— 这是过去性能分析里一直没算进去的一块。
 * 换成向量后 e32m8 一次算 256 个元素、十余条指令。
 *
 * 另一重必要性：CLP 下激活在 0x1xxx 窗口，**那段空间只有 RVV 指令能访存**，
 * 标量访问会触发 PMA 异常。所以这两个算子本来也必须彻底去标量。
 *
 * 精度：与 libm 对拍，相对误差 2.5e-7（float 精度极限附近）。 */
static inline vfloat32m8_t vexpf_m8(vfloat32m8_t x, size_t vl) {
    const float LOG2E  = 1.44269504088896341f;
    /* ln2 拆成 HI/LO 两半：k 最大到 127，单精度 ln2 乘上去尾部有效位会丢光，
     * 分两步减才能保住精度。 */
    const float LN2_HI = 0.693359375f;
    const float LN2_LO = -2.12194440e-4f;

    /* 上界防 2^k 溢出；下界取 −87 是因为再小 k+127 会掉进非规格化区，
     * 移位构造 2^k 的做法在那里失效。exp(−87)≈1.6e−38，对 softmax 无影响。 */
    x = __riscv_vfmin_vf_f32m8(x,  88.0f, vl);
    x = __riscv_vfmax_vf_f32m8(x, -87.0f, vl);

    vint32m8_t   ki = __riscv_vfcvt_x_f_v_i32m8(
                          __riscv_vfmul_vf_f32m8(x, LOG2E, vl), vl);
    vfloat32m8_t kf = __riscv_vfcvt_f_x_v_f32m8(ki, vl);

    vfloat32m8_t r = __riscv_vfnmsac_vf_f32m8(x, LN2_HI, kf, vl);
    r = __riscv_vfnmsac_vf_f32m8(r, LN2_LO, kf, vl);

    /* exp(r) 泰勒 6 阶，Horner 展开 */
    vfloat32m8_t q = __riscv_vfmv_v_f_f32m8(0.0013888889f, vl);   /* 1/720 */
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, r, vl), 0.008333334f, vl);
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, r, vl), 0.041666668f, vl);
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, r, vl), 0.16666667f,  vl);
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, r, vl), 0.5f,         vl);
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, r, vl), 1.0f,         vl);
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, r, vl), 1.0f,         vl);

    /* 2^k：(k+127)<<23 就是 float 的位模式 */
    vint32m8_t bits = __riscv_vsll_vx_i32m8(
                          __riscv_vadd_vx_i32m8(ki, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m8(
               q, __riscv_vreinterpret_v_i32m8_f32m8(bits), vl);
}

void qwen3_op_softmax(float *x, int n) {
    float mx;
    {
        /* 初值取 −inf，不读 x[0] —— 那是标量访存，CLP 下会触发 PMA 异常。
         * 数值上与从 x[0] 起始等价。 */
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(-__builtin_inff(),
                                                  __riscv_vsetvlmax_e32m1());
        size_t rest = (size_t)n; const float *p = x;
        for (size_t vl; rest > 0; rest -= vl, p += vl) {
            vl = __riscv_vsetvl_e32m8(rest);
            acc = __riscv_vfredmax_vs_f32m8_f32m1(
                      __riscv_vle32_v_f32m8(p, vl), acc, vl);
        }
        mx = __riscv_vfmv_f_s_f32m1_f32(acc);
    }

    /* 减最大值后取指数并累加。减最大值是数值稳定性所必需；
     * 指数走向量版本 —— 既是这里最大的一笔开销，也因为标量在 CLP 下不能碰。 */
    float sum;
    {
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        size_t rest = (size_t)n; float *p = x;
        for (size_t vl; rest > 0; rest -= vl, p += vl) {
            vl = __riscv_vsetvl_e32m8(rest);
            vfloat32m8_t e = vexpf_m8(
                __riscv_vfsub_vf_f32m8(__riscv_vle32_v_f32m8(p, vl), mx, vl), vl);
            __riscv_vse32_v_f32m8(p, e, vl);
            acc = __riscv_vfredusum_vs_f32m8_f32m1(e, acc, vl);
        }
        sum = __riscv_vfmv_f_s_f32m1_f32(acc);
    }

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
    size_t rest = (size_t)n; float *g = gate; const float *u = up;
    for (size_t vl; rest > 0; rest -= vl, g += vl, u += vl) {
        vl = __riscv_vsetvl_e32m8(rest);
        vfloat32m8_t vg = __riscv_vle32_v_f32m8(g, vl);
        /* sigmoid(g) = 1/(1+exp(−g)) */
        vfloat32m8_t den = __riscv_vfadd_vf_f32m8(
            vexpf_m8(__riscv_vfneg_v_f32m8(vg, vl), vl), 1.0f, vl);
        vfloat32m8_t sig = __riscv_vfdiv_vv_f32m8(vg, den, vl);
        __riscv_vse32_v_f32m8(
            g, __riscv_vfmul_vv_f32m8(sig, __riscv_vle32_v_f32m8(u, vl), vl), vl);
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
