/* GEMM kernel —— RVV 的 BF16 widening 乘加（zvfbfwma）实现。
 *
 * 与 kernels_ame.c 算同一件事：C[M,N] = A[M,K] · B[N,K]ᵀ，
 * 区别只在用什么硬件单元。
 *
 * ── 为什么要有这一份 ────────────────────────────────────────
 * decode 路径 M 恒为 1，而 AME 的 tile 是 128×32×128 —— 一条 mfmacc 名义上
 * 做 524288 次乘加，M=1 时只有 4096 次有效，利用率 1/128。
 * 真实硬件上这不构成问题（decode 本就是访存瓶颈，算力浪费不转化为时间），
 * 但 QEMU 的 helper 按满 tile 模拟，于是模拟开销也放大了 128 倍。
 *
 * vfwmaccbf16 做的正是 FP32 += BF16 × BF16，VLEN=1024 时一次处理 256 对，
 * 没有任何浪费。因此它在 QEMU 上应当明显快于 AME —— 这不改变最终方案
 * （FPGA 上仍用 AME，那里的并行阵列一个周期出结果），只是给 QEMU 阶段
 * 一条更快的验证路径，让每轮全模型验证从十几分钟降下来。
 *
 * 用 KERNEL=rvvbf16 选择本实现。正确性由 tests/ame_gemm_test.c 保证
 * （该测试与具体 kernel 无关，测的是 qwen3_gemm 接口）。
 */
#include <riscv_vector.h>

#include "kernels.h"
#include "qwen3.h"

/* A 的 BF16 暂存，上界 = MAX_SEQ × INTERMEDIATE_SIZE（1.5 MB BSS）。
 * 当前 forward 逐 token（M=1）只用到 K 个元素，但按 prefill 批量化的上界
 * 预留 —— 否则 M>1 时会静默返回，表现为输出"未写入"。 */
#define RVVBF_A_MAX ((size_t)QWEN3_MAX_SEQ * QWEN3_INTERMEDIATE_SIZE)
static uint16_t g_abuf[RVVBF_A_MAX] __attribute__((aligned(64)));

void qwen3_gemm(float *c, const float *a, const uint16_t *b,
                int M, int K, int N) {
    /* 激活 FP32 -> BF16。vfwmaccbf16 的两个源都必须是 BF16，
     * 与 AME 一致 —— 这是硬件的实际行为，不是实现取巧。 */
    const size_t n_a = (size_t)M * (size_t)K;
    if (n_a > RVVBF_A_MAX) return;              /* 调用方违约，见上面注释 */
    for (size_t i = 0; i < n_a; i++) g_abuf[i] = f32_to_bf16(a[i]);

    const size_t vlmax = __riscv_vsetvlmax_e32m8();

    for (int m = 0; m < M; m++) {
        const uint16_t *arow = g_abuf + (size_t)m * K;
        float *crow = c + (size_t)m * N;

        for (int n = 0; n < N; n++) {
            const uint16_t *brow = b + (size_t)n * K;

            /* K 维累加在向量寄存器里，最后只归约一次。
             * 注意 tail：K 不是 vlmax 整数倍时最后一次 vl 变小，
             * intrinsics 默认 tail-undisturbed，acc 的高位元素保持先前累加值，
             * 故最后按 vlmax 归约仍然正确。 */
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
            size_t rest = (size_t)K;
            const uint16_t *pa = arow, *pb = brow;
            for (size_t vl; rest > 0; rest -= vl, pa += vl, pb += vl) {
                vl = __riscv_vsetvl_e32m8(rest);
                vbfloat16m4_t va = __riscv_vle16_v_bf16m4((const __bf16 *)pa, vl);
                vbfloat16m4_t vb = __riscv_vle16_v_bf16m4((const __bf16 *)pb, vl);
                acc = __riscv_vfwmaccbf16_vv_f32m8(acc, va, vb, vl);
            }

            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f,
                                    __riscv_vsetvlmax_e32m1());
            vfloat32m1_t sum = __riscv_vfredusum_vs_f32m8_f32m1(acc, zero, vlmax);
            crow[n] = __riscv_vfmv_f_s_f32m1_f32(sum);
        }
    }
}

const char *qwen3_kernel_name(void) { return "rvvbf16"; }
