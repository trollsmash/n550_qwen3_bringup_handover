/* 可替换的计算 kernel 接口。
 *
 * 全模型 99% 的计算量在 GEMM 上，所以只有它值得做多版本实现：
 *   kernels_scalar.c  朴素三重循环          —— 正确性基准，永不为性能改动
 *   kernels_tiled.c   128×32×128 分块       —— 验证 tiling 逻辑（仍是标量）
 *   kernels_ame.c     AME intrinsics        —— Phase 3 目标
 *
 * 三者必须通过完全相同的测试（tools/05_e2e_test.py + test_kernels）。
 */
#ifndef QWEN3_KERNELS_H
#define QWEN3_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ---- AME tile 参数（用户 2026-07-29 确认）----
 * 输出 tile 128×128 由 AR 容量决定：64KB / 4B(FP32) = 16384 = 128×128
 * K 维由 TR 容量决定：8KB / 2B(BF16) = 4096 = 128×32
 * AME 支持 M < 128 的部分 tile，因此 decode(M=1) 无需 padding。 */
#define AME_TILE_M 128
#define AME_TILE_N 128
#define AME_TILE_K_BF16 32
#define AME_TILE_K_INT8 64

/* ★ 调用方必须保证的内存契约（AME 实现的硬性要求）★
 *
 * 权重缓冲 B 的末尾之后，必须至少还有 QWEN3_WEIGHT_TAIL_PAD 字节是**可读**的。
 *
 * 原因：AME 的 tile 装载会读超出 mtilen×mtilek 所需的最小范围（实测多读约一行
 * stride）。这在权重区中部无害 —— 多读的是相邻权重，且超出 mtilen 的行不参与
 * 计算；但对**最后一个**权重张量就会越过缓冲末尾。
 *
 * 本模型里最后一个大权重是 layers[27].mlp.down_proj，其后仅剩 2 KB 的
 * model.norm 就到文件末尾，因此只有它会触发 —— 前 27 层因后面还有大量数据而
 * 侥幸无恙。表现为「跑到最后一层才 SIGSEGV」，极易误判成 kernel 逻辑错误。
 *
 * 各平台如何满足：
 *   x86/QEMU  mmap 之后在尾部再挂一段匿名只读内存（见 main_qemu.c: mmap_guard）
 *   FPGA      权重区之后本就是 DDR 的其他区域，天然可读；只需确保权重不是
 *             紧贴 DDR 末端摆放
 */
#define QWEN3_WEIGHT_TAIL_PAD 8192

/* C[M,N] = A[M,K] · B[N,K]ᵀ
 *
 *   A  FP32 激活，行主序 [M,K]
 *   B  BF16 权重，行主序 [N,K]  —— 即 PyTorch 的 [out_features, in_features]
 *   C  FP32 输出，行主序 [M,N]
 *
 * B 按输出通道排列（[N,K]）这件事曾经令人担心，但 ISA spec 5.3 已确认：
 * AME 的 tile 形状基于 C = A × Bᵀ 模式，B 加载为 mtilen × mtilek —— 正是 [N,K]。
 * 因此权重可直接用 mlbe16 加载，无需任何离线 repack 或运行时转置。
 * 本接口的签名与 AME 原生语义天然一致。
 *
 * M = 1 时即 decode 路径的 GEMV。 */
void qwen3_gemm(float *c, const float *a, const uint16_t *b,
                int M, int K, int N);

/* 当前链接进来的 kernel 实现的名字，用于日志与测试报告。 */
const char *qwen3_kernel_name(void);

#endif /* QWEN3_KERNELS_H */
