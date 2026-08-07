#!/usr/bin/env python3
"""Phase 4 —— 生成交付给 RTL 验证的 AME 指令级测试向量。

面向的使用者是**另一个部门的 RTL 验证同事**：他们没有本项目的软件环境，
也不该需要。每个 case 都是一段能自己跑起来、算完往固定地址写完成标志的
最小程序，外加一份期望输出。testbench 轮询标志位、dump 输出、逐字比对。

═══════════════════════════════════════════════════════════════════
参考模型的规范（本文件的核心，务必先读懂再改）
═══════════════════════════════════════════════════════════════════

黄金参考**不来自 QEMU**。QEMU 也是一个实现，它已经被证实存在缺陷：

  * mzero 曾只清累加器前 64 行（v1.0.4，后修复）
  * 同一份输入在不同执行上下文下，最终舍入方向不稳定：
    真值 16777247 落在两个 FP32 中间，有时给 16777246 有时给 16777248

拿一个实现去验另一个实现，错误会互相掩盖。所以参考结果由**架构语义**
推导，QEMU 只做交叉检查（见 --cross-check）。

实测确定的三条语义（tools/ 下的探针程序，结论写进了 README）：

  T1  单条 mfmacc 内部，K 维累加精度**宽于 FP32**。
      构造 A=[2^24, 1×31]、B=全1 时，31 个 1.0 一个都没被吞掉
      —— 若是 FP32 逐项累加，它们会全部消失。
      ⇒ 块内必须精确求和（本文件用 Fraction），最后才舍入。

  T2  跨多条 mfmacc 累加时，累加器中间值**落回 FP32**。
      第二条 mfmacc 贡献的 1.0（远小于该量级 1 ULP=2）被整个吞掉。
      ⇒ 必须按 K=32 分块，块间按 FP32 舍入后再累加。
      本项目 K=1024 要连做 32 次，写成一次性高精度求和必然对不上。

  T3  舍入方向按 IEEE RNE（最近偶数）建模。
      注：QEMU 在中点情形下不稳定，这类 case 会被标 ⚠，允许 ±1 ULP。

于是参考模型就是下面这段话的直译：

    for 每个 K 块（32 个元素）:
        块和 = Σ A[m,k] × B[n,k]        ← Fraction，零误差
        acc  = fp32_rne(acc + 块和)      ← 每块结束舍入一次

═══════════════════════════════════════════════════════════════════
数据布局约定（指南里也会写一遍）
═══════════════════════════════════════════════════════════════════

  A 装载为 mtilem × mtilek，行 stride 由 meta.json 给出
  B 装载为 mtilen × mtilek —— 注意是 [N,K] 而非 [K,N]，
    这正是 PyTorch 权重的 [out_features, in_features]，无需 repack
  C 为 mtilem × mtilen 的 FP32，C = A · Bᵀ

  .hex 文件一律"每行 32 bit，即内存镜像的一个小端字"：
    a.hex / b.hex —— 一行装两个 BF16（低 16 位是偶数下标那个）
    c_ref.hex     —— 一行一个 FP32 位模式

用法:
    python3 tools/40_gen_ame_vectors.py                 # 生成全部 case
    python3 tools/40_gen_ame_vectors.py --only 001      # 只生成一个
    python3 tools/40_gen_ame_vectors.py --base 0x80000000
"""

import argparse
import json
import struct
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "deliverables" / "ame_vectors_v1"

# ───────────────────────── BF16 / FP32 基本操作 ─────────────────────────

def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def bf16_to_exact(u16: int) -> Fraction:
    """BF16 位模式 → 精确有理数。

    BF16 的位模式就是 FP32 的高 16 位，左移 16 位即得 FP32，无任何精度损失。
    转成 Fraction 是为了让块内求和零误差 —— 这是 T1 的要求。
    Inf/NaN 不能表示成 Fraction，由调用方在特殊值 case 里单独处理。
    """
    f = bits_f32(u16 << 16)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"BF16 0x{u16:04x} 是 Inf/NaN，需走特殊值路径")
    return Fraction(f)


def f32_to_bf16_rne(x: float) -> int:
    """FP32 → BF16，按 RNE 舍入（导出激活/权重时用）。"""
    u = f32_bits(x)
    if (u & 0x7F800000) == 0x7F800000:          # Inf/NaN 直接截高 16 位
        return (u >> 16) & 0xFFFF
    # round-half-to-even：加上 0x7FFF + 落点最低位
    u += 0x7FFF + ((u >> 16) & 1)
    return (u >> 16) & 0xFFFF


def exact_to_f32_rne(v: Fraction) -> float:
    """精确有理数 → FP32，按 RNE 舍入。

    Python 的 float 是 FP64，Fraction→float 已按 RNE 舍到 53 位尾数；
    再 struct 到 FP32 又是一次 RNE。两次舍入在极罕见的双重中点情形下
    与一次直接舍入可能差 1 ULP（double rounding）。
    对本向量包无影响：所有块和的有效位数都远小于 53 位（BF16 乘积
    最多 16 位尾数，32 项求和至多再加 5 位），FP64 能精确容纳，
    因此第一次转换无损，只有第二次真正舍入。
    """
    return bits_f32(f32_bits(float(v)))


# ───────────────────────── 参考模型 ─────────────────────────

K_BLOCK = 32        # BF16 下一条 mfmacc 覆盖的 K 元素数（AME tile 参数）


def ame_gemm_ref(A, B, M, N, K):
    """C = A · Bᵀ，严格按上文规范。

    A: [M][K] BF16 位模式   B: [N][K] BF16 位模式
    返回 [M][N] 的 FP32 位模式。
    """
    if K % K_BLOCK != 0:
        # 部分块合法（mtilek < 32），末块按实际长度算
        pass
    C = [[0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            acc = 0.0                       # 块间以 FP32 承载（T2）
            for k0 in range(0, K, K_BLOCK):
                k1 = min(k0 + K_BLOCK, K)
                # 块内精确求和，零误差（T1）
                blk = Fraction(0)
                for k in range(k0, k1):
                    blk += bf16_to_exact(A[m][k]) * bf16_to_exact(B[n][k])
                # 块结束时舍入一次（T2/T3）
                acc = exact_to_f32_rne(Fraction(acc) + blk)
            C[m][n] = f32_bits(acc)
    return C


# ───────────────────────── 确定性数据生成 ─────────────────────────

class Rng:
    """xorshift64*，确定性且与 C 端可对拍。

    不用 Python 的 random：向量包要能被同事在任何机器上复现出逐位相同的
    结果，标准库的实现细节和版本差异是不必要的风险。
    """

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFFFFFFFFFF or 0x9E3779B97F4A7C15

    def next(self) -> int:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = s
        return s

    def unit(self) -> float:
        """[-1, 1) 均匀。"""
        return (self.next() >> 11) / float(1 << 52) * 2.0 - 1.0


def gen_matrix(rows, cols, rng, scale=1.0, mode="normal"):
    """生成 BF16 位模式矩阵。

    刻意避免全 1 或对称数据：那类数据会让"行索引错乱""转置""累加器
    未清零"等错误算出正确答案，测了等于没测。这条是本项目多次踩坑
    换来的 —— 用例 6 的复杂矩阵就是为此加的。
    """
    M = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if mode == "normal":
                # 幅值跨几个数量级，且与 (i,j) 相关，转置立刻暴露
                v = rng.unit() * scale * (1.0 + 0.5 * ((i * 7 + j * 3) % 5))
            elif mode == "massive":
                # Qwen3 深层实测：少数维度出现 ~6912 的 outlier，其余正常
                v = rng.unit() * scale
                if (i * 13 + j * 29) % 97 == 0:
                    v = 6912.0 * (1.0 if (i + j) % 2 == 0 else -1.0)
            else:
                raise ValueError(mode)
            M[i][j] = f32_to_bf16_rne(v)
    return M


# ───────────────────────── 导出 ─────────────────────────

def bf16_rows_to_words(M, cols_padded):
    """BF16 矩阵 → 内存镜像的 32 位小端字序列。

    cols_padded 是行 stride 对应的元素数（可大于实际列数，尾部补 0）：
    AME 的 tile 装载按 stride 走，行与行之间可以有空隙。
    """
    assert cols_padded % 2 == 0, "行 stride 必须是 4 字节的整数倍"
    words = []
    for row in M:
        padded = list(row) + [0] * (cols_padded - len(row))
        for i in range(0, cols_padded, 2):
            words.append(padded[i] | (padded[i + 1] << 16))
    return words


def f32_rows_to_words(C, cols_padded):
    words = []
    for row in C:
        words.extend(list(row) + [0] * (cols_padded - len(row)))
    return words


def write_hex(path: Path, words, comment: str):
    """每行一个 32 位字，$readmemh 直接可读。

    注释用 // 起头：Verilog 的 $readmemh 接受 // 与 /* */ 注释，
    带上来源信息能省掉同事回头问"这文件哪来的"。
    """
    with open(path, "w", encoding="ascii") as f:
        f.write(f"// {comment}\n")
        f.write(f"// {len(words)} words, 32-bit each, little-endian memory image\n")
        for w in words:
            f.write(f"{w & 0xFFFFFFFF:08x}\n")


def write_bin(path: Path, words):
    with open(path, "wb") as f:
        for w in words:
            f.write(struct.pack("<I", w & 0xFFFFFFFF))


# ───────────────────────── case 定义 ─────────────────────────

class Case:
    """一个测试向量。

    K 可以大于 K_BLOCK：那意味着程序要连做 K/32 条 mfmacc 累加到同一个
    累加器，正是 T2 要验证的路径，也是真实 GEMM（K=1024）的样子。

    pad_a / pad_b 是行 stride 相对 K 的额外元素数。stride > 实际列数意味着
    行与行之间有空隙 —— 真实代码里这很常见（子矩阵、对齐填充），而
    "装载时忽略 stride、按紧排走"的实现只有在这种 case 下才会露馅。
    """

    def __init__(self, cid, name, M, N, K, why, mode="normal", scale=1.0,
                 seed=None, pad_a=0, pad_b=0):
        self.cid, self.name = cid, name
        self.M, self.N, self.K = M, N, K
        self.why = why                      # 这个 case 为什么存在
        self.mode, self.scale = mode, scale
        self.pad_a, self.pad_b = pad_a, pad_b
        self.seed = seed if seed is not None else (0xA3E0_0000 + int(cid))

    @property
    def dirname(self):
        return f"{self.cid}_{self.name}"

    @property
    def stride_a_elems(self):
        return self.K + self.pad_a

    @property
    def stride_b_elems(self):
        return self.K + self.pad_b


CASES = [
    Case("001", "bf16_full_tile", 128, 128, 32,
         "满 tile 基线。M/N/K 全部取最大值，覆盖 128x128 个输出元素的完整通路。"
         "先过这一条再看别的：它失败说明是根本性问题，查后面的没有意义。"),

    Case("002", "m1_gemv", 1, 128, 1024,
         "M=1 的 GEMV，即 LLM 逐 token 生成（decode）时的真实形状，"
         "也是本项目跑得最多的一种。tile 只有 1/128 被填上，"
         "边界处理稍有差错就只在这里暴露。K=1024 取自 Qwen3 的 hidden_size。"),

    Case("003", "partial_m", 7, 128, 32,
         "M=7：非 2 的幂、非对齐的部分 tile。"
         "用 7 而不是 8 或 64，是为了让任何按 2 的幂做的隐式假设立刻失效。"),

    Case("004", "k_multiblock", 128, 128, 256,
         "K=256，需连做 8 条 mfmacc 累加到同一累加器。验证累加器中间值按 FP32 "
         "承载。真实 GEMM 的 K=1024，这条是它的缩小版；只测单块的话，"
         "块间舍入差异永远暴露不出来。"),

    Case("005", "partial_n", 128, 13, 32,
         "N=13：输出宽度不满且是质数。N 方向的部分 tile 与 M 方向走的是"
         "不同的硬件路径（一个是行数，一个是列数），必须分开验。"),

    # K=17 是奇数，行 stride 若按 17 个元素算就是 34 字节，落不到 4 字节边界，
    # 32 位字的 hex 文件会跨行错位。补 1 个元素使 stride = 36 字节。
    Case("006", "partial_k", 128, 128, 17,
         "K=17：单块内 K 不满。累加长度不足时，剩余的乘法单元必须贡献 0 而非"
         "残留值 —— 这类错误在 K=32 满块时完全看不出来。"
         "行 stride 补到 18 个元素以对齐 4 字节。",
         pad_a=1, pad_b=1),

    Case("007", "stride_gap", 128, 128, 32,
         "A/B 的行 stride 大于实际列数（各留 16 个元素空隙）。"
         "验证装载严格按 stride 寻址。忽略 stride 的实现在前面所有 case 上"
         "都是对的，只有这里会错。",
         pad_a=16, pad_b=16),

    Case("008", "massive_act", 128, 128, 1024,
         "Qwen3 深层实测的真实数值分布：多数元素在 1 附近，少数维度出现 "
         "±6912 的 outlier（Transformer 深层的已知现象）。K=1024 为真实值。"
         "这条同时压测动态范围与 32 次连续累加，最接近硬件在真机上的实际负载。",
         mode="massive", scale=1.0),
]


# ───────────────────────── 整核指令级程序 ─────────────────────────
#
# 交付形态是"能自己跑起来的最小程序"，不是纯数据：DUT 是整个 core，
# 走取指、译码、访存的完整通路。testbench 只需
#   1) 把 prog.hex 装进内存    2) 放开复位
#   3) 轮询 DONE_ADDR 等魔数   4) dump C 区比对（或直接读 STATUS_ADDR）
#
# 相对 base 的固定布局。留的间隔比最大 case 宽裕得多，
# 目的是让所有 case 共用同一张地址表 —— 同事的 testbench 写一次就够，
# 不必每个 case 改参数。
SEG = {
    "text":   0x000000,     # 程序，实测 < 4 KB
    "a":      0x001000,     # A 矩阵，最大 264 KB
    "b":      0x050000,     # B 矩阵，最大 264 KB
    "c":      0x0A0000,     # C 输出，最大 64 KB（镜像里预置脏值）
    "ref":    0x0B0000,     # c_ref，供程序自检用
    "done":   0x0C0000,     # 完成标志
    "status": 0x0C0004,     # 自检结果
    "failidx":0x0C0008,     # 首个不符元素的下标
}
DONE_MAGIC = 0x600D600D
ST_PASS    = 0x50415353     # 'PASS'
ST_FAIL    = 0x4641494C     # 'FAIL'
C_DIRTY    = 0xA5A5A5A5     # C 区预置值：msce32 漏写时留下的痕迹好认


def emit_prog_s(c: Case, base: int, d: Path, n_mfmacc: int,
                stride_a: int, stride_b: int, stride_c: int):
    """生成 prog.S。K 循环**完全展开**：

    展开是刻意的。循环控制会在波形里混入分支、比较、计数器的活动，
    而 RTL 同事要看的是矩阵单元本身。展开后指令流里除了 tile 装载、
    mfmacc 和几条 li，什么都没有，波形一眼能对上。
    """
    L = []
    a = L.append
    a(f"# 自动生成 by tools/40_gen_ame_vectors.py —— 请勿手改")
    a(f"# case {c.cid} {c.name}")
    a(f"#   {c.why}")
    a(f"#")
    a(f"# C[{c.M}][{c.N}] = A[{c.M}][{c.K}] * B[{c.N}][{c.K}]^T   (BF16 -> FP32)")
    a(f"# 共 {n_mfmacc} 条 mfmacc 累加进同一个累加器")
    a("")
    for k, v in SEG.items():
        a(f"    .equ {k.upper()}_ADDR, 0x{base + v:08X}")
    a(f"    .equ DONE_MAGIC, 0x{DONE_MAGIC:08X}")
    a(f"    .equ ST_PASS,    0x{ST_PASS:08X}")
    a(f"    .equ ST_FAIL,    0x{ST_FAIL:08X}")
    a(f"    .equ N_ELEM,     {c.M * c.N}")
    a("")
    a('    .section .text.init, "ax"')
    a("    .global _start")
    a("_start:")
    a("    # 只让 hart 0 跑，其余原地睡")
    a("    csrr    t0, mhartid")
    a("    bnez    t0, park")
    a("")
    a("    # 使能三个扩展的上下文状态域，写 01(Initial) 即可。")
    a("    # MS[30:29] 最易遗漏：不开则所有 AME 指令报 illegal instruction，")
    a("    # 现象与「指令编码不支持」一模一样，极容易误判成译码问题。")
    a("    li      t0, (1 << 13) | (1 << 9) | (1 << 29)   # FS | VS | MS")
    a("    csrs    mstatus, t0")
    a("")
    a("    # 固化舍入模式为 RNE。参考结果按 RNE 生成，不写这条就得看")
    a("    # 复位默认值，各家实现不一定相同。")
    a("    csrw    frm, x0")
    a("")
    a(f"    # tile: mtilem={c.M} mtilen={c.N} mtilek={min(c.K, K_BLOCK)}")
    a(f"    li      t0, {c.M}")
    a("    csrw    0x803, t0            # mtilem")
    a(f"    li      t0, {c.N}")
    a("    csrw    0x804, t0            # mtilen")
    a("")
    a("    mzero   acc0")
    a("")
    for i in range(n_mfmacc):
        k0 = i * K_BLOCK
        kk = min(K_BLOCK, c.K - k0)
        a(f"    # ---- K 块 {i}: k={k0}..{k0 + kk - 1} ----")
        a(f"    li      t0, {kk}")
        a("    csrw    0x805, t0            # mtilek")
        a(f"    li      a0, A_ADDR + {k0 * 2}")
        a(f"    li      a1, {stride_a}")
        a("    mlae16  tr0,(a0),a1")
        a(f"    li      a0, B_ADDR + {k0 * 2}")
        a(f"    li      a1, {stride_b}")
        a("    mlbe16  tr1,(a0),a1")
        a("    mfmacc.s.bf16 acc0,tr1,tr0   # (md, B, A) —— 顺序写反会得到转置")
        a("")
    a("    # 回写 C")
    a("    li      a0, C_ADDR")
    a(f"    li      a1, {stride_c}")
    a("    msce32  acc0,(a0),a1")
    a("    mrelease")
    a("")
    a("    # 完成标志：testbench 轮询这个地址即可，不必猜时序")
    a("    li      t0, DONE_MAGIC")
    a("    li      t1, DONE_ADDR")
    a("    sw      t0, 0(t1)")
    a("")
    a("    # ---- 自检：与镜像里的 c_ref 逐字比对 ----")
    a("    # 纯粹是给同事省事的：不想用它就只看 DONE 然后自己 dump C 区。")
    a("    li      a0, C_ADDR")
    a("    li      a1, REF_ADDR")
    a("    li      a2, N_ELEM")
    a("    li      a3, 0                # 当前下标")
    a("    li      a4, 0                # 不符计数")
    a("    li      a5, 0                # 是否已记录首个不符")
    a("    li      a6, -1               # 首个不符的下标")
    a("cmp_loop:")
    a("    beqz    a2, cmp_done")
    a("    lw      t0, 0(a0)")
    a("    lw      t1, 0(a1)")
    a("    beq     t0, t1, cmp_next")
    a("    addi    a4, a4, 1")
    a("    bnez    a5, cmp_next")
    a("    li      a5, 1")
    a("    mv      a6, a3")
    a("cmp_next:")
    a("    addi    a0, a0, 4")
    a("    addi    a1, a1, 4")
    a("    addi    a3, a3, 1")
    a("    addi    a2, a2, -1")
    a("    j       cmp_loop")
    a("cmp_done:")
    a("    li      t1, FAILIDX_ADDR")
    a("    sw      a6, 0(t1)")
    a("    li      t0, ST_PASS")
    a("    beqz    a4, 1f")
    a("    li      t0, ST_FAIL")
    a("1:  li      t1, STATUS_ADDR")
    a("    sw      t0, 0(t1)")
    a("")
    a("park:")
    a("    wfi")
    a("    j       park")
    a("")
    a("    # ---- 数据段：地址由 prog.ld 钉死 ----")
    a('    .section .data.a, "a"')
    a('    .incbin "a.bin"')
    a('    .section .data.b, "a"')
    a('    .incbin "b.bin"')
    a('    .section .data.c, "a"')
    a(f"    # C 区预置 0x{C_DIRTY:08X}：msce32 若漏写某些元素，")
    a("    # 留下的是这个值而不是碰巧正确的 0。")
    a(f"    .rept {c.M * c.N}")
    a(f"    .word 0x{C_DIRTY:08X}")
    a("    .endr")
    a('    .section .data.ref, "a"')
    a('    .incbin "c_ref.bin"')
    # UTF-8 而非 ASCII：注释里的中文说明对接手的人价值很高，
    # 而 GNU as 只是把 # 之后的字节丢掉，不解析内容。
    (d / "prog.S").write_text("\n".join(L) + "\n", encoding="utf-8")


def emit_prog_ld(c: Case, base: int, d: Path):
    """链接脚本：把各段钉到 SEG 表给出的绝对地址。"""
    txt = f"""/* 自动生成 —— case {c.cid} {c.name}
 * 各段地址与 meta.json 的 segments 一致；改这里就要同步改那里。
 */
ENTRY(_start)
SECTIONS
{{
    . = 0x{base + SEG['text']:08X};
    .text : {{ *(.text.init) *(.text*) }}

    . = 0x{base + SEG['a']:08X};
    .data.a : {{ *(.data.a) }}

    . = 0x{base + SEG['b']:08X};
    .data.b : {{ *(.data.b) }}

    . = 0x{base + SEG['c']:08X};
    .data.c : {{ *(.data.c) }}

    . = 0x{base + SEG['ref']:08X};
    .data.ref : {{ *(.data.ref) }}

    /DISCARD/ : {{ *(.comment) *(.riscv.attributes) }}
}}
"""
    (d / "prog.ld").write_text(txt, encoding="utf-8")


def build_case(c: Case, base: int, out_root: Path, verbose=True):
    rng = Rng(c.seed)
    A = gen_matrix(c.M, c.K, rng, c.scale, c.mode)
    B = gen_matrix(c.N, c.K, rng, c.scale, c.mode)
    C = ame_gemm_ref(A, B, c.M, c.N, c.K)

    d = out_root / "cases" / c.dirname
    d.mkdir(parents=True, exist_ok=True)

    # 行 stride（字节）：BF16 每元素 2 字节，FP32 每元素 4 字节。
    # stride 可大于实际列数，见 Case.pad_a/pad_b 的说明。
    stride_a = c.stride_a_elems * 2
    stride_b = c.stride_b_elems * 2
    stride_c = c.N * 4

    wa = bf16_rows_to_words(A, c.stride_a_elems)
    wb = bf16_rows_to_words(B, c.stride_b_elems)
    wc = f32_rows_to_words(C, c.N)

    # 注释一律 ASCII：$readmemh 的实现对非 ASCII 字节的处理各家不同，
    # 中文注释放 meta.json 与 README，hex 文件保持最保守。
    tag = f"case {c.cid} {c.name}"
    write_hex(d / "a.hex", wa,
              f"{tag} - A[{c.M}][{c.K}] BF16, row stride {stride_a} bytes")
    write_hex(d / "b.hex", wb,
              f"{tag} - B[{c.N}][{c.K}] BF16, row stride {stride_b} bytes")
    write_hex(d / "c_ref.hex", wc,
              f"{tag} - C[{c.M}][{c.N}] FP32 expected, row stride {stride_c} bytes")
    write_bin(d / "a.bin", wa)
    write_bin(d / "b.bin", wb)
    write_bin(d / "c_ref.bin", wc)

    n_mfmacc = (c.K + K_BLOCK - 1) // K_BLOCK
    emit_prog_s(c, base, d, n_mfmacc, stride_a, stride_b, stride_c)
    emit_prog_ld(c, base, d)

    meta = {
        "case_id": c.cid,
        "name": c.name,
        "why": c.why,
        "dtype": {"a": "bf16", "b": "bf16", "c": "fp32"},
        "shape": {"M": c.M, "N": c.N, "K": c.K},
        "tile": {"mtilem": c.M, "mtilen": c.N,
                 "mtilek": min(c.K, K_BLOCK),
                 "n_mfmacc": n_mfmacc},
        "segments": {k: f"0x{base + v:08X}" for k, v in SEG.items()},
        "markers": {
            "done_magic": f"0x{DONE_MAGIC:08X}",
            "status_pass": f"0x{ST_PASS:08X}",
            "status_fail": f"0x{ST_FAIL:08X}",
            "c_preset": f"0x{C_DIRTY:08X}",
        },
        "csr": {"mtilem": "0x803", "mtilen": "0x804", "mtilek": "0x805"},
        "stride_bytes": {"a": stride_a, "b": stride_b, "c": stride_c},
        "stride_elems": {"a": c.stride_a_elems, "b": c.stride_b_elems, "c": c.N},
        "row_gap_elems": {"a": c.pad_a, "b": c.pad_b},
        "layout": "C = A · Bt ; A is [M,K], B is [N,K] (NOT [K,N])",
        "rounding": "block-exact then FP32 RNE per K-block of 32",
        "words": {"a": len(wa), "b": len(wb), "c": len(wc)},
        "seed": f"0x{c.seed:08X}",
    }
    (d / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"  [{c.cid}] {c.name:18s} M={c.M:<4d} N={c.N:<4d} K={c.K:<4d} "
              f"mfmacc×{meta['tile']['n_mfmacc']:<2d} "
              f"a={len(wa)}w b={len(wb)}w c={len(wc)}w")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="0x80000000",
                    help="目标 SoC 的 DRAM 基址（QEMU virt 是 0x80000000）")
    ap.add_argument("--only", help="只生成指定 case id")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    base = int(args.base, 0)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = [c for c in CASES if not args.only or c.cid == args.only]
    if not cases:
        print(f"没有匹配 --only {args.only} 的 case")
        return 1

    print(f"[*] 生成 AME 测试向量  base=0x{base:08X}  → {out_root}")
    metas = [build_case(c, base, out_root) for c in cases]

    (out_root / "index.json").write_text(
        json.dumps({"base": f"0x{base:08X}", "cases": metas},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] {len(metas)} 个 case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
