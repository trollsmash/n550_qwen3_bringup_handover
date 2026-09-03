"""扫描每条 AME 指令的**所有可能前驱**（含跳转来源），
检查是否存在「前驱是非 AME 指令且与本条无数据相关」的路径。

相比上一版的改进：
  - 无条件跳转(j/jr/ret/tail)之后的指令没有 fall-through 前驱
  - 跳转目标要把所有 b*/j 的来源算进去
  - 按 AME 指令类型分组，因为它们读的通用寄存器不同
"""
import re
import subprocess
import sys
from collections import defaultdict

AME_MNEM = re.compile(
    r"^(ml[abc](te|t|e)?\d*|ms[abc](te|t|e)?\d*|mlme\d*|msme\d*|"
    r"mfmacc\S*|mmacc\S*|mzero|mmov\S*|mbce\S*|mpack\S*|mrelease)$")
AME_CSR = {"mtilem", "mtilen", "mtilek", "xmcsr", "xmfrm",
           "xmsaten", "xmerr", "xmisa", "xtlenb", "xtrlenb", "xalenb"}

COND_BR = re.compile(r"^(beq|bne|blt|bge|bltu|bgeu|beqz|bnez|blez|bgez|"
                     r"bltz|bgtz|bgt|ble|bgtu|bleu)$")
UNCOND  = re.compile(r"^(j|jr|jal|jalr|ret|tail)$")
STORE   = re.compile(r"^(sb|sh|sw|sd|fsw|fsd|c\.sw|c\.sd|c\.swsp|c\.sdsp)$")
LINE    = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f ]+)\s+\t(\S+)\s*(.*)$")
REG     = r"\b(?:zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b"


def regs_of(ops):
    return re.findall(REG, ops)


def is_ame(mnem, ops):
    if AME_MNEM.match(mnem):
        return True
    if mnem.startswith("csr"):
        return ops.split(",")[0].strip() in AME_CSR
    return False


def dest_reg(mnem, ops):
    if COND_BR.match(mnem) or STORE.match(mnem) or mnem.startswith("csrw"):
        return None
    if mnem in ("nop", "unimp", "ecall", "ebreak") or mnem.startswith("fence"):
        return None
    if mnem in ("j", "jr", "ret", "tail"):
        return None
    r = regs_of(ops)
    return r[0] if r else None


def src_regs(mnem, ops):
    if mnem.startswith("csr"):
        parts = [p.strip() for p in ops.split(",")]
        return regs_of(",".join(parts[1:]))
    return regs_of(ops)


def kind(mnem, ops):
    if mnem.startswith("csr"):
        return "CSR 写 " + ops.split(",")[0].strip()
    if mnem.startswith(("mla", "mlb", "mlc", "mlm")):
        return "矩阵 load"
    if mnem.startswith(("msa", "msb", "msc", "msm")):
        return "矩阵 store"
    if mnem.startswith("mfmacc") or mnem.startswith("mmacc"):
        return "矩阵乘加"
    return "矩阵 misc"


def main(elf, cross):
    out = subprocess.run([cross + "objdump", "-d", elf],
                         capture_output=True, text=True).stdout
    ins = []
    for ln in out.splitlines():
        m = LINE.match(ln)
        if m:
            ins.append((int(m.group(1), 16), m.group(3), m.group(4).strip()))
    idx = {a: i for i, (a, _, _) in enumerate(ins)}

    # 跳转来源：目标地址 -> [来源下标]
    tgt_src = defaultdict(list)
    for i, (a, mn, ops) in enumerate(ins):
        if COND_BR.match(mn) or mn in ("j", "jal"):
            m = re.search(r"\b([0-9a-f]{6,})\b", ops)
            if m:
                t = int(m.group(1), 16)
                if t in idx:
                    tgt_src[t].append(i)

    rows = []
    for i, (a, mn, ops) in enumerate(ins):
        if not is_ame(mn, ops):
            continue
        s = src_regs(mn, ops)
        preds = []
        # fall-through：前一条不是无条件跳转时才成立
        if i > 0:
            pa, pm, po = ins[i - 1]
            if not UNCOND.match(pm):
                preds.append(("顺序", i - 1))
        for j in tgt_src.get(a, []):
            preds.append(("跳转自", j))
        for tag, j in preds:
            pa, pm, po = ins[j]
            if is_ame(pm, po):
                continue
            d = dest_reg(pm, po)
            if d is not None and d in s:
                continue                      # 有数据相关，硬件会等
            rows.append((a, mn, ops, kind(mn, ops), tag, pa, pm, po, d, s))
    return len(ins), rows


if __name__ == "__main__":
    total, rows = main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
                       else "riscv-unknown-elf-")
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r[3]].append(r)
    print("  扫描 %d 条指令，命中 %d 处（按类型分组）" % (total, len(rows)))
    for k in sorted(by_kind):
        print()
        print("  【%s】%d 处" % (k, len(by_kind[k])))
        for a, mn, ops, _k, tag, pa, pm, po, d, s in by_kind[k]:
            print("    0x%08x %-26s <- %s 0x%08x %-22s  前写:%-5s 本读:%s"
                  % (a, (mn + " " + ops)[:26], tag, pa,
                     (pm + " " + po)[:22], d or "-",
                     "/".join(s) if s else "-"))
