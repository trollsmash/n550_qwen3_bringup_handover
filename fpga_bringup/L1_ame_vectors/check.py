#!/usr/bin/env python3
"""比对 RTL 仿真 dump 出来的 C 区与本包的期望值。

用法:
    python3 check.py 001 my_dump.hex          # hex，每行 32 bit
    python3 check.py 001 my_dump.bin          # 原始小端字节
    python3 check.py 001 my_dump.hex -v       # 附带前若干个不符元素的明细

只依赖标准库，Python 3.8+ 即可，不需要 numpy。

本脚本的重点不是"报个 PASS/FAIL"，而是**把失败模式翻译成原因**。
下面 diagnose() 里的每一条都对应一类真实踩过的坑，读一遍能省很多时间。
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIRTY = 0xA5A5A5A5        # 镜像里 C 区的预置值，见 meta.json 的 markers


def load_words(path: Path):
    """读 .hex（每行 32 位十六进制，允许 // 与 /* */ 注释）或 .bin（小端）。

    判别方式是"剥掉注释后能否整份解析成十六进制"，而不是看扩展名或
    字符白名单：注释里出现逗号、括号、中文都很正常，按白名单判别会把
    正经的 hex 文件误当成二进制。
    """
    data = path.read_bytes()

    text = None
    for enc in ("utf-8", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is not None:
        body = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)     # /* 块注释 */
        body = re.sub(r"//[^\n]*", " ", body)                   # // 行注释
        toks = body.split()
        if toks and all(re.fullmatch(r"[0-9a-fA-F_]{1,8}", t) for t in toks):
            return [int(t.replace("_", ""), 16) & 0xFFFFFFFF for t in toks]

    if len(data) % 4:
        sys.exit(f"{path} 既不是合法的 hex 文本，长度 {len(data)} 也不是 4 的倍数")
    return list(struct.unpack(f"<{len(data)//4}I", data))


def bits_f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def ulp_diff(x, y):
    """两个 FP32 位模式相差几个 ULP。异号返回 None（不按 ULP 论）。"""
    if x == y:
        return 0
    if (x >> 31) != (y >> 31):
        return None
    a, b = x & 0x7FFFFFFF, y & 0x7FFFFFFF
    return abs(a - b)


def find_case(cid):
    hits = sorted((HERE / "cases").glob(f"{cid}*"))
    if not hits:
        sys.exit(f"找不到 case '{cid}'，可选：" +
                 ", ".join(p.name for p in sorted((HERE / 'cases').iterdir())))
    return hits[0]


def diagnose(got, ref, M, N, dirty_n, zero_n):
    """把失败模式翻译成可能的原因。每条都来自真实踩过的坑。"""
    out = []
    total = M * N

    if dirty_n == total:
        out.append("输出区**原封未动**（全是镜像预置的 0xA5A5A5A5）。"
                   "msce32 没有执行，或写到了别的地址。"
                   "先确认程序真的跑到了结尾：读 DONE 标志。")
        return out
    if dirty_n:
        out.append(f"有 {dirty_n} 个元素仍是预置值 0xA5A5A5A5 —— "
                   f"msce32 **漏写了一部分**。若漏的是尾部，多半是 mtilem/mtilen "
                   f"比预期小；若漏的是零散位置，查回写的 stride。")

    if zero_n == total:
        out.append("输出**全为 0**。累加器被清零后没有累加进任何东西："
                   "检查 mfmacc 是否真的发射了，以及 tr0/tr1 是否装载成功。")
        return out

    # 逐行/逐列统计正确率，能一眼看出是行的问题还是列的问题
    row_ok = [sum(1 for n in range(N) if got[m*N+n] == ref[m*N+n]) for m in range(M)]
    col_ok = [sum(1 for m in range(M) if got[m*N+n] == ref[m*N+n]) for n in range(N)]

    good_rows = [m for m in range(M) if row_ok[m] == N]
    good_cols = [n for n in range(N) if col_ok[n] == M]

    if len(good_rows) == 1 and good_rows[0] == 0 and M > 1:
        out.append("**只有第 0 行正确**，其余行全错。典型症状是 mtilem 没生效，"
                   "矩阵单元只算了一行。QEMU 早期版本正是这个缺陷。")
    elif 0 < len(good_rows) < M:
        out.append(f"{len(good_rows)}/{M} 行完全正确。"
                   f"若正好是前一半，查累加器清零是否只覆盖了一半"
                   f"（QEMU 的 mzero 曾只清前 64 行）。")

    if 0 < len(good_cols) < N and len(good_cols) != N:
        out.append(f"{len(good_cols)}/{N} 列完全正确 —— 查 mtilen 与回写 stride。")

    # 转置：只有方阵才能这样比
    if M == N:
        t_hit = sum(1 for m in range(M) for n in range(N)
                    if got[m*N+n] == ref[n*N+m])
        if t_hit > total * 0.9:
            out.append("结果与**期望值的转置**高度吻合。"
                       "mfmacc 的操作数顺序写反了：正确写法是 "
                       "`mfmacc.s.bf16 md, B, A` —— 第二个操作数放右矩阵、"
                       "第三个放左矩阵。")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="case id 或目录名，如 001 或 001_bf16_full_tile")
    ap.add_argument("dump", help="RTL dump 出来的 C 区（.hex 或 .bin）")
    ap.add_argument("-v", "--verbose", action="store_true", help="列出不符元素明细")
    ap.add_argument("--max-show", type=int, default=10)
    args = ap.parse_args()

    d = find_case(args.case)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    M, N = meta["shape"]["M"], meta["shape"]["N"]
    total = M * N

    ref = load_words(d / "c_ref.hex")
    got = load_words(Path(args.dump))

    print(f"case      : {meta['case_id']} {meta['name']}")
    print(f"形状      : M={M} N={N} K={meta['shape']['K']}  "
          f"mfmacc x{meta['tile']['n_mfmacc']}")
    print(f"期望元素数: {total}")
    print(f"你的 dump : {len(got)} 个字  ({args.dump})")

    if len(got) < total:
        sys.exit(f"\n★ dump 只有 {len(got)} 个字，不足 {total} 个。"
                 f"\n  C 区起始地址是 {meta['segments']['c']}，"
                 f"需要 dump {total*4} 字节。")
    if len(got) > total:
        print(f"  (多出的 {len(got)-total} 个字将被忽略)")
        got = got[:total]

    n_exact = n_1ulp = n_bad = 0
    dirty_n = zero_n = 0
    worst = 0
    bad = []
    for i in range(total):
        r, g = ref[i], got[i]
        if g == C_DIRTY:
            dirty_n += 1
        if g == 0:
            zero_n += 1
        d_ulp = ulp_diff(r, g)
        if d_ulp == 0:
            n_exact += 1
        elif d_ulp == 1:
            n_1ulp += 1
        else:
            n_bad += 1
            worst = max(worst, d_ulp if d_ulp is not None else 1 << 30)
            if len(bad) < args.max_show:
                bad.append((i, r, g, d_ulp))

    print()
    print(f"逐位相同  : {n_exact} / {total}")
    print(f"差 1 ULP  : {n_1ulp}   （真值落在两个 FP32 中点，可接受）")
    print(f"差 >1 ULP : {n_bad}")

    if n_bad == 0:
        if n_1ulp:
            print("\n>>> PASS（含中点舍入差异）")
            print("    这类差异出现在真值恰好落在两个 FP32 正中间时。"
                  "参考值按 IEEE RNE（最近偶数）生成。")
        else:
            print("\n>>> PASS  完全一致")
        return 0

    print(f"\n>>> FAIL  最大偏差 {worst} ULP")
    if args.verbose or bad:
        print("\n前几个不符元素：")
        print("   下标   (m, n)      期望        实测       差(ULP)")
        for i, r, g, du in bad:
            m, n = divmod(i, N)
            print(f"  {i:6d}  ({m:3d},{n:3d})  0x{r:08x}  0x{g:08x}  "
                  f"{'异号' if du is None else du}")
            if args.verbose:
                print(f"           期望={bits_f32(r):.9g}  实测={bits_f32(g):.9g}")

    hints = diagnose(got, ref, M, N, dirty_n, zero_n)
    if hints:
        print("\n可能的原因：")
        for h in hints:
            print(f"  * {h}")
    print("\n（若以上都不像，把本输出连同 prog.hex 的 md5 发回软件侧）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
