#!/usr/bin/env bash
# 把 40_gen_ame_vectors.py 生成的每个 case 汇编成可运行镜像。
#
#   source tools/env.sh
#   tools/41_build_vectors.sh              # 编译全部 case
#   tools/41_build_vectors.sh 001          # 只编译一个
#   RUN=1 tools/41_build_vectors.sh 001    # 编译并在 qemu-system 上跑一遍
#
# 产物（放在各 case 目录里，随包交付）：
#   prog.elf   带符号，便于同事用 objdump 看指令流
#   prog.bin   从 BASE 起的连续内存镜像
#   prog.hex   同上，每行 32 bit，$readmemh 直接可读
#
# prog.hex 是**完整镜像**：代码、A、B、C(预置脏值)、c_ref 全在里面，
# 一次 $readmemh 就位。不想要自检数据的话，按 meta.json 的 segments
# 只取前面几段即可。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${RISCV_ISA:-}" ]; then
    echo "错误：请先 source tools/env.sh" >&2
    exit 1
fi

PKG="deliverables/ame_vectors_v1"
[ -d "$PKG/cases" ] || { echo "先跑 python3 tools/40_gen_ame_vectors.py" >&2; exit 1; }

ONLY="${1:-}"
ok=0; fail=0

for d in "$PKG"/cases/*/; do
    name="$(basename "$d")"
    [ -z "$ONLY" ] || [[ "$name" == "$ONLY"* ]] || continue

    # -nostdlib 等一串：不要 crt0、不要 libc，镜像里只有我们写的指令。
    # 在 case 目录里编译，.incbin 才能用相对路径找到 a.bin/b.bin。
    ( cd "$d" && ${CROSS}gcc -march="$RISCV_ISA" -mabi=lp64d -mcmodel=medany \
        -ffreestanding -nostdlib -nostartfiles -Wl,--build-id=none \
        -T prog.ld prog.S -o prog.elf ) 2>&1 \
        | grep -viE "LOAD segment with RWX" || true

    if [ ! -f "$d/prog.elf" ]; then
        echo "  [FAIL] $name  汇编失败"; fail=$((fail+1)); continue
    fi

    ${CROSS}objcopy -O binary --gap-fill 0 "$d/prog.elf" "$d/prog.bin"

    # bin → hex（每行一个 32 位小端字）。用 od 而非自写循环：
    # 40 万行的纯 bash 转换会慢到无法接受。
    { echo "// $name — full memory image from BASE, 32-bit words, little-endian"
      od -An -tx4 -v -w4 "$d/prog.bin" | tr -d ' ' | grep -v '^$'
    } > "$d/prog.hex"

    sz=$(stat -c%s "$d/prog.bin")
    lines=$(( $(wc -l < "$d/prog.hex") - 1 ))
    printf "  [ OK ] %-22s prog.bin %8d B   prog.hex %7d words\n" "$name" "$sz" "$lines"
    ok=$((ok+1))

    if [ "${RUN:-0}" = "1" ]; then
        # 在 qemu-system 上跑一遍，回读程序自己写的标志字。
        # 没有串口可用（交付镜像刻意不依赖任何外设），所以走 QEMU monitor
        # 读物理内存 —— 这与 RTL testbench 的做法是同构的：等标志、读内存。
        # sleep 是给 QEMU 启动和程序跑完留的时间；K=1024 的 case 稍慢。
        st=$( { sleep "${WAIT:-6}"; echo "xp/3xw ${DONE_ADDR:-0x800C0000}"; echo quit; } | \
            timeout 180 "$QEMU_SYSTEM_RISCV64" -machine virt -bios none -m 512M \
                -cpu rv64,matrix=true,v=true,vlen=1024 -kernel "$d/prog.elf" \
                -display none -serial none -monitor stdio 2>/dev/null \
            | tr -d '\r' | grep -oE '0x600d600d 0x[0-9a-f]+ 0x[0-9a-f]+' | tail -1 )
        case "$st" in
            *0x50415353*) echo "        自检 PASS" ;;
            *0x4641494c*) echo "        ★自检 FAIL  ($st)"; fail=$((fail+1)); ok=$((ok-1)) ;;
            *)            echo "        ★未跑完或标志未写：'$st'"; fail=$((fail+1)); ok=$((ok-1)) ;;
        esac
    fi
done

echo
echo "汇编完成：$ok 个成功，$fail 个失败"
[ "$fail" -eq 0 ]
