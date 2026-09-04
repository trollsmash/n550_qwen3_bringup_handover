#!/usr/bin/env bash
# RISC-V 交叉编译 + QEMU 运行。
#
#   source tools/env.sh
#   tools/build_riscv.sh                    # 编译全模型
#   tools/build_riscv.sh run 5              # 编译并跑 5 步
#   tools/build_riscv.sh smoke              # 编译并跑 AME 冒烟测试
#   tools/build_riscv.sh KERNEL=ame run 5   # 换 GEMM kernel 实现
#
# 产物放 $RISCV_BUILD（WSL 原生盘）—— /tmp 是 tmpfs，实例回收后文件会消失。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${RISCV_ISA:-}" ]; then
    echo "错误：请先 source tools/env.sh" >&2
    exit 1
fi

KERNEL="${KERNEL:-scalar}"
OUT="$RISCV_BUILD"

# 从 src/bsp/board.h 展开取地址常量，供各目标打印部署提示用。
# 一律走这里，绝不在脚本里手抄地址 —— 抄一次就会有失同步的一天，
# 而且是"二进制里对、提示里错"这种最能骗过人的失同步（已经发生过一次：
# UART 被写成 0x00201000，少了两位）。
# 调用前 BOARD_DEF 必须已设好。
# -D__ASSEMBLER__ 让 board.h 跳过 #include <stdint.h>，
# 否则展开结果里混进整个 stdint.h 的 typedef，无法参与算术。
bval() {
    printf '%s\n' "$1" \
        | ${CROSS}cpp -Isrc ${BOARD_DEF:-} -D__ASSEMBLER__ -P \
                     -include bsp/board.h - 2>/dev/null \
        | grep -v '^\s*$' | tail -1 | tr -d ' '
}
bhex() { printf "0x%08X" "$(( $(bval "$1") ))"; }

# BOARD -> 预处理器宏。两个目标共用。
board_def() {
    case "${BOARD:-qemu}" in
    qemu) BOARD_DEF="-DBOARD_QEMU_VIRT" ;;
    s2c)  BOARD_DEF="-DBOARD_S2C" ;;
    *)    echo "未知 BOARD=$BOARD（可选 qemu / s2c）" >&2; exit 1 ;;
    esac
    # 版本标识：让串口第一行就能认出跑的是哪一版镜像。
    # 用**十六进制**字面量 0xMMDDhhmm，打印出来直接读作 月日-时分。
    # 不能写成十进制：08 开头会被 C 当八进制，而 08/09 是非法八进制。
    BOARD_DEF="$BOARD_DEF -DBOARD_BUILD_ID=0x$(date +%m%d%H%M)"
}

# -nostdlib：不链 libc（newlib 的 IO 走 semihosting，user-mode QEMU 不认）
#            平台层 main_qemu.c 直接发 Linux syscall
# -lm      ：链 newlib 的 libm（sqrtf/expf/sinf/cosf/powf），
#            配 main_qemu.c 里的 __errno 桩即可满足链接
# string_min.c：提供 memcpy/memset —— 编译器在任何优化级别都可能自行生成这些调用
CFLAGS_ALL="$RISCV_CFLAGS -ffreestanding -nostdlib -nostartfiles -Wl,-e,_start"

# 从 ELF 派生调试辅助文件。纯派生物，不影响烧录的 bin。
#
# 真机出错时手里通常只有一个 mepc（trap handler 打出来的），
# 没有这两个文件就只能靠猜：
#   .diss  源码交织反汇编 —— 按地址直接查到出错的指令与对应源码行
#   .sym   按地址排序的符号表 —— 先看 mepc 落在哪个函数
# 也可以用 ${CROSS}addr2line -e <elf> <mepc>，但那要求手边装了工具链，
# 上板调试的同事未必有。
emit_debug() {
    local elf="$1" base="${1%.elf}"
    ${CROSS}objdump -d -S "$elf" > "$base.diss" 2>/dev/null || true
    ${CROSS}nm -n "$elf"         > "$base.sym"  2>/dev/null || true
}
OPS="${OPS:-scalar}"
BSP="src/bsp/string_min.c src/bsp/libm_stub.c"
SRCS="src/qwen3.c src/tokenizer.c src/kernels_${KERNEL}.c src/ops_${OPS}.c src/main_qemu.c $BSP"

build() {
    echo "== 交叉编译 (ISA=$RISCV_ISA, kernel=$KERNEL) =="
    # 先删旧产物：否则编译失败时下面的 -f 检查会被上一次的 elf 蒙混过去，
    # 报“成功”却跑的是旧二进制。
    rm -f "$OUT/qwen3.elf"
    # shellcheck disable=SC2086
    ${CROSS}gcc $CFLAGS_ALL $SRCS -o "$OUT/qwen3.elf" -lm 2>&1 \
        | grep -viE "LOAD segment with RWX" || true
    emit_debug "$OUT/qwen3.elf"
    [ -f "$OUT/qwen3.elf" ] || { echo "链接失败" >&2; exit 1; }
    local sz und
    sz=$(stat -c%s "$OUT/qwen3.elf")
    und=$(${CROSS}nm -u "$OUT/qwen3.elf" || true)
    printf "   %s  (%.1f KB)\n" "$OUT/qwen3.elf" "$(echo "$sz/1024" | bc -l)"
    echo "   未定义符号: ${und:-无}"
}

# 权重文件与其大小 —— 编译期就要用到（-DWEIGHTS_SIZE），所以在分支之前算好。
# WEIGHTS 可指向 tile-major 版；两种布局的文件大小不同，写死会让
# qwen3_init 在最后几个张量上报 QWEN3_ERR_SIZE。
WSRC="${WEIGHTS:-golden/qwen3-0.6b-bf16.bin}"
WSIZE_DEF=""
[ -f "$WSRC" ] && WSIZE_DEF="-DWEIGHTS_SIZE=$(stat -c%s "$WSRC")UL"

# 固件把权重总长编进了二进制，qwen3_init 会严格核对（c.pos != blob_size ->
# QWEN3_ERR_SIZE），所以「固件 + 权重」必须成套，不能混搭。
# 产物按布局具名让两套并存，打包脚本才能一次取到配对的那一份；
# 只用一个固定名的话，后编的会盖掉先编的，而且盖完看不出来。
# 布局从 header 偏移 76 读 —— 文件名可以改，header 不会。
WTAG=row
if [ -f "$WSRC" ]; then
    case "$(od -An -tu4 -j76 -N4 "$WSRC" 2>/dev/null | tr -d ' ')" in
        1) WTAG=tile ;;
    esac
fi

# 对话模式每次回答的 token 上限。源码里默认 1 —— 那是给 QEMU 的，
# 那里每多一个 token 就要再遍历一遍 1.1 GB 权重。真机 demo 要说完整的话，
# 用 CHAT_GEN=N 注入。源码侧按 KV cache 余量做了钳制，设大不会越界。
CHAT_GEN_DEF=""
[ -n "${CHAT_GEN:-}" ] && CHAT_GEN_DEF="-DCHAT_MAX_GEN=$CHAT_GEN"

# 上下文窗口（KV cache 容量）。qwen3.h 默认 256，x86 与 QEMU user-mode
# 沿用它；真机 demo 要多轮对话，用 MAX_SEQ=N 注入。
#
# 注意它按 N x 229 KB 线性吃 arena：512 -> 117 MB，而 arena 是 BSS 里的
# 静态数组，夹在程序镜像与 mailbox(0x87F00000) 之间，上限约 127 MB。
# 再往上要把 g_scratch 挪出 BSS，那会动到内存布局。
#
# 它**不影响** decode 速度：attention 只读到当前位置，MAX_SEQ 仅参与
# 地址计算。变慢的是实际聊得更长，不是容量设得更大。
MAXSEQ_DEF=""
[ -n "${MAX_SEQ:-}" ] && MAXSEQ_DEF="-DQWEN3_MAX_SEQ=$MAX_SEQ"

# 单次 prefill 的 token 上限。>128 时 AME 的 M 维要分多个 tile —— 
# gemm_impl 的 m0 循环本来就支持，但那条路径此前从未被触发过。
MAXBATCH_DEF=""
[ -n "${MAX_BATCH:-}" ] && MAXBATCH_DEF="-DQWEN3_MAX_BATCH=$MAX_BATCH"

# CLP=1 走 RVV 高速接口窗口（见 board.h 的 BOARD_CLP 段）：
#   激活与 KV cache 落在 0x1000_0000 窗口，RVV 绕 L1D 访问，
#   AME 用重映射后的 0xF000_0000，两者共享同一物理内存，
#   于是 GEMM 前后的缓存维护指令全部省掉。
# 产物名带 -clp 后缀，与 L4 的镜像并存、不互相覆盖 —— 两版要做对比。
CLP_DEF=""
CLP_TAG=""
if [ "${CLP:-0}" = "1" ]; then
    CLP_DEF="-DBOARD_CLP -DQWEN3_ATT_STATIC"
    CLP_TAG="-clp"
fi

case "${1:-build}" in
build) build ;;
run)
    build
    STEPS="${2:-5}"
    # 权重放原生盘：跨 /mnt/c 读 1.1GB 走 9p，首次加载明显更慢
    W="$OUT/w-$(basename "$WSRC" .bin).bin"
    [ -f "$W" ] || cp "$WSRC" "$W"
    # 打包脚本按固定名 w.bin 取权重。这里让它指向本次构建实际用的那一份，
    # 免得两处各自拼文件名而失同步 —— 那种错的表现是打包进了上一次布局的
    # 权重，镜像与权重对不上，却不会有任何提示。
    ln -sf "$(basename "$W")" "$OUT/w.bin"
    echo "== QEMU 运行 ($STEPS 步) =="
    ( cd "$OUT" && time "$QEMU_RISCV64" -cpu "$QEMU_CPU_FULL" ./qwen3.elf w.bin "$STEPS" )
    ;;
baremetal)
    # 目标形态：无 OS + AME + RVV。权重由 QEMU 的 generic loader 直接放进 DDR，
    # 模拟 FPGA 上 host 经 PCIe 后门写入的做法。
    # BOARD 选目标板：qemu(默认) 或 s2c。板级地址全在 src/bsp/board.h，
    # start.S 也 include 它，所以 UART 基址只存在一份。
    board_def
    echo "== 交叉编译 bare-metal (board=${BOARD:-qemu}, kernel=$KERNEL, ops=$OPS) =="
    BM_ELF="$OUT/qwen3_bm-$WTAG$CLP_TAG.elf"
    rm -f "$BM_ELF"
    # -Isrc 是为了让 start.S 能 #include "board.h"
    # shellcheck disable=SC2086
    ${CROSS}gcc $RISCV_CFLAGS -Isrc $BOARD_DEF $WSIZE_DEF $CHAT_GEN_DEF $MAXSEQ_DEF $MAXBATCH_DEF $CLP_DEF -ffreestanding -nostdlib -nostartfiles \
        -T src/bsp/qemu_virt.ld src/bsp/start.S \
        src/qwen3.c src/tokenizer.c src/kernels_${KERNEL}.c src/ops_${OPS}.c \
        src/main_baremetal.c $BSP \
        -o "$BM_ELF" -lm 2>&1 | grep -viE "LOAD segment with RWX" || true
    [ -f "$BM_ELF" ] || { echo "链接失败" >&2; exit 1; }
    emit_debug "$BM_ELF"
    # 固定名指向本次构建，既有引用（QEMU 运行、旧脚本）不受具名化影响。
    ln -sf "$(basename "$BM_ELF")"             "$OUT/qwen3_bm.elf"
    ln -sf "$(basename "${BM_ELF%.elf}.diss")" "$OUT/qwen3_bm.diss"
    ln -sf "$(basename "${BM_ELF%.elf}.sym")"  "$OUT/qwen3_bm.sym"
    printf "   %s  (%.1f KB, 布局 %s, 回答 %s, 上下文 %s, 单批 %s, CLP %s)\n" "$BM_ELF" \
        "$(echo "$(stat -c%s "$BM_ELF")/1024" | bc -l)" "$WTAG" \
        "${CHAT_GEN:-1}" "${MAX_SEQ:-256}" "${MAX_BATCH:-128}" "${CLP:-0}"
    printf "   %s  (反汇编，按 mepc 定位用)\n" "${BM_ELF%.elf}.diss"

    W="$OUT/w-$(basename "$WSRC" .bin).bin"
    [ -f "$W" ] || cp "$WSRC" "$W"
    # 打包脚本按固定名 w.bin 取权重。这里让它指向本次构建实际用的那一份，
    # 免得两处各自拼文件名而失同步 —— 那种错的表现是打包进了上一次布局的
    # 权重，镜像与权重对不上，却不会有任何提示。
    ln -sf "$(basename "$W")" "$OUT/w.bin"
    T="$OUT/tok.bin"
    [ -f "$T" ] || cp golden/tokenizer.bin "$T"

    # 真机不走 QEMU：板上没有 virt 的那套外设，跑了也只是空转。
    # 这里改为产出 host 经 PCIe 后门要写的三块裸镜像，并打印地址表。
    # ELF 不能直接灌进 DDR —— 必须 objcopy 成 raw binary。
    if [ "${BOARD:-qemu}" = "s2c" ]; then
        S2C_BIN="$OUT/qwen3_s2c-$WTAG$CLP_TAG.bin"
        ${CROSS}objcopy -O binary --gap-fill 0 "$BM_ELF" "$S2C_BIN"
        ln -sf "$(basename "$S2C_BIN")" "$OUT/qwen3_s2c.bin"
        echo
        echo "== S2C 部署镜像 (权重布局 $WTAG) =="
        printf "  %-16s -> %s   %12s B   程序\n" \
            "$(basename "$S2C_BIN")" "$(bhex BOARD_DRAM_BASE)" "$(stat -c%s "$S2C_BIN")"
        printf "  %-16s -> %s   %12s B   权重\n" \
            "w.bin"         "$(bhex BOARD_WEIGHTS_ADDR)"   "$(stat -c%s "$W")"
        printf "  %-16s -> %s   %12s B   分词器\n" \
            "tok.bin"       "$(bhex BOARD_TOKENIZER_ADDR)" "$(stat -c%s "$T")"
        echo "  产物目录: $OUT"
        echo
        echo "  上板顺序：三块写完 -> 放开复位 -> 轮询 mailbox $(bhex BOARD_MBOX_STATUS)"
        printf "  串口 %s (16550)；状态字取值见 src/bsp/board.h 的 BOARD_ST_*\n" \
            "$(bhex BOARD_UART_BASE)"
        exit 0
    fi
    # 两个 -device loader 把权重与分词器分别放进 DDR 的固定地址，
    # 正对应 FPGA 上 host 经 PCIe 后门写两块区域。
    # 地址必须与 src/main_baremetal.c 的 WEIGHTS_ADDR / TOKENIZER_ADDR 一致。
    # 改动时看那里的注释：权重区起点要留在程序镜像（含 BSS）末端之后。
    # ★ CPU 必须用 $QEMU_CPU_FULL，不能手写一个短的。
    # 这里曾经写成 "rv64,matrix=true,v=true,vlen=1024"，漏掉了 zfh/zfbfmin/
    # zvfh/zvfbfmin/zvfbfwma —— ops_rvv.c 里的 vfwcvtbf16 立刻非法指令。
    # 而当时 mtvec 未设，异常直接跳到地址 0 空转，看起来就像"跑得很慢"。
    # 地址同样从 board.h 取，不手抄 —— QEMU 侧灌入的位置必须与
    # main_baremetal.c 编译进去的常量完全一致，两边失同步的现象是
    # "权重 magic 不对"，看起来却像加载没生效。
    WA="$(bhex BOARD_WEIGHTS_ADDR)"
    TA="$(bhex BOARD_TOKENIZER_ADDR)"
    # NO_TOK=1 不灌分词器 —— 程序转入回归模式（固定 4 token 与黄金数据
    # 逐一自检），这是在 QEMU 上验证数值的唯一可行路径。
    # 带分词器时程序进的是交互模式，会停在串口等一行输入，而 QEMU 这边
    # 没有人打字，于是永远挂着 —— 看起来像"跑得很慢"，其实根本不会结束。
    if [ "${NO_TOK:-0}" = 1 ]; then
        echo "== qemu-system-riscv64 (权重@$WA, 无分词器 -> 回归模式) =="
        time "$QEMU_SYSTEM_RISCV64" -machine virt -nographic -bios none -m 2G \
            -cpu "$QEMU_CPU_FULL" \
            -kernel "$OUT/qwen3_bm.elf" \
            -device loader,file="$W",addr="$WA"
    else
        echo "== qemu-system-riscv64 (权重@$WA, tokenizer@$TA) =="
        time "$QEMU_SYSTEM_RISCV64" -machine virt -nographic -bios none -m 2G \
            -cpu "$QEMU_CPU_FULL" \
            -kernel "$OUT/qwen3_bm.elf" \
            -device loader,file="$W",addr="$WA" \
            -device loader,file="$T",addr="$TA"
    fi
    ;;
bringup)
    # L0：上板第一个程序。不依赖权重与分词器，几 KB，加载一秒。
    #   BOARD=s2c tools/build_riscv.sh bringup     产出真机镜像
    #   tools/build_riscv.sh bringup               在 QEMU 上先验一遍
    board_def
    # 产物名带板子标识：两块板的镜像绝不能同名互相覆盖。
    # 拿错镜像上板的现象是"串口一片死寂"，与地址配错完全一样，极难分辨。
    ELF="$OUT/bringup_${BOARD:-qemu}.elf"
    BIN="$OUT/bringup_${BOARD:-qemu}.bin"
    echo "== 交叉编译 bringup (board=${BOARD:-qemu}) =="
    rm -f "$ELF"
    # 复用 src/bsp/start.S —— 顺带验证 gp/栈/BSS/mstatus/trap handler，
    # 而它们正是后面跑全模型要用的同一份启动代码。
    # shellcheck disable=SC2086
    ${CROSS}gcc $RISCV_CFLAGS -Isrc $BOARD_DEF $WSIZE_DEF -ffreestanding -nostdlib -nostartfiles \
        -T src/bsp/qemu_virt.ld src/bsp/start.S tests/bringup.c \
        src/bsp/string_min.c -o "$ELF" 2>&1 \
        | grep -viE "LOAD segment with RWX" || true
    [ -f "$ELF" ] || { echo "链接失败" >&2; exit 1; }
    ${CROSS}objcopy -O binary --gap-fill 0 "$ELF" "$BIN"
    emit_debug "$ELF"
    printf "   %s  (%s B)\n" "$BIN" "$(stat -c%s "$BIN")"
    printf "   %s  (反汇编，按 mepc 定位用)\n" "${ELF%.elf}.diss"

    MAGIC_ADDR="$(bhex '(BOARD_MBOX_ADDR + 0x80)')"
    ECHO_ADDR="$(bhex '(BOARD_MBOX_ADDR + 0x84)')"

    if [ "${BOARD:-qemu}" = "s2c" ]; then
        echo
        echo "== 上板步骤 =="
        echo "  1) PCIe 后门写 $(basename "$BIN")  -> $(bhex BOARD_DRAM_BASE)"
        echo "  2) PCIe 后门写 0x5A5AC3C3         -> $MAGIC_ADDR   ★ 别漏，这是 PCIe 自证"
        echo "  3) 放开复位"
        echo "  4) 轮询 $(bhex BOARD_MBOX_STATUS)，期望依次"
        echo "       0xB0000010 取指+DDR写 -> 0xB0000020 串口 -> 0xB0000030 PCIe"
        echo "       -> 0xB0000040 RVV -> 0xB0000050 AME -> 0xB0000060 性能"
        echo "       -> 0xB00000AA 全部通过；0xB00000Fx 表示第 x 步失败"
        echo "  5) 串口 $(bhex BOARD_UART_BASE) (16550) 有完整日志；没输出就只看 mailbox"
        echo "  失败时读 $ECHO_ADDR 看回读到的 magic 实际是什么"
        echo
        echo "== mailbox 字段（基址 $(bhex BOARD_MBOX_ADDR)，全部 32 位小端）=="
        echo "    +0x00  STATUS      阶段推进 / 最终结论"
        echo "    +0x04  VLENB       实测 vlenb，应为 128"
        echo "    +0x08  AME_RESULT  AME 自测结果，应为 0x42000000 (=32.0f)"
        echo "    +0x0C  CYCLES_LO   性能采样的 mcycle 差值（低 32 位）"
        echo "    +0x10  CYCLES_HI   同上（高 32 位）"
        echo "    +0x14  ITERS       采样跑了多少次 mfmacc"
        echo "    +0x18  MTILE_CAP   mtilem 回读值，核对硬件配置"
        echo "    +0x1C  FAILMASK    失败位图，bit n = 第 n 步未过；0 表示全过"
        echo "    +0x80  HOST_MAGIC  ← host 上电前写 0x5A5AC3C3"
        echo "    +0x84  HOST_ECHO   程序回读到的值，不符时看这里"
        echo
        echo "  ★ 别只看 STATUS：软失败后程序会继续跑，STATUS 会被后续阶段覆盖。"
        echo "    判定通过与否一律看 FAILMASK 是否为 0。"
        echo
        echo "  性能换算：每次 mfmacc 周期数 = CYCLES / ITERS"
        echo "            decode 一个 token 约需 145500 次 => 时间 = 145500 x 周期 / 40MHz"
        exit 0
    fi
    echo "== QEMU 预演 =="
    # QEMU 上没有 host 来写 magic，用 -device loader 直接写一个 32 位值代劳。
    "$QEMU_SYSTEM_RISCV64" -machine virt -nographic -bios none -m 2G \
        -cpu "$QEMU_CPU_FULL" -kernel "$ELF" \
        -device loader,addr="$MAGIC_ADDR",data=0x5A5AC3C3,data-len=4
    ;;
smoke)
    echo "== AME 冒烟测试 =="
    # shellcheck disable=SC2086
    ${CROSS}gcc $CFLAGS_ALL tests/ame_smoke.c -o "$OUT/smoke.elf" 2>&1 \
        | grep -viE "LOAD segment with RWX" || true
    "$QEMU_RISCV64" -cpu "$QEMU_CPU_FULL" "$OUT/smoke.elf"
    ;;
*) echo "未知命令: $1" >&2; exit 1 ;;
esac
