# 每次在 WSL 中执行本项目脚本前先 source 本文件。
# 非交互式 shell (bash -c) 不读 ~/.bashrc，所以这些设置必须显式加载。
#   source tools/env.sh

# ---------------- Python / HuggingFace ----------------
export HF_ENDPOINT=https://hf-mirror.com    # 大陆镜像；直连 huggingface.co 不通
export HF_HOME="$HOME/.cache/huggingface"   # 缓存放 WSL 原生盘，比 /mnt/c 快
export TOKENIZERS_PARALLELISM=false         # 静音 tokenizers 的 fork 告警
# 新版 huggingface_hub 默认走 Xet 存储 (cas-server.xethub.hf.co)，hf-mirror 不代理该域名，
# 大文件会 401。禁用 Xet 回退到传统 HTTP 下载。
export HF_HUB_DISABLE_XET=1
source "$HOME/qwen3-venv/bin/activate"

# ---------------- RISC-V 交叉工具链 ----------------
# ESWIN GCC 14.1.1。注意命令前缀是 riscv-unknown-elf-（不带 64）。
export RISCV_TOOLCHAIN="$HOME/toolchain/riscv-elf-toolchain"
export PATH="$RISCV_TOOLCHAIN/bin:$PATH"
export CROSS=riscv-unknown-elf-

# ---- 目标 ISA（统一基准）----
# QEMU 与真实硬件使用同一 ISA，不做区分。
# 两套 ISA 会带来「QEMU 上通过、硬件上不通过」这类最难排查的问题，得不偿失。
#
#   g = imafd    基础 + 乘除 + 原子 + 单双精度浮点
#   c            压缩指令
#   v            RVV 1.0
#   zfh/zvfh     半精度（标量/向量）
#   zfbfmin      标量 BF16 <-> FP32 转换
#   zvfbfmin     向量 BF16 <-> FP32 转换（vfwcvtbf16 / vfncvtbf16）
#   zvfbfwma     向量 BF16 widening 乘加：FP32 += BF16*BF16（vfwmaccbf16）
#   xewmatrix    AME 矩阵扩展（链接阶段必须带版本号 1p0，
#                否则报 "x ISA extension must be set with the versions"）
#
# 刻意不启用的扩展及原因（实测）：
#   zcb   编译器会生成 c.sb 等压缩指令，QEMU 默认 CPU 不支持 -> illegal instruction
#   xext  编译器把 expf 编译成 custom-1 空间的自定义指令，QEMU 未实现 -> illegal
# AME 的访存绕过 L1D，与 RVV/标量之间的一致性只能靠软件维护（见 src/bsp/board.h）。
# 两套原语各有用途，两个扩展都要带上：
#   zicbom/zicbop/zicboz —— 标准的 cbo.clean/inval/flush/zero（按地址，一次一行）
#                           QEMU 能执行，是上板前验证同步逻辑的唯一手段
#   xdcache              —— l1d_clean_all / l1d_inv_all（全量）
#                           QEMU 报非法指令，只有真机能跑；L1D 仅 32 KB，
#                           数据量远大于它时全量明显更划算，故真机默认用它
# 少了对应扩展，开启同步的构建会在 board.h 里直接 #error 报出人话。
export RISCV_ISA="rv64gcv_zfh_zfbfmin_zvfh_zvfbfmin_zvfbfwma_xewmatrix1p0_zicbom_zicbop_zicboz_xdcache"

# 交叉编译公共选项。
#   -mcmodel=medany  链接到 0x80000000 时必需，否则 relocation truncated to fit
#   -std=gnu11       内联汇编、register asm、AME intrinsic 全是 GNU 扩展
export RISCV_CFLAGS="-march=$RISCV_ISA -mabi=lp64d -mcmodel=medany -O2 -std=gnu11"

# ---------------- QEMU (AME 定制版) ----------------
# 只有 user-mode 支持 AME；qemu-system-riscv64 不支持矩阵扩展。
# 2026-08-06 起用 qemu-latest：该版本修复了 mzero 只清累加器前 64 行的缺陷，
# 项目已据此把 acc_clear() 从 mlce32 装零换回原生 mzero。
export QEMU_RISCV64="$HOME/qemu-latest/bin/qemu-riscv64"
export QEMU_SYSTEM_RISCV64="$HOME/qemu-latest/bin/qemu-system-riscv64"
# 该版本额外依赖 libdw/libnuma，用 apt download + dpkg-deb -x 解到用户目录（无需 sudo）：
#   apt download libdw1t64 libnuma1 && dpkg-deb -x *.deb ~/qemu-new/libs
export LD_LIBRARY_PATH="$HOME/qemu-new/libs/usr/lib/x86_64-linux-gnu:$HOME/qemu/slirp/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
# CPU 配置必须与 RISCV_ISA 对应，且 BF16 那几个扩展 QEMU 不会默认打开 ——
# 实测不加 zvfbfmin/zvfbfwma 时，vfwcvtbf16 直接非法指令。
export QEMU_CPU_AME="rv64,matrix=true"
export QEMU_CPU_FULL="rv64,matrix=true,v=true,vlen=1024,zfh=true,zfbfmin=true,zvfh=true,zvfbfmin=true,zvfbfwma=true"

# 便捷函数：qrun <elf> [args...]   在 QEMU user-mode 下跑，AME+RVV 全开
qrun() { "$QEMU_RISCV64" -cpu "$QEMU_CPU_FULL" "$@"; }

# 交叉编译产物放 WSL 原生盘：/tmp 在本机是 tmpfs，WSL 实例回收后文件会消失。
export RISCV_BUILD="$HOME/qwen3-build"
mkdir -p "$RISCV_BUILD"
