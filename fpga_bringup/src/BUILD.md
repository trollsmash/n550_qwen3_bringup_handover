# 源码与重新编译

252K，包含 L0/L2/L3 全部源码。改完能自己编出镜像，
不必回头找软件侧。

## 目录

```
src/
├── bsp/board.h          ★ 板级地址全在这里，改地址只改这一个文件
├── bsp/start.S          启动代码 + trap handler（异常时打印 mcause/mepc/mtval）
├── qwen3.c              模型前向，28 层循环
├── kernels_ame.c        ★ AME GEMM + 与 L1D 的缓存同步
├── kernels_hwsim.c      在 PC 上复刻 RTL 累加语义，用于不上板预测数值行为
├── ops_rvv.c            RVV 实现的 rmsnorm/rope/softmax/silu 等
├── tokenizer.c          BPE 分词
├── main_baremetal.c     ★ L2/L3 的入口，prompt 和生成长度在这里
└── ...
tests/bringup.c          ★ L0 的全部代码，六步自检
tools/build_riscv.sh     构建脚本
tools/env.sh             工具链与 ISA 串
```

## 编译

```bash
source tools/env.sh                                   # 先改里面的工具链路径
BOARD=s2c tools/build_riscv.sh bringup                # L0  -> bringup_s2c.bin
KERNEL=ame OPS=rvv BOARD=s2c tools/build_riscv.sh baremetal   # L2/L3 -> qwen3_s2c.bin
```

## 可选：tile-major 权重布局（访存优化，需 RTL 配合）

AME 装载一个 tile 要读 128 行、每行 32 个 BF16 = 64 字节。权重按 PyTorch 原始的
 行优先存放时，**行间跨度是 K x 2 = 2048 字节**（down_proj 是 6144）——
读 64 B 就要跳过 2 KB，AXI 侧只能发 128 个单拍事务，无从合并。

把权重按 tile 重排后，行间跨度变成 **64 字节**，整个 tile 是连续的 8 KB：

```bash
python tools/02_export_weights.py --layout tile   # -> golden/qwen3-0.6b-bf16-tile.bin
KERNEL=ame OPS=rvv BOARD=s2c WEIGHTS=golden/qwen3-0.6b-bf16-tile.bin \n    tools/build_riscv.sh baremetal
```

**⚠ 只有当 RTL 支持「stride 等于行宽时合并成 burst」，这个布局才有收益**；
否则访存模式虽然连续，硬件仍按单拍发，性能与行优先一样。两边都到位才有效果。

两种布局的权重**不能混用**：布局标记在文件 header 偏移 76， 会校验，
不匹配时立刻报错而不是算出垃圾。tile 版文件大约 326 KB（张量对齐从 128 B 提到
4096 B，好让每个 8 KB 的 tile 落在 4 KB 边界上 —— AXI burst 不得跨 4 KB 边界，
对齐后正好切成两段而非三段）。

数值上两种布局完全等价，已在 QEMU 上用带黄金数据的回归模式验证过。

```

每次构建除了 `.bin` 还会产出三个文件，烧录只用 `.bin`，另外三个留着排障：

| 文件 | 用途 |
|---|---|
| `.elf` | 带完整调试信息（`-g3 -gdwarf-4`），GDB / addr2line 用 |
| `.diss` | 反汇编与 C 源码交织。串口打出 `*** TRAP ***` 时，拿 `mepc` 在这里搜地址，直接看到是哪条指令、哪一行 C |
| `.sym` | 按地址排序的符号表，先看 `mepc` 落在哪个函数 |

调试信息只存在于 `.elf`，`objcopy` 出来的 `.bin` 不受影响，**烧录镜像大小与不带调试信息时完全一致**。

`tools/env.sh` 里两处按你们的环境改：`CROSS`（工具链前缀）和
`RISCV_BUILD`（产物目录）。ISA 串**不要动**：

```
rv64gcv_zfh_zfbfmin_zvfh_zvfbfmin_zvfbfwma_xewmatrix1p0_zicbom_zicbop_zicboz_xdcache
```

少任何一段都会出事，而且症状具有迷惑性：
少 `zvfbfmin` → `vfwcvtbf16` 非法指令；少 `zicbom`/`xdcache` → 缓存同步编不过；
少 `xewmatrix1p0` → 所有 AME 指令不认。

## 你最可能要改的四处

| 想改什么 | 改哪里 |
|---|---|
| DDR / UART / 权重地址 | `src/bsp/board.h` 的 `BOARD_*`（S2C 段） |
| 串口没输出 | 寄存器排布已确认（DW_apb_uart，32 位、间距 4 字节），程序即按此编译；先查线序/电平与 UART 时钟 |
| 串口乱码 | 波特率按 UART 输入时钟 **40 MHz** 配（DLL=21 + DLF=11，误差 +0.06%）。若实际时钟不同，改 `board.h` 的 `BOARD_UART_CLK_HZ` 重编。这里 DLF 不能省：只用整数分频偏 +3.34%，已超 16550 容限 |
| demo 生成几个 token | `src/main_baremetal.c` 的 `CHAT_MAX_GEN`（默认 1） |
| demo 问什么问题 | `src/main_baremetal.c` 的 `DEMO_PROMPT` |

## 数值语义：真机与 QEMU 并不相同

AMU 的 PE 一次吃 **16** 路乘积（`ma_pkg.sv` 的 `ARRAY_K_FP`），每个 `k_iter`
结束就把累加器写回 AR（FP32）。也就是说 `mtilek=32` 的一条 `mfmacc`
内部会舍入**两次**。而 QEMU 的粒度是 32，只舍入一次 —— 两者对同一条指令
给出的结果不逐位相同。

这对 demo 没有影响：用 `src/kernels_hwsim.c`（复刻 RTL 语义）在 PC 上跑
完整模型，生成的 token 与黄金数据完全一致。但如果你要做**逐位**的数值比对，
记住以 RTL 为准，别拿 QEMU 的输出当标准答案。

```bash
gcc -O2 -Isrc -o q_hwsim src/main.c src/qwen3.c src/tokenizer.c     src/kernels_hwsim.c src/ops_scalar.c -lm
./q_hwsim golden/qwen3-0.6b-bf16.bin 4        # 应输出 3837 101889 106525 56568
```

**改 `board.h` 之后地址表会自动跟着变** —— 构建脚本用 `cpp` 从
`board.h` 展开取值，不存在"改了代码忘了改文档"。

## 改地址时必须注意

程序镜像（含 BSS 与栈）不能越过 `BOARD_WEIGHTS_ADDR`。
`start.S` 清 BSS 时会把重叠部分擦成 0，症状是"权重 magic 不对"，
**看起来完全像是 PCIe 加载没生效**。
`main_baremetal.c` 的 `check_layout()` 就是为此设的护栏，启动时会把
镜像末端和权重起点都打出来，别删。

## 在 QEMU 上先验

没有板子也能跑（用 QEMU virt 的地址，去掉 `BOARD=s2c`）：

```bash
tools/build_riscv.sh bringup                          # L0，几秒
KERNEL=ame OPS=rvv tools/build_riscv.sh baremetal      # 全模型，约 3 分钟
```

注意 QEMU 与真机的两处差异：`l1d_clean_all`/`l1d_inv_all` 在 QEMU 上是
非法指令（所以 QEMU 版走 `cbo.*`），`mcycle` 也不是真实周期计数。
