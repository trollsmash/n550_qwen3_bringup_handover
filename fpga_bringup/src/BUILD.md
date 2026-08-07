# 源码与重新编译

208K，包含 L0/L2/L3 全部源码。改完能自己编出镜像，
不必回头找软件侧。

## 目录

```
src/
├── bsp/board.h          ★ 板级地址全在这里，改地址只改这一个文件
├── bsp/start.S          启动代码 + trap handler（异常时打印 mcause/mepc/mtval）
├── qwen3.c              模型前向，28 层循环
├── kernels_ame.c        ★ AME GEMM + 与 L1D 的缓存同步
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
| 串口没输出，要配波特率 | 编译时加 `-DUART_NEEDS_INIT -DUART_DIVISOR=<值>`，见 `tests/bringup.c` |
| demo 生成几个 token | `src/main_baremetal.c` 的 `CHAT_MAX_GEN`（默认 1） |
| demo 问什么问题 | `src/main_baremetal.c` 的 `DEMO_PROMPT` |

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
