# 上板运行说明（FPGA）

本目录的 `README.md` 是写给 **RTL 验证**的，讲 testbench 与 `$readmemh`。
在 FPGA 上跑这些 case 用不着那一套，流程与 L0 bringup 完全相同：

1. PCIe 后门写 `cases/<case>/prog_fpga.bin` -> `0x8000_0000`
   **用 `prog_fpga.bin`**（带串口输出的 FPGA 版，见下节），
   **也不要用 .hex** —— 那是给 Verilog `$readmemh` 的文本格式。
2. 放开复位
3. 轮询 `0x800C_0000`，等它变成 `0x600D600D`
4. 读 `0x800C_0004`：`0x50415353`(PASS) / `0x4641494C`(FAIL)
5. 失败时 `0x800C_0008` 是首个不符元素的下标，`0x800A_0000` 起是结果矩阵 C

`prog_fpga.bin` 是自包含的完整内存镜像，不需要额外加载权重或分词器。

## 两份镜像的区别

| 文件 | 给谁 | 区别 |
|---|---|---|
| `prog.bin` | RTL 验证 | 结果只写内存；**完全不依赖 cache 行为** |
| **`prog_fpga.bin`** | **FPGA 上板** | 同样的计算，额外有串口输出与 cache 维护 |

两版的差别不只是串口。**上板必须用 `prog_fpga.bin`**，因为它多了三处
cache 操作，缺一处就会出现「看着像硬件坏了、其实是 cache 没同步」的现象：

| 位置 | 做什么 | 不做会怎样 |
|---|---|---|
| 复位后 | `l1d_inv_all` + `fence.i` | 上电时 cache 里是随机值，CPU 命中垃圾 —— **后门明明写进去了，CPU 却看不到镜像** |
| AME 写完 C | `l1d_clean_all` + `l1d_inv_all` | `msce32` 绕过 L1D 直写 DDR，CPU 却拿 cache 里预置的 `0xA5A5A5A5` 去比对 |
| 写完标志 | `l1d_clean_all` | 标志停在 L1D，**host 后门永远读不到 PASS/FAIL** |

复位那两条的顺序不能反：本核的 `fence.i` 实现是先对 D-cache 做 clean all
再失效 I-cache，若 cache 里有上电残留的随机脏行，那一步会把它们写回 DDR，
**反过来破坏刚加载的镜像**。所以必须先 `l1d_inv_all` 丢弃。

RTL 版刻意不含这些：前 8 个纯 AME case 的价值就在于 cache 子系统还没做好时，
矩阵单元本身的正确性照样能验 —— 加了 cache 指令会把这条价值抹掉。

两者装载地址与流程完全相同，只差一个串口输出。跑完串口直接给出：

```
=== case 001 bf16_full_tile  M=128 N=128 K=32 ===
RESULT: PASS
```

失败时连出错元素一起打出来：

```
RESULT: FAIL  idx=00000003  got=4351eec8  exp=4351eec7
```

`got`/`exp` 是 FP32 的位模式。**这两个值是读标志地址看不到的**，
省掉一轮"再去 dump C 区找是哪个元素"的往返。

串口参数与 L0 一致：`0x20100000`，32 位寄存器、间距 4 字节。
换板重编时加 `-DUART_BASE=...`，见 `tools/41_build_vectors.sh`。

### ⚠ FAIL 不等于硬件有错

程序自检做的是**逐位比对**，而矩阵扩展规范把累加的数值行为定为
implementation-defined —— **逐位不一致是允许的**，不代表实现有问题。
（在 QEMU 上跑这些 case 必然 FAIL：它的累加舍入粒度与本核不同，
差异约 1 ULP。这正是该判据不能直接用的例证。）

判定要用 `check.py` 的**归一化误差判据**（以 `c_peak.bin` 归一化），
而不是看串口这行 PASS/FAIL。串口的价值在**快速定位**：
`idx` 指出哪个元素，`got`/`exp` 指出差多少 ——
差 1 ULP 和差一个数量级，含义完全不同。

## 目录里那些 .bin 要不要一起装？——不用

`a.bin` / `b.bin` / `c_ref.bin` 的内容**已经逐字节嵌在 `prog.bin` 里**了，
在各自的地址上（见 `meta.json` 的 `segments`）。**只装 `prog.bin` 这一个文件。**

它们是同一份数据的独立副本，留着是为了**区分「装载错了」和「算错了」**：

| 想确认什么 | 怎么做 |
|---|---|
| 输入数据是否真的写进去了 | 后门读 `segments.a` 起 8192 字节，与 `a.bin` 比对；`b` 同理 |
| 结果是否正确 | 后门读 `segments.c`，与 `c_ref.bin` 比对（程序自检也在做这件事） |
| 误差是否在容限内 | `c_peak.bin` 是各元素累加过程的峰值，判据要用它归一化，见 `check.py` |

case 失败时先做第一行：相同说明输入到位、问题在计算；不同说明是后门写入本身
出了问题，那跟 AME 没关系。这一步能省掉一整轮猜测。

`.hex` 是各 `.bin` 的文本版（每行一个 32 位小端字），只给 Verilog `$readmemh` 用，
上板一律用 `.bin`。

什么时候需要跑它：L0 的 step5 只验了 1x1x32 全 1 这一种最简情形。
若 L0 过了但全模型数值不对，用这 12 个 case 定位是哪一类情况出错
（partial M/N/K、多 k 块累加、RVV 与 AME 混合等）。
**先跑完前 8 个纯 AME 的**，再碰混合的 —— 后者涉及 cache 与 AME 的数据交换。

其余内容（case 说明、数值判据、check.py 用法）两边通用，见 `README.md`。
