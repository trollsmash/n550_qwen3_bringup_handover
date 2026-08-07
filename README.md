# n550 · Qwen3-0.6B 移植交付

把 Qwen3-0.6B 跑在自研 RISC-V AI CPU（RVV 1.0 + AME 矩阵扩展）上，
目标是 FPGA demo。**成功标准是能跑起来证明兼容性，不是性能。**

本仓库是交给两个下游团队的东西，各看各的目录。

## 给 RTL 验证 → [`rtl_verification/`](rtl_verification/)

AME 矩阵扩展的指令级测试向量，8 个 case。每个是一段能自己跑起来、
算完往固定地址写完成标志的最小裸机程序，testbench 只需"装镜像 → 放复位 →
等标志 → dump 内存"。

期望值**不是从模拟器抄的**，而是按架构语义推导（块内精确求和、块间 FP32
舍入、RNE），再与 QEMU 交叉验证：8 个 case 共 84608 个输出元素逐位相同。

从 [`rtl_verification/README.md`](rtl_verification/README.md) 开始读。

## 给 FPGA 上板 → [`fpga_bringup/`](fpga_bringup/)

分四级递进，每级失败都直接指向一个子系统：

| 级别 | 镜像 | 验什么 |
|---|---|---|
| L0 | 3.6 KB | PCIe 写入、复位、取指、串口、RVV、AME、时钟 |
| L1 | 786 KB | AME 矩阵单元数值正确性 |
| L2 | 1.11 GB | 全模型前向，对黄金数据自检 |
| L3 | +3.3 MB | 中文进中文出的完整 demo |

**先跑 L0**，3.6 KB 加载一秒。跳级直接跑全模型，出了问题分不清是 PCIe、
时钟、串口、矩阵单元还是模型本身。

从 [`fpga_bringup/README.md`](fpga_bringup/README.md) 开始读。

## 权重不在仓库里

`w.bin`（1.11 GB）超过 GitHub 单文件 100 MB 限制，且它是可完全复现的产物。
获取方式与 md5 见
[`fpga_bringup/L2_L3_fullmodel/GET_WEIGHTS.md`](fpga_bringup/L2_L3_fullmodel/GET_WEIGHTS.md)。

## 几条踩出来的经验，可能对你们也有用

* **AME 的访存绕过 L1D**，与 RVV/标量之间的一致性只能靠软件维护。
  同步点全在 GEMM 边界上，序列写反（GEMM 后 flush 输出）会把结果覆盖掉。
* **`mstatus.MS` 不使能时，所有 AME 指令报 illegal instruction**，
  现象与"指令编码不支持"一模一样，极易误判成译码问题。
* **tile 装载会读满整个 tile 的地址范围**，即使 `mtilem=1`。
  缓冲按有效数据大小紧贴分配会读到非法地址。
* **裸机上没有 trap handler 就是盲飞**：异常跳到 `mtvec`(=0) 空转，
  CPU 占满，与"正在慢慢算"完全无法区分。我们为此浪费过三个小时。
* **换镜像先断电**：L1I 与 D-cache 不自动一致，软复位可能跑的还是上一个镜像。
