/* 板级配置 —— 换板子只改这一个文件。
 *
 * 本文件同时被 C 和汇编（start.S）包含，所以：
 *   * 只能有 #define 和注释，不能有 typedef / 结构体 / 函数声明
 *   * 地址常量**不加 UL 后缀** —— 汇编器不认。C 侧用 BOARD_PTR() 转换
 *
 * 选板子：编译时 -DBOARD_QEMU_VIRT 或 -DBOARD_S2C，默认前者。
 */
#ifndef QWEN3_BOARD_H
#define QWEN3_BOARD_H

#if !defined(BOARD_QEMU_VIRT) && !defined(BOARD_S2C)
#define BOARD_QEMU_VIRT 1
#endif

/* ══════════════════════════ QEMU virt ══════════════════════════ */
#ifdef BOARD_QEMU_VIRT

#define BOARD_NAME            "QEMU virt"

#define BOARD_UART_BASE       0x10000000
#define BOARD_UART_16550      1           /* THR=idx0, LSR=idx5, THRE=bit5 */
/* QEMU virt 的 16550 是 8 位寄存器、间距 1 字节，且开箱即用无需配波特率。 */
#ifndef BOARD_UART_32BIT
#define BOARD_UART_32BIT      0
#endif
#ifndef BOARD_UART_REG_SHIFT
#define BOARD_UART_REG_SHIFT  0
#endif
#define BOARD_UART_NEEDS_INIT 0
#ifndef BOARD_UART_CLK_HZ
#define BOARD_UART_CLK_HZ     0
#endif
#ifndef BOARD_UART_BAUD
#define BOARD_UART_BAUD       115200
#endif
#ifndef BOARD_UART_HAS_DLF
#define BOARD_UART_HAS_DLF    0
#endif
#define BOARD_UART_DLF_IDX    48
#ifndef BOARD_UART_DLF_BITS
#define BOARD_UART_DLF_BITS   4
#endif

#define BOARD_DRAM_BASE       0x80000000
/* QEMU 以 -m 2G 启动（见 build_riscv.sh），故 DRAM 到 0xFFFFFFFF。 */
#define BOARD_DRAM_END        0xFFFFFFFF
/* 权重起点。必须在程序镜像（含 BSS 与栈）末端之后 ——
 * start.S 清 BSS 时会把重叠区域擦成 0，症状是"权重 magic 不对"，
 * 看起来却像外部加载没生效。main_baremetal.c 的 check_layout() 守这条。 */
#define BOARD_WEIGHTS_ADDR    0x88000000
#define BOARD_TOKENIZER_ADDR  0xD0000000

/* sifive_test：写 0x5555 让 QEMU 正常退出。真机上没有这个外设。 */
#define BOARD_HAS_POWEROFF    1
#define BOARD_POWEROFF_ADDR   0x100000
#define BOARD_POWEROFF_PASS   0x5555
#define BOARD_POWEROFF_FAIL   0x3333

/* QEMU 是模拟执行，mcycle 与真实时间无对应关系。
 * 置 0 表示"不要用它换算耗时"，计时代码会跳过。 */
#define BOARD_CPU_HZ          0

/* ══════════════════════════ S2C FPGA ══════════════════════════ */
#elif defined(BOARD_S2C)

#define BOARD_NAME            "S2C FPGA"

/* 串口基址与型号均由硬件团队确认：16550 兼容
 * （THR@0x00, LSR@0x05, THRE=bit5），与 QEMU virt 布局相同，
 * 所以 putc_() 和 start.S 的 trap handler 都不用改，只换基址。
 *
 * 硬件给的写法是 0x00_2010_0000 —— 十个十六进制数字，即 0x20100000。
 * 别按八位读成 0x00201000，那会差 256 倍，落到一片没有外设的地址上，
 * 写进去不报错也没输出，查起来毫无线索。 */
#define BOARD_UART_BASE       0x20100000
#define BOARD_UART_16550      1

/* ★ 寄存器访问宽度与地址间距 —— 这两个值必须与 SoC 的集成方式一致，
 *   任何一个不对，串口都会彻底不工作（而串口是最直观的观测手段）。
 *
 *   BOARD_UART_32BIT      1 = 用 32 位 load/store 访问寄存器
 *   BOARD_UART_REG_SHIFT  寄存器索引左移几位得到字节偏移
 *                         2 => 间距 4 字节，LSR(idx5) 在 +0x14   ← 最常见
 *                         0 => 间距 1 字节，LSR 在 +0x05
 *
 *   若上板后 mailbox 能推进到 ST_UART 之后、屏幕却一个字都没有，
 *   先把 REG_SHIFT 在 2 和 0 之间换一次再排查别的。 */
#ifndef BOARD_UART_32BIT
#define BOARD_UART_32BIT      1
#endif
/* 寄存器按 32 位排布、间距 4 字节（已向硬件确认）。
 * 于是 index n 的字节偏移 = n << 2，与 DW_apb_uart 的寄存器映射一致。 */
#ifndef BOARD_UART_REG_SHIFT
#define BOARD_UART_REG_SHIFT  2
#endif

/* UART 的输入时钟。**与 CPU 主频是两个不同的值**：CPU 40 MHz、UART 10 MHz。
 *     BOARD_CPU_HZ        -> 把 mcycle 换算成秒
 *     BOARD_UART_CLK_HZ   -> 只用来算波特率分频
 *   两者是各自独立的配置项，取值也确实不同。拿 CPU 主频去算分频会偏 4 倍，
 *   现象是串口满屏乱码而不是没有输出，很容易往别的方向查。
 *
 * 分频因子 = UART_CLK / (16 x 波特率)，见下面的 BOARD_UART_DIVISOR。
 * 10 MHz / (16 x 115200) = 5.43，除不尽，小数部分靠 DLF 补，
 * 详见下面 BOARD_UART_HAS_DLF 处的说明。 */
#ifndef BOARD_UART_CLK_HZ
#define BOARD_UART_CLK_HZ     10000000
#endif
#ifndef BOARD_UART_BAUD
#define BOARD_UART_BAUD       115200
#endif
/* 本板的 UART 需要程序自己配波特率，不能指望 bootrom 已配好。 */
#define BOARD_UART_NEEDS_INIT 1

/* DLF（Divisor Latch Fraction）—— 小数分频寄存器。
 * 本板 UART 用的是 **Synopsys DW_apb_uart**（已确认），所以这个寄存器存在；
 * 标准 16550 是没有它的。有了它才能在 10 MHz 这种除不尽的时钟下配准波特率。
 *
 * ★ 在 10 MHz 下 DLF 不是锦上添花，而是**唯一可行的办法**：
 *   理想分频 5.43，整数只能取 5 -> 125000 baud，偏 +8.5%，
 *   远超 16550 常见的 ±2~3% 容限，串口根本收不对。
 *   补上 DLF = 7/16 后实际 114943 baud，偏 -0.22%。
 *   注意此时**没有降级路径**：四舍五入同样得 5，偏差不变。
 *   （40 MHz 时尚可退到 22 勉强用，10 MHz 时这条路走不通。）
 *
 *   位置：寄存器索引 48，字节偏移 0xC0 = 48 << REG_SHIFT(2)，
 *         即 DW_apb_uart 手册里 DLF 的标准偏移。
 *
 * ── 两个"位宽"别搞混 ──
 *   访问宽度   32 位 / 4 字节，与其它 UART 寄存器一样 -> BOARD_UART_32BIT
 *   有效位数   DW_apb_uart 的综合参数 DLF_SIZE，决定小数的分母是 2^N
 *              -> BOARD_UART_DLF_BITS，本板为 4（已确认），分母 16
 *   前者说的是怎么访问，后者说的是里面实际实现了几位，两者没有关系。
 *
 * 万一位数填错也不会让串口哑掉，只影响精度：写进去的值超出实际位数时高位
 * 被硬件丢弃 —— 按 4 位算出 11 而实际只有 3 位的话，硬件收到 11 & 7 = 3，
 * 波特率 116959（+1.53%），仍在 16550 容限内。
 *
 * 访问时机：DLL/DLM 需要 LCR.DLAB=1，而 DLF 不需要，
 * 因此在恢复 DLAB=0 之后再写它。 */
#ifndef BOARD_UART_HAS_DLF
#define BOARD_UART_HAS_DLF    1
#endif
#define BOARD_UART_DLF_IDX    48
#ifndef BOARD_UART_DLF_BITS
#define BOARD_UART_DLF_BITS   4
#endif

/* DDR 可用范围（硬件团队确认）：0x8000_0000 ~ 0xEFFF_FFFF，共 1.75 GB。
 * 注意不是整块 8 GB —— 布局余量没有想象中宽裕，核算过：
 *   程序镜像   0x8000_0000  67 MB（含 56.6 MB scratch）
 *   mailbox    0x87F0_0000   1 MB
 *   权重       0x8800_0000   1.11 GB  末端 0xCF12_0000
 *   分词器     0xD000_0000   3.3 MB   末端 0xD034_0000
 *   剩余                      511 MB
 * 加长上下文或加大 scratch 时要重新核这张表。 */
#define BOARD_DRAM_BASE       0x80000000
#define BOARD_DRAM_END        0xEFFFFFFF

/* 权重与分词器的位置由 host 经 PCIe 后门写入，我们自己定。
 * 留 128 MB 给程序镜像 —— 必须大于镜像实际末端，否则 start.S 清 BSS
 * 时会把权重头部擦掉，症状是"权重 magic 不对"，看起来像加载没生效。 */
#define BOARD_WEIGHTS_ADDR    (BOARD_DRAM_BASE + 0x08000000)
#define BOARD_TOKENIZER_ADDR  (BOARD_DRAM_BASE + 0x50000000)

/* 真机没有 sifive_test，跑完停在死循环里等 host 读结果。 */
#define BOARD_HAS_POWEROFF    0

/* CPU 与 AME 同为 40 MHz（注意 UART 是另一个时钟，10 MHz）。
 * 用来把 mcycle 换算成真实秒数 ——
 * demo 里报"生成 N 个 token 用时 X 秒"就靠它。 */
#define BOARD_CPU_HZ          40000000

#endif

/* 波特率分频。BOARD_UART_CLK_HZ 为 0 表示该平台无需配置。
 *
 * DLL/DLM 存**整数**部分，小数部分交给 DLF（见下）：
 *   10 MHz / (16 x 115200) = 5.4253
 *   整数 5 单独使用时实际波特率 125000，误差 +8.51%，远超容限；
 *   补上 DLF = 7/16 之后实际 114943，误差 -0.22%。 */
#if BOARD_UART_CLK_HZ
/* 16 x 波特率：整数分频与小数分频共用的步长 */
#define BOARD_UART_STEP      (16 * BOARD_UART_BAUD)
#if BOARD_UART_HAS_DLF
/* 有 DLF：整数部分取**截断**，余数交给 DLF 补 */
#define BOARD_UART_DIVISOR   (BOARD_UART_CLK_HZ / BOARD_UART_STEP)
#else
/* 没有 DLF（标准 16550 就没有）：只剩整数分频，此时**四舍五入**才是最优解。
 * 注意：10 MHz 下这条降级路径救不了 —— 截断与四舍五入同样得 5，
 * 都偏 +8.51%。该分支只在时钟较高（如 40 MHz 得 21/22）时才有意义。
 * 所以这里不能沿用上面那条截断式 —— 把 BOARD_UART_HAS_DLF 关掉时，
 * 分频值会自动跟着切换，不需要再手工算一遍。 */
#define BOARD_UART_DIVISOR       ((BOARD_UART_CLK_HZ + BOARD_UART_STEP / 2) / BOARD_UART_STEP)
#endif
#define BOARD_UART_REM       (BOARD_UART_CLK_HZ % BOARD_UART_STEP)
/* 小数部分 x 2^DLF_BITS，四舍五入（分子加半个步长再整除）：
 *   小数 0.7014 x 16 = 11.22 -> DLF = 11
 *   实际波特率 = 40e6 / (16 x (21 + 11/16)) = 115274，误差 +0.06% */
#define BOARD_UART_DLF_VAL   ((BOARD_UART_REM * (1 << BOARD_UART_DLF_BITS) + BOARD_UART_STEP / 2) / BOARD_UART_STEP)
#else
/* 该平台（QEMU virt）无需配波特率，下面几个值不会被用到。
 * 仍然逐个给出定义，是因为 BOARD_UART_DLF_VAL 引用了 BOARD_UART_REM，
 * 少定义一个，别处展开时就会撞上未定义标识符。 */
#define BOARD_UART_STEP      1
#define BOARD_UART_DIVISOR   1
#define BOARD_UART_REM       0
#define BOARD_UART_DLF_VAL   0
#endif

/* ══════════════════════ 与 host 的结果交换区 ══════════════════════
 * PCIe 后门可以回读 DDR，所以 bring-up 不必先调通串口：
 * host 写完权重放开复位，轮询 STATUS，再把 RESULT 区读回去比对即可。
 * 串口只作为辅助观测手段。
 *
 * 放在权重之前的低地址区，与程序镜像、权重都不重叠。 */
#define BOARD_MBOX_ADDR       (BOARD_DRAM_BASE + 0x07F00000)
#define BOARD_MBOX_STATUS     (BOARD_MBOX_ADDR + 0x00)  /* 见下面的取值 */
#define BOARD_MBOX_LAYER      (BOARD_MBOX_ADDR + 0x04)  /* 当前层号，看进度 */
#define BOARD_MBOX_NTOKEN     (BOARD_MBOX_ADDR + 0x08)  /* 已生成 token 数 */
#define BOARD_MBOX_CYCLES_LO  (BOARD_MBOX_ADDR + 0x0C)  /* mcycle 低 32 位 */
#define BOARD_MBOX_CYCLES_HI  (BOARD_MBOX_ADDR + 0x10)
#define BOARD_MBOX_TOKENS     (BOARD_MBOX_ADDR + 0x20)  /* 生成的 token id 数组 */
#define BOARD_MBOX_TEXT       (BOARD_MBOX_ADDR + 0x400) /* 解码后的 UTF-8 文本 */
/* 两个区的容量。程序按这两个数截断，绝不越界往后写 ——
 * mailbox 之后就是权重区，写过界会把权重悄悄改掉，
 * 而那种错误只会表现为"模型胡言乱语"，几乎无从查起。 */
#define BOARD_MBOX_TOKENS_MAX 240                       /* (0x400-0x20)/4 = 248，留些余量 */
#define BOARD_MBOX_TEXT_BYTES 4096                      /* UTF-8 中文约 1300 字 */

/* STATUS 取值。用有辨识度的魔数，避免与未初始化内存混淆。 */
#define BOARD_ST_BOOT         0xB0000001      /* 启动，尚未初始化 */
#define BOARD_ST_READY        0xB0000002      /* 权重校验通过 */
#define BOARD_ST_RUNNING      0xB0000003      /* 正在前向 */
#define BOARD_ST_DONE         0xB0000004      /* 正常完成 */
#define BOARD_ST_TRAP         0xB00000FF      /* 发生异常，见串口 */

/* ══════════════════ L1D 缓存一致性 ══════════════════
 *
 * 本核的 L1D 只服务标量与向量访存；**AME 的 load/store 直接走 DDR，
 * 不经过 L1D**，硬件不维护两者之间的一致性，必须软件显式同步。
 *
 * 两个方向都危险：
 *   RVV 写 -> AME 读：数据可能还在 L1D 里是脏的，AME 从 DDR 读到旧值
 *   AME 写 -> RVV 读：AME 已写进 DDR，L1D 里却还留着旧拷贝
 *
 * FENCE 覆盖 AME 的访存缓冲（硬件团队确认），fence 退休即代表
 * AME 的访存已落到 DDR，所以用标准 `fence rw,rw` 即可定序。
 *
 * 用按地址的 Zicbom（cbo.clean/inval/flush）而不是全量 clean/inval：
 *   * 只碰 A 缓冲与 C 矩阵，**不动栈和全局** —— 全量 INV 会丢弃栈上
 *     未写回的脏数据，那种 bug 不当场崩，而是过一会儿在别处读到垃圾
 *   * 代价可控：每 token 约 10 万条 cbo，@40MHz 只有几 ms
 * march 需要带 _zicbom_zicbop_zicboz。
 */
#ifdef BOARD_S2C
#define BOARD_NEEDS_CACHE_SYNC 1
#else
/* QEMU 是模拟器，不存在这个一致性问题；且其 CPU 模型未必实现 cbo.*，
 * 贸然发出会变成非法指令，把已经跑通的 QEMU 路径打死。
 * 想在 QEMU 上验证这段代码本身，编译时加 -DFORCE_CACHE_SYNC。 */
#ifdef FORCE_CACHE_SYNC
#define BOARD_NEEDS_CACHE_SYNC 1
#else
#define BOARD_NEEDS_CACHE_SYNC 0
#endif
#endif

/* cache 参数（硬件团队确认）：L1D 32 KB、L1I 32 KB、行 64 B。
 * 行大小若填错，后果不对称：偏大只是多做无用功，
 * 偏小会跳过尾部的行，症状是偶发数值错误，极难查。 */
#ifndef BOARD_CACHE_LINE
#define BOARD_CACHE_LINE 64
#endif
#define BOARD_L1D_BYTES  (32 * 1024)
#define BOARD_L1I_BYTES  (32 * 1024)
#define BOARD_L1D_LINES  (BOARD_L1D_BYTES / BOARD_CACHE_LINE)   /* = 512 */

/* 真机默认用全量（xdcache），不用按地址（Zicbom）—— 这是算过账的：
 *
 *   L1D 只有 512 行，而要同步的数据远大于它。decode 时 lm_head 的
 *   输出是 151936×4 = 607 KB，按地址失效要发 9496 条 cbo.inval，
 *   其中 9000 多条打在根本不在 cache 里的地址上，纯属空转；
 *   而 l1d_inv_all 无论数据多大都只处理 512 行。
 *
 * 按地址方案保留给 QEMU：那里 cbo.* 能执行，而 l1d_*_all 是非法指令
 * （实测 0x0020702b），同步逻辑只有靠 cbo 才能在上板前验证。
 * 真机若想改用按地址，编译时加 -DBOARD_USE_CBO_RANGE。 */
#if defined(BOARD_S2C) && !defined(BOARD_USE_CBO_RANGE)
#define BOARD_USE_L1D_ALL 1
#endif

/* 镜像版本标识。构建脚本会用 -D 注入 0xMMDDhhmm；
 * 为 0 表示这份镜像不是走标准构建流程编出来的。
 * 加它的原因：反馈回来的串口截图无法判断对方跑的是哪一版，
 * 只能靠比对某行输出是否存在来倒推，每次迭代都要重来一遍。 */
#ifndef BOARD_BUILD_ID
#define BOARD_BUILD_ID 0u
#endif

/* C 侧把上面的裸数字转成指针用这个；汇编侧直接 li 即可。 */
#ifndef __ASSEMBLER__
#include <stdint.h>
#include <stddef.h>
#define BOARD_PTR(t, a)  ((volatile t *)(uintptr_t)(a))

/* 编译期把关：开了 cache 同步却没在 -march 里带 zicbom，
 * 汇编器只会吐一堆 "unrecognized opcode" 指向内联汇编，很难联想到是 march。
 * 这里直接报出人话。 */
#if BOARD_NEEDS_CACHE_SYNC && !defined(__riscv_zicbom)
#error "cache 同步已开启，但 -march 缺少 zicbom。请追加 _zicbom_zicbop_zicboz（见 tools/env.sh）"
#endif

/* ── 两套实现，语义等价，调用方（kernels_ame.c）无需区分 ──
 *
 * 默认按地址（Zicbom 的 cbo.*）：
 *   + QEMU 能执行，同步逻辑可以在上板前就验证
 *   + 只碰指定范围，不动栈与全局
 *   - 指令数随数据量增长
 *
 * -DBOARD_USE_L1D_ALL 切到全量（xdcache 的 l1d_clean_all / l1d_inv_all）：
 *   + 代价只受 L1D 容量限制，与数据量无关
 *   - QEMU 报非法指令（实测 0x0020702b），只有真机能跑
 *   - l1d_inv_all **直接丢弃脏行**，会波及栈与全局，
 *     所以下面的 INVAL 必须先 clean_all 再 inv_all，缺一不可
 * 全量需要 -march 追加 _xdcache。 */
#if BOARD_NEEDS_CACHE_SYNC && defined(BOARD_USE_L1D_ALL)

#ifndef __riscv_xdcache
#error "BOARD_USE_L1D_ALL 需要 -march 追加 _xdcache"
#endif
#define BOARD_DCACHE_CLEAN(p, n)                                             \
    do { (void)(p); (void)(n);                                               \
         __asm__ volatile("l1d_clean_all" ::: "memory"); } while (0)
/* 写回并失效。本核没有单条 flush，用 clean_all + inv_all 组合表达，
 * 与 Zicbom 的 cbo.flush 语义对齐：不丢数据，且拷贝确实被失效。
 *
 * ★ 不能只做 clean_all。那样脏行虽然写回了 DDR，**拷贝仍留在 cache 里**，
 *   随后的读依旧命中旧拷贝 —— 写方向没问题，读方向完全失效。
 *   需要"读到别人（host/AME）写进 DDR 的新值"时，必须用这个而不是 CLEAN。 */
#define BOARD_DCACHE_FLUSH(p, n)  BOARD_DCACHE_INVAL(p, n)
/* ★ 先 clean 再 inv：inv 丢弃脏行，不先写回就会丢掉栈上的数据。 */
#define BOARD_DCACHE_INVAL(p, n)                                             \
    do { (void)(p); (void)(n);                                               \
         __asm__ volatile("l1d_clean_all\n\tl1d_inv_all" ::: "memory");      \
    } while (0)
#define BOARD_FENCE()             __asm__ volatile("fence rw,rw" ::: "memory")

/* ★ 关于 fence.i：本核的实现是**先对 D-cache 做 clean all，再失效整个
 * I-cache**。它不是一条纯粹的取指同步指令 —— 它会把脏行写回 DDR。
 *
 * 后果与上面 INVAL 宏要防的是同一类问题：AME 写过的输出缓冲，若之后执行
 * fence.i，CPU 侧残留的旧脏行会被写回，把 AME 的结果盖掉。
 *
 * 目前全项目只有 start.S 在复位处用了一次 fence.i，且已在它之前先做
 * l1d_inv_all 把 cache 清空。若将来要动态加载代码（自修改、加载 overlay），
 * 用 fence.i 之前务必确认此刻 D-cache 里没有不能被写回的内容。 */

#elif BOARD_NEEDS_CACHE_SYNC
#define BOARD_CBO_RANGE(op, p, bytes)                                        \
    do {                                                                     \
        uintptr_t _a = (uintptr_t)(p) & ~(uintptr_t)(BOARD_CACHE_LINE - 1);  \
        uintptr_t _e = (uintptr_t)(p) + (bytes);                             \
        for (; _a < _e; _a += BOARD_CACHE_LINE)                              \
            __asm__ volatile(op " (%0)" :: "r"(_a) : "memory");              \
    } while (0)
/* 写回脏行，保留拷贝。用于"CPU 写完、AME 要读"。 */
#define BOARD_DCACHE_CLEAN(p, n)  BOARD_CBO_RANGE("cbo.clean", p, n)
/* 失效拷贝，**脏行直接丢弃**。只可用于确知无脏行的区域。 */
#define BOARD_DCACHE_INVAL(p, n)  BOARD_CBO_RANGE("cbo.inval", p, n)
/* 写回并失效。不丢数据，首尾不对齐时也安全。 */
#define BOARD_DCACHE_FLUSH(p, n)  BOARD_CBO_RANGE("cbo.flush", p, n)
#define BOARD_FENCE()             __asm__ volatile("fence rw,rw" ::: "memory")
#else
#define BOARD_DCACHE_CLEAN(p, n)  ((void)(p), (void)(n))
#define BOARD_DCACHE_INVAL(p, n)  ((void)(p), (void)(n))
#define BOARD_DCACHE_FLUSH(p, n)  ((void)(p), (void)(n))
/* 即使不做 cache 操作也保留编译器屏障：防止编译器把访存跨过 AME 指令重排。 */
#define BOARD_FENCE()             __asm__ volatile("" ::: "memory")
#endif

#endif /* !__ASSEMBLER__ */

#endif /* QWEN3_BOARD_H */
