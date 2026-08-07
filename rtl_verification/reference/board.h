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
#define BOARD_UART_16550      1           /* THR@0x00, LSR@0x05, THRE=bit5 */

#define BOARD_DRAM_BASE       0x80000000
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

/* CPU 与 AME 同为 50 MHz。用来把 mcycle 换算成真实秒数 ——
 * demo 里报"生成 N 个 token 用时 X 秒"就靠它。 */
#define BOARD_CPU_HZ          50000000

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
 *   * 代价可控：每 token 约 10 万条 cbo，@50MHz 只有几 ms
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
/* 全量 clean 已把所有脏行写回，与 FLUSH 语义在此等价。 */
#define BOARD_DCACHE_FLUSH(p, n)  BOARD_DCACHE_CLEAN(p, n)
/* ★ 先 clean 再 inv：inv 丢弃脏行，不先写回就会丢掉栈上的数据。 */
#define BOARD_DCACHE_INVAL(p, n)                                             \
    do { (void)(p); (void)(n);                                               \
         __asm__ volatile("l1d_clean_all\n\tl1d_inv_all" ::: "memory");      \
    } while (0)
#define BOARD_FENCE()             __asm__ volatile("fence rw,rw" ::: "memory")

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
