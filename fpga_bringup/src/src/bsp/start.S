/* 最小启动代码 —— QEMU virt / 裸机通用。
 *
 * 职责：设栈、清 BSS、开扩展、装 trap handler、跳 C 入口。
 * 若你们的 BSP 已有 crt0，本文件可整个替换掉，只需保证跳转到 qwen3_main()。
 *
 * 板级地址来自 board.h。本文件后缀是大写 .S，会经过 C 预处理器，
 * 所以能直接 include —— UART 基址不要在这里再写一份。
 */
#include "board.h"
    .section .text.init, "ax"
    .global _start
_start:
    /* ══════════ 复位后第一件事：丢掉 cache 里的一切 ══════════
     *
     * 镜像是 host 经 PCIe 后门直接写进 DDR 的，绕开了 CPU 的 cache。
     * 软复位不一定清 L1，于是上一次运行的内容可能还留在里面。
     * bring-up 时反复换镜像正是这个场景：
     *   写镜像 A -> 跑 -> 写镜像 B 到同一地址 -> 软复位
     * 旧行若还在，CPU 就接着跑旧程序，现象是"明明重新烧了，行为却没变"。
     *
     * ★★ 这两条的顺序不能反 ★★
     *
     * 本核的 fence.i 实现是：**先对 D-cache 做 clean all，再失效整个
     * I-cache**。也就是说 fence.i 自己会把 D-cache 里的脏行写回 DDR。
     * 若先 fence.i，上一次运行残留的脏行就会被写回，**盖掉后门刚写进去
     * 的新镜像** —— 这是破坏性的，且现象与"程序有 bug"无法区分。
     *
     * 所以必须先 l1d_inv_all：复位后 DDR 里才是权威内容，cache 里的一切
     * 都是垃圾，直接丢弃（inv_all 不写回，正是这里要的语义）。等 D-cache
     * 空了，再让 fence.i 去做它该做的 I-cache 失效。
     *
     * 局限：fence.i 管不了它**自己之前**的取指。若 I-cache 里 _start 开头
     * 这几条就是旧指令，CPU 根本执行不到这里。那种情况只能靠硬件复位清
     * I-cache，软件无能为力 —— 值得向 FPGA 团队确认复位是否清 L1I。
     *
     * QEMU 没有 xdcache，l1d_inv_all 在那边是非法指令，故条件编译。
     * QEMU 的镜像由 generic loader 装载，也不存在陈旧 cache 的问题。 */
#ifdef BOARD_USE_L1D_ALL
    l1d_inv_all
    /* 与下面的 fence.i 隔开：inv_all 和 fence.i 内部那次 clean all 之间
     * 没有架构层面的顺序保证，万一 clean 跑到 inv 前面就前功尽弃。 */
    fence   rw, rw
#endif
    fence.i

    /* 只让 hart 0 继续，其余核原地睡眠。
     * 放在 cache 清理之后：每个核清的都是自己的私有 L1，各清各的无副作用。 */
    csrr    t0, mhartid
    bnez    t0, park

    /* 初始化 gp（global pointer）。
     * 编译器会把小的全局变量改成 gp 相对寻址访问，gp 没设就是往垃圾地址读写，
     * 现象是 store/load access fault，且 epc 指向的指令看起来毫无问题 ——
     * 极难联想到是 gp 的锅。
     * .option norelax 是必需的：否则这条 la 自己会被 relax 成 gp 相对寻址（鸡生蛋）。 */
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    /* 设栈指针（栈顶由 linker script 给出） */
    la      sp, _stack_top

    /* 清 BSS：静态 arena 依赖 BSS 被清零 */
    la      t0, _bss_start
    la      t1, _bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:

    /* 使能三个扩展上下文状态域，缺一个对应的指令就会非法。
     * 三者都是两位域，写 01(Initial) 即可：
     *   mstatus.FS [14:13] 浮点   —— 不开则任何浮点指令陷入
     *   mstatus.VS [10:9]  向量   —— 不开则 RVV 指令陷入
     *   mstatus.MS [30:29] 矩阵   —— 不开则所有 AME 指令报 illegal instruction
     * MS 这一位极易遗漏：QEMU 里由 require_matrix() 检查，
     * 漏了的现象与「指令编码不被支持」完全一样，非常容易误判。 */
    li      t0, (1 << 13)         /* FS = Initial */
    csrs    mstatus, t0
    li      t0, (1 << 9)          /* VS = Initial */
    csrs    mstatus, t0
    li      t0, (1 << 29)         /* MS = Initial */
    csrs    mstatus, t0

    /* 安装 trap handler。
     * 不装的话 mtvec 是 0，任何异常都跳到地址 0 去执行垃圾 ——
     * 现象是程序"卡住"且 CPU 占满，与"正在慢慢算"完全无法区分。
     * 真机上没有别的观测手段，只会更难查。 */
    la      t0, trap_entry
    csrw    mtvec, t0             /* Direct 模式：低两位为 0 */

    call    qwen3_main

park:
    wfi
    j       park

/* ===================== Trap Handler =====================
 * 目标只有一个：把异常原因打出来，然后停下。
 *
 * 全汇编、直写 UART、不碰栈、不调用任何 C 函数 —— 因为触发 trap 时
 * 不能假设 C 运行环境还是完好的（栈溢出本身就是常见的 trap 原因）。
 * 寄存器随便用：到这里就不打算回去了。
 *
 * FPGA 上换板子只需改下面的 UART 基址。
 */
    .align 4
    .global trap_entry
trap_entry:
    li      t6, BOARD_UART_BASE   /* 来自 board.h，勿在此硬编码 */

    /* 先给 host 留个记号：PCIe 能回读 DDR，即使串口没接好也能看出是 trap。
     * 用 t5 而不是 t0，t0 后面要装 mcause。 */
    li      t5, BOARD_MBOX_STATUS
    li      t0, BOARD_ST_TRAP
    sw      t0, 0(t5)

    la      t4, msg_trap
    jal     t3, put_str
    csrr    t0, mcause
    jal     t3, put_hex

    la      t4, msg_epc
    jal     t3, put_str
    csrr    t0, mepc
    jal     t3, put_hex

    la      t4, msg_tval
    jal     t3, put_str
    csrr    t0, mtval
    jal     t3, put_hex

    la      t4, msg_hint
    jal     t3, put_str

1:  wfi
    j       1b

/* 打印以 NUL 结尾的字符串，地址在 t4，返回地址在 t3 */
put_str:
1:  lbu     t1, 0(t4)          /* 读字符串，来自内存，始终按字节 */
    beqz    t1, 2f
    /* 写 UART THR。THR 的寄存器索引为 0，字节偏移恒为 0，
     * 不受 REG_SHIFT 影响；这里只需匹配访问宽度。 */
#if BOARD_UART_32BIT
    sw      t1, 0(t6)
#else
    sb      t1, 0(t6)
#endif
    addi    t4, t4, 1
    j       1b
2:  jr      t3

/* 打印 t0 的 16 位十六进制，返回地址在 t3 */
put_hex:
    li      t1, 60
1:  srl     t2, t0, t1
    andi    t2, t2, 0xf
    li      t5, 10
    blt     t2, t5, 2f
    addi    t2, t2, 'a' - 10
    j       3f
2:  addi    t2, t2, '0'
3:
#if BOARD_UART_32BIT
    sw      t2, 0(t6)
#else
    sb      t2, 0(t6)
#endif
    addi    t1, t1, -4
    bgez    t1, 1b
    jr      t3

    .section .rodata
msg_trap:
    .asciz "\r\n\r\n*** TRAP ***\r\n  mcause = 0x"
msg_epc:
    .asciz "\r\n  mepc   = 0x"
msg_tval:
    .asciz "\r\n  mtval  = 0x"
msg_hint:
    .asciz "\r\n\r\n  mcause 最高位为 1 表示中断，否则是异常。常见异常码：\r\n    1=取指访问错  2=非法指令  5=load访问错  7=store访问错\r\n    4/6=地址未对齐  12/13/15=取指/load/store页错\r\n  mtval 通常是出错的地址或指令编码。\r\n  用 riscv-unknown-elf-addr2line -e qwen3_bm.elf <mepc> 定位到源码行。\r\n\r\n"
