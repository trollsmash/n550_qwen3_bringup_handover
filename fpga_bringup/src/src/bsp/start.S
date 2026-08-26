/* 最小启动代码 —— QEMU virt / 裸机通用。
 *
 * 职责：设栈、清 BSS、开扩展、装 trap handler、跳 C 入口。
 * 若你们的 BSP 已有 crt0，本文件可整个替换掉，只需保证跳转到 qwen3_main()。
 *
 * 板级地址来自 board.h。本文件后缀是大写 .S，会经过 C 预处理器，
 * 所以能直接 include —— UART 基址不要在这里再写一份。
 */
#include "board.h"

/* fence.i 重试上限。取 3：够区分"刷一次就好"与"怎么刷都不行"，
 * 又不至于在真正的非法指令上原地打转刷屏。 */
#define TRAP_MAX_RETRY 3
    .section .text.init, "ax"
    .global _start
_start:
    /* ══════════ 同步指令缓存 ══════════
     *
     * 2026-08-26 FPGA 确认：**硬件复位会清 L1 的 valid 位**。
     * 原先这里有一条 l1d_inv_all 用来丢弃上电残留，现已不需要，去掉。
     *
     * fence.i 留着 —— 它防的是另一个场景：bring-up 时反复换镜像
     *   写镜像 A -> 跑 -> 写镜像 B 到同一地址 -> **软复位**
     * 硬件复位清 valid 位，不等于软复位也清。I-cache 里若残留旧指令，
     * CPU 会接着跑上一版程序，现象是"明明重新烧了，行为却没变"，
     * 越是反复调试越容易撞上，而代价只有一条指令。
     *
     * ★ 它必须待在这个位置：本核的 fence.i 实现是先对 D-cache 做 clean all
     *   再失效 I-cache。此刻还没清 BSS、没写过任何数据，没有脏行可写回，
     *   所以那次 clean all 是空操作。**若挪到写过数据之后**，它就会把脏行
     *   刷进 DDR —— 在别的场景下这正是覆盖掉后门刚写内容的元凶。
     *
     * 局限：fence.i 管不了它**自己之前**的取指。若 I-cache 里 _start 开头
     * 这几条就是旧指令，CPU 根本执行不到这里 —— 那只能靠硬件复位解决。 */
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
    .section .bss
    .align 3
/* trap 时被破坏的寄存器存这里，**不用栈** —— trap 的起因可能正是栈跑飞了，
 * 那时再往栈上写只会把现场彻底毁掉。 */
trap_save:
    .space 8 * 10
/* handler 自己出错时的护栏。没有它，第二次 trap 会覆盖 trap_save，
 * 第一现场就永远拿不回来了 —— 而第一现场才是真正要看的那个。 */
trap_busy:
    .space 8
/* fence.i 重试次数。不设上限的话，真正的非法指令会让它无限重试。 */
trap_retry:
    .space 8

    .section .text.init, "ax"
    .align 4
    .global trap_entry
trap_entry:
    /* 先把要用到的寄存器全存下来，之后才有资格谈"返回现场"。
     * mscratch 解开鸡生蛋：保存基址本身也要占一个寄存器。 */
    csrw    mscratch, t6
    /* 已经在处理 trap 了？那说明 handler 自己出了错。立刻打一行短消息停机，
     * **绝不覆盖 trap_save** —— 保住第一现场比报告第二次错误重要得多。
     * 这一步只许用 t6（唯一已存进 mscratch 的），别的寄存器还没保存。 */
    la      t6, trap_busy
    ld      t6, 0(t6)
    bnez    t6, trap_nested
    csrr    t6, mscratch          /* 取回原 t6，下面按常规流程保存 */

    csrw    mscratch, t6
    la      t6, trap_save
    sd      t0, 0(t6)
    sd      t1, 8(t6)
    sd      t2, 16(t6)
    sd      t3, 24(t6)
    sd      t4, 32(t6)
    sd      t5, 40(t6)
    sd      a1, 48(t6)
    sd      a2, 56(t6)
    csrr    t0, mscratch
    sd      t0, 64(t6)            /* 原 t6 */
    /* 寄存器都存好了，现在才置忙标志 —— 早置会破坏尚未保存的 t5。 */
    la      t0, trap_busy
    li      t1, 1
    sd      t1, 0(t0)

    li      t6, BOARD_UART_BASE   /* 来自 board.h，勿在此硬编码 */

    /* ── 先把完整现场写进 mailbox，再去打串口 ──
     * 顺序是有意的：串口是最不可靠的一环（波特率、线序、终端编码），
     * 若打印中途挂掉，至少 host 用 PCIe 后门还能把现场完整取走。
     * 反过来先打串口的话，串口一挂就什么都不剩了。 */
    li      t5, BOARD_MBOX_TRAP
    csrr    t0, mcause
    sw      t0, 0(t5)
    csrr    t0, mepc
    sw      t0, 4(t5)
    csrr    t0, mtval
    sw      t0, 8(t5)
    csrr    t0, mstatus
    sw      t0, 12(t5)
    sw      ra, 16(t5)          /* 谁调过来的 */
    sw      sp, 20(t5)          /* 栈指针，看是否跑飞 */
    li      t5, BOARD_MBOX_STATUS
    li      t0, BOARD_ST_TRAP
    sw      t0, 0(t5)
    /* CPU 的写会停在 L1D，而 host 后门读 DDR 绕过 cache —— 不写回就白记了。
     * trap 时写回脏行有可能带出半完成的数据，但相比丢掉整个现场，值得。 */
#ifdef BOARD_USE_L1D_ALL
    l1d_clean_all
#endif
    fence   rw, rw

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

    la      t4, msg_mstatus
    jal     t3, put_str
    csrr    t0, mstatus
    jal     t3, put_hex

    la      t4, msg_ra
    jal     t3, put_str
    mv      t0, ra
    jal     t3, put_hex

    la      t4, msg_sp
    jal     t3, put_str
    mv      t0, sp
    jal     t3, put_hex

    /* ★ 关键一项：用 load 把 mepc 处的四字节读回来。
     * 它走 D-cache / DDR 通路，而 mtval 是**取指**通路拿到的编码。
     * 两者不同就当场证明：取指与数据读看到的不是同一份内容，
     * 问题在 I-cache 而不在代码本身。 */
    la      t4, msg_mem
    jal     t3, put_str
    /* ★ mepc 只保证 **2 字节**对齐：压缩指令让它可以停在 4 的非整数倍上。
     * 用 lwu 直接读会触发 misaligned load，handler 自己再 trap 一次，
     * 把第一现场彻底冲掉 —— 拿两个 lhu 拼，2 字节对齐总是满足的。 */
    csrr    t0, mepc
    lhu     t1, 0(t0)
    lhu     t2, 2(t0)
    slli    t2, t2, 16
    or      t0, t1, t2
    jal     t3, put_hex

    la      t4, msg_retry
    jal     t3, put_str
    la      t0, trap_retry
    ld      t0, 0(t0)
    jal     t3, put_hex

    la      t4, msg_hint
    jal     t3, put_str

    /* ── 尝试恢复：fence.i 之后重跑同一条指令 ──
     * 若确是 I-cache 与 DDR 不一致，刷新后重取就拿到正确编码，程序继续往下跑，
     * 串口会看到 decode 正常推进 —— 那就坐实了 I-cache。
     * 仍然 trap 则说明不在 I-cache，重试到上限后按原样停机。
     * mret 回到 mepc 那条指令**重新执行**（而非下一条），正是这里要的语义。 */
    la      t5, trap_retry
    ld      t1, 0(t5)
    li      t2, TRAP_MAX_RETRY
    bge     t1, t2, 3f
    addi    t1, t1, 1
    sd      t1, 0(t5)

    la      t4, msg_try
    jal     t3, put_str

    /* 清忙标志。不清的话，返回后再次 trap 会被当成"handler 内部出错"，
     * 于是只打一行就停机 —— 而那本该是一次完整的新现场。 */
    la      t6, trap_busy
    sd      zero, 0(t6)

    la      t6, trap_save         /* 恢复现场 */
    ld      t0, 0(t6)
    ld      t1, 8(t6)
    ld      t2, 16(t6)
    ld      t3, 24(t6)
    ld      t4, 32(t6)
    ld      t5, 40(t6)
    ld      a1, 48(t6)
    ld      a2, 56(t6)
    ld      t6, 64(t6)            /* 最后恢复 t6，此后基址失效 */
    fence.i
    mret

3:  la      t4, msg_give
    jal     t3, put_str
1:  wfi
    j       1b

/* handler 内部再次出错时走这里。只用 t4/t6 与 put_str，
 * 不动 trap_save，好让上面已经打出的第一现场保持可信。 */
trap_nested:
    li      t6, BOARD_UART_BASE
    la      t4, msg_nested
    jal     t3, put_str
2:  wfi
    j       2b

/* LSR 的寄存器索引是 5；字节偏移随 REG_SHIFT 变（本核 32 位寄存器 -> 0x14）。 */
#define UART_LSR_OFF  (5 << BOARD_UART_REG_SHIFT)
#define UART_LSR_THRE 0x20

/* 打印以 NUL 结尾的字符串，地址在 t4，返回地址在 t3。破坏 t1/a1/a2。
 *
 * ★ 每个字符都要先等 THR 空。原先直接写，FIFO 一满就丢字符 ——
 *   现象是 trap 信息打到一半断掉（"mcause = 0x00 m" 就是这么来的），
 *   而 trap 现场只打印一次，丢了没有第二次机会。
 *   等待带计数上限：UART 若根本没配好，死等会把"发生了 trap"本身也吞掉。 */
put_str:
1:  lbu     t1, 0(t4)          /* 读字符串，来自内存，始终按字节 */
    beqz    t1, 3f
    li      a2, 100000
2:
#if BOARD_UART_32BIT
    lw      a1, UART_LSR_OFF(t6)
#else
    lbu     a1, UART_LSR_OFF(t6)
#endif
    andi    a1, a1, UART_LSR_THRE
    bnez    a1, 4f
    addi    a2, a2, -1
    bnez    a2, 2b
4:
    /* THR 的寄存器索引为 0，字节偏移恒为 0，不受 REG_SHIFT 影响；
     * 这里只需匹配访问宽度。 */
#if BOARD_UART_32BIT
    sw      t1, 0(t6)
#else
    sb      t1, 0(t6)
#endif
    addi    t4, t4, 1
    j       1b
3:  jr      t3

/* 打印 t0 的 16 位十六进制，返回地址在 t3。破坏 t1/t2/t5/a1/a2。
 * 与 put_str 同样要等 THR 空，理由见上。 */
put_hex:
    li      t1, 60
1:  srl     t2, t0, t1
    andi    t2, t2, 0xf
    li      t5, 10
    blt     t2, t5, 2f
    addi    t2, t2, 'a' - 10
    j       3f
2:  addi    t2, t2, '0'
3:  li      a2, 100000
5:
#if BOARD_UART_32BIT
    lw      a1, UART_LSR_OFF(t6)
#else
    lbu     a1, UART_LSR_OFF(t6)
#endif
    andi    a1, a1, UART_LSR_THRE
    bnez    a1, 6f
    addi    a2, a2, -1
    bnez    a2, 5b
6:
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
msg_mstatus:
    .asciz "\r\n  mstatus= 0x"
msg_ra:
    .asciz "\r\n  ra     = 0x"
msg_sp:
    .asciz "\r\n  sp     = 0x"
msg_mem:
    .asciz "\r\n  [mepc] = 0x"
msg_retry:
    .asciz "\r\n  已重试 = 0x"
msg_try:
    .asciz "\r\n\r\n  >> fence.i 后返回原处重试 <<\r\n"
msg_give:
    .asciz "\r\n\r\n  >> 已达重试上限，停机 <<\r\n"
msg_nested:
    .asciz "\r\n\r\n  >> handler 内部再次 trap，以上为第一现场，停机 <<\r\n"
msg_hint:
    .asciz "\r\n\r\n  mcause 最高位为 1 表示中断，否则是异常。常见异常码：\r\n    1=取指访问错  2=非法指令  5=load访问错  7=store访问错\r\n    4/6=地址未对齐  12/13/15=取指/load/store页错\r\n  mtval 通常是出错的地址或指令编码。\r\n  用 riscv-unknown-elf-addr2line -e qwen3_bm.elf <mepc> 定位到源码行。\r\n\r\n"
