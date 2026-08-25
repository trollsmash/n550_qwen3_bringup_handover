# 自动生成 by tools/40_gen_ame_vectors.py —— 请勿手改
# case 001 bf16_full_tile
#   满 tile 基线。M/N/K 全部取最大值，覆盖 128x128 个输出元素的完整通路。先过这一条再看别的：它失败说明是根本性问题，查后面的没有意义。
#
# C[128][128] = A[128][32] * B[128][32]^T   (BF16 -> FP32)
# 共 1 条 mfmacc 累加进同一个累加器

    .equ TEXT_ADDR, 0x80000000
    .equ A_ADDR, 0x80001000
    .equ B_ADDR, 0x80050000
    .equ C_ADDR, 0x800A0000
    .equ REF_ADDR, 0x800B0000
    .equ DONE_ADDR, 0x800C0000
    .equ STATUS_ADDR, 0x800C0004
    .equ FAILIDX_ADDR, 0x800C0008
    .equ A_SRC_ADDR, 0x800D0000
    .equ XOR_ADDR, 0x80160000
    .equ XOR_REF_ADDR, 0x80160004
    .equ STAGE_A_ADDR, 0x80160008
    .equ STAGE_B_ADDR, 0x8016000C
    .equ PROBE1_ADDR, 0x80160010
    .equ PROBE2_ADDR, 0x80160014
    .equ PROBE_EQ_ADDR, 0x80160018
    .equ TRAP_CAUSE_ADDR, 0x8016001C
    .equ TRAP_EPC_ADDR, 0x80160020
    .equ TRAP_TVAL_ADDR, 0x80160024
    .equ DONE_MAGIC, 0x600D600D
    .equ ST_PASS,    0x50415353
    .equ ST_FAIL,    0x4641494C
    .equ N_ELEM,     16384

    .section .text.init, "ax"
    .global _start
_start:
    # 只让 hart 0 跑，其余原地睡
    csrr    t0, mhartid
    bnez    t0, park

#ifdef FOR_FPGA
    # ---- 复位后先把 cache 清空（详见函数注释）----
    # ★ 这两条的顺序不能反：本核的 fence.i 实现是先对 D-cache 做
    #   clean all，再失效 I-cache。若 cache 里有上电残留的随机脏行，
    #   那一步会把它们写回 DDR，反过来破坏刚刚加载进去的镜像。
    #   所以必须先 inv_all 丢弃（不写回），再让 fence.i 去清 I-cache。
    l1d_inv_all
    fence   rw, rw
    fence.i
#endif

    # 使能三个扩展的上下文状态域，写 01(Initial) 即可。
    # MS[30:29] 最易遗漏：不开则所有 AME 指令报 illegal instruction，
    # 现象与「指令编码不支持」一模一样，极容易误判成译码问题。
    li      t0, (1 << 13) | (1 << 9) | (1 << 29)   # FS | VS | MS
    csrs    mstatus, t0

    # 固化舍入模式为 RNE。参考结果按 RNE 生成，不写这条就得看
    # 复位默认值，各家实现不一定相同。
    csrw    frm, x0

    # tile: mtilem=128 mtilen=128 mtilek=32
    li      t0, 128
    csrw    0x803, t0            # mtilem
    li      t0, 128
    csrw    0x804, t0            # mtilen

    mzero   acc0

    # ---- K 块 0: k=0..31 ----
    li      t0, 32
    csrw    0x805, t0            # mtilek
    li      a0, A_ADDR + 0
    li      a1, 64
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 0
    li      a1, 64
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0   # (md, B, A) —— 顺序写反会得到转置

    # 回写 C
    li      a0, C_ADDR
    li      a1, 512
    msce32  acc0,(a0),a1
    mrelease

#ifdef FOR_FPGA
    # AME 直写 DDR，CPU 侧需失效旧拷贝才能读到真值（详见函数注释）
    l1d_clean_all
    l1d_inv_all
    fence   rw, rw
#endif

    # 完成标志：testbench 轮询这个地址即可，不必猜时序
    li      t0, DONE_MAGIC
    li      t1, DONE_ADDR
    sw      t0, 0(t1)

    # ---- 自检：与镜像里的 c_ref 逐字比对 ----
    # 纯粹是给同事省事的：不想用它就只看 DONE 然后自己 dump C 区。
    li      a0, C_ADDR
    li      a1, REF_ADDR
    li      a2, N_ELEM
    li      a3, 0                # 当前下标
    li      a4, 0                # 不符计数
    li      a5, 0                # 是否已记录首个不符
    li      a6, -1               # 首个不符的下标
cmp_loop:
    beqz    a2, cmp_done
    lw      t0, 0(a0)
    lw      t1, 0(a1)
    beq     t0, t1, cmp_next
    addi    a4, a4, 1
    bnez    a5, cmp_next
    li      a5, 1
    mv      a6, a3
cmp_next:
    addi    a0, a0, 4
    addi    a1, a1, 4
    addi    a3, a3, 1
    addi    a2, a2, -1
    j       cmp_loop
cmp_done:
    li      t1, FAILIDX_ADDR
    sw      a6, 0(t1)
    li      t0, ST_PASS
    beqz    a4, 1f
    li      t0, ST_FAIL
1:  li      t1, STATUS_ADDR
    sw      t0, 0(t1)

#ifdef FOR_FPGA
    # 让 host 后门能看到上面写的标志（详见函数注释）
    l1d_clean_all
    fence   rw, rw
#endif

#ifdef FOR_FPGA
# ================= 串口输出（仅 FPGA 版编入） =================
# UART 地址按板而异，默认 S2C；换板用 -DUART_BASE=... 覆盖。
# 16550 的 LSR 在寄存器索引 5；本核寄存器 32 位、间距 4 字节，故偏移 5<<2。
#ifndef UART_BASE
#define UART_BASE 0x20100000
#endif
#ifndef UART_LSR_OFF
#define UART_LSR_OFF 0x14
#endif
# 寄存器宽度也随板而变：本核的 16550 是 32 位寄存器、间距 4 字节，
# 而 QEMU virt 那种是 8 位、间距 1 字节，对它用 sw 根本不响应。
# 在 QEMU 上验证时加 -DUART_BYTE_ACCESS -DUART_LSR_OFF=5。
#ifdef UART_BYTE_ACCESS
#define UART_LD lbu
#define UART_ST sb
#else
#define UART_LD lw
#define UART_ST sw
#endif

    la      a0, umsg_hdr
    jal     ra, uputs
    li      t1, STATUS_ADDR
    lw      t0, 0(t1)
    li      t2, ST_PASS
    beq     t0, t2, upass
    la      a0, umsg_fail
    jal     ra, uputs
    li      t1, FAILIDX_ADDR
    lw      s1, 0(t1)
    mv      a0, s1
    jal     ra, uputhex
    # 顺带把该元素的实际值与期望值打出来：光有下标还得再去读内存。
    slli    s2, s1, 2
    la      a0, umsg_got
    jal     ra, uputs
    li      t3, C_ADDR
    add     t3, t3, s2
    lw      a0, 0(t3)
    jal     ra, uputhex
    la      a0, umsg_exp
    jal     ra, uputs
    li      t3, REF_ADDR
    add     t3, t3, s2
    lw      a0, 0(t3)
    jal     ra, uputhex
    la      a0, umsg_nl
    jal     ra, uputs
    j       udone
upass:
    la      a0, umsg_pass
    jal     ra, uputs
udone:
    j       park

# 打印 NUL 结尾字符串，a0 = 首地址。
uputs:
    li      t6, UART_BASE
1:  lbu     t0, 0(a0)
    beqz    t0, 3f
2:  UART_LD t1, UART_LSR_OFF(t6)
    andi    t1, t1, 0x20
    beqz    t1, 2b
    UART_ST t0, 0(t6)
    addi    a0, a0, 1
    j       1b
3:  ret

# 打印 a0 的 8 位十六进制。
uputhex:
    li      t6, UART_BASE
    li      t2, 28
1:  srl     t0, a0, t2
    andi    t0, t0, 0xf
    li      t1, 10
    blt     t0, t1, 2f
    addi    t0, t0, 'a' - 10
    j       3f
2:  addi    t0, t0, '0'
3:  UART_LD t1, UART_LSR_OFF(t6)
    andi    t1, t1, 0x20
    beqz    t1, 3b
    UART_ST t0, 0(t6)
    addi    t2, t2, -4
    bgez    t2, 1b
    ret

# 字符串放进独立的 .text.umsg 段。
# 不能用 .rodata：prog.ld 只给 .text 划了位置，孤立的 .rodata 没有归宿。
# 也不能直接塞在 .text.init 里：字符串长度不是 4 的倍数时会把后面的
# 指令挤歪，链接器在 relax 阶段报 bad value，且不指出真正原因。
# 用独立段名，prog.ld 的 *(.text*) 会收走它并排在所有代码之后。
    .section .text.umsg, "a"
    .align 2
umsg_hdr:  .asciz "\r\n=== case 001 bf16_full_tile  M=128 N=128 K=32 ===\r\n"
umsg_pass: .asciz "RESULT: PASS\r\n"
umsg_fail: .asciz "RESULT: FAIL  idx="
umsg_got:  .asciz "  got="
umsg_exp:  .asciz "  exp="
umsg_nl:   .asciz "\r\n"
    # 切回代码段。.section 之后的内容都归新段，后面还有 park 循环，
    # 留在数据段里链接器会在 relax 阶段报 bad value。
    .section .text.init, "ax"
#endif

park:
    wfi
    j       park

    # ---- 数据段：地址由 prog.ld 钉死 ----
    .section .data.a, "a"
    .incbin "a.bin"
    .section .data.b, "a"
    .incbin "b.bin"
    .section .data.c, "a"
    # C 区预置 0xA5A5A5A5：msce32 若漏写某些元素，
    # 留下的是这个值而不是碰巧正确的 0。
    .rept 16384
    .word 0xA5A5A5A5
    .endr
    .section .data.ref, "a"
    .incbin "c_ref.bin"
