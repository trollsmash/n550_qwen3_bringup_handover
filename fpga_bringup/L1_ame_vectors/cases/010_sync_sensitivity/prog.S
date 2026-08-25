# 自动生成 by tools/40_gen_ame_vectors.py —— 请勿手改
# case 010 sync_sensitivity
#   同一计算做两遍：第一遍带完整同步，第二遍故意去掉所有 cache 操作，
#   把两次的 XOR 校验和都回填，并给出是否相同。
#   ★ 第二遍不作判据 —— 它依赖 cache 当时的状态，通过与否都不能证明什么。
#     两次不同 => 证明软件同步确实必要、序列也确实有效
#     两次相同 => **不能**推出'不需要同步'，可能只是数据恰好未驻留 cache
#   它把'到底需不需要软件维护一致性'从口头约定变成可测量的事实。
#
# ★ 本 case 混合 RVV 与 AME，依赖缓存同步与访存定序。
#   请先确保 001-008（纯 AME，不依赖 cache）全部通过再跑这里，
#   否则 cache 子系统的问题会掩盖矩阵单元本身的问题。

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

    .section .text.init, "ax"
    .global _start
_start:
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
    # FS|VS|MS 三个上下文状态域都要开：本 case 同时用到浮点、向量、矩阵
    li      t0, (1 << 13) | (1 << 9) | (1 << 29)
    csrs    mstatus, t0
    csrw    frm, x0                # RNE，与参考模型一致
    # 装 trap handler。不装的话 mtvec 默认为 0，任何异常都会跳到
    # 地址 0 执行垃圾：现象是标志位全 0、CPU 占满，与死循环无法区分。
    la      t0, trap_entry
    csrw    mtvec, t0

    # ═══════ 第一遍：带完整同步（这一遍的结果才是判据）═══════
    li      a0, A_SRC_ADDR
    li      a1, A_ADDR
    li      a2, 8192
    li      t3, 0x7fff
2:  vsetvli t0, a2, e32, m4, ta, ma
    vle32.v v8, (a0)
    vsrl.vi v12, v8, 16
    vand.vi v12, v12, 1
    vadd.vx v12, v12, t3
    vadd.vv v8, v8, v12
    vsetvli zero, t0, e16, m2, ta, ma
    vnsrl.wi v4, v8, 16
    vse16.v v4, (a1)
    slli    t1, t0, 2
    add     a0, a0, t1
    slli    t1, t0, 1
    add     a1, a1, t1
    sub     a2, a2, t0
    bnez    a2, 2b
    li      a0, A_ADDR
    li      a1, A_ADDR + 16384
1:  cbo.clean (a0)
    addi    a0, a0, 64
    bltu    a0, a1, 1b
    fence   rw,rw
    # ---- AME GEMM (带同步) ----
    li      t0, 32
    csrw    0x803, t0
    li      t0, 128
    csrw    0x804, t0
    mzero   acc0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 0
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 0
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 64
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 64
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 128
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 128
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 192
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 192
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 256
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 256
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 320
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 320
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 384
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 384
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 448
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 448
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      a0, C_ADDR
    li      a1, 512
    msce32  acc0,(a0),a1
    mrelease
    fence   rw,rw
    li      a0, C_ADDR
    li      a1, C_ADDR + 16384
1:  cbo.inval (a0)
    addi    a0, a0, 64
    bltu    a0, a1, 1b
    fence   rw,rw
    li      a0, C_ADDR
    li      a2, 4096
    vsetvli t0, a2, e32, m8, ta, ma
    vmv.v.i v16, 0
3:  vsetvli t0, a2, e32, m8, ta, ma
    vle32.v v8, (a0)
    vxor.vv v16, v16, v8
    slli    t1, t0, 2
    add     a0, a0, t1
    sub     a2, a2, t0
    bnez    a2, 3b
    vsetvli zero, zero, e32, m1, ta, ma
    vmv.v.i v1, 0
    vsetvli t0, zero, e32, m8, ta, ma
    vredxor.vs v1, v16, v1
    vmv.x.s t2, v1
    li      a0, PROBE1_ADDR
    sw      t2, 0(a0)

    # ═══════ 第二遍：故意去掉所有 cache 操作 ═══════
    # ★ 这一遍的结果**不作判据**。它依赖 cache 当时的状态，
    #   通过与否都不能证明什么。它的价值是当观察项：
    #     两次不同 => 证明软件同步确实必要，序列也确实有效
    #     两次相同 => 不能推出'不需要同步'，可能只是数据恰好未驻留 cache
    #   千万不要拿'不加同步也过了'去论证软件可以省掉同步。
    # 先改写 A 源数据，制造出'L1D 里是新值、DDR 里是旧值'的局面。
    li      a0, A_SRC_ADDR
    li      a2, 64
    vsetvli t0, a2, e32, m4, ta, ma
    vle32.v v8, (a0)
    vfadd.vv v8, v8, v8              # 值翻倍，制造差异
    vse32.v v8, (a0)
    li      a0, A_SRC_ADDR
    li      a1, A_ADDR
    li      a2, 8192
    li      t3, 0x7fff
2:  vsetvli t0, a2, e32, m4, ta, ma
    vle32.v v8, (a0)
    vsrl.vi v12, v8, 16
    vand.vi v12, v12, 1
    vadd.vx v12, v12, t3
    vadd.vv v8, v8, v12
    vsetvli zero, t0, e16, m2, ta, ma
    vnsrl.wi v4, v8, 16
    vse16.v v4, (a1)
    slli    t1, t0, 2
    add     a0, a0, t1
    slli    t1, t0, 1
    add     a1, a1, t1
    sub     a2, a2, t0
    bnez    a2, 2b
    # 这里**不做** clean，AME 若绕过 L1D 就会读到旧值
    fence   rw,rw
    # ---- AME GEMM (无同步) ----
    li      t0, 32
    csrw    0x803, t0
    li      t0, 128
    csrw    0x804, t0
    mzero   acc0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 0
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 0
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 64
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 64
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 128
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 128
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 192
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 192
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 256
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 256
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 320
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 320
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 384
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 384
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      t0, 32
    csrw    0x805, t0
    li      a0, A_ADDR + 448
    li      a1, 512
    mlae16  tr0,(a0),a1
    li      a0, B_ADDR + 448
    li      a1, 512
    mlbe16  tr1,(a0),a1
    mfmacc.s.bf16 acc0,tr1,tr0
    li      a0, C_ADDR
    li      a1, 512
    msce32  acc0,(a0),a1
    mrelease
    fence   rw,rw
    # 这里**不做** inval
    li      a0, C_ADDR
    li      a2, 4096
    vsetvli t0, a2, e32, m8, ta, ma
    vmv.v.i v16, 0
3:  vsetvli t0, a2, e32, m8, ta, ma
    vle32.v v8, (a0)
    vxor.vv v16, v16, v8
    slli    t1, t0, 2
    add     a0, a0, t1
    sub     a2, a2, t0
    bnez    a2, 3b
    vsetvli zero, zero, e32, m1, ta, ma
    vmv.v.i v1, 0
    vsetvli t0, zero, e32, m8, ta, ma
    vredxor.vs v1, v16, v1
    vmv.x.s t2, v1
    li      a0, PROBE2_ADDR
    sw      t2, 0(a0)

    # 记录两次是否相同，供 testbench 直接读
    li      a0, PROBE1_ADDR
    lw      t0, 0(a0)
    li      a0, PROBE2_ADDR
    lw      t1, 0(a0)
    li      t2, 0
    bne     t0, t1, 4f
    li      t2, 1
4:  li      a0, PROBE_EQ_ADDR
    sw      t2, 0(a0)
    li      t0, DONE_MAGIC
    li      t1, STAGE_A_ADDR
    sw      t0, 0(t1)

    li      t0, DONE_MAGIC
    li      t1, DONE_ADDR
    sw      t0, 0(t1)

    # ---- 自检：只看第一遍（带同步）的 XOR ----
    # 不比对 C：第二遍故意改了输入，C 已非 c_ref，拿它当判据必然误报。
    li      a4, 0
    li      a0, PROBE1_ADDR
    lw      t0, 0(a0)
    li      a0, XOR_REF_ADDR
    lw      t1, 0(a0)
    beq     t0, t1, 9f
    li      a4, 1
9:  li      t1, FAILIDX_ADDR
    li      t0, -1
    sw      t0, 0(t1)
    li      t0, ST_PASS
    beqz    a4, 7f
    li      t0, ST_FAIL
7:  li      t1, STATUS_ADDR
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
umsg_hdr:  .asciz "\r\n=== case 010 sync_sensitivity  M=32 N=128 K=256 ===\r\n"
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

    # ---- Trap handler ----
    # 把异常原因原样记到 mailbox 并停机。不做恢复：这里出异常
    # 意味着测试环境或硬件有问题，继续跑只会得到无意义的结果。
    # 没有它的话，异常会跳到 mtvec(=0) 执行垃圾，现象是标志位全 0、
    # CPU 占满 —— 与死循环完全无法区分。
    # 常见 mcause：2=非法指令 5=load访问错 7=store访问错 4/6=地址未对齐
    # mtval 通常是出错的地址或指令编码。
    .align 4
trap_entry:
    csrr    t0, mcause
    li      t1, TRAP_CAUSE_ADDR
    sw      t0, 0(t1)
    csrr    t0, mepc
    li      t1, TRAP_EPC_ADDR
    sw      t0, 0(t1)
    csrr    t0, mtval
    li      t1, TRAP_TVAL_ADDR
    sw      t0, 0(t1)
    li      t0, ST_FAIL
    li      t1, STATUS_ADDR
    sw      t0, 0(t1)
8:  wfi
    j       8b

    .section .data.a, "a"
    # A 区由 RVV 在运行时填写，这里预置 0x5A5A：
    # RVV 若漏写，AME 读到的会是这个值而不是碰巧正确的 0。
    .rept 8192
    .short 0x5A5A
    .endr
    .section .data.b, "a"
    .incbin "b.bin"
    .section .data.c, "a"
    .rept 4096
    .word 0xA5A5A5A5
    .endr
    .section .data.ref, "a"
    .incbin "c_ref.bin"
    .section .data.asrc, "a"
    .incbin "a_src.bin"
    .section .data.xorref, "a"
    .incbin "xor_ref.bin"
