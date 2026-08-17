# 自动生成 by tools/40_gen_ame_vectors.py —— 请勿手改
# case 009a rvv_ame_handoff_cbo
#   RVV 与 AME 的双向数据交接，缓存同步用标准 Zicbom（cbo.clean/inval）。
#   拆成两段各自报状态，否则失败时分不清是哪个方向坏的：
#     阶段A  RVV 写 BF16 -> clean -> AME 读。失败 => clean 没生效
#     阶段B  AME 写 C -> inval -> RVV 读并 XOR 归约。失败 => inval 没生效
#   这是真实负载的形态：模型里 RVV 与 AME 每层都要交替七次。
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
    # FS|VS|MS 三个上下文状态域都要开：本 case 同时用到浮点、向量、矩阵
    li      t0, (1 << 13) | (1 << 9) | (1 << 29)
    csrs    mstatus, t0
    csrw    frm, x0                # RNE，与参考模型一致
    # 装 trap handler。不装的话 mtvec 默认为 0，任何异常都会跳到
    # 地址 0 执行垃圾：现象是标志位全 0、CPU 占满，与死循环无法区分。
    la      t0, trap_entry
    csrw    mtvec, t0

    # ═══════ 阶段 A：验 RVV -> AME 方向 ═══════
    # RVV 把 FP32 激活转成 BF16 写进 A 区（数据此刻在 L1D 里是脏的），
    # 经 clean 写回 DDR 后，AME 才可能读到正确值。
    # 若此段失败 => clean 没生效，AME 从 DDR 读到了旧值/垃圾。
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

    # ---- AME GEMM (阶段A) ----
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

    # 阶段 A 的产物 C 由 AME 直接写 DDR，testbench 可不经 RVV 直接比对。
    li      t0, DONE_MAGIC
    li      t1, STAGE_A_ADDR
    sw      t0, 0(t1)

    # ═══════ 阶段 B：验 AME -> RVV 方向 ═══════
    # AME 已把 C 写进 DDR，但 L1D 里可能还留着 C 地址的旧拷贝，
    # 必须失效后 RVV 才能读到新值。
    # 若此段失败 => inval 没生效，RVV 命中了 L1D 里的旧数据。
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
    li      a0, XOR_ADDR
    sw      t2, 0(a0)
    li      t0, DONE_MAGIC
    li      t1, STAGE_B_ADDR
    sw      t0, 0(t1)

    li      t0, DONE_MAGIC
    li      t1, DONE_ADDR
    sw      t0, 0(t1)

    # ---- 自检：比对 C 与镜像里的 c_ref ----
    li      a0, C_ADDR
    li      a1, REF_ADDR
    li      a2, 4096
    li      a3, 0
    li      a4, 0
    li      a5, 0
    li      a6, -1
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
    # 阶段 B 的 XOR 结果也要核：它证明 RVV 确实读到了 AME 写的新值
    li      a0, XOR_ADDR
    lw      t0, 0(a0)
    li      a0, XOR_REF_ADDR
    lw      t1, 0(a0)
    beq     t0, t1, 6f
    addi    a4, a4, 1
6:
    li      t0, ST_PASS
    beqz    a4, 7f
    li      t0, ST_FAIL
7:  li      t1, STATUS_ADDR
    sw      t0, 0(t1)

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
