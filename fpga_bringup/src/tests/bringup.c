/* L0 —— FPGA 上板第一个程序。
 *
 * 目的：在跑任何模型代码之前，把板子的基础通路一项项验穿。
 * 全模型镜像 1.1 GB，加载慢、出错难定位；基础通路不通时跑它纯属浪费。
 *
 * ══════════════ 设计要点：为什么每步都写 mailbox ══════════════
 *
 * 串口是**最不可靠**的观测手段：波特率、分频、引脚、电平，任何一环没配好
 * 就是一片死寂，而"串口没配好"和"CPU 没跑起来"现象完全一样。
 *
 * PCIe 后门能回读 DDR，所以真正的观测主通道是 mailbox：每完成一件事
 * 写一个魔数，host 读回就知道走到第几步。串口只是锦上添花。
 *
 * ══════════════ 六个阶段 ══════════════
 *
 *   0xB0000010  取指执行 + DDR 可写      能读到 => 复位放开了、CPU 活着
 *   0xB0000020  串口已输出               读到但屏幕没字 => 串口配置问题
 *   0xB0000030  回读 host 预置 magic     反证 PCIe 写 DDR 真的生效
 *   0xB0000040  RVV 可用，vlenb 已回填
 *   0xB0000050  AME 最小 GEMM 结果正确
 *   0xB0000060  性能采样完成，周期数已回填
 *   0xB00000AA  全部通过
 *   0xB00000Fx  第 x 步失败
 *
 * 第三步最关键：host 先往 MBOX+0x80 写 0x5A5AC3C3，程序读回来比对。
 * 没有这一步，"PCIe 没写进去"和"程序崩了"分不开：两者的现象一模一样，
 * 都是标志位全 0、串口没动静。
 *
 * 复用 src/bsp/start.S 而不是自己写启动代码：这样 L0 顺带验证了
 * gp/栈/BSS 清零/mstatus 三个扩展位/trap handler，而它们正是
 * L2、L3 要用的同一份代码。
 *
 * 编译运行见 tools/build_riscv.sh bringup
 */
#include <stdint.h>

#include "bsp/board.h"

/* ══════════════ mailbox 布局（host 侧按这张表读写） ══════════════ */
#define MB(off)         (BOARD_MBOX_ADDR + (off))
#define MB_STATUS       MB(0x00)   /* [读] 阶段魔数，见文件头 */
#define MB_VLENB        MB(0x04)   /* [读] 实测 vlenb，应为 128 (VLEN=1024bit) */
#define MB_AME_RESULT   MB(0x08)   /* [读] AME 自测结果的 FP32 位模式，应 0x42000000 */
#define MB_CYCLES_LO    MB(0x0C)   /* [读] 性能采样的 mcycle 差值，低 32 位 */
#define MB_CYCLES_HI    MB(0x10)
#define MB_ITERS        MB(0x14)   /* [读] 采样跑了多少次 mfmacc */
#define MB_MTILE_CAP    MB(0x18)   /* [读] AME 能力 CSR，便于核对硬件配置 */
/* [读] 失败步骤的位掩码，bit n = 第 n 步未通过；0 表示全过。
 * 单靠 MB_STATUS 不够：软失败（如 step3）之后程序继续跑，
 * 后面几步的状态写入会把失败痕迹覆盖掉。这个字段只置位、不清除。 */
#define MB_FAILMASK     MB(0x1C)
#define MB_HOST_MAGIC   MB(0x80)   /* [写] host 上电前写 0x5A5AC3C3 */
#define MB_HOST_ECHO    MB(0x84)   /* [读] 程序读回的值，不符时看这里是什么 */

#define HOST_MAGIC      0x5A5AC3C3u

#define ST_BOOT         0xB0000010u
#define ST_UART         0xB0000020u
#define ST_PCIE         0xB0000030u
#define ST_RVV          0xB0000040u
#define ST_AME          0xB0000050u
#define ST_PERF         0xB0000060u
#define ST_ALLPASS      0xB00000AAu
#define ST_FAIL(n)      (0xB00000F0u | (n))

static void mb_set(unsigned long a, uint32_t v) {
    *BOARD_PTR(uint32_t, a) = v;
}
static uint32_t mb_get(unsigned long a) {
    return *BOARD_PTR(uint32_t, a);
}

/* ══════════════ 串口 ══════════════
 *
 * 默认**不做初始化**：QEMU virt 与多数 FPGA 参考设计里 bootrom 已配好。
 * 若上板后 mailbox 能走到 ST_UART 之后但屏幕一个字都没有，
 * 就是波特率没配 —— 打开 -DUART_NEEDS_INIT 并填对下面的分频值。
 * 分频值 = UART 输入时钟 / (16 × 波特率)，需要硬件团队给输入时钟。 */
#define UART_THR   0x00
#define UART_IER   0x01
#define UART_FCR   0x02
#define UART_LCR   0x03
#define UART_LSR   0x05
#define UART_DLL   0x00      /* DLAB=1 时 */
#define UART_DLM   0x01
#define LSR_THRE   0x20

static volatile uint8_t *uart_reg(int off) {
    return BOARD_PTR(uint8_t, BOARD_UART_BASE + off);
}

static void uart_init(void) {
#ifdef UART_NEEDS_INIT
#ifndef UART_DIVISOR
#define UART_DIVISOR 27      /* ⚠ 占位。50MHz/(16*115200)=27.1 —— 但 UART 的
                              *   输入时钟未必等于 CPU 时钟，务必向硬件确认。 */
#endif
    *uart_reg(UART_IER) = 0x00;              /* 关中断 */
    *uart_reg(UART_LCR) = 0x80;              /* DLAB=1，露出分频寄存器 */
    *uart_reg(UART_DLL) = UART_DIVISOR & 0xFF;
    *uart_reg(UART_DLM) = (UART_DIVISOR >> 8) & 0xFF;
    *uart_reg(UART_LCR) = 0x03;              /* DLAB=0, 8N1 */
    *uart_reg(UART_FCR) = 0x07;              /* 开 FIFO 并清空 */
#endif
}

static void putc_(char c) {
    /* 有上限的轮询：真机上等 THRE 是对的，但 LSR 读不到预期值时不能死等，
     * 否则整个程序静默挂死 —— 而这恰恰是串口没配好时的表现。
     * 宁可字符发飞，也要让程序继续往下走，把 mailbox 填完。 */
    for (int i = 0; i < 100000; i++)
        if (*uart_reg(UART_LSR) & LSR_THRE) break;
    *uart_reg(UART_THR) = (uint8_t)c;
}
static void P(const char *s) { while (*s) { if (*s == '\n') putc_('\r'); putc_(*s++); } }
static void U(unsigned long v) {
    char b[24]; int i = 0;
    if (!v) { putc_('0'); return; }
    while (v) { b[i++] = (char)('0' + v % 10); v /= 10; }
    while (i) putc_(b[--i]);
}
static void X(unsigned long v) {
    P("0x");
    for (int s = 28; s >= 0; s -= 4) putc_("0123456789abcdef"[(v >> s) & 0xf]);
}

/* ══════════════ AME 自测 ══════════════
 * 最小规模：1×1×32，全 1 × 全 1 = 32.0（FP32 位模式 0x42000000）。
 * 结果精确、与累加顺序无关，任何舍入策略都得到同一个值 ——
 * 作为"矩阵单元是否活着"的判据，不掺杂数值语义的争议。 */
#define TAIL_PAD_ELEMS (8192 / 2)
/* 尾部填充不是保险：tile 装载按 stride 走满整个 tile 的地址范围，
 * 即使 mtilem=1 也一样。按有效数据大小紧贴分配会读到非法地址。 */
static uint16_t g_a[32 + TAIL_PAD_ELEMS] __attribute__((aligned(64)));
static uint16_t g_b[32 + TAIL_PAD_ELEMS] __attribute__((aligned(64)));
static float    g_c[1 + 8192 / 4]        __attribute__((aligned(64)));

static void set_tile(unsigned long m, unsigned long n, unsigned long k) {
    __asm__ volatile("csrw 0x803, %0" :: "r"(m));
    __asm__ volatile("csrw 0x804, %0" :: "r"(n));
    __asm__ volatile("csrw 0x805, %0" :: "r"(k));
}

static uint32_t ame_selftest(void) {
    for (int i = 0; i < 32; i++) { g_a[i] = 0x3F80; g_b[i] = 0x3F80; }  /* 1.0f */
    g_c[0] = -1.0f;
    set_tile(1, 1, 32);
    __asm__ volatile("mzero acc0");
    { register const void *p __asm__("a0") = g_a; register long s __asm__("a1") = 64;
      __asm__ volatile("mlae16 tr0,(%0),%1" :: "r"(p), "r"(s) : "memory"); }
    { register const void *p __asm__("a0") = g_b; register long s __asm__("a1") = 64;
      __asm__ volatile("mlbe16 tr1,(%0),%1" :: "r"(p), "r"(s) : "memory"); }
    __asm__ volatile("mfmacc.s.bf16 acc0,tr1,tr0");
    { register void *p __asm__("a0") = g_c; register long s __asm__("a1") = 4;
      __asm__ volatile("msce32 acc0,(%0),%1" :: "r"(p), "r"(s) : "memory"); }
    __asm__ volatile("mrelease");
    union { float f; uint32_t u; } v; v.f = g_c[0];
    return v.u;
}

/* ══════════════ 性能采样 ══════════════
 *
 * 跑 N 次满 tile（128×128×32）的装载+mfmacc，报告 mcycle 差值。
 * 这一个数就能外推整个 demo 的时间预算：
 *
 *   decode 一个 token 需要约 145500 次 mfmacc
 *     （28 层 × 3840，加 lm_head 的 1187×32；M=1 时 tile 利用率仅 1/128）
 *   每次 X 周期 => 时间 = 145500 × X / 50MHz
 *     X=100  => 0.29 秒/token
 *     X=1000 => 2.9 秒/token
 *
 * 在此之前这个数只能靠猜，猜的区间跨了一个数量级。 */
#define PERF_TILE_M 128
#define PERF_TILE_N 128
#define PERF_ITERS  1000
static uint16_t g_pa[128 * 32 + TAIL_PAD_ELEMS] __attribute__((aligned(64)));
static uint16_t g_pb[128 * 32 + TAIL_PAD_ELEMS] __attribute__((aligned(64)));
static float    g_pc[128 * 128 + 8192 / 4]      __attribute__((aligned(64)));

static uint64_t rd_mcycle(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

static uint64_t perf_sample(void) {
    for (int i = 0; i < 128 * 32; i++) { g_pa[i] = 0x3F80; g_pb[i] = 0x3F80; }
    set_tile(PERF_TILE_M, PERF_TILE_N, 32);
    __asm__ volatile("mzero acc0");

    uint64_t t0 = rd_mcycle();
    for (int it = 0; it < PERF_ITERS; it++) {
        { register const void *p __asm__("a0") = g_pa; register long s __asm__("a1") = 64;
          __asm__ volatile("mlae16 tr0,(%0),%1" :: "r"(p), "r"(s) : "memory"); }
        { register const void *p __asm__("a0") = g_pb; register long s __asm__("a1") = 64;
          __asm__ volatile("mlbe16 tr1,(%0),%1" :: "r"(p), "r"(s) : "memory"); }
        __asm__ volatile("mfmacc.s.bf16 acc0,tr1,tr0");
    }
    uint64_t t1 = rd_mcycle();

    /* 把累加器写出来，防止编译器或硬件把整个循环当成无副作用而优化/跳过。 */
    { register void *p __asm__("a0") = g_pc; register long s __asm__("a1") = 128 * 4;
      __asm__ volatile("msce32 acc0,(%0),%1" :: "r"(p), "r"(s) : "memory"); }
    __asm__ volatile("mrelease");
    return t1 - t0;
}

/* 失败步骤的位掩码。软失败（继续跑）与硬失败（当场停机）都记在这里，
 * 最后统一裁决 —— 否则后续阶段的状态写入会盖掉先前的失败痕迹。 */
static uint32_t g_failmask;

static void mark_fail(int step) {
    g_failmask |= (1u << step);
    mb_set(MB_FAILMASK, g_failmask);
    mb_set(MB_STATUS, ST_FAIL(step));
}

/* 硬失败：这一步不过，后面的测试没有意义，当场停机。 */
static void fail(int step) {
    mark_fail(step);
    P("\n*** FAIL at step "); U((unsigned long)step); P(" ***\n");
    for (;;) { }
}

void qwen3_main(void) {
    /* ---- 步骤 1：能执行到这里，说明复位放开、取指正常、DDR 可写 ---- */
    mb_set(MB_STATUS, ST_BOOT);

    /* ---- 步骤 2：串口 ---- */
    uart_init();
    P("\n\n=== S2C BRINGUP (" BOARD_NAME ") ===\n");
    P("step1 boot        OK   (mailbox @ "); X(BOARD_MBOX_ADDR); P(")\n");
    P("step2 uart        OK   (base "); X(BOARD_UART_BASE); P(")\n");
    mb_set(MB_STATUS, ST_UART);

    /* ---- 步骤 3：回读 host 预置的 magic，反证 PCIe 写 DDR 生效 ---- */
    uint32_t got = mb_get(MB_HOST_MAGIC);
    mb_set(MB_HOST_ECHO, got);
    P("step3 pcie magic  ");
    if (got == HOST_MAGIC) {
        P("OK   ("); X(got); P(")\n");
        mb_set(MB_STATUS, ST_PCIE);
    } else {
        /* 不直接停机：后面几步与 PCIe 无关，继续跑能一次收集到更多信息。
         * host 侧看 MB_HOST_ECHO 就知道实际读到的是什么。 */
        P("MISMATCH  期望 "); X(HOST_MAGIC); P("  实际 "); X(got); P("\n");
        P("      => host 是否在放开复位前往 "); X(MB_HOST_MAGIC); P(" 写过 magic？\n");
        mark_fail(3);
    }

    /* ---- 步骤 4：RVV ---- */
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    mb_set(MB_VLENB, (uint32_t)vlenb);
    P("step4 rvv vlenb   ");
    U(vlenb); P(" bytes (VLEN="); U(vlenb * 8); P(" bit)");
    if (vlenb != 128) { P("  ★ 期望 128\n"); fail(4); }
    P("   OK\n");
    mb_set(MB_STATUS, ST_RVV);

    /* ---- 步骤 5：AME 最小 GEMM ---- */
    unsigned long cap;
    __asm__ volatile("csrr %0, 0x803" : "=r"(cap));   /* 回读 mtilem，核对配置 */
    mb_set(MB_MTILE_CAP, (uint32_t)cap);

    uint32_t r = ame_selftest();
    mb_set(MB_AME_RESULT, r);
    P("step5 ame 1x1x32  "); X(r);
    if (r != 0x42000000u) { P("  ★ 期望 0x42000000 (=32.0f)\n"); fail(5); }
    P("  (=32.0f)   OK\n");
    mb_set(MB_STATUS, ST_AME);

    /* ---- 步骤 6：性能采样 ---- */
    P("step6 perf        采样 "); U(PERF_ITERS); P(" 次 128x128x32 ...\n");
    uint64_t cyc = perf_sample();
    mb_set(MB_CYCLES_LO, (uint32_t)cyc);
    mb_set(MB_CYCLES_HI, (uint32_t)(cyc >> 32));
    mb_set(MB_ITERS, PERF_ITERS);

    P("      总周期 "); U((unsigned long)cyc);
    P("   每次 "); U((unsigned long)(cyc / PERF_ITERS)); P(" 周期\n");
#if !BOARD_CPU_HZ
    /* QEMU 的 mcycle 跟着宿主机时间跑，与被模拟 CPU 的周期没有对应关系。
     * 不加这句警告，上面那个数会被当成真实性能读走。 */
    P("      ⚠ 本平台 mcycle 非真实周期计数，此数仅证明采样跑通，\n");
    P("        不可据此估算性能。真机上会给出可用的数值。\n");
#endif
#if BOARD_CPU_HZ
    /* 用实测周期数外推 decode 一个 token 的时间。
     * 145500 = 28 层 × 3840 + lm_head 的 1187×32，M=1 的 mfmacc 总数。 */
    unsigned long per = (unsigned long)(cyc / PERF_ITERS);
    unsigned long ms  = (unsigned long)((145500ULL * per * 1000ULL) / BOARD_CPU_HZ);
    P("      @ "); U(BOARD_CPU_HZ / 1000000); P(" MHz 外推：约 ");
    U(ms); P(" ms/token");
    if (ms) { P("  ("); U(1000 / (ms ? ms : 1)); P(" token/s)"); }
    P("\n");
#endif
    if (!g_failmask) mb_set(MB_STATUS, ST_PERF);

    /* ---- 裁决 ----
     * 必须看累计的 failmask，不能只看跑到了最后一步：
     * step3 是软失败（继续跑以便一次收齐信息），若在这里无条件报 ALL PASS，
     * 上板时会把"PCIe 没写进去"当成全部通过，是最坏的一种误导。 */
    if (g_failmask) {
        P("\n>>> FAIL —— 以下步骤未通过：");
        for (int i = 1; i <= 6; i++)
            if (g_failmask & (1u << i)) { P(" step"); U((unsigned long)i); }
        P("\n    failmask "); X(g_failmask);
        P("  (mailbox "); X(MB_FAILMASK); P(")\n");
        mb_set(MB_FAILMASK, g_failmask);
#if BOARD_HAS_POWEROFF
        *BOARD_PTR(uint32_t, BOARD_POWEROFF_ADDR) = BOARD_POWEROFF_FAIL;
#endif
        for (;;) { }
    }

    P("\n>>> ALL PASS —— 基础通路全通，可以进 L1（AME 向量包）\n");
    mb_set(MB_FAILMASK, 0);
    mb_set(MB_STATUS, ST_ALLPASS);
#if BOARD_HAS_POWEROFF
    *BOARD_PTR(uint32_t, BOARD_POWEROFF_ADDR) = BOARD_POWEROFF_PASS;
#endif
    for (;;) { }
}
