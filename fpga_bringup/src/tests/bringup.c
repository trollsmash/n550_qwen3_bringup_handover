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
/* [读] 内存带宽，单位 MB/s（整数）。host 不必换算周期数即可直接看。
 * ★ 偏移接在 MB_BUILD_ID(0x28) 之后。加新字段前先看下面这张表把偏移排完，
 *   两个字段落到同一地址时后写的会覆盖先写的，host 读到什么全凭调用顺序，
 *   而且不会有任何报错。 */
#define MB_BW_AME       MB(0x2C)
#define MB_BW_RVV       MB(0x30)
/* 下一个可用偏移：0x34（0x80 起是 host 侧的区域，别越过去） */
/* [读] 同步前直接读到的 AME 结果。与 MB_AME_RESULT 对照可一眼区分故障类型：
 *   RAW=哨兵 且 RESULT=32.0f  -> 缓存一致性问题，同步已生效
 *   RAW 与 RESULT 都是哨兵      -> msce32 没写进来，查 AME 或地址
 *   RESULT 是别的值             -> AME 算错了 */
#define MB_AME_RAW      MB(0x20)
/* [读] 结果缓冲的物理地址。host 可经 PCIe 直读该地址，
 * 绕开 CPU 的 L1D 看 DDR 里的真值，独立判断 AME 到底写没写。 */
#define MB_C_ADDR       MB(0x24)
#define MB_HOST_MAGIC   MB(0x80)   /* [写] host 上电前写 0x5A5AC3C3 */
#define MB_HOST_ECHO    MB(0x84)   /* [读] 程序同步后读回的值 */

/* [读] 镜像版本，0xMMDDhhmm。host 一读就知道板上跑的是哪一版。 */
#define MB_BUILD_ID     MB(0x28)



#define HOST_MAGIC      0x5A5AC3C3u

#define ST_BOOT         0xB0000010u
#define ST_UART         0xB0000020u
#define ST_PCIE         0xB0000030u
#define ST_RVV          0xB0000040u
#define ST_AME          0xB0000050u
#define ST_PERF         0xB0000060u
#define ST_ALLPASS      0xB00000AAu
#define ST_FAIL(n)      (0xB00000F0u | (n))

/* mailbox 是 CPU 与 host 共享的一块 DDR，host 侧经 PCIe 后门读写，
 * **完全绕过 CPU 的 cache**。所以每一次交接都要显式同步：
 *   写完 -> CLEAN，把值压到 DDR，host 才看得见
 *   读前 -> FLUSH，写回并失效本地拷贝，才拿得到 host 写进去的值
 *
 * 少了写侧同步，host 读到的永远是旧值；少了读侧同步，程序读到的可能是
 * 上电时 cache 里的随机内容。两种情况下 mailbox 这条"主观测通道"都是哑的，
 * 而它恰恰是串口没调通时唯一的观测手段。
 *
 * 读侧用 FLUSH 而不是 INVAL：mailbox 各字段挨得很近，MB_HOST_MAGIC(0x80)
 * 与 MB_HOST_ECHO(0x84) 就同处一条 64 字节 cache line，
 * 用 INVAL 会把刚写进 ECHO 的脏数据直接丢掉。 */
static void mb_set(unsigned long a, uint32_t v) {
    *BOARD_PTR(uint32_t, a) = v;
    BOARD_DCACHE_CLEAN((void *)a, sizeof(uint32_t));
    BOARD_FENCE();
}
static uint32_t mb_get(unsigned long a) {
    BOARD_DCACHE_FLUSH((void *)a, sizeof(uint32_t));
    BOARD_FENCE();
    return *BOARD_PTR(uint32_t, a);
}

/* ══════════════ 串口 ══════════════
 *
 * S2C 板上**由程序自己配波特率**（board.h 的 BOARD_UART_NEEDS_INIT），
 * 不指望 bootrom 已配好；QEMU virt 则开箱即用，那边这一项为 0。
 *
 * 上板后若 mailbox 走过 ST_UART 但屏幕一个字都没有，先怀疑寄存器位宽或
 * 间距（BOARD_UART_32BIT / BOARD_UART_REG_SHIFT）；若有输出但是乱码，
 * 才是分频不对，改 board.h 的 BOARD_UART_CLK_HZ。 */
/* 下面是**寄存器索引**，不是字节偏移。
 * 实际地址 = BOARD_UART_BASE + (索引 << BOARD_UART_REG_SHIFT)。 */
#define UART_THR   0
#define UART_IER   1
#define UART_FCR   2
#define UART_LCR   3
#define UART_LSR   5
#define UART_DLL   0      /* DLAB=1 时与 THR 同址 */
#define UART_DLM   1
#define LSR_THRE   0x20

/* 访问宽度由 board.h 决定：本核的 16550 是 32 位寄存器，
 * 用字节访问读 LSR 会得到未定义结果。 */
static inline unsigned long uart_addr(int idx) {
    return BOARD_UART_BASE + ((unsigned long)idx << BOARD_UART_REG_SHIFT);
}
static inline void uart_wr(int idx, uint32_t v) {
#if BOARD_UART_32BIT
    *BOARD_PTR(uint32_t, uart_addr(idx)) = v;
#else
    *BOARD_PTR(uint8_t, uart_addr(idx)) = (uint8_t)v;
#endif
}
static inline uint32_t uart_rd(int idx) {
#if BOARD_UART_32BIT
    return *BOARD_PTR(uint32_t, uart_addr(idx));
#else
    return *BOARD_PTR(uint8_t, uart_addr(idx));
#endif
}

static void uart_init(void) {
#if BOARD_UART_NEEDS_INIT
    /* 分频因子由 board.h 从 BOARD_UART_CLK_HZ 算出，改时钟只改那一处。
     * 配错的现象是满屏乱码，而不是没有输出。 */
    uart_wr(UART_IER, 0x00);                          /* 关中断 */
    uart_wr(UART_LCR, 0x80);                          /* DLAB=1，露出分频寄存器 */
    uart_wr(UART_DLL, BOARD_UART_DIVISOR & 0xFF);
    uart_wr(UART_DLM, (BOARD_UART_DIVISOR >> 8) & 0xFF);
    uart_wr(UART_LCR, 0x03);                          /* DLAB=0, 8N1 */
#if BOARD_UART_HAS_DLF
    /* 小数分频：DLF 不受 DLAB 控制，放在恢复 DLAB=0 之后写。
     * 40 MHz 下只用整数分频误差 +3.34%（超容限），补上 DLF 后为 +0.06%。 */
    uart_wr(BOARD_UART_DLF_IDX, BOARD_UART_DLF_VAL);
#endif
    uart_wr(UART_FCR, 0x07);                          /* 开 FIFO 并清空 */
#endif
}

static void putc_(char c) {
    /* 有上限的轮询：真机上等 THRE 是对的，但 LSR 读不到预期值时不能死等，
     * 否则整个程序静默挂死 —— 而这恰恰是串口没配好时的表现。
     * 宁可字符发飞，也要让程序继续往下走，把 mailbox 填完。 */
    for (int i = 0; i < 100000; i++)
        if (uart_rd(UART_LSR) & LSR_THRE) break;
    uart_wr(UART_THR, (uint8_t)c);
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

/* 把编译期假定的板级参数原样打出来。
 *
 * 这些值全部来自 board.h，是"我们以为的硬件"。一旦它与实际硬件不符，
 * 后面所有诊断的推理都会跟着错 —— 例如 cache line 填错，
 * "两个字段不在同一行、不会互相污染"这类判断就整个不成立。
 * 与其在远端反复推测，不如把前提打出来让板上的人一眼核对。 */
static void print_board_config(void) {
    P("-- 本镜像编译时假定的板级配置（请与实际硬件核对）--\n");
    P("  DRAM      "); X(BOARD_DRAM_BASE); P(" .. "); X(BOARD_DRAM_END); P("\n");
    P("  mailbox   "); X(BOARD_MBOX_ADDR); P("\n");
    P("  权重      "); X(BOARD_WEIGHTS_ADDR);
    P("   分词器 "); X(BOARD_TOKENIZER_ADDR); P("\n");
    P("  UART      "); X(BOARD_UART_BASE);
    P("  "); U(BOARD_UART_32BIT ? 32 : 8); P("bit  shift=");
    U(BOARD_UART_REG_SHIFT); P("  div="); U(BOARD_UART_DIVISOR);
#if BOARD_UART_HAS_DLF
    P("+"); U(BOARD_UART_DLF_VAL); P("/"); U(1u << BOARD_UART_DLF_BITS);
#endif
    P("\n");
    P("  CPU       "); U(BOARD_CPU_HZ); P(" Hz\n");
#ifdef BOARD_CACHE_LINE
    P("  cacheline "); U(BOARD_CACHE_LINE); P(" B");
#ifdef BOARD_L1D_BYTES
    P("   L1D "); U(BOARD_L1D_BYTES); P(" B");
#endif
    P("\n");
#endif
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

static uint32_t g_ame_raw;      /* 同步前读到的值，仅用于故障定位 */

static uint32_t ame_selftest(void) {
    for (int i = 0; i < 32; i++) { g_a[i] = 0x3F80; g_b[i] = 0x3F80; }
    g_c[0] = -1.0f;             /* 哨兵：msce32 若没写，读回来就是它 */

    /* ---- 进入 AME 前的缓存同步 ----
     * AME 的访存绕过 L1D 直连 DDR，而上面几行是 CPU 写的、还留在 L1D 里。
     * 不写回，AME 从 DDR 读到的就是旧内容；
     * g_c 的脏行也必须先清掉，否则它稍后被写回时会覆盖 AME 的结果。 */
    BOARD_DCACHE_CLEAN(g_a, 32 * sizeof g_a[0]);
    BOARD_DCACHE_CLEAN(g_b, 32 * sizeof g_b[0]);
    BOARD_DCACHE_FLUSH(g_c, sizeof g_c[0]);
    BOARD_FENCE();

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

    /* ---- 退出 AME 后的缓存同步 ----
     * fence 让 AME 的写落到 DDR，再失效 L1D 里可能残留的旧拷贝。 */
    BOARD_FENCE();
    { union { float f; uint32_t u; } r; r.f = g_c[0]; g_ame_raw = r.u; }
    BOARD_DCACHE_INVAL(g_c, sizeof g_c[0]);
    BOARD_FENCE();

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
 *   每次 X 周期 => 时间 = 145500 × X / 40MHz
 *     X=100  => 0.36 秒/token
 *     X=1000 => 3.6 秒/token
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
    /* 同样要写回 DDR：本函数虽只计时、不校验结果，但要让 AME 读到确定的数据，
     * 计时才有可比性 —— 全零输入在某些实现上可能走捷径。
     * 放在计时起点之前，不计入采样周期。 */
    BOARD_DCACHE_CLEAN(g_pa, 128 * 32 * sizeof g_pa[0]);
    BOARD_DCACHE_CLEAN(g_pb, 128 * 32 * sizeof g_pb[0]);
    BOARD_FENCE();
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

/* ══════════════ 内存带宽 ══════════════
 *
 * step6 报的是"每次 tile 装载多少周期"，那是个相对量；带宽是绝对量，
 * 能直接和 DDR 理论峰值比，一眼看出访存效率是 5% 还是 80%。
 *
 * 测两条路径，因为它们在本核上是分开的：
 *   AME  绕过 L1D 直连 DDR —— 全模型 99% 的访存走这条
 *   RVV  经过 L1D          —— 激活值的搬运走这条
 * 两者差得远时就知道该查哪一侧，不必两边都翻。
 *
 * 数据量取 4 MiB：远超 32 KB 的 L1D，RVV 那条不会退化成测 cache 带宽。
 * 缓冲放在 DRAM 16 MiB 处 —— 程序自身连栈带 BSS 不到 300 KB，
 * mailbox 在 127 MiB 处，两边都碰不着。内容是什么无所谓，只测读取速度。 */
#define BW_BUF_ADDR  (BOARD_DRAM_BASE + 0x01000000)
#define BW_BYTES     (4u * 1024 * 1024)

static uint64_t bw_ame_read(void) {
    set_tile(128, 128, 32);
    unsigned long p = BW_BUF_ADDR, end = BW_BUF_ADDR + BW_BYTES;
    uint64_t t0 = rd_mcycle();
    while (p + 8192 <= end) {
        /* 一条 mlae16 读 128 行 x 64 B = 8 KiB */
        register const void *q __asm__("a0") = (const void *)(uintptr_t)p;
        register long st __asm__("a1") = 64;
        __asm__ volatile("mlae16 tr0,(%0),%1" :: "r"(q), "r"(st) : "memory");
        p += 8192;
    }
    __asm__ volatile("mrelease");
    return rd_mcycle() - t0;
}

static uint64_t bw_rvv_read(void) {
    unsigned long p = BW_BUF_ADDR, end = BW_BUF_ADDR + BW_BYTES;
    uint64_t t0 = rd_mcycle();
    /* e32 m8：VLEN=1024 bit=128 B，八个寄存器一次搬 1 KiB */
    __asm__ volatile("vsetvli t0, zero, e32, m8, ta, ma" ::: "t0");
    while (p + 1024 <= end) {
        __asm__ volatile("vle32.v v0, (%0)" :: "r"(p) : "memory");
        p += 1024;
    }
    return rd_mcycle() - t0;
}

/* 打印 X.XX GB/s。裸机没有浮点输出，先放大 100 倍再拆整数与小数两段。
 * 中间量必须用 64 位：4 MiB x 40 MHz x 100 已到 1.7e16，32 位差得远。 */
static void print_gbps(const char *tag, uint64_t cycles) {
    P(tag);
    if (cycles == 0) { P("  n/a\n"); return; }
#if BOARD_CPU_HZ
    {
        uint64_t centi = (uint64_t)BW_BYTES * (uint64_t)BOARD_CPU_HZ * 100u
                         / cycles / 1000000000u;
        P("  "); U((unsigned long)(centi / 100)); P(".");
        if (centi % 100 < 10) P("0");
        U((unsigned long)(centi % 100)); P(" GB/s");
        P("   ("); U((unsigned long)cycles); P(" 周期)\n");
    }
#else
    P("  "); U((unsigned long)cycles);
    P(" 周期（本平台 mcycle 非真实周期，无法换算带宽）\n");
#endif
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
    P("build "); X(BOARD_BUILD_ID); P("   (0xMMDDhhmm)\n");
    mb_set(MB_BUILD_ID, BOARD_BUILD_ID);
    print_board_config();
    P("step1 boot        OK   (mailbox @ "); X(BOARD_MBOX_ADDR); P(")\n");
    P("step2 uart        OK   (base "); X(BOARD_UART_BASE); P(")\n");
    mb_set(MB_STATUS, ST_UART);

#ifdef TRAP_TEST
    /* 故意触发一次非法指令，验证 trap handler 本身可用。
     * 默认不编入；上板前想确认"真出事时能看到现场"，加 -DTRAP_TEST 编一版即可。
     * 0xFFFFFFFF 不是任何合法编码，低两位为 11 会被当作 32 位指令去译码。 */
    P("[TRAP_TEST] 下面故意执行非法指令，应打印完整 trap 现场\n");
    __asm__ volatile(".word 0xffffffff");
#endif

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
    mb_set(MB_AME_RAW, g_ame_raw);
    mb_set(MB_C_ADDR, (uint32_t)(uintptr_t)g_c);
    P("step5 ame 1x1x32  "); X(r);
    if (r != 0x42000000u) {
        P("  * 期望 0x42000000 (=32.0f)\n");
        P("      同步前读到 ");
        X(g_ame_raw);
        P("\n      结果缓冲 @ ");
        X((unsigned long)(uintptr_t)g_c);
        P("\n      两者都是 0xbf800000 则说明 msce32 没写进来；\n");
        P("      可让 host 经 PCIe 直读上面的地址，绕开 L1D 看 DDR 真值\n");
        fail(5);
    }
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

    /* ---- 步骤 7：内存带宽 ----
     * 与 step6 互补：那边是"每次 tile 多少周期"，这边是绝对带宽，
     * 可以直接和 DDR 理论峰值比，看出访存效率的量级。 */
    P("step7 带宽        读 "); U(BW_BYTES >> 20); P(" MiB x2 ...\n");
    {
        uint64_t ca = bw_ame_read();
        uint64_t cr = bw_rvv_read();
        print_gbps("      AME 装载(绕 L1D)", ca);
        print_gbps("      RVV 读取(经 L1D)", cr);
#if BOARD_CPU_HZ
        mb_set(MB_BW_AME, (uint32_t)((uint64_t)BW_BYTES * BOARD_CPU_HZ
                                     / (ca ? ca : 1) / 1000000u));
        mb_set(MB_BW_RVV, (uint32_t)((uint64_t)BW_BYTES * BOARD_CPU_HZ
                                     / (cr ? cr : 1) / 1000000u));
#endif
    }

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
