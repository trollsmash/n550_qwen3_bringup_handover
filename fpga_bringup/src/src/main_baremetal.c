/* FPGA bare-metal 入口 —— 取代 PC 侧的 main.c / main_qemu.c。
 *
 * 与另两个平台层的区别只在 IO：
 *   权重  ── 不 open/mmap，而是由外部预先放进 DDR 的固定地址
 *            QEMU:  -device loader,file=w.bin,addr=0x84000000
 *            FPGA:  host 经 PCIe 后门写入同一地址
 *   输出  ── UART 寄存器直写，不用 printf
 *   结束  ── QEMU 用 sifive_test 关机；FPGA 上换成死循环或回填 mailbox
 *
 * qwen3.c / kernels_*.c / ops_*.c 原样复用，一行不改。
 *
 * 编译运行见 tools/build_riscv.sh baremetal
 */
#include <stdint.h>

#include "bsp/board.h"
#include "kernels.h"
#include "ops.h"
#include "qwen3.h"
#include "tokenizer.h"

/* ==================== 内存映射 ====================
 * QEMU virt 的 DRAM 从 0x80000000 起。布局：
 *   0x80000000  程序（.text/.data/.bss/.stack，约 67 MiB，其中 scratch 56.6 MiB）
 *   0x88000000  权重 1.11 GiB，末尾约 0xCF120000
 *   0xD0000000  分词器 3.3 MiB
 * 权重之后仍在 DRAM 内，天然满足 kernels.h 的 QWEN3_WEIGHT_TAIL_PAD 契约
 * —— 只要 -m 给够（2G 以上）。FPGA 上同理：别把权重紧贴 DDR 末端摆放。
 *
 * ★ 128 MiB 的程序区不是随手取的整数：start.S 会把整个 BSS 清零，
 *   若 BSS 末端越过 WEIGHTS_ADDR，启动瞬间就把权重头部擦成 0，
 *   症状是"magic 不对"，看起来却像外部加载没生效。
 *   下面的 check_layout() 就是为此设的护栏，改 scratch 大小时别删。 */
/* ★ 具体数值全部来自 bsp/board.h —— 换板子只改那一个文件。
 * 这里曾经与 start.S 的 trap handler 各写了一份 UART 基址，
 * 上板时改一处漏一处是必然的。 */
#define DRAM_BASE      ((unsigned long)BOARD_DRAM_BASE)
#define WEIGHTS_ADDR   ((unsigned long)BOARD_WEIGHTS_ADDR)
#define TOKENIZER_ADDR ((unsigned long)BOARD_TOKENIZER_ADDR)

#define UART_BASE     ((unsigned long)BOARD_UART_BASE)
/* 寄存器**索引**，实际地址 = BASE + (索引 << BOARD_UART_REG_SHIFT) */
#define UART_THR      0
#define UART_DLL      0      /* 与 THR 同址，靠 LCR.DLAB 切换 */
#define UART_IER      1
#define UART_DLM      1      /* 与 IER 同址，靠 LCR.DLAB 切换 */
#define UART_FCR      2
#define UART_LCR      3
#define UART_LSR      5
#define UART_LSR_THRE 0x20

/* ==================== UART ====================
 * 访问宽度与间距由 board.h 给出：本核的 16550 为 32 位寄存器，
 * 用字节访问读 LSR 得到的是未定义结果。 */
static inline unsigned long uart_addr(int idx) {
    return UART_BASE + ((unsigned long)idx << BOARD_UART_REG_SHIFT);
}
/* 寄存器索引 -> 一次写。只有 uart_init 用得到，与 putc_ 分开是为了让
 * putc_ 保持在最热路径上不做多余判断。 */
static inline void uart_wr(int idx, uint32_t v) {
#if BOARD_UART_32BIT
    *BOARD_PTR(uint32_t, uart_addr(idx)) = v;
#else
    *BOARD_PTR(uint8_t, uart_addr(idx)) = (uint8_t)v;
#endif
}

/* 配波特率。本板不能指望 bootrom 已配好，程序必须自己来。
 *
 * 分频参数全部来自 board.h，由 BOARD_UART_CLK_HZ（UART 输入时钟，10 MHz）
 * 算出 —— 它与 BOARD_CPU_HZ（40 MHz）是两个不同的值，别拿错。
 * 拿错时钟的现象是满屏乱码，而不是没有输出。 */
static void uart_init(void) {
#if BOARD_UART_NEEDS_INIT
    uart_wr(UART_IER, 0x00);                          /* 关中断 */
    uart_wr(UART_LCR, 0x80);                          /* DLAB=1，露出分频寄存器 */
    uart_wr(UART_DLL, BOARD_UART_DIVISOR & 0xFF);
    uart_wr(UART_DLM, (BOARD_UART_DIVISOR >> 8) & 0xFF);
    uart_wr(UART_LCR, 0x03);                          /* DLAB=0, 8N1 */
#if BOARD_UART_HAS_DLF
    /* DLF 小数分频，不受 DLAB 控制，故放在恢复 DLAB=0 之后写 */
    uart_wr(BOARD_UART_DLF_IDX, BOARD_UART_DLF_VAL);
#endif
    uart_wr(UART_FCR, 0x07);                          /* 开 FIFO 并清空 */
#endif
}

static void putc_(char c) {
    /* 有上限的轮询：真机上等 THRE 是对的，但若 LSR 读不到预期值也不能死等，
     * 否则整个程序静默挂死、毫无线索。 */
    for (int i = 0; i < 100000; i++) {
#if BOARD_UART_32BIT
        if (*BOARD_PTR(uint32_t, uart_addr(UART_LSR)) & UART_LSR_THRE) break;
#else
        if (*BOARD_PTR(uint8_t, uart_addr(UART_LSR)) & UART_LSR_THRE) break;
#endif
    }
#if BOARD_UART_32BIT
    *BOARD_PTR(uint32_t, uart_addr(UART_THR)) = (uint8_t)c;
#else
    *BOARD_PTR(uint8_t, uart_addr(UART_THR)) = (uint8_t)c;
#endif
}
static void P(const char *s) { while (*s) { if (*s == '\n') putc_('\r'); putc_(*s++); } }
static void U(unsigned long v) {
    char b[24]; int i = 0;
    if (!v) { putc_('0'); return; }
    while (v) { b[i++] = (char)('0' + v % 10); v /= 10; }
    while (i) putc_(b[--i]);
}
static void I(long v) { if (v < 0) { putc_('-'); v = -v; } U((unsigned long)v); }
static void X(unsigned long v) {
    P("0x");
    for (int s = 60; s >= 0; s -= 4) putc_("0123456789abcdef"[(v >> s) & 0xf]);
}

/* ==================== 串口输入 ====================
 * 之前只有发送方向。做交互 demo 需要收方向：LSR 的 bit0 是 Data Ready，
 * RBR 与 THR 同址（写是 THR，读是 RBR）。 */
#define UART_RBR    0
#define UART_LSR_DR 0x01

static int uart_getc(void) {
    for (;;) {
#if BOARD_UART_32BIT
        if (*BOARD_PTR(uint32_t, uart_addr(UART_LSR)) & UART_LSR_DR) break;
#else
        if (*BOARD_PTR(uint8_t, uart_addr(UART_LSR)) & UART_LSR_DR) break;
#endif
    }
#if BOARD_UART_32BIT
    return (int)(*BOARD_PTR(uint32_t, uart_addr(UART_RBR)) & 0xFF);
#else
    return (int)(*BOARD_PTR(uint8_t, uart_addr(UART_RBR)) & 0xFF);
#endif
}

/* 只看标志、不取字符、不阻塞。用于在生成过程中探测有没有人按键。
 * 每个 token 才查一次，开销可以忽略。 */
static int uart_haschar(void) {
#if BOARD_UART_32BIT
    return (*BOARD_PTR(uint32_t, uart_addr(UART_LSR)) & UART_LSR_DR) != 0;
#else
    return (*BOARD_PTR(uint8_t, uart_addr(UART_LSR)) & UART_LSR_DR) != 0;
#endif
}

/* 读一行，带回显与退格。返回字节数（不含结尾 NUL）。
 *
 * 中文经终端发来的是 UTF-8 多字节序列。回显逐字节发回去没问题 ——
 * 终端自己会把字节流重新组装成字符。但**退格必须整字删除**：
 * 只退一个字节会在缓冲里留下半个字符，那对 tokenizer 是非法序列。
 * UTF-8 的续字节高两位恒为 10，据此往回退到字符边界。 */
static int uart_readline(char *buf, int cap) {
    int n = 0;
    for (;;) {
        int c = uart_getc();
        if (c == '\r' || c == '\n') { P("\n"); break; }
        if (c == 8 || c == 127) {                   /* BS / DEL */
            if (n > 0) {
                int nb = 0;
                do { n--; nb++; }
                while (n > 0 && ((unsigned char)buf[n] & 0xC0) == 0x80);
                /* 中文在终端占两列，ASCII 占一列，擦除宽度得跟着变 */
                P(nb > 1 ? "\b\b  \b\b" : "\b \b");
            }
            continue;
        }
        if (c < 0x20) continue;                     /* 其余控制字符丢弃 */
        if (n + 1 < cap) { buf[n++] = (char)c; putc_((char)c); }
    }
    buf[n] = 0;
    return n;
}

static uint64_t rd_mcycle(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

/* ==================== 模型 ==================== */
static uint8_t g_scratch[QWEN3_SCRATCH_BYTES];

/* golden_meta.json 里 raw 模式的 prompt token（"你好，请介绍一下你自己"）。
 * 回归模式用它，因为有黄金数据可比对。 */
static const int g_prompt[] = { 108386, 37945, 109432, 107828 };
#define N_PROMPT ((int)(sizeof g_prompt / sizeof g_prompt[0]))
/* QEMU 上每次前向约 40~90 秒（模拟开销，非硬件性能）。
 * 真机上可以调大 —— 那里 AME 是并行阵列，不是这个量级。 */
#define MAX_GEN  5

/* 对话模式生成几个 token。默认 1：要证明的是端到端形态跑得通，
 * 而每多一个 token 就要把 1.1 GB 权重再完整遍历一遍。
 * 真机上调大，编译时 -DCHAT_MAX_GEN=N 即可。 */
#ifndef CHAT_MAX_GEN
#define CHAT_MAX_GEN 1
#endif

/* ==================== 对话模式 ====================
 * bare-metal 没有命令行，问题硬编码在此。真机 demo 可改成从 UART 读一行。
 * chat template 与另两个平台层一致，由黄金数据反推得到；
 * 末尾预置空的 <think></think> 用于关闭思考模式。 */
#define DEMO_PROMPT "你好，请介绍一下你自己"
#define CHAT_PREFIX "<|im_start|>user\n"
#define CHAT_MIDDLE "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

static uint32_t g_tok_htab[1u << 19];      /* merges 哈希表 */
static char     g_chat_buf[1024];
static int      g_chat_ids[QWEN3_TOK_MAX];

static unsigned long cat(char *dst, unsigned long at, const char *s) {
    while (*s) dst[at++] = *s++;
    return at;
}
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* 权重文件大小：从 QW3M header 的字段推出总长，不必外部告知。
 * 张量区按 QW3M_ALIGN 对齐依次排列，末尾无填充，故总长即最后一个张量的终点。
 * 这里直接用导出时的实际值做交叉校验即可 —— qwen3_init 内部会核对 header
 * 与编译期常量是否一致，用错文件会立刻报错而不是算出垃圾。 */
/* 构建脚本按权重文件的实际大小注入 -DWEIGHTS_SIZE。
 * 不同布局的文件大小不同（tile-major 因 4 KB 对齐会多出约 326 KB），
 * 写死一个数就会在换布局时触发 QWEN3_ERR_SIZE。下面只是兜底默认值。 */
#ifndef WEIGHTS_SIZE
#define WEIGHTS_SIZE 1192100096UL
#endif

/* 覆盖 qwen3.c 里的弱符号，把 28 层的推进过程吐到串口。
 *
 * 裸机上这是唯一的观测手段。没有它，"卡在第 3 层"和"正常跑到第 27 层"
 * 在外面看来一模一样，都是串口没动静。FPGA 上只会更需要它。
 *
 * 输出形如 [....|....|....|....|....|...] ，每 5 层一个分隔符便于数。 */
/* ---- 与 host 的 mailbox。PCIe 后门可回读 DDR，所以即使串口没接好，
 *      host 也能看到进度和结果。真机 bring-up 优先走这条路。 ---- */
/* 必须 CLEAN 之后 host 才看得见：mailbox 在普通 DDR 上，CPU 的写会停在 L1D，
 * 而 host 经 PCIe 后门读 DDR 完全绕过 cache。少了这一步，上面那句
 * "真机 bring-up 优先走这条路"就不成立 —— host 读到的永远是旧值。 */
static void mbox_set(unsigned long addr, uint32_t v) {
    *BOARD_PTR(uint32_t, addr) = v;
    BOARD_DCACHE_CLEAN((void *)addr, sizeof(uint32_t));
    BOARD_FENCE();
}

/* mailbox 写文本：host 经 PCIe 后门读 DDR 就能拿到中文答案，
 * 完全不依赖串口。L0 的教训就是别把观测全压在串口上 ——
 * 波特率、线序、终端编码任何一环出问题，串口就是一片死寂，
 * 而 mailbox 走的是 host 本来就在用的后门通路。 */
static unsigned long g_text_used;

static void mbox_text_reset(void) {
    g_text_used = 0;
    *BOARD_PTR(uint8_t, BOARD_MBOX_TEXT) = 0;
    BOARD_DCACHE_CLEAN((void *)(uintptr_t)BOARD_MBOX_TEXT, 1);
    BOARD_FENCE();
}

/* 追加一段 UTF-8 并保持 NUL 结尾。留 1 字节给结束符，超长就丢弃后续 —— 
 * 宁可截断也不能越界写进权重区。 */
static void mbox_text_append(const uint8_t *p, size_t n) {
    unsigned long cap = BOARD_MBOX_TEXT_BYTES - 1;
    for (size_t i = 0; i < n && g_text_used < cap; i++)
        *BOARD_PTR(uint8_t, BOARD_MBOX_TEXT + g_text_used++) = p[i];
    *BOARD_PTR(uint8_t, BOARD_MBOX_TEXT + g_text_used) = 0;
    BOARD_DCACHE_CLEAN((void *)(uintptr_t)BOARD_MBOX_TEXT, g_text_used + 1);
    BOARD_FENCE();
}

void qwen3_on_layer(int layer, int n_layers) {
    if (layer == 0) putc_('[');
    putc_('.');
    if (layer % 5 == 4 && layer != n_layers - 1) putc_('|');
    if (layer == n_layers - 1) putc_(']');
    mbox_set(BOARD_MBOX_LAYER, (uint32_t)layer);
}

/* 停机。QEMU 上写 sifive_test 让进程退出并带上退出码；
 * 真机没有这个外设，只能停在死循环里等 host 通过 PCIe 读结果。
 * 两种情况都先把状态写进 mailbox —— 那是真机上唯一可靠的出口。 */
static void halt(int ok) {
    mbox_set(BOARD_MBOX_STATUS, ok ? BOARD_ST_DONE : BOARD_ST_TRAP);
#if BOARD_HAS_POWEROFF
    *BOARD_PTR(uint32_t, BOARD_POWEROFF_ADDR) =
        ok ? BOARD_POWEROFF_PASS : BOARD_POWEROFF_FAIL;
#endif
    for (;;) { }
}

static void die(const char *msg) {
    P("\nFATAL: "); P(msg); P("\n");
    halt(0);
}

/* 链接脚本导出的镜像末端（含 .bss 与 .stack）。 */
extern char _end[];

/* 护栏：程序镜像不得越过权重区。
 * 越过时 start.S 清 BSS 会把权重头部擦零，表现为"magic 不对"，
 * 极易误判成外部加载失败。这里把两个地址直接打出来，一眼定性。 */
static void check_layout(void) {
    unsigned long img_end = (unsigned long)_end;
    P("镜像 "); X(DRAM_BASE); P(" .. "); X(img_end);
    P("  ("); U((img_end - DRAM_BASE) >> 20); P(" MiB)\n");
    if (img_end > WEIGHTS_ADDR) {
        P("镜像末端越过 WEIGHTS_ADDR="); X(WEIGHTS_ADDR);
        P("\n  → 请调大 WEIGHTS_ADDR 并同步 build_riscv.sh 的 -device loader\n");
        die("内存布局冲突：BSS 清零会擦掉权重");
    }
}

/* ==================== 交互式对话 ====================
 * 演示形态：一个串口终端，输入中文回车，答案逐字冒出来。
 * 不需要 host 侧程序，也不用 PCIe 后门参与 —— 现场只有一根串口线。
 *
 * 单轮对话：每次都从 pos0=0 重新 prefill，KV 缓存被新一轮覆盖。
 * 多轮上下文对 demo 不是必需，而每轮独立反倒让现场更可控。 */

/* 现场用输入法打中文有卡壳风险（终端编码、输入法切换），
 * 预置几条按数字键直接触发，手输作为加分项而非唯一路径。 */
static const char *g_presets[] = {
    "你好，请介绍一下你自己",
    "用一句话解释什么是 RISC-V",
    "写一首关于秋天的短诗",
};
#define N_PRESET ((int)(sizeof g_presets / sizeof g_presets[0]))

static char g_line[512];

static void chat_once(qwen3_t *m, qwen3_tok_t *tk, const char *prompt) {
    unsigned long n = 0;
    n = cat(g_chat_buf, n, CHAT_PREFIX);
    n = cat(g_chat_buf, n, prompt);
    n = cat(g_chat_buf, n, CHAT_MIDDLE);

    int n_in = qwen3_tok_encode(tk, g_chat_buf, n, g_chat_ids, QWEN3_TOK_MAX);
    if (n_in < 0 || n_in > QWEN3_MAX_BATCH) {
        P("  [prompt 编码失败或超过单批上限，换短一点的问题]\n");
        return;
    }

    mbox_set(BOARD_MBOX_STATUS, BOARD_ST_RUNNING);
    mbox_set(BOARD_MBOX_NTOKEN, 0);
    mbox_text_reset();

    uint64_t c0 = rd_mcycle();
    qwen3_forward_batch(m, g_chat_ids, n_in, 0);
    int next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);
    int got = 0, aborted = 0;
    for (int i = 0; i < CHAT_MAX_GEN; i++) {
        if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
        size_t plen;
        const uint8_t *piece = qwen3_tok_piece(tk, next, &plen);
        for (size_t q = 0; q < plen; q++) putc_((char)piece[q]);   /* 流式 */
        /* 同一份结果也写进 mailbox：串口若不通，host 后门照样读得到 */
        if (got < BOARD_MBOX_TOKENS_MAX)
            *BOARD_PTR(uint32_t, BOARD_MBOX_TOKENS + 4u * (unsigned)got) =
                (uint32_t)next;
        mbox_text_append(piece, plen);
        got++;
        mbox_set(BOARD_MBOX_NTOKEN, (uint32_t)got);
        /* 最后一个 token 不必再前向：那一整遍 1.1 GB 权重算出的 logits
         * 没有任何人会用到。 */
        if (i + 1 >= CHAT_MAX_GEN) break;
        /* 按任意键中止。检查放在前向**之前** —— 放在之后的话，按键那一刻
         * 还得再等一整轮 1.1 GB 的遍历才停得下来。
         * 不挑特定键是有意的：ESC 与方向键的转义序列同头、Ctrl-C 是否发到
         * 串口取决于终端配置，而现场那台机器什么设置往往到了才知道。 */
        if (uart_haschar()) {
            /* 排空缓冲：现场可能连按几下，残留字符会被下一次 readline
             * 当成输入的开头。 */
            while (uart_haschar()) (void)uart_getc();
            aborted = 1;
            P("  [已中止]");
            break;
        }
        qwen3_forward(m, next, n_in + i);
        next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);
    }
    uint64_t c1 = rd_mcycle();
    mbox_set(BOARD_MBOX_CYCLES_LO, (uint32_t)(c1 - c0));
    mbox_set(BOARD_MBOX_CYCLES_HI, (uint32_t)((c1 - c0) >> 32));
    mbox_set(BOARD_MBOX_STATUS, BOARD_ST_DONE);

    /* 现场把速度算出来。这个数比幻灯片上任何数字都有说服力，
     * 因为它是当场跑出来的。 */
    P("\n\n  [prompt "); I(n_in); P(" token，生成 "); I(got); P(" token");
#if BOARD_CPU_HZ
    {
        unsigned long ms = (unsigned long)((c1 - c0) / (BOARD_CPU_HZ / 1000));
        P("，"); U(ms); P(" ms");
        if (got > 0) { P("，"); U(ms / (unsigned long)got); P(" ms/token"); }
    }
#else
    (void)c0; (void)c1;
    P("（本平台 mcycle 非真实周期，未换算时间）");
#endif
    P(aborted ? "，已中止]\n" : "]\n");
}

/* 把 mailbox 的关键字段读回来打一遍。
 *
 * 这不是调试残留：上板时它能立刻回答一个关键问题 ——
 * "host 后门读到的，和串口上看到的，是同一份东西吗？"
 * 两者不一致就说明写侧的 cache 同步没生效，而那正是 L0 上踩过的坑。
 * 读之前先 FLUSH，确保拿到的是 DDR 里的值而非本地拷贝。 */
static void mbox_selfread(void) {
    BOARD_DCACHE_FLUSH((void *)(uintptr_t)BOARD_MBOX_ADDR,
                       0x400 + BOARD_MBOX_TEXT_BYTES);
    BOARD_FENCE();
    P("\nmailbox 回读（host 后门应看到同样内容）：\n");
    P("  STATUS  "); X(*BOARD_PTR(uint32_t, BOARD_MBOX_STATUS)); P("\n");
    P("  NTOKEN  "); U(*BOARD_PTR(uint32_t, BOARD_MBOX_NTOKEN)); P("\n");
    P("  TEXT    ");
    {
        const volatile uint8_t *t = BOARD_PTR(uint8_t, BOARD_MBOX_TEXT);
        for (unsigned i = 0; i < 256 && t[i]; i++) putc_((char)t[i]);
    }
    P("\n");
}

static void chat_repl(qwen3_t *m, qwen3_tok_t *tk) {
    /* 权重与分词器都就位了。host 轮到这个状态就知道可以开始交互 —— 
     * 1.11 GB 权重的加载与校验要花不少时间，没有这个标志只能靠猜。 */
    mbox_set(BOARD_MBOX_STATUS, BOARD_ST_READY);
    P("\n============ 交互模式 ============\n");
    P("直接输入中文问题后回车；输入数字选预置问题；q 退出\n");
    P("生成过程中按任意键可中止\n");
    for (int i = 0; i < N_PRESET; i++) {
        P("  "); I(i + 1); P(") "); P(g_presets[i]); P("\n");
    }
    for (;;) {
        P("\n> ");
        int len = uart_readline(g_line, (int)sizeof g_line);
        if (len == 0) continue;
        if (len == 1 && (g_line[0] == 'q' || g_line[0] == 'Q')) break;

        const char *prompt = g_line;
        if (len == 1 && g_line[0] >= '1' && g_line[0] < '1' + N_PRESET) {
            prompt = g_presets[g_line[0] - '1'];
            P(prompt); P("\n");
        }
        P("\n");
        chat_once(m, tk, prompt);
    }
    mbox_selfread();
    P("\n再见。\n");
}


void qwen3_main(void) {
    uart_init();                /* 必须早于第一次 putc_ */
    P("\n=== Qwen3-0.6B on RISC-V (bare-metal) ===\n");
    P("build "); X(BOARD_BUILD_ID); P("   (0xMMDDhhmm)\n");
    P("kernel="); P(qwen3_kernel_name());
    P("  ops="); P(qwen3_ops_name()); P("\n");

    check_layout();

    /* 权重应已由外部放入 DDR。先验 magic，避免在垃圾数据上跑几分钟才发现不对。 */
    const void *blob = (const void *)WEIGHTS_ADDR;
    const uint8_t *b = (const uint8_t *)blob;
    if (b[0] != 'Q' || b[1] != 'W' || b[2] != '3' || b[3] != 'M')
        die("权重 magic 不对 —— 是否忘了 -device loader？");
    P("权重 magic OK @ WEIGHTS_ADDR\n");

    static qwen3_t m;
    int rc = qwen3_init(&m, blob, WEIGHTS_SIZE, g_scratch, sizeof g_scratch);
    if (rc != QWEN3_OK) { P(qwen3_strerror(rc)); die("qwen3_init 失败"); }

    P("arena "); U(m.arena.used >> 20); P(" MiB   层数 "); U((unsigned long)m.n_layers);
    P("\n");

    /* ---------------- 分词器：同样来自 DDR 固定地址 ---------------- */
    const uint8_t *tb = (const uint8_t *)TOKENIZER_ADDR;
    static qwen3_tok_t tk;
    int have_tok = 0;
    if (tb[0] == 'Q' && tb[1] == 'W' && tb[2] == '3' && tb[3] == 'T') {
        uint32_t tsz = rd32(tb + 56);          /* header 里的总长度字段 */
        int trc = qwen3_tok_init(&tk, tb, tsz, g_tok_htab, sizeof g_tok_htab);
        if (trc == QWEN3_TOK_OK) {
            have_tok = 1;
            P("tokenizer OK  vocab "); U(tk.n_vocab);
            P("  merges "); U(tk.n_merges); P("\n");
        } else {
            P("tokenizer 初始化失败: "); P(qwen3_tok_strerror(trc)); P("\n");
        }
    } else {
        P("未检测到 tokenizer（缺 -device loader），转入回归模式\n");
    }

    /* ================= 对话模式：中文进、中文出 ================= */
    if (have_tok) {
        chat_repl(&m, &tk);
        halt(1);
    }

    /* ================= 回归模式：与黄金数据比对 ================= */
    P("\nprefill ");
    int gen[MAX_GEN + N_PROMPT];
    int n_gen = 0;

    /* ★ 一次批量前向：GEMM 的 M = N_PROMPT，AME tile 才被填上。 */
    qwen3_forward_batch(&m, g_prompt, N_PROMPT, 0);
    P("done\ndecode  ");

    int next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
    gen[n_gen++] = next;
    for (int step = 0; step < MAX_GEN - 1; step++) {
        qwen3_forward(&m, next, N_PROMPT + step);
        next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
        gen[n_gen++] = next;
        putc_('.');
        if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
    }
    P("\n\nTOKENS:");
    for (int i = 0; i < n_gen; i++) { putc_(' '); I(gen[i]); }
    P("\n");

    /* 与黄金数据的前 4 个 token 比对 —— 无需 host 参与即可自检。 */
    /* 黄金 token，来自 PyTorch 参考实现。
     *
     * 已按**真机的累加语义**验证过，不是只在 QEMU 上对过：
     * AMU RTL 每 16 个 K 元素就把累加器落回 FP32（ma_pkg.sv 的 ARRAY_K_FP=16），
     * 而 QEMU 的粒度是 32，两者数值并不逐位相同。
     * 用 src/kernels_hwsim.c（复刻 RTL 语义）在 PC 上跑完整模型，
     * 得到的 token 与下面完全一致 —— 说明这点数值差异不影响 argmax，
     * 上板后这四个值依然成立。 */
    static const int expect[] = { 3837, 101889, 106525, 56568 };
    int ok = 1;
    for (int i = 0; i < 4 && i < n_gen; i++) if (gen[i] != expect[i]) ok = 0;
    P(ok ? "\n>>> PASS: 前 4 个 token 与黄金数据一致\n"
         : "\n>>> FAIL: 与黄金数据不符\n");

    halt(ok);
}
