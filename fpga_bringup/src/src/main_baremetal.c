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

static void putc_raw(char c) {
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

/* ★ LF -> CRLF 的转换必须放在**最底层**。
 *
 * 串口终端在 raw 模式下收到 LF 只下移一行、不回行首，光标停在原列，
 * 后续文字便从那一列开始，整段输出呈阶梯状。
 *
 * 这里曾经只在 P() 里转换，而模型生成的 token 是逐字节直接送 putc_ 的 ——
 * 单段回答看不出问题，一旦模型吐出 markdown 列表（带 \n）就原形毕露。
 * 放到 putc_ 里，所有输出路径（P/U/X/流式 token/进度条）一次覆盖。 */
static void putc_(char c) {
    if (c == '\n') putc_raw('\r');
    putc_raw(c);
}
static void P(const char *s) { while (*s) putc_(*s++); }
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

/* ==================== 开机标识与时钟 ==================== */

static void D2(unsigned v) {          /* 两位、补零 —— 时间字段全靠它对齐 */
    putc_((char)(0x30 + (v / 10) % 10));
    putc_((char)(0x30 + v % 10));
}

/* 开机第一屏。logo 是 Unicode 立体块字符，串口通道本来就跑 UTF-8
 * （中文回答一直是这么出的），所以能显示。 */
static void print_banner(void) {
    P("\n");
    P("  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2557    \u2588\u2588\u2557\u2588\u2588\u2557\u2588\u2588\u2588\u2557   \u2588\u2588\u2557\n");
    P("  \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2551    \u2588\u2588\u2551\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2551\n");
    P("  \u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2551 \u2588\u2557 \u2588\u2588\u2551\u2588\u2588\u2551\u2588\u2588\u2554\u2588\u2588\u2557 \u2588\u2588\u2551\n");
    P("  \u2588\u2588\u2554\u2550\u2550\u255d  \u255a\u2550\u2550\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2551\u2588\u2588\u2588\u2557\u2588\u2588\u2551\u2588\u2588\u2551\u2588\u2588\u2551\u255a\u2588\u2588\u2557\u2588\u2588\u2551\n");
    P("  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u255a\u2588\u2588\u2588\u2554\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551\u2588\u2588\u2551 \u255a\u2588\u2588\u2588\u2588\u2551\n");
    P("  \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u255d\u255a\u2550\u2550\u255d \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u2550\u2550\u255d\n");
    P("   C  O  M  P  U  T  I  N  G\n\n");
    P("  ==================================================================\n");
    P("   Hardware    AI Core N550            developed by Zhufeng Institute\n");
    P("   Software    Qwen3-0.6B LLM port     developed by Zhufeng Institute\n");
    P("   Toolchain   RISC-V GCC + AME        developed by Zhufeng Institute\n");
    P("  ==================================================================\n");
    P("   Copyright (c) 2026 Beijing ESWIN Computing Technology Co., Ltd.\n");
    P("   All rights reserved.\n");
    P("  ==================================================================\n");
}

/* 板上没有 RTC：mcycle 只知道开机多久，不知道今天几号。
 * host 下载镜像时往 BOARD_MBOX_EPOCH 写一个 UTC 秒数，这里加上开机时长
 * 再加 8 小时即北京时间。**没授时也能跑** —— 读到 0 就显示开机时长；
 * 屏幕上跳出 1970 年，比老实说"开机 12 分钟"难看得多。 */
static void print_clock(void) {
    uint32_t base = *BOARD_PTR(volatile uint32_t, BOARD_MBOX_EPOCH);
#if BOARD_CPU_HZ
    uint64_t up_s = rd_mcycle() / (uint64_t)BOARD_CPU_HZ;
#else
    uint64_t up_s = 0;                /* QEMU 下 mcycle 不对应真实时间 */
#endif

    if (base == 0) {                  /* host 没授时 */
        P("[\u5f00\u673a ");
        D2((unsigned)(up_s / 3600));      putc_(0x3a);
        D2((unsigned)(up_s / 60 % 60));   putc_(0x3a);
        D2((unsigned)(up_s % 60));        P("]");
        return;
    }

    uint64_t t    = (uint64_t)base + up_s + 8u * 3600u;   /* -> 北京时间 */
    uint32_t days = (uint32_t)(t / 86400u);
    uint32_t sod  = (uint32_t)(t % 86400u);

    /* civil_from_days：1970 起的天数还原成年月日。以 3 月为岁首、
     * 400 年一循环，于是闰年只在最后一步修正一次，不必查表。 */
    uint32_t z   = days + 719468u;
    uint32_t era = z / 146097u, doe = z % 146097u;
    uint32_t yoe = (doe - doe/1460u + doe/36524u - doe/146096u) / 365u;
    uint32_t y   = yoe + era * 400u;
    uint32_t doy = doe - (365u*yoe + yoe/4u - yoe/100u);
    uint32_t mp  = (5u*doy + 2u) / 153u;
    uint32_t d   = doy - (153u*mp + 2u)/5u + 1u;
    uint32_t mo  = mp + (mp < 10u ? 3u : (uint32_t)-9);
    y += (mo <= 2u);

    P("[");  U((unsigned long)y); putc_(0x2d); D2(mo); putc_(0x2d); D2(d);
    putc_(0x20);
    D2(sod / 3600u); putc_(0x3a); D2(sod / 60u % 60u); putc_(0x3a);
    D2(sod % 60u);  P("]");
}

/* ==================== 模型 ==================== */
/* arena 指向 DDR 里的固定地址，不再占 BSS。容量由 board.h 给出，
 * qwen3_init 会用 QWEN3_SCRATCH_BYTES 校验够不够。
 *
 * CLP 下改指向 RVV 高速窗口：RVV/标量看到的是 0x1xxx 视图，AME 侧由
 * kernels_ame.c 换算成 0xFxxx —— 同一物理内存，两条路都绕 L1D。 */
#ifdef BOARD_CLP
static uint8_t *const g_scratch = BOARD_PTR(uint8_t, BOARD_CLP_ARENA_ADDR);
#define ARENA_LIMIT_BYTES  BOARD_CLP_ARENA_BYTES
#else
static uint8_t *const g_scratch = BOARD_PTR(uint8_t, BOARD_ARENA_ADDR);
#define ARENA_LIMIT_BYTES  BOARD_ARENA_BYTES
#endif

/* golden_meta.json 里 raw 模式的 prompt token（"你好，请介绍一下你自己"）。
 * 回归模式用它，因为有黄金数据可比对。 */
static const int g_prompt[] = { 108386, 37945, 109432, 107828 };
#define N_PROMPT ((int)(sizeof g_prompt / sizeof g_prompt[0]))
/* QEMU 上每次前向约 40~90 秒（模拟开销，非硬件性能）。
 * 真机上可以调大 —— 那里 AME 是并行阵列，不是这个量级。 */
#define MAX_GEN  5

/* 对话模式每次回答最多生成几个 token（是上限，遇 EOS 会提前停）。
 * 单位是 token 不是字：实测中文约 4.7 字节/token ≈ 1.5 个汉字。
 *
 * 默认 1 是为 QEMU 定的 —— 那里每多一个 token 就要把 1.1 GB 权重
 * 再完整遍历一遍，跑长回答没法用。真机由构建脚本注入
 * -DCHAT_MAX_GEN=N（见 tools/build_riscv.sh 的 CHAT_GEN）。
 *
 * 设多大不必算安全边界：下面的生成循环会按 KV cache 的剩余容量钳制。 */
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

/* 续轮时补上一轮 assistant 的结束标记。生成循环遇到 EOS 就 break 了，
 * 那个 <|im_end|> 从没进过 KV cache，不补的话模型看到的是一段
 * 没有收尾的 assistant 发言，接下来的 user 轮次会被理解成它的续写。 */
#define CHAT_TURN_SEP "<|im_end|>\n"

/* 一轮至少要留这么多生成位置，否则这轮没有意义，不如直接开新对话。 */
#define CHAT_MIN_GEN 8

/* ---------------- 多轮对话状态 ----------------
 *
 * KV cache 里躺着前几轮的 K/V，新一轮从 g_pos 往后写 —— 模型于是
 * "记得"之前说过什么。这比每轮重新 prefill 整段历史更省：
 * 省下的正好是一次完整的权重遍历。
 *
 * g_last_tok 的由来：生成循环里最后一个 token 只打印、不 forward
 *   （再 forward 一次要多遍历 1.1 GB 权重，而它的 logits 没人用）。
 * 于是它没进 KV cache。下一轮把它接在开头补回来，历史才是连续的 ——
 * 比补一次前向便宜得多。 */
static int g_pos      = 0;    /* KV cache 已占用的位置数 */
static int g_last_tok = -1;   /* 上一轮最后一个 token，尚未进 KV cache */

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

/* decode 阶段是否把逐层进度打到串口。
 *
 * 这个进度条是 L2 调试期的救命功能 —— 真机上"卡在第 3 层"和"跑到第 27 层"
 * 在串口外面看来都是没动静。但 demo 时它是纯噪声：一行 40 个字符，
 * 而一个 token 通常只有 1~3 个字符，答案被切得读不成句子。
 *
 * 折中：prefill 保留（那是最长的一次等待，要让人知道没死机），
 * decode 静音（答案连续吐出）。默认开，只有 chat_once 的生成循环
 * 临时关掉 —— 回归模式与其他路径的可观测性一点不减。 */
static int g_show_layer = 1;

void qwen3_on_layer(int layer, int n_layers) {
    /* mailbox 无条件更新：host 侧要看进度不该受串口开关影响，
     * 而这正是串口静音时唯一还能看到"跑到哪一层"的出口。 */
    mbox_set(BOARD_MBOX_LAYER, (uint32_t)layer);
    if (!g_show_layer) return;
    if (layer == 0) putc_('[');
    putc_('.');
    if (layer % 5 == 4 && layer != n_layers - 1) putc_('|');
    if (layer == n_layers - 1) putc_(']');
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
    P("  ("); U((img_end - DRAM_BASE) >> 20); P(" MiB");
    /* 余量随 QWEN3_MAX_SEQ 线性缩小（arena 按 N x 229 KB 吃 BSS），
     * 印出来才好判断还能不能再调大上下文。 */
    P("，距 mailbox 余 "); U((BOARD_MBOX_ADDR - img_end) >> 10); P(" KiB)\n");

    /* ★ 下界是 mailbox 而不是权重区 —— 它比 WEIGHTS_ADDR 靠前 1 MB。
     * 只查 WEIGHTS_ADDR 会漏掉 [MBOX, WEIGHTS) 这一段：镜像落在那里时
     * BSS 清零把 mailbox 擦成 0，host 侧永远读到 0，看起来像程序没跑起来，
     * 而串口那边一切正常 —— 这种错最难往内存布局上想。 */
    if (img_end > BOARD_MBOX_ADDR) {
        P("镜像末端越过 mailbox="); X(BOARD_MBOX_ADDR);
        P("\n  → 调小 QWEN3_MAX_SEQ，或把 arena 移出 BSS\n");
        die("内存布局冲突：BSS 清零会擦掉 mailbox");
    }
    if (img_end > WEIGHTS_ADDR) {
        P("镜像末端越过 WEIGHTS_ADDR="); X(WEIGHTS_ADDR);
        P("\n  → 请调大 WEIGHTS_ADDR 并同步 build_riscv.sh 的 -device loader\n");
        die("内存布局冲突：BSS 清零会擦掉权重");
    }

    /* arena 现在独立于镜像，单独校验：既要装得下，也不能越过 DRAM 顶。 */
#ifdef BOARD_CLP
    P("arena "); X(BOARD_CLP_ARENA_ADDR); P(" .. ");
    X(BOARD_CLP_ARENA_ADDR + QWEN3_SCRATCH_BYTES);
    P("  ("); U(QWEN3_SCRATCH_BYTES >> 20); P(" MiB / 窗口余 ");
    U(BOARD_CLP_ARENA_BYTES >> 20); P(" MiB)   [CLP: RVV 视图]\n");
    if (QWEN3_SCRATCH_BYTES > BOARD_CLP_ARENA_BYTES) {
        P("arena 需求超出 RVV 窗口\n");
        die("内存布局冲突：CLP 窗口容量不足");
    }
    /* 重映射自检。配错时的症状是"数值莫名不对"而非崩溃 —— 那种错极难查，
     * 所以在启动第一秒就挡住：往 RVV 视图写、从 AME 视图读回核对。 */
    {
        int rc = board_clp_selftest();
        /* 失败分支自己会把窗口地址打全，这里不再预告。 */
        if (rc == 0) {
            /* 成功时不出声：这行对操作者没有信息量，只占演示画面。
             * 失败才需要人看见 —— 见下面的分支。 */
        } else {
            P("CLP 窗口 "); X(BOARD_CLP_RVV_BASE); P(" -> ");
            X(BOARD_CLP_DDR_BASE); P("\n");
            P("自检失败 rc="); U((unsigned long)rc); P("\n");
            P("  rc=1: RVV 写入后 AME 视图读不到；rc=2: 反向读不到\n");
            P("  → 检查 NoC 是否已把该窗口重映射到 DDR\n");
            die("CLP 窗口重映射未生效");
        }
    }
#else
    P("arena "); X(BOARD_ARENA_ADDR); P(" .. ");
    X(BOARD_ARENA_ADDR + QWEN3_SCRATCH_BYTES);
    P("  ("); U(QWEN3_SCRATCH_BYTES >> 20); P(" MiB / 上限 ");
    U(BOARD_ARENA_BYTES >> 20); P(" MiB)\n");
    if (QWEN3_SCRATCH_BYTES > BOARD_ARENA_BYTES) {
        P("arena 需求超过 BOARD_ARENA_BYTES\n");
        die("内存布局冲突：arena 容量不足");
    }
    if ((unsigned long)BOARD_ARENA_ADDR < TOKENIZER_ADDR) {
        P("arena 起点在分词器之前\n");
        die("内存布局冲突：arena 会覆盖 tok.bin");
    }
#endif
}

/* ==================== 交互式对话 ====================
 * 演示形态：一个串口终端，输入中文回车，答案逐字冒出来。
 * 不需要 host 侧程序，也不用 PCIe 后门参与 —— 现场只有一根串口线。
 *
 * 多轮对话：KV cache 跨轮保留，新一轮从 g_pos 往后写，模型记得前文。
 * 这比每轮重新 prefill 整段历史更省 —— 省下的正好是一次权重遍历。
 * 上下文由 QWEN3_MAX_SEQ 封顶，满了自动重开，也可以输入 new 手动清空。 */

/* 现场用输入法打中文有卡壳风险（终端编码、输入法切换），
 * 预置几条按数字键直接触发，手输作为加分项而非唯一路径。 */
static const char *g_presets[] = {
    "你好，请介绍一下你自己",
    "用一句话解释什么是 RISC-V",
    "写一首关于秋天的短诗",
};
#define N_PRESET ((int)(sizeof g_presets / sizeof g_presets[0]))

static char g_line[512];

/* 拼出这一轮要喂进去的 token。cont=1 表示接着上一轮说，
 * 此时要补上一轮的收尾标记与那个没进 cache 的 token。
 * 返回 token 数，负数是错误码。 */
static int build_turn(qwen3_tok_t *tk, const char *prompt, int cont) {
    unsigned long n = 0;
    if (cont) n = cat(g_chat_buf, n, CHAT_TURN_SEP);
    n = cat(g_chat_buf, n, CHAT_PREFIX);
    n = cat(g_chat_buf, n, prompt);
    n = cat(g_chat_buf, n, CHAT_MIDDLE);

    int k = 0;
    if (cont && g_last_tok >= 0) g_chat_ids[k++] = g_last_tok;
    int e = qwen3_tok_encode(tk, g_chat_buf, n, g_chat_ids + k,
                             QWEN3_TOK_MAX - k);
    return (e < 0) ? e : (k + e);
}

static void chat_once(qwen3_t *m, qwen3_tok_t *tk, const char *prompt) {
    int cont = (g_pos > 0);
    int n_in = build_turn(tk, prompt, cont);
    if (n_in < 0 || n_in > QWEN3_MAX_BATCH) {
        P("  [prompt 编码失败或超过单批上限，换短一点的问题]\n");
        return;
    }

    /* 装不下就重开一轮。宁可丢掉历史，也不能让位置越过 MAX_SEQ ——
     * 那会静默写进下一层的 KV 区域，输出变胡话却看不出是内存问题。 */
    if (cont && g_pos + n_in + CHAT_MIN_GEN > QWEN3_MAX_SEQ) {
        P("  [上下文已满 "); I(g_pos); P("/"); I(QWEN3_MAX_SEQ);
        P("，自动开始新对话]\n");
        g_pos = 0; g_last_tok = -1;
        n_in = build_turn(tk, prompt, 0);
        if (n_in < 0 || n_in > QWEN3_MAX_BATCH) {
            P("  [prompt 编码失败或超过单批上限，换短一点的问题]\n");
            return;
        }
    }
    if (g_pos + n_in >= QWEN3_MAX_SEQ) {
        P("  [prompt 装不进 KV cache，换短一点的问题]\n");
        return;
    }

    mbox_set(BOARD_MBOX_STATUS, BOARD_ST_RUNNING);
    mbox_set(BOARD_MBOX_NTOKEN, 0);
    mbox_text_reset();

    /* KV cache 只有 QWEN3_MAX_SEQ 个位置，而 qwen3_forward_batch 只校验
     * 批大小、不校验 pos —— 写到 pos >= MAX_SEQ 会跨进下一层的 KV 区域，
     * 不报错也不 trap，只是后面几层读到污染的 K/V，输出变成胡话。
     * 这种错查起来极贵（看着像模型算错，其实是内存），所以在这里挡住：
     * 让 CHAT_MAX_GEN 表达"想要多少"，剩余容量不够时自动收敛。 */
    int budget = QWEN3_MAX_SEQ - g_pos - n_in;
    int limit  = CHAT_MAX_GEN < budget ? CHAT_MAX_GEN : budget;
    if (limit <= 0) {
        P("  [KV cache 没有生成空间了，输入 new 清空上下文]\n");
        return;
    }

    uint64_t c0 = rd_mcycle();
    qwen3_forward_batch(m, g_chat_ids, n_in, g_pos);  /* prefill：进度条照常 */
    int next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);
    /* 进度条到此为止。换行让答案从行首开始，读起来才是一段话。 */
    g_show_layer = 0;
    P("\n");
    int got = 0, aborted = 0;
    for (int i = 0; i < limit; i++) {
        if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
        size_t plen;
        const uint8_t *piece = qwen3_tok_piece(tk, next, &plen);
        for (size_t q = 0; q < plen; q++) putc_((char)piece[q]);   /* 流式 */
        /* 同一份结果也写进 mailbox：串口若不通，host 后门照样读得到 */
        if (got < BOARD_MBOX_TOKENS_MAX)
            *BOARD_PTR(uint32_t, BOARD_MBOX_TOKENS + 4u * (unsigned)got) =
                (uint32_t)next;
        mbox_text_append(piece, plen);
        g_last_tok = next;       /* 它不会被 forward，留给下一轮补 */
        got++;
        mbox_set(BOARD_MBOX_NTOKEN, (uint32_t)got);
        /* 最后一个 token 不必再前向：那一整遍 1.1 GB 权重算出的 logits
         * 没有任何人会用到。 */
        if (i + 1 >= limit) break;
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
        qwen3_forward(m, next, g_pos + n_in + i);
        next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);
    }
    /* 推进位置：prefill 占了 n_in 个；生成循环里只有前 got-1 个被 forward
     * 进 KV cache，最后一个留在 g_last_tok 里等下一轮补。 */
    g_pos += n_in + (got > 0 ? got - 1 : 0);

    g_show_layer = 1;          /* 恢复，别影响下一轮 prefill 与其他路径 */
    uint64_t c1 = rd_mcycle();
    mbox_set(BOARD_MBOX_CYCLES_LO, (uint32_t)(c1 - c0));
    mbox_set(BOARD_MBOX_CYCLES_HI, (uint32_t)((c1 - c0) >> 32));
    mbox_set(BOARD_MBOX_STATUS, BOARD_ST_DONE);

    /* 现场把速度算出来。这个数比幻灯片上任何数字都有说服力，
     * 因为它是当场跑出来的。 */
    P("\n\n  [prompt "); I(n_in); P(" token，生成 "); I(got); P(" token");
    if (limit < CHAT_MAX_GEN) {
        /* 说清楚是容量到顶而不是模型自己收尾 —— 两者现象一样。 */
        P("（受 KV cache 余量限制，上限收到 "); I(limit); P("）");
    }
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
    P("多轮对话：上一轮的内容会被记住；输入 new 清空重来\n");
    for (int i = 0; i < N_PRESET; i++) {
        P("  "); I(i + 1); P(") "); P(g_presets[i]); P("\n");
    }
    for (;;) {
        /* 提示符里带上 KV cache 占用：多轮对话下"还能聊几句"是个
         * 随时在变的量，等它满了才发现就晚了。 */
        P("\n"); print_clock(); P(" \u73e0\u5cf0\u5c0f\u52a9\u624b ");
        if (g_pos > 0) { P("["); I(g_pos); P("/"); I(QWEN3_MAX_SEQ); P("] "); }
        P("> ");
        int len = uart_readline(g_line, (int)sizeof g_line);
        if (len == 0) continue;
        if (len == 1 && (g_line[0] == 'q' || g_line[0] == 'Q')) break;
        if (len == 3 && g_line[0] == 'n' && g_line[1] == 'e' && g_line[2] == 'w') {
            g_pos = 0; g_last_tok = -1;
            P("  [上下文已清空]\n");
            continue;
        }

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
    print_banner();
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
    int rc = qwen3_init(&m, blob, WEIGHTS_SIZE, g_scratch, ARENA_LIMIT_BYTES);
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
