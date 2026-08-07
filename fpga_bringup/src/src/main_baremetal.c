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
 *   症状是"magic 不对"，看起来却像加载没生效 —— 曾经真的踩过。
 *   下面的 check_layout() 就是为此设的护栏，改 scratch 大小时别删。 */
/* ★ 具体数值全部来自 bsp/board.h —— 换板子只改那一个文件。
 * 这里曾经与 start.S 的 trap handler 各写了一份 UART 基址，
 * 上板时改一处漏一处是必然的。 */
#define DRAM_BASE      ((unsigned long)BOARD_DRAM_BASE)
#define WEIGHTS_ADDR   ((unsigned long)BOARD_WEIGHTS_ADDR)
#define TOKENIZER_ADDR ((unsigned long)BOARD_TOKENIZER_ADDR)

#define UART_BASE     ((unsigned long)BOARD_UART_BASE)
#define UART_THR      0x00
#define UART_LSR      0x05
#define UART_LSR_THRE 0x20

/* ==================== UART ==================== */
static void putc_(char c) {
    /* 有上限的轮询：真机上等 THRE 是对的，但若 LSR 读不到预期值也不能死等，
     * 否则整个程序静默挂死、毫无线索。 */
    for (int i = 0; i < 100000; i++)
        if (*(volatile uint8_t *)(UART_BASE + UART_LSR) & UART_LSR_THRE) break;
    *(volatile uint8_t *)(UART_BASE + UART_THR) = (uint8_t)c;
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
#define WEIGHTS_SIZE 1192100096UL

/* 覆盖 qwen3.c 里的弱符号，把 28 层的推进过程吐到串口。
 *
 * 裸机上这是唯一的观测手段。没有它，"卡在第 3 层"和"正常跑到第 27 层"
 * 在外面看来一模一样，都是串口没动静 —— QEMU 上那次跑了 3 小时不知道
 * 卡在哪，就是这个原因。FPGA 上只会更需要它。
 *
 * 输出形如 [....|....|....|....|....|...] ，每 5 层一个分隔符便于数。 */
/* ---- 与 host 的 mailbox。PCIe 后门可回读 DDR，所以即使串口没接好，
 *      host 也能看到进度和结果。真机 bring-up 优先走这条路。 ---- */
static void mbox_set(unsigned long addr, uint32_t v) {
    *BOARD_PTR(uint32_t, addr) = v;
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

void qwen3_main(void) {
    P("\n=== Qwen3-0.6B on RISC-V (bare-metal) ===\n");
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
        unsigned long n = 0;
        n = cat(g_chat_buf, n, CHAT_PREFIX);
        n = cat(g_chat_buf, n, DEMO_PROMPT);
        n = cat(g_chat_buf, n, CHAT_MIDDLE);

        int n_in = qwen3_tok_encode(&tk, g_chat_buf, n, g_chat_ids, QWEN3_TOK_MAX);
        if (n_in < 0 || n_in > QWEN3_MAX_BATCH) die("prompt 编码失败或超过单批上限");

        P("\n提问: " DEMO_PROMPT "\n");
        P("(prompt "); I(n_in); P(" 个 token)\n回答: ");

        qwen3_forward_batch(&m, g_chat_ids, n_in, 0);
        int next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
        for (int i = 0; i < CHAT_MAX_GEN; i++) {
            if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
            size_t plen;
            const uint8_t *piece = qwen3_tok_piece(&tk, next, &plen);
            for (size_t q = 0; q < plen; q++) putc_((char)piece[q]);  /* 流式输出 */
            /* 最后一个 token 不必再前向：那一整遍 1.1 GB 权重算出来的
             * logits 没有任何人会用到。之前的写法白算了一次。 */
            if (i + 1 >= CHAT_MAX_GEN) break;
            qwen3_forward(&m, next, n_in + i);
            next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
        }
        P("\n\n>>> DONE\n");
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
    static const int expect[] = { 3837, 101889, 106525, 56568 };
    int ok = 1;
    for (int i = 0; i < 4 && i < n_gen; i++) if (gen[i] != expect[i]) ok = 0;
    P(ok ? "\n>>> PASS: 前 4 个 token 与黄金数据一致\n"
         : "\n>>> FAIL: 与黄金数据不符\n");

    halt(ok);
}
