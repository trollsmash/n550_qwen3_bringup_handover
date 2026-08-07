/* QEMU user-mode 平台层 —— 交叉编译验证用。
 *
 * 与 main.c(x86/glibc) 的唯一区别是 IO 实现：这里直接发 Linux syscall，
 * 不链接 libc。原因是手上只有 elf(newlib) 工具链，其 printf/open 走
 * semihosting，user-mode QEMU 不认；而 syscall 是 QEMU 直接转译的。
 *
 * qwen3.c / kernels_*.c 原样复用，一行不改。
 *
 * 编译（见 tools/env.sh）：
 *   $CROSS-gcc $RISCV_CFLAGS -ffreestanding -nostdlib -nostartfiles \
 *       -Wl,-e,_start src/qwen3.c src/kernels_scalar.c src/main_qemu.c \
 *       -o $RISCV_BUILD/qwen3.elf -lm
 * 运行：
 *   qrun $RISCV_BUILD/qwen3.elf golden/qwen3-0.6b-bf16.bin 20
 */
#include <stdint.h>

#include "kernels.h"
#include "ops.h"
#include "qwen3.h"
#include "tokenizer.h"

/* libm 所需的 __errno 桩已移到 src/bsp/libm_stub.c —— 那是共享的 freestanding
 * 支持代码，测试程序也要用。链接时须一并带上。 */

/* ==================== Linux syscall ==================== */
#define SYS_openat 56
#define SYS_close  57
#define SYS_lseek  62
#define SYS_write  64
#define SYS_exit   93
#define SYS_mmap   222

#define AT_FDCWD     (-100)
#define O_RDONLY     0
#define SEEK_END     2
#define PROT_READ    1
#define MAP_PRIVATE  2

static long sc3(long n, long a, long b, long c) {
    register long a0 asm("a0") = a, a1 asm("a1") = b, a2 asm("a2") = c;
    register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}
static long sc6(long n, long a, long b, long c, long d, long e, long f) {
    register long a0 asm("a0") = a, a1 asm("a1") = b, a2 asm("a2") = c;
    register long a3 asm("a3") = d, a4 asm("a4") = e, a5 asm("a5") = f;
    register long a7 asm("a7") = n;
    asm volatile("ecall" : "+r"(a0)
                 : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a7)
                 : "memory");
    return a0;
}

static long sys_open(const char *p) { return sc3(SYS_openat, AT_FDCWD, (long)p, O_RDONLY); }
static long sys_close(long fd)      { return sc3(SYS_close, fd, 0, 0); }
static long sys_size(long fd)       { return sc3(SYS_lseek, fd, 0, SEEK_END); }
static long sys_write(long fd, const void *b, unsigned long n) {
    return sc3(SYS_write, fd, (long)b, (long)n);
}
static void sys_exit(long c) {
    sc3(SYS_exit, c, 0, 0);
    __builtin_unreachable();
}
#define MAP_FIXED     0x10
#define MAP_ANONYMOUS 0x20

static void *sys_mmap_ro(long fd, unsigned long len) {
    long r = sc6(SYS_mmap, 0, (long)len, PROT_READ, MAP_PRIVATE, fd, 0);
    return (r < 0 && r > -4096) ? 0 : (void *)r;
}

/* 在权重映射的尾部再挂一段匿名只读内存作为护栏。
 *
 * 起因：AME 的 tile 装载按 mtilen 行 × 行 stride 读取，末尾那个 tile 会一直
 * 摸到权重区的最后一个字节。而 L27 的 down_proj 是文件里最后一个大权重，
 * 其后仅剩 2 KB 的 model.norm 就到文件末尾 —— 一旦装载多读哪怕一行，
 * 就越过 mmap 区域触发 SIGSEGV。前 27 层因后面还有大量数据而侥幸无恙，
 * 这正是"只有最后一层崩"的原因。
 *
 * 护栏让越界读落到合法的零页上，从而把"崩溃"变成"可观测的数值误差"，
 * 便于判断究竟是不是越界。 */
static int mmap_guard(const void *blob, unsigned long len, unsigned long guard) {
    unsigned long end = ((unsigned long)blob + len + 4095u) & ~4095ul;
    long r = sc6(SYS_mmap, (long)end, (long)guard, PROT_READ,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    return (r < 0 && r > -4096) ? -1 : 0;
}

/* ==================== 输出 ==================== */
static unsigned long slen(const char *s) { unsigned long n = 0; while (s[n]) n++; return n; }
static void P(const char *s)  { sys_write(1, s, slen(s)); }
static void PE(const char *s) { sys_write(2, s, slen(s)); }
static void PI(long v) {
    char b[24]; int i = 0;
    if (v < 0) { P("-"); v = -v; }
    if (!v) { P("0"); return; }
    while (v) { b[i++] = (char)('0' + v % 10); v /= 10; }
    char o[24]; int j = 0; while (i) o[j++] = b[--i];
    sys_write(1, o, (unsigned long)j);
}
/* 打印一位小数，避免为了显示浮点去引入整个 printf */
static void PF1(long milli) { PI(milli / 1000); P("."); PI((milli / 100) % 10); }

/* ==================== 主体 ==================== */
static uint8_t g_scratch[QWEN3_SCRATCH_BYTES];

/* golden_meta.json 里 raw 模式的 prompt token（"你好，请介绍一下你自己"） */
static const int g_prompt[] = { 108386, 37945, 109432, 107828 };
#define N_PROMPT ((int)(sizeof g_prompt / sizeof g_prompt[0]))
#define MAX_GEN  64

static int parse_int(const char *s) {
    int v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
    return v;
}

/* ==================== 对话模式 ====================
 * chat template 与 main.c 一致，由黄金数据反推：
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * 末尾预置空的 <think></think> 用于关闭思考模式 —— 否则 0.6B 会先自言自语
 * 一两百个 token 才进正题。 */
#define CHAT_PREFIX "<|im_start|>user\n"
#define CHAT_MIDDLE "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

static uint32_t g_tok_htab[1u << 19];
static char     g_chat_buf[4096];
static int      g_chat_ids[QWEN3_TOK_MAX];

static unsigned long cat(char *dst, unsigned long at, const char *s) {
    while (*s) dst[at++] = *s++;
    return at;
}

static int chat_mode(qwen3_t *m, const char *prompt, int max_new) {
    size_t tsz = 0;
    long fd = sys_open("golden/tokenizer.bin");
    if (fd < 0) { PE("缺少 golden/tokenizer.bin\n"); return 1; }
    tsz = (size_t)sys_size(fd);
    const void *tblob = sys_mmap_ro(fd, tsz);
    sys_close(fd);
    if (!tblob) { PE("tokenizer mmap 失败\n"); return 1; }

    static qwen3_tok_t tk;
    int rc = qwen3_tok_init(&tk, tblob, tsz, g_tok_htab, sizeof g_tok_htab);
    if (rc != QWEN3_TOK_OK) { PE("tokenizer 初始化失败: ");
                              PE(qwen3_tok_strerror(rc)); PE("\n"); return 1; }

    unsigned long n = 0;
    n = cat(g_chat_buf, n, CHAT_PREFIX);
    n = cat(g_chat_buf, n, prompt);
    n = cat(g_chat_buf, n, CHAT_MIDDLE);

    int n_in = qwen3_tok_encode(&tk, g_chat_buf, n, g_chat_ids, QWEN3_TOK_MAX);
    if (n_in < 0) { PE("编码失败\n"); return 1; }
    if (n_in > QWEN3_MAX_BATCH) { PE("prompt 超过单批上限（分块尚未实现）\n"); return 1; }

    P("提问: "); P(prompt); P("\n(prompt "); PI(n_in);
    P(" 个 token, kernel="); P(qwen3_kernel_name());
    P(" ops="); P(qwen3_ops_name()); P(")\n回答: ");

    qwen3_forward_batch(m, g_chat_ids, n_in, 0);
    int next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);

    for (int i = 0; i < max_new; i++) {
        if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
        size_t plen;
        const uint8_t *piece = qwen3_tok_piece(&tk, next, &plen);
        sys_write(1, piece, plen);          /* 流式输出，syscall 无缓冲 */
        qwen3_forward(m, next, n_in + i);
        next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);
    }
    P("\n");
    return 0;
}

/* 调试用：把每个算子出口的名字实时打出来。syscall 直接写、无缓冲，
 * 所以程序崩溃时最后一行就是现场。加第三个参数 trace 开启。 */
static void trace_cb(void *ctx, int layer, const char *name,
                     const float *data, size_t n) {
    (void)ctx; (void)data;
    P("  L"); PI(layer); P(" "); P(name); P(" n="); PI((long)n); P("\n");
}

void qwen3_main(long argc, char **argv) {
    const char *wpath = (argc > 1) ? argv[1] : "golden/qwen3-0.6b-bf16.bin";
    int steps = (argc > 2) ? parse_int(argv[2]) : 20;
    if (steps < 1) steps = 1;
    if (steps > MAX_GEN) steps = MAX_GEN;

    long fd = sys_open(wpath);
    if (fd < 0) { PE("open failed: "); PE(wpath); PE("\n"); sys_exit(1); }
    long wsize = sys_size(fd);
    const void *blob = sys_mmap_ro(fd, (unsigned long)wsize);
    sys_close(fd);
    if (!blob) { PE("mmap failed\n"); sys_exit(1); }
    /* 满足 kernels.h 的 QWEN3_WEIGHT_TAIL_PAD 契约：AME 的 tile 装载会读越过
     * 权重末尾。实测 4 KB 已足够，取 8 KB 留余量。 */
    if (mmap_guard(blob, (unsigned long)wsize, QWEN3_WEIGHT_TAIL_PAD) != 0)
        PE("警告: 权重尾部护栏挂载失败，AME kernel 可能越界读\n");

    static qwen3_t m;
    int rc = qwen3_init(&m, blob, (unsigned long)wsize, g_scratch, sizeof g_scratch);
    if (rc != QWEN3_OK) {
        PE("qwen3_init failed: "); PE(qwen3_strerror(rc)); PE("\n");
        sys_exit(1);
    }

    P("weights "); P(wpath); P("  ");
    PF1((long)(wsize / 1073741ll));  P(" GiB   kernel="); P(qwen3_kernel_name()); P("\n");
    P("arena "); PF1((long)(m.arena.used / 1048ll)); P(" MiB\n");

    /* 对话模式：./qwen3.elf w.bin <最多生成几个token> chat "你的问题"
     * QEMU 上每个 token 要几十秒，故生成长度由第二个参数控制。 */
    if (argc > 4 && argv[3][0] == 'c')
        sys_exit(chat_mode(&m, argv[4], steps));

    if (argc > 3 && argv[3][0] == 't') {  /* trace：逐算子打印名字，定位崩溃点 */
        P("== trace 模式 ==\n");
        qwen3_set_dump(&m, trace_cb, 0);
    }

    /* ---- 性能对比模式 ----
     *   ./qwen3.elf w.bin 1 batch 128   一次批量前向 128 个 token（M=128）
     *   ./qwen3.elf w.bin 1 seq   128   逐 token 前向 128 次（M=1）
     * 两者算的是同一批 token，唯一区别是 GEMM 的 M 维 ——
     * 这正是衡量 AME tile 利用率的实验：M=128 填满 tile，M=1 只用 1/128。
     * 合成 token 仅为压测，不校验输出。 */
    if (argc > 4) {
        int nb = parse_int(argv[4]);
        if (nb < 1) nb = 1;
        if (nb > QWEN3_MAX_BATCH) nb = QWEN3_MAX_BATCH;
        static int toks[QWEN3_MAX_BATCH];
        for (int i = 0; i < nb; i++) toks[i] = g_prompt[i % N_PROMPT];

        int seq = (argv[3][0] == 's');
        P(seq ? "bench: 逐 token (M=1) x" : "bench: 批量 (M=");
        PI(nb); P(seq ? "\n" : ")\n");

        if (seq) for (int i = 0; i < nb; i++) qwen3_forward(&m, toks[i], i);
        else     qwen3_forward_batch(&m, toks, nb, 0);

        P("bench done\n");
        sys_exit(0);
    }

    int gen[MAX_GEN + N_PROMPT];
    int n_gen = 0;

    /* ★ prefill 一次批量前向：GEMM 的 M = N_PROMPT 而非 1，
     *   AME 的 tile 才有机会被填上。prompt 超过 QWEN3_MAX_BATCH 需分块。 */
    qwen3_forward_batch(&m, g_prompt, N_PROMPT, 0);
    int next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
    gen[n_gen++] = next;

    for (int step = 0; step < steps - 1; step++) {
        qwen3_forward(&m, next, N_PROMPT + step);
        next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
        gen[n_gen++] = next;
        if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
    }

    P("TOKENS:");
    for (int i = 0; i < n_gen; i++) { P(" "); PI(gen[i]); }
    P("\n");
    sys_exit(0);
}

/* 入口：从栈上取 argc/argv 传给 C。
 * Linux ABI 下 _start 时 sp 指向 argc，其后是 argv[]。
 * 必须初始化 gp，否则访问全局变量时段错误（且崩溃点与真实原因毫不相干）。 */
asm(
    ".global _start\n"
    "_start:\n"
    ".option push\n.option norelax\n"
    "  la   gp, __global_pointer$\n"
    ".option pop\n"
    "  ld   a0, 0(sp)\n"          /* argc */
    "  addi a1, sp, 8\n"          /* argv */
    "  call qwen3_main\n");
