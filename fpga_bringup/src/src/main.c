/* PC 侧驱动程序 —— 所有平台相关代码（文件 IO、打印）都在这里。
 * qwen3.c/kernels_*.c 保持零 libc 依赖，可原样编译进 bare-metal 固件。
 *
 * 用法:  qwen3 <权重.bin> [生成步数]
 * 输出:  TOKENS: <id> <id> ...        <- tools/05_e2e_test.py 解析这一行
 *        build/c_prefill.bin + .manifest.json
 *        build/c_decode{0,1,2}.bin + .manifest.json
 *
 * FPGA 上本文件被 main_baremetal.c 取代：权重由 host 经 PCIe 灌进 DDR，
 * dump 改写到 DDR 约定地址供 host 回读 —— 同一套比对工具通用。
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kernels.h"
#include "qwen3.h"
#include "tokenizer.h"

/* 激活缓冲：静态分配，模拟 bare-metal 无 malloc 的环境。
 * 大小取自 qwen3.h 的单一来源，不在此重复推导。 */
static uint8_t g_scratch[QWEN3_SCRATCH_BYTES];

/* ==================== dump 收集 ====================
 *
 * 批量前向之后，每个张量一次 dump 出来就是完整的 [n_token, elems]，
 * 与黄金数据的 [1,N,...] 布局天然一致 —— 直接顺序追加即可。
 *
 * （逐 token 前向时代这里有一套按名字重排的逻辑：同名张量分散在 N 次调用中，
 *   必须重排成连续的 [N,elems] 才能比对。批量化把这个问题消掉了。）
 */
#define DUMP_MAX_BYTES (64u << 20)      /* 64 MB，prefill 4 token 实测约 14 MB */
#define DUMP_MAX_RECS  700              /* 每次前向 479 个张量 */

typedef struct {
    char   name[64];
    size_t off;          /* 在 g_dumpbuf 中的字节偏移 */
    size_t nbytes;
    size_t elems;
} dump_rec_t;

static uint8_t     g_dumpbuf[DUMP_MAX_BYTES];
static dump_rec_t  g_recs[DUMP_MAX_RECS];
static int         g_nrec;
static size_t      g_used;
static int         g_ntok;          /* 本批 token 数，写 manifest 的 shape 用 */
static int         g_enabled;
static int         g_overflow;

static void dump_begin(int n_token) {
    g_nrec = 0; g_used = 0; g_ntok = n_token;
    g_enabled = 1; g_overflow = 0;
}

static void dump_cb(void *ctx, int layer, const char *name,
                    const float *data, size_t n) {
    (void)ctx;
    if (!g_enabled) return;
    if (g_nrec >= DUMP_MAX_RECS) { g_overflow = 1; return; }

    size_t nb = n * sizeof(float);
    if (g_used + nb > DUMP_MAX_BYTES) { g_overflow = 1; return; }

    dump_rec_t *r = &g_recs[g_nrec++];
    if (layer >= 0) snprintf(r->name, sizeof r->name, "L%02d.%s", layer, name);
    else            snprintf(r->name, sizeof r->name, "%s", name);
    r->off = g_used;
    r->nbytes = nb;
    r->elems = n;
    memcpy(g_dumpbuf + g_used, data, nb);
    g_used += nb;
}

static int dump_write(const char *stem) {
    char pb[256], pj[256];
    snprintf(pb, sizeof pb, "build/%s.bin", stem);
    snprintf(pj, sizeof pj, "build/%s.manifest.json", stem);

    FILE *fb = fopen(pb, "wb");
    FILE *fj = fopen(pj, "w");
    if (!fb || !fj) { perror(pb); if (fb) fclose(fb); if (fj) fclose(fj); return -1; }

    fwrite(g_dumpbuf, 1, g_used, fb);
    fprintf(fj, "{\n  \"tag\": \"%s\",\n  \"dtype\": \"fp32\",\n"
                "  \"count\": %d,\n  \"tensors\": [\n", stem, g_nrec);
    for (int i = 0; i < g_nrec; i++) {
        dump_rec_t *r = &g_recs[i];
        /* logits 只算最后一个 token，故其 shape 是 [1,1,V] 而非 [1,N,V]；
         * 其余张量都是 [1,N,elems/N]。这里按实际元素数如实记录。 */
        int nt = (r->elems % (size_t)g_ntok == 0) ? g_ntok : 1;
        fprintf(fj, "    {\"name\": \"%s\", \"shape\": [1, %d, %zu], "
                    "\"offset\": %zu, \"nbytes\": %zu, \"dtype\": \"fp32\"}%s\n",
                r->name, nt, r->elems / (size_t)nt, r->off, r->nbytes,
                (i + 1 < g_nrec) ? "," : "");
    }
    fprintf(fj, "  ]\n}\n");
    fclose(fb); fclose(fj);
    if (g_overflow) { fprintf(stderr, "[WARN] dump 溢出，%s 不完整\n", stem); return -1; }
    return 0;
}

/* ==================== 权重加载 ==================== */
static const void *map_file(const char *path, size_t *out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return NULL; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return NULL; }
    *out_size = (size_t)st.st_size;
    return p;
}

/* golden_meta.json 里 raw 模式的 prompt token（"你好，请介绍一下你自己"） */
static const int g_prompt[] = { 108386, 37945, 109432, 107828 };
#define N_PROMPT ((int)(sizeof g_prompt / sizeof g_prompt[0]))
#define MAX_GEN  64

/* ==================== 对话模式 ====================
 *
 * Qwen3 的 chat template（enable_thinking=False 时）。由黄金数据反推得到：
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * 那几个 <|...|> 会被 tokenizer 当作特殊 token 整体匹配，不参与 BPE。
 * 末尾预置空的 <think></think> 是关闭思考模式的做法 —— 否则 0.6B 会先自言自语
 * 一两百个 token 才进入正题，demo 上观感很差。 */
#define CHAT_PREFIX "<|im_start|>user\n"
#define CHAT_MIDDLE "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

static uint32_t g_tok_htab[1u << 19];      /* merges 哈希表，见 qwen3_tok_scratch_bytes */
static char     g_chat_buf[4096];
static int      g_chat_ids[QWEN3_TOK_MAX];

static int chat_mode(qwen3_t *m, const char *weights_dir_hint, const char *prompt,
                     int max_new) {
    (void)weights_dir_hint;
    size_t tsz = 0;
    const void *tblob = map_file("golden/tokenizer.bin", &tsz);
    if (!tblob) { fprintf(stderr, "缺少 golden/tokenizer.bin，先跑 tools/06_export_tokenizer.py\n");
                  return 1; }

    static qwen3_tok_t tk;
    int rc = qwen3_tok_init(&tk, tblob, tsz, g_tok_htab, sizeof g_tok_htab);
    if (rc != QWEN3_TOK_OK) {
        fprintf(stderr, "tokenizer 初始化失败: %s\n", qwen3_tok_strerror(rc));
        return 1;
    }

    int n = snprintf(g_chat_buf, sizeof g_chat_buf, "%s%s%s",
                     CHAT_PREFIX, prompt, CHAT_MIDDLE);
    if (n < 0 || (size_t)n >= sizeof g_chat_buf) {
        fprintf(stderr, "prompt 过长\n"); return 1;
    }

    int n_in = qwen3_tok_encode(&tk, g_chat_buf, (size_t)n, g_chat_ids, QWEN3_TOK_MAX);
    if (n_in < 0) { fprintf(stderr, "编码失败: %s\n", qwen3_tok_strerror(n_in)); return 1; }
    if (n_in > QWEN3_MAX_BATCH) {
        fprintf(stderr, "prompt %d 个 token，超过单批上限 %d（分块尚未实现）\n",
                n_in, QWEN3_MAX_BATCH);
        return 1;
    }

    printf("提问: %s\n", prompt);
    printf("(prompt %d 个 token, kernel=%s)\n回答: ", n_in, qwen3_kernel_name());
    fflush(stdout);

    qwen3_forward_batch(m, g_chat_ids, n_in, 0);
    int next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);

    for (int i = 0; i < max_new; i++) {
        if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
        size_t plen;
        const uint8_t *piece = qwen3_tok_piece(&tk, next, &plen);
        fwrite(piece, 1, plen, stdout);     /* 流式输出：逐 token 打印 */
        fflush(stdout);
        qwen3_forward(m, next, n_in + i);
        next = qwen3_argmax(m->s.logits, QWEN3_VOCAB_SIZE);
    }
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    /* 两种用法：
     *   qwen3 <权重> [steps]          回归测试：跑黄金 prompt，输出 TOKENS 与 dump
     *   qwen3 <权重> -c "你的问题"     对话：中文进、中文出，流式输出
     */
    const char *wpath = (argc > 1) ? argv[1] : "golden/qwen3-0.6b-bf16.bin";
    const char *chat = NULL;
    int steps = 20;
    if (argc > 2) {
        if (argv[2][0] == '-' && argv[2][1] == 'c')
            chat = (argc > 3) ? argv[3] : "你好";
        else
            steps = atoi(argv[2]);
    }
    if (steps < 1) steps = 1;
    if (steps > MAX_GEN) steps = MAX_GEN;

    size_t wsize = 0;
    const void *blob = map_file(wpath, &wsize);
    if (!blob) return 1;

    static qwen3_t m;
    int rc = qwen3_init(&m, blob, wsize, g_scratch, sizeof g_scratch);
    if (rc != QWEN3_OK) {
        fprintf(stderr, "qwen3_init 失败: %s (%d)\n", qwen3_strerror(rc), rc);
        return 1;
    }

    printf("权重 %s  (%.3f GiB)   kernel=%s\n",
           wpath, (double)wsize / 1073741824.0, qwen3_kernel_name());
    printf("激活 arena %.1f MiB (KV cache %.1f MiB @ MAX_SEQ=%d)\n",
           (double)m.arena.used / 1048576.0,
           (double)QWEN3_N_LAYERS * QWEN3_MAX_SEQ * QWEN3_KV_DIM * 4 * 2 / 1048576.0,
           QWEN3_MAX_SEQ);

    if (chat) return chat_mode(&m, wpath, chat, MAX_GEN);

    int gen[MAX_GEN + N_PROMPT];
    int n_gen = 0;

    /* ---------------- Prefill：4 个 prompt token 一次批量前向 ----------------
     * ★ 这里 GEMM 的 M = N_PROMPT，不再是 1 —— AME 的 tile 才有机会被填上。
     *   prompt 超过 QWEN3_MAX_BATCH 时需要分块，当前测试 prompt 只有 4 个。 */
    qwen3_set_dump(&m, dump_cb, NULL);
    dump_begin(N_PROMPT);
    qwen3_forward_batch(&m, g_prompt, N_PROMPT, 0);
    g_enabled = 0;
    dump_write("c_prefill");

    int next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
    gen[n_gen++] = next;

    /* ---------------- Decode：前 3 步各自成批并 dump ---------------- */
    for (int step = 0; step < steps - 1; step++) {
        char stem[32];
        int do_dump = (step < 3);
        if (do_dump) { dump_begin(1); }
        else         { g_enabled = 0; qwen3_set_dump(&m, NULL, NULL); }

        qwen3_forward(&m, next, N_PROMPT + step);

        if (do_dump) {
            g_enabled = 0;
            snprintf(stem, sizeof stem, "c_decode%d", step);
            dump_write(stem);
        }
        next = qwen3_argmax(m.s.logits, QWEN3_VOCAB_SIZE);
        gen[n_gen++] = next;
        if (next == QWEN3_EOS_TOKEN_ID_0 || next == QWEN3_EOS_TOKEN_ID_1) break;
    }

    printf("TOKENS:");
    for (int i = 0; i < n_gen; i++) printf(" %d", gen[i]);
    printf("\n");
    return 0;
}
