/* Qwen3 分词器 —— GPT-2 风格的 byte-level BPE，bare-metal 可用。
 *
 * 纪律与 qwen3.c 相同：不 include stdio/stdlib，不调用 malloc。
 *
 * ── 编码流程 ────────────────────────────────────────────────
 *   输入 UTF-8 字节
 *     ① 特殊 token 优先整体匹配（<|im_start|> 等，不参与 BPE）
 *     ② 正则切分为片段（依赖 Unicode \p{L} \p{N} 分类）
 *     ③ 每个片段先拆成单字节 token
 *     ④ 按 merges 的 rank 反复合并相邻对
 *   输出 token id
 *
 * ── 与 HuggingFace 的两点差异 ───────────────────────────────
 * 1. byte-level 编码已在导出阶段消掉（见 tools/06_export_tokenizer.py）：
 *    词表里存的直接是原始字节，故此处直接在 UTF-8 字节上工作。
 * 2. NFC 规范化未实现 —— 对 ASCII 与常见 CJK 它是恒等变换。
 *    tests/tokenizer_test.c 用 34 组真实文本验证该简化是否成立。
 */
#ifndef QWEN3_TOKENIZER_H
#define QWEN3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/* 一次编码最多产出多少 token —— 也限制了单次输入的长度。 */
#ifndef QWEN3_TOK_MAX
#define QWEN3_TOK_MAX 2048
#endif

typedef struct {
    uint32_t id, len, off;
} qwen3_tok_special_t;

typedef struct {
    const uint32_t *index;      /* [n_vocab+1] 每个 token 在 data 中的起始偏移 */
    const uint8_t  *data;       /* 所有 token 的原始字节，连续存放 */
    const uint32_t *merges;     /* [n_merges][3] = (left_id, right_id, merged_id)，序即 rank */
    const uint32_t *lrange;     /* [n_lrange][2] Unicode 字母区间 */
    const uint32_t *nrange;     /* [n_nrange][2] Unicode 数字区间 */
    const uint32_t *byte_id;    /* [256] 单字节 -> token id */
    const qwen3_tok_special_t *specials;
    const uint8_t  *sdata;      /* 特殊 token 的字符串数据 */
    uint32_t n_vocab, n_merges, n_lrange, n_nrange, n_special;

    /* merges 的哈希表：把 (left,right) 映射到 rank。
     * 线性探测开放寻址，容量取 2 的幂且约为条目数的 2 倍。
     * 不用二分是因为 merges 按 rank 排序而非按 key 排序。 */
    uint32_t *htab;             /* [hcap] 存 merge 下标 +1，0 表示空槽 */
    uint32_t  hcap;
} qwen3_tok_t;

enum {
    QWEN3_TOK_OK = 0,
    QWEN3_TOK_ERR_MAGIC = -1,
    QWEN3_TOK_ERR_VERSION = -2,
    QWEN3_TOK_ERR_SIZE = -3,
    QWEN3_TOK_ERR_OOM = -4,      /* scratch 不足以建哈希表 */
    QWEN3_TOK_ERR_OVERFLOW = -5, /* 输出 token 数超过 max_out */
};

/* 从 tools/06_export_tokenizer.py 产出的 tokenizer.bin 建立分词器。
 * blob 须在 t 的生命周期内有效。scratch 用于哈希表，
 * 所需字节数见 qwen3_tok_scratch_bytes()。 */
int qwen3_tok_init(qwen3_tok_t *t, const void *blob, size_t blob_size,
                   void *scratch, size_t scratch_size);

size_t qwen3_tok_scratch_bytes(void);

/* 编码：UTF-8 文本 -> token id。返回 token 数，负数为错误码。 */
int qwen3_tok_encode(const qwen3_tok_t *t, const char *text, size_t len,
                     int *out, int max_out);

/* 解码：token id -> UTF-8 文本。返回写入的字节数（不含结尾 '\0'），
 * 负数为错误码。out 至少要有 max_out 字节。 */
int qwen3_tok_decode(const qwen3_tok_t *t, const int *ids, int n,
                     char *out, size_t max_out);

/* 单个 token 的原始字节，供流式输出用。*len 返回字节数。 */
const uint8_t *qwen3_tok_piece(const qwen3_tok_t *t, int id, size_t *len);

const char *qwen3_tok_strerror(int err);

#endif /* QWEN3_TOKENIZER_H */
