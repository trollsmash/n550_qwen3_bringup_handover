/* Qwen3 分词器实现 —— 见 tokenizer.h 的流程说明。
 *
 * 纪律：不 include stdio/stdlib，不调用 malloc，可原样编译进 bare-metal 固件。
 */
#include "tokenizer.h"

#include <string.h>     /* memcpy —— freestanding 亦提供 */

/* ==================== 二进制解析 ==================== */

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* 哈希表容量：取不小于 2*n 的 2 的幂。 */
#define TOK_HCAP_FOR(n) tok_hcap_for(n)
static uint32_t tok_hcap_for(uint32_t n) {
    uint32_t c = 1024;
    while (c < n * 2u) c <<= 1;
    return c;
}
/* merges 上限固定（151387 条），故 scratch 大小可静态确定。 */
#define TOK_MERGES_MAX 200000u

size_t qwen3_tok_scratch_bytes(void) {
    return (size_t)tok_hcap_for(TOK_MERGES_MAX) * sizeof(uint32_t);
}

/* (left,right) 的哈希。两个 id 都 < 2^18，拼成 64 位再混淆。 */
static uint32_t pair_hash(uint32_t l, uint32_t r) {
    uint64_t k = ((uint64_t)l << 32) | r;
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)k;
}

static void htab_put(qwen3_tok_t *t, uint32_t idx) {
    const uint32_t *m = t->merges + (size_t)idx * 3;
    uint32_t h = pair_hash(m[0], m[1]) & (t->hcap - 1);
    while (t->htab[h]) {                       /* 线性探测 */
        h = (h + 1) & (t->hcap - 1);
    }
    t->htab[h] = idx + 1;                      /* 存 下标+1，0 表示空槽 */
}

/* 查 (l,r) 对应的 merge。命中返回其在 merges 中的下标（即 rank），否则 -1。 */
static int htab_get(const qwen3_tok_t *t, uint32_t l, uint32_t r) {
    uint32_t h = pair_hash(l, r) & (t->hcap - 1);
    while (t->htab[h]) {
        uint32_t idx = t->htab[h] - 1;
        const uint32_t *m = t->merges + (size_t)idx * 3;
        if (m[0] == l && m[1] == r) return (int)idx;
        h = (h + 1) & (t->hcap - 1);
    }
    return -1;
}

int qwen3_tok_init(qwen3_tok_t *t, const void *blob, size_t blob_size,
                   void *scratch, size_t scratch_size) {
    const uint8_t *b = (const uint8_t *)blob;
    memset(t, 0, sizeof *t);

    if (blob_size < 64) return QWEN3_TOK_ERR_SIZE;
    if (b[0] != 'Q' || b[1] != 'W' || b[2] != '3' || b[3] != 'T')
        return QWEN3_TOK_ERR_MAGIC;
    if (rd32(b + 4) != 1) return QWEN3_TOK_ERR_VERSION;

    t->n_vocab   = rd32(b + 8);
    t->n_merges  = rd32(b + 12);
    t->n_lrange  = rd32(b + 16);
    t->n_nrange  = rd32(b + 20);
    uint32_t off_index  = rd32(b + 24);
    uint32_t off_data   = rd32(b + 28);
    uint32_t off_merges = rd32(b + 32);
    uint32_t off_lr     = rd32(b + 36);
    uint32_t off_nr     = rd32(b + 40);
    uint32_t off_byte   = rd32(b + 44);
    uint32_t off_spec   = rd32(b + 48);
    uint32_t off_sdata  = rd32(b + 52);
    uint32_t total      = rd32(b + 56);
    if (total != blob_size) return QWEN3_TOK_ERR_SIZE;
    if (t->n_merges > TOK_MERGES_MAX) return QWEN3_TOK_ERR_SIZE;

    t->index   = (const uint32_t *)(b + off_index);
    t->data    = b + off_data;
    t->merges  = (const uint32_t *)(b + off_merges);
    t->lrange  = (const uint32_t *)(b + off_lr);
    t->nrange  = (const uint32_t *)(b + off_nr);
    t->byte_id = (const uint32_t *)(b + off_byte);
    t->n_special = rd32(b + off_spec);
    t->specials  = (const qwen3_tok_special_t *)(b + off_spec + 4);
    t->sdata     = b + off_sdata;

    t->hcap = tok_hcap_for(t->n_merges);
    if (scratch_size < (size_t)t->hcap * sizeof(uint32_t))
        return QWEN3_TOK_ERR_OOM;
    t->htab = (uint32_t *)scratch;
    memset(t->htab, 0, (size_t)t->hcap * sizeof(uint32_t));
    for (uint32_t i = 0; i < t->n_merges; i++) htab_put(t, i);

    return QWEN3_TOK_OK;
}

const uint8_t *qwen3_tok_piece(const qwen3_tok_t *t, int id, size_t *len) {
    if (id < 0 || (uint32_t)id >= t->n_vocab) { *len = 0; return t->data; }
    uint32_t a = t->index[id], c = t->index[id + 1];
    *len = c - a;
    return t->data + a;
}

/* ==================== UTF-8 与 Unicode 分类 ==================== */

/* 解码一个码点，返回消耗的字节数；非法序列按单字节处理（与 byte-level 回退一致）。 */
static int utf8_next(const uint8_t *p, size_t remain, uint32_t *cp) {
    uint8_t c = p[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && remain >= 2 && (p[1] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | (p[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && remain >= 3 && (p[1] & 0xC0) == 0x80
        && (p[2] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6)
            | (p[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && remain >= 4 && (p[1] & 0xC0) == 0x80
        && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12)
            | ((uint32_t)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        return 4;
    }
    *cp = c;
    return 1;
}

static int in_ranges(const uint32_t *r, uint32_t n, uint32_t cp) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {                          /* 二分：区间互不重叠且已排序 */
        uint32_t mid = (lo + hi) / 2;
        if (cp < r[mid * 2])       hi = mid;
        else if (cp > r[mid * 2 + 1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}
static int is_letter(const qwen3_tok_t *t, uint32_t cp) {
    return in_ranges(t->lrange, t->n_lrange, cp);
}
static int is_number(const qwen3_tok_t *t, uint32_t cp) {
    return in_ranges(t->nrange, t->n_nrange, cp);
}
static int is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'
        || cp == 0x0B || cp == 0x0C || cp == 0x85 || cp == 0xA0
        || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029
        || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static int is_nl(uint32_t cp) { return cp == '\n' || cp == '\r'; }

/* ==================== BPE ==================== */

/* 对一个片段做 BPE：先拆成单字节 token，再按 rank 反复合并相邻对。
 * 朴素 O(n²) 实现 —— 片段通常只有几到几十字节，够用且不易写错。 */
static int bpe_encode(const qwen3_tok_t *t, const uint8_t *p, size_t n,
                      int *out, int max_out) {
    if (n == 0) return 0;
    if (n > QWEN3_TOK_MAX) return QWEN3_TOK_ERR_OVERFLOW;

    static int buf[QWEN3_TOK_MAX];         /* 非重入，但本项目单线程 */
    int cnt = (int)n;
    for (size_t i = 0; i < n; i++) buf[i] = (int)t->byte_id[p[i]];

    for (;;) {
        int best_rank = -1, best_at = -1;
        for (int i = 0; i + 1 < cnt; i++) {
            int rk = htab_get(t, (uint32_t)buf[i], (uint32_t)buf[i + 1]);
            if (rk >= 0 && (best_rank < 0 || rk < best_rank)) {
                best_rank = rk;
                best_at = i;
            }
        }
        if (best_at < 0) break;
        buf[best_at] = (int)t->merges[(size_t)best_rank * 3 + 2];
        for (int i = best_at + 1; i + 1 < cnt; i++) buf[i] = buf[i + 1];
        cnt--;
    }

    if (cnt > max_out) return QWEN3_TOK_ERR_OVERFLOW;
    for (int i = 0; i < cnt; i++) out[i] = buf[i];
    return cnt;
}

/* ==================== 正则切分 ====================
 *
 * 手写这一条正则的状态机，不引入正则引擎：
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)      英文缩写
 *   | [^\r\n\p{L}\p{N}]?\p{L}+        可选前导符号 + 连续字母
 *   | \p{N}                           单个数字（注意：一次只吃一个）
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*        可选前导空格 + 连续符号 + 尾随换行
 *   | \s*[\r\n]+                      空白后跟换行
 *   | \s+(?!\S)                       尾部空白（后面没有非空白）
 *   | \s+                             其余空白
 * 按顺序尝试，第一个匹配成功的胜出 —— 与正则的交替语义一致。
 */
static size_t match_contraction(const uint8_t *p, size_t n) {
    if (n < 2 || p[0] != '\'') return 0;
    uint8_t a = p[1] | 0x20;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (n >= 3) {
        uint8_t c = p[2] | 0x20;
        if ((a == 'r' && c == 'e') || (a == 'v' && c == 'e')
            || (a == 'l' && c == 'l')) return 3;
    }
    return 0;
}

/* 切出下一个片段，返回其字节数（至少 1）。 */
static size_t next_piece(const qwen3_tok_t *t, const uint8_t *p, size_t n) {
    uint32_t cp;
    int adv = utf8_next(p, n, &cp);

    size_t m = match_contraction(p, n);
    if (m) return m;

    /* [^\r\n\p{L}\p{N}]?\p{L}+ —— 可选一个前导非字母数字非换行符，再吃连续字母 */
    {
        size_t i = 0;
        uint32_t c0 = cp;
        int a0 = adv;
        if (!is_nl(c0) && !is_letter(t, c0) && !is_number(t, c0)) {
            /* 试着把它当前导符号，后面必须紧跟字母 */
            if ((size_t)a0 < n) {
                uint32_t c1; int a1 = utf8_next(p + a0, n - a0, &c1);
                if (is_letter(t, c1)) {
                    i = (size_t)a0 + a1;
                    while (i < n) {
                        uint32_t c; int a = utf8_next(p + i, n - i, &c);
                        if (!is_letter(t, c)) break;
                        i += a;
                    }
                    return i;
                }
            }
        } else if (is_letter(t, c0)) {
            i = (size_t)a0;
            while (i < n) {
                uint32_t c; int a = utf8_next(p + i, n - i, &c);
                if (!is_letter(t, c)) break;
                i += a;
            }
            return i;
        }
    }

    /* \p{N} —— 一次只吃一个数字字符 */
    if (is_number(t, cp)) return (size_t)adv;

    /* ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        size_t i = 0;
        if (cp == ' ' && (size_t)adv < n) {
            uint32_t c1; int a1 = utf8_next(p + adv, n - adv, &c1);
            (void)a1;
            if (!is_space(c1) && !is_letter(t, c1) && !is_number(t, c1))
                i = (size_t)adv;                /* 前导空格归入本片段 */
        }
        uint32_t c; int a;
        size_t start = i;
        while (i < n) {
            a = utf8_next(p + i, n - i, &c);
            if (is_space(c) || is_letter(t, c) || is_number(t, c)) break;
            i += a;
        }
        if (i > start) {
            while (i < n) {                     /* 尾随的 \r\n */
                a = utf8_next(p + i, n - i, &c);
                if (!is_nl(c)) break;
                i += a;
            }
            return i;
        }
    }

    /* \s*[\r\n]+ —— 空白后跟至少一个换行 */
    if (is_space(cp)) {
        size_t i = 0, last_nl = 0;
        while (i < n) {
            uint32_t c; int a = utf8_next(p + i, n - i, &c);
            if (!is_space(c)) break;
            i += a;
            if (is_nl(c)) last_nl = i;
        }
        if (last_nl > 0) return last_nl;

        /* \s+(?!\S) —— 若后面没有非空白字符，整段空白算一个片段 */
        if (i >= n) return i;
        /* \s+ 但要给后面的非空白留出一个前导空格：
         * HF 的行为是 " abc" 里的空格归到后面那段，故此处最多吃到剩一个空格 */
        return (i > 1) ? i - 1 : i;
    }

    return (size_t)adv;    /* 兜底：单个字符成片段 */
}

/* ==================== 对外接口 ==================== */

int qwen3_tok_encode(const qwen3_tok_t *t, const char *text, size_t len,
                     int *out, int max_out) {
    const uint8_t *p = (const uint8_t *)text;
    int total = 0;
    size_t i = 0;

    while (i < len) {
        /* ① 特殊 token 整体匹配，不参与 BPE。specials 已按长度降序排列。 */
        int hit = 0;
        for (uint32_t s = 0; s < t->n_special; s++) {
            uint32_t sl = t->specials[s].len;
            if (sl <= len - i &&
                memcmp(p + i, t->sdata + t->specials[s].off, sl) == 0) {
                if (total >= max_out) return QWEN3_TOK_ERR_OVERFLOW;
                out[total++] = (int)t->specials[s].id;
                i += sl;
                hit = 1;
                break;
            }
        }
        if (hit) continue;

        /* ② 切出片段  ③④ 对片段做 BPE */
        size_t plen = next_piece(t, p + i, len - i);
        if (plen == 0) plen = 1;
        int got = bpe_encode(t, p + i, plen, out + total, max_out - total);
        if (got < 0) return got;
        total += got;
        i += plen;
    }
    return total;
}

int qwen3_tok_decode(const qwen3_tok_t *t, const int *ids, int n,
                     char *out, size_t max_out) {
    size_t w = 0;
    for (int i = 0; i < n; i++) {
        size_t l;
        const uint8_t *s = qwen3_tok_piece(t, ids[i], &l);
        if (w + l >= max_out) return QWEN3_TOK_ERR_OVERFLOW;
        memcpy(out + w, s, l);
        w += l;
    }
    out[w] = '\0';
    return (int)w;
}

const char *qwen3_tok_strerror(int err) {
    switch (err) {
    case QWEN3_TOK_OK:            return "ok";
    case QWEN3_TOK_ERR_MAGIC:     return "bad magic (不是 QW3T 文件)";
    case QWEN3_TOK_ERR_VERSION:   return "版本不符";
    case QWEN3_TOK_ERR_SIZE:      return "文件长度不符";
    case QWEN3_TOK_ERR_OOM:       return "scratch 不足以建 merges 哈希表";
    case QWEN3_TOK_ERR_OVERFLOW:  return "token 数超过上限";
    default:                      return "unknown";
    }
}
