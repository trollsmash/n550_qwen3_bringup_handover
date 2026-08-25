/* Qwen3-0.6B 推理核心 —— 标量参考实现。
 *
 * 纪律：本文件不 include stdio/stdlib，不调用 malloc。
 *       它必须能原样编译进 bare-metal 固件。
 */
/* 本文件不依赖任何外部库：GEMM 走 kernels.h，其余算子走 ops.h，
 * libm 的依赖（sqrtf/expf/sinf/cosf/powf）全部收敛在 ops_*.c 里。 */
#include "kernels.h"
#include "ops.h"
#include "qwen3.h"

/* ==================== arena ==================== */

void arena_init(arena_t *a, void *mem, size_t size) {
    a->base = (uint8_t *)mem;
    a->size = size;
    a->used = 0;
}

void *arena_alloc(arena_t *a, size_t nbytes, size_t align) {
    size_t p = (a->used + align - 1) & ~(align - 1);
    if (p + nbytes > a->size) return 0;
    a->used = p + nbytes;
    return a->base + p;
}

/* ==================== 权重文件解析 ====================
 *
 * 文件布局: [256B header][张量0][pad][张量1][pad]...
 * 每个张量起始按 QW3M_ALIGN(128) 对齐。张量顺序见 qwen3_config.h 末尾注释。
 */

/* 读一个小端 int32，不假设宿主字节序也不做未对齐访问。 */
static int32_t rd_i32(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static float rd_f32(const uint8_t *p) {
    uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* 顺序游标：按导出顺序依次切出每个张量的指针。 */
typedef struct {
    const uint8_t *base;
    size_t         size;
    size_t         pos;
    /* 张量对齐粒度取自 header 而非编译期常量：tile-major 布局下它是 4096
     * （让每个 8 KB 的 tile 都落在 4 KB 边界上，AXI burst 才不会被切三段），
     * 行优先下仍是 128。写死常量会让所有张量偏移算错。 */
    size_t         align;
    int            err;
} cursor_t;

/* 只认行优先的 kernel，两个入口本就等价。做成弱符号让 AME kernel 覆盖。 */
__attribute__((weak))
void qwen3_gemm_row(float *c, const float *a, const uint16_t *b,
                    int M, int K, int N) {
    qwen3_gemm(c, a, b, M, K, N);
}

/* 默认实现：只认行优先。做成弱符号，好让认识其它布局的 kernel 覆盖它 ——
 * 目前只有 kernels_ame.c 提供强符号。 */
__attribute__((weak))
int qwen3_set_weight_layout(int layout, int tile_n, int tile_k) {
    (void)tile_n; (void)tile_k;
    return layout == QW3M_LAYOUT_ROW ? 0 : -1;
}

static const uint16_t *cur_take(cursor_t *c, size_t n_elem) {
    if (c->err) return 0;
    size_t p = (c->pos + c->align - 1) & ~(c->align - 1);
    size_t nbytes = n_elem * sizeof(uint16_t);
    if (p + nbytes > c->size) {
        c->err = QWEN3_ERR_SIZE;
        return 0;
    }
    c->pos = p + nbytes;
    return (const uint16_t *)(c->base + p);
}

int qwen3_init(qwen3_t *m, const void *blob, size_t blob_size,
               void *scratch, size_t scratch_size) {
    const uint8_t *b = (const uint8_t *)blob;

    memset(m, 0, sizeof *m);

    if (blob_size < QW3M_HEADER_BYTES) return QWEN3_ERR_SIZE;
    if (b[0] != 'Q' || b[1] != 'W' || b[2] != '3' || b[3] != 'M')
        return QWEN3_ERR_MAGIC;
    if (rd_i32(b + 4) != QW3M_VERSION) return QWEN3_ERR_VERSION;
    if (rd_i32(b + 44) != QW3M_DTYPE_BF16) return QWEN3_ERR_DTYPE;

    /* header 里的形状必须与编译期常量一致，否则说明用错了权重文件。 */
    if (rd_i32(b + 8)  != QWEN3_HIDDEN_SIZE)       return QWEN3_ERR_SHAPE;
    if (rd_i32(b + 12) != QWEN3_INTERMEDIATE_SIZE) return QWEN3_ERR_SHAPE;
    if (rd_i32(b + 16) != QWEN3_N_LAYERS)          return QWEN3_ERR_SHAPE;
    if (rd_i32(b + 20) != QWEN3_N_HEADS)           return QWEN3_ERR_SHAPE;
    if (rd_i32(b + 24) != QWEN3_N_KV_HEADS)        return QWEN3_ERR_SHAPE;
    if (rd_i32(b + 28) != QWEN3_HEAD_DIM)          return QWEN3_ERR_SHAPE;
    if (rd_i32(b + 32) != QWEN3_VOCAB_SIZE)        return QWEN3_ERR_SHAPE;
    /* 对齐粒度不再要求等于编译期常量：tile-major 用 4096，行优先用 128。
     * 只校验它是 2 的幂且不小于默认值，具体值交给 cursor 使用。 */
    const int align = rd_i32(b + 72);
    if (align < QW3M_ALIGN || (align & (align - 1)) != 0)
        return QWEN3_ERR_SHAPE;

    /* 布局必须让当前 kernel 认得，否则算出的每个数都取自错误的位置。 */
    const int layout = rd_i32(b + 76);
    if (qwen3_set_weight_layout(layout, rd_i32(b + 80), rd_i32(b + 84)) != 0)
        return QWEN3_ERR_LAYOUT;

    /* 浮点超参也校验一下，防止用了别的模型导出的同版本文件。 */
    if (rd_f32(b + 48) != QWEN3_ROPE_THETA)        return QWEN3_ERR_SHAPE;
    if (rd_f32(b + 52) != QWEN3_RMS_NORM_EPS)      return QWEN3_ERR_SHAPE;

    m->n_layers = QWEN3_N_LAYERS;

    /* ---- 切权重指针 ---- */
    cursor_t c = { b, blob_size, (size_t)rd_i32(b + 68), (size_t)align, 0 };

    m->w.embed_tokens = cur_take(&c, (size_t)QWEN3_VOCAB_SIZE * QWEN3_HIDDEN_SIZE);
    for (int l = 0; l < QWEN3_N_LAYERS; l++) {
        qwen3_layer_w_t *w = &m->w.layers[l];
        w->input_layernorm     = cur_take(&c, QWEN3_HIDDEN_SIZE);
        w->wq                  = cur_take(&c, (size_t)QWEN3_Q_DIM  * QWEN3_HIDDEN_SIZE);
        w->wk                  = cur_take(&c, (size_t)QWEN3_KV_DIM * QWEN3_HIDDEN_SIZE);
        w->wv                  = cur_take(&c, (size_t)QWEN3_KV_DIM * QWEN3_HIDDEN_SIZE);
        w->q_norm              = cur_take(&c, QWEN3_HEAD_DIM);
        w->k_norm              = cur_take(&c, QWEN3_HEAD_DIM);
        w->wo                  = cur_take(&c, (size_t)QWEN3_HIDDEN_SIZE * QWEN3_Q_DIM);
        w->post_attn_layernorm = cur_take(&c, QWEN3_HIDDEN_SIZE);
        w->wgate               = cur_take(&c, (size_t)QWEN3_INTERMEDIATE_SIZE * QWEN3_HIDDEN_SIZE);
        w->wup                 = cur_take(&c, (size_t)QWEN3_INTERMEDIATE_SIZE * QWEN3_HIDDEN_SIZE);
        w->wdown               = cur_take(&c, (size_t)QWEN3_HIDDEN_SIZE * QWEN3_INTERMEDIATE_SIZE);
    }
    m->w.final_norm = cur_take(&c, QWEN3_HIDDEN_SIZE);
    if (c.err) return c.err;

    /* 全部张量切完后应恰好用尽文件（末尾无对齐填充）。 */
    if (c.pos != blob_size) return QWEN3_ERR_SIZE;

    /* ---- 分配激活缓冲 ---- */
    arena_init(&m->arena, scratch, scratch_size);
    qwen3_state_t *s = &m->s;
#define ALLOC(field, n)                                                    \
    do {                                                                   \
        s->field = (float *)arena_alloc(&m->arena, (size_t)(n) * sizeof(float), 64); \
        if (!s->field) return QWEN3_ERR_OOM;                               \
    } while (0)

    /* 激活按 [MAX_BATCH][...] 分配，行主序 —— 正好是 GEMM 的 A[M,K] 布局。
     * 顺序必须与 qwen3.h 的 QWEN3_SCRATCH_BYTES 一致。 */
    ALLOC(x,        QWEN3_B * QWEN3_HIDDEN_SIZE);
    ALLOC(xb,       QWEN3_B * QWEN3_HIDDEN_SIZE);
    ALLOC(xb2,      QWEN3_B * QWEN3_HIDDEN_SIZE);
    ALLOC(q,        QWEN3_B * QWEN3_Q_DIM);
    ALLOC(k,        QWEN3_B * QWEN3_KV_DIM);
    ALLOC(v,        QWEN3_B * QWEN3_KV_DIM);
    ALLOC(attn_out, QWEN3_B * QWEN3_Q_DIM);
    ALLOC(att,      (size_t)QWEN3_N_HEADS * QWEN3_MAX_SEQ);
    ALLOC(hb,       QWEN3_B * QWEN3_INTERMEDIATE_SIZE);
    ALLOC(hb2,      QWEN3_B * QWEN3_INTERMEDIATE_SIZE);
    ALLOC(logits,   QWEN3_VOCAB_SIZE);
    ALLOC(kcache, (size_t)QWEN3_N_LAYERS * QWEN3_MAX_SEQ * QWEN3_KV_DIM);
    ALLOC(vcache, (size_t)QWEN3_N_LAYERS * QWEN3_MAX_SEQ * QWEN3_KV_DIM);
#undef ALLOC

    m->loaded = 1;
    return QWEN3_OK;
}

size_t qwen3_scratch_bytes(void) {
    /* 单一来源见 qwen3.h 的 QWEN3_SCRATCH_BYTES。
     * 不要在此重新推导 —— 那正是曾经与 main.c 失同步的原因。 */
    return QWEN3_SCRATCH_BYTES;
}

const char *qwen3_strerror(int err) {
    switch (err) {
    case QWEN3_OK:           return "ok";
    case QWEN3_ERR_MAGIC:    return "bad magic (不是 QW3M 权重文件)";
    case QWEN3_ERR_VERSION:  return "版本不符";
    case QWEN3_ERR_DTYPE:    return "dtype 不是 BF16";
    case QWEN3_ERR_SHAPE:    return "形状/超参与编译期常量不符 (权重文件与 qwen3_config.h 不匹配)";
    case QWEN3_ERR_SIZE:     return "文件长度不符";
    case QWEN3_ERR_OOM:      return "scratch 空间不足";
    case QWEN3_ERR_LAYOUT:   return "权重布局本 kernel 不支持 "
                                    "(tile-major 的权重只能配 AME kernel，"
                                    "行优先的权重请用 --layout row 重新导出)";
    default:                 return "unknown";
    }
}

/* ==================== 算子 ====================
 *
 * 标量参考实现，刻意写得直白：这是判定"算法对不对"的基准，
 * 一切 RVV/AME 优化版本都必须与它逐层对齐。永远不要为了性能改动本节。
 *
 * 唯一的外部依赖是 <math.h> 的 5 个函数：
 *   sqrtf expf sinf cosf powf
 * bare-metal 若无 libm，需自行提供（或把 RoPE 的 sin/cos 改成离线查表）。
 */

void qwen3_set_dump(qwen3_t *m, qwen3_dump_fn fn, void *ctx) {
    m->dump = fn;
    m->dump_ctx = ctx;
}

#define DUMP(m, layer, name, ptr, n)                                          \
    do {                                                                      \
        if ((m)->dump) (m)->dump((m)->dump_ctx, (layer), (name), (ptr), (size_t)(n)); \
    } while (0)

/* -DQWEN3_BF16_ROUND：在每个算子出口把结果舍入到 BF16，
 * 复现 PyTorch BF16 模型的数值行为（它每个算子的输出都是 BF16）。
 * 用途有二：
 *   1. 验证"误差只来自精度而非逻辑" —— 开启后误差应塌到 BF16 量级；
 *   2. Phase 2 的 BF16 数值适配，也更接近 FPGA 上 AME 的实际行为。
 * 默认关闭：标量参考实现保持全 FP32，用于判定算法本身是否正确。 */
#ifdef QWEN3_BF16_ROUND
static void round_bf16(float *v, int n) {
    for (int i = 0; i < n; i++) v[i] = bf16_to_f32(f32_to_bf16(v[i]));
}
#else
#define round_bf16(v, n) ((void)(v), (void)(n))
#endif

/* 算子出口：先按需舍入，再 dump。 */
#define EMIT(m, layer, name, ptr, n)                                          \
    do {                                                                      \
        round_bf16((ptr), (int)(n));                                          \
        DUMP((m), (layer), (name), (ptr), (n));                               \
    } while (0)

/* 查 embedding 表：BF16 权重 -> FP32 激活。是索引 + 转换，不是矩阵乘。 */
void qwen3_embed(const qwen3_t *m, int token, float *out) {
    const uint16_t *row = m->w.embed_tokens + (size_t)token * QWEN3_HIDDEN_SIZE;
    qwen3_op_bf16_to_f32(out, row, QWEN3_HIDDEN_SIZE);
}

/* 批量化之后 forward 直接调 qwen3_gemm（M = n_token），
 * 原先那个 M=1 的 matmul 包装已无必要，删去以免有人误用。
 * ★ 全模型 99% 的计算量在 qwen3_gemm 里，换实现只需换链接的 kernels_*.c。 */

/* QK-Norm（Qwen3 特有）：对每个头的 head_dim 维做 RMSNorm，各头共享同一份权重。
 * ★ llama2.c 一类参考实现没有这个算子，漏掉不会崩，但输出是垃圾。 */
static void qk_norm(float *v, const uint16_t *w, int n_heads) {
    for (int h = 0; h < n_heads; h++) {
        float *p = v + (size_t)h * QWEN3_HEAD_DIM;
        qwen3_op_rmsnorm(p, p, w, QWEN3_HEAD_DIM);
    }
}

/* 1/sqrt(head_dim)，head_dim=128。写成字面量而非 sqrtf(128)，
 * 这样本文件不必为一个编译期常数引入 math.h。 */
#define QWEN3_ATTN_SCALE 0.08838834764831845f

/* GQA + causal attention，批量版。
 * ★ 第 h 个 Q 头用第 h/KV_GROUP 个 KV 头 —— 是整除，不是取模。
 *
 * 逐 query 位置处理，每个位置只看 0..pos（causal），因此不需要显式 mask 矩阵。
 * 没有做 N×N 的矩阵化：prefill 阶段 GEMM 占绝对主导，attention 占比很小，
 * 正确性优先。若日后成为瓶颈再改。 */
static void attention(qwen3_t *m, int layer, int n_token, int pos0) {
    qwen3_state_t *s = &m->s;
    const size_t lstride = (size_t)QWEN3_MAX_SEQ * QWEN3_KV_DIM;
    const float *kc = s->kcache + (size_t)layer * lstride;
    const float *vc = s->vcache + (size_t)layer * lstride;

    for (int i = 0; i < n_token; i++) {
        const int pos = pos0 + i;
        const float *qrow = s->q        + (size_t)i * QWEN3_Q_DIM;
        float       *orow = s->attn_out + (size_t)i * QWEN3_Q_DIM;

        for (int h = 0; h < QWEN3_N_HEADS; h++) {
            const float *qh = qrow + (size_t)h * QWEN3_HEAD_DIM;
            const int kvh = h / QWEN3_KV_GROUP;
            float *att = s->att + (size_t)h * QWEN3_MAX_SEQ;

            for (int t = 0; t <= pos; t++) {
                const float *kh = kc + (size_t)t * QWEN3_KV_DIM
                                     + (size_t)kvh * QWEN3_HEAD_DIM;
                att[t] = qwen3_op_dot(qh, kh, QWEN3_HEAD_DIM) * QWEN3_ATTN_SCALE;
            }
            qwen3_op_softmax(att, pos + 1);

            float *oh = orow + (size_t)h * QWEN3_HEAD_DIM;
            for (int j = 0; j < QWEN3_HEAD_DIM; j++) oh[j] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float *vh = vc + (size_t)t * QWEN3_KV_DIM
                                     + (size_t)kvh * QWEN3_HEAD_DIM;
                qwen3_op_axpy(oh, att[t], vh, QWEN3_HEAD_DIM);
            }
        }
    }
}

/* 逐行调用某个作用于单行的算子。批量化后大量算子都是这个模式：
 * GEMM 天然吃 [M,K]，而 norm/rope/silu 这些是按行独立的。 */
#define EACH_ROW(n, stride, body)                                             \
    do { for (int _r = 0; _r < (n); _r++) { const size_t _o = (size_t)_r * (stride); body } } while (0)

/* 进度回调的默认实现：什么都不做。
 *
 * 平台层若定义了同名强符号就会覆盖它。这样做的理由是裸机上没有别的观测
 * 手段 —— 没有 gdb、没有 printf、没有性能计数器，程序一旦跑进 28 层循环
 * 就是个黑盒，卡死和"还在算"从外面看完全一样。
 *
 * 用弱符号而不是条件编译或函数指针：x86 与 user-mode 平台层不实现它，
 * 链接进来的就是这个空函数，对现有两条路径零影响、零开销。 */
__attribute__((weak)) void qwen3_on_layer(int layer, int n_layers) {
    (void)layer; (void)n_layers;
}

void qwen3_forward_batch(qwen3_t *m, const int *tokens, int n_token, int pos0) {
    qwen3_state_t *s = &m->s;
    const size_t lstride = (size_t)QWEN3_MAX_SEQ * QWEN3_KV_DIM;
    const int NT = n_token;
    const size_t H = QWEN3_HIDDEN_SIZE, QD = QWEN3_Q_DIM;
    const size_t KV = QWEN3_KV_DIM, IN = QWEN3_INTERMEDIATE_SIZE;

    if (NT <= 0 || NT > QWEN3_MAX_BATCH) return;

    for (int t = 0; t < NT; t++) qwen3_embed(m, tokens[t], s->x + (size_t)t * H);
    EMIT(m, -1, "embed_out", s->x, (size_t)NT * H);

    for (int l = 0; l < QWEN3_N_LAYERS; l++) {
        const qwen3_layer_w_t *w = &m->w.layers[l];
        qwen3_on_layer(l, QWEN3_N_LAYERS);

        /* ---- Attention ---- */
        EACH_ROW(NT, H, qwen3_op_rmsnorm(s->xb + _o, s->x + _o,
                                         w->input_layernorm, (int)H););
        EMIT(m, l, "input_layernorm_out", s->xb, (size_t)NT * H);

        /* 顺序与黄金数据一致：q_proj -> q_norm -> k_proj -> k_norm -> v_proj。
         * ★ 这几个 GEMM 的 M 就是 NT —— prefill 时填满 AME tile 的关键。 */
        qwen3_gemm(s->q, s->xb, w->wq, NT, (int)H, (int)QD);
        EMIT(m, l, "q_proj_out", s->q, (size_t)NT * QD);
        EACH_ROW(NT, QD, qk_norm(s->q + _o, w->q_norm, QWEN3_N_HEADS););
        EMIT(m, l, "q_norm_out", s->q, (size_t)NT * QD);

        qwen3_gemm(s->k, s->xb, w->wk, NT, (int)H, (int)KV);
        EMIT(m, l, "k_proj_out", s->k, (size_t)NT * KV);
        EACH_ROW(NT, KV, qk_norm(s->k + _o, w->k_norm, QWEN3_N_KV_HEADS););
        EMIT(m, l, "k_norm_out", s->k, (size_t)NT * KV);

        qwen3_gemm(s->v, s->xb, w->wv, NT, (int)H, (int)KV);
        EMIT(m, l, "v_proj_out", s->v, (size_t)NT * KV);

        /* RoPE 的角度依赖各 token 自己的位置，故逐行传 pos0+r。 */
        EACH_ROW(NT, QD, qwen3_op_rope(s->q + _o, QWEN3_N_HEADS, pos0 + _r););
        EMIT(m, l, "q_rope_out", s->q, (size_t)NT * QD);
        EACH_ROW(NT, KV, qwen3_op_rope(s->k + _o, QWEN3_N_KV_HEADS, pos0 + _r););
        EMIT(m, l, "k_rope_out", s->k, (size_t)NT * KV);

        /* 写 KV cache —— decode 阶段全靠它，写偏一格前几个 token 还正常，
         * 后面才乱，是最难查的一类 bug。 */
        for (int t = 0; t < NT; t++) {
            const size_t dst = (size_t)l * lstride + (size_t)(pos0 + t) * KV;
            memcpy(s->kcache + dst, s->k + (size_t)t * KV, KV * sizeof(float));
            memcpy(s->vcache + dst, s->v + (size_t)t * KV, KV * sizeof(float));
        }

        attention(m, l, NT, pos0);
        EMIT(m, l, "attn_out", s->attn_out, (size_t)NT * QD);

        qwen3_gemm(s->xb2, s->attn_out, w->wo, NT, (int)QD, (int)H);
        EMIT(m, l, "o_proj_out", s->xb2, (size_t)NT * H);

        EACH_ROW(NT, H, qwen3_op_residual_add(s->x + _o, s->xb2 + _o, (int)H););
        EMIT(m, l, "x_after_attn_residual", s->x, (size_t)NT * H);

        /* ---- MLP (SwiGLU) ---- */
        EACH_ROW(NT, H, qwen3_op_rmsnorm(s->xb + _o, s->x + _o,
                                         w->post_attn_layernorm, (int)H););
        EMIT(m, l, "post_attn_norm_out", s->xb, (size_t)NT * H);

        qwen3_gemm(s->hb,  s->xb, w->wgate, NT, (int)H, (int)IN);
        EMIT(m, l, "gate_proj_out", s->hb, (size_t)NT * IN);
        qwen3_gemm(s->hb2, s->xb, w->wup,   NT, (int)H, (int)IN);
        EMIT(m, l, "up_proj_out", s->hb2, (size_t)NT * IN);

        /* SiLU 只作用于 gate，再与 up 逐元素相乘。整块连续，无需逐行。 */
        qwen3_op_silu_mul(s->hb, s->hb2, (int)((size_t)NT * IN));
        EMIT(m, l, "silu_gate_mul_up", s->hb, (size_t)NT * IN);

        qwen3_gemm(s->xb2, s->hb, w->wdown, NT, (int)IN, (int)H);
        EMIT(m, l, "down_proj_out", s->xb2, (size_t)NT * H);

        qwen3_op_residual_add(s->x, s->xb2, (int)((size_t)NT * H));
        EMIT(m, l, "layer_out", s->x, (size_t)NT * H);
    }

    EACH_ROW(NT, H, qwen3_op_rmsnorm(s->x + _o, s->x + _o,
                                     m->w.final_norm, (int)H););
    EMIT(m, -1, "final_norm_out", s->x, (size_t)NT * H);

    /* lm_head 复用 embed_tokens（tie_word_embeddings=1）。
     * ★ 只算最后一个 token：生成只需要它，而这是全模型最大的一次 GEMM
     *   （N=151936），算全部 NT 行纯属浪费。
     *   代价是 dump 出的 logits 只有 1 行，而黄金数据是 NT 行 ——
     *   tools/04_compare.py 对此做了特殊处理（取黄金数据的最后一行比对）。 */
    /* ★ 必须走行优先入口：embed_tokens 从不重排（见 kernels.h 的说明）。 */
    qwen3_gemm_row(s->logits, s->x + (size_t)(NT - 1) * H, m->w.embed_tokens,
               1, (int)H, QWEN3_VOCAB_SIZE);
    EMIT(m, -1, "logits", s->logits, QWEN3_VOCAB_SIZE);
}

void qwen3_forward(qwen3_t *m, int token, int pos) {
    qwen3_forward_batch(m, &token, 1, pos);
}

int qwen3_argmax(const float *v, int n) {
    return qwen3_op_argmax(v, n);
}
