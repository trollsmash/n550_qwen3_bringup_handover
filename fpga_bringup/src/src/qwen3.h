/* Qwen3-0.6B 推理核心 —— bare-metal 可用。
 *
 * 纪律：本头文件与 qwen3.c 不依赖 stdio / stdlib / malloc。
 *       所有平台相关操作（读文件、打印）由调用方在 main.c 里完成。
 */
#ifndef QWEN3_H
#define QWEN3_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>     /* memcpy —— freestanding 环境亦提供 */

#include "qwen3_config.h"

/* 支持的最大序列长度。KV cache 大小与之成正比：
 * 28层 × 2(K,V) × MAX_SEQ × KV_DIM × 4B = MAX_SEQ × 229 KB */
#ifndef QWEN3_MAX_SEQ
#define QWEN3_MAX_SEQ 256
#endif

/* ---------------- BF16 <-> FP32 ----------------
 * BF16 就是 FP32 砍掉低 16 位尾数，指数位完全相同。
 * 转换无需任何浮点指令，纯位操作。 */
static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    /* round-to-nearest-even */
    uint32_t r = 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)((u + r) >> 16);
}

/* ---------------- 静态 arena ----------------
 * bump pointer 分配，只增不减，无碎片，无 free。
 * bare-metal 上比 malloc 更可控，也更容易审计内存用量。 */
typedef struct {
    uint8_t *base;
    size_t   size;
    size_t   used;
} arena_t;

void  arena_init(arena_t *a, void *mem, size_t size);
void *arena_alloc(arena_t *a, size_t nbytes, size_t align);

/* ---------------- scratch 大小（编译期常量）----------------
 * 各平台的入口文件（main.c / main_qemu.c / main_baremetal.c）都用这个宏
 * 来开静态缓冲，qwen3_scratch_bytes() 也返回它 —— 单一来源。
 *
 * 曾经这里是「qwen3.c 里 ALLOC 一份、main.c 里手算一份」，
 * 加了 attn_out 之后两边立刻失同步，表现为 qwen3_init 返回 ERR_OOM。
 * 新增激活缓冲时，只需同时改这里和 qwen3.c 的 ALLOC 列表。 */
#define QWEN3_A(n)  ((((size_t)(n) * sizeof(float) + 63u) & ~(size_t)63u))

#define QWEN3_B ((size_t)QWEN3_MAX_BATCH)

#define QWEN3_SCRATCH_BYTES (                                              \
      QWEN3_A(QWEN3_B * QWEN3_HIDDEN_SIZE)       /* x        */            \
    + QWEN3_A(QWEN3_B * QWEN3_HIDDEN_SIZE)       /* xb       */            \
    + QWEN3_A(QWEN3_B * QWEN3_HIDDEN_SIZE)       /* xb2      */            \
    + QWEN3_A(QWEN3_B * QWEN3_Q_DIM)             /* q        */            \
    + QWEN3_A(QWEN3_B * QWEN3_KV_DIM)            /* k        */            \
    + QWEN3_A(QWEN3_B * QWEN3_KV_DIM)            /* v        */            \
    + QWEN3_A(QWEN3_B * QWEN3_Q_DIM)             /* attn_out */            \
    + QWEN3_A((size_t)QWEN3_N_HEADS * QWEN3_MAX_SEQ)          /* att    */ \
    + QWEN3_A(QWEN3_B * QWEN3_INTERMEDIATE_SIZE) /* hb       */            \
    + QWEN3_A(QWEN3_B * QWEN3_INTERMEDIATE_SIZE) /* hb2      */            \
    + QWEN3_A(QWEN3_VOCAB_SIZE)         /* logits（只留最后一个 token）*/    \
    + QWEN3_A((size_t)QWEN3_N_LAYERS * QWEN3_MAX_SEQ * QWEN3_KV_DIM)       \
    + QWEN3_A((size_t)QWEN3_N_LAYERS * QWEN3_MAX_SEQ * QWEN3_KV_DIM)       \
    + 64                                /* 首次对齐余量 */                  \
)

/* ---------------- 权重 ---------------- */
typedef struct {
    const uint16_t *input_layernorm;      /* [HIDDEN]          */
    const uint16_t *wq;                   /* [Q_DIM,  HIDDEN]  */
    const uint16_t *wk;                   /* [KV_DIM, HIDDEN]  */
    const uint16_t *wv;                   /* [KV_DIM, HIDDEN]  */
    const uint16_t *q_norm;               /* [HEAD_DIM]        */
    const uint16_t *k_norm;               /* [HEAD_DIM]        */
    const uint16_t *wo;                   /* [HIDDEN, Q_DIM]   */
    const uint16_t *post_attn_layernorm;  /* [HIDDEN]          */
    const uint16_t *wgate;                /* [INTER,  HIDDEN]  */
    const uint16_t *wup;                  /* [INTER,  HIDDEN]  */
    const uint16_t *wdown;                /* [HIDDEN, INTER]   */
} qwen3_layer_w_t;

typedef struct {
    const uint16_t *embed_tokens;         /* [VOCAB, HIDDEN]，lm_head 复用之 */
    qwen3_layer_w_t layers[QWEN3_N_LAYERS];
    const uint16_t *final_norm;           /* [HIDDEN] */
} qwen3_weights_t;

/* 一次前向最多处理多少个 token。
 * 取 128 = AME_TILE_M：prefill 分块时恰好填满 AME 的 M 维，
 * 这是让矩阵扩展真正发挥作用的关键（M=1 时利用率只有 1/128）。
 * 代价是各激活缓冲放大 128 倍，约 +7.9 MiB。 */
#ifndef QWEN3_MAX_BATCH
#define QWEN3_MAX_BATCH 128
#endif

/* ---------------- 运行时状态 ----------------
 * 所有激活按 [MAX_BATCH][...] 行主序排布 —— 正好是 GEMM 接口要的 A[M,K]，
 * 无需任何重排即可直接传给 kernel。 */
typedef struct {
    float *x;        /* [B][HIDDEN]  残差流 */
    float *xb;       /* [B][HIDDEN]  norm 输出等临时量 */
    float *xb2;      /* [B][HIDDEN]  */
    float *q;        /* [B][Q_DIM]   */
    float *k;        /* [B][KV_DIM]  */
    float *v;        /* [B][KV_DIM]  */
    float *attn_out; /* [B][Q_DIM]   attention 的输出，o_proj 的输入 */
    float *att;      /* [N_HEADS * MAX_SEQ]  注意力分数（逐 query 位置复用）*/
    float *hb;       /* [B][INTER]   */
    float *hb2;      /* [B][INTER]   */
    float *logits;   /* [VOCAB]      只保留最后一个 token 的 */
    float *kcache;   /* [N_LAYERS][MAX_SEQ][KV_DIM] */
    float *vcache;   /* [N_LAYERS][MAX_SEQ][KV_DIM] */
} qwen3_state_t;

/* 中间激活的 dump 回调。核心代码只负责"在正确的位置调用"，
 * 数据落到哪里由平台层决定：
 *   PC   -> 写文件，交给 tools/04_compare.py 比对
 *   FPGA -> 写 DDR 约定地址，host 经 PCIe 回读后比对
 * layer < 0 表示非层内张量（embed_out / final_norm_out / logits）。 */
typedef void (*qwen3_dump_fn)(void *ctx, int layer, const char *name,
                              const float *data, size_t n);

typedef struct {
    qwen3_weights_t w;
    qwen3_state_t   s;
    arena_t         arena;
    int             n_layers;      /* 由文件 header 决定，应等于宏 */
    int             loaded;
    qwen3_dump_fn   dump;          /* 可为 NULL，此时零开销 */
    void           *dump_ctx;
} qwen3_t;

/* ---------------- API ---------------- */

/* 错误码 */
enum {
    QWEN3_OK = 0,
    QWEN3_ERR_MAGIC = -1,
    QWEN3_ERR_VERSION = -2,
    QWEN3_ERR_DTYPE = -3,
    QWEN3_ERR_SHAPE = -4,
    QWEN3_ERR_SIZE = -5,
    QWEN3_ERR_OOM = -6,
};

/* 从已载入内存的权重 blob 建立权重指针表。
 * blob 必须是 02_export_weights.py 产出的 .bin 全文，且在 m 的生命周期内有效。
 * scratch/scratch_size 用于激活缓冲的 arena。 */
int qwen3_init(qwen3_t *m, const void *blob, size_t blob_size,
               void *scratch, size_t scratch_size);

/* 计算本配置所需的 scratch 字节数（编译期常量表达式亦可，但用函数更清晰）。 */
size_t qwen3_scratch_bytes(void);

const char *qwen3_strerror(int err);

/* ---------------- 算子 ---------------- */

/* embedding 查表：BF16 权重 -> FP32 激活。out 需 [HIDDEN] 个 float。 */
void qwen3_embed(const qwen3_t *m, int token, float *out);

/* 设置 dump 回调；传 NULL 关闭。 */
void qwen3_set_dump(qwen3_t *m, qwen3_dump_fn fn, void *ctx);

/* 批量前向：一次处理 n_token 个 token，它们在序列中的位置是 pos0 .. pos0+n_token-1。
 * n_token 上限为 QWEN3_MAX_BATCH；prefill 时由调用方分块。
 *
 * 这是让 AME 发挥作用的关键路径：GEMM 的 M 维等于 n_token，
 * n_token=128 时恰好填满一个 AME tile；而逐 token 调用时 M=1，利用率仅 1/128。
 *
 * logits 只保留最后一个 token 的 —— 生成时只需要它。 */
void qwen3_forward_batch(qwen3_t *m, const int *tokens, int n_token, int pos0);

/* 单 token 前向，等价于 n_token=1 的批量前向。decode 路径用。
 * 结果写入 m->s.logits[VOCAB]。KV cache 由内部按 pos 维护。 */
void qwen3_forward(qwen3_t *m, int token, int pos);

/* logits 取 argmax。 */
int qwen3_argmax(const float *v, int n);

/* 每进入一层调用一次。qwen3.c 里有个什么都不做的弱符号实现，
 * 平台层定义同名函数即可覆盖。
 *
 * 存在的理由是裸机上没有别的观测手段：没有 gdb、没有 printf、
 * 没有性能计数器，程序跑进 28 层循环后，"卡死"和"还在算"从外面
 * 看起来完全一样。FPGA 上尤其如此 —— 你只有一根串口线。
 * 不实现它的平台（x86 / user-mode）零开销、行为不变。 */
void qwen3_on_layer(int layer, int n_layers);

#endif /* QWEN3_H */
