/* 自动生成，请勿手改 —— 由 tools/02_export_weights.py 生成。
 * 所有常量取自模型仓库的原始 config.json，非估算。
 */
#ifndef QWEN3_CONFIG_H
#define QWEN3_CONFIG_H

/* ---- 模型超参 ---- */
#define QWEN3_HIDDEN_SIZE        1024
#define QWEN3_INTERMEDIATE_SIZE  3072
#define QWEN3_N_LAYERS           28
#define QWEN3_N_HEADS            16
#define QWEN3_N_KV_HEADS         8
#define QWEN3_HEAD_DIM           128   /* 注意: N_HEADS*HEAD_DIM != HIDDEN_SIZE */
#define QWEN3_VOCAB_SIZE         151936
#define QWEN3_MAX_POS            40960
#define QWEN3_ROPE_THETA         1000000.0f
#define QWEN3_RMS_NORM_EPS       1e-06f
#define QWEN3_TIE_EMBEDDINGS     1

/* 每 2 个 Q 头共享 1 个 KV 头 (GQA)。第 h 个 Q 头用第 h/2 个 KV 头。 */
#define QWEN3_KV_GROUP           2

/* 投影层的输出维度（由 head_dim 决定，不能用 hidden_size 推） */
#define QWEN3_Q_DIM              2048   /* N_HEADS    * HEAD_DIM */
#define QWEN3_KV_DIM             1024   /* N_KV_HEADS * HEAD_DIM */

/* ---- 特殊 token ---- */
#define QWEN3_BOS_TOKEN_ID       151643
#define QWEN3_EOS_TOKEN_ID_0     151645
#define QWEN3_EOS_TOKEN_ID_1     151643   /* Qwen3 有两个 EOS，都要判！ */

/* ---- 权重文件格式 (.bin) ---- */
#define QW3M_MAGIC               0x4D335751u  /* "QW3M" 小端 */
#define QW3M_VERSION             1
#define QW3M_HEADER_BYTES        256
#define QW3M_ALIGN               128   /* 每个张量按此对齐 = RVV VLEN(bit)/8 */
#define QW3M_DTYPE_BF16          1
#define QW3M_DTYPE_FP32          2

/* 256 字节定长 header 的字段布局（全部小端） */
typedef struct {
    char  magic[4];          /* +0   "QW3M" */
    int   version;           /* +4  */
    int   hidden_size;       /* +8  */
    int   intermediate_size; /* +12 */
    int   n_layers;          /* +16 */
    int   n_heads;           /* +20 */
    int   n_kv_heads;        /* +24 */
    int   head_dim;          /* +28 */
    int   vocab_size;        /* +32 */
    int   max_pos;           /* +36 */
    int   tie_embeddings;    /* +40 */
    int   dtype_code;        /* +44  1=BF16 2=FP32 */
    float rope_theta;        /* +48 */
    float rms_norm_eps;      /* +52 */
    int   bos_token_id;      /* +56 */
    int   eos_token_id_0;    /* +60 */
    int   eos_token_id_1;    /* +64 */
    int   data_start;        /* +68  权重区起始字节偏移 */
    int   align;             /* +72 */
    char  _pad[180];      /* 填充至 256 字节 */
} qw3m_header_t;

/* 权重区中张量的排列顺序（C 端必须按完全相同顺序解析）:
 *   model.embed_tokens.weight                    [VOCAB, HIDDEN]
 *   for l in 0..N_LAYERS-1:
 *     input_layernorm.weight                     [HIDDEN]
 *     self_attn.q_proj.weight                    [Q_DIM,  HIDDEN]
 *     self_attn.k_proj.weight                    [KV_DIM, HIDDEN]
 *     self_attn.v_proj.weight                    [KV_DIM, HIDDEN]
 *     self_attn.q_norm.weight                    [HEAD_DIM]
 *     self_attn.k_norm.weight                    [HEAD_DIM]
 *     self_attn.o_proj.weight                    [HIDDEN, Q_DIM]
 *     post_attention_layernorm.weight            [HIDDEN]
 *     mlp.gate_proj.weight                       [INTER,  HIDDEN]
 *     mlp.up_proj.weight                         [INTER,  HIDDEN]
 *     mlp.down_proj.weight                       [HIDDEN, INTER]
 *   model.norm.weight                            [HIDDEN]
 * 共 1 + N_LAYERS*11 + 1 = 310 个张量。
 * lm_head 不单独存储（tie_word_embeddings=1），复用 embed_tokens。
 */

#endif /* QWEN3_CONFIG_H */
