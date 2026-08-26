/* freestanding 环境的最小 string 实现。
 *
 * C 标准允许编译器在任何优化级别自行生成 memcpy/memset/memmove/memcmp 调用
 * （即使源码里没写），所以 -nostdlib 时必须自行提供，否则链接失败。
 * x86 上这几个由 glibc 兜底，交叉编译到裸机就暴露了。
 *
 * 注意 optimize("no-tree-loop-distribute-patterns")：
 * 没有它，GCC 会把下面的复制循环识别成 memcpy 再生成一次 memcpy 调用，
 * 结果是函数调用自己 —— 无限递归直到爆栈。
 *
 * 这里是朴素标量实现，正确性优先。真机上若成为瓶颈，可换 RVV 版本，
 * 但 memcpy 在本项目里只用于 embedding 查表与 KV cache 写入，占比很小。
 */
#include <stddef.h>
#include <stdint.h>

#define NO_LOOP_IDIOM __attribute__((optimize("no-tree-loop-distribute-patterns")))

/* ★ 诊断期版本：退回最朴素的逐字节复制。
 *
 * 原本有一条「双方同余就按 8 字节走」的快路径，逻辑本身没有问题
 * （同余 + 先补齐，保证 d 与 s 都落在 8 字节边界上）。改成这样是为了
 * 排除嫌疑：真机上两次故障都落在 memcpy 前 32 字节内，而这个函数
 * 每 token 只调用约 57 次，远少于 GEMM 的指令数 —— 指令占比与故障
 * 分布不成比例。把实现换掉会改变函数大小与后续代码的布局，
 * 若故障随之转移或消失，就说明问题跟地址相关而非跟逻辑相关。
 *
 * 代价：KV cache 每 token 约 229 KB 的搬运慢 8 倍，等效多花约 1.8 MB
 * 的访存 —— 相对每 token 必读的 1.11 GB 权重可以忽略。
 * 结论明确之后可以改回快路径版本。 */
NO_LOOP_IDIOM
void *memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    while (n--) *d++ = *s++;
    return dst;
}

NO_LOOP_IDIOM
void *memset(void *dst, int c, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    uint8_t v = (uint8_t)c;
    while (n && ((uintptr_t)d & 7u)) { *d++ = v; n--; }
    uint64_t w = 0x0101010101010101ull * v;
    uint64_t *d8 = (uint64_t *)d;
    while (n >= 8) { *d8++ = w; n -= 8; }
    d = (uint8_t *)d8;
    while (n--) *d++ = v;
    return dst;
}

NO_LOOP_IDIOM
void *memmove(void *dst, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    if (d == s || n == 0) return dst;
    if (d < s) { while (n--) *d++ = *s++; }
    else       { d += n; s += n; while (n--) *--d = *--s; }
    return dst;
}

NO_LOOP_IDIOM
int memcmp(const void *a, const void *b, size_t n) {
    const uint8_t *p = (const uint8_t *)a, *q = (const uint8_t *)b;
    while (n--) { if (*p != *q) return (int)*p - (int)*q; p++; q++; }
    return 0;
}

NO_LOOP_IDIOM
size_t strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return (size_t)(p - s);
}
