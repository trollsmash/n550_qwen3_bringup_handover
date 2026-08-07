/* freestanding 环境下使用 newlib libm 所需的最小桩。
 *
 * newlib 的 libm 会引用 __errno() 取 errno 的地址 —— 那本是 libc 的
 * reentrancy 机制（每线程一份 errno）。裸机上不需要真正的 errno 语义，
 * 给个静态变量即可满足链接。
 *
 * 有了它就能直接用 newlib 的 sqrtf/expf/sinf/cosf/powf，不必自己实现
 * （实测 5 个函数取值全部正确）。
 *
 * 凡是链接 -lm 的目标都要带上本文件：ops_scalar.c 用到 sqrtf/expf/sinf/cosf/powf。
 */
static int g_errno;

int *__errno(void) { return &g_errno; }
