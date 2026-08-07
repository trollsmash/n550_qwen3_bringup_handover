# Phase 1/3: PC 上的参考实现与 kernel 开发
#
#   make                构建（默认 scalar kernel）
#   make KERNEL=tiled   构建（128×32×128 分块 kernel）
#   make test           跑 kernel 单元测试（秒级，不需要模型权重）
#   make check          纪律检查：核心文件不得依赖 libc
#   make e2e            端到端验证（需要 golden/ 数据）

CC      ?= gcc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wpedantic -Wshadow
LDFLAGS ?= -lm
BUILD   := ../build

# 可替换的 GEMM 实现：scalar / tiled / ame（占 99% 计算量）
KERNEL  ?= scalar
KSRC    := kernels_$(KERNEL).c

# 可替换的其余算子实现：scalar / rvv（剩下的 1%）
# x86 上只有 scalar 可用，rvv 需交叉编译（见 tools/build_riscv.sh）
OPS     ?= scalar
OSRC    := ops_$(OPS).c

TARGET  := $(BUILD)/qwen3
TESTBIN := $(BUILD)/test_kernels

CORE := qwen3.c tokenizer.c $(KSRC) $(OSRC)
OBJ  := $(CORE:%.c=$(BUILD)/%.o) $(BUILD)/main.o

.PHONY: all run test check e2e clean

all: $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)
	@echo "== 已构建 $(TARGET)  (GEMM kernel: $(KERNEL)) =="

# kernel 单测：不链接 main.c，秒级反馈。写 AME kernel 时先让它过。
$(TESTBIN): $(CORE:%.c=$(BUILD)/%.o) $(BUILD)/test_kernels.o
	$(CC) $(BUILD)/$(KSRC:%.c=%.o) $(BUILD)/$(OSRC:%.c=%.o) \
	      $(BUILD)/qwen3.o $(BUILD)/test_kernels.o -o $@ $(LDFLAGS)

test: $(TESTBIN)
	@$(TESTBIN)

run: $(TARGET)
	cd .. && ./build/qwen3

e2e: $(TARGET)
	cd .. && python tools/05_e2e_test.py

# 纪律检查：qwen3.c 与 kernel 都要进 bare-metal，不能依赖 stdio/stdlib/malloc。
# -ffreestanding 下若引用了 libc 符号，这里就会暴露。
check: | $(BUILD)
	@echo "== freestanding 编译 + 未定义符号检查 =="
	@for f in qwen3.c $(KSRC); do \
	    $(CC) $(CFLAGS) -ffreestanding -c $$f -o $(BUILD)/fs_check.o || exit 1; \
	    echo "-- $$f --"; \
	    nm -u $(BUILD)/fs_check.o | grep -v '^\s*$$' || echo "   (无未定义符号)"; \
	done
	@echo "== OK: 允许的依赖仅 memcpy/memset 与 libm 的 sqrtf/expf/sinf/cosf/powf =="

# 验证 bare-metal 入口同样零 libc 依赖。
# 这里只做编译期检查（未交叉编译），确认它不引用任何 libc 符号。
baremetal-check: | $(BUILD)
	@echo "== bare-metal 入口的 libc 依赖检查 =="
	@$(CC) $(CFLAGS) -ffreestanding -c main_baremetal.c -o $(BUILD)/bm.o
	@echo "-- main_baremetal.c 的未定义符号（应只有 qwen3_* 与 libm）--"
	@nm -u $(BUILD)/bm.o
	@echo "== 对照：PC 版 main.c --"
	@$(CC) $(CFLAGS) -c main.c -o $(BUILD)/pc.o 2>/dev/null
	@nm -u $(BUILD)/pc.o | grep -vE 'qwen3_' | head -12
	@echo "== 上面那些 open/mmap/malloc/printf 正是 bare-metal 没有的东西 =="

clean:
	rm -rf $(BUILD)
