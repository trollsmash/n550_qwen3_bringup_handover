/* Linker script for QEMU virt machine.
 *
 * QEMU virt 的 DRAM 从 0x80000000 起，-m 参数决定大小。
 * FPGA 上把 RAM 起始地址和大小改成实际值即可，其余不用动。
 *
 * 注意 KV cache 等静态缓冲在 BSS 里，体量很大（MAX_SEQ=256 时约 59 MiB），
 * 所以 RAM 长度必须给够，且 QEMU 要用 -m 指定足够内存。
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwxa) : ORIGIN = 0x80000000, LENGTH = 3072M
}

_stack_size = 256K;

SECTIONS
{
    .text : {
        KEEP(*(.text.init))     /* _start 必须在最前面 */
        *(.text .text.*)
    } > RAM

    .rodata : {
        . = ALIGN(8);
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > RAM

    .data : {
        . = ALIGN(8);
        *(.data .data.*)
        . = ALIGN(8);
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
    } > RAM

    .bss (NOLOAD) : {
        . = ALIGN(8);
        _bss_start = .;
        *(.sbss .sbss.*)
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(8);
        _bss_end = .;
    } > RAM

    /* 栈放在所有静态数据之后 */
    .stack (NOLOAD) : {
        . = ALIGN(16);
        _stack_bottom = .;
        . += _stack_size;
        _stack_top = .;
    } > RAM

    _end = .;

    /DISCARD/ : { *(.comment) *(.note.*) *(.eh_frame) }
}
