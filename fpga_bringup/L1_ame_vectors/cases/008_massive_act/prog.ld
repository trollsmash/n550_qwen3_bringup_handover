/* 自动生成 —— case 008 massive_act
 * 各段地址与 meta.json 的 segments 一致；改这里就要同步改那里。
 */
ENTRY(_start)
SECTIONS
{
    . = 0x80000000;
    .text : { *(.text.init) *(.text*) }

    . = 0x80001000;
    .data.a : { *(.data.a) }

    . = 0x80050000;
    .data.b : { *(.data.b) }

    . = 0x800A0000;
    .data.c : { *(.data.c) }

    . = 0x800B0000;
    .data.ref : { *(.data.ref) }

    /DISCARD/ : { *(.comment) *(.riscv.attributes) }
}
