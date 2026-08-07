/* 自动生成 —— case 010 sync_sensitivity
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

    /* 以下仅混合 case 使用。a_src 与 a/c 刻意隔开 64 KB 以上：
     * 009b 用的是全量失效，各区贴太近时很难分辨"同步没生效"
     * 与"相邻区域被误伤"。 */
    . = 0x800D0000;
    .data.asrc : { *(.data.asrc) }

    . = 0x80160004;
    .data.xorref : { *(.data.xorref) }

    /DISCARD/ : { *(.comment) *(.riscv.attributes) }
}
