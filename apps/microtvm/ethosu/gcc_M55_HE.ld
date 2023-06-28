/* This file was ported to work on Alif Semiconductor Ensemble family of devices. */

/* Copyright (C) 2022 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

__STACK_SIZE = 0x00001000;
__HEAP_SIZE  = 0x00001000;
__ROM_BASE = 0x80001000;
__ROM_SIZE = 0x00580000;

MEMORY
{
  ITCM  (rwx) : ORIGIN = 0x00000000, LENGTH = 0x00040000
  DTCM  (rwx) : ORIGIN = 0x20000000, LENGTH = 0x00040000
  SRAM0 (rwx) : ORIGIN = 0x02000000, LENGTH = 0x00400000
  SRAM1 (rwx) : ORIGIN = 0x08000000, LENGTH = 0x00280000
  SRAM6 (rwx) : ORIGIN = 0x62000000, LENGTH = 0x00200000
  SRAM7 (rwx) : ORIGIN = 0x63000000, LENGTH = 0x00080000
  SRAM8 (rwx) : ORIGIN = 0x63100000, LENGTH = 0x00200000
  SRAM9 (rwx) : ORIGIN = 0x64000000, LENGTH = 0x000C0000
  MRAM  (rx)  : ORIGIN = __ROM_BASE, LENGTH = __ROM_SIZE
}

ENTRY(Reset_Handler)

SECTIONS
{
  .startup.at_mram : ALIGN(16)
  {
    KEEP(*(.vectors))

    *startup_M55_*(.text .rodata*)
    *system_M55_*(.text .rodata*)
    *mpu_M55_*(.text .rodata*)

    *(startup_ro_data)
  } > MRAM

  .copy.table : ALIGN(4)
  {
    __copy_table_start__ = .;
    LONG ( LOADADDR(.ARM.extab) )
    LONG ( ADDR(.ARM.extab) )
    LONG ( SIZEOF(.ARM.extab)/4 )
    LONG ( LOADADDR(.ARM.exidx) )
    LONG ( ADDR(.ARM.exidx) )
    LONG ( SIZEOF(.ARM.exidx)/4 )
    LONG ( LOADADDR(.code.at_itcm) )
    LONG ( ADDR(.code.at_itcm) )
    LONG ( SIZEOF(.code.at_itcm)/4 )
    LONG ( LOADADDR(.data.at_dtcm) )
    LONG ( ADDR(.data.at_dtcm) )
    LONG ( SIZEOF(.data.at_dtcm)/4 )
    LONG ( LOADADDR(.weights) )
    LONG ( ADDR(.weights) )
    LONG ( SIZEOF(.weights)/4 )
    __copy_table_end__ = .;
  } > MRAM

  .zero.table : ALIGN(4)
  {
    __zero_table_start__ = .;
    LONG (ADDR(.bss))
    LONG (SIZEOF(.bss)/4)
    LONG (ADDR(.arena))
    LONG (SIZEOF(.arena)/4)
    __zero_table_end__ = .;
  } > MRAM

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > ITCM AT> MRAM

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > ITCM AT> MRAM
  __exidx_end = .;

  .vectors.at_itcm (NOLOAD) : ALIGN(2048)
  {
    *(.bss.noinit.ram_vectors)
  } > ITCM

  .code.at_itcm : ALIGN(8)
  {
    *(.text*)
    *(fast_code)
  } > ITCM AT> MRAM

  .data.at_dtcm : ALIGN(8)
  {
    *(vtable)
    *(.data)
    *(.data*)
    KEEP(*(.jcr*))
  } > DTCM AT> MRAM

  .weights : ALIGN(8)
  {
    *(.rodata.tvm)
    *(.rodata.tflite)
  } > SRAM0 AT> MRAM

  .arena (NOLOAD) : ALIGN(8)
  {
    *(.bss.noinit.tvm)
    *(.bss.noinit.tflite_tensor_arena)
  } > SRAM1

  .bss (NOLOAD) : ALIGN(8)
  {
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end__ = .;
  } > DTCM

  .heap (NOLOAD) : ALIGN(8)
  {
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM

  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (NOLOAD) : ALIGN(8)
  {
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM
  PROVIDE(__stack = __StackTop);

  .readonly.at_mram : ALIGN(8)
  {
    /* Use wildcards to mop up any read-only not directed to TCM */
    KEEP(*(.init))
    KEEP(*(.fini))

    . = ALIGN(4);
    /* preinit data */
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    /* init data */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);

    . = ALIGN(4);
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    *(.rodata*)

    KEEP(*(.eh_frame*))
  } > MRAM

}
