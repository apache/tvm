/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * Linker script for STM32H7 series
 *   1024Kbytes FLASH and 192Kbytes RAM
 *
 * Set heap size, stack size and stack location according
 * to application requirements.
 *
 * Set memory bank area and size if external memory is used.
 *
 *  Target      : STMicroelectronics STM32
 */

/* Entry Point */
ENTRY(Reset_Handler)

/* Highest address of the user mode stack */
/* _estack = 0x24080000; */  /* end of AXIRAM */
/*_estack = 0x20020000; */   /* end of DTCMRAM */
/* Generate a link error if heap and stack don't fit into RAM */
/* _Min_Heap_Size =  0x100000; */ /* 1MB 0x4000; */      /* required amount of heap  */
_Min_Heap_Size =  0x40000; /* 256KB */
_Min_Stack_Size = 0x8000; /* required amount of stack */

/* Specify the memory areas */
MEMORY
{
ITCMRAM (xrw)   : ORIGIN = 0x00000000, LENGTH = 64K
FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 2048K
DTCMRAM (xrw)   : ORIGIN = 0x20000000, LENGTH = 128K
AXIRAM (xrw)    : ORIGIN = 0x24000000, LENGTH = 512K
SRAM123 (xrw)   : ORIGIN = 0x30000000, LENGTH = 288K
SRAM4   (xrw)   : ORIGIN = 0x38000000, LENGTH = 64K
BKPSRAM (xrw)   : ORIGIN = 0x38800000, LENGTH = 64K
QSPIFLASH (rx)  : ORIGIN = 0x90000000, LENGTH = 128M
SDRAM (xrw)     : ORIGIN = 0xD0000000, LENGTH = 32M
}

/* Define output sections */
SECTIONS
{
  /* The startup code goes first into FLASH */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Startup code */
    . = ALIGN(4);
  } >FLASH
  
  
  /* The program code and other data goes into FLASH */
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)

    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >FLASH

  /* Constant data goes into FLASH */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)         /* .rodata sections (constants, strings, etc.) */
    *(.rodata*)        /* .rodata* sections (constants, strings, etc.) */
    . = ALIGN(4);
  } >FLASH

  .ARM.extab   : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM : {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array     :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >FLASH
  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >FLASH
  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >FLASH

  .nn_models :
  {
    . = ALIGN(4);
    __models_section_start__ = .;
    KEEP (*(.nn_models*))
    __models_section_end__ = .;
  } >FLASH

  /* User_heap_stack section, used to check that there is enough RAM left */
  ._user_stack :
  {
    . = ALIGN(8);
    _sstack = .;
    . = . + _Min_Stack_Size;
    _estack = .;
    . = ALIGN(8);
  } >DTCMRAM   /* AXIRAM DTCMRAM*/

  /* Check if MY_MEMORY usage exceeds MY_MEMORY size */
/*
  ASSERT( LENGTH(MY_MEMORY) >= (__mysection_end__ - __mysection_start__), "MY_MEMORY memory overflowed !")
*/

  /* used by the startup to initialize data */
  _sidata = LOADADDR(.data);

  /* Initialized data sections goes into RAM, load LMA copy after code */
  .data : 
  {
    . = ALIGN(4);
    _sdata = .;        /* create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data*)          /* .data* sections */

    . = ALIGN(4);
    _edata = .;        /* define a global symbol at data end */
  } >DTCMRAM AT> FLASH

  
  /* Uninitialized data section */
  . = ALIGN(4);
  .bss :
  {
    /* This is used by the startup in order to initialize the .bss secion */
    _sbss = .;         /* define a global symbol at bss start */
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;         /* define a global symbol at bss end */
    __bss_end__ = _ebss;
  } >DTCMRAM

  /* Warning: looks like there is no heap overlow check in sbrk */
  ._user_heap :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    __HeapLimit = .;
    . = ALIGN(8);
  } >SDRAM /* >SDRAM AXIRAM */

  .params_external_flash :
  {
    *(.nn_weights*)
  } >QSPIFLASH   /* QSPIFLASH FLASH*/

  .external_sdram (NOLOAD) :
  {
    *(*.nn_data_act)
    /**(.nn_data*)*/
  }  >SDRAM  /*> SDRAM */ /* >AXIRAM*/  /* DTCMRAM */

  /* Remove information from the standard libraries */
  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}


