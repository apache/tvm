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
 */

/*!
 * \file aiTestUtility.c
 * \brief STM32 Utility functions for STM32 AI test application
 */

/*
 * Description:
 *
 * History:
 *  - v1.0 - initial version (from initial aiSystemPerformance file - v5.1)
 *  - v1.1 - add Arm Compiler 6 support (MDK)
 *  - v1.2 - add io low level code to manage a COM through the STM32 USB CDC profile
 *           enabled with the USE_USB_CDC_CLASS = 1 define.
 *  - v1.3 - Fix compilation issue for H7 dual core
 */

/*
 *  NOTE about the USE_USB_CDC_CLASS support
 *
 *   When USE_USB_CDC_CLASS is set to 1 the code to manage the
 *   STM32 USB_DEVICE and the CDC class should added in the project.
 *
 *   CDC_Receive_FS()/CDC_Control_FS() functions (file: "usb_cdc_if.c")
 *   should be adapted/patched
 *
 *   o CDC_Receive_FS() to handle the received data
 *    ...
 *    static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len) {
 *      extern void ioPushInUserUsb(uint8_t *pw, uint32_t *len);
 *      ioPushInUserUsb(Buf, Len);
 *      USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
 *      USBD_CDC_ReceivePacket(&hUsbDeviceFS);
 *      return (USBD_OK);
 *      }
 *    ...
 *
 *   o CDC_Control_FS() to return the valid data for CDC_GET_LINE_CODING
 *     request.This is requested by the Python module in charge of the
 *     Serial COM with the validation process.
 *
 *      ...
 *      case CDC_GET_LINE_CODING:
 *
 *        if (length == 7) {
 *        *(uint32_t *)pbuf = 115200;
 *        pbuf[4] = 0;
 *        pbuf[5] = 0;
 *        pbuf[6] = 8;
 *        }
 *
 *      break;
 *      ...
 *
 */

#include <stdio.h>
#include <string.h>

#include <aiTestUtility.h>

#ifndef USE_USB_CDC_CLASS
#define USE_USB_CDC_CLASS 0
#endif


/* -----------------------------------------------------------------------------
 * Sanity check
 * -----------------------------------------------------------------------------
 */

#if defined(__GNUC__) && !defined(__ARMCC_VERSION)

#elif defined(__ICCARM__)

#if _APP_HEAP_MONITOR_ == 1
#undef _APP_HEAP_MONITOR_
#define _APP_HEAP_MONITOR_ 0
#error HEAP monitor is not YET supported
#endif

#elif defined(__CC_ARM) || defined(__ARMCC_VERSION)

#if _APP_STACK_MONITOR_ == 1
#undef _APP_STACK_MONITOR_
#define _APP_STACK_MONITOR_ 0
#error STACK monitor is not YET supported
#endif

#if _APP_HEAP_MONITOR_ == 1
#undef _APP_HEAP_MONITOR_
#define _APP_HEAP_MONITOR_ 0
#error HEAP monitor is not YET supported
#endif

#else

#error ARM MCU tool chain is not supported

#endif


/* -----------------------------------------------------------------------------
 * IO functions
 * -----------------------------------------------------------------------------
 */

#if defined(__GNUC__) && !defined(__ARMCC_VERSION)
#include <errno.h>
#include <sys/unistd.h> /* STDOUT_FILENO, STDERR_FILENO */
#elif defined (__ICCARM__)
#if (__IAR_SYSTEMS_ICC__ <= 8)
/* Temporary workaround - LowLevelIOInterface.h seems not available
   with IAR 7.80.4 */
#define _LLIO_STDIN  0
#define _LLIO_STDOUT 1
#define _LLIO_STDERR 2
#define _LLIO_ERROR ((size_t)-1) /* For __read and __write. */
#else
#include <LowLevelIOInterface.h> /* _LLIO_STDOUT, _LLIO_STDERR */
#endif

#elif defined (__CC_ARM) || defined(__ARMCC_VERSION)

#endif

#if defined(USE_USB_CDC_CLASS) && USE_USB_CDC_CLASS == 1

#include "usbd_cdc_if.h"

#define _MAX_USB_USER_ELEM (128)

uint8_t _usb_fifo[_MAX_USB_USER_ELEM];
volatile uint32_t _usb_nb_w_item = 0;
volatile uint32_t _usb_nb_r_item = 0;
uint8_t *_usb_p_read = &_usb_fifo[0];
uint8_t *_usb_p_write = &_usb_fifo[0];

#endif

extern UART_HandleTypeDef UartHandle;

static bool _ioWriteAllowed = true;

int ioRawGetUint8(uint8_t *c, uint32_t timeout)
{
  HAL_StatusTypeDef status;

  if (!c)
    return -1;

  status = HAL_UART_Receive(&UartHandle, (uint8_t *)c, 1,
      timeout);

  if (status == HAL_TIMEOUT)
    return -1;

  return (status == HAL_OK ? 1 : 0);
}

#if defined(USE_USB_CDC_CLASS) && USE_USB_CDC_CLASS == 1

void ioPushInUserUsb(uint8_t *pw, uint32_t *len)
{
  uint8_t *pr = pw;
  for (int i=0; i < *len; i++) {
    *_usb_p_write++ = *pr++;
    if (_usb_p_write == &_usb_fifo[_MAX_USB_USER_ELEM])
      _usb_p_write = &_usb_fifo[0];
    _usb_nb_w_item++;
  }
}

bool ioRawWriteBuffer(uint8_t *buff, int count)
{
  HAL_StatusTypeDef status = HAL_OK;
  while (USBD_BUSY == CDC_Transmit_FS(buff, count));
  // CDC_Transmit_FS(buff, count);
  return (status == HAL_OK);
}

bool ioRawReadBuffer(uint8_t *buff, int count)
{
  HAL_StatusTypeDef status = HAL_OK;
  while ((_usb_nb_w_item - _usb_nb_r_item) < count) {};

  uint8_t *pw = buff;
  for (int i=0; i < count; i++) {
    *pw++ = *_usb_p_read++;
    if (_usb_p_read == &_usb_fifo[_MAX_USB_USER_ELEM])
      _usb_p_read = &_usb_fifo[0];
    _usb_nb_r_item++;
  }
  return (status == HAL_OK);
}

void ioRawDisableLLWrite(void)
{
}

#else

bool ioRawWriteBuffer(uint8_t *buff, int count)
{
  HAL_StatusTypeDef status;

  status = HAL_UART_Transmit(&UartHandle, buff, count, HAL_MAX_DELAY);

  return (status == HAL_OK);
}

bool ioRawReadBuffer(uint8_t *buff, int count)
{
  HAL_StatusTypeDef status;

  status = HAL_UART_Receive(&UartHandle, buff, count, HAL_MAX_DELAY);

  return (status == HAL_OK);
}

void ioRawDisableLLWrite(void)
{
  _ioWriteAllowed = false;
}

#endif

#if defined(__GNUC__) && !defined(__ARMCC_VERSION)

int _write(int fd, const void *buff, int count);

int _write(int fd, const void *buff, int count)
{
  HAL_StatusTypeDef status;

  if ((count < 0) && (fd != STDOUT_FILENO) && (fd != STDERR_FILENO)) {
    errno = EBADF;
    return -1;
  }

  if (_ioWriteAllowed)
    status = HAL_UART_Transmit(&UartHandle, (uint8_t *)buff, count,
        HAL_MAX_DELAY);
  else
    status = HAL_OK;

  return (status == HAL_OK ? count : 0);
}

#elif defined (__ICCARM__)

__ATTRIBUTES  size_t __write(int handle, const unsigned char *buffer,
    size_t size);

__ATTRIBUTES  size_t __write(int handle, const unsigned char *buffer,
    size_t size)
{
  HAL_StatusTypeDef status;

  /*
   * This means that we should flush internal buffers.  Since we
   * don't we just return.  (Remember, "handle" == -1 means that all
   * handles should be flushed.)
   */
  if (buffer == 0)
    return 0;

  /* This template only writes to "standard out" and "standard err",
   * for all other file handles it returns failure.
   */
  if ((handle != _LLIO_STDOUT) && (handle != _LLIO_STDERR))
    return _LLIO_ERROR;

  if (_ioWriteAllowed)
    status = HAL_UART_Transmit(&UartHandle, (uint8_t *)buffer, size,
        HAL_MAX_DELAY);
  else
    status = HAL_OK;

  return (status == HAL_OK ? size : _LLIO_ERROR);
}

#elif defined (__CC_ARM) || defined(__ARMCC_VERSION)

int fputc(int ch, FILE *f)
{
  if (_ioWriteAllowed)
    HAL_UART_Transmit(&UartHandle, (uint8_t *)&ch, 1,
        HAL_MAX_DELAY);

  return ch;
}

#else
#error ARM MCU tool-chain is not supported.
#endif


/* -----------------------------------------------------------------------------
 * HEAP Monitor functions
 * -----------------------------------------------------------------------------
 */


#if defined(__GNUC__) && !defined(__ARMCC_VERSION)

#define MAGIC_MALLOC_NUMBER 0xefdcba98

struct io_malloc io_malloc;

void* __real_malloc(size_t bytes);
void __real_free(void *ptr);

void* __wrap_malloc(size_t bytes)
{
  uint8_t *ptr;

  io_malloc.cfg |= 1 << 1;

  /* ensure alignment for magic number */
  bytes = (bytes + 3) & ~3;

  /* add 2x32-bit for size and magic  number */
  ptr = (uint8_t*)__real_malloc(bytes + 8);

  /* remember size */
  if (ptr) {
    *((uint32_t*)ptr) = bytes;
    *((uint32_t*)(ptr + 4 + bytes)) = MAGIC_MALLOC_NUMBER;
  }

  if ((ptr) && (io_malloc.cfg & 1UL)) {
    io_malloc.alloc_req++;
    io_malloc.alloc += bytes;

    io_malloc.used += bytes;

    if (io_malloc.used > io_malloc.max) {
      io_malloc.max = io_malloc.used;
    }

#if _IO_MALLOC_TRACK_MODE == 1
    io_malloc.a_ptr[io_malloc.a_idx] = (ptr + 4);
    io_malloc.a_s[io_malloc.a_idx] = bytes;
    io_malloc.a_idx++;

    if (io_malloc.a_idx >= _IO_MALLOC_TRACK_MODE)
      io_malloc.a_idx = 0;
#endif
  }
  return ptr?(ptr + 4):NULL;
}

void __wrap_free(void *ptr)
{
  uint8_t* p;
  uint32_t bytes;

  io_malloc.cfg |= 1 << 2;

  if (!ptr)
    return;

  p = (uint8_t*)ptr - 4;
  bytes = *((uint32_t*)p);

  if (*((uint32_t*)(p + 4 + bytes)) == MAGIC_MALLOC_NUMBER) {
    *((uint32_t*)(p + 4 + bytes)) = 0;
  }

  if (io_malloc.cfg & 1UL) {
    io_malloc.free_req++;
    io_malloc.free += bytes;
    io_malloc.used -= bytes;

#if _IO_MALLOC_TRACK_MODE == 1
    io_malloc.f_ptr[io_malloc.f_idx] = ptr;
    io_malloc.f_s[io_malloc.f_idx] = bytes;
    io_malloc.f_idx++;

    if (io_malloc.f_idx >= _IO_MALLOC_TRACK_MODE)
      io_malloc.f_idx = 0;
#endif
  }
  __real_free(p);
}

#endif


/* -----------------------------------------------------------------------------
 * STACK Monitor functions
 * -----------------------------------------------------------------------------
 */

#if defined(__GNUC__) && !defined(__ARMCC_VERSION)
extern uint32_t _estack[];
#elif defined (__ICCARM__)
extern int CSTACK$$Limit;
extern int CSTACK$$Base;
#endif

struct io_stack io_stack;

void stackMonInit(uint32_t ctrl, uint32_t cstack, uint32_t msize)
{
  memset(&io_stack, 0, sizeof(struct io_stack));

  /* Reading ARM Core registers */
  io_stack.ctrl = ctrl;
  io_stack.cstack = cstack;

#if defined(__GNUC__) && !defined(__ARMCC_VERSION)
  io_stack.estack = (uint32_t)_estack;
  io_stack.bstack = io_stack.estack - msize;
  io_stack.mstack_size = msize;
#elif defined (__ICCARM__)
  io_stack.estack = (uint32_t)&CSTACK$$Limit;
  io_stack.bstack = (uint32_t)&CSTACK$$Base;
  io_stack.mstack_size = (uint32_t)&CSTACK$$Limit - (uint32_t)&CSTACK$$Base;
#endif

  /* Check that MSP is the active stack */
  if (io_stack.ctrl & CONTROL_SPSEL_Msk) {
    printf("E: MSP is not the active stack (stack monitoring is disabled)\r\n");
    io_stack.stack_mon = false;
  } else
    io_stack.stack_mon = true;

  /* Calculating used stack before test */
  io_stack.ustack_size = io_stack.estack - io_stack.cstack;

  if ((io_stack.stack_mon) && (io_stack.ustack_size > io_stack.mstack_size)) {
    printf("E: !stack overflow detected %d > %d\r\n", (int)io_stack.ustack_size,
        (int)io_stack.mstack_size);
    io_stack.stack_mon = false;
  }
}

/* -----------------------------------------------------------------------------
 * HW-setting functions
 * -----------------------------------------------------------------------------
 */

struct cyclesCount cyclesCount;

__STATIC_INLINE void crcIpInit(void)
{
#if defined(STM32H7)
  /* By default the CRC IP clock is enabled */
  __HAL_RCC_CRC_CLK_ENABLE();
#elif defined(STM32MP1)
  __HAL_RCC_CRC2_CLK_ENABLE();
#else
  if (!__HAL_RCC_CRC_IS_CLK_ENABLED())
    printf("W: CRC IP clock is NOT enabled\r\n");

  /* By default the CRC IP clock is enabled */
  __HAL_RCC_CRC_CLK_ENABLE();
#endif
}

void dwtIpInit(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

#ifdef STM32F7
  DWT->LAR = 0xC5ACCE55;
#endif

  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk | DWT_CTRL_CPIEVTENA_Msk;

}

uint32_t systemCoreClock(void)
{
#if defined(STM32MP1)
  return HAL_RCC_GetSystemCoreClockFreq();
#elif !defined(STM32H7)
  return HAL_RCC_GetHCLKFreq();
#else
  return HAL_RCC_GetSysClockFreq();
#endif
}

int dwtCyclesToTime(uint64_t clks, struct dwtTime *t)
{
  if (!t)
    return -1;
  uint32_t fcpu = systemCoreClock();
  uint64_t s  = clks / fcpu;
  uint64_t ms = (clks * 1000) / fcpu;
  uint64_t us = (clks * 1000 * 1000) / fcpu;
  ms -= (s * 1000);
  us -= (ms * 1000 + s * 1000000);
  t->fcpu = fcpu;
  t->s = s;
  t->ms = ms;
  t->us = us;
  return 0;
}

float dwtCyclesToFloatMs(uint64_t clks)
{
  float res;
  float fcpu = (float)systemCoreClock();
  res = ((float)clks * (float)1000.0) / fcpu;
  return res;
}

__STATIC_INLINE const char *devIdToStr(uint16_t dev_id)
{
  /* DEV_ID field from DBGMCU register */
  const char *str;
  switch (dev_id) {
  case 0x422: str = "STM32F303xB/C"; break;
  case 0x438: str = "STM32F303x6/8"; break;
  case 0x446: str = "STM32F303xD/E"; break;
  case 0x431: str = "STM32F411xC/E"; break;
  case 0x423: str = "STM32F401xB/C"; break;
  case 0x433: str = "STM32F401xD/E"; break;
  case 0x435: str = "STM32L43xxx"; break;
  case 0x462: str = "STM32L45xxx"; break;
  case 0x415: str = "STM32L4x6xx"; break;
  case 0x470: str = "STM32L4Rxxx"; break;
  case 0x472: str = "STM32L5[5,6]2xx"; break;
  case 0x449: str = "STM32F74xxx"; break;
  case 0x450: str = "STM32H743/53/50xx and STM32H745/55/47/57xx"; break; /* see RM0433 & RM0399 */
  case 0x451: str = "STM32F7[6,7]xxx"; break; /* see RM0410 */
  default:    str = "UNKNOWN";
  }
  return str;
}

#if defined(DUAL_CORE) && defined(STM32H7) && defined(CORE_CM4)
#define _STM32H7_DUAL_CORE_CM4
#endif

#if !defined(STM32F3) && !defined(STM32MP1) && !defined(_STM32H7_DUAL_CORE_CM4)
__STATIC_INLINE const char* bitToStr(uint32_t val)
{
  if (val)
    return "True";
  else
    return "False";
}
#endif

static uint32_t mconf;

uint32_t getFlashCacheConf(void)
{
  return mconf;
}

__STATIC_INLINE void _stackInfoLog(void) 
{
#if defined(__GNUC__) || defined(__ICCARM__)
  uint32_t sstack;     /* start of stack @ */
  uint32_t estack;     /* end of stack @ */
  uint32_t stack_size; /* stack size */
#if defined(__GNUC__)
  extern uint32_t _sstack[];
  extern uint32_t _estack[];
  sstack = (uint32_t)_sstack;
  estack = (uint32_t)_estack;
  stack_size = (uint32_t)_estack - (uint32_t)_sstack;
#elif defined(__ICCARM__)
  extern int CSTACK$$Limit;
  extern int CSTACK$$Base;
  sstack = (uint32_t)&CSTACK$$Base;
  estack = (uint32_t)&CSTACK$$Limit;
  stack_size = (uint32_t)&CSTACK$$Limit - (uint32_t)&CSTACK$$Base;
#endif

  printf(" System stack : 0x%08x:0x%08x (%u bytes)\r\n",
          (unsigned int)sstack, (unsigned int)estack,
          (unsigned int)stack_size);
#endif
}

void systemSettingLog(void)
{
  struct dwtTime t;
  uint32_t st;

#if !defined(STM32F3) && !defined(STM32L5)
  uint32_t acr = FLASH->ACR ;
#endif
  uint32_t val;

  /* Display ARM Complier version */

#if defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
  printf("Compiled with MDK-ARM Keil (Arm Compiler 6) %d\r\n", __ARMCC_VERSION);
#elif defined(__GNUC__)
  printf("Compiled with GCC %d.%d.%d\r\n", __GNUC__, __GNUC_MINOR__,
      __GNUC_PATCHLEVEL__);
#elif defined(__ICCARM__)
  printf("Compiled with IAR %d (build %d)\r\n", __IAR_SYSTEMS_ICC__,
      __BUILD_NUMBER__
  );
#elif defined (__CC_ARM)
  printf("Compiled with MDK-ARM Keil %d\r\n", __ARMCC_VERSION);
#endif


  /* Display Device configuration */

  printf("\r\nSTM32 Runtime configuration\r\n");

  printf(" Device       : DevID:0x%04x (%s) RevID:0x%04x\r\n",
      (int)HAL_GetDEVID(),
      devIdToStr(HAL_GetDEVID()),
      (int)HAL_GetREVID()
  );

  printf(" Core Arch.   : M%d - %s %s\r\n",
      __CORTEX_M,
#if (__FPU_PRESENT == 1)
      "FPU PRESENT",
      __FPU_USED ? "and used" : "and not used!"
#else
          "!FPU NOT PRESENT",
          ""
#endif
  );

#if (__FPU_PRESENT == 1) && __FPU_USED
  mconf = (1 << 16);  /* FPU present and used */
#endif

  printf(" HAL version  : 0x%08x\r\n", (int)HAL_GetHalVersion());

  val = systemCoreClock()/1000000;

#if !defined(STM32H7)
  printf(" system clock : %u MHz\r\n", (int)val);
#else
  printf(" SYSCLK clock : %u MHz\r\n", (int)val);
  printf(" HCLK clock   : %u MHz\r\n", (int)HAL_RCC_GetHCLKFreq()/1000000);
#endif

#if defined(STM32F7) || defined(STM32H7)
  val = SCB->CCR;
#if !defined(STM32H7)
  mconf |= (2 << 24);  /* F7 conf. */
  mconf |= ((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos);
  if ((acr & FLASH_ACR_PRFTEN_Msk) >> FLASH_ACR_PRFTEN_Pos)
    mconf |= (1 << 8);
  if ((acr & FLASH_ACR_ARTEN_Msk) >> FLASH_ACR_ARTEN_Pos)
    mconf |= (1 << 9);
  printf(" FLASH conf.  : ACR=0x%08x - Prefetch=%s ART=%s latency=%d\r\n",
      (int)acr,
      bitToStr((acr & FLASH_ACR_PRFTEN_Msk) >> FLASH_ACR_PRFTEN_Pos),
      bitToStr((acr & FLASH_ACR_ARTEN_Msk) >> FLASH_ACR_ARTEN_Pos),
      (int)((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos));
#else
  mconf |= (3 << 24);  /* H7 conf. */
  mconf |= (acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos;
  printf(" FLASH conf.  : ACR=0x%08x - latency=%d\r\n",
      (int)acr,
      (int)((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos));
#endif
#if !defined(CORE_CM4)
  if (val & SCB_CCR_IC_Msk)
    mconf |= (1 << 10);
  if (val & SCB_CCR_DC_Msk)
    mconf |= (1 << 11);
  printf(" CACHE conf.  : $I/$D=(%s,%s)\r\n",
      bitToStr(val & SCB_CCR_IC_Msk),
      bitToStr(val & SCB_CCR_DC_Msk));
#endif
#else
#if !defined(STM32F3) && !defined(STM32L5) && !defined(STM32MP1)
  mconf |= (1 << 24);  /* F4/L4 conf. */
  mconf |= ((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos);
  if ((acr & FLASH_ACR_PRFTEN_Msk) >> FLASH_ACR_PRFTEN_Pos)
    mconf |= (1 << 8);
  if ((acr & FLASH_ACR_ICEN_Msk) >> FLASH_ACR_ICEN_Pos)
    mconf |= (1 << 9);
  if ((acr & FLASH_ACR_DCEN_Msk) >> FLASH_ACR_DCEN_Pos)
    mconf |= (1 << 10);
  printf(" FLASH conf.  : ACR=0x%08x - Prefetch=%s $I/$D=(%s,%s) latency=%d\r\n",
      (int)acr,
      bitToStr((acr & FLASH_ACR_PRFTEN_Msk) >> FLASH_ACR_PRFTEN_Pos),
      bitToStr((acr & FLASH_ACR_ICEN_Msk) >> FLASH_ACR_ICEN_Pos),
      bitToStr((acr & FLASH_ACR_DCEN_Msk) >> FLASH_ACR_DCEN_Pos),
      (int)((acr & FLASH_ACR_LATENCY_Msk) >> FLASH_ACR_LATENCY_Pos));
#endif
#if defined(STM32L5)
  printf(" ICACHE       : %s\r\n", bitToStr(READ_BIT(ICACHE->CR, ICACHE_CR_EN)));
#endif
#endif

  /* Init CRC and DWT */
  crcIpInit();
  dwtIpInit();

  /* Display HAL tick Calibration */
  dwtReset();
  HAL_Delay(100);
  st = dwtGetCycles();
  dwtCyclesToTime(st/100, &t);

  printf(" Calibration  : HAL_Delay(1)=%d.%03d ms\r\n",
      t.s * 100 + t.ms, t.us);

  _stackInfoLog();
}

