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
 * \file utvm_timer.c
 * \brief uTVM timer API definitions for STM32F746XX-series boards
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "utvm_runtime.h"

// There are two implementations of cycle counters on the STM32F7X: SysTick and
// CYCCNT.  SysTick is preferred, as it gives better error handling, but the
// counter is only 24 bits wide.  If a larger timer is needed, use the CYCCNT
// implementation, which has a 32-bit counter.
#define USE_SYSTICK

#ifdef USE_SYSTICK

#define SYST_CSR    (*((volatile uint32_t *) 0xE000E010))
#define SYST_RVR    (*((volatile uint32_t *) 0xE000E014))
#define SYST_CVR    (*((volatile uint32_t *) 0xE000E018))
#define SYST_CALIB  (*((volatile uint32_t *) 0xE000E01C))

#define SYST_CSR_ENABLE     0
#define SYST_CSR_TICKINT    1
#define SYST_CSR_CLKSOURCE  2
#define SYST_COUNTFLAG      16

#define SYST_CALIB_NOREF  31
#define SYST_CALIB_SKEW   30

uint32_t start_time = 0;
uint32_t stop_time = 0;

int32_t UTVMTimerStart() {
  SYST_CSR = (1 << SYST_CSR_ENABLE) | (1 << SYST_CSR_CLKSOURCE);
  // wait until timer starts
  while (SYST_CVR == 0) {}
  start_time = SYST_CVR;
  return 0;
}

void UTVMTimerStop() {
  SYST_CSR = 0;
  stop_time = SYST_CVR;
}

void UTVMTimerReset() {
  SYST_CSR = 0;
  // maximum reload value (24-bit)
  SYST_RVR = (~((uint32_t) 0)) >> 8;
  SYST_CVR = 0;
}

uint32_t UTVMTimerRead() {
  if (SYST_CSR & SYST_COUNTFLAG) {
    TVMAPISetLastError("timer overflowed");
    return -1;
  } else {
    return start_time - stop_time;
  }
}

#else  // !USE_SYSTICK

#define DWT_CTRL    (*((volatile uint32_t *) 0xE0001000))
#define DWT_CYCCNT  (*((volatile uint32_t *) 0xE0001004))

#define DWT_CTRL_NOCYCCNT   25
#define DWT_CTRL_CYCCNTENA  0

uint32_t start_time = 0;
uint32_t stop_time = 0;

void UTVMTimerReset() {
  DWT_CYCCNT = 0;
}

int32_t UTVMTimerStart() {
  if (DWT_CTRL & DWT_CTRL_NOCYCCNT) {
    TVMAPISetLastError("cycle counter not implemented on device");
    return -1;
  }
  start_time = DWT_CYCCNT;
  DWT_CTRL |= (1 << DWT_CTRL_CYCCNTENA);
}

void UTVMTimerStop() {
  stop_time = DWT_CYCCNT;
  DWT_CTRL &= ~(1 << DWT_CTRL_CYCCNTENA);
}

int32_t UTVMTimerRead() {
  if (stop_time > stop_time) {
    return stop_time - start_time;
  } else {
    uint32_t largest = ~0;
    return (largest - start_time) + stop_time;
  }
}

#endif  // USE_SYSTICK

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
