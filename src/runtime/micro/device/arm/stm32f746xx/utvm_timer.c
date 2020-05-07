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
// NOTE: This expects ST CMSIS to be in your include path.
// Download STM32CubeF7 here:
// https://www.st.com/content/st_com/en/products/embedded-software/mcu-mpu-embedded-software/stm32-embedded-software/stm32cube-mcu-mpu-packages/stm32cubef7.html
// and add Drivers/CMSIS to your C include path.
#include "Device/ST/STM32F7xx/Include/stm32f746xx.h"


#define utvm_SystemCoreClock 216000000UL

int32_t UTVMTimerStart() {
  UTVMTimerReset();
  TIM2->CR1 =
    TIM_CR1_CEN;  // Start counter
  return UTVM_ERR_OK;
}

uint32_t UTVMTimerStop(int32_t* err) {
  TIM2->CR1 &= TIM_CR1_CEN;
  if (TIM2->SR & TIM_SR_UIF_Msk) {
    *err = UTVM_ERR_TIMER_OVERFLOW;
    return 0;
  }
  *err = UTVM_ERR_OK;
  uint32_t tim_cnt = TIM2->CNT;
  uint32_t millis = tim_cnt / (utvm_SystemCoreClock / 1000);
  uint32_t micros =
    (tim_cnt - (millis * (utvm_SystemCoreClock / 1000))) /
    (utvm_SystemCoreClock / 1000000);
  return millis * 1000 + micros;
}

void UTVMTimerReset() {
  RCC->APB1RSTR |= RCC_APB1RSTR_TIM2RST;  // Hold TIM2 in reset
  RCC->DCKCFGR1 = (RCC->DCKCFGR1 & ~RCC_DCKCFGR1_TIMPRE_Msk);  // disable 2x clock boost to TIM2
  RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_PPRE1_Msk);  // No AHB clock division to APB1 (1:1).
  RCC->APB1ENR |= RCC_APB1ENR_TIM2EN;  // Enable TIM2 clock.
  RCC->APB1RSTR &= ~RCC_APB1RSTR_TIM2RST;  // Exit TIM2 reset.

  DBGMCU->APB1FZ |= DBGMCU_APB1_FZ_DBG_TIM2_STOP;  // stop TIM2 clock during debug halt.
  TIM2->ARR = 0xffffffff;
  if (TIM2->SR & TIM_SR_UIF_Msk) {
    for (;;) ;
  }
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
