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
 * \file stm32lib.h
 * \brief Support for the TVM code generation for STM32 ARM targets.
 */

// LINT_C_FILE

#ifndef TVM_RUNTIME_CRT_CONTRIB_STM32_STM32LIB_H_
#define TVM_RUNTIME_CRT_CONTRIB_STM32_STM32LIB_H_

#include <stdint.h>

#ifdef __arm__
#include "cmsis_compiler.h"
#endif

// =========================================================
//   conv2d_nhwc_int8_smlad_reset_
// =========================================================
__attribute__((always_inline)) inline static int32_t
conv2d_nhwc_int8_smlad_reset_(
  int32_t * conv
) {
  *conv = 0;
  return 0;
}

// =========================================================
//   conv2d_nhwc_int8_smlad_update_
// =========================================================
__attribute__((always_inline)) inline static int32_t
conv2d_nhwc_int8_smlad_update_(
  int8_t * input,
  int8_t * weights,
  int32_t * conv,
  int16_t omaps,
  int16_t channels
) {
  // te.sum((a[k]*b[k,i]).astype('int32'), axis=k

#ifdef __arm__
    //
    // Load and extend int8x2 => int16x2
    //
  int32_t in = *(int32_t*)input;  // TODO(stoa): fix it !!
    //
    // Pack 2 kernel values
    //
    int8_t ker0 = *(weights+0*channels);
    int8_t ker1 = *(weights+1*channels);
    int32_t ker = ((ker1<<16)&0x00FF)|(ker0&0x00FF);

    *(conv) = __SMLAD(in, ker, *(conv));
#else
    int8_t in0 = *(input+0);
    int8_t ker0 = *(weights+0*channels);
    int8_t in1 = *(input+1);
    int8_t ker1 = *(weights+1*channels);
    *(conv) = *(conv) + in0*ker0 + in1*ker1;
#endif

  return 0;
}

#if 0
// =========================================================
//   conv2d_nhwc_int16_smlad_reset_
// =========================================================
__attribute__((always_inline)) inline static int32_t
conv2d_nhwc_int16_smlad_reset_(
  int32_t * conv
) {
  *conv = 0;
  return 0;
}

// =========================================================
//   conv2d_nhwc_int16_smlad_update_
// =========================================================
__attribute__((always_inline)) inline static int32_t
conv2d_nhwc_int16_smlad_update_(
  int16_t * input,
  int16_t * weights,
  int32_t * conv,
  int16_t channels
) {
  // te.sum((a[k]*b[k,i]).astype('int32'), axis=k

  //
  // Load int16x2
  //
#ifdef __arm__

  int32_t in = *(int32_t*)input;

  int16_t ker0 = *(weights+0*channels);
  int16_t ker1 = *(weights+1*channels);
  //
  // Pack 2 kernel values
  //
  int32_t ker = (ker1<<16)|(ker0&0xFFFF);

  *conv = __SMLAD(in, ker, *conv);

#else

  int16_t in0 = *(input+0);
  int16_t in1 = *(input+1);

  int16_t ker0 = *(weights+0*channels);
  int16_t ker1 = *(weights+1*channels);

  *conv = *conv + in0*ker0 + in1*ker1;

#endif

  return 0;
}
#endif  // 0

// =========================================================
//   conv2d_NCHWc_int16_smlad_reset_
// =========================================================
__attribute__((always_inline)) inline static int32_t
conv2d_NCHWc_int16_smlad_reset_(
  int32_t * conv
) {
  *conv = 0;
  return 0;
}

// =========================================================
//   conv2d_NCHWc_int16_smlad_update_
// =========================================================
__attribute__((always_inline)) inline static int32_t
conv2d_NCHWc_int16_smlad_update_(
  int16_t * input,
  int16_t * weights,
  int32_t * conv,
  int16_t channels
) {
  //
  // Load input: int16x2
  //
#ifdef __arm__

  int32_t in = *(int32_t*)input;

  //
  // Load weights:
  //
  int16_t ker0 = *(weights+0*channels);
  int16_t ker1 = *(weights+1*channels);
  //
  // Pack 2 kernel values
  //
  int32_t ker = (ker1<<16)|(ker0&0xFFFF);

  *conv = __SMLAD(in, ker, *conv);

#else

  int16_t in0 = *(input+0);
  int16_t in1 = *(input+1);

  //
  // Load weights:
  //
  int16_t ker0 = *(weights+0*channels);
  int16_t ker1 = *(weights+1*channels);

  *conv = *conv + in0*ker0 + in1*ker1;

#endif

  return 0;
}

#endif  // TVM_RUNTIME_CRT_CONTRIB_STM32_STM32LIB_H_
