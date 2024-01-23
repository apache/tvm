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

/*
 * Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#ifndef __KWS_MFCC_H__
#define __KWS_MFCC_H__

extern "C" {
#include <math.h>
#include <string.h>
}

#include <vector>

#include "dsp/fast_math_functions.h"
#include "dsp/transform_functions.h"

#define SAMP_FREQ 16000
#define NUM_FBANK_BINS 40
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000

#define M_2PI 6.283185307179586476925286766559005
#ifndef M_PI    /* M_PI might not be defined for non-gcc based tc */
#define M_PI PI /* Comes from math.h */
#endif          /* M_PI */

class MFCC {
 private:
  int numMfccFeatures;
  int frameLen;
  int frameLenPadded;
  std::vector<float> frame;
  std::vector<float> buffer;
  std::vector<float> melEnergies;
  std::vector<float> windowFunc;
  std::vector<int32_t> fbankFilterFirst;
  std::vector<int32_t> fbankFilterLast;
  std::vector<std::vector<float>> melFbank;
  std::vector<float> dctMatrix;
  riscv_rfft_fast_instance_f32 fft;
  static std::vector<float> CreateDctMatrix(int32_t inputLength, int32_t coefficientCount);
  std::vector<std::vector<float>> CreateMelFbank();

  static inline float InverseMelScale(float melFreq) {
    return 700.0f * (expf(melFreq / 1127.0f) - 1.0f);
  }

  static inline float MelScale(float freq) { return 1127.0f * logf(1.0f + freq / 700.0f); }

 public:
  MFCC(int numMfccFeatures, int frameLen);
  ~MFCC() = default;

  void MfccCompute(const int16_t* data, float* mfccOut);
};

#endif
