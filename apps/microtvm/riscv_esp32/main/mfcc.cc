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

/*
 * Description: MFCC feature extraction to match with TensorFlow MFCC Op
 */

#include "mfcc.h"

#include <cfloat>
#include <cstring>

MFCC::MFCC(int numMfccFeatures, int frameLen)
    : numMfccFeatures(numMfccFeatures), frameLen(frameLen) {
  // Round-up to nearest power of 2.
  frameLenPadded = pow(2, ceil((log(frameLen) / log(2))));

  frame = std::vector<float>(frameLenPadded, 0.0);
  buffer = std::vector<float>(frameLenPadded, 0.0);
  melEnergies = std::vector<float>(NUM_FBANK_BINS, 0.0);

  // Create window function.
  windowFunc = std::vector<float>(frameLen, 0.0);
  for (int i = 0; i < frameLen; i++)
    windowFunc[i] = 0.5 - 0.5 * riscv_cos_f32(M_2PI * (static_cast<float>(i)) / (frameLen));

  // Create mel filterbank.
  fbankFilterFirst = std::vector<int32_t>(NUM_FBANK_BINS, 0);
  fbankFilterLast = std::vector<int32_t>(NUM_FBANK_BINS, 0);
  melFbank = CreateMelFbank();

  // Create DCT matrix.
  dctMatrix = CreateDctMatrix(NUM_FBANK_BINS, numMfccFeatures);

  // Initialize FFT.
  riscv_rfft_fast_init_f32(&fft, frameLenPadded);
}

std::vector<float> MFCC::CreateDctMatrix(int32_t inputLength, int32_t coefficientCount) {
  int32_t k, n;
  std::vector<float> M(inputLength * coefficientCount);
  float normalizer;
  normalizer = sqrt(2.0 / static_cast<float>(inputLength));
  for (k = 0; k < coefficientCount; k++) {
    for (n = 0; n < inputLength; n++) {
      M[k * inputLength + n] =
          normalizer * riscv_cos_f32((static_cast<double>(M_PI)) / inputLength * (n + 0.5) * k);
    }
  }
  return M;
}

std::vector<std::vector<float>> MFCC::CreateMelFbank() {
  int32_t bin, i;

  int32_t numFftBins = frameLenPadded / 2;
  float fftBinWidth = (static_cast<float>(SAMP_FREQ)) / frameLenPadded;
  float melLowFreq = MelScale(MEL_LOW_FREQ);
  float melHighFreq = MelScale(MEL_HIGH_FREQ);
  float melFreqDelta = (melHighFreq - melLowFreq) / (NUM_FBANK_BINS + 1);

  std::vector<float> thisBin(numFftBins);
  std::vector<std::vector<float>> melFbank(NUM_FBANK_BINS);

  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    float leftMel = melLowFreq + bin * melFreqDelta;
    float centerMel = melLowFreq + (bin + 1) * melFreqDelta;
    float rightMel = melLowFreq + (bin + 2) * melFreqDelta;

    int32_t firstIndex = -1, lastIndex = -1;

    for (i = 0; i < numFftBins; i++) {
      float freq = (fftBinWidth * i);  // Center freq of this FFT bin.
      float mel = MelScale(freq);
      thisBin[i] = 0.0;

      if (mel > leftMel && mel < rightMel) {
        float weight;
        if (mel <= centerMel) {
          weight = (mel - leftMel) / (centerMel - leftMel);
        } else {
          weight = (rightMel - mel) / (rightMel - centerMel);
        }
        thisBin[i] = weight;
        if (firstIndex == -1) firstIndex = i;
        lastIndex = i;
      }
    }

    fbankFilterFirst[bin] = firstIndex;
    fbankFilterLast[bin] = lastIndex;

    // Copy the part we care about.
    for (i = firstIndex; i <= lastIndex; i++) {
      melFbank[bin].push_back(thisBin[i]);
    }
  }

  return melFbank;
}

void MFCC::MfccCompute(const int16_t* audioData, float* mfccOut) {
  int32_t i, j, bin;

  // TensorFlow way of normalizing .wav data to (-1,1).
  for (i = 0; i < frameLen; i++) {
    frame[i] = static_cast<float>(audioData[i]) / (1 << 15);
  }

  // Fill up remaining with zeros.
  memset(&frame[frameLen], 0, sizeof(float) * (frameLenPadded - frameLen));

  for (i = 0; i < frameLen; i++) {
    frame[i] *= windowFunc[i];
  }

  // Compute FFT.
  riscv_rfft_fast_f32(&fft, frame.data(), buffer.data(), 0);

  // Convert to power spectrum.
  // frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32_t halfDim = frameLenPadded / 2;
  float firstEnergy = buffer[0] * buffer[0],
        lastEnergy = buffer[1] * buffer[1];  // Handle this special case.
  for (i = 1; i < halfDim; i++) {
    float real = buffer[i * 2], im = buffer[i * 2 + 1];
    buffer[i] = real * real + im * im;
  }
  buffer[0] = firstEnergy;
  buffer[halfDim] = lastEnergy;

  float sqrtData;
  // Apply mel filterbanks.
  for (bin = 0; bin < NUM_FBANK_BINS; bin++) {
    j = 0;
    float melEnergy = 0;
    int32_t firstIndex = fbankFilterFirst[bin];
    int32_t lastIndex = fbankFilterLast[bin];
    for (i = firstIndex; i <= lastIndex; i++) {
      sqrtData = sqrt(buffer[i]);
      melEnergy += (sqrtData)*melFbank[bin][j++];
    }
    melEnergies[bin] = melEnergy;

    // Avoid log of zero.
    if (melEnergy == 0.0) melEnergies[bin] = FLT_MIN;
  }

  // Take log.
  for (bin = 0; bin < NUM_FBANK_BINS; bin++) melEnergies[bin] = logf(melEnergies[bin]);

  // Take DCT. Uses matrix mul.
  for (i = 0; i < numMfccFeatures; i++) {
    float sum = 0.0;
    for (j = 0; j < NUM_FBANK_BINS; j++) {
      sum += dctMatrix[i * NUM_FBANK_BINS + j] * melEnergies[j];
    }

    mfccOut[i] = sum;
  }
}
