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

#include "mfcc_preprocessor.h"

#include <algorithm>
#include <cstring>

#include "def.h"

MfccPreprocessor::MfccPreprocessor() : mfccBufferTemplate_(nullptr), errors_(0) {
  if (!(mfcc_ = new MFCC(KWS_NUM_MFCC, KWS_FRAME_LEN))) errors_++;
#if defined(A_MFCC_ENVIRONMENT_PREPARED_TEMPLATE) || defined(A_MFCC_ZERO_PREPARED_TEMPLATE) || \
    defined(A_MFCC_NOISE_PREPARED_TEMPLATE)
  if (!(mfccBufferTemplate_ = new float[KWS_NUM_MFCC * KWS_NUM_FRAMES])) errors_++;
#endif
}

MfccPreprocessor::~MfccPreprocessor() {
  if (mfccBufferTemplate_) delete[] mfccBufferTemplate_;
  if (mfcc_) delete mfcc_;
}

// Push and calculate MFCC for the pushed data
void MfccPreprocessor::Apply(const void* src, const size_t src_bytes, void* dst) {
  if (!src_bytes) return;

  float* buf;
  const size_t new_frames =
      std::min(static_cast<size_t>(KWS_NUM_FRAMES),
               static_cast<size_t>(src_bytes / (KWS_FRAME_SHIFT * A_BYTES_PER_SAMPLE_DATA)));
  const size_t old_frames = KWS_NUM_FRAMES - new_frames;

  if (!dst) {
    if (!mfccBufferTemplate_) return;
    buf = reinterpret_cast<float*>(mfccBufferTemplate_);
    std::memmove(buf, buf + new_frames * KWS_NUM_MFCC, old_frames * KWS_NUM_MFCC * sizeof(*buf));
#if defined(A_MFCC_ZERO_PREPARED_TEMPLATE)
    for (int f = old_frames * KWS_NUM_MFCC; f < KWS_NUM_FRAMES * KWS_NUM_MFCC; f++) buf[f] = 0.0f;
    return;
#elif defined(A_MFCC_NOISE_PREPARED_TEMPLATE)
    get_random_data(buf + old_frames * KWS_NUM_MFCC, -1.0f, 1.0f, new_frames * KWS_NUM_MFCC);
    return;
#endif
  } else {
    buf = reinterpret_cast<float*>(dst);
    std::memmove(buf, buf + new_frames * KWS_NUM_MFCC, old_frames * KWS_NUM_MFCC * sizeof(*buf));
  }

  if (src) {
    for (int f = old_frames; f < KWS_NUM_FRAMES; f++)
      mfcc_->MfccCompute(reinterpret_cast<const audio_t*>(src) + (f - old_frames) * KWS_FRAME_SHIFT,
                         buf + f * KWS_NUM_MFCC);
  } else {
    if (dst && mfccBufferTemplate_)
      std::memmove(reinterpret_cast<float*>(dst) + old_frames * KWS_NUM_MFCC,
                   reinterpret_cast<float*>(mfccBufferTemplate_) + old_frames * KWS_NUM_MFCC,
                   new_frames * KWS_NUM_MFCC * sizeof(float));
  }
}

int MfccPreprocessor::Errors() { return errors_; }
