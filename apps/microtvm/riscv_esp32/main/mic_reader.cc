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

#include "mic_reader.h"

#include <cstring>

#include "frag.h"
#include "fragment_detector.h"
#include "spec.h"

MicReader::MicReader() : errors_(0), fz_(0), sz_(0) {
  AudioRxSlot::plan(A_SAMPLE_RATE, A_HW_BITS_PER_SAMPLE_DATA, A_SLOT_NUM, A_POLLING_TIME_MS,
                    A_FRAGMENT_TIME_MS, io_plan_);
  mic_ = new AudioRxSlot(BCK_PIN, WS_PIN, DATA_PIN, CHANNEL_SELECT_PIN);

  hw_fragment_size_ = GetFragmentSize();
  audio_buffer_size_ = PARTS(A_AUDIO_DURATION_MS, A_FRAGMENT_TIME_MS) * hw_fragment_size_;
  kws_duration_size_ = PARTS(A_SAMPLE_RATE * KWS_DURATION_MS, 1000) * A_BYTES_PER_SAMPLE_DATA;
  hw_kws_duration_size_ = PARTS(A_SAMPLE_RATE * KWS_DURATION_MS, 1000) * A_HW_BYTES_PER_SAMPLE_DATA;
  hw_kws_half_duration_size_ =
      PARTS(A_SAMPLE_RATE * KWS_DURATION_MS, 2 * 1000) * A_HW_BYTES_PER_SAMPLE_DATA;
  hw_kws_dbl_duration_size_ = hw_kws_duration_size_ * 2;

  audioBuffer_ = new audio_t[PARTS(audio_buffer_size_, sizeof(audio_t))];
  if (audioBuffer_)
    std::memset(audioBuffer_, 0, audio_buffer_size_);
  else
    errors_++;

  det_.word_det_fragment_time_msec = A_DET_WORD_FRAGMENT_TIME_MS;
  det_.word_det_time_msec = A_DET_WORD_TIME_MS;
  det_.interword_det_time_msec = A_DET_INTERWORD_TIME_MS;
  det_.before_word_time_msec = A_DET_BEFORE_WORD_MS;
  det_.syllable_det_time_msec = A_DET_SYLLABLE_TIME_MS;
#ifdef A_COLLECT_FRAGMENTS
  if (!(det_.fragments = new std::vector<DetectItem>())) errors_++;
#else
  det_.fragments = nullptr;
#endif  // A_COLLECT_FRAGMENTS
}

MicReader::~MicReader() {
  Close();
  if (mic_) delete mic_;
  if (audioBuffer_) delete[] audioBuffer_;
  if (det_.fragments) {
    det_.fragments->clear();
    delete det_.fragments;
  }
}

// Calculate the microphone interface settings and start the microphone
void MicReader::Setup() {
  DEBUG_PRINTF(
      "dma_frame_num: %ld, dma_desc_num: %ld,  polling_buffer_size: %ld, fragment_buffer_size: "
      "%ld\n",
      io_plan_.dma_frame_num, io_plan_.dma_desc_num, io_plan_.polling_buffer_size,
      io_plan_.read_fragment_size);
  mic_->end();
  errors_ += mic_->begin(A_SAMPLE_RATE, io_plan_, 0,
#if defined(A_HW_BITS_PER_SAMPLE_DATA_8BIT)
                         bwBitWidth8,
#elif defined(A_HW_BITS_PER_SAMPLE_DATA_16BIT)
                         bwBitWidth16,
#elif defined(A_HW_BITS_PER_SAMPLE_DATA_24BIT)
                         bwBitWidth24,
#elif defined(A_HW_BITS_PER_SAMPLE_DATA_32BIT)
                         bwBitWidth32,
#else
#error Unsupported BPS!
#endif
                         stStdRight, smMono,
#if defined(A_HW_BIT_SHIFT)
                         bsEnable
#else
                         bsDisable
#endif
  );
  fz_ = 0;

  {  // Start
    DEBUG_SCOPE_TIMER("Start (Setup)");
    const size_t start_size =
        PARTS(A_SAMPLE_RATE * A_DET_START_TIME_MS, 1000) * A_HW_BYTES_PER_SAMPLE_DATA;
    size_t start_sz = 0;
    while (start_sz < start_size) start_sz += Read(audioBuffer_, hw_fragment_size_);
  }

  env_ = {.average_abs = 0, .max_abs = 0};
  {  // Environment
    DEBUG_SCOPE_TIMER("Environment (Setup)");
    const size_t env_size =
        PARTS(A_SAMPLE_RATE * A_DET_ENVIRONMENT_TIME_MS, 1000) * A_HW_BYTES_PER_SAMPLE_DATA;
    size_t env_sz = 0;
    do {
      size_t sz = 0;
      while ((sz <= audio_buffer_size_ - hw_fragment_size_) && ((env_sz + sz) < env_size))
        sz += Read(reinterpret_cast<uint8_t*>(audioBuffer_) + sz, hw_fragment_size_);
      env_sz += sz;
      EnvironmentProcess(audioBuffer_, sz, A_HW_BYTES_PER_SAMPLE_DATA, env_);
    } while (env_sz < env_size);
  }
}

// Stop the microphone
void MicReader::Close() { mic_->end(); }

bool MicReader::Prepare() {
  det_.fragment_byte_size = 0;
  det_.environment = env_.max_abs;
  do {
    size_t kws_sz = 0;
    size_t read_size = std::min(hw_kws_duration_size_ - sz_, hw_kws_half_duration_size_);
    while ((sz_ <= audio_buffer_size_ - hw_fragment_size_) && (kws_sz < read_size)) {
      size_t bytes = Read(reinterpret_cast<uint8_t*>(audioBuffer_) + sz_, hw_fragment_size_);
      kws_sz += bytes;
      sz_ += bytes;
    }
    if (det_.fragment_byte_size) {
      if (sz_ >= hw_kws_duration_size_) break;
      continue;
    }
    if (DetectionProcess(audioBuffer_, sz_, A_HW_BYTES_PER_SAMPLE_DATA, &det_)) {
      if (det_.fragment_byte_offset) {
        sz_ -= det_.fragment_byte_offset;
        std::memmove(audioBuffer_,
                     reinterpret_cast<uint8_t*>(audioBuffer_) + det_.fragment_byte_offset, sz_);
      }
      if (sz_ >= hw_kws_duration_size_) break;
      continue;
    }
    if (sz_ > hw_kws_half_duration_size_) {
      std::memmove(audioBuffer_,
                   reinterpret_cast<uint8_t*>(audioBuffer_) + (sz_ - hw_kws_half_duration_size_),
                   hw_kws_half_duration_size_);
      sz_ = hw_kws_half_duration_size_;
    }
    return false;
  } while (true);
  fz_ = sz_;
  sz_ = 0;
  return true;
}

// Main stream function: read, detect, preprocess
bool MicReader::Collect() {
  if (!fz_) return false;

  {
    DEBUG_SCOPE_TIMER("Process");
    fz_ = SampleProcess(audioBuffer_, fz_);
  }
#ifdef A_MON_ENABLE
  mon(audioBuffer_, fz_, A_BYTES_PER_SAMPLE_DATA);
#endif  // A_MON_ENABLE
  fz_ = 0;
  return true;
}

size_t MicReader::Read(void* dst, size_t bytes) {
  size_t sz = 0;
  if (dst && bytes)
    while (bytes >= (sz + io_plan_.read_fragment_size))
      sz += mic_->read(reinterpret_cast<uint8_t*>(dst) + sz, io_plan_.read_fragment_size);
  return sz;
}

// Return the desired size of the audio stream reading transaction
size_t MicReader::GetFragmentSize() { return io_plan_.read_fragment_size; }

// Return microphone interface driver buffer size
size_t MicReader::GetBufferSize() { return io_plan_.polling_buffer_size; }

/*!
 * \brief Convert a stream
 *
 * \param buf[in] buffer of analyzed stream data
 * \param bytes[in] buffer size
 * \return conversion stream data buffer size
 */
size_t MicReader::SampleProcess(void* buf, size_t bytes) {
#if A_BYTES_PER_SAMPLE_DATA > A_HW_BYTES_PER_SAMPLE_DATA
#error Unsupported BPS!
#endif

#define SP_DATA_OFFSET (A_HW_BYTES_PER_SAMPLE_DATA - A_BYTES_PER_SAMPLE_DATA)

#if defined(A_VOLUME_X2)
#define SP_VOLUME_SHT 1
#define SP_VOLUME_MSK 0x40
#elif defined(A_VOLUME_X4)
#define SP_VOLUME_SHT 2
#define SP_VOLUME_MSK 0x60
#elif defined(A_VOLUME_X8)
#define SP_VOLUME_SHT 3
#define SP_VOLUME_MSK 0x70
#elif defined(A_VOLUME_X16)
#define SP_VOLUME_SHT 4
#define SP_VOLUME_MSK 0x78
#elif defined(A_VOLUME_X32)
#define SP_VOLUME_SHT 5
#define SP_VOLUME_MSK 0x7C
#elif defined(A_VOLUME_X64)
#define SP_VOLUME_SHT 6
#define SP_VOLUME_MSK 0x7E
#elif defined(A_VOLUME_X128)
#define SP_VOLUME_SHT 7
#define SP_VOLUME_MSK 0x7F
#else
#define SP_VOLUME_SHT 0
#define SP_VOLUME_MSK 0x00
#endif

  if ((A_BYTES_PER_SAMPLE_DATA == A_HW_BYTES_PER_SAMPLE_DATA) && (SP_VOLUME_SHT == 0)) return bytes;

  uint8_t* s = reinterpret_cast<uint8_t*>(buf);
  uint8_t* d = reinterpret_cast<uint8_t*>(buf);
  const size_t samples = bytes / A_HW_BYTES_PER_SAMPLE_DATA;

  if (SP_VOLUME_SHT == 0) {
    // No sample reference level increase
    for (size_t z = 0; z < samples; z++) {
      s += SP_DATA_OFFSET;
      for (int i = 0; i < A_BYTES_PER_SAMPLE_DATA; i++) {
        *d++ = *s++;
      }
    }
    return samples * A_BYTES_PER_SAMPLE_DATA;
  }
  // Sample reference level increase
  uint32_t val;
  int sat;
  for (size_t z = 0; z < samples; z++) {
    // Collect, convert and increase the sample reference value
    sat = 0;
    if (s[A_HW_BYTES_PER_SAMPLE_DATA - 1] & 0x80) {
      if ((s[A_HW_BYTES_PER_SAMPLE_DATA - 1] & SP_VOLUME_MSK) != SP_VOLUME_MSK) {
        sat = 1;
        for (int i = 0; i < A_BYTES_PER_SAMPLE_DATA; i++) d[i] = 0;
        d[A_BYTES_PER_SAMPLE_DATA - 1] |= 0x80;
      }
    } else if (s[A_HW_BYTES_PER_SAMPLE_DATA - 1] & SP_VOLUME_MSK) {
      sat = 1;
      for (int i = 0; i < A_BYTES_PER_SAMPLE_DATA; i++) d[i] = 0xFF;
      d[A_BYTES_PER_SAMPLE_DATA - 1] &= 0x7F;
    }
    if (!sat) {
      val = s[0] & 0xFF;
#if A_HW_BYTES_PER_SAMPLE_DATA == 2
      val |= s[1] << 8;
#elif A_HW_BYTES_PER_SAMPLE_DATA == 3
      val |= ((s[2] << 8) | s[1]) << 8;
#elif A_HW_BYTES_PER_SAMPLE_DATA == 4
      val |= ((((s[3] << 8) | s[2]) << 8) | s[1]) << 8;
#endif
      val <<= SP_VOLUME_SHT;
      val >>= (SP_DATA_OFFSET * 8);
      for (int i = 0; i < A_BYTES_PER_SAMPLE_DATA; i++) {
        d[i] = static_cast<uint8_t>(val);
        val >>= 8;
      }
      if (s[A_HW_BYTES_PER_SAMPLE_DATA - 1] & 0x80)
        d[A_BYTES_PER_SAMPLE_DATA - 1] |= 0x80;
      else
        d[A_BYTES_PER_SAMPLE_DATA - 1] &= 0x7F;
    }
    d += A_BYTES_PER_SAMPLE_DATA;
    s += A_HW_BYTES_PER_SAMPLE_DATA;
  }  // for
  return samples * A_BYTES_PER_SAMPLE_DATA;
}

// Analysis of the environment ('silence') signal level
void MicReader::EnvironmentProcess(void* buf, const size_t bytes, const size_t bytes_width,
                                   Environment& env) {
  const size_t samples = bytes / bytes_width;
  if (samples && buf && bytes && bytes_width) {
    proc_t sum = proc_t(0);
    uint8_t* s = reinterpret_cast<uint8_t*>(buf);
    int32_t val;
    for (size_t z = 0; z < samples; z++) {
      val = (s[bytes_width - 1] & 0x80) ? -1 : 0;
      for (int b = bytes_width - 1; b >= 0; b--) val = (val << 8) | (s[b] & 0xFF);
      s += bytes_width;

      val = std::abs(val);
      sum += static_cast<proc_t>(val);
      if (env.max_abs < val) env.max_abs = val;
    }
    if (env.average_abs < (static_cast<float>(sum) / samples))
      env.average_abs = static_cast<float>(sum) / samples;
  }
}

/*!
 * \brief Detect a useful stream fragment
 *
 * \param buf[in] buffer of analyzed stream data
 * \param bytes[in] buffer size
 * \param bytes_width[in] sample size
 * \param det[in,out] pointer to a structure with fragment detection parameters and detection
 * results \return the number of detected words
 */
size_t MicReader::DetectionProcess(void* buf, const size_t bytes, const size_t bytes_width,
                                   Detection* det) {
  if (!buf || !bytes || !bytes_width || !det) return 0;

  uint8_t* s = reinterpret_cast<uint8_t*>(buf);
  const size_t samples = bytes / bytes_width;
  FragmentDetector detector(*det);
  detector.Init(samples, bytes_width);

  int32_t val;
  for (size_t z = 0; z < samples; z++) {
    // Collect the value of the sample reference
    val = (s[bytes_width - 1] & 0x80) ? -1 : 0;
    for (int b = bytes_width - 1; b >= 0; b--) val = (val << 8) | (s[b] & 0xFF);
    s += bytes_width;
    detector.Detect(val);
  }  // for z

  return (det->fragments) ? det->fragments->size() : (det->fragment_byte_size) ? 1 : 0;
}
