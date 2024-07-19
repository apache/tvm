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

#ifndef _MIC_READER_H_
#define _MIC_READER_H_

#include "audio_rx_slot.h"

class MicReader {
 public:
  MicReader();
  ~MicReader();

  void Setup();
  void Close();
  bool Prepare();
  bool Collect();
  int Errors() { return errors_; }
  const audio_t* AudioBuffer() { return audioBuffer_; }
  size_t KwsDurationSize() { return kws_duration_size_; }

 protected:
  size_t Read(void* dst, size_t bytes);
  size_t GetFragmentSize();
  size_t GetBufferSize();
  void EnvironmentProcess(void* buf, const size_t bytes, const size_t bytes_width,
                          Environment& env);
  size_t DetectionProcess(void* buf, const size_t bytes, const size_t bytes_width, Detection* det);

 private:
  size_t SampleProcess(void* buf, size_t bytes);

 protected:
  audio_t* audioBuffer_;
  size_t audio_buffer_size_;
  size_t hw_fragment_size_;
  size_t kws_duration_size_;
  size_t hw_kws_duration_size_;
  size_t hw_kws_half_duration_size_;
  size_t hw_kws_dbl_duration_size_;

  Environment env_;
  Detection det_;
  int errors_;

 private:
  AudioRxSlot* mic_;
  IOPlan io_plan_;
  size_t fz_;
  size_t sz_;
};

#endif  // _MIC_READER_H_
