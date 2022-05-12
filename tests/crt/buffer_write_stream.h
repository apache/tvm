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

#ifndef TESTS_CRT_BUFFER_WRITE_STREAM_H_
#define TESTS_CRT_BUFFER_WRITE_STREAM_H_

#include <inttypes.h>
#include <tvm/runtime/crt/rpc_common/frame_buffer.h>
#include <tvm/runtime/crt/rpc_common/write_stream.h>

#include <string>

using ::tvm::runtime::micro_rpc::FrameBuffer;
using ::tvm::runtime::micro_rpc::WriteStream;

template <unsigned int N>
class BufferWriteStream : public WriteStream {
 public:
  ssize_t Write(const uint8_t* data, size_t data_size_bytes) override {
    return buffer_.Write(data, data_size_bytes);
  }

  void Reset() {
    buffer_.Clear();
    packet_done_ = false;
  }

  inline bool packet_done() { return packet_done_; }

  inline bool is_valid() { return is_valid_; }

  void PacketDone(bool is_valid) override {
    EXPECT_FALSE(packet_done_);
    packet_done_ = true;
    is_valid_ = is_valid;
  }

  std::string BufferContents() { return std::string((const char*)buffer_data_, buffer_.Size()); }

  static constexpr unsigned int capacity() { return N; }

 private:
  bool packet_done_{false};
  bool is_valid_{false};
  uint8_t buffer_data_[N];
  FrameBuffer buffer_{buffer_data_, N};
};

#endif  // TESTS_CRT_BUFFER_WRITE_STREAM_H_
