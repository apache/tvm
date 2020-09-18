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
 * \file tvm/runtime/crt/rpc_common/frame_buffer.h
 * \brief Defines a buffer for use by the RPC framing layer.
 */

#ifndef TVM_RUNTIME_CRT_RPC_COMMON_FRAME_BUFFER_H_
#define TVM_RUNTIME_CRT_RPC_COMMON_FRAME_BUFFER_H_

#include <inttypes.h>
#include <stdlib.h>

namespace tvm {
namespace runtime {
namespace micro_rpc {

class FrameBuffer {
 public:
  FrameBuffer(uint8_t* data, size_t data_size_bytes)
      : data_{data}, capacity_{data_size_bytes}, num_valid_bytes_{0}, read_cursor_{0} {}

  size_t Write(const uint8_t* data, size_t data_size_bytes);

  size_t Read(uint8_t* data, size_t data_size_bytes);

  size_t Peek(uint8_t* data, size_t data_size_bytes);

  void Clear();

  size_t ReadAvailable() const { return num_valid_bytes_ - read_cursor_; }

  size_t Size() const { return num_valid_bytes_; }

 private:
  /*! \brief pointer to data buffer. */
  uint8_t* data_;

  /*! \brief The total number of bytes available in data_. Always a power of 2. */
  size_t capacity_;

  /*! \brief index into data_ of the next potentially-available byte in the buffer.
   * The byte is available when tail_ != data_ + capacity_.
   */
  size_t num_valid_bytes_;

  /*! \brief Read cursor position. */
  size_t read_cursor_;
};

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CRT_RPC_COMMON_FRAME_BUFFER_H_
