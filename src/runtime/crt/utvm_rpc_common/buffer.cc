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
 * \file buffer.h
 * \brief Buffer
 */

#include <stdio.h>
#include <string.h>
#include <tvm/runtime/crt/rpc_common/buffer.h>

namespace tvm {
namespace runtime {

size_t Buffer::Write(const uint8_t* data, size_t data_size_bytes) {
  size_t num_bytes_available = capacity_ - num_valid_bytes_;
  size_t num_bytes_to_copy = data_size_bytes;
  if (num_bytes_available < num_bytes_to_copy) {
    num_bytes_to_copy = num_bytes_available;
  }

  memcpy(&data_[num_valid_bytes_], data, num_bytes_to_copy);
  num_valid_bytes_ += num_bytes_to_copy;
  return num_bytes_to_copy;
}

size_t Buffer::Read(uint8_t* data, size_t data_size_bytes) {
  size_t num_bytes_to_copy = data_size_bytes;
  size_t num_bytes_available = num_valid_bytes_ - read_cursor_;
  if (num_bytes_available < num_bytes_to_copy) {
    num_bytes_to_copy = num_bytes_available;
  }

  memcpy(data, &data_[read_cursor_], num_bytes_to_copy);
  read_cursor_ += num_bytes_to_copy;
  return num_bytes_to_copy;
}

void Buffer::Clear() {
  num_valid_bytes_ = 0;
  read_cursor_ = 0;
}

}  // namespace runtime
}  // namespace tvm
