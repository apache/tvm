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

#include "../../../src/support/ring_buffer.h"

#include <gtest/gtest.h>

namespace tvm {
namespace support {
namespace {

TEST(RingBuffer, ReadWrite) {
  RingBuffer buffer;
  std::vector<int> data = {1, 2, 3, 4};
  std::vector<int> output;

  buffer.Write(data.data(), data.size() * 4);
  ASSERT_EQ(buffer.bytes_available(), data.size() * 4);

  output.resize(4);
  buffer.Read(output.data(), data.size() * 4);

  for (size_t i = 0; i < output.size(); ++i) {
    ASSERT_EQ(output[i], data[i]);
  }
}

TEST(RingBuffer, ReadWithCallback) {
  RingBuffer buffer;
  std::vector<int> data = {1, 2, 3, 4};
  std::vector<int> output;

  buffer.Write(data.data(), data.size() * 4);

  auto callback0 = [](const char* data, size_t size) -> size_t {
    const int* iptr = reinterpret_cast<const int*>(data);
    ICHECK_EQ(iptr[0], 1);
    ICHECK_EQ(iptr[1], 2);
    return size;
  };
  buffer.ReadWithCallback(callback0, 2 * sizeof(int));
  auto callback1 = [](const char* data, size_t size) -> size_t {
    const int* iptr = reinterpret_cast<const int*>(data);
    ICHECK_EQ(iptr[0], 3);
    ICHECK_EQ(iptr[1], 4);
    return size;
  };
  buffer.ReadWithCallback(callback1, 2 * sizeof(int));
}
}  // namespace
}  // namespace support
}  // namespace tvm
