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

#include <gtest/gtest.h>

#include "../src/runtime/hexagon/ring_buffer.h"

using namespace tvm::runtime;
using namespace tvm::runtime::hexagon;

class RingBufferTest : public ::testing::Test {
  void SetUp() override {
    always_in_flight_rb = new RingBuffer<int>(10, always_in_flight);
    ring_buff = new RingBuffer<int>(10, check_answer);
  }
  void TearDown() override { delete always_in_flight_rb; }

 public:
  std::function<bool(int*)> always_in_flight = [](int* ptr) { return true; };
  RingBuffer<int>* always_in_flight_rb;

  std::function<bool(int*)> check_answer = [](int* ptr) {
    if (*ptr == 42) {
      // complete, retired, done
      return false;
    }
    // in flight
    return true;
  };
  RingBuffer<int>* ring_buff;
};

TEST_F(RingBufferTest, zero_size_ring_buffer) {
  ASSERT_THROW(RingBuffer<int>(0, always_in_flight), InternalError);
}

TEST_F(RingBufferTest, in_flight) { ASSERT_EQ(always_in_flight_rb->InFlight(), 0); }

TEST_F(RingBufferTest, next) {
  int* ptr = always_in_flight_rb->Next();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(always_in_flight_rb->InFlight(), 1);
}

TEST_F(RingBufferTest, full) {
  for (int i = 0; i < 10; ++i) {
    int* ptr = always_in_flight_rb->Next();
    ASSERT_NE(ptr, nullptr);
  }
  ASSERT_EQ(always_in_flight_rb->InFlight(), 10);
  ASSERT_EQ(always_in_flight_rb->Next(), nullptr);
  ASSERT_EQ(always_in_flight_rb->InFlight(), 10);
}

TEST_F(RingBufferTest, half_full) {
  // these will complete
  for (int i = 0; i < 5; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = 42;
  }

  // these will not complete
  for (int i = 0; i < 5; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = 43;
  }

  ASSERT_EQ(ring_buff->InFlight(), 5);
  ASSERT_NE(ring_buff->Next(), nullptr);
  ASSERT_EQ(ring_buff->InFlight(), 6);
}

TEST_F(RingBufferTest, still_full) {
  // these will not complete
  for (int i = 0; i < 5; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = 43;
  }

  // these would complete, but they are blocked
  for (int i = 0; i < 5; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = 42;
  }

  ASSERT_EQ(ring_buff->InFlight(), 10);
  ASSERT_EQ(ring_buff->Next(), nullptr);
  ASSERT_EQ(ring_buff->InFlight(), 10);
}
