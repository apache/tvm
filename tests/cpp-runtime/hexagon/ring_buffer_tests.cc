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
  void SetUp() override { ring_buff = new RingBuffer<int>(size, in_flight); }
  void TearDown() override { delete ring_buff; }

 public:
  std::function<bool(int*)> in_flight = [](int* ptr) {
    if (*ptr == 42) {
      // finished
      return false;
    }
    // in flight
    return true;
  };

  int finished = 42;
  int inflight = 43;
  uint32_t size = 4;
  uint32_t half = size / 2;
  RingBuffer<int>* ring_buff = nullptr;
};

TEST_F(RingBufferTest, zero_size_ring_buffer) {
  ASSERT_THROW(RingBuffer<int>(0, in_flight), InternalError);
}

TEST_F(RingBufferTest, in_flight) { ASSERT_EQ(ring_buff->InFlight(), 0); }

TEST_F(RingBufferTest, next) {
  // get pointer to first item
  int* ptr = ring_buff->Next();
  ASSERT_NE(ptr, nullptr);

  // mark it in flight and check
  *ptr = inflight;
  ASSERT_EQ(ring_buff->InFlight(), 1);

  // mark it finished and check
  *ptr = finished;
  ASSERT_EQ(ring_buff->InFlight(), 0);
}

TEST_F(RingBufferTest, full) {
  // fill the ring buffer
  for (int i = 0; i < size; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);

    // mark in flight and check
    *ptr = inflight;
    ASSERT_EQ(ring_buff->InFlight(), i + 1);
  }

  // check that the ring buffer is full
  ASSERT_EQ(ring_buff->Next(), nullptr);
  ASSERT_EQ(ring_buff->InFlight(), size);
}

TEST_F(RingBufferTest, wrap) {
  // fill the ring buffer, but mark each finished
  bool first = true;
  int* firstptr = nullptr;
  for (int i = 0; i < size; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);

    // save first ptr for later comparison
    if (first) {
      firstptr = ptr;
      first = false;
    }

    // mark finished and check
    *ptr = finished;
    ASSERT_EQ(ring_buff->InFlight(), 0);
  }

  // reuse the first ring buffer entry
  int* ptr = ring_buff->Next();
  ASSERT_EQ(ptr, firstptr);

  // mark it in flight and check
  *ptr = inflight;
  ASSERT_EQ(ring_buff->InFlight(), 1);

  // mark it finished and check
  *ptr = finished;
  ASSERT_EQ(ring_buff->InFlight(), 0);
}

TEST_F(RingBufferTest, wrap_corner) {
  for (int i = 0; i < size; ++i) {
    int* ptr = ring_buff->Next();
    *ptr = finished;
  }

  // reuse the first ring buffer entry
  int* ptr = ring_buff->Next();
  ASSERT_NE(ptr, nullptr);

  // user must mark the item "inflight" before checking in flight count
  // here the "finished" status is inherited from the reused ring buffer entry
  // thus the in flight count is zero instead one; which the user might expect
  ASSERT_EQ(ring_buff->InFlight(), 0);

  // marking the item "inflight" after checking the in flight count
  // will not change the outcome; the ring buffer considers the item "finished"
  *ptr = inflight;
  ASSERT_EQ(ring_buff->InFlight(), 0);
}

TEST_F(RingBufferTest, half_in_flight) {
  // these will complete
  for (int i = 0; i < half; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = finished;
    ASSERT_EQ(ring_buff->InFlight(), 0);
  }

  // these will not complete
  for (int i = 0; i < half; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = inflight;
    ASSERT_EQ(ring_buff->InFlight(), i + 1);
  }

  // check half in flight
  ASSERT_EQ(ring_buff->InFlight(), half);

  // get pointer to next item
  int* ptr = ring_buff->Next();
  ASSERT_NE(ptr, nullptr);

  // mark it inflight and check
  *ptr = inflight;
  ASSERT_EQ(ring_buff->InFlight(), 3);

  // mark it finished and check also blocked
  *ptr = finished;
  ASSERT_EQ(ring_buff->InFlight(), 3);
}

TEST_F(RingBufferTest, half_in_flight_blocked) {
  // these will not complete
  for (int i = 0; i < half; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = inflight;
    ASSERT_EQ(ring_buff->InFlight(), i + 1);
  }

  // these would complete, but they are blocked
  for (int i = half; i < size; ++i) {
    int* ptr = ring_buff->Next();
    ASSERT_NE(ptr, nullptr);
    *ptr = finished;
    ASSERT_EQ(ring_buff->InFlight(), i + 1);
  }

  // check that the ring buffer is full
  ASSERT_EQ(ring_buff->Next(), nullptr);
  ASSERT_EQ(ring_buff->InFlight(), size);
}

class QueuedRingBufferTest : public RingBufferTest {
  void SetUp() override { queued_ring_buff = new QueuedRingBuffer<int>(size, in_flight); }
  void TearDown() override { delete queued_ring_buff; }

 public:
  QueuedRingBuffer<int>* queued_ring_buff = nullptr;
};

TEST_F(QueuedRingBufferTest, two_queues) {
  int* q0 = queued_ring_buff->Next(0);
  *q0 = inflight;
  ASSERT_EQ(queued_ring_buff->InFlight(0), 1);
  ASSERT_EQ(queued_ring_buff->InFlight(1), 0);

  int* q1 = queued_ring_buff->Next(1);
  *q1 = inflight;
  ASSERT_EQ(queued_ring_buff->InFlight(0), 1);
  ASSERT_EQ(queued_ring_buff->InFlight(1), 1);

  *q0 = finished;
  ASSERT_EQ(queued_ring_buff->InFlight(0), 0);
  ASSERT_EQ(queued_ring_buff->InFlight(1), 1);

  *q1 = finished;
  ASSERT_EQ(queued_ring_buff->InFlight(0), 0);
  ASSERT_EQ(queued_ring_buff->InFlight(1), 0);
}
