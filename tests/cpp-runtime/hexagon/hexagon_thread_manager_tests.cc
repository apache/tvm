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
#include <tvm/runtime/logging.h>

#include "../src/runtime/hexagon/hexagon_device_api.h"
#include "../src/runtime/hexagon/hexagon_thread_manager.h"

using namespace tvm::runtime;
using namespace tvm::runtime::hexagon;

class HexagonThreadManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create with no hardware resources so we don't conflict with session HexagonThreadManager
    htm = new HexagonThreadManager(threads, stack_size, pipe_size);
    streams = htm->GetStreamHandles();
  }
  void TearDown() override { delete htm; }
  HexagonThreadManager* htm{nullptr};
  std::vector<TVMStreamHandle> streams;
  int answer{0};
  const unsigned threads{6};
  const unsigned pipe_size{100};
  const unsigned stack_size{0x4000};  // 16KB
};

TEST_F(HexagonThreadManagerTest, ctor_errors) {
  // zero threads
  ASSERT_THROW(HexagonThreadManager(0, stack_size, pipe_size), InternalError);
  // too many threads
  ASSERT_THROW(HexagonThreadManager(0x10000000, stack_size, pipe_size), InternalError);
  // stack too small
  ASSERT_THROW(HexagonThreadManager(6, 0, pipe_size), InternalError);
  // stack too big
  ASSERT_THROW(HexagonThreadManager(6, 0x10000000, pipe_size), InternalError);
  // pipe too small
  ASSERT_THROW(HexagonThreadManager(6, stack_size, 9), InternalError);
  // pipe too big
  ASSERT_THROW(HexagonThreadManager(6, stack_size, 0x10000000), InternalError);
  // hw resources count doesn't match thread count
  ASSERT_THROW(HexagonThreadManager(6, stack_size, pipe_size, {DMA_0}), InternalError);
  // hw resources doesn't match specific supported configuration
  ASSERT_THROW(
      HexagonThreadManager(6, stack_size, pipe_size, {DMA_0, HTP_0, HVX_0, HVX_1, HVX_2, DMA_0}),
      InternalError);
  // hw resources doesn't match specific supported configuration
  ASSERT_THROW(HexagonThreadManager(5, stack_size, pipe_size, {DMA_0, HTP_0, HVX_0, HVX_1, HVX_2}),
               InternalError);
}

TEST_F(HexagonThreadManagerTest, init) {
  CHECK(htm != nullptr);
  CHECK_EQ(streams.size(), threads);
}

void get_the_answer(void* answer) { *reinterpret_cast<int*>(answer) = 42; }

TEST_F(HexagonThreadManagerTest, dispatch) {
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->Start();
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, dispatch_wait) {
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, wait_signal) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, re_signal) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->Signal(streams[1], 0);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, re_wait) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->Wait(streams[0], 0);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, wait_signal_x2) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->Wait(streams[0], 1);
  htm->Signal(streams[1], 1);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, signal_wait) {
  htm->Signal(streams[1], 0);
  htm->Wait(streams[0], 0);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to) {
  htm->SyncFromTo(streams[1], streams[0]);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to_self) {
  htm->SyncFromTo(streams[0], streams[0]);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to_x2) {
  htm->SyncFromTo(streams[0], streams[1]);
  htm->SyncFromTo(streams[1], streams[0]);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to_all) {
  htm->SyncFromTo(streams[5], streams[4]);
  htm->SyncFromTo(streams[4], streams[3]);
  htm->SyncFromTo(streams[3], streams[2]);
  htm->SyncFromTo(streams[2], streams[1]);
  htm->SyncFromTo(streams[1], streams[0]);
  htm->Dispatch(streams[0], get_the_answer, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

TEST_F(HexagonThreadManagerTest, pipe_fill) {
  // fill the pipe
  for (int i = 0; i < pipe_size; ++i) {
    htm->Dispatch(streams[0], get_the_answer, &answer);
  }
  htm->WaitOnThreads();
  CHECK_EQ(answer, 42);
}

// TODO(HWE): Create a temporary thread manager with a smaller pipe for this test
TEST_F(HexagonThreadManagerTest, pipe_overflow) {
  // fill the pipe
  for (int i = 0; i < pipe_size; ++i) {
    htm->Dispatch(streams[0], get_the_answer, &answer);
  }
  // overflow the pipe
  bool space = htm->Dispatch(streams[0], get_the_answer, &answer);
  CHECK_EQ(space, false);
}

void increment(void* voidptr) {
  int* intptr = reinterpret_cast<int*>(voidptr);
  *intptr = *intptr + 1;
}

TEST_F(HexagonThreadManagerTest, producer_consumer) {
  htm->Dispatch(streams[5], increment, &answer);
  htm->SyncFromTo(streams[5], streams[4]);
  htm->Dispatch(streams[4], increment, &answer);
  htm->SyncFromTo(streams[4], streams[3]);
  htm->Dispatch(streams[3], increment, &answer);
  htm->SyncFromTo(streams[3], streams[2]);
  htm->Dispatch(streams[2], increment, &answer);
  htm->SyncFromTo(streams[2], streams[1]);
  htm->Dispatch(streams[1], increment, &answer);
  htm->SyncFromTo(streams[1], streams[0]);
  htm->Dispatch(streams[0], increment, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 6);
}

TEST_F(HexagonThreadManagerTest, producer_consumer_signal_wait) {
  htm->Wait(streams[0], 0);
  htm->Wait(streams[1], 1);
  htm->Wait(streams[2], 2);
  htm->Wait(streams[3], 3);
  htm->Wait(streams[4], 4);

  htm->Dispatch(streams[5], increment, &answer);
  htm->Signal(streams[5], 4);
  htm->Dispatch(streams[4], increment, &answer);
  htm->Signal(streams[4], 3);
  htm->Dispatch(streams[3], increment, &answer);
  htm->Signal(streams[3], 2);
  htm->Dispatch(streams[2], increment, &answer);
  htm->Signal(streams[2], 1);
  htm->Dispatch(streams[1], increment, &answer);
  htm->Signal(streams[1], 0);
  htm->Dispatch(streams[0], increment, &answer);
  htm->WaitOnThreads();
  CHECK_EQ(answer, 6);
}

struct ToAppend {
  std::vector<int>* arr;
  int value;
  ToAppend(std::vector<int>* addr, int value) : arr(addr), value(value){};
};

void append(void* toappend) {
  ToAppend* cmd = reinterpret_cast<ToAppend*>(toappend);
  cmd->arr->push_back(cmd->value);
}

TEST_F(HexagonThreadManagerTest, thread_order) {
  std::vector<int> arr;

  ToAppend cmd0(&arr, 0);
  htm->Dispatch(streams[0], append, &cmd0);
  htm->SyncFromTo(streams[0], streams[1]);

  ToAppend cmd1(&arr, 1);
  htm->Dispatch(streams[1], append, &cmd1);
  htm->SyncFromTo(streams[1], streams[2]);

  ToAppend cmd2(&arr, 2);
  htm->Dispatch(streams[2], append, &cmd2);
  htm->SyncFromTo(streams[2], streams[3]);

  ToAppend cmd3(&arr, 3);
  htm->Dispatch(streams[3], append, &cmd3);
  htm->SyncFromTo(streams[3], streams[4]);

  ToAppend cmd4(&arr, 4);
  htm->Dispatch(streams[4], append, &cmd4);
  htm->SyncFromTo(streams[4], streams[5]);

  ToAppend cmd5(&arr, 5);
  htm->Dispatch(streams[5], append, &cmd5);
  htm->WaitOnThreads();
  for (int i = 0; i < threads; ++i) {
    CHECK_EQ(arr[i], i);
  }
}

TEST_F(HexagonThreadManagerTest, thread_order_signal_wait) {
  std::vector<int> arr;

  htm->Wait(streams[1], 1);
  htm->Wait(streams[2], 2);
  htm->Wait(streams[3], 3);
  htm->Wait(streams[4], 4);
  htm->Wait(streams[5], 5);

  ToAppend cmd0(&arr, 0);
  htm->Dispatch(streams[0], append, &cmd0);
  htm->Signal(streams[0], 1);

  ToAppend cmd1(&arr, 1);
  htm->Dispatch(streams[1], append, &cmd1);
  htm->Signal(streams[1], 2);

  ToAppend cmd2(&arr, 2);
  htm->Dispatch(streams[2], append, &cmd2);
  htm->Signal(streams[2], 3);

  ToAppend cmd3(&arr, 3);
  htm->Dispatch(streams[3], append, &cmd3);
  htm->Signal(streams[3], 4);

  ToAppend cmd4(&arr, 4);
  htm->Dispatch(streams[4], append, &cmd4);
  htm->Signal(streams[4], 5);

  ToAppend cmd5(&arr, 5);
  htm->Dispatch(streams[5], append, &cmd5);
  htm->WaitOnThreads();
  for (int i = 0; i < threads; ++i) {
    CHECK_EQ(arr[i], i);
  }
}

struct ToWrite {
  int* addr;
  int value;
  ToWrite(int* addr, int value) : addr(addr), value(value){};
};

void thread_write_val(void* towrite) {
  ToWrite* cmd = reinterpret_cast<ToWrite*>(towrite);
  *(cmd->addr) = cmd->value;
  delete cmd;
}

TEST_F(HexagonThreadManagerTest, dispatch_writes) {
  std::vector<int> array;
  std::vector<int> truth;
  array.resize(streams.size());
  truth.resize(streams.size());
  for (int i = 0; i < streams.size(); i++) {
    int val = i * 2;
    ToWrite* cmd = new ToWrite(&array[i], val);
    htm->Dispatch(streams[i], thread_write_val, cmd);
    truth[i] = val;
  }
  htm->Start();
  htm->WaitOnThreads();
  for (int i = 0; i < streams.size(); i++) {
    CHECK_EQ(array[i], truth[i]);
  }
}
