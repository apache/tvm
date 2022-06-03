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
#include <tvm/runtime/container/optional.h>

#include "../src/runtime/hexagon/hexagon_threadmanager.h"
#include "tvm/runtime/logging.h"

using namespace tvm::runtime;
using namespace tvm::runtime::hexagon;

class HexagonThreadManagerTest : public ::testing::Test {
  protected:
  void SetUp() override {
    htm = new HexagonThreadManager(6, stack_size, pipe_size);
    htm->GetStreamHandles(&streams);
  }
  void TearDown() override {
    delete htm;
  }
  HexagonThreadManager *htm {nullptr};
  std::vector<TVMStreamHandle> streams;
  int answer{0};
  unsigned pipe_size {100};
  unsigned stack_size {0x4000}; // 16KB
};

TEST_F(HexagonThreadManagerTest, ctor_errors) {
  // zero threads
  ASSERT_THROW(HexagonThreadManager(0, stack_size, pipe_size), InternalError);
  // too many threads
  ASSERT_THROW(HexagonThreadManager(60, stack_size, pipe_size), InternalError);
  // stack too small
  ASSERT_THROW(HexagonThreadManager(6, MIN_STACK_SIZE_BYTES - 1, pipe_size), InternalError);
  // stack too big
  ASSERT_THROW(HexagonThreadManager(6, MAX_STACK_SIZE_BYTES + 1, pipe_size), InternalError);
  // pipe too small
  ASSERT_THROW(HexagonThreadManager(6, stack_size, MIN_PIPE_SIZE_WORDS - 1), InternalError);
  // pipe too big
  ASSERT_THROW(HexagonThreadManager(6, stack_size, MAX_PIPE_SIZE_WORDS + 1), InternalError);
}

TEST_F(HexagonThreadManagerTest, init) {
  CHECK(htm != nullptr);
  CHECK_EQ(streams.size(), 6);
}

void get_the_answer(void* answer) {
  *(int*)answer = 42;
}

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

TEST_F(HexagonThreadManagerTest, pipe_overflow) {
  // fill the pipe
  for (int i = 0; i < pipe_size; ++i) {
    htm->Dispatch(streams[0], get_the_answer, &answer);
  }
  // overflow the pipe
  bool space = htm->Dispatch(streams[0], get_the_answer, &answer);
  CHECK_EQ(space, false);
}

struct ToWrite {
  int* addr;
  int value;
  ToWrite(int* addr, int value) : addr(addr), value(value) {};
};

void thread_write_val(void* towrite) {
  ToWrite* cmd = (ToWrite*)towrite;
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
    DBG(std::to_string(array[i]) << " " << std::to_string(truth[i]));
    CHECK_EQ(array[i], truth[i]);
  }
}

void thread_print(void* msg) {
  LOG(WARNING) << (char*)msg << "\n";
}

TEST_F(HexagonThreadManagerTest, dispatch_prints) {
  std::vector<std::string> strings;
  strings.resize(streams.size());
  DBG(std::to_string(streams.size()) << " streams");
  for (int i = 0; i < streams.size(); i++) {
    strings[i] += "In thread " + std::to_string(i);
    htm->Dispatch(streams[i], thread_print, (void*)strings[0].c_str());
  }
  htm->Start();
  htm->WaitOnThreads();
}

