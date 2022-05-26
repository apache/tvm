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
    htm = new HexagonThreadManager(6, MIN_STACK_SIZE_BYTES, MIN_PIPE_SIZE_WORDS);
    htm->GetStreamHandles(&streams);
  }
  void TearDown() override {
    delete htm;
  }
  HexagonThreadManager *htm;
  std::vector<TVMStreamHandle> streams;
};

TEST_F(HexagonThreadManagerTest, init) {
  CHECK(htm != nullptr);
  CHECK_EQ(streams.size(), 6);
}

TEST_F(HexagonThreadManagerTest, ctor_errors) {
  ASSERT_ANY_THROW(HexagonThreadManager(60, 16*1024, 1024));
  ASSERT_ANY_THROW(HexagonThreadManager(6, MAX_STACK_SIZE_BYTES + 1, 1024));
  ASSERT_ANY_THROW(HexagonThreadManager(6, 16*1024, MAX_PIPE_SIZE_WORDS + 1));
}

TEST_F(HexagonThreadManagerTest, wait_on_threads) {
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, start_wait) {
  htm->Start();
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, wait_signal) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, re_signal) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->Signal(streams[1], 0);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, re_wait) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->Wait(streams[0], 0);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, wait_signal_x2) {
  htm->Wait(streams[0], 0);
  htm->Signal(streams[1], 0);
  htm->Wait(streams[1], 1);
  htm->Signal(streams[0], 1);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, signal_wait) {
  htm->Signal(streams[1], 0);
  htm->Wait(streams[0], 0);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to) {
  htm->SyncFromTo(streams[0], streams[1]);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to_self) {
  htm->SyncFromTo(streams[0], streams[0]);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to_x2) {
  htm->SyncFromTo(streams[0], streams[1]);
  htm->SyncFromTo(streams[1], streams[0]);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
}

TEST_F(HexagonThreadManagerTest, sync_from_to_all) {
  htm->SyncFromTo(streams[0], streams[1]);
  htm->SyncFromTo(streams[1], streams[2]);
  htm->SyncFromTo(streams[2], streams[3]);
  htm->SyncFromTo(streams[3], streams[4]);
  htm->SyncFromTo(streams[4], streams[5]);
  htm->WaitOnThreads();
  CHECK_EQ(42, 42);
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

