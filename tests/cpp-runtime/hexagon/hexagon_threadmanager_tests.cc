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

// TEST(suitename, testname)
// TEST(HexagonThreadManager, _)
// EXPECT_EQ(a, b)
// EXPECT_THROW({...}, exception)

// TEST(HexagonThreadManager, ) {
// }

TEST(HexagonThreadManager, init) {
  auto tm = new HexagonThreadManager(6, 16*1024, 1024);
  delete tm;
  CHECK_EQ(0,0);
}

void thread_print(void* msg) {
  LOG(WARNING) << (char*)msg << "\n";
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

TEST(HexagonThreadManager, dispatch_prints) {
  auto tm = new HexagonThreadManager(6, 16*1024, 1024);
  std::vector<TVMStreamHandle> streams;
  tm->GetStreamHandles(&streams);
  std::vector<std::string> strings;
  strings.resize(streams.size());
  DBG(std::to_string(streams.size()) << " streams");
  for (int i = 0; i < streams.size(); i++) {
    strings[i] += "In thread " + std::to_string(i);
    tm->Dispatch(streams[i], thread_print, (void*)strings[0].c_str());
  }
  tm->Start();
  delete tm;
}

TEST(HexagonThreadManager, dispatch_writes) {
  auto tm = new HexagonThreadManager(6, 16*1024, 1024);
  std::vector<TVMStreamHandle> streams;
  tm->GetStreamHandles(&streams);
  std::vector<int> array;
  std::vector<int> truth;
  array.resize(streams.size());
  truth.resize(streams.size());
  for (int i = 0; i < streams.size(); i++) {
    int val = i * 2;
    ToWrite* cmd = new ToWrite(&array[i], val);
    tm->Dispatch(streams[i], thread_write_val, cmd);
    truth[i] = val;
  }
  tm->Start();
  tm->WaitOnThreads();
  delete tm;
  for (int i = 0; i < streams.size(); i++) {
    DBG(std::to_string(array[i]) << " " << std::to_string(truth[i]));
    CHECK_EQ(array[i], truth[i]);
  }
}

