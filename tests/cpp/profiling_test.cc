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
#include <tvm/runtime/profiling.h>

#include <chrono>
#include <thread>

namespace tvm {
namespace runtime {
TEST(DefaultTimer, Basic) {
  using namespace tvm::runtime;
  Device dev;
  dev.device_type = kDLCPU;
  dev.device_id = 0;

  Timer t = Timer::Start(dev);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  t->Stop();
  int64_t elapsed = t->SyncAndGetElapsedNanos();
  CHECK_GT(elapsed, 9 * 1e6);
}
}  // namespace runtime
}  // namespace tvm
