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
#include <thread>
#include <chrono>

namespace tvm {
namespace runtime {
TEST(DefaultTimer, Basic) {
  using namespace tvm::runtime;
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;

  Timer t = StartTimer(ctx);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  t.Stop();
  int64_t elapsed = t.SyncAndGetTime();
  CHECK_GT(elapsed, 0);
}
}  // namespace topi
}  // namespace tvm

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
