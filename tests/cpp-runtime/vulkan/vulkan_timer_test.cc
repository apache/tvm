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

#include "../src/runtime/vulkan/vulkan_timer.h"

#include <gtest/gtest.h>
#include <tvm/runtime/profiling.h>

#include "../src/runtime/vulkan/vulkan_device_api.h"

using namespace tvm::runtime;
using namespace tvm::runtime::vulkan;

#define BUFF_SIZE 1024
#define NUM_REPEAT 10

TEST(VulkanTimerNode, TimerCorrectness) {
  VulkanDeviceAPI* api = VulkanDeviceAPI::Global();
  auto device_id = api->GetActiveDeviceID();
  tvm::Device dev{kDLVulkan, device_id};

  constexpr int32_t kBufferSize = 1024;
  Tensor src = Tensor::Empty({kBufferSize}, {kDLInt, 32, 1}, {kDLCPU, 0});
  Tensor dst = Tensor::Empty({kBufferSize}, {kDLInt, 32, 1}, {kDLVulkan, device_id});

  // Fill CPU array with dummy data
  for (int32_t i = 0; i < kBufferSize; ++i) {
    static_cast<int32_t*>(src->data)[i] = i;
  }

  // Create a Timer
  Timer timer = Timer::Start(dev);

  // Perform a CPU -> Vulkan copy
  src.CopyTo(dst);

  // Important: Force Vulkan to flush and sync work
  api->StreamSync(dev, nullptr);

  timer->Stop();
  int64_t elapsed_nanos = timer->SyncAndGetElapsedNanos();

  std::cout << "Elapsed time (nanoseconds): " << elapsed_nanos << std::endl;

  // Check that some time was measured
  ASSERT_GT(elapsed_nanos, 0);
}
