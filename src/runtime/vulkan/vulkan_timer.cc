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

#include "vulkan_timer.h"

#include <tvm/runtime/logging.h>

#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanTimerNode::VulkanTimerNode(Device dev) : dev_(dev) {
  // Get the Vulkan device and stream
  auto& vk_dev = VulkanDeviceAPI::Global()->device(dev_.device_id);
  stream_ = &vk_dev.ThreadLocalStream();
  device_ = vk_dev;

  // Retrieve the timestamp period from device properties
  timestamp_period_ = vk_dev.device_properties.timestamp_period;

  CreateQueryPool();
}

VulkanTimerNode::~VulkanTimerNode() { Cleanup(); }

void VulkanTimerNode::CreateQueryPool() {
  VkQueryPoolCreateInfo query_pool_info{};
  query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_pool_info.queryCount = 2;

  VkResult res = vkCreateQueryPool(device_, &query_pool_info, nullptr, &query_pool_);
  ICHECK(res == VK_SUCCESS) << "Failed to create Vulkan query pool.";
}

void VulkanTimerNode::Start() {
  stream_->Launch([this](VulkanStreamState* state) {
    // Reset the query pool before writing timestamps
    vkCmdResetQueryPool(state->cmd_buffer_, query_pool_, start_query_, 2);
    vkCmdWriteTimestamp(state->cmd_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_,
                        start_query_);
  });
}

void VulkanTimerNode::Stop() {
  stream_->Launch([this](VulkanStreamState* state) {
    vkCmdWriteTimestamp(state->cmd_buffer_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_,
                        end_query_);
  });

  // Ensure GPU has finished writing timestamps before collecting them
  stream_->Synchronize();
  CollectTimestamps();
}

int64_t VulkanTimerNode::SyncAndGetElapsedNanos() { return duration_; }

void VulkanTimerNode::CollectTimestamps() {
  uint64_t timestamps[2] = {0};

  VkResult result =
      vkGetQueryPoolResults(device_, query_pool_, 0, 2, sizeof(timestamps), timestamps,
                            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  ICHECK(result == VK_SUCCESS) << "Failed to get Vulkan query pool results.";

  // Calculate the duration in nanoseconds
  uint64_t diff = timestamps[1] - timestamps[0];
  duration_ = static_cast<int64_t>(diff * timestamp_period_);
}

void VulkanTimerNode::Cleanup() {
  if (query_pool_ != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device_, query_pool_, nullptr);
    query_pool_ = VK_NULL_HANDLE;
  }
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
