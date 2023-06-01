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

#include "vulkan_device.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanStreamProfiler::VulkanStreamProfiler(const VulkanDevice* device)
    : device_(device), curr_state_(READY), available_(device->UseDebugUtilsLabel()) {}

void AmdRgpProfiler::capture() {
  if (!available_) {
    return;
  }

  // Trigger RGP capture by using dummy present and switch state from READY to RUNNING
  if (curr_state_ == READY) {
    VkDebugUtilsLabelEXT frame_end_label = {
        VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, "AmdFrameEnd", {0.0f, 0.0f, 0.0f, 0.0f}};
    device_->queue_insert_debug_utils_label_functions->vkQueueInsertDebugUtilsLabelEXT(
        device_->Queue(), &frame_end_label);

    VkDebugUtilsLabelEXT frame_begin_label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                              nullptr,
                                              "AmdFrameBegin",
                                              {0.0f, 0.0f, 0.0f, 0.0f}};
    device_->queue_insert_debug_utils_label_functions->vkQueueInsertDebugUtilsLabelEXT(
        device_->Queue(), &frame_begin_label);

    // Set state as RUNNING
    curr_state_ = RUNNING;
  }
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
