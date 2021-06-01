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

#include "vulkan_buffer.h"

#include "vulkan_device_api.h"
#include "vulkan_thread_entry.h"

namespace tvm {
namespace runtime {
namespace vulkan {

void DeleteHostVisibleBuffer(VulkanHostVisibleBuffer* buf) {
  if (buf && buf->vk_buf) {
    if (buf->host_addr != nullptr) {
      vkUnmapMemory(buf->device, buf->vk_buf->memory);
    }
    if (buf->vk_buf->memory != VK_NULL_HANDLE) {
      vkFreeMemory(buf->device, buf->vk_buf->memory, nullptr);
    }
    if (buf->vk_buf->buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(buf->device, buf->vk_buf->buffer, nullptr);
    }
    buf->host_addr = nullptr;
    delete buf->vk_buf;
  }
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
