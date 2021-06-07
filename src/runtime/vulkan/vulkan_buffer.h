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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_BUFFER_H_
#define TVM_RUNTIME_VULKAN_VULKAN_BUFFER_H_

#include <vulkan/vulkan_core.h>

#include <memory>
#include <unordered_map>

namespace tvm {
namespace runtime {
namespace vulkan {

struct VulkanBuffer {
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
};

/*! \brief A struct to represent Vulkan buffers backed by host visible memory */
struct VulkanHostVisibleBuffer {
  // A device where the buffer is allocated
  VkDevice device{nullptr};
  // Vulkan buffer and memory
  VulkanBuffer* vk_buf{nullptr};
  // The corresponding pointer to the host memory
  void* host_addr{nullptr};
  // The size of the buffer in bytes
  size_t size{0};
};

using VulkanStagingBuffer = VulkanHostVisibleBuffer;
using VulkanUniformBuffer = VulkanHostVisibleBuffer;

VulkanHostVisibleBuffer* GetOrAllocate(
    int device_id, size_t size, VkBufferUsageFlags usage, uint32_t mem_type_index,
    std::unordered_map<size_t, std::unique_ptr<VulkanHostVisibleBuffer>>* buffers_ptr,
    bool sync_before_realloc = false);

void DeleteHostVisibleBuffer(VulkanHostVisibleBuffer* buf);

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_BUFFER_H_
