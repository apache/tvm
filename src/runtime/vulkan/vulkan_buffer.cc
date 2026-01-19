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

#include <utility>

#include "vulkan_device_api.h"
#include "vulkan_resource.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VkBufferCreateInfo MakeBufferCreateInfo(size_t nbytes, VkBufferUsageFlags usage) {
  VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};

  info.size = nbytes;
  // Since sharingMode is not VK_SHARING_MODE_CONCURRENT, no need to
  // specify the queue families.
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.usage = usage;
  return info;
}

VulkanBuffer::VulkanBuffer(const VulkanDevice& device, size_t nbytes, VkBufferUsageFlags usage,
                           uint32_t mem_type_index, std::optional<std::string> mem_scope,
                           std::shared_ptr<VulkanMemory> back_memory)
    : VulkanResource(device, mem_scope, back_memory), size(nbytes) {
  VkBufferCreateInfo buffer_info = MakeBufferCreateInfo(nbytes, usage);
  VULKAN_CALL(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

  // Allocate new memory if no memory is passed in or if the existing memory is not compatible
  if (!memory) {
    AllocateMemory(mem_reqs, mem_type_index);
  }

  // Bind the buffer to the allocated memory
  VULKAN_CALL(vkBindBufferMemory(device, buffer, memory->memory_, 0));
}

void VulkanBuffer::AllocateMemory(const VkMemoryRequirements& mem_reqs, uint32_t mem_type_index) {
  VkMemoryAllocateInfo mem_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mem_info.allocationSize = mem_reqs.size;
  mem_info.memoryTypeIndex = mem_type_index;

  // Allocate memory
  VkDeviceMemory raw_memory;
  VULKAN_CALL(vkAllocateMemory(device_, &mem_info, nullptr, &raw_memory));

  // Store the allocated memory along with its requirements
  memory = std::make_shared<VulkanMemory>(raw_memory, mem_reqs);
}

VulkanBuffer::~VulkanBuffer() {
  if (buffer) {
    vkDestroyBuffer(device_, buffer, nullptr);
    buffer = VK_NULL_HANDLE;
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other)
    : VulkanResource(std::move(other)), buffer(other.buffer) {
  other.buffer = VK_NULL_HANDLE;
  other.size = 0;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) {
  std::swap(device_, other.device_);
  std::swap(buffer, other.buffer);
  std::swap(memory, other.memory);
  return *this;
}

bool VulkanBuffer::UseDedicatedAllocation(const VulkanDevice& device, VkBuffer buffer,
                                          VkDeviceSize* nbytes) {
  if (device.get_buffer_memory_requirements_2_functions) {
    // Which buffer to request information about
    VkBufferMemoryRequirementsInfo2KHR req_info2 = {
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR};
    req_info2.buffer = buffer;

    // What information to request
    VkMemoryDedicatedRequirementsKHR dedicated_req;
    dedicated_req.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
    dedicated_req.pNext = nullptr;

    VkMemoryRequirements2KHR req2 = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR};
    req2.pNext = &dedicated_req;

    device.get_buffer_memory_requirements_2_functions->vkGetBufferMemoryRequirements2KHR(
        device, &req_info2, &req2);
    if (dedicated_req.requiresDedicatedAllocation || dedicated_req.prefersDedicatedAllocation) {
      *nbytes = req2.memoryRequirements.size;
      return true;
    }
  }

  return false;
}

VulkanHostVisibleBuffer::VulkanHostVisibleBuffer(const VulkanDevice& device, size_t nbytes,
                                                 VkBufferUsageFlags usage, uint32_t mem_type_index,
                                                 std::optional<std::string> mem_scope)
    : vk_buf(device, nbytes, usage, mem_type_index, mem_scope), size(nbytes) {
  VULKAN_CALL(vkMapMemory(device, vk_buf.memory->memory_, 0, size, 0, &host_addr));
}

VulkanHostVisibleBuffer::~VulkanHostVisibleBuffer() {
  if (host_addr) {
    vkUnmapMemory(vk_buf.device_, vk_buf.memory->memory_);
  }
}

VulkanHostVisibleBuffer::VulkanHostVisibleBuffer(VulkanHostVisibleBuffer&& other)
    : vk_buf(std::move(other.vk_buf)), host_addr(other.host_addr), size(other.size) {
  other.host_addr = nullptr;
  other.size = 0;
}

VulkanHostVisibleBuffer& VulkanHostVisibleBuffer::operator=(VulkanHostVisibleBuffer&& other) {
  std::swap(vk_buf, other.vk_buf);
  std::swap(host_addr, other.host_addr);
  std::swap(size, other.size);

  return *this;
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
