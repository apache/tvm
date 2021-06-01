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

#include "vulkan_context.h"

#include <unordered_map>

#include "vulkan_common.h"
#include "vulkan_device_api.h"
#include "vulkan_thread_entry.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanDescriptorTemplateKHRFunctions::VulkanDescriptorTemplateKHRFunctions(VkDevice device) {
  vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)ICHECK_NOTNULL(
      vkGetDeviceProcAddr(device, "vkCreateDescriptorUpdateTemplateKHR"));
  vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)ICHECK_NOTNULL(
      vkGetDeviceProcAddr(device, "vkDestroyDescriptorUpdateTemplateKHR"));
  vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)ICHECK_NOTNULL(
      vkGetDeviceProcAddr(device, "vkUpdateDescriptorSetWithTemplateKHR"));
  vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)ICHECK_NOTNULL(
      vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetWithTemplateKHR"));
}

VulkanGetBufferMemoryRequirements2Functions::VulkanGetBufferMemoryRequirements2Functions(
    VkDevice device) {
  vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)ICHECK_NOTNULL(
      vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2KHR"));
}

uint32_t FindMemoryType(const VulkanContext& vctx, VkBufferCreateInfo info,
                        VkMemoryPropertyFlags req_prop) {
  VkBuffer buffer;
  VULKAN_CALL(vkCreateBuffer(vctx.device, &info, nullptr, &buffer));

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(vctx.device, buffer, &mem_reqs);
  uint32_t type_bits = mem_reqs.memoryTypeBits;
  VkPhysicalDeviceMemoryProperties phy_mem_prop;
  vkGetPhysicalDeviceMemoryProperties(vctx.phy_device, &phy_mem_prop);
  for (uint32_t i = 0; i < phy_mem_prop.memoryTypeCount; i++) {
    if ((type_bits & 1) == 1 &&
        (phy_mem_prop.memoryTypes[i].propertyFlags & req_prop) == req_prop) {
      return i;
    }
    type_bits >>= 1;
  }
  LOG(FATAL) << "Requested memory type not found";
  return 0;
}

VkBufferCreateInfo MakeBufferCreateInfo(const VulkanContext& vctx, size_t nbytes,
                                        VkBufferUsageFlags usage) {
  VkBufferCreateInfo info;
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0;
  info.size = nbytes;
  info.queueFamilyIndexCount = 1;
  info.pQueueFamilyIndices = &(vctx.queue_family_index);
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.usage = usage;
  return info;
}

VulkanBuffer* CreateBuffer(const VulkanContext& vctx, size_t nbytes, VkBufferUsageFlags usage,
                           uint32_t mem_type_index) {
  auto info = MakeBufferCreateInfo(vctx, nbytes, usage);
  // create buffer
  VkBuffer buffer;
  VULKAN_CALL(vkCreateBuffer(vctx.device, &info, nullptr, &buffer));

  // bind to memory
  bool dedicated_allocation = false;
  VkMemoryRequirements2KHR req2;

  if (vctx.get_buffer_memory_requirements_2_functions) {
    VkBufferMemoryRequirementsInfo2KHR req_info2;
    req_info2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR;
    req_info2.pNext = 0;
    req_info2.buffer = buffer;

    req2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
    req2.pNext = 0;

    VkMemoryDedicatedRequirementsKHR dedicated_req;
    dedicated_req.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
    dedicated_req.pNext = 0;
    req2.pNext = &dedicated_req;

    vctx.get_buffer_memory_requirements_2_functions->vkGetBufferMemoryRequirements2KHR(
        vctx.device, &req_info2, &req2);
    dedicated_allocation =
        dedicated_req.requiresDedicatedAllocation || dedicated_req.prefersDedicatedAllocation;
  }

  VkDeviceMemory memory;
  if (!dedicated_allocation) {
    VkMemoryAllocateInfo minfo;
    minfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    minfo.pNext = nullptr;
    minfo.allocationSize = info.size;
    minfo.memoryTypeIndex = mem_type_index;
    VULKAN_CALL(vkAllocateMemory(vctx.device, &minfo, nullptr, &memory));
  } else {
    VkMemoryAllocateInfo minfo;
    minfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    minfo.pNext = nullptr;
    minfo.allocationSize = req2.memoryRequirements.size;
    minfo.memoryTypeIndex = mem_type_index;

    VkMemoryDedicatedAllocateInfoKHR mdinfo;
    mdinfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
    mdinfo.pNext = 0;
    mdinfo.image = 0;
    mdinfo.buffer = buffer;
    minfo.pNext = &mdinfo;
    VULKAN_CALL(vkAllocateMemory(vctx.device, &minfo, nullptr, &memory));
  }
  VULKAN_CALL(vkBindBufferMemory(vctx.device, buffer, memory, 0));
  VulkanBuffer* pbuf = new VulkanBuffer();
  pbuf->memory = memory;
  pbuf->buffer = buffer;
  return pbuf;
}

VulkanHostVisibleBuffer* GetOrAllocate(
    int device_id, size_t size, VkBufferUsageFlags usage, uint32_t mem_type_index,
    std::unordered_map<size_t, std::unique_ptr<VulkanHostVisibleBuffer>>* buffers_ptr,
    bool sync_before_realloc) {
  auto& buffers = *buffers_ptr;
  if (!buffers[device_id]) {
    buffers[device_id] = std::make_unique<VulkanHostVisibleBuffer>();
  }

  auto& buf = *(buffers[device_id]);
  if (buf.device != nullptr && buf.size < size) {
    // free previous buffer
    if (sync_before_realloc) {
      // For the deferred execution mode, we need to make sure that old tasks that use
      // the older, smaller buffer get finished
      // Synchronization on staging buffers is done after host to device memory copy
      // For UBO, we sync here before we reallocate a larger buffer, to minimize synchronization
      // points
      VulkanThreadEntry::ThreadLocal()->Stream(device_id)->Synchronize();
    }
    DeleteHostVisibleBuffer(&buf);
  }

  const auto& vctx = VulkanDeviceAPI::Global()->context(device_id);

  if (buf.device == nullptr) {
    buf.device = vctx.device;
  }
  if (buf.host_addr == nullptr) {
    buf.vk_buf = CreateBuffer(vctx, size, usage, mem_type_index);
    VULKAN_CALL(vkMapMemory(vctx.device, buf.vk_buf->memory, 0, size, 0, &(buf.host_addr)));
    buf.size = size;
  }
  return &buf;
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
