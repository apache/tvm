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

#include <algorithm>
#include <unordered_map>

#include "vulkan_common.h"
#include "vulkan_device_api.h"
#include "vulkan_thread_entry.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanDeviceProperties::VulkanDeviceProperties(VkInstance instance, VkPhysicalDevice phy_dev,
                                               const std::vector<const char*> instance_extensions,
                                               const std::vector<const char*> device_extensions) {
  auto has_instance_extension = [&](const char* query) {
    return std::any_of(instance_extensions.begin(), instance_extensions.end(),
                       [&](const char* extension) { return std::strcmp(query, extension) == 0; });
  };

  auto has_device_extension = [&](const char* query) {
    return std::any_of(device_extensions.begin(), device_extensions.end(),
                       [&](const char* extension) { return std::strcmp(query, extension) == 0; });
  };

  ///////////////////////////////////////////////////////////////
  //           Query properties from Vulkan API                //
  ///////////////////////////////////////////////////////////////

  // Declare output locations for properties
  VkPhysicalDeviceProperties2 properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceDriverProperties driver = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties subgroup = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};

  // Need to do initial query in order to check the apiVersion.
  vkGetPhysicalDeviceProperties(phy_dev, &properties.properties);

  // Set up linked list for property query
  {
    void** pp_next = &properties.pNext;
    if (has_device_extension("VK_KHR_driver_properties")) {
      *pp_next = &driver;
      pp_next = &driver.pNext;
    }
    if (properties.properties.apiVersion >= VK_API_VERSION_1_1) {
      *pp_next = &subgroup;
      pp_next = &subgroup.pNext;
    }
  }

  // Declare output locations for features
  VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDevice8BitStorageFeatures storage_8bit = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
  VkPhysicalDevice16BitStorageFeatures storage_16bit = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceShaderFloat16Int8Features float16_int8 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};

  // Set up linked list for feature query
  {
    void** pp_next = &features.pNext;
    if (has_device_extension("VK_KHR_8bit_storage")) {
      *pp_next = &storage_8bit;
      pp_next = &storage_8bit.pNext;
    }
    if (has_device_extension("VK_KHR_16bit_storage")) {
      *pp_next = &storage_16bit;
      pp_next = &storage_16bit.pNext;
    }
    if (has_device_extension("VK_KHR_shader_float16_int8")) {
      *pp_next = &float16_int8;
      pp_next = &float16_int8.pNext;
    }
  }

  if (has_instance_extension("VK_KHR_get_physical_device_properties2")) {
    // Preferred method, call to get all properties that can be queried.
    auto vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)ICHECK_NOTNULL(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR"));
    vkGetPhysicalDeviceProperties2KHR(phy_dev, &properties);

    auto vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)ICHECK_NOTNULL(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
    vkGetPhysicalDeviceFeatures2KHR(phy_dev, &features);
  } else {
    // Fallback, get as many features as we can from the Vulkan1.0
    // API.  Corresponding vkGetPhysicalDeviceProperties was already done earlier.
    vkGetPhysicalDeviceFeatures(phy_dev, &features.features);
  }

  ///////////////////////////////////////////////////////////////
  //     Fill member variables from Vulkan structures          //
  ///////////////////////////////////////////////////////////////

  supports_float16 = float16_int8.shaderFloat16;
  supports_float32 = true;
  supports_float64 = features.features.shaderFloat64;
  supports_int8 = float16_int8.shaderInt8;
  supports_int16 = features.features.shaderInt16;
  supports_int32 = true;
  supports_int64 = features.features.shaderInt64;
  supports_8bit_buffer = storage_8bit.storageBuffer8BitAccess;
  supports_16bit_buffer = storage_16bit.storageBuffer16BitAccess;
  supports_storage_buffer_storage_class =
      has_device_extension("VK_KHR_storage_buffer_storage_class");

  // Support is available based on these extensions, but allow it to
  // be disabled based on an environment variable.
  supports_push_descriptor = has_device_extension("VK_KHR_push_descriptor") &&
                             has_device_extension("VK_KHR_descriptor_update_template");
  {
    const char* disable = std::getenv("TVM_VULKAN_DISABLE_PUSH_DESCRIPTOR");
    if (disable && *disable) {
      supports_push_descriptor = false;
    }
  }

  // Support is available based on these extensions, but allow it to
  // be disabled based on an environment variable.
  supports_dedicated_allocation = has_device_extension("VK_KHR_get_memory_requirements2") &&
                                  has_device_extension("VK_KHR_dedicated_allocation");
  {
    const char* disable = std::getenv("TVM_VULKAN_DISABLE_DEDICATED_ALLOCATION");
    if (disable && *disable) {
      supports_dedicated_allocation = false;
    }
  }

  // The check of VK_SHADER_STAGE_COMPUTE_BIT isn't technically
  // needed, since it will be set so long at least one queue has
  // VK_QUEUE_COMPUTE_BIT.  Including it to avoid potential future
  // confusion..
  supported_subgroup_operations =
      (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? subgroup.supportedOperations : 0;

  max_num_threads = properties.properties.limits.maxComputeWorkGroupInvocations;

  // Even if we can't query it, warp size must be at least 1.
  thread_warp_size = std::max(subgroup.subgroupSize, 1U);

  max_block_size_x = properties.properties.limits.maxComputeWorkGroupSize[0];
  max_block_size_y = properties.properties.limits.maxComputeWorkGroupSize[1];
  max_block_size_z = properties.properties.limits.maxComputeWorkGroupSize[2];
  max_push_constants_size = properties.properties.limits.maxPushConstantsSize;
  max_uniform_buffer_range = properties.properties.limits.maxUniformBufferRange;
  max_storage_buffer_range = properties.properties.limits.maxStorageBufferRange;
  max_per_stage_descriptor_storage_buffer =
      properties.properties.limits.maxPerStageDescriptorStorageBuffers;
  max_shared_memory_per_block = properties.properties.limits.maxComputeSharedMemorySize;
  device_name = properties.properties.deviceName;
  driver_version = properties.properties.driverVersion;

  // By default, use the maximum API version that the driver allows,
  // so that any supported features can be used by TVM shaders.
  // However, if we can query the conformance version, then limit to
  // only using the api version that passes the vulkan conformance
  // tests.
  vulkan_api_version = properties.properties.apiVersion;
  if (has_device_extension("VK_KHR_driver_properties")) {
    auto api_major = VK_VERSION_MAJOR(vulkan_api_version);
    auto api_minor = VK_VERSION_MINOR(vulkan_api_version);
    if ((api_major > driver.conformanceVersion.major) ||
        ((api_major == driver.conformanceVersion.major) &&
         (api_minor > driver.conformanceVersion.minor))) {
      vulkan_api_version =
          VK_MAKE_VERSION(driver.conformanceVersion.major, driver.conformanceVersion.minor, 0);
    }
  }

  // From "Versions and Formats" section of Vulkan spec.
  max_spirv_version = 0x10000;
  if (vulkan_api_version >= VK_API_VERSION_1_2) {
    max_spirv_version = 0x10500;
  } else if (has_device_extension("VK_KHR_spirv_1_4")) {
    max_spirv_version = 0x10400;
  } else if (vulkan_api_version >= VK_API_VERSION_1_1) {
    max_spirv_version = 0x10300;
  }
}

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
