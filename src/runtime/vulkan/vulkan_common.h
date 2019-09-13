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
#pragma once

#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>

#include <vulkan/vulkan.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace vulkan {

inline const char* VKGetErrorString(VkResult error) {
  switch (error) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_EVENT_SET:
      return "VK_EVENT_SET";
    case VK_EVENT_RESET:
      return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
      return "VK_ERROR_FRAGMENTED_POOL";
    default:
      return "Unknown Vulkan error code";
  }
}

/*!
 * \brief Protected Vulkan call
 * \param func Expression to call.
 */
#define VULKAN_CHECK_ERROR(__e)                                     \
  {                                                                 \
    CHECK(__e == VK_SUCCESS) << "Vulan Error, code=" << __e << ": " \
                             << vulkan::VKGetErrorString(__e);      \
  }

#define VULKAN_CALL(func)    \
  {                          \
    VkResult __e = (func);   \
    VULKAN_CHECK_ERROR(__e); \
  }

struct VulkanDescriptorTemplateKHRFunctions {
  PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR{nullptr};
  PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR{nullptr};
  PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR{nullptr};
  PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR{nullptr};
};

struct VulkanGetBufferMemoryRequirements2Functions {
  PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR{nullptr};
};

struct VulkanStagingBuffer {
  VkDevice device{nullptr};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  void* host_addr{nullptr};
  size_t size{0};
};

struct VulkanContext {
  // phyiscal device
  VkPhysicalDevice phy_device{nullptr};
  // Phyiscal device property
  VkPhysicalDeviceProperties phy_device_prop;
  // Memory type index for staging.
  uint32_t staging_mtype_index{0};
  // whether staging is coherent
  bool coherent_staging{false};

  std::unique_ptr<VulkanDescriptorTemplateKHRFunctions> descriptor_template_khr_functions{nullptr};
  std::unique_ptr<VulkanGetBufferMemoryRequirements2Functions>
      get_buffer_memory_requirements_2_functions{nullptr};
  // Memory type index for compute
  uint32_t compute_mtype_index{0};
  // The logical device
  VkDevice device{nullptr};
  // command queue

  std::unique_ptr<std::mutex> queue_mutex;
  VkQueue queue{nullptr};
  // queue family_index;
  uint32_t queue_family_index{0};
  // Queue family index.
  VkQueueFamilyProperties queue_prop;

  bool UseImmediate() const { return descriptor_template_khr_functions.get() != nullptr; }
};


}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
