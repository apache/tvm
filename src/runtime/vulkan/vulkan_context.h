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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_CONTEXT_H_
#define TVM_RUNTIME_VULKAN_VULKAN_CONTEXT_H_

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>

#include <memory>
#include <string>
#include <vector>

#include "vulkan/vulkan_core.h"
#include "vulkan_buffer.h"

namespace tvm {
namespace runtime {
namespace vulkan {

struct VulkanDescriptorTemplateKHRFunctions {
  explicit VulkanDescriptorTemplateKHRFunctions(VkDevice device);

  PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR{nullptr};
  PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR{nullptr};
  PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR{nullptr};
  PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR{nullptr};
};

struct VulkanGetBufferMemoryRequirements2Functions {
  explicit VulkanGetBufferMemoryRequirements2Functions(VkDevice device);

  PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR{nullptr};
};

/*!
 * \brief Stores the capabilities/limits queried from the physical device.
 *
 * The member variables here have a 1-1 mapping to Target parameters,
 * if target->kind->device_type==kDLVulkan.  A separate struct is used
 * to maintain the boundary between the Vulkan runtime in
 * libtvm_runtime.so, and the Target object in libtvm.so.
 */
struct VulkanDeviceProperties {
  VulkanDeviceProperties() {}
  VulkanDeviceProperties(VkInstance instance, VkPhysicalDevice phy_device,
                         const std::vector<const char*> instance_extensions,
                         const std::vector<const char*> device_extensions);

  bool supports_float16{false};
  bool supports_float32{true};
  bool supports_float64{false};
  bool supports_int8{false};
  bool supports_int16{false};
  bool supports_int32{true};
  bool supports_int64{false};
  bool supports_8bit_buffer{false};
  bool supports_16bit_buffer{false};
  bool supports_storage_buffer_storage_class{false};
  bool supports_push_descriptor{false};
  bool supports_dedicated_allocation{false};
  uint32_t supported_subgroup_operations{0};
  uint32_t max_num_threads{1};
  uint32_t thread_warp_size{1};
  uint32_t max_block_size_x{1};
  uint32_t max_block_size_y{1};
  uint32_t max_block_size_z{1};
  uint32_t max_push_constants_size{128};
  uint32_t max_uniform_buffer_range{16384};
  uint32_t max_storage_buffer_range{1 << 27};
  uint32_t max_per_stage_descriptor_storage_buffer{4};
  uint32_t max_shared_memory_per_block{16384};
  std::string device_name{"unknown device name"};
  uint32_t driver_version{0};
  uint32_t vulkan_api_version{VK_API_VERSION_1_0};
  uint32_t max_spirv_version{0x10000};
};

struct VulkanContext {
  // physical device
  VkPhysicalDevice phy_device{nullptr};

  // Cached device properties, queried through Vulkan API.
  VulkanDeviceProperties device_properties;

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

  bool UseImmediate() const { return descriptor_template_khr_functions != nullptr; }
};

uint32_t FindMemoryType(const VulkanContext& vctx, VkBufferCreateInfo info,
                        VkMemoryPropertyFlags req_prop);

VkBufferCreateInfo MakeBufferCreateInfo(const VulkanContext& vctx, size_t nbytes,
                                        VkBufferUsageFlags usage);

VulkanBuffer* CreateBuffer(const VulkanContext& vctx, size_t nbytes, VkBufferUsageFlags usage,
                           uint32_t mem_type_index);

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_CONTEXT_H_
