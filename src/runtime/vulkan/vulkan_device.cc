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

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "../../support/utils.h"
#include "vulkan_common.h"
#include "vulkan_device.h"
#include "vulkan_device_api.h"
#include "vulkan_instance.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanDeviceProperties::VulkanDeviceProperties(const VulkanInstance& instance,
                                               const VulkanDevice& device) {
  ///////////////////////////////////////////////////////////////
  //           Query properties from Vulkan API                //
  ///////////////////////////////////////////////////////////////

  // Declare output locations for properties
  VkPhysicalDeviceProperties2 properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceDriverProperties driver = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties subgroup = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};

  // Need to do initial query in order to check the apiVersion.
  vkGetPhysicalDeviceProperties(device, &properties.properties);

  // Set up linked list for property query
  {
    void** pp_next = &properties.pNext;
    if (device.HasExtension("VK_KHR_driver_properties")) {
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
    if (device.HasExtension("VK_KHR_8bit_storage")) {
      *pp_next = &storage_8bit;
      pp_next = &storage_8bit.pNext;
    }
    if (device.HasExtension("VK_KHR_16bit_storage")) {
      *pp_next = &storage_16bit;
      pp_next = &storage_16bit.pNext;
    }
    if (device.HasExtension("VK_KHR_shader_float16_int8")) {
      *pp_next = &float16_int8;
      pp_next = &float16_int8.pNext;
    }
  }

  if (instance.HasExtension("VK_KHR_get_physical_device_properties2")) {
    // Preferred method, call to get all properties that can be queried.
    auto vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)ICHECK_NOTNULL(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR"));
    vkGetPhysicalDeviceProperties2KHR(device, &properties);

    auto vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)ICHECK_NOTNULL(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
    vkGetPhysicalDeviceFeatures2KHR(device, &features);
  } else {
    // Fallback, get as many features as we can from the Vulkan1.0
    // API.  Corresponding vkGetPhysicalDeviceProperties was already done earlier.
    vkGetPhysicalDeviceFeatures(device, &features.features);
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
      device.HasExtension("VK_KHR_storage_buffer_storage_class");

  // Support is available based on these extensions, but allow it to
  // be disabled based on an environment variable.
  supports_push_descriptor = device.HasExtension("VK_KHR_push_descriptor") &&
                             device.HasExtension("VK_KHR_descriptor_update_template") &&
                             !support::BoolEnvironmentVar("TVM_VULKAN_DISABLE_PUSH_DESCRIPTOR");

  // Support is available based on these extensions, but allow it to
  // be disabled based on an environment variable.
  supports_dedicated_allocation =
      device.HasExtension("VK_KHR_get_memory_requirements2") &&
      device.HasExtension("VK_KHR_dedicated_allocation") &&
      !support::BoolEnvironmentVar("TVM_VULKAN_DISABLE_DEDICATED_ALLOCATION");

  supports_integer_dot_product = device.HasExtension("VK_KHR_shader_integer_dot_product");

  supports_cooperative_matrix = device.HasExtension("VK_NV_cooperative_matrix");

  // The check of VK_SHADER_STAGE_COMPUTE_BIT isn't technically
  // needed, since it will be set so long at least one queue has
  // VK_QUEUE_COMPUTE_BIT.  Including it to avoid potential future
  // confusion..
  supported_subgroup_operations =
      (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? subgroup.supportedOperations : 0;

  max_num_threads = properties.properties.limits.maxComputeWorkGroupInvocations;

  // Even if we can't query it, warp size must be at least 1.
  // thread_warp_size = std::max(subgroup.subgroupSize, 1U);
  // vulkan's subgroup may not directly map to warp and atm
  // can cause issues in softmax allreduce in NVidia GPU
  // disable warp setting to be safe.
  thread_warp_size = 1U;

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

  if (device.HasExtension("VK_KHR_driver_properties")) {
    driver_name = driver.driverName;
  }

  switch (properties.properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
      device_type = "other";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      device_type = "integrated";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      device_type = "discrete";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      device_type = "virtual";
      break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      device_type = "cpu";
      break;
    default:
      LOG(FATAL) << "Unknown vulkan device type: " << properties.properties.deviceType;
      break;
  }

  // By default, use the maximum API version that the driver allows,
  // so that any supported features can be used by TVM shaders.
  // However, if we can query the conformance version, then limit to
  // only using the api version that passes the vulkan conformance
  // tests.
  vulkan_api_version = properties.properties.apiVersion;
  if (device.HasExtension("VK_KHR_driver_properties")) {
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
  } else if (device.HasExtension("VK_KHR_spirv_1_4")) {
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

VulkanQueueInsertDebugUtilsLabelFunctions::VulkanQueueInsertDebugUtilsLabelFunctions(
    VkInstance instance) {
  vkQueueInsertDebugUtilsLabelEXT = (PFN_vkQueueInsertDebugUtilsLabelEXT)ICHECK_NOTNULL(
      vkGetInstanceProcAddr(instance, "vkQueueInsertDebugUtilsLabelEXT"));
}

VulkanDevice::VulkanDevice(const VulkanInstance& instance, VkPhysicalDevice phy_device)
    : physical_device_(phy_device) {
  queue_family_index = SelectComputeQueueFamily();
  if (queue_family_index == uint32_t(-1)) {
    // The GPU doesn't support compute, cannot use
    return;
  }

  enabled_extensions = SelectEnabledExtensions();
  device_properties = VulkanDeviceProperties(instance, *this);
  CreateVkDevice(instance);

  // Currently, any exceptions called after this point will prevent
  // vkDestroyDevice from being called in the destructor.  If this
  // becomes an issue, can split out the VulkanDevice into two
  // classes, one of which strictly holds the VkDevice, and one which
  // holds the ancillary handles that TVM needs.

  vkGetDeviceQueue(device_, queue_family_index, 0, &queue);

  // Find suitable memory type for staging and compute
  // Find suitable compute index.
  VkBuffer buffer;
  VkMemoryRequirements req_staging, req_compute;
  VkBufferCreateInfo info;
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0;
  info.size = 1024;
  info.queueFamilyIndexCount = 1;
  info.pQueueFamilyIndices = &queue_family_index;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  // get staging requirement
  info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VULKAN_CALL(vkCreateBuffer(device_, &info, nullptr, &buffer));
  vkGetBufferMemoryRequirements(device_, buffer, &req_staging);
  vkDestroyBuffer(device_, buffer, nullptr);
  // get compute requirement
  info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VULKAN_CALL(vkCreateBuffer(device_, &info, nullptr, &buffer));
  vkGetBufferMemoryRequirements(device_, buffer, &req_compute);
  vkDestroyBuffer(device_, buffer, nullptr);

  // Query phyiscal device property
  // find a memory that is host visible, no need to be consistent
  int win_rank = -1;
  VkPhysicalDeviceMemoryProperties prop;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &prop);

  for (uint32_t k = 0; k < prop.memoryTypeCount; ++k) {
    VkMemoryType ty = prop.memoryTypes[k];
    int64_t heap_size = static_cast<int64_t>(prop.memoryHeaps[ty.heapIndex].size);
    // host visible
    if (!(ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
    // match copy requirment
    if (!(req_staging.memoryTypeBits & (1 << k))) continue;
    if (heap_size < 1024) continue;
    int rank = 0;
    rank += ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    if (rank > win_rank) {
      win_rank = rank;
      staging_mtype_index = k;
      coherent_staging = ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
  }
  ICHECK_GE(win_rank, 0) << "Cannot find suitable staging memory on device.";

  win_rank = -1;
  for (uint32_t k = 0; k < prop.memoryTypeCount; ++k) {
    VkMemoryType ty = prop.memoryTypes[k];
    int64_t heap_size = static_cast<int64_t>(prop.memoryHeaps[ty.heapIndex].size);
    // host visible
    if (!(ty.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
    // match copy requirment
    if (!(req_staging.memoryTypeBits & (1 << k))) continue;
    if (heap_size < 1024) continue;
    int rank = 0;
    // prefer not host visible
    rank += !(ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (rank > win_rank) {
      win_rank = rank;
      compute_mtype_index = k;
      compute_memory_size = heap_size;
    }
  }

  ICHECK_GE(win_rank, 0) << "Cannot find suitable local memory on device.";

  if (device_properties.supports_push_descriptor) {
    descriptor_template_khr_functions =
        std::make_unique<VulkanDescriptorTemplateKHRFunctions>(device_);
  }

  if (device_properties.supports_dedicated_allocation) {
    get_buffer_memory_requirements_2_functions =
        std::make_unique<VulkanGetBufferMemoryRequirements2Functions>(device_);
  }

  if (instance.HasExtension("VK_EXT_debug_utils")) {
    queue_insert_debug_utils_label_functions =
        std::make_unique<VulkanQueueInsertDebugUtilsLabelFunctions>(instance);
  }
}

VulkanDevice::~VulkanDevice() {
  // Need to clear anything that uses this device calling
  // vkDestroyDevice.  Might be a sign that the VkDevice should be
  // held by member variable rather than beind owned directly by
  // VulkanDevice.
  stream_per_thread.Clear();
  staging_buffer_per_thread.Clear();
  uniform_buffer_per_thread.Clear();

  if (device_) {
    vkDestroyDevice(device_, nullptr);
  }
}

VulkanDevice::VulkanDevice(VulkanDevice&& other) { do_swap(std::move(other)); }

VulkanDevice& VulkanDevice::operator=(VulkanDevice&& other) {
  do_swap(std::move(other));
  return *this;
}

void VulkanDevice::do_swap(VulkanDevice&& other) {
  if (this == &other) {
    return;
  }

  std::lock(queue_mutex, other.queue_mutex);
  std::lock_guard<std::mutex> lock_self(queue_mutex, std::adopt_lock);
  std::lock_guard<std::mutex> lock_other(other.queue_mutex, std::adopt_lock);

  std::swap(device_properties, other.device_properties);
  std::swap(staging_mtype_index, other.staging_mtype_index);
  std::swap(coherent_staging, other.coherent_staging);
  std::swap(descriptor_template_khr_functions, other.descriptor_template_khr_functions);
  std::swap(get_buffer_memory_requirements_2_functions,
            other.get_buffer_memory_requirements_2_functions);
  std::swap(queue_insert_debug_utils_label_functions,
            other.queue_insert_debug_utils_label_functions);
  std::swap(compute_mtype_index, other.compute_mtype_index);
  std::swap(compute_memory_size, other.compute_memory_size);
  std::swap(queue, other.queue);
  std::swap(queue_family_index, other.queue_family_index);
  std::swap(physical_device_, other.physical_device_);
  std::swap(enabled_extensions, other.enabled_extensions);
  std::swap(device_, other.device_);
}

bool VulkanDevice::SupportsCompute() const { return queue_family_index != uint32_t(-1); }

void VulkanDevice::QueueSubmit(VkSubmitInfo submit_info, VkFence fence) const {
  // Multiple streams (on different threads) use the same VulkanDevice
  // instance, so we need to externally synchronize accesses.
  std::lock_guard<std::mutex> lock(queue_mutex);
  VULKAN_CALL(vkQueueSubmit(queue, 1, &submit_info, fence));
}

uint32_t VulkanDevice::SelectComputeQueueFamily() const {
  // Get a queue family that supports compute. We currently only use
  // one queue from one family.
  uint32_t queue_prop_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_prop_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_props(queue_prop_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_prop_count,
                                           dmlc::BeginPtr(queue_props));

  std::vector<uint32_t> result;
  // Prefer compute-only queues. On certain devices supporting this (e.g. Mesa RADV), using
  // compute-only queues gives better responsiveness for other graphics workload (e.g. desktop).
  for (uint32_t i = 0; i != queue_prop_count; ++i) {
    if ((VK_QUEUE_COMPUTE_BIT & queue_props[i].queueFlags) != 0 &&
        (VK_QUEUE_GRAPHICS_BIT & queue_props[i].queueFlags) == 0) {
      return i;
    }
  }
  // Now, push the compute queues that we skipped above into the list.
  for (uint32_t i = 0; i != queue_prop_count; ++i) {
    if ((VK_QUEUE_COMPUTE_BIT & queue_props[i].queueFlags) != 0 &&
        (VK_QUEUE_GRAPHICS_BIT & queue_props[i].queueFlags) != 0) {
      return i;
    }
  }

  // No queues support compute capability, this GPU cannot be used.
  return -1;
}

std::vector<const char*> VulkanDevice::SelectEnabledExtensions() const {
  std::vector<const char*> required_extensions{};
  std::vector<const char*> optional_extensions{"VK_KHR_driver_properties",
                                               "VK_KHR_storage_buffer_storage_class",
                                               "VK_KHR_8bit_storage",
                                               "VK_KHR_16bit_storage",
                                               "VK_KHR_shader_float16_int8",
                                               "VK_KHR_push_descriptor",
                                               "VK_KHR_descriptor_update_template",
                                               "VK_KHR_get_memory_requirements2",
                                               "VK_KHR_dedicated_allocation",
                                               "VK_KHR_spirv_1_4",
                                               "VK_KHR_shader_integer_dot_product",
                                               "VK_NV_cooperative_matrix"};

  uint32_t device_extension_prop_count;
  VULKAN_CALL(vkEnumerateDeviceExtensionProperties(physical_device_, nullptr,
                                                   &device_extension_prop_count, nullptr));
  std::vector<VkExtensionProperties> device_extension_prop(device_extension_prop_count);
  VULKAN_CALL(vkEnumerateDeviceExtensionProperties(
      physical_device_, nullptr, &device_extension_prop_count, device_extension_prop.data()));

  return FindEnabledExtensions(device_extension_prop, required_extensions, optional_extensions);
}

bool VulkanDevice::HasExtension(const char* query) const {
  return std::any_of(enabled_extensions.begin(), enabled_extensions.end(),
                     [&](const char* extension) { return std::strcmp(query, extension) == 0; });
}

void VulkanDevice::CreateVkDevice(const VulkanInstance& instance) {
  // Enable all features we may use that a device supports.
  VkPhysicalDeviceFeatures2 enabled_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDevice8BitStorageFeatures storage_8bit = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
  VkPhysicalDevice16BitStorageFeatures storage_16bit = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceShaderFloat16Int8Features float16_int8 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};

  void** pp_next = &enabled_features.pNext;
  bool needs_float16_int8 = false;

  if (device_properties.supports_float16) {
    float16_int8.shaderFloat16 = true;
    needs_float16_int8 = true;
  }
  if (device_properties.supports_float64) {
    enabled_features.features.shaderFloat64 = true;
  }
  if (device_properties.supports_int8) {
    float16_int8.shaderInt8 = true;
    needs_float16_int8 = true;
  }
  if (device_properties.supports_int16) {
    enabled_features.features.shaderInt16 = true;
  }
  if (device_properties.supports_int64) {
    enabled_features.features.shaderInt64 = true;
  }
  if (device_properties.supports_8bit_buffer) {
    storage_8bit.storageBuffer8BitAccess = true;
    *pp_next = &storage_8bit;
    pp_next = &storage_8bit.pNext;
  }
  if (device_properties.supports_16bit_buffer) {
    storage_16bit.storageBuffer16BitAccess = true;
    *pp_next = &storage_16bit;
    pp_next = &storage_16bit.pNext;
  }

  if (needs_float16_int8) {
    *pp_next = &float16_int8;
    pp_next = &float16_int8.pNext;
  }

  float priority = 1.0f;

  struct VkDeviceQueueCreateInfo queue_create_info;
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.pNext = nullptr;
  queue_create_info.flags = 0;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_create_info;
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pNext = nullptr;
  device_create_info.flags = 0;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pQueueCreateInfos = &queue_create_info;
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = nullptr;
  device_create_info.enabledExtensionCount = enabled_extensions.size();
  device_create_info.ppEnabledExtensionNames = enabled_extensions.data();

  if (instance.HasExtension("VK_KHR_get_physical_device_properties2")) {
    device_create_info.pEnabledFeatures = nullptr;
    device_create_info.pNext = &enabled_features;
  } else {
    device_create_info.pNext = nullptr;
    device_create_info.pEnabledFeatures = &enabled_features.features;
  }
  VULKAN_CALL(vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_));
}

VulkanStream& VulkanDevice::ThreadLocalStream() {
  return const_cast<VulkanStream&>(const_cast<const VulkanDevice*>(this)->ThreadLocalStream());
}

const VulkanStream& VulkanDevice::ThreadLocalStream() const {
  return stream_per_thread.GetOrMake(this);
}

VulkanStagingBuffer& VulkanDevice::ThreadLocalStagingBuffer(size_t min_size) {
  auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VulkanStagingBuffer& result =
      staging_buffer_per_thread.GetOrMake(*this, min_size, usage, staging_mtype_index);

  if (result.size < min_size) {
    result = VulkanStagingBuffer(*this, min_size, usage, staging_mtype_index);
  }

  return result;
}

void VulkanDevice::AllocateThreadLocalUniformBuffer(size_t min_size) {
  auto usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  auto buffer_info = MakeBufferCreateInfo(min_size, usage);
  auto prop = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  auto mem_type_index = FindMemoryType(*this, buffer_info, prop);

  VulkanUniformBuffer& result =
      uniform_buffer_per_thread.GetOrMake(*this, min_size, usage, mem_type_index);

  if (result.size < min_size) {
    result = VulkanUniformBuffer(*this, min_size, usage, mem_type_index);
  }
}

VulkanStagingBuffer& VulkanDevice::ThreadLocalUniformBuffer(size_t min_size) {
  VulkanStagingBuffer* buffer = uniform_buffer_per_thread.Get();
  ICHECK(buffer) << "Vulkan uniform buffer requested, but not previously allocated.";
  ICHECK_GE(buffer->size, min_size)
      << "Vulkan uniform buffer of size " << min_size << " requested, but only " << buffer->size
      << " was previously allocated.";
  return *buffer;
}

uint32_t FindMemoryType(const VulkanDevice& device, VkBufferCreateInfo info,
                        VkMemoryPropertyFlags req_prop) {
  VkBuffer buffer;
  VULKAN_CALL(vkCreateBuffer(device, &info, nullptr, &buffer));

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
  uint32_t type_bits = mem_reqs.memoryTypeBits;
  VkPhysicalDeviceMemoryProperties phy_mem_prop;
  vkGetPhysicalDeviceMemoryProperties(device, &phy_mem_prop);
  for (uint32_t i = 0; i < phy_mem_prop.memoryTypeCount; i++) {
    if ((type_bits & 1) == 1 &&
        (phy_mem_prop.memoryTypes[i].propertyFlags & req_prop) == req_prop) {
      return i;
    }
    type_bits >>= 1;
  }
  LOG(FATAL) << "Requested memory type not found";
}

VulkanHostVisibleBuffer* GetOrAllocate(
    int device_id, size_t size, VkBufferUsageFlags usage, uint32_t mem_type_index,
    std::unordered_map<size_t, std::unique_ptr<VulkanHostVisibleBuffer>>* buffers_ptr,
    bool sync_before_realloc) {
  auto& device = VulkanDeviceAPI::Global()->device(device_id);

  auto& buffers = *buffers_ptr;

  bool needs_alloc = !buffers[device_id] || (buffers[device_id]->size < size);
  bool is_realloc = buffers[device_id] && (buffers[device_id]->size < size);
  if (is_realloc && sync_before_realloc) {
    device.ThreadLocalStream().Synchronize();
  }

  if (needs_alloc) {
    auto new_buffer =
        std::make_unique<VulkanHostVisibleBuffer>(device, size, usage, mem_type_index);
    buffers[device_id] = std::move(new_buffer);
  }
  return buffers[device_id].get();
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
