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

#include "vulkan_device_api.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "vulkan_thread_entry.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanDeviceAPI* VulkanDeviceAPI::Global() {
  // Most of the TVM Global() functions allocate with "new" and do
  // not deallocate, as the OS can clean up any leftover buffers at
  // the end.  In this case, we need the VulkanDeviceAPI destructor
  // to call vkDestroyInstance, to prevent a segfault on exit when
  // using some nvidia drivers.
  static VulkanDeviceAPI inst;
  return &inst;
}

VulkanDeviceAPI::VulkanDeviceAPI() {
  const auto layers = []() -> std::vector<const char*> {
    uint32_t inst_layer_prop_count;
    VULKAN_CALL(vkEnumerateInstanceLayerProperties(&inst_layer_prop_count, nullptr));
    std::vector<VkLayerProperties> inst_layer_prop(inst_layer_prop_count);
    VULKAN_CALL(vkEnumerateInstanceLayerProperties(&inst_layer_prop_count, inst_layer_prop.data()));
    std::vector<const char*> l;

    const char* enable = std::getenv("TVM_VULKAN_ENABLE_VALIDATION_LAYERS");
    bool validation_enabled = enable && *enable;
    if (validation_enabled) {
      for (const auto& lp : inst_layer_prop) {
        if (std::strcmp(lp.layerName, "VK_LAYER_LUNARG_standard_validation") == 0) {
          l.push_back("VK_LAYER_LUNARG_standard_validation");
        }
        if (std::strcmp(lp.layerName, "VK_LAYER_LUNARG_parameter_validation") == 0) {
          l.push_back("VK_LAYER_LUNARG_parameter_validation");
        }
        if (std::strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
          l.push_back("VK_LAYER_KHRONOS_validation");
        }
      }
    }
    return l;
  }();

  const auto instance_extensions = [this]() {
    std::vector<const char*> required_extensions{};
    std::vector<const char*> optional_extensions{"VK_KHR_get_physical_device_properties2"};

    uint32_t inst_extension_prop_count;
    VULKAN_CALL(
        vkEnumerateInstanceExtensionProperties(nullptr, &inst_extension_prop_count, nullptr));
    std::vector<VkExtensionProperties> inst_extension_prop(inst_extension_prop_count);
    VULKAN_CALL(vkEnumerateInstanceExtensionProperties(nullptr, &inst_extension_prop_count,
                                                       inst_extension_prop.data()));

    return FindEnabledExtensions(inst_extension_prop, required_extensions, optional_extensions);
  }();

  auto has_instance_extension = [&instance_extensions](const char* query) {
    return std::any_of(instance_extensions.begin(), instance_extensions.end(),
                       [&](const char* extension) { return std::strcmp(query, extension) == 0; });
  };

  const auto instance_api_version = []() {
    uint32_t api_version = VK_MAKE_VERSION(1, 0, 0);

    // Result from vkGetInstanceProcAddr is NULL if driver only
    // supports vulkan 1.0.
    auto vkEnumerateInstanceVersion =
        (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
    if (vkEnumerateInstanceVersion) {
      vkEnumerateInstanceVersion(&api_version);
    }

    return api_version;
  }();

  {
    VkApplicationInfo app_info;
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = nullptr;
    app_info.pApplicationName = "TVM";
    app_info.applicationVersion = 0;
    app_info.pEngineName = "";
    app_info.engineVersion = 0;
    app_info.apiVersion = instance_api_version;

    VkInstanceCreateInfo inst_info;
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = nullptr;
    inst_info.flags = 0;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledLayerCount = layers.size();
    inst_info.ppEnabledLayerNames = layers.data();
    inst_info.enabledExtensionCount = instance_extensions.size();
    inst_info.ppEnabledExtensionNames = instance_extensions.data();

    VULKAN_CALL(vkCreateInstance(&inst_info, nullptr, &instance_));
  }

  uint32_t phy_dev_count = 0;
  VULKAN_CALL(vkEnumeratePhysicalDevices(instance_, &phy_dev_count, nullptr));
  std::vector<VkPhysicalDevice> all_phy_devs(phy_dev_count);
  VULKAN_CALL(vkEnumeratePhysicalDevices(instance_, &phy_dev_count, dmlc::BeginPtr(all_phy_devs)));
  for (VkPhysicalDevice phy_dev : all_phy_devs) {
    // Get a list of queue families supporting compute, in order of preference. We currently only
    // make use of the most preferred one family.
    std::vector<uint32_t> queue_family_indexes = GetComputeQueueFamilies(phy_dev);
    if (queue_family_indexes.empty()) continue;
    uint32_t queue_family_index = queue_family_indexes[0];
    float priority = 1.0f;

    struct VkDeviceQueueCreateInfo queue_create_info;
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.pNext = nullptr;
    queue_create_info.flags = 0;
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &priority;

    VulkanContext ctx;
    // setup context
    ctx.phy_device = phy_dev;
    vkGetPhysicalDeviceProperties(ctx.phy_device, &(ctx.phy_device_prop));

    const auto device_extensions = [&]() {
      std::vector<const char*> required_extensions{};
      std::vector<const char*> optional_extensions{
          "VK_KHR_driver_properties",
          "VK_KHR_storage_buffer_storage_class",
          "VK_KHR_8bit_storage",
          "VK_KHR_16bit_storage",
          "VK_KHR_shader_float16_int8",
          "VK_KHR_push_descriptor",
          "VK_KHR_descriptor_update_template",
          "VK_KHR_get_memory_requirements2",
          "VK_KHR_dedicated_allocation",
          "VK_KHR_spirv_1_4",
      };

      uint32_t device_extension_prop_count;
      VULKAN_CALL(vkEnumerateDeviceExtensionProperties(ctx.phy_device, nullptr,
                                                       &device_extension_prop_count, nullptr));
      std::vector<VkExtensionProperties> device_extension_prop(device_extension_prop_count);
      VULKAN_CALL(vkEnumerateDeviceExtensionProperties(
          ctx.phy_device, nullptr, &device_extension_prop_count, device_extension_prop.data()));

      return FindEnabledExtensions(device_extension_prop, required_extensions, optional_extensions);
    }();

    ctx.device_properties =
        VulkanDeviceProperties(instance_, phy_dev, instance_extensions, device_extensions);

    {
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

      if (ctx.device_properties.supports_float16) {
        float16_int8.shaderFloat16 = true;
        needs_float16_int8 = true;
      }
      if (ctx.device_properties.supports_float64) {
        enabled_features.features.shaderFloat64 = true;
      }
      if (ctx.device_properties.supports_int8) {
        float16_int8.shaderInt8 = true;
        needs_float16_int8 = true;
      }
      if (ctx.device_properties.supports_int16) {
        enabled_features.features.shaderInt16 = true;
      }
      if (ctx.device_properties.supports_int64) {
        enabled_features.features.shaderInt64 = true;
      }
      if (ctx.device_properties.supports_8bit_buffer) {
        storage_8bit.storageBuffer8BitAccess = true;
        *pp_next = &storage_8bit;
        pp_next = &storage_8bit.pNext;
      }
      if (ctx.device_properties.supports_16bit_buffer) {
        storage_16bit.storageBuffer16BitAccess = true;
        *pp_next = &storage_16bit;
        pp_next = &storage_16bit.pNext;
      }

      if (needs_float16_int8) {
        *pp_next = &float16_int8;
        pp_next = &float16_int8.pNext;
      }

      VkDeviceCreateInfo device_create_info;
      device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      device_create_info.pNext = nullptr;
      device_create_info.flags = 0;
      device_create_info.queueCreateInfoCount = 1;
      device_create_info.pQueueCreateInfos = &queue_create_info;
      device_create_info.enabledLayerCount = 0;
      device_create_info.ppEnabledLayerNames = nullptr;
      device_create_info.enabledExtensionCount = device_extensions.size();
      device_create_info.ppEnabledExtensionNames = device_extensions.data();

      if (has_instance_extension("VK_KHR_get_physical_device_properties2")) {
        device_create_info.pEnabledFeatures = nullptr;
        device_create_info.pNext = &enabled_features;
      } else {
        device_create_info.pNext = nullptr;
        device_create_info.pEnabledFeatures = &enabled_features.features;
      }
      VULKAN_CALL(vkCreateDevice(phy_dev, &device_create_info, nullptr, &(ctx.device)));
    }

    ctx.queue_mutex.reset(new std::mutex());
    vkGetDeviceQueue(ctx.device, queue_family_index, 0, &(ctx.queue));
    ctx.queue_family_index = queue_family_index;
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
    info.pQueueFamilyIndices = &(ctx.queue_family_index);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // get staging requirement
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VULKAN_CALL(vkCreateBuffer(ctx.device, &info, nullptr, &buffer));
    vkGetBufferMemoryRequirements(ctx.device, buffer, &req_staging);
    vkDestroyBuffer(ctx.device, buffer, nullptr);
    // get compute requirement
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VULKAN_CALL(vkCreateBuffer(ctx.device, &info, nullptr, &buffer));
    vkGetBufferMemoryRequirements(ctx.device, buffer, &req_compute);
    vkDestroyBuffer(ctx.device, buffer, nullptr);

    // Query phyiscal device property
    // find a memory that is host visible, no need to be consistent
    int win_rank = -1;
    VkPhysicalDeviceMemoryProperties prop;
    vkGetPhysicalDeviceMemoryProperties(ctx.phy_device, &prop);

    for (uint32_t k = 0; k < prop.memoryTypeCount; ++k) {
      VkMemoryType ty = prop.memoryTypes[k];
      size_t heap_size = prop.memoryHeaps[ty.heapIndex].size;
      // host visible
      if (!(ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
      // match copy requirment
      if (!(req_staging.memoryTypeBits & (1 << k))) continue;
      if (heap_size < 1024) continue;
      int rank = 0;
      rank += ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      if (rank > win_rank) {
        win_rank = rank;
        ctx.staging_mtype_index = k;
        ctx.coherent_staging = ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      }
    }
    ICHECK_GE(win_rank, 0) << "Cannot find suitable staging memory on device.";

    win_rank = -1;
    for (uint32_t k = 0; k < prop.memoryTypeCount; ++k) {
      VkMemoryType ty = prop.memoryTypes[k];
      size_t heap_size = prop.memoryHeaps[ty.heapIndex].size;
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
        ctx.compute_mtype_index = k;
      }
    }
    ICHECK_GE(win_rank, 0) << "Cannot find suitable local memory on device.";

    if (ctx.device_properties.supports_push_descriptor) {
      ctx.descriptor_template_khr_functions =
          std::make_unique<VulkanDescriptorTemplateKHRFunctions>(ctx.device);
    }

    if (ctx.device_properties.supports_dedicated_allocation) {
      ctx.get_buffer_memory_requirements_2_functions =
          std::make_unique<VulkanGetBufferMemoryRequirements2Functions>(ctx.device);
    }

    context_.push_back(std::move(ctx));
  }

  LOG(INFO) << "Initialize Vulkan with " << context_.size() << " devices..";
  for (size_t i = 0; i < context_.size(); ++i) {
    LOG(INFO) << "vulkan(" << i << ")=\'" << context_[i].phy_device_prop.deviceName
              << "\' phy_dev_id=" << context_[i].phy_device
              << " use_immediate=" << context_[i].UseImmediate();
  }
}

VulkanDeviceAPI::~VulkanDeviceAPI() {
  for (auto& vctx : context_) {
    vkDestroyDevice(vctx.device, nullptr);
  }
  if (instance_) {
    vkDestroyInstance(instance_, nullptr);
  }
}

void VulkanDeviceAPI::SetDevice(Device dev) { VulkanThreadEntry::ThreadLocal()->device = dev; }

void VulkanDeviceAPI::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  size_t index = static_cast<size_t>(dev.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < context_.size());
    return;
  }

  const auto& prop = context(index).device_properties;

  switch (kind) {
    case kMaxThreadsPerBlock: {
      *rv = int64_t(prop.max_num_threads);
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      *rv = int64_t(prop.max_shared_memory_per_block);
      break;
    }
    case kWarpSize: {
      *rv = int64_t(prop.thread_warp_size);
      break;
    }
    case kComputeVersion: {
      int64_t value = prop.vulkan_api_version;
      std::ostringstream os;
      os << VK_VERSION_MAJOR(value) << "." << VK_VERSION_MINOR(value) << "."
         << VK_VERSION_PATCH(value);
      *rv = os.str();
      break;
    }
    case kDeviceName:
      *rv = prop.device_name;
      break;

    case kMaxClockRate:
      break;

    case kMultiProcessorCount:
      break;

    case kExist:
      break;

    case kMaxThreadDimensions: {
      std::stringstream ss;  // use json string to return multiple int values;
      ss << "[" << prop.max_block_size_x << ", " << prop.max_block_size_y << ", "
         << prop.max_block_size_z << "]";
      *rv = ss.str();
      break;
    }

    case kMaxRegistersPerBlock:
      break;

    case kGcnArch:
      break;

    case kApiVersion:
      *rv = VK_HEADER_VERSION;
      break;

    case kDriverVersion: {
      int64_t value = prop.driver_version;
      std::ostringstream os;
      os << VK_VERSION_MAJOR(value) << "." << VK_VERSION_MINOR(value) << "."
         << VK_VERSION_PATCH(value);
      *rv = os.str();
      break;
    }
  }
}

void VulkanDeviceAPI::GetTargetProperty(Device dev, const std::string& property, TVMRetValue* rv) {
  size_t index = static_cast<size_t>(dev.device_id);
  const auto& prop = context(index).device_properties;

  if (property == "supports_float16") {
    *rv = prop.supports_float16;
  }
  if (property == "supports_float32") {
    *rv = prop.supports_float32;
  }
  if (property == "supports_float64") {
    *rv = prop.supports_float64;
  }
  if (property == "supports_int8") {
    *rv = prop.supports_int8;
  }
  if (property == "supports_int16") {
    *rv = prop.supports_int16;
  }
  if (property == "supports_int32") {
    *rv = prop.supports_int32;
  }
  if (property == "supports_int64") {
    *rv = prop.supports_int64;
  }
  if (property == "supports_8bit_buffer") {
    *rv = prop.supports_8bit_buffer;
  }
  if (property == "supports_16bit_buffer") {
    *rv = prop.supports_16bit_buffer;
  }
  if (property == "supports_storage_buffer_storage_class") {
    *rv = prop.supports_storage_buffer_storage_class;
  }
  if (property == "supports_push_descriptor") {
    *rv = prop.supports_push_descriptor;
  }
  if (property == "supports_dedicated_allocation") {
    *rv = prop.supports_dedicated_allocation;
  }
  if (property == "supported_subgroup_operations") {
    *rv = int64_t(prop.supported_subgroup_operations);
  }
  if (property == "max_num_threads") {
    *rv = int64_t(prop.max_num_threads);
  }
  if (property == "thread_warp_size") {
    *rv = int64_t(prop.thread_warp_size);
  }
  if (property == "max_block_size_x") {
    *rv = int64_t(prop.max_block_size_x);
  }
  if (property == "max_block_size_y") {
    *rv = int64_t(prop.max_block_size_y);
  }
  if (property == "max_block_size_z") {
    *rv = int64_t(prop.max_block_size_z);
  }
  if (property == "max_push_constants_size") {
    *rv = int64_t(prop.max_push_constants_size);
  }
  if (property == "max_uniform_buffer_range") {
    *rv = int64_t(prop.max_uniform_buffer_range);
  }
  if (property == "max_storage_buffer_range") {
    *rv = int64_t(prop.max_storage_buffer_range);
  }
  if (property == "max_per_stage_descriptor_storage_buffer") {
    *rv = int64_t(prop.max_per_stage_descriptor_storage_buffer);
  }
  if (property == "max_shared_memory_per_block") {
    *rv = int64_t(prop.max_shared_memory_per_block);
  }
  if (property == ":string device_name") {
    *rv = prop.device_name;
  }
  if (property == "driver_version") {
    *rv = int64_t(prop.driver_version);
  }
  if (property == "vulkan_api_version") {
    *rv = int64_t(prop.vulkan_api_version);
  }
  if (property == "max_spirv_version") {
    *rv = int64_t(prop.max_spirv_version);
  }
}

void* VulkanDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                      DLDataType type_hint) {
  if (nbytes == 0) {
    // Vulkan seems to have issues if we return nullptr on zero size alloc
    nbytes = 1;
  }
  const auto& vctx = context(dev.device_id);
  auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  return CreateBuffer(vctx, nbytes, usage, vctx.compute_mtype_index);
}

void VulkanDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  // Before releasing the vkBuffer, call sync to
  // finish all the vulkan commands that reference the buffer.
  StreamSync(dev, nullptr);

  const auto& vctx = context(dev.device_id);
  auto* pbuf = static_cast<VulkanBuffer*>(ptr);
  vkDestroyBuffer(vctx.device, pbuf->buffer, nullptr);
  vkFreeMemory(vctx.device, pbuf->memory, nullptr);
  delete pbuf;
}

void* VulkanDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return VulkanThreadEntry::ThreadLocal()->pool->AllocWorkspace(dev, size);
}

void VulkanDeviceAPI::FreeWorkspace(Device dev, void* data) {
  VulkanThreadEntry::ThreadLocal()->pool->FreeWorkspace(dev, data);
}

TVMStreamHandle VulkanDeviceAPI::CreateStream(Device dev) { return nullptr; }

void VulkanDeviceAPI::FreeStream(Device dev, TVMStreamHandle stream) {
  ICHECK_EQ(stream, static_cast<void*>(nullptr));
}

// Syncing two streams is a nop, since there is only one stream.
void VulkanDeviceAPI::SyncStreamFromTo(Device dev, TVMStreamHandle event_src,
                                       TVMStreamHandle event_dst) {
  ICHECK_EQ(event_src, static_cast<void*>(nullptr));
  ICHECK_EQ(event_dst, static_cast<void*>(nullptr));
}

void VulkanDeviceAPI::StreamSync(Device dev, TVMStreamHandle stream) {
  ICHECK_EQ(stream, static_cast<void*>(nullptr));
  VulkanThreadEntry::ThreadLocal()->Stream(dev.device_id)->Synchronize();
}

void VulkanDeviceAPI::SetStream(Device dev, TVMStreamHandle stream) {
  ICHECK_EQ(stream, static_cast<void*>(nullptr));
}

void VulkanDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                     size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                     DLDataType type_hint, TVMStreamHandle stream) {
  ICHECK(stream == nullptr);
  Device dev = dev_from;
  if (dev_from.device_type == kDLCPU) {
    dev = dev_to;
  }

  int from_dev_type = static_cast<int>(dev_from.device_type);
  int to_dev_type = static_cast<int>(dev_to.device_type);
  if (from_dev_type == kDLVulkan && to_dev_type == kDLVulkan) {
    VulkanThreadEntry::ThreadLocal()
        ->Stream(dev_from.device_id)
        ->Launch([=](VulkanStreamState* state) {
          // 1: copy
          const auto* from_buf = static_cast<const VulkanBuffer*>(from);
          auto* to_buf = static_cast<VulkanBuffer*>(to);
          VkBufferCopy copy_info;
          copy_info.srcOffset = from_offset;
          copy_info.dstOffset = to_offset;
          copy_info.size = size;
          vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, to_buf->buffer, 1, &copy_info);
          // 2: barrier(transfer-> compute|transfer)
          ICHECK_EQ(dev_from.device_id, dev_to.device_id) << "Vulkan disallow cross device copy.";
          VkMemoryBarrier barrier_info;
          barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
          barrier_info.pNext = nullptr;
          barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier_info.dstAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
          vkCmdPipelineBarrier(
              state->cmd_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
              &barrier_info, 0, nullptr, 0, nullptr);
        });

  } else if (from_dev_type == kDLVulkan && to_dev_type == kDLCPU) {
    const auto* from_buf = static_cast<const VulkanBuffer*>(from);
    const auto& vctx = context(dev_from.device_id);
    auto* temp = VulkanThreadEntry::ThreadLocal()->StagingBuffer(dev_from.device_id, size);
    VulkanThreadEntry::ThreadLocal()
        ->Stream(dev_from.device_id)
        ->Launch([&](VulkanStreamState* state) {
          VkBufferCopy copy_info;
          copy_info.srcOffset = from_offset;
          copy_info.dstOffset = 0;
          copy_info.size = size;
          vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, temp->vk_buf->buffer, 1,
                          &copy_info);
        });
    VulkanThreadEntry::ThreadLocal()->Stream(dev_from.device_id)->Synchronize();
    if (!vctx.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = temp->vk_buf->memory;
      mrange.offset = 0;
      mrange.size = VK_WHOLE_SIZE;  // size;
      VULKAN_CALL(vkInvalidateMappedMemoryRanges(vctx.device, 1, &mrange));
    }
    memcpy(static_cast<char*>(to) + to_offset, static_cast<char*>(temp->host_addr), size);
  } else if (from_dev_type == kDLCPU && to_dev_type == kDLVulkan) {
    const auto& vctx = context(dev_to.device_id);
    const auto* to_buf = static_cast<const VulkanBuffer*>(to);
    VulkanStagingBuffer* temp =
        VulkanThreadEntry::ThreadLocal()->StagingBuffer(dev_to.device_id, size);
    memcpy(temp->host_addr, static_cast<const char*>(from) + from_offset, size);
    // host side flush if access is not coherent.
    // so writes from CPU is visible to GPU
    if (!vctx.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = temp->vk_buf->memory;
      mrange.offset = 0;
      mrange.size = VK_WHOLE_SIZE;  // size;
      VULKAN_CALL(vkFlushMappedMemoryRanges(vctx.device, 1, &mrange));
    }

    VulkanThreadEntry::ThreadLocal()
        ->Stream(dev_to.device_id)
        ->Launch([&](VulkanStreamState* state) {
          // 0: barrier(host->transfer)
          VkMemoryBarrier barrier_info;
          barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
          barrier_info.pNext = nullptr;
          barrier_info.srcAccessMask = 0;
          barrier_info.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          vkCmdPipelineBarrier(state->cmd_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier_info, 0, nullptr, 0,
                               nullptr);
          // 1: copy
          VkBufferCopy copy_info;
          copy_info.srcOffset = 0;
          copy_info.dstOffset = to_offset;
          copy_info.size = size;
          vkCmdCopyBuffer(state->cmd_buffer_, temp->vk_buf->buffer, to_buf->buffer, 1, &copy_info);
        });
    // TODO(tulloch): should we instead make the staging buffer a property of the
    // Stream? This would allow us to elide synchronizations here.
    VulkanThreadEntry::ThreadLocal()->Stream(dev_to.device_id)->Synchronize();
  } else {
    LOG(FATAL) << "Expect copy from/to Vulkan or between Vulkan"
               << ", from=" << from_dev_type << ", to=" << to_dev_type;
  }
}

std::vector<const char*> VulkanDeviceAPI::FindEnabledExtensions(
    const std::vector<VkExtensionProperties>& ext_prop,
    const std::vector<const char*>& required_extensions,
    const std::vector<const char*>& optional_extensions) {
  std::set<std::string> available_extensions;
  for (const auto& prop : ext_prop) {
    if (prop.specVersion > 0) {
      available_extensions.insert(prop.extensionName);
    }
  }

  std::vector<const char*> enabled_extensions;
  for (const auto& ext : required_extensions) {
    ICHECK(available_extensions.count(ext))
        << "Required vulkan extension \"" << ext << "\" not supported by driver";
    enabled_extensions.push_back(ext);
  }

  for (const auto& ext : optional_extensions) {
    if (available_extensions.count(ext)) {
      enabled_extensions.push_back(ext);
    }
  }

  return enabled_extensions;
}

const VulkanContext& VulkanDeviceAPI::context(size_t device_id) const {
  ICHECK_LT(device_id, context_.size()) << "Requested Vulkan device_id=" << device_id
                                        << ", but only " << context_.size() << " devices present";
  return context_[device_id];
}

std::vector<uint32_t> VulkanDeviceAPI::GetComputeQueueFamilies(VkPhysicalDevice phy_dev) {
  uint32_t queue_prop_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queue_prop_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_props(queue_prop_count);
  vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queue_prop_count, dmlc::BeginPtr(queue_props));

  std::vector<uint32_t> result;
  // Prefer compute-only queues. On certain devices supporting this (e.g. Mesa RADV), using
  // compute-only queues gives better responsiveness for other graphics workload (e.g. desktop).
  for (uint32_t i = 0; i != queue_prop_count; ++i) {
    if ((VK_QUEUE_COMPUTE_BIT & queue_props[i].queueFlags) != 0 &&
        (VK_QUEUE_GRAPHICS_BIT & queue_props[i].queueFlags) == 0) {
      result.push_back(i);
    }
  }
  // Now, push the compute queues that we skipped above into the list.
  for (uint32_t i = 0; i != queue_prop_count; ++i) {
    if ((VK_QUEUE_COMPUTE_BIT & queue_props[i].queueFlags) != 0 &&
        (VK_QUEUE_GRAPHICS_BIT & queue_props[i].queueFlags) != 0) {
      result.push_back(i);
    }
  }
  return result;
}

TVM_REGISTER_GLOBAL("device_api.vulkan").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = VulkanDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("device_api.vulkan.get_target_property")
    .set_body_typed([](Device dev, const std::string& property) {
      TVMRetValue rv;
      VulkanDeviceAPI::Global()->GetTargetProperty(dev, property, &rv);
      return rv;
    });

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
