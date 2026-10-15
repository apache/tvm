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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/memory_scope.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "vulkan_common.h"
#include "vulkan_timer.h"

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
  std::vector<VkPhysicalDevice> vulkan_physical_devices = instance_.GetPhysicalDevices();
  for (VkPhysicalDevice phy_dev : vulkan_physical_devices) {
    VulkanDevice device(instance_, phy_dev);

    if (device.SupportsCompute()) {
      devices_.push_back(std::move(device));
    }
  }

  // Move discrete GPUs to the start of the list, so the default
  // device_id=0 preferentially uses a discrete GPU.
  auto preference = [](const VulkanDevice& device) {
    const std::string& type = device.device_properties.device_type;
    if (type == "discrete") {
      return 0;
    } else if (type == "integrated") {
      return 1;
    } else if (type == "virtual") {
      return 2;
    } else if (type == "cpu") {
      return 3;
    } else {
      return 4;
    }
  };

  std::stable_sort(devices_.begin(), devices_.end(),
                   [&preference](const VulkanDevice& a, const VulkanDevice& b) {
                     return preference(a) < preference(b);
                   });
}

VulkanDeviceAPI::~VulkanDeviceAPI() {}

void VulkanDeviceAPI::SetDevice(Device dev) {
  TVM_FFI_ICHECK_EQ(dev.device_type, kDLVulkan)
      << "Active vulkan device cannot be set to non-vulkan device" << dev;

  TVM_FFI_ICHECK_LE(dev.device_id, static_cast<int>(devices_.size()))
      << "Attempted to set active vulkan device to device_id==" << dev.device_id << ", but only "
      << devices_.size() << " devices present";

  active_device_id_per_thread.GetOrMake(0) = dev.device_id;
}

int VulkanDeviceAPI::GetActiveDeviceID() { return active_device_id_per_thread.GetOrMake(0); }

VulkanDevice& VulkanDeviceAPI::GetActiveDevice() { return device(GetActiveDeviceID()); }

ffi::Any VulkanDeviceAPI::GetAttr(Device dev, const ffi::String& name) {
  ffi::Any rv;
  size_t index = static_cast<size_t>(dev.device_id);
  if (name == "exist") {
    rv = index < devices_.size();
    return rv;
  }

  const auto& prop = device(index).device_properties;

  if (name == "max_threads_per_block") {
    rv = int64_t(prop.max_num_threads);
    return rv;
  }
  if (name == "max_shared_memory_per_block") {
    rv = int64_t(prop.max_shared_memory_per_block);
    return rv;
  }
  if (name == "warp_size") {
    rv = int64_t(prop.thread_warp_size);
    return rv;
  }
  if (name == "compute_version") {
    int64_t value = prop.vulkan_api_version;
    std::ostringstream os;
    os << VK_VERSION_MAJOR(value) << "." << VK_VERSION_MINOR(value) << "."
       << VK_VERSION_PATCH(value);
    rv = os.str();
    return rv;
  }
  if (name == "device_name") {
    rv = std::string(prop.device_name);
    return rv;
  }
  if (name == "max_thread_dimensions") {
    std::stringstream ss;  // use json string to return multiple int values;
    ss << "[" << prop.max_block_size_x << ", " << prop.max_block_size_y << ", "
       << prop.max_block_size_z << "]";
    rv = ss.str();
    return rv;
  }
  if (name == "api_version") {
    rv = VK_HEADER_VERSION;
    return rv;
  }
  if (name == "driver_version") {
    int64_t value = prop.driver_version;
    std::ostringstream os;
    os << VK_VERSION_MAJOR(value) << "." << VK_VERSION_MINOR(value) << "."
       << VK_VERSION_PATCH(value);
    rv = os.str();
    return rv;
  }
  if (name == "total_global_memory") {
    rv = device(index).compute_memory_size;
    return rv;
  }
  if (name == "available_global_memory") {
    // Not currently implemented.  Will only be implementable for
    // devices that support the VK_EXT_memory_budget extension.
    return rv;
  }

  if (name == "supports_float16") {
    return prop.supports_float16;
  }
  if (name == "supports_float32") {
    return prop.supports_float32;
  }
  if (name == "supports_float64") {
    return prop.supports_float64;
  }
  if (name == "supports_int8") {
    return prop.supports_int8;
  }
  if (name == "supports_int16") {
    return prop.supports_int16;
  }
  if (name == "supports_int32") {
    return prop.supports_int32;
  }
  if (name == "supports_int64") {
    return prop.supports_int64;
  }
  if (name == "supports_8bit_buffer") {
    return prop.supports_8bit_buffer;
  }
  if (name == "supports_16bit_buffer") {
    return prop.supports_16bit_buffer;
  }
  if (name == "supports_storage_buffer_storage_class") {
    return prop.supports_storage_buffer_storage_class;
  }
  if (name == "supports_push_descriptor") {
    return prop.supports_push_descriptor;
  }
  if (name == "supports_dedicated_allocation") {
    return prop.supports_dedicated_allocation;
  }
  if (name == "supported_subgroup_operations") {
    return int64_t(prop.supported_subgroup_operations);
  }
  if (name == "max_block_size_x") {
    return int64_t(prop.max_block_size_x);
  }
  if (name == "max_block_size_y") {
    return int64_t(prop.max_block_size_y);
  }
  if (name == "max_block_size_z") {
    return int64_t(prop.max_block_size_z);
  }
  if (name == "max_push_constants_size") {
    return int64_t(prop.max_push_constants_size);
  }
  if (name == "max_uniform_buffer_range") {
    return int64_t(prop.max_uniform_buffer_range);
  }
  if (name == "max_storage_buffer_range") {
    return int64_t(prop.max_storage_buffer_range);
  }
  if (name == "max_per_stage_descriptor_storage_buffer") {
    return int64_t(prop.max_per_stage_descriptor_storage_buffer);
  }

  if (name == "supports_integer_dot_product") {
    return prop.supports_integer_dot_product;
  }

  if (name == "supports_cooperative_matrix") {
    return prop.supports_cooperative_matrix;
  }

  if (name == "device_type") {
    return prop.device_type;
  }
  if (name == "driver_name") {
    return prop.driver_name;
  }
  if (name == "vulkan_driver_version") {
    return int64_t(prop.driver_version);
  }
  if (name == "vulkan_api_version") {
    return int64_t(prop.vulkan_api_version);
  }
  if (name == "max_spirv_version") {
    return int64_t(prop.max_spirv_version);
  }

  return rv;
}

ffi::Tensor VulkanDeviceAPI::Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                                   ffi::Optional<runtime::MemoryScope> mem_scope) {
  TVM_FFI_CHECK(!mem_scope.has_value() || mem_scope.value() == runtime::MemoryScope::Global(),
                ValueError)
      << "Unsupported memory scope: " << mem_scope.value_or(runtime::MemoryScope::Global()).name();
  struct Allocator {
    VulkanDeviceAPI* api;
    void AllocData(DLTensor* tensor) {
      size_t alignment =
          std::max(static_cast<size_t>(kAllocAlignment),
                   static_cast<size_t>((tensor->dtype.bits / 8) * tensor->dtype.lanes));
      tensor->data =
          api->AllocDataSpace(tensor->device, ffi::GetDataSize(*tensor), alignment, tensor->dtype);
    }
    void FreeData(DLTensor* tensor) { api->FreeDataSpace(tensor->device, tensor->data); }
  };
  return ffi::Tensor::FromNDAlloc(Allocator{this}, shape, dtype, device);
}

void* VulkanDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                      DLDataType type_hint) {
  if (nbytes == 0) {
    // Vulkan seems to have issues if we return nullptr on zero size alloc
    nbytes = 1;
  }
  const auto& device = this->device(dev.device_id);
  auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  return new VulkanBuffer(device, nbytes, usage, device.compute_mtype_index);
}

void VulkanDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  // Before releasing the vkBuffer, call sync to
  // finish all the vulkan commands that reference the buffer.
  StreamSync(dev, nullptr);

  auto* pbuf = static_cast<VulkanBuffer*>(ptr);
  delete pbuf;
}

void* VulkanDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  auto& pool = pool_per_thread.GetOrMake(kDLVulkan, this);
  return pool.AllocWorkspace(dev, size);
}

void VulkanDeviceAPI::FreeWorkspace(Device dev, void* data) {
  auto* pool = pool_per_thread.Get();
  TVM_FFI_ICHECK(pool) << "Attempted to free a vulkan workspace on a CPU-thread "
                       << "that has never allocated a workspace";
  pool->FreeWorkspace(dev, data);
}

TVMStreamHandle VulkanDeviceAPI::CreateStream(Device dev) { return nullptr; }

void VulkanDeviceAPI::FreeStream(Device dev, TVMStreamHandle stream) {
  TVM_FFI_ICHECK_EQ(stream, static_cast<void*>(nullptr));
}

// Syncing two streams is a nop, since there is only one stream.
void VulkanDeviceAPI::SyncStreamFromTo(Device dev, TVMStreamHandle event_src,
                                       TVMStreamHandle event_dst) {
  TVM_FFI_ICHECK_EQ(event_src, static_cast<void*>(nullptr));
  TVM_FFI_ICHECK_EQ(event_dst, static_cast<void*>(nullptr));
}

void VulkanDeviceAPI::StreamSync(Device dev, TVMStreamHandle stream) {
  TVM_FFI_ICHECK_EQ(stream, static_cast<void*>(nullptr));
  device(dev.device_id).ThreadLocalStream().Synchronize();
}

void VulkanDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                     size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                     DLDataType type_hint, TVMStreamHandle stream) {
  TVM_FFI_ICHECK(stream == nullptr);
  Device dev = dev_from;
  if (dev_from.device_type == kDLCPU) {
    dev = dev_to;
  }

  int from_dev_type = static_cast<int>(dev_from.device_type);
  int to_dev_type = static_cast<int>(dev_to.device_type);
  if (from_dev_type == kDLVulkan && to_dev_type == kDLVulkan) {
    TVM_FFI_ICHECK_EQ(dev_from.device_id, dev_to.device_id)
        << "The Vulkan runtime does not support deviceA to deviceB copies. "
        << "This should be changed to a deviceA to CPU copy, followed by a CPU to deviceB copy";

    device(dev_from.device_id).ThreadLocalStream().Launch([=](VulkanStreamState* state) {
      // 1: copy
      const auto* from_buf = static_cast<const VulkanBuffer*>(from);
      auto* to_buf = static_cast<VulkanBuffer*>(to);
      VkBufferCopy copy_info;
      copy_info.srcOffset = from_offset;
      copy_info.dstOffset = to_offset;
      copy_info.size = size;
      vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, to_buf->buffer, 1, &copy_info);
      // 2: barrier(transfer-> compute|transfer)
      VkMemoryBarrier barrier_info;
      barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier_info.pNext = nullptr;
      barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier_info.dstAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      vkCmdPipelineBarrier(state->cmd_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           1, &barrier_info, 0, nullptr, 0, nullptr);
    });

  } else if (from_dev_type == kDLVulkan && to_dev_type == kDLCPU) {
    const auto* from_buf = static_cast<const VulkanBuffer*>(from);
    auto& device = this->device(dev_from.device_id);
    auto& stream = device.ThreadLocalStream();
    auto& staging_buffer = device.ThreadLocalStagingBuffer(size);
    stream.Launch([&](VulkanStreamState* state) {
      VkBufferCopy copy_info;
      copy_info.srcOffset = from_offset;
      copy_info.dstOffset = 0;
      copy_info.size = size;
      vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, staging_buffer.vk_buf.buffer, 1,
                      &copy_info);
    });
    stream.Synchronize();
    stream.ProfilerReset();
    if (!device.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = staging_buffer.vk_buf.memory;
      mrange.offset = 0;
      mrange.size = VK_WHOLE_SIZE;  // size;
      VULKAN_CALL(vkInvalidateMappedMemoryRanges(device, 1, &mrange));
    }
    memcpy(static_cast<char*>(to) + to_offset, static_cast<char*>(staging_buffer.host_addr), size);
  } else if (from_dev_type == kDLCPU && to_dev_type == kDLVulkan) {
    auto& device = this->device(dev_to.device_id);
    auto& stream = device.ThreadLocalStream();
    const auto* to_buf = static_cast<const VulkanBuffer*>(to);
    auto& staging_buffer = device.ThreadLocalStagingBuffer(size);
    memcpy(staging_buffer.host_addr, static_cast<const char*>(from) + from_offset, size);
    // host side flush if access is not coherent.
    // so writes from CPU is visible to GPU
    if (!device.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = staging_buffer.vk_buf.memory;
      mrange.offset = 0;
      mrange.size = VK_WHOLE_SIZE;  // size;
      VULKAN_CALL(vkFlushMappedMemoryRanges(device, 1, &mrange));
    }

    stream.Launch([&](VulkanStreamState* state) {
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
      vkCmdCopyBuffer(state->cmd_buffer_, staging_buffer.vk_buf.buffer, to_buf->buffer, 1,
                      &copy_info);
    });

    stream.ProfilerReady();
    // TODO(tulloch): should we instead make the staging buffer a property of the
    // Stream? This would allow us to elide synchronizations here.
    stream.Synchronize();
  } else {
    TVM_FFI_THROW(InternalError) << "Expect copy from/to Vulkan or between Vulkan"
                                 << ", from=" << from_dev_type << ", to=" << to_dev_type;
  }
}

const VulkanDevice& VulkanDeviceAPI::device(size_t device_id) const {
  TVM_FFI_ICHECK_LT(device_id, devices_.size())
      << "Requested Vulkan device_id=" << device_id << ", but only " << devices_.size()
      << " devices present";
  return devices_[device_id];
}

VulkanDevice& VulkanDeviceAPI::device(size_t device_id) {
  return const_cast<VulkanDevice&>(const_cast<const VulkanDeviceAPI*>(this)->device(device_id));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("profiling.timer.vulkan", [](Device dev) {
    return Timer(ffi::UnsafeInit{}, ffi::make_object<VulkanTimerObj>(dev));
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.vulkan", [](Device dev) {
    return Timer(ffi::UnsafeInit{}, ffi::make_object<VulkanTimerObj>(dev));
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.vulkan", [](ffi::PackedArgs args, ffi::Any* rv) {
    DeviceAPI* ptr = VulkanDeviceAPI::Global();
    *rv = static_cast<void*>(ptr);
  });
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
