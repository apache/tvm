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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "../memory/pooled_allocator.h"
#include "vulkan_buffer.h"
#include "vulkan_common.h"
#include "vulkan_image.h"
#include "vulkan_timer.h"

namespace tvm {
namespace runtime {
namespace vulkan {

using tvm::runtime::memory::Buffer;

struct ImageInfo {
  VkOffset3D origin;
  VkExtent3D region;
  uint32_t layer_count;
};

ImageInfo GetImageInfo(const VulkanImage* image, const DLTensor* tensor) {
  ImageInfo info{};

  ICHECK(tensor->dtype.lanes == 1) << "Image dtype has lanes: " << tensor->dtype.lanes;

  info.origin = {0, 0, 0};
  info.layer_count = 0;
  size_t axis = DefaultTextureLayoutSeparator(tensor->ndim,
                                              VulkanResource::ScopeFromMemoryLayout(image->layout));
  auto texture_shape = ApplyTexture2DFlattening<int64_t>(tensor->shape, tensor->ndim, axis);
  info.region = {static_cast<uint32_t>(texture_shape.width),
                 static_cast<uint32_t>(texture_shape.height), 1};
  info.layer_count = static_cast<uint32_t>(texture_shape.depth);
  return info;
}

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
  ICHECK_EQ(dev.device_type, kDLVulkan)
      << "Active vulkan device cannot be set to non-vulkan device" << dev;

  ICHECK_LE(dev.device_id, static_cast<int>(devices_.size()))
      << "Attempted to set active vulkan device to device_id==" << dev.device_id << ", but only "
      << devices_.size() << " devices present";

  active_device_id_per_thread.GetOrMake(0) = dev.device_id;
}

int VulkanDeviceAPI::GetActiveDeviceID() { return active_device_id_per_thread.GetOrMake(0); }

VulkanDevice& VulkanDeviceAPI::GetActiveDevice() { return device(GetActiveDeviceID()); }

void VulkanDeviceAPI::GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) {
  size_t index = static_cast<size_t>(dev.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < devices_.size());
    return;
  }

  const auto& prop = device(index).device_properties;

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
      *rv = std::string(prop.device_name);
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

    case kL2CacheSizeBytes:
      break;

    case kTotalGlobalMemory: {
      *rv = device(index).compute_memory_size;
      return;
    }
    case kAvailableGlobalMemory:
      // Not currently implemented.  Will only be implementable for
      // devices that support the VK_EXT_memory_budget extension.
      break;
    case kImagePitchAlignment:
      *rv = int64_t(prop.image_row_align);
      break;
  }
}

void VulkanDeviceAPI::GetTargetProperty(Device dev, const std::string& property, ffi::Any* rv) {
  size_t index = static_cast<size_t>(dev.device_id);
  const auto& prop = device(index).device_properties;

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

  if (property == "supports_integer_dot_product") {
    *rv = prop.supports_integer_dot_product;
  }

  if (property == "supports_cooperative_matrix") {
    *rv = prop.supports_cooperative_matrix;
  }

  if (property == "device_name") {
    *rv = prop.device_name;
  }
  if (property == "device_type") {
    *rv = prop.device_type;
  }
  if (property == "driver_name") {
    *rv = prop.driver_name;
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
  if (property == "image_row_align") {
    *rv = int64_t(prop.image_row_align);
  }
}

size_t VulkanDeviceAPI::GetImageAlignment(Device dev) {
  const auto& device = this->device(dev.device_id);
  return device.device_properties.image_row_align;
}

size_t VulkanDeviceAPI::GetDataSize(const DLTensor& arr, ffi::Optional<ffi::String> mem_scope) {
  if (!mem_scope.has_value() || mem_scope.value().empty() || mem_scope.value() == "global") {
    return DeviceAPI::GetDataSize(arr);
  }

  uint32_t row_align = static_cast<uint32_t>(GetImageAlignment(arr.device));
  std::vector<int64_t> shape;
  shape.assign(arr.shape, arr.shape + arr.ndim);
  return runtime::GetTextureMemorySize<std::vector<int64_t>>(shape, arr.dtype.bits, arr.dtype.lanes,
                                                             mem_scope.value(), row_align);
}

static size_t GetMemObjectSize(Device dev, int ndim, const int64_t* shape, DLDataType dtype) {
  DLTensor temp;
  temp.data = nullptr;
  temp.device = dev;
  temp.ndim = ndim;
  temp.dtype = dtype;
  temp.shape = const_cast<int64_t*>(shape);
  temp.strides = nullptr;
  temp.byte_offset = 0;
  size_t size = DeviceAPI::Get(dev)->GetDataSize(temp);
  return size;
}

void* VulkanDeviceAPI::AllocVulkanBuffer(Device dev, size_t nbytes, DLDataType type_hint,
                                         std::shared_ptr<VulkanMemory> memory) {
  if (nbytes == 0) {
    // Vulkan seems to have issues if we return nullptr on zero size alloc
    nbytes = 1;
  }

  // For a standard buffer allocation, use the default layout (1D Buffer)
  auto mem_scope = std::optional<std::string>("global");

  const auto& device = this->device(dev.device_id);
  auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  return new VulkanBuffer(device, nbytes, usage, device.compute_mtype_index, mem_scope, memory);
}

void* VulkanDeviceAPI::AllocVulkanImage(Device dev, size_t width, size_t height, size_t layers,
                                        DLDataType type_hint, ffi::Optional<ffi::String> mem_scope,
                                        std::shared_ptr<VulkanMemory> memory) {
  const auto& device = this->device(dev.device_id);
  auto format = DTypeToVulkanFormat(type_hint);  // Use the new function to get the format
  auto usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  // image and view creation
  VulkanImage* image = new VulkanImage(device, format, width, height, layers, usage,
                                       device.compute_mtype_index, mem_scope.value(), memory);
  image->CreateImageView(format);
  return image;
}

void* VulkanDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                      DLDataType type_hint) {
  return AllocVulkanBuffer(dev, nbytes, type_hint, nullptr);
}

void* VulkanDeviceAPI::AllocDataSpace(Device dev, size_t width, size_t height, size_t depth,
                                      DLDataType type_hint, ffi::Optional<ffi::String> mem_scope) {
  if (!mem_scope.has_value()) {
    mem_scope = ffi::String("global.texture");
  }
  return AllocVulkanImage(dev, width, height, depth, type_hint, mem_scope, nullptr);
}

void* VulkanDeviceAPI::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                      ffi::Optional<ffi::String> mem_scope) {
  if (!mem_scope.has_value() || mem_scope.value().empty() || mem_scope.value() == "global") {
    size_t size = GetMemObjectSize(dev, ndim, shape, dtype);
    auto buf = MemoryManager::GetOrCreateAllocator(dev, AllocatorType::kPooled)
                   ->Alloc(dev, size, kTempAllocaAlignment, dtype);
    return buf.data;
  }

  size_t axis = DefaultTextureLayoutSeparator(ndim, mem_scope.value());
  auto texture = ApplyTexture2DFlattening<int64_t>(shape, ndim, axis);

  return AllocDataSpace(dev, texture.width, texture.height, texture.depth, dtype, mem_scope);
}

void* VulkanDeviceAPI::AllocDataSpaceView(Device dev, void* data, ffi::Shape shape,
                                          DLDataType dtype, ffi::Optional<ffi::String> mem_scope) {
  const auto* res = static_cast<const VulkanResource*>(data);

  if (!mem_scope.has_value() || mem_scope.value().empty() || mem_scope.value() == "global") {
    size_t nbytes = GetMemObjectSize(dev, shape.size(), shape.data(), dtype);
    return AllocVulkanBuffer(dev, nbytes, dtype, res->memory);
  }
  size_t axis = DefaultTextureLayoutSeparator(shape.size(), mem_scope.value());
  auto texture = ApplyTexture2DFlattening<int64_t>(shape.data(), shape.size(), axis);
  return AllocVulkanImage(dev, texture.width, texture.height, texture.depth, dtype, mem_scope,
                          res->memory);
}

void VulkanDeviceAPI::FreeDataSpaceView(Device dev, void* ptr) {
  StreamSync(dev, nullptr);
  const auto* res = static_cast<const VulkanResource*>(ptr);

  if (const auto* buf_res = dynamic_cast<const VulkanBuffer*>(res)) {
    delete buf_res;
  } else if (const auto* img_res = dynamic_cast<const VulkanImage*>(res)) {
    delete img_res;
  }
}

void VulkanDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  // Get Vulkan stream associated with the device
  VulkanStream& stream = device(dev.device_id).ThreadLocalStream();
  const auto* res = static_cast<const VulkanResource*>(ptr);

  if (const auto* buf_res = dynamic_cast<const VulkanBuffer*>(res)) {
    // Defer buffer destruction by scheduling it in VulkanStream
    stream.Launch([buf_res](VulkanStreamState* state) { delete buf_res; });
  } else if (const auto* img_res = dynamic_cast<const VulkanImage*>(res)) {
    // Defer image destruction in VulkanStream
    stream.Launch([img_res](VulkanStreamState* state) { delete img_res; });
  }
}

void* VulkanDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  // Use MemoryManager to allocate workspace memory.
  auto buffer = MemoryManager::GetOrCreateAllocator(dev, AllocatorType::kPooled)
                    ->Alloc(dev, size, kTempAllocaAlignment, type_hint);
  return buffer.data;
}

void VulkanDeviceAPI::FreeWorkspace(Device dev, void* data) {
  // Use MemoryManager to free workspace memory.
  Allocator* allocator = MemoryManager::GetAllocator(dev, AllocatorType::kPooled);
  Buffer buffer;
  buffer.data = data;
  allocator->Free(buffer);
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
  device(dev.device_id).ThreadLocalStream().Synchronize();
}

void VulkanDeviceAPI::SetStream(Device dev, TVMStreamHandle stream) {
  ICHECK_EQ(stream, static_cast<void*>(nullptr));
}

TVMStreamHandle VulkanDeviceAPI::GetCurrentStream(Device dev) { return nullptr; }

void VulkanDeviceAPI::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  ICHECK(stream == nullptr);
  ICHECK(from->device.device_type == kDLVulkan || from->device.device_type == kDLCPU);
  ICHECK(to->device.device_type == kDLVulkan || to->device.device_type == kDLCPU);

  size_t nbytes = GetDataSize(*from);
  ICHECK_EQ(nbytes, GetDataSize(*to));
  ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only supports contiguous array for now";

  Device dev_from = from->device;
  Device dev_to = to->device;
  const auto* from_res = static_cast<const VulkanResource*>(from->data);
  const auto* to_res = static_cast<const VulkanResource*>(to->data);

  int from_dev_type = static_cast<int>(dev_from.device_type);
  int to_dev_type = static_cast<int>(dev_to.device_type);

  if (from_dev_type == kDLVulkan && to_dev_type == kDLVulkan) {
    ICHECK_EQ(dev_from.device_id, dev_to.device_id)
        << "The Vulkan runtime does not support deviceA to deviceB copies. "
        << "This should be changed to a deviceA to CPU copy, followed by a CPU to deviceB copy";

    device(dev_from.device_id).ThreadLocalStream().Launch([=](VulkanStreamState* state) {
      // Buffer to Buffer Copy
      if (const auto* from_buf = dynamic_cast<const VulkanBuffer*>(from_res)) {
        if (const auto* to_buf = dynamic_cast<const VulkanBuffer*>(to_res)) {
          VkBufferCopy copy_info = {};
          copy_info.srcOffset = from->byte_offset;
          copy_info.dstOffset = to->byte_offset;
          copy_info.size = nbytes;
          vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, to_buf->buffer, 1, &copy_info);
        } else if (const auto* to_img = dynamic_cast<const VulkanImage*>(to_res)) {
          auto image_info = GetImageInfo(to_img, to);

          VkBufferImageCopy copy_info = {};
          copy_info.bufferOffset = from->byte_offset;
          copy_info.bufferRowLength = 0;
          copy_info.bufferImageHeight = 0;
          copy_info.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          copy_info.imageSubresource.mipLevel = 0;
          copy_info.imageSubresource.baseArrayLayer = 0;
          copy_info.imageSubresource.layerCount = image_info.layer_count;
          copy_info.imageOffset = {0, 0, 0};
          copy_info.imageExtent = image_info.region;
          vkCmdCopyBufferToImage(state->cmd_buffer_, from_buf->buffer, to_img->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
        }
      } else if (const auto* from_img = dynamic_cast<const VulkanImage*>(from_res)) {
        if (const auto* to_buf = dynamic_cast<const VulkanBuffer*>(to_res)) {
          auto image_info = GetImageInfo(from_img, from);

          VkBufferImageCopy copy_info = {};
          copy_info.bufferOffset = to->byte_offset;
          copy_info.bufferRowLength = 0;
          copy_info.bufferImageHeight = 0;
          copy_info.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          copy_info.imageSubresource.mipLevel = 0;
          copy_info.imageSubresource.baseArrayLayer = 0;
          copy_info.imageSubresource.layerCount = image_info.layer_count;
          copy_info.imageOffset = {0, 0, 0};
          copy_info.imageExtent = image_info.region;
          vkCmdCopyImageToBuffer(state->cmd_buffer_, from_img->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, to_buf->buffer, 1,
                                 &copy_info);
        } else if (const auto* to_img = dynamic_cast<const VulkanImage*>(to_res)) {
          auto image_info = GetImageInfo(from_img, from);

          VkImageCopy copy_info = {};
          copy_info.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          copy_info.srcSubresource.mipLevel = 0;
          copy_info.srcSubresource.baseArrayLayer = 0;
          copy_info.srcSubresource.layerCount = image_info.layer_count;
          copy_info.srcOffset = {0, 0, 0};
          copy_info.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          copy_info.dstSubresource.mipLevel = 0;
          copy_info.dstSubresource.baseArrayLayer = 0;
          copy_info.dstSubresource.layerCount = image_info.layer_count;
          copy_info.dstOffset = {0, 0, 0};
          copy_info.extent = image_info.region;
          vkCmdCopyImage(state->cmd_buffer_, from_img->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         to_img->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
        }
      }

      // Memory barrier to ensure proper synchronization
      VkMemoryBarrier barrier_info = {};
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
    auto& device = this->device(dev_from.device_id);
    auto& stream = device.ThreadLocalStream();
    auto& staging_buffer = device.ThreadLocalStagingBuffer(nbytes);

    stream.Launch([&](VulkanStreamState* state) {
      if (const auto* from_buf = dynamic_cast<const VulkanBuffer*>(from_res)) {
        VkBufferCopy copy_info = {};
        copy_info.srcOffset = from->byte_offset;
        copy_info.dstOffset = 0;
        copy_info.size = nbytes;
        vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, staging_buffer.vk_buf.buffer, 1,
                        &copy_info);
      } else if (const auto* from_img = dynamic_cast<const VulkanImage*>(from_res)) {
        auto image_info = GetImageInfo(from_img, from);

        // Ensure the image is in the correct layout for transfer
        VkImageMemoryBarrier img_barrier = {};
        img_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        img_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;  // Original layout
        img_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        img_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        img_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        img_barrier.image = from_img->image;
        img_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        img_barrier.subresourceRange.baseMipLevel = 0;
        img_barrier.subresourceRange.levelCount = 1;
        img_barrier.subresourceRange.baseArrayLayer = 0;
        img_barrier.subresourceRange.layerCount = image_info.layer_count;
        img_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        img_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(state->cmd_buffer_, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &img_barrier);
        VkBufferImageCopy copy_info = {};
        copy_info.bufferOffset = 0;
        copy_info.bufferRowLength = 0;
        copy_info.bufferImageHeight = 0;
        copy_info.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_info.imageSubresource.mipLevel = 0;
        copy_info.imageSubresource.baseArrayLayer = 0;
        copy_info.imageSubresource.layerCount = image_info.layer_count;
        copy_info.imageOffset = {0, 0, 0};
        copy_info.imageExtent = image_info.region;
        vkCmdCopyImageToBuffer(state->cmd_buffer_, from_img->image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer.vk_buf.buffer,
                               1, &copy_info);

        // Restore the image layout
        img_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        img_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        img_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        img_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(state->cmd_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &img_barrier);
      }
    });

    stream.Synchronize();
    stream.ProfilerReset();
    if (!device.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = staging_buffer.vk_buf.memory->memory_;
      mrange.offset = 0;
      mrange.size = VK_WHOLE_SIZE;
      VULKAN_CALL(vkInvalidateMappedMemoryRanges(device, 1, &mrange));
    }
    memcpy(static_cast<char*>(to->data) + to->byte_offset,
           static_cast<char*>(staging_buffer.host_addr), nbytes);

  } else if (from_dev_type == kDLCPU && to_dev_type == kDLVulkan) {
    auto& device = this->device(dev_to.device_id);
    auto& stream = device.ThreadLocalStream();
    auto& staging_buffer = device.ThreadLocalStagingBuffer(nbytes);
    memcpy(staging_buffer.host_addr, static_cast<const char*>(from->data) + from->byte_offset,
           nbytes);

    // host side flush if access is not coherent.
    // so writes from CPU is visible to GPU
    if (!device.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = staging_buffer.vk_buf.memory->memory_;
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

      if (const auto* to_buf = dynamic_cast<const VulkanBuffer*>(to_res)) {
        VkBufferCopy copy_info;
        copy_info.srcOffset = 0;
        copy_info.dstOffset = to->byte_offset;
        copy_info.size = nbytes;
        vkCmdCopyBuffer(state->cmd_buffer_, staging_buffer.vk_buf.buffer, to_buf->buffer, 1,
                        &copy_info);
      } else if (const auto* to_img = dynamic_cast<const VulkanImage*>(to_res)) {
        auto image_info = GetImageInfo(to_img, to);

        VkBufferImageCopy copy_info = {};
        copy_info.bufferOffset = 0;
        copy_info.bufferRowLength = 0;
        copy_info.bufferImageHeight = 0;
        copy_info.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_info.imageSubresource.mipLevel = 0;
        copy_info.imageSubresource.baseArrayLayer = 0;
        copy_info.imageSubresource.layerCount = image_info.layer_count;
        copy_info.imageOffset = {0, 0, 0};
        copy_info.imageExtent = image_info.region;
        vkCmdCopyBufferToImage(state->cmd_buffer_, staging_buffer.vk_buf.buffer, to_img->image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_info);
      }
    });

    stream.ProfilerReady();
    stream.Synchronize();

  } else {
    LOG(FATAL) << "Expect copy from/to Vulkan or between Vulkan"
               << ", from=" << from_dev_type << ", to=" << to_dev_type;
  }
}

const VulkanDevice& VulkanDeviceAPI::device(size_t device_id) const {
  ICHECK_LT(device_id, devices_.size()) << "Requested Vulkan device_id=" << device_id
                                        << ", but only " << devices_.size() << " devices present";
  return devices_[device_id];
}

VulkanDevice& VulkanDeviceAPI::device(size_t device_id) {
  return const_cast<VulkanDevice&>(const_cast<const VulkanDeviceAPI*>(this)->device(device_id));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.vulkan",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    DeviceAPI* ptr = VulkanDeviceAPI::Global();
                    *rv = static_cast<void*>(ptr);
                  })
      .def("device_api.vulkan.get_target_property",
           [](Device dev, const std::string& property) {
             ffi::Any rv;
             VulkanDeviceAPI::Global()->GetTargetProperty(dev, property, &rv);
             return rv;
           })
      .def_packed("device_api.vulkan.alloc_nd",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    int32_t device_type = args[0].cast<int32_t>();
                    int32_t device_id = args[1].cast<int32_t>();
                    int32_t dtype_code_hint = args[2].cast<int32_t>();
                    int32_t dtype_bits_hint = args[3].cast<int32_t>();
                    std::string scope = args[4].cast<std::string>();

                    CHECK(scope.find("texture") != std::string::npos);
                    int64_t ndim = args[5].cast<int64_t>();
                    CHECK_EQ(ndim, 2);
                    int64_t* shape = static_cast<int64_t*>(args[6].cast<void*>());
                    int64_t width = shape[0];
                    int64_t height = shape[1];
                    int64_t depth = shape[2];

                    Device dev;
                    dev.device_type = static_cast<DLDeviceType>(device_type);
                    dev.device_id = device_id;

                    DLDataType type_hint;
                    type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
                    type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
                    type_hint.lanes = 1;

                    *rv = VulkanDeviceAPI::Global()->AllocDataSpace(
                        dev, static_cast<size_t>(width), static_cast<size_t>(height),
                        static_cast<size_t>(depth), type_hint,
                        ffi::Optional<ffi::String>("global.texture"));
                  })
      .def_packed("device_api.vulkan.free_nd", [](ffi::PackedArgs args, ffi::Any* rv) {
        int32_t device_type = args[0].cast<int32_t>();
        int32_t device_id = args[1].cast<int32_t>();
        std::string scope = args[2].cast<std::string>();
        CHECK(scope.find("texture") != std::string::npos);
        void* data = args[3].cast<void*>();
        Device dev;
        dev.device_type = static_cast<DLDeviceType>(device_type);
        dev.device_id = device_id;
        VulkanDeviceAPI::Global()->FreeDataSpace(dev, data);
        *rv = static_cast<int32_t>(0);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("profiling.timer.vulkan",
                        [](Device dev) { return Timer(ffi::make_object<VulkanTimerNode>(dev)); });
}

class VulkanPooledAllocator final : public memory::PooledAllocator {
 public:
  explicit VulkanPooledAllocator() : PooledAllocator() {}

  bool AllowMemoryScope(const std::string& mem_scope) const final {
    return ((mem_scope.find("texture") != std::string::npos) || mem_scope.empty() ||
            ("global" == mem_scope));
  }

  Buffer Alloc(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) override {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    size_t size = ((nbytes + page_size_ - 1) / page_size_) * page_size_;
    auto&& it = memory_pool_.find(size);
    if (it != memory_pool_.end() && !it->second.empty()) {
      auto&& pool = it->second;
      auto ret = pool.back();
      pool.pop_back();
      return ret;
    }
    Buffer buf;
    buf.device = dev;
    buf.size = size;
    buf.alloc_type = AllocatorType::kPooled;
    try {
      buf.data = DeviceAllocDataSpace(dev, size, alignment, type_hint);
    } catch (InternalError& err) {
      LOG(WARNING) << "PooledAllocator got InternalError during allocation: " << err.message();
      LOG(WARNING) << "Trying to release all unused memory and reallocate...";
      ReleaseAll();
      buf.data = DeviceAllocDataSpace(dev, size, alignment, type_hint);
    }

    used_memory_.fetch_add(size, std::memory_order_relaxed);
    VLOG(1) << "allocate " << size << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  Buffer Alloc(Device dev, ffi::Shape shape, DLDataType type_hint,
               const std::string& mem_scope) override {
    if (AllowMemoryScope(mem_scope)) {
      size_t size = GetMemObjectSize(dev, shape.size(), shape.data(), type_hint);
      Buffer buf;
      buf.device = dev;
      buf.size = size;
      buf.alloc_type = AllocatorType::kPooled;
      buf.data = DeviceAPI::Get(dev)->AllocDataSpace(dev, shape.size(), shape.data(), type_hint,
                                                     ffi::String(mem_scope));
      if (mem_scope.find("texture") == std::string::npos) {
        // All textures are backed by buffers - don't count in total memory
        used_memory_.fetch_add(size, std::memory_order_relaxed);
      }
      DLOG(INFO) << "allocate " << size << " B, used memory " << used_memory_ << " B";
      return buf;
    }
    LOG(FATAL) << "Unsupported memory scope for this Allocator:" << mem_scope;
    return {};
  }

  void Free(const Buffer& buffer) override {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (memory_pool_.find(buffer.size) == memory_pool_.end()) {
      memory_pool_.emplace(buffer.size, std::vector<Buffer>{});
    }
    memory_pool_.at(buffer.size).push_back(buffer);
    VLOG(1) << "reclaim buffer " << buffer.size;
  }

  void* CreateView(const Buffer& buffer, ffi::Shape shape, DLDataType type_hint,
                   const std::string& mem_scope) final {
    return VulkanDeviceAPI::Global()->AllocDataSpaceView(
        buffer.device, buffer.data, shape, type_hint, ffi::Optional<ffi::String>(mem_scope));
  }

  void FreeView(Device dev, void* data) final {
    return VulkanDeviceAPI::Global()->FreeDataSpaceView(dev, data);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("DeviceAllocator.vulkan", [](ffi::PackedArgs args, ffi::Any* rv) {
    Allocator* alloc = new VulkanPooledAllocator();
    *rv = static_cast<void*>(alloc);
  });
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
