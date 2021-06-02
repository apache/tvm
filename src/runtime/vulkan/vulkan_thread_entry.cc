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

#include "vulkan_thread_entry.h"

#include "vulkan_buffer.h"
#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanThreadEntry::~VulkanThreadEntry() {
  // Because the thread entry refers to Device API
  // The command buffer always will be destroyed before
  // the instance and device get destroyed.
  // The destruction need to be manually called
  // to ensure the destruction order.

  pool.reset();
  streams_.clear();
  for (const auto& kv : staging_buffers_) {
    DeleteHostVisibleBuffer(kv.second.get());
  }
}

VulkanThreadEntry* VulkanThreadEntry::ThreadLocal() { return VulkanThreadStore::Get(); }

void VulkanThreadEntry::AllocateUniformBuffer(int device_id, size_t size) {
  const auto& vctx = VulkanDeviceAPI::Global()->context(device_id);
  auto prop = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  auto info = MakeBufferCreateInfo(vctx, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  auto mem_type_index = FindMemoryType(vctx, info, prop);
  GetOrAllocate(device_id, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, mem_type_index,
                &uniform_buffers_, true);
}

VulkanUniformBuffer* VulkanThreadEntry::GetUniformBuffer(int device_id, size_t size) {
  auto& buf = uniform_buffers_[device_id];
  ICHECK(buf);
  ICHECK_GE(buf->size, size);
  return buf.get();
}

VulkanStagingBuffer* VulkanThreadEntry::StagingBuffer(int device_id, size_t size) {
  const auto& vctx = VulkanDeviceAPI::Global()->context(device_id);
  auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  return GetOrAllocate(device_id, size, usage, vctx.staging_mtype_index, &staging_buffers_);
}

VulkanThreadEntry::VulkanThreadEntry()
    : pool(std::make_unique<WorkspacePool>(static_cast<DLDeviceType>(kDLVulkan),
                                           VulkanDeviceAPI::Global())) {
  device.device_id = 0;
  device.device_type = static_cast<DLDeviceType>(kDLVulkan);
}

VulkanStream* VulkanThreadEntry::Stream(size_t device_id) {
  if (!streams_[device_id]) {
    streams_[device_id] = std::unique_ptr<VulkanStream>(
        new VulkanStream(&VulkanDeviceAPI::Global()->context(device_id)));
  }
  return streams_[device_id].get();
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
