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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_THREAD_ENTRY_H_
#define TVM_RUNTIME_VULKAN_VULKAN_THREAD_ENTRY_H_

#include <dmlc/thread_local.h>

#include <memory>
#include <unordered_map>

#include "../workspace_pool.h"
#include "vulkan_buffer.h"
#include "vulkan_stream.h"

namespace tvm {
namespace runtime {
namespace vulkan {

/*! \brief Contains all per-CPU-thread resources.
 */
class VulkanThreadEntry {
 public:
  VulkanThreadEntry();
  static VulkanThreadEntry* ThreadLocal();

  ~VulkanThreadEntry();

  Device device;
  std::unique_ptr<WorkspacePool> pool;
  VulkanStream* Stream(size_t device_id);
  VulkanStagingBuffer* StagingBuffer(int device_id, size_t size);
  void AllocateUniformBuffer(int device_id, size_t size);
  VulkanUniformBuffer* GetUniformBuffer(int device_id, size_t size);

 private:
  //! Map from device to the VulkanStream for it
  std::unordered_map<size_t, std::unique_ptr<VulkanStream>> streams_;
  //! Map from device to the StagingBuffer for it
  std::unordered_map<size_t, std::unique_ptr<VulkanStagingBuffer>> staging_buffers_;
  //! Map from device to the UniformBuffer associated with it
  std::unordered_map<size_t, std::unique_ptr<VulkanUniformBuffer>> uniform_buffers_;
};

typedef dmlc::ThreadLocalStore<VulkanThreadEntry> VulkanThreadStore;

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_THREAD_ENTRY_H_
