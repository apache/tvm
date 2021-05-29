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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_DEVICE_API_H_
#define TVM_RUNTIME_VULKAN_VULKAN_DEVICE_API_H_

#include <tvm/runtime/device_api.h>

#include <vector>

#include "vulkan/vulkan_core.h"
#include "vulkan_context.h"
#include "vulkan_thread_entry.h"

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanDeviceAPI final : public DeviceAPI {
 public:
  static VulkanDeviceAPI* Global();
  VulkanDeviceAPI();
  ~VulkanDeviceAPI();

  // Implement active device
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;

  // Implement memory management required by DeviceAPI
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;

  // Current vulkan implementation has one "stream" per CPU thread,
  // with all commands writing into a single command buffer that is
  // submitted on a call to StreamSync.  Therefore, for now, these are
  // mostly no-ops.  If needed in the future, could have multiple
  // command buffers to act as multiple streams.
  TVMStreamHandle CreateStream(Device dev) final;
  void FreeStream(Device dev, TVMStreamHandle stream) final;
  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
  void SetStream(Device dev, TVMStreamHandle stream) final;

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;

  // End of required methods for the DeviceAPI interface

 public:
  /*! \brief Return the context associated with a specific device.
   *
   * These are constructed during VulkanDeviceAPI initialization, so
   * this function returns immediately.
   */
  const VulkanContext& context(size_t device_id) const;

  /*! \brief Get a Target that best describes a particular device.
   *
   * Returns the results of feature/property queries done during the
   * device initialization.
   */
  Target GenerateTarget(size_t device_id) const;

 private:
  std::vector<uint32_t> GetComputeQueueFamilies(VkPhysicalDevice phy_dev);

  Target GetDeviceDescription(VkInstance instance, VkPhysicalDevice dev,
                              const std::vector<const char*>& instance_extensions,
                              const std::vector<const char*>& device_extensions);

  std::vector<const char*> FindEnabledExtensions(
      const std::vector<VkExtensionProperties>& ext_prop,
      const std::vector<const char*>& required_extensions,
      const std::vector<const char*>& optional_extensions);

  VkInstance instance_{nullptr};
  // The physical devices, have 1 to 1 mapping to devices
  std::vector<VulkanContext> context_;
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_DEVICE_API_H_
