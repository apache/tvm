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
#include <vulkan/vulkan_core.h>

#include <string>
#include <vector>

#include "../thread_map.h"
#include "../workspace_pool.h"
#include "vulkan/vulkan_core.h"
#include "vulkan_device.h"
#include "vulkan_instance.h"

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
  /*! \brief Return the currently active VulkanDevice
   *
   * The active device can be set using VulkanDeviceAPI::SetDevice.
   * Each CPU thread has its own active device, mimicking the
   * semantics of cudaSetDevice.
   */
  VulkanDevice& GetActiveDevice();

  /*! \brief Return the currently active VulkanDevice
   *
   * The active device can be set using VulkanDeviceAPI::SetDevice.
   * Each CPU thread has its own active device, mimicking the
   * semantics of cudaSetDevice.
   */
  int GetActiveDeviceID();

  /*! \brief Return the VulkanDevice associated with a specific device_id
   *
   * These are constructed during VulkanDeviceAPI initialization, so
   * this function returns immediately.
   */
  const VulkanDevice& device(size_t device_id) const;

  /*! \brief Return the VulkanDevice associated with a specific device_id
   *
   * These are constructed during VulkanDeviceAPI initialization, so
   * this function returns immediately.
   */
  VulkanDevice& device(size_t device_id);

  /*! \brief Returns a property to be stored in a target.
   *
   * Returns the results of feature/property queries done during the
   * device initialization.
   */
  void GetTargetProperty(Device dev, const std::string& property, TVMRetValue* rv) final;

 private:
  std::vector<uint32_t> GetComputeQueueFamilies(VkPhysicalDevice phy_dev);

  /*! \brief The Vulkan API instance owned by the VulkanDeviceAPI
   *
   * Holds and manages VkInstance.
   */
  VulkanInstance instance_;

  /*! \brief Handles to the Vulkan devices
   *
   * The physical devices.  These are constructed after the instance_,
   * and must be destructed before the instance_.
   */
  std::vector<VulkanDevice> devices_;

  /*! \brief One pool of device memory for each CPU thread.
   *
   * These allocate memory based on the devices stored in devices_.
   * The memory pools must be destructed before devices_.
   */
  ThreadMap<WorkspacePool> pool_per_thread;

  /*! \brief The index of the active device for each CPU thread.
   *
   * To mimic the semantics of cudaSetDevice, each CPU thread can set
   * the device on which functions should run.  If unset, the active
   * device defaults to device_id == 0.
   */
  ThreadMap<int> active_device_id_per_thread;
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_DEVICE_API_H_
