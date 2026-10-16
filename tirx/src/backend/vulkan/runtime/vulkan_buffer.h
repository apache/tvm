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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_BUFFER_H_
#define TVM_RUNTIME_VULKAN_VULKAN_BUFFER_H_

#include <vulkan/vulkan_core.h>

#include <memory>
#include <unordered_map>

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanDevice;

class VulkanBuffer {
 public:
  /* \brief Allocate memory on the device
   *
   * \param device Which device should have the memory allocation.
   * The VulkanDevice given should outlive the VulkanBuffer.
   *
   * \param nbytes Size of the buffer in bytes
   *
   * \param usage The usage flags for the buffer (e.g. transfer
   * source, transfer dest, storage buffer, etc.)
   *
   * \param mem_type_index The memory type to index.  This should be
   * an index to a compatible memory located in
   * VkPhysicalDeviceMemoryProperties.
   */
  VulkanBuffer(const VulkanDevice& device, size_t nbytes, VkBufferUsageFlags usage,
               uint32_t mem_type_index);

  //! \brief Destructor, deallocates the memory and buffer.
  ~VulkanBuffer();

  // Forbid copy assignment/constructor
  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;

  // Allow move assignment/constructor
  VulkanBuffer(VulkanBuffer&&);
  VulkanBuffer& operator=(VulkanBuffer&&);

 private:
  /*! \brief Whether this buffer should be allocated using dedicated
   * allocation
   *
   * In typical usage, there will be one VkDeviceMemory that has a
   * large number of VkBuffers pointing to it.  Currently, the TVM
   * Vulkan runtime has a single VkBuffer for each VkDeviceMemory.  In
   * this case, there can be performance benefits by explicitly
   * marking this as a dedicated allocation.  The function returns
   * true if the device supports the dedicated allocation extension,
   * and the buffer either requires or has better performance with a
   * dedicated allocation.
   *
   * \param[out] nbytes If using dedicated allocation, the number of
   * bytes required for the allocation.  If not using dedicated
   * allocation, this value is unchanged.
   *
   * \returns Whether the allocation should use the dedicated
   * allocation extension.
   */
  static bool UseDedicatedAllocation(const VulkanDevice& device, VkBuffer buffer,
                                     VkDeviceSize* nbytes);

  // TODO(elunderberg): Move copy functionality into the buffer class
  // so these don't need to be public.
 public:
  /*! \brief Pointer to the device that owns this buffer.
   *
   * Assumes that the VulkanBuffer will be destructed before the
   * VulkanDevice, and this will never be a dangling reference.
   * Stores a VkDevice and not a VulkanDevice, because the
   * VulkanDevice may be moved to a different location while the
   * VulkanBuffer is alive.
   */
  VkDevice device_{VK_NULL_HANDLE};

  //! \brief Handle to the logical buffer on the device
  VkBuffer buffer{VK_NULL_HANDLE};

  //! \brief Handle to the physical device memory
  VkDeviceMemory memory{VK_NULL_HANDLE};

  friend class VulkanHostVisibleBuffer;
};

/*! \brief A struct to represent Vulkan buffers backed by host visible memory */
class VulkanHostVisibleBuffer {
 public:
  /* \brief Allocate memory on the device, visible to the host
   *
   * \param device Which GPU device should have the memory allocation.
   * The VulkanDevice specified should outlive the VulkanBuffer.
   *
   * \param nbytes Size of the buffer in bytes
   *
   * \param usage The usage flags for the buffer (e.g. transfer
   * source, transfer dest, storage buffer, etc.)
   *
   * \param mem_type_index The memory type to index.  This should be
   * an index to a compatible memory located in
   * VkPhysicalDeviceMemoryProperties.
   */
  VulkanHostVisibleBuffer(const VulkanDevice& device, size_t nbytes, VkBufferUsageFlags usage,
                          uint32_t mem_type_index);

  //! \brief Unmap memory and deallocate.
  ~VulkanHostVisibleBuffer();

  // Forbid copy assignment/constructor
  VulkanHostVisibleBuffer(const VulkanHostVisibleBuffer&) = delete;
  VulkanHostVisibleBuffer& operator=(const VulkanHostVisibleBuffer&) = delete;

  // Allow move assignment/constructor
  VulkanHostVisibleBuffer(VulkanHostVisibleBuffer&&);
  VulkanHostVisibleBuffer& operator=(VulkanHostVisibleBuffer&&);

 private:
  // TODO(elunderberg): Move copy functionality into the buffer class
  // so these don't need to be public.
 public:
  VulkanBuffer vk_buf;
  void* host_addr{nullptr};
  size_t size{0};
};

using VulkanStagingBuffer = VulkanHostVisibleBuffer;
using VulkanUniformBuffer = VulkanHostVisibleBuffer;

VulkanHostVisibleBuffer* GetOrAllocate(
    int device_id, size_t size, VkBufferUsageFlags usage, uint32_t mem_type_index,
    std::unordered_map<size_t, std::unique_ptr<VulkanHostVisibleBuffer>>* buffers_ptr,
    bool sync_before_realloc = false);

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_BUFFER_H_
