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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_IMAGE_H_
#define TVM_RUNTIME_VULKAN_VULKAN_IMAGE_H_

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "vulkan_resource.h"

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanImage : public VulkanResource {
 public:
  /* \brief Allocate and create an image on the device
   *
   * \param device Which device should have the image allocation.
   * The VulkanDevice given should outlive the VulkanImage.
   *
   * \param format The format of the image (e.g., VK_FORMAT_R32_SFLOAT)
   *
   * \param width The width of the image
   *
   * \param height The height of the image
   *
   * \param layers The array layers of the image
   *
   * \param usage The usage flags for the image (e.g. sampled, transfer destination, etc.)
   *
   * \param mem_type_index The memory type to index. This should be
   * an index to a compatible memory located in
   * VkPhysicalDeviceMemoryProperties.
   */
  VulkanImage(const VulkanDevice& device, VkFormat format, uint32_t width, uint32_t height,
              uint32_t depth, VkImageUsageFlags usage, uint32_t mem_type_index,
              std::optional<std::string> mem_scope = std::nullopt,
              std::shared_ptr<VulkanMemory> back_memory = nullptr);

  ~VulkanImage();

  // Forbid copy assignment/constructor
  VulkanImage(const VulkanImage&) = delete;
  VulkanImage& operator=(const VulkanImage&) = delete;

  // Allow move assignment/constructor
  VulkanImage(VulkanImage&&);
  VulkanImage& operator=(VulkanImage&&);

  void AllocateMemory(const VkMemoryRequirements& mem_reqs, uint32_t mem_type_index);

  void CreateImageView(VkFormat format);

 private:
  /*!
   * \brief Whether this image should be allocated using dedicated allocation
   *
   * In typical usage, there will be one VkDeviceMemory that has a
   * large number of VkImages pointing to it. Currently, the TVM
   * Vulkan runtime has a single VkImage for each VkDeviceMemory. In
   * this case, there can be performance benefits by explicitly
   * marking this as a dedicated allocation. The function returns
   * true if the device supports the dedicated allocation extension,
   * and the image either requires or has better performance with a
   * dedicated allocation.
   *
   * \param[out] nbytes If using dedicated allocation, the number of
   * bytes required for the allocation. If not using dedicated
   * allocation, this value is unchanged.
   *
   * \returns Whether the allocation should use the dedicated
   * allocation extension.
   */
  static bool UseDedicatedAllocation(const VulkanDevice& device, VkImage image,
                                     VkDeviceSize* nbytes);

 public:
  /*! \brief Pointer to the device that owns this image.
   *
   * Assumes that the VulkanImage will be destructed before the
   * VulkanDevice, and this will never be a dangling reference.
   * Stores a VkDevice and not a VulkanDevice, because the
   * VulkanDevice may be moved to a different location while the
   * VulkanImage is alive.
   */

  //! \brief Handle to the logical image on the device
  VkImage image{VK_NULL_HANDLE};

  //! \brief Handle to the image view
  VkImageView imageView{VK_NULL_HANDLE};

  // capture the memory requirements.
  // VkMemoryRequirements mem_reqs;

  // Add width and height members
  uint32_t width{0};   // Width of the image
  uint32_t height{0};  // Height of the image
  uint32_t layers{0};  // Depth of the image
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_IMAGE_H_
