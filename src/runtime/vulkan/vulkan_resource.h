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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_RESOURCE_H_
#define TVM_RUNTIME_VULKAN_VULKAN_RESOURCE_H_

#include <vulkan/vulkan_core.h>

#include <memory>
#include <optional>
#include <string>

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanDevice;

/*!
 * \brief Class representing Vulkan device memory allocations.
 *
 * This class encapsulates a Vulkan device memory allocation and its memory requirements.
 * It provides functionality to check memory compatibility with new resource requirements.
 */
class VulkanMemory {
 public:
  /*!
   * \brief Constructor to create a VulkanMemory instance.
   *
   * \param mem The Vulkan device memory handle.
   * \param mem_reqs The memory requirements associated with this allocation.
   */
  VulkanMemory(VkDeviceMemory mem, const VkMemoryRequirements& mem_reqs)
      : memory_(mem), mem_reqs_(mem_reqs) {}

  /*!
   * \brief Destructor to free the Vulkan device memory.
   */
  ~VulkanMemory() {
    if (memory_ != VK_NULL_HANDLE) {
      memory_ = VK_NULL_HANDLE;
    }
  }

  VkDeviceMemory memory_;
  VkMemoryRequirements mem_reqs_;
};

/*!
 * \brief Base class for Vulkan resources such as buffers and images.
 *
 * This class holds common properties and functionalities for Vulkan resources,
 * including device association, memory layout, and memory management.
 */
class VulkanResource {
 public:
  /*!
   * \brief Enumeration of memory layout types.
   */
  enum class MemoryLayout {
    kBuffer1D,
    kImage2DActivation,
    kImage2DWeight,
    kImage2DNHWC,
  };

  /*!
   * \brief Constructor to create a VulkanResource.
   *
   * \param device The Vulkan device associated with this resource.
   * \param mem_scope Optional memory scope string specifying the memory layout.
   * \param back_memory Optional shared pointer to existing VulkanMemory.
   */
  VulkanResource(const VulkanDevice& device, std::optional<std::string> mem_scope,
                 std::shared_ptr<VulkanMemory> back_memory = nullptr);

  /*!
   * \brief Virtual destructor.
   */
  virtual ~VulkanResource();

  // Forbid copy assignment/constructor
  VulkanResource(const VulkanResource&) = delete;
  VulkanResource& operator=(const VulkanResource&) = delete;

  // Allow move assignment/constructor
  VulkanResource(VulkanResource&& other);
  VulkanResource& operator=(VulkanResource&& other);

  /*!
   * \brief Converts a memory scope string to a MemoryLayout enumeration.
   *
   * \param mem_scope The optional memory scope string.
   * \return The corresponding MemoryLayout value.
   */
  static MemoryLayout MemoryLayoutFromScope(std::optional<std::string> mem_scope);

  /*!
   * \brief Converts a MemoryLayout enumeration to a memory scope string.
   *
   * \param layout The MemoryLayout value.
   * \return The corresponding memory scope string.
   */
  static std::string ScopeFromMemoryLayout(MemoryLayout layout);

  VkDevice device_{VK_NULL_HANDLE};
  MemoryLayout layout{MemoryLayout::kBuffer1D};
  std::shared_ptr<VulkanMemory> memory{nullptr};
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_RESOURCE_H_
