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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_INSTANCE_H_
#define TVM_RUNTIME_VULKAN_VULKAN_INSTANCE_H_

#include <vector>

#include "vulkan/vulkan_core.h"

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanInstance {
 public:
  VulkanInstance();
  ~VulkanInstance();

  // Allow move assignment/construction
  VulkanInstance(VulkanInstance&&);
  VulkanInstance& operator=(VulkanInstance&&);

  // Forbid copy assignment/construction
  VulkanInstance(const VulkanInstance&) = delete;
  VulkanInstance& operator=(const VulkanInstance&) = delete;

  /*! \brief Expose the internal VkInstance
   *
   * Allows the managed class to be passed to Vulkan APIs as if it
   * were the VkInstance handler itself.
   */
  operator VkInstance() const { return instance_; }

  /*! \brief Checks if the device has an extension enabled
   *
   * Returns true if the device was initialized with the extension
   * given.
   *
   * \param query The name of the extension to check.
   */
  bool HasExtension(const char* query) const;

  /*! \brief Return all accessible physical devices
   *
   * Wrapper around vkEnumeratePhysicalDevices.
   */
  std::vector<VkPhysicalDevice> GetPhysicalDevices() const;

 private:
  /*! \brief Helper function for move assignment/construction
   *
   * Named "do_swap" instead of "swap" because otherwise cpplint.py
   * thinks that it needs the <utility> header include.
   */
  void do_swap(VulkanInstance&& other);

  /*! \brief Extensions enabled for this instance
   *
   * Based on supported extensions queried through
   * vkEnumerateInstanceExtensionProperties, prior to creating
   * instance_.  Contains only statically allocated string literals,
   * no cleanup required.
   */
  std::vector<const char*> enabled_extensions_;

  //! \brief The Vulkan API instance handle
  VkInstance instance_{nullptr};
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_INSTANCE_H_
