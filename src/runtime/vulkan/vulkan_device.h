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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_DEVICE_H_
#define TVM_RUNTIME_VULKAN_VULKAN_DEVICE_H_

#include <tvm/runtime/logging.h>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../thread_map.h"
#include "vulkan/vulkan_core.h"
#include "vulkan_buffer.h"
#include "vulkan_stream.h"

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanInstance;
class VulkanDevice;

struct VulkanDescriptorTemplateKHRFunctions {
  explicit VulkanDescriptorTemplateKHRFunctions(VkDevice device);

  PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR{nullptr};
  PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR{nullptr};
  PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR{nullptr};
  PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR{nullptr};
};

struct VulkanGetBufferMemoryRequirements2Functions {
  explicit VulkanGetBufferMemoryRequirements2Functions(VkDevice device);

  PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR{nullptr};
};

struct VulkanQueueInsertDebugUtilsLabelFunctions {
  explicit VulkanQueueInsertDebugUtilsLabelFunctions(VkInstance instance);

  PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT{nullptr};
};

/*!
 * \brief Stores the capabilities/limits queried from the physical device.
 *
 * The member variables here have a 1-1 mapping to Target parameters,
 * if target->GetTargetDeviceType()==kDLVulkan.  A separate struct is used
 * to maintain the boundary between the Vulkan runtime in
 * libtvm_runtime.so, and the Target object in libtvm.so.
 */
struct VulkanDeviceProperties {
  VulkanDeviceProperties() {}
  VulkanDeviceProperties(const VulkanInstance& instance, const VulkanDevice& device);

  bool supports_float16{false};
  bool supports_float32{true};
  bool supports_float64{false};
  bool supports_int8{false};
  bool supports_int16{false};
  bool supports_int32{true};
  bool supports_int64{false};
  bool supports_8bit_buffer{false};
  bool supports_16bit_buffer{false};
  bool supports_storage_buffer_storage_class{false};
  bool supports_push_descriptor{false};
  bool supports_dedicated_allocation{false};
  bool supports_integer_dot_product{false};
  bool supports_cooperative_matrix{false};
  uint32_t supported_subgroup_operations{0};
  uint32_t max_num_threads{1};
  uint32_t thread_warp_size{1};
  uint32_t max_block_size_x{1};
  uint32_t max_block_size_y{1};
  uint32_t max_block_size_z{1};
  uint32_t max_push_constants_size{128};
  uint32_t max_uniform_buffer_range{16384};
  uint32_t max_storage_buffer_range{1 << 27};
  uint32_t max_per_stage_descriptor_storage_buffer{4};
  uint32_t max_shared_memory_per_block{16384};
  std::string device_type{"unknown_device_type"};
  std::string device_name{"unknown_device_name"};
  std::string driver_name{"unknown_driver_name"};
  uint32_t driver_version{0};
  uint32_t vulkan_api_version{VK_API_VERSION_1_0};
  uint32_t max_spirv_version{0x10000};
};

/*! \brief Handle to the Vulkan API's VkDevice
 *
 * Handles all setup and teardown of the class.  The owner of the
 * VulkanDevice object is responsible for ensuring that it remains
 * alive as long as any object that accesses that device is used.
 */
class VulkanDevice {
 public:
  VulkanDevice(const VulkanInstance& instance, VkPhysicalDevice phy_dev);
  ~VulkanDevice();

  // Allow move constructor/assignment
  VulkanDevice(VulkanDevice&&);
  VulkanDevice& operator=(VulkanDevice&&);

  // Disable copy constructor/assignment
  VulkanDevice(const VulkanDevice&) = delete;
  VulkanDevice& operator=(const VulkanDevice&) = delete;

  /*! \brief Expose the internal VkDevice
   *
   * Allows the managed class to be passed to Vulkan APIs as if it
   * were the VkDevice handler itself.
   */
  operator VkDevice() const { return device_; }

  /*! \brief Expose the internal VkPhysicalDevice
   *
   * Allows the managed class to be passed to Vulkan APIs as if it
   * were the VkPhysicalDevice handler itself.
   */
  operator VkPhysicalDevice() const { return physical_device_; }

  /*! \brief Returns whether this device supports Vulkan compute operations.
   *
   * If the device does not support Vulkan compute operations, it
   * should not be used any further.
   */
  bool SupportsCompute() const;

  /*! \brief Calls vkQueueSubmit to run work on the GPU
   *
   * Currently only supports submitting a single VkSubmitInfo at a
   * time.  Handles mutexing internally, safe to call from multiple
   * CPU threads.
   *
   * \param submit_info The job submission information to be passed to
   * vkQueueSubmit.
   *
   * \param fence Optional fence to be passed to vkQueueSubmit,
   * signals once the command buffers submitted have completed.
   */
  void QueueSubmit(VkSubmitInfo submit_info, VkFence fence) const;

  /*! \brief Checks if the device has an extension enabled
   *
   * Returns true if the device was initialized with the extension
   * given.
   *
   * \param query The name of the extension to check.
   */
  bool HasExtension(const char* query) const;

  //! \brief Return the VulkanStream for the current CPU thread
  VulkanStream& ThreadLocalStream();

  //! \brief Return the VulkanStream for the current CPU thread
  const VulkanStream& ThreadLocalStream() const;

  /*! \brief Return the staging buffer for the current CPU thread
   *
   * This function may re-allocate the staging buffer depending on the
   * size of the previously allocated buffer.
   *
   * \param min_size The size in bytes of the staging buffer to be
   * returned.  The buffer may be larger than requested, depending on
   * previous use.
   */
  VulkanStagingBuffer& ThreadLocalStagingBuffer(size_t min_size);

  /*! \brief Allocate the uniform buffer for the current CPU thread
   *
   * \param min_size The minimum size in bytes of the uniformn buffer
   * to be allocated.  If a larger uniform buffer has already been
   * allocated, no allocation is performed.
   */
  void AllocateThreadLocalUniformBuffer(size_t min_size);

  /*! \brief Return the uniform buffer for the current CPU thread
   *
   * Assumes that AllocateThreadLocalUniformBuffer has previously been
   * called, with a min_size greater than or equal to the min_size of
   * the current call.  If this is not the case, will throw an
   * exception.
   *
   * \param min_size The minimum size in bytes of the uniform buffer to be
   * returned.
   */
  VulkanUniformBuffer& ThreadLocalUniformBuffer(size_t min_size);

  // Cached device properties, queried through Vulkan API.
  VulkanDeviceProperties device_properties{};

  // Memory type index for staging.
  uint32_t staging_mtype_index{0};
  // whether staging is coherent
  bool coherent_staging{false};

  std::unique_ptr<VulkanDescriptorTemplateKHRFunctions> descriptor_template_khr_functions{nullptr};
  std::unique_ptr<VulkanGetBufferMemoryRequirements2Functions>
      get_buffer_memory_requirements_2_functions{nullptr};
  std::unique_ptr<VulkanQueueInsertDebugUtilsLabelFunctions>
      queue_insert_debug_utils_label_functions{nullptr};
  // Memory type index for compute
  uint32_t compute_mtype_index{0};
  // maximum memory size for compute
  int64_t compute_memory_size{0};

  // queue family_index;
  uint32_t queue_family_index{uint32_t(-1)};

  bool UseImmediate() const { return descriptor_template_khr_functions != nullptr; }

  bool UseDebugUtilsLabel() const { return queue_insert_debug_utils_label_functions != nullptr; }

  VkQueue Queue() const { return queue; }

 private:
  /*! \brief Helper function for move assignment/construction
   *
   * Named "do_swap" instead of "swap" because otherwise cpplint.py
   * thinks that it needs the <utility> header include.
   */
  void do_swap(VulkanDevice&& other);

  /*! \brief Returns a queue family capable of running Vulkan compute
   * operations
   */
  uint32_t SelectComputeQueueFamily() const;

  /*! \brief Returns the extensions to be enabled.
   *
   * All char* in the returned vector point to static memory
   * allocations, and do not require cleanup.
   */
  std::vector<const char*> SelectEnabledExtensions() const;

  /*! \brief Initialize the VkDevice
   *
   * Called during VulkanDevice construction.  Assumes that
   * queue_family_index, device_properties, and enabled_extensions
   * have been set.
   */
  void CreateVkDevice(const VulkanInstance& instance);

  //! \brief Handle to the Vulkan API physical device
  VkPhysicalDevice physical_device_{nullptr};

  /*! \brief Extensions enabled for this device
   *
   * Based on supported extensions queried from physical_device_ prior
   * to creating device_.  Contains only statically allocated string
   * literals, no cleanup required.
   */
  std::vector<const char*> enabled_extensions;

  //! \brief Handle to the Vulkan API logical device
  VkDevice device_{nullptr};

  //! \brief Mutex to protect access to queue
  mutable std::mutex queue_mutex;

  /*! \brief Handle to Vulkan API VkQueue.
   *
   * Work can be executed by submitted to this queue using
   * VulkanDevice::QueueSubmit.
   */
  VkQueue queue{nullptr};

  /*! \brief The VulkanStream for each CPU thread.
   *
   * To mimic the semantics of cudaSetDevice and cuLaunchKernel, each
   * CPU thread must have a separate stream of execution.  The
   * ThreadMap is declared mutable so that the streams can be lazily
   * generated.
   */
  mutable ThreadMap<VulkanStream> stream_per_thread;

  //! \brief The VulkanStagingBuffer for each CPU thread.
  ThreadMap<VulkanStagingBuffer> staging_buffer_per_thread;

  //! \brief The VulkanUniformBuffer for each CPU thread.
  ThreadMap<VulkanUniformBuffer> uniform_buffer_per_thread;
};

uint32_t FindMemoryType(const VulkanDevice& device, VkBufferCreateInfo info,
                        VkMemoryPropertyFlags req_prop);

VkBufferCreateInfo MakeBufferCreateInfo(size_t nbytes, VkBufferUsageFlags usage);

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_DEVICE_H_
