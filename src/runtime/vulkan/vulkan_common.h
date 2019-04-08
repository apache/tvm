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

/*!
 *  Copyright (c) 2017 by Contributors
 * \file vulkan_common.h
 * \brief Vulkan common header
 */
#ifndef TVM_RUNTIME_VULKAN_VULKAN_COMMON_H_
#define TVM_RUNTIME_VULKAN_VULKAN_COMMON_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <dmlc/logging.h>

#include <vulkan/vulkan.h>
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {
namespace vulkan {

inline const char* VKGetErrorString(VkResult error) {
  switch (error) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    default: return "Unknown Vulkan error code";
  }
}

/*!
 * \brief Protected Vulkan call
 * \param func Expression to call.
 */
#define VULKAN_CHECK_ERROR(__e)                                         \
  {                                                                     \
    CHECK(__e == VK_SUCCESS)                                            \
        << "Vulan Error, code=" << __e << ": " << vulkan::VKGetErrorString(__e); \
  }

#define VULKAN_CALL(func)                                             \
  {                                                                   \
    VkResult __e = (func);                                            \
    VULKAN_CHECK_ERROR(__e);                                          \
  }

/*! \brief Auxiliary context structure for vulkan */
struct VulkanContext {
  // phyiscal device
  VkPhysicalDevice phy_device{nullptr};
  // Phyiscal device property
  VkPhysicalDeviceProperties phy_device_prop;
  // Memory type index for staging.
  uint32_t staging_mtype_index{0};
  // whether staging is coherent
  bool coherent_staging{false};
  // Memory type index for compute
  uint32_t compute_mtype_index{0};
  // The logical device
  VkDevice device{nullptr};
  // command queue
  VkQueue queue{nullptr};
  // queue family_index;
  uint32_t queue_family_index{0};
  // Queue family index.
  VkQueueFamilyProperties queue_prop;
};

/*! \brief The buffer object */
struct VulkanBuffer {
  /*! \brief underlying buffer */
  VkBuffer buffer{VK_NULL_HANDLE};
  /*! \brief underlying buffer */
  VkDeviceMemory memory{VK_NULL_HANDLE};
};

/*! \brief Buffer only used for stagging */
struct VulkanStagingBuffer {
  /*! \brief the corresponding device */
  VkDevice device{nullptr};
  /*! \brief underlying buffer */
  VkBuffer buffer{VK_NULL_HANDLE};
  /*! \brief underlying buffer */
  VkDeviceMemory memory{VK_NULL_HANDLE};
  /*! \brief host address */
  void* host_addr{nullptr};
  /*! \brief size of the memory */
  size_t size{0};
};

/*!
 * \brief Process global Vulkan workspace.
 */
class VulkanWorkspace final : public DeviceAPI {
 public:
  // global mutex
  std::mutex mu;
  // whether the workspace it initialized.
  bool initialized_{false};
  // vulkan instance
  VkInstance instance_{nullptr};
  // The physical devices, have 1 to 1 mapping to devices
  std::vector<VulkanContext> context_;
  // Destructor
  ~VulkanWorkspace();
  // Initialize workspace
  // Return false if already initialized, otherwise return true.
  void Init();
  // override device API
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from,
                      size_t from_size,
                      void* to,
                      size_t to_size,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final;
  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;
  // get the global workspace
  static const std::shared_ptr<VulkanWorkspace>& Global();
};

/*! \brief Helper command buffer resource */
struct VulkanCommandBuffer {
  /*! \brief fence to signal the resource is ready to use */
  VkFence fence{VK_NULL_HANDLE};
  /*! \brief The internal command buffer */
  VkCommandBuffer cmd_buffer{nullptr};
  /*! \brief Descriptor set used to bind arguments */
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  /*! \brief Internal utilities for write command */
  VkWriteDescriptorSet write_descriptor_set;

  VulkanCommandBuffer() {
    write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set.pNext = nullptr;
    write_descriptor_set.dstSet = VK_NULL_HANDLE;
    write_descriptor_set.dstBinding = 0;
    write_descriptor_set.dstArrayElement = 0;
    write_descriptor_set.descriptorCount = 1;
    write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set.pImageInfo = nullptr;
    write_descriptor_set.pBufferInfo = nullptr;
    write_descriptor_set.pTexelBufferView = nullptr;
  }
};

/*!
 * \brief Command pool backed by a fixed size ring buffer.
 *
 *  Vulkan requires us not to reuse command buffer until
 *  All its corresponding jobs have finished.
 *
 *  This class to faciliate automatic management
 *  of the command buffers. A fence is created
 *  for each launch of command buffer jobs
 *  and when we try to reuse the same entry
 *  in the ring, we need to make sure that
 *  the previous pending job already finishes.
 *
 */
class VulkanCommandPool {
 public:
  /*! \brief Maximum number of pending jobs in the pool */
  static constexpr const int kMaxPending = 4;
  /*! \brief Maximum number of pending jobs in the pool */
  static constexpr const int kMaxNumArgs = 16;
  /*!
   * \brief constructor
   * \param vctx The corresponding vulkan context.
   */
  explicit VulkanCommandPool(const VulkanContext& vctx);
  /*! \brief destructor */
  ~VulkanCommandPool();
  /*!
   * \brief Allocate a new command buffer entry
   *
   *  The caller must only submit the entry once
   *  with the given fence in the entry,
   *  before calling next Alloc.
   *
   *  This function may block to wait for a
   *  previously unfinished command when
   *  there is more than kMaxPending jobs.
   *
   * \returns The allocated entry.
   */
  VulkanCommandBuffer* Alloc();

  /*!
   * \brief Allocate a new command buffer entry
   * \param dlayout the descriptor layout.
   *
   * \returns The allocated entry.
   */
  VulkanCommandBuffer* Alloc(const VkDescriptorSetLayout* dlayout);

 private:
  /*! \brief Local ring buffer */
  std::vector<VulkanCommandBuffer> ring_;
  /*! \brief clock pointer */
  size_t clock_ptr_{0};
  /*! \brief the corresponding device*/
  VkDevice device_{nullptr};
  /*! \brief internal command buffer pool */
  VkCommandPool cmd_pool_{VK_NULL_HANDLE};
  /*! \brief Descriptor pool */
  VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
};

/*! \brief Thread local workspace */
class VulkanThreadEntry {
 public:
  /*! \brief The current context */
  TVMContext context;
  /*! \brief workspace pool */
  WorkspacePool pool;
  /*! \brief The staging buffers */
  std::vector<VulkanStagingBuffer> staging_buffer_;
  /*!
   * \brief Get the command pool of corresponding device;
   * \param device_id The device id
   * \return The corresponding command buffer.
   */
  VulkanCommandPool* CommandPool(int device_id);
  /*!
   * \brief Get the stagging buffer.
   * \param device_id The device id
   * \return The corresponding stagging buffer.
   */
  VulkanStagingBuffer* StagingBuffer(int device_id, size_t size);

  // constructor
  VulkanThreadEntry()
      : pool(static_cast<DLDeviceType>(kDLVulkan), VulkanWorkspace::Global()) {
    context.device_id = 0;
    context.device_type = static_cast<DLDeviceType>(kDLVulkan);
  }
  ~VulkanThreadEntry();
  // get the global workspace
  static VulkanThreadEntry* ThreadLocal();

 private:
  /*! \brief the command pools */
  std::vector<std::unique_ptr<VulkanCommandPool> > pool_;
};

// inline implementation


}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_COMMON_H_
