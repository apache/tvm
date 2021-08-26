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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_COMMON_H_
#define TVM_RUNTIME_VULKAN_VULKAN_COMMON_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <vulkan/vulkan.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace vulkan {

/*! \brief Maximum number of GPU supported in VulkanModule. */
static constexpr const int kVulkanMaxNumDevice = 8;

/*! \brief TVM Vulkan binary pack magic number */
static constexpr const int kVulkanModuleMagic = 0x02700027;

const int kMaxPushConstantsBytes = 128;

/*! \brief A mask used when we attach additional information to shaders */
enum ShaderMetaDataFlagMask { kUseUBO = 0 };

inline const char* VKGetErrorString(VkResult error) {
  switch (error) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_EVENT_SET:
      return "VK_EVENT_SET";
    case VK_EVENT_RESET:
      return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
      return "VK_ERROR_FRAGMENTED_POOL";
    default:
      return "Unknown Vulkan error code";
  }
}

/*!
 * \brief Protected Vulkan call
 * \param func Expression to call.
 */
#define VULKAN_CHECK_ERROR(__e)                                       \
  {                                                                   \
    ICHECK(__e == VK_SUCCESS) << "Vulkan Error, code=" << __e << ": " \
                              << vulkan::VKGetErrorString(__e);       \
  }

#define VULKAN_CALL(func)    \
  {                          \
    VkResult __e = (func);   \
    VULKAN_CHECK_ERROR(__e); \
  }

std::vector<const char*> FindEnabledExtensions(const std::vector<VkExtensionProperties>& ext_prop,
                                               const std::vector<const char*>& required_extensions,
                                               const std::vector<const char*>& optional_extensions);

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_COMMON_H_
