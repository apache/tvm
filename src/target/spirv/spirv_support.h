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
 * \file spirv_support
 *
 * \brief Utility for determining which spirv capabilities a TVM
 * target supports.
 */
#ifndef TVM_TARGET_SPIRV_SPIRV_SUPPORT_H_
#define TVM_TARGET_SPIRV_SPIRV_SUPPORT_H_

#include <tvm/target/target.h>
#include <vulkan/vulkan_core.h>

namespace tvm {
namespace codegen {

/*! \brief Represents which support a Vulkan driver has that are relevant to codegen */
struct SPIRVSupport {
  /*! \brief Determine spirv capabilities from a vulkan target.
   */
  explicit SPIRVSupport(Target target);

  /*! \brief The Vulkan API version supported by the device.
   *
   *  Vulkan struct: VkPhysicalDeviceProperties
   *  Device property: apiVersion
   *
   *  If VK_KHR_driver_properties is present, will also check the
   *  driver conformance version.  If the version advertised does not
   *  pass the Vulkan conformance test, vulkan_api_version will be the
   *  latest Vulkan version that does pass the conformance test
   *  instead.
   */
  uint32_t vulkan_api_version{VK_MAKE_VERSION(1, 0, 0)};

  /*!
   * \brief The supported subgroup operations
   *
   * Vulkan extension: VK_KHR_driver_properties
   * Minimum vulkan version: 1.1
   * Vulkan struct: VkPhysicalDeviceSubgroupProperties
   * Device property: supportedOperations
   *
   * Requires vulkan 1.1 or higher to use.  If the
   * VK_KHR_driver_properties extension is not present in order to
   * query this value, or if the driver does not support vulkan 1.0,
   * then this value will be set to 0.
   *
   */
  uint32_t supported_subgroup_operations{0};

  /*!
   * \brief The maximum size (bytes) of push constants
   *
   * Vulkan struct: VkPhysicalDeviceLimits
   * Device property: maxPushConstantsSize
   *
   * The maxPushConstantsSize from VkPhysicalDeviceLimits.
   * Default value is from Vulkan spec, "Required Limits" table.
   * Implementations may have a larger limit.
   */
  uint32_t max_push_constants_size{128};

  /*!
   * \brief The maximum size (bytes) of a uniform buffer.
   *
   * Vulkan struct: VkPhysicalDeviceLimits
   * Device property: maxUniformBufferRange
   *
   * Default value is from Vulkan spec, "Required Limits" table.
   * Implementations may have a larger limit.
   */
  uint32_t max_uniform_buffer_range{16384};

  /*!
   * \brief The maximum size (bytes) of a storage buffer.
   *
   * Vulkan struct: VkPhysicalDeviceLimits
   * Device property: maxStorageBufferRange
   *
   * Default value is from Vulkan spec, "Required Limits" table.
   * Implementations may have a larger limit.
   */
  uint32_t max_storage_buffer_range{1 << 27};

  /*!
   * \brief The maximum amount of shared memory usable by a shader
   *
   * Vulkan extension: N/A
   * Vulkan struct: VkPhysicalDeviceLimits
   * Device Property: maxComputeSharedMemorySize
   * SPV Extension name: N/A
   * SPV Capability: N/A
   *
   * The maximum amount of shared memory (Workgroup scope) that may be
   * allocated by a shader.  Default value is from Vulkan spec,
   * "Required Limits" table.  Implementations may have a larger
   * limit.
   */
  uint32_t max_shared_memory_per_block{16384};

  /*!
   * \brief The maximum number of storage buffers accessible by a single shader.
   *
   * Vulkan struct: VkPhysicalDeviceLimits
   * Device property: maxPerStageDescriptorStorageBuffers
   *
   * Default value is from Vulkan spec, "Required Limits" table.
   * Implementations may have a larger limit, frequently much larger.
   * (e.g. GTX 1080 has max of 2^20)
   */
  uint32_t max_per_stage_descriptor_storage_buffers{4};

  /*!
   * \brief Whether the driver supports StorageClassStorageBuffer
   *
   * Vulkan extension: VK_KHR_storage_buffer_storage_class
   * Device property: N/A
   * SPV Extension: SPV_KHR_storage_buffer_storage_class
   * SPV Capability: N/A
   *
   * If support is present, access push constants and UBO as
   * block-decorated StorageClassStorageBuffer.  Otherwise, access as
   * buffer-block-decorated StorageClassUniform.  SPIRV 1.3 deprecated
   * BufferBlock, so this should always be true drivers that support
   * SPIRV 1.3.
   *
   */
  bool supports_storage_buffer_storage_class{false};

  /*!
   * \brief Whether the driver supports reading/writing to 16-bit values
   *
   * Vulkan extension: VK_KHR_8bit_storage
   * Vulkan struct: VkPhysicalDevice8BitStorageFeaturesKHR
   * Device property: storageBuffer8BitAccess
   * SPV extension: SPV_KHR_8bit_storage
   * SPV Capability: StorageBuffer8BitAccess
   *
   * If support is present, can read/write 8-bit values, but doesn't
   * necessarily provide 8-bit operations.
   *
   * If support is present, will declare StorageBuffer8BitAccess as
   * needed.  If support is not present, will throw error if a
   * PrimFunc calls for this functionality.  Unlike
   * StorageUniform16BitAccess, no fallback to
   * "StorageUniformBufferBlock8" is needed, as VK_KHR_8bit_storage
   * requires VK_KHR_storage_buffer_storage_class to also be present.
   *
   */
  bool supports_storage_buffer_8bit_access{false};

  /*!
   * \brief Whether the driver supports reading/writing to 16-bit values
   *
   * Vulkan extension: VK_KHR_16bit_storage
   * Vulkan struct: VkPhysicalDevice16BitStorageFeaturesKHR
   * Device property: storageBuffer16BitAccess
   * SPV extension: SPV_KHR_16bit_storage
   * SPV Capability: StorageBuffer16BitAccess, StorageUniformBufferBlock16
   *
   * If support is present, can read/write 16-bit values, but doesn't
   * necessarily provide 16-bit operations.
   *
   * If support is present, will declare either
   * StorageBuffer16BitAccess or StorageUniformBufferBlock16 as
   * needed, selecting based on the value of
   * supports_StorageBufferStorageClass.  If support is not present,
   * will throw error if a PrimFunc calls for this functionality.
   */
  bool supports_storage_buffer_16bit_access{false};

  /*!
   * \brief Whether the driver supports operations involving 16-bit floats
   *
   * Vulkan extension: VK_KHR_shader_float16_int8
   * Vulkan struct: VkPhysicalDeviceShaderFloat16Int8FeaturesKHR
   * Device Property: shaderFloat16
   * SPV Extension name: N/A
   * SPV Capability: Float16, Float16Buffer
   *
   * If support is present, can perform 16-bit float operations.  If
   * support is not present, codegen will throw exception on
   * attempting to create a 16-bit float.
   */
  bool supports_float16{false};

  /*!
   * \brief Whether the driver supports operations involving 16-bit floats
   *
   * Vulkan extension: N/A
   * Vulkan struct: VkPhysicalDeviceFeatures
   * Device Property: shaderFloat64
   * SPV Extension name: N/A
   * SPV Capability: Float64
   *
   * If support is present, can perform 64-bit float operations.  If
   * support is not present, codegen will throw exception on
   * attempting to create a 64-bit float.
   */
  bool supports_float64{false};

  /*!
   * \brief Whether the driver supports operations involving 8-bit ints
   *
   * Vulkan extension: VK_KHR_shader_float16_int8
   * Vulkan struct: VkPhysicalDeviceShaderFloat16Int8FeaturesKHR
   * Device Property: shaderInt8
   * SPV Extension name: N/A
   * SPV Capability: Int8
   *
   * If support is present, can perform 8-bit int operations.  If
   * support is not present, codegen will throw exception on
   * attempting to create a 8-bit int.
   */
  bool supports_int8{false};

  /*!
   * \brief Whether the driver supports operations involving 8-bit ints
   *
   * Vulkan extension: N/A
   * Vulkan struct: VkPhysicalDeviceFeatures
   * Device Property: shaderInt16
   * SPV Extension name: N/A
   * SPV Capability: Int16
   *
   * If support is present, can perform 16-bit int operations.  If
   * support is not present, codegen will throw exception on
   * attempting to create a 16-bit int.
   */
  bool supports_int16{false};

  /*!
   * \brief Whether the driver supports operations involving 64-bit ints
   *
   * Vulkan extension: N/A
   * Vulkan struct: VkPhysicalDeviceFeatures
   * Device Property: shaderInt64
   * SPV Extension name: N/A
   * SPV Capability: Int64
   *
   * If support is present, can perform 64-bit int operations.  If
   * support is not present, codegen will throw exception on
   * attempting to create a 64-bit int.
   */
  bool supports_int64{false};

  /*!
   * \brief Whether the driver supports operations involving integer dot product.
   *
   * Vulkan extension: VK_KHR_shader_integer_dot_product
   * SPV Extension name: SPV_KHR_integer_dot_product
   * SPV Capability: spv::CapabilityDotProductKHR,
   *                 spv::CapabilityDotProductInput4x8BitPackedKHR);
   *
   * If support is present, can perform integer dot product operations.  If
   * support is not present, codegen will throw exception on
   * attempting to perform integer dot product.
   */
  bool supports_integer_dot_product{false};

  /*!
   * \brief  Whether the driver supports operations involving cooperative matrix.
   *
   * Vulkan extension: VK_NV_cooperative_matrix
   * SPV Extension name: SPV_NV_cooperative_matrix
   * SPV Capability: spv::CapabilityCooperativeMatrixNV
   *
   * If support is present, can perform cooperative matrix operations.  If
   * support is not present, codegen will throw exception on
   * attempting to perform cooperative matrix.
   */

  bool supports_cooperative_matrix{false};
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_SPIRV_SPIRV_SUPPORT_H_
