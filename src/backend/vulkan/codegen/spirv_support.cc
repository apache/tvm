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

#include "spirv_support.h"

#include <spirv.hpp>

namespace tvm {
namespace codegen {

SPIRVSupport::SPIRVSupport(tvm::Target target) {
  auto device_type = target->GetTargetDeviceType();
  TVM_FFI_ICHECK(device_type == kDLVulkan || device_type == kDLOpenCL || device_type == kDLWebGPU)
      << "Unsupported device type for SPIRV codegen:" << device_type;

  vulkan_api_version = target->GetAttr<int64_t>("vulkan_api_version").value_or(vulkan_api_version);
  max_spirv_version = target->GetAttr<int64_t>("max_spirv_version").value_or(max_spirv_version);
  supported_subgroup_operations = target->GetAttr<int64_t>("supported_subgroup_operations")
                                      .value_or(supported_subgroup_operations);
  max_push_constants_size =
      target->GetAttr<int64_t>("max_push_constants_size").value_or(max_push_constants_size);
  max_uniform_buffer_range =
      target->GetAttr<int64_t>("max_uniform_buffer_range").value_or(max_uniform_buffer_range);
  max_storage_buffer_range =
      target->GetAttr<int64_t>("max_storage_buffer_range").value_or(max_storage_buffer_range);
  max_shared_memory_per_block =
      target->GetAttr<int64_t>("max_shared_memory_per_block").value_or(max_shared_memory_per_block);
  max_per_stage_descriptor_storage_buffers =
      target->GetAttr<int64_t>("max_per_stage_descriptor_storage_buffer")
          .value_or(max_per_stage_descriptor_storage_buffers);
  supports_storage_buffer_storage_class =
      target->GetAttr<bool>("supports_storage_buffer_storage_class")
          .value_or(supports_storage_buffer_storage_class);
  supports_storage_buffer_8bit_access =
      target->GetAttr<bool>("supports_8bit_buffer").value_or(supports_storage_buffer_8bit_access);
  supports_storage_buffer_16bit_access =
      target->GetAttr<bool>("supports_16bit_buffer").value_or(supports_storage_buffer_16bit_access);
  supports_float16 = target->GetAttr<bool>("supports_float16").value_or(supports_float16);
  supports_float64 = target->GetAttr<bool>("supports_float64").value_or(supports_float64);
  supports_int8 = target->GetAttr<bool>("supports_int8").value_or(supports_int8);
  supports_int16 = target->GetAttr<bool>("supports_int16").value_or(supports_int16);
  supports_int64 = target->GetAttr<bool>("supports_int64").value_or(supports_int64);
  // Check whether integer dot product is enabled in the target string.
  supports_integer_dot_product =
      target->GetAttr<bool>("supports_integer_dot_product").value_or(supports_integer_dot_product);
  // Check whether integer dot product is enabled in mattr.
  if (const ffi::Optional<ffi::Array<ffi::String>>& v =
          target->GetAttr<ffi::Array<ffi::String>>("mattr")) {
    for (const ffi::String& s : v.value()) {
      if (s.compare("+dotprod") == 0) {
        supports_integer_dot_product = true;
        break;
      }
    }
  }
  // Check whether cooperative matrix is enabled in the target string.
  supports_cooperative_matrix =
      target->GetAttr<bool>("supports_cooperative_matrix").value_or(supports_cooperative_matrix);
}

}  // namespace codegen
}  // namespace tvm
