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
  ICHECK(device_type == kDLVulkan || device_type == kDLOpenCL || device_type == kDLWebGPU)
      << "Unsupported device type for SPIRV codegen:" << device_type;

  if (target->GetAttr<Integer>("vulkan_api_version")) {
    vulkan_api_version = target->GetAttr<Integer>("vulkan_api_version").value().IntValue();
  }

  if (target->GetAttr<Integer>("supported_subgroup_operations")) {
    supported_subgroup_operations =
        target->GetAttr<Integer>("supported_subgroup_operations").value().IntValue();
  }
  if (target->GetAttr<Integer>("max_push_constants_size")) {
    max_push_constants_size =
        target->GetAttr<Integer>("max_push_constants_size").value().IntValue();
  }
  if (target->GetAttr<Integer>("max_uniform_buffer_range")) {
    max_uniform_buffer_range =
        target->GetAttr<Integer>("max_uniform_buffer_range").value().IntValue();
  }
  if (target->GetAttr<Integer>("max_storage_buffer_range")) {
    max_storage_buffer_range =
        target->GetAttr<Integer>("max_storage_buffer_range").value().IntValue();
  }
  if (target->GetAttr<Integer>("max_shared_memory_per_block")) {
    max_shared_memory_per_block =
        target->GetAttr<Integer>("max_shared_memory_per_block").value().IntValue();
  }
  if (target->GetAttr<Integer>("max_per_stage_descriptor_storage_buffer")) {
    max_per_stage_descriptor_storage_buffers =
        target->GetAttr<Integer>("max_per_stage_descriptor_storage_buffer").value().IntValue();
  }
  if (target->GetAttr<Bool>("supports_storage_buffer_storage_class")) {
    supports_storage_buffer_storage_class =
        target->GetAttr<Bool>("supports_storage_buffer_storage_class").value();
  }
  if (target->GetAttr<Bool>("supports_8bit_buffer")) {
    supports_storage_buffer_8bit_access = target->GetAttr<Bool>("supports_8bit_buffer").value();
  }
  if (target->GetAttr<Bool>("supports_16bit_buffer")) {
    supports_storage_buffer_16bit_access = target->GetAttr<Bool>("supports_16bit_buffer").value();
  }
  if (target->GetAttr<Bool>("supports_float16")) {
    supports_float16 = target->GetAttr<Bool>("supports_float16").value();
  }
  if (target->GetAttr<Bool>("supports_float64")) {
    supports_float64 = target->GetAttr<Bool>("supports_float64").value();
  }
  if (target->GetAttr<Bool>("supports_int8")) {
    supports_int8 = target->GetAttr<Bool>("supports_int8").value();
  }
  if (target->GetAttr<Bool>("supports_int16")) {
    supports_int16 = target->GetAttr<Bool>("supports_int16").value();
  }
  if (target->GetAttr<Bool>("supports_int64")) {
    supports_int64 = target->GetAttr<Bool>("supports_int64").value();
  }
  // Check whether integer dot product is enabled in the target string.
  if (target->GetAttr<Bool>("supports_integer_dot_product")) {
    supports_integer_dot_product = target->GetAttr<Bool>("supports_integer_dot_product").value();
  }
  // Check whether integer dot product is enabled in mattr.
  if (const Optional<Array<String>>& v = target->GetAttr<Array<String>>("mattr")) {
    for (const String& s : v.value()) {
      if (s.compare("+dotprod") == 0) {
        supports_integer_dot_product = true;
        break;
      }
    }
  }
  // Check whether cooperative matrix is enabled in the target string.
  if (target->GetAttr<Bool>("supports_cooperative_matrix")) {
    supports_cooperative_matrix = target->GetAttr<Bool>("supports_cooperative_matrix").value();
  }
}

}  // namespace codegen
}  // namespace tvm
