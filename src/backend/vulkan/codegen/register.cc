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
 * \file register.cc
 * \brief Vulkan compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {
namespace backend {
namespace vulkan {

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  TVM_REGISTER_TARGET_KIND("vulkan", kDLVulkan)
      .add_attr_option<ffi::Array<ffi::String>>("mattr")
      .add_attr_option<bool>("supports_float16")
      .add_attr_option<bool>("supports_float32", refl::DefaultValue(true))
      .add_attr_option<bool>("supports_float64")
      .add_attr_option<bool>("supports_int8")
      .add_attr_option<bool>("supports_int16")
      .add_attr_option<bool>("supports_int32", refl::DefaultValue(true))
      .add_attr_option<bool>("supports_int64")
      .add_attr_option<bool>("supports_8bit_buffer")
      .add_attr_option<bool>("supports_16bit_buffer")
      .add_attr_option<bool>("supports_storage_buffer_storage_class")
      .add_attr_option<bool>("supports_push_descriptor")
      .add_attr_option<bool>("supports_dedicated_allocation")
      .add_attr_option<bool>("supports_integer_dot_product")
      .add_attr_option<bool>("supports_cooperative_matrix")
      .add_attr_option<int64_t>("supported_subgroup_operations")
      .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .add_attr_option<int64_t>("max_threads_per_block", refl::DefaultValue(256))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(1))
      .add_attr_option<int64_t>("max_block_size_x")
      .add_attr_option<int64_t>("max_block_size_y")
      .add_attr_option<int64_t>("max_block_size_z")
      .add_attr_option<int64_t>("max_push_constants_size")
      .add_attr_option<int64_t>("max_uniform_buffer_range")
      .add_attr_option<int64_t>("max_storage_buffer_range")
      .add_attr_option<int64_t>("max_per_stage_descriptor_storage_buffer")
      .add_attr_option<int64_t>("max_shared_memory_per_block")
      .add_attr_option<ffi::String>("device_type")
      .add_attr_option<ffi::String>("device_name")
      .add_attr_option<ffi::String>("driver_name")
      .add_attr_option<int64_t>("driver_version")
      .add_attr_option<int64_t>("vulkan_api_version")
      .add_attr_option<int64_t>("max_spirv_version")
      .set_default_keys({"vulkan", "gpu"});
}

}  // namespace vulkan
}  // namespace backend

#ifdef TVM_ENABLE_SPIRV
namespace codegen {
void RegisterVulkanCodegen();
namespace spirv {
void RegisterVulkanIntrinRules();
}  // namespace spirv
}  // namespace codegen
#endif
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::vulkan::RegisterTargetKind();
#ifdef TVM_ENABLE_SPIRV
  tvm::codegen::spirv::RegisterVulkanIntrinRules();
  tvm::codegen::RegisterVulkanCodegen();
#endif
}
