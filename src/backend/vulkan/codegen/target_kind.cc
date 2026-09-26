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
 * \file target_kind.cc
 * \brief Vulkan compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

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
  using namespace tvm;
  namespace refl = tvm::ffi::reflection;

  TargetKindDef("vulkan")
      .set_default_device_type(kDLVulkan)
      .def_option<ffi::Array<ffi::String>>("mattr")
      .def_option<bool>("supports_float16")
      .def_option<bool>("supports_float32", refl::DefaultValue(true))
      .def_option<bool>("supports_float64")
      .def_option<bool>("supports_int8")
      .def_option<bool>("supports_int16")
      .def_option<bool>("supports_int32", refl::DefaultValue(true))
      .def_option<bool>("supports_int64")
      .def_option<bool>("supports_8bit_buffer")
      .def_option<bool>("supports_16bit_buffer")
      .def_option<bool>("supports_storage_buffer_storage_class")
      .def_option<bool>("supports_push_descriptor")
      .def_option<bool>("supports_dedicated_allocation")
      .def_option<bool>("supports_integer_dot_product")
      .def_option<bool>("supports_cooperative_matrix")
      .def_option<int64_t>("supported_subgroup_operations")
      .def_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .def_option<int64_t>("max_threads_per_block", refl::DefaultValue(256))
      .def_option<int64_t>("thread_warp_size", refl::DefaultValue(1))
      .def_option<int64_t>("max_block_size_x")
      .def_option<int64_t>("max_block_size_y")
      .def_option<int64_t>("max_block_size_z")
      .def_option<int64_t>("max_push_constants_size")
      .def_option<int64_t>("max_uniform_buffer_range")
      .def_option<int64_t>("max_storage_buffer_range")
      .def_option<int64_t>("max_per_stage_descriptor_storage_buffer")
      .def_option<int64_t>("max_shared_memory_per_block")
      .def_option<ffi::String>("device_type")
      .def_option<ffi::String>("device_name")
      .def_option<ffi::String>("driver_name")
      .def_option<int64_t>("driver_version")
      .def_option<int64_t>("vulkan_api_version")
      .def_option<int64_t>("max_spirv_version")
      .set_default_keys({"vulkan", "gpu"});
#ifdef TVM_ENABLE_SPIRV
  tvm::codegen::spirv::RegisterVulkanIntrinRules();
  tvm::codegen::RegisterVulkanCodegen();
#endif
}
