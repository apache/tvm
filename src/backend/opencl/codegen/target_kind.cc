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
 * \brief OpenCL compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {
namespace backend {
namespace opencl {

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  TVM_REGISTER_TARGET_KIND("opencl", kDLOpenCL)
      .add_attr_option<int64_t>("max_threads_per_block", refl::DefaultValue(256))
      .add_attr_option<int64_t>("max_shared_memory_per_block", refl::DefaultValue(16384))
      .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(1))
      .add_attr_option<int64_t>("texture_spatial_limit", refl::DefaultValue(16384))
      .add_attr_option<int64_t>("texture_depth_limit", refl::DefaultValue(2048))
      // Qualcomm OpenCL runtimes may crash when the number of kernel arguments is too large.
      .add_attr_option<int64_t>("max_function_args", refl::DefaultValue(128))
      .add_attr_option<int64_t>("image_base_address_alignment", refl::DefaultValue(64))
      .set_default_keys({"opencl", "gpu"});
}

}  // namespace opencl
}  // namespace backend

namespace codegen {
void RegisterOpenCLCodegen();
void RegisterOpenCLDeviceScopeCompatibility();
namespace intrin {
void RegisterOpenCLIntrinRules();
}  // namespace intrin
}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::opencl::RegisterTargetKind();
  tvm::codegen::intrin::RegisterOpenCLIntrinRules();
  tvm::codegen::RegisterOpenCLCodegen();
  tvm::codegen::RegisterOpenCLDeviceScopeCompatibility();
}
