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
 * \brief WebGPU compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {
namespace backend {
namespace webgpu {

ffi::Map<ffi::String, ffi::Any> UpdateWebGPUAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  bool subgroups = false;
  if (target.count("supports_subgroups")) {
    subgroups = target.at("supports_subgroups").cast<bool>();
  }

  if (target.count("thread_warp_size")) {
    int64_t thread_warp_size = target.at("thread_warp_size").cast<int64_t>();
    TVM_FFI_ICHECK(subgroups || thread_warp_size <= 1)
        << "WebGPU target with thread_warp_size=" << thread_warp_size
        << " requires supports_subgroups=true";
  }

  if (subgroups) {
    target.Set("thread_warp_size", int64_t(32));
  }
  return target;
}

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  TVM_REGISTER_TARGET_KIND("webgpu", kDLWebGPU)
      .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .add_attr_option<bool>("supports_subgroups", refl::DefaultValue(false))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(1))
      .add_attr_option<int64_t>("max_shared_memory_per_block", refl::DefaultValue(32768))
      .set_target_canonicalizer(UpdateWebGPUAttrs)
      .set_default_keys({"webgpu", "gpu"});
}

}  // namespace webgpu
}  // namespace backend

namespace codegen {
void RegisterWebGPUCodegen();
namespace intrin {
void RegisterWebGPUIntrinRules();
}  // namespace intrin
}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::webgpu::RegisterTargetKind();
  tvm::codegen::intrin::RegisterWebGPUIntrinRules();
  tvm::codegen::RegisterWebGPUCodegen();
}
