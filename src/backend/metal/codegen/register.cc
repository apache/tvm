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
 * \brief Metal compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

namespace backend {
namespace metal {

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  // Metal limits the number of kernel arguments.  `max_function_args` captures that bound.
  TVM_REGISTER_TARGET_KIND("metal", kDLMetal)
      .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .add_attr_option<int64_t>("max_threads_per_block", refl::DefaultValue(256))
      .add_attr_option<int64_t>("max_shared_memory_per_block", refl::DefaultValue(32768))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(16))
      .add_attr_option<int64_t>("max_function_args", refl::DefaultValue(31))
      .set_default_keys({"metal", "gpu"});
}

}  // namespace metal
}  // namespace backend

namespace codegen {
void RegisterMetalCodegen();
namespace intrin {
void RegisterMetalIntrinRules();
}  // namespace intrin
}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::metal::RegisterTargetKind();
  tvm::codegen::intrin::RegisterMetalIntrinRules();
  tvm::codegen::RegisterMetalCodegen();
}
