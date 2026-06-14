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
 * \brief Hexagon compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {
namespace backend {
namespace hexagon {

void RegisterTargetKind() {
  TVM_REGISTER_TARGET_KIND("hexagon", kDLHexagon)
      .add_attr_option<ffi::Array<ffi::String>>("mattr")
      .add_attr_option<ffi::String>("mcpu")
      .add_attr_option<ffi::String>("mtriple")
      .add_attr_option<ffi::Array<ffi::String>>("llvm-options")
      .add_attr_option<int64_t>("num-cores")
      .add_attr_option<int64_t>("vtcm-capacity")
      .set_default_keys({"hexagon", "cpu"});
}

}  // namespace hexagon
}  // namespace backend

#ifdef TVM_LLVM_VERSION
namespace codegen {
void RegisterHexagonCodegen();
namespace llvm {
void RegisterHexagonIntrinRules();
}  // namespace llvm
}  // namespace codegen
#endif
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::hexagon::RegisterTargetKind();
#ifdef TVM_LLVM_VERSION
  tvm::codegen::llvm::RegisterHexagonIntrinRules();
  tvm::codegen::RegisterHexagonCodegen();
#endif
}
