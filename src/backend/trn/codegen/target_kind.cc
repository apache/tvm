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
 * \brief Trainium compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

namespace backend {
namespace trn {

void RegisterTargetKind() {
  TVM_REGISTER_TARGET_KIND("trn", kDLTrn)
      .add_attr_option<int64_t>("partition_size", 128)
      .add_attr_option<int64_t>("max_sbuf_size_per_partition", 196608)
      .add_attr_option<int64_t>("max_psum_size_per_partition", 16384)
      .add_attr_option<int64_t>("num-cores");
}

}  // namespace trn
}  // namespace backend

namespace codegen {
void RegisterTRNCodegen();
}  // namespace codegen

namespace tirx {
namespace transform {
void RegisterTRNTransforms();
}  // namespace transform
}  // namespace tirx
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::trn::RegisterTargetKind();
  tvm::codegen::RegisterTRNCodegen();
  tvm::tirx::transform::RegisterTRNTransforms();
}
