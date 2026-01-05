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
 *  Lower TVM related builtin intrinsics such as packed call.
 * \file tir/transforms/inject_torch_mps_stream.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "ir_utils.h"

namespace tvm {
namespace tir {

// Calculate the statistics of packed function.
// These information are needed during codegen.
class InjectMPSStream : public StmtExprMutator {
 public:
  static PrimFunc Build(PrimFunc func) {
    return func;
  }
};

namespace transform {

Pass InjectTorchMPSStream() {
  auto pass_func = [](PrimFunc func, IRModule m, PassContext ctx) {
    if (IsHostFunc(func).value_or(false)) {
      func = InjectMPSStream::Build(func);
      VLOG(2) << "InjectTorchMPSStream: " << func;
    }
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectTorchMPSStream", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.InjectTorchMPSStream", InjectTorchMPSStream);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
