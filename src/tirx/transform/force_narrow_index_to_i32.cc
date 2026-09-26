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
 * \file force_narrow_index_to_i32.cc
 * \brief Force narrow down indexing expressions and integer buffers to int32 dtype.
 * \note This pass is not used in default cases.
 */

#include "force_narrow_index_to_i32.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {

class Int32DTypeNarrower : public Int32DTypeNarrowerBase<IndexDataTypeNormalizer> {
 public:
  static PrimFunc RewriteDataType(PrimFunc func) {
    // The TIRX normalizer does not rewrite S-TIR block iterators, regions, or match buffers, so
    // narrowing a function that still contains blocks would leave their index types inconsistent.
    if (ContainsNode<s_tir::SBlockRealizeNode>(func->body)) {
      TVM_FFI_THROW(ValueError)
          << "tirx.transform.ForceNarrowIndexToInt32 requires a function without S-TIR blocks. "
          << "Use s_tir.transform.ForceNarrowIndexToInt32 before block lowering.";
    }
    CheckBufferParams(func);
    auto narrower = ffi::make_object<Int32DTypeNarrower>(func);
    return narrower->Rewrite(func);
  }

  explicit Int32DTypeNarrower(PrimFunc func) : Int32DTypeNarrowerBase(std::move(func)) {}
};

PrimFunc ForceNarrowIndexToInt32(PrimFunc func) {
  return Int32DTypeNarrower::RewriteDataType(func);
}

namespace transform {

Pass ForceNarrowIndexToInt32() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return ForceNarrowIndexToInt32(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.NarrowDataType", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.ForceNarrowIndexToInt32", ForceNarrowIndexToInt32);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
