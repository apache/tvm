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
 * \brief Force narrow down indexing expressions and integer buffers to int32 dtype in functions
 *        that still contain S-TIR blocks.
 * \note This pass is not used in default cases.
 */

#include "../../tirx/transform/force_narrow_index_to_i32.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/transform.h>

#include "../ir/data_type_rewriter.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class Int32DTypeNarrower : public Int32DTypeNarrowerBase<IndexDataTypeNormalizer> {
 public:
  using Int32DTypeNarrowerBase::Mutate;
  using Int32DTypeNarrowerBase::Mutate_;
  static PrimFunc RewriteDataType(PrimFunc func) {
    CheckBufferParams(func);
    auto narrower = ffi::make_object<Int32DTypeNarrower>(func);
    return narrower->Rewrite(func);
  }

  explicit Int32DTypeNarrower(PrimFunc func) : Int32DTypeNarrowerBase(std::move(func)) {}

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) final {
    auto result = IndexDataTypeNormalizer::Mutate_(op, inplace_mode);
    auto block = std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op)).as_or_throw<SBlock>();
    for (const BufferVar& buf : block->alloc_buffers) {
      CheckAllocatedBuffer(buf);
    }
    return block;
  }
};

namespace transform {

Pass ForceNarrowIndexToInt32() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return Int32DTypeNarrower::RewriteDataType(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.ForceNarrowIndexToInt32", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.ForceNarrowIndexToInt32", ForceNarrowIndexToInt32);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
