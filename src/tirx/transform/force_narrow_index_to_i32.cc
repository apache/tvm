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
 * \brief Force narrow down indexing expressions and integer tensors to int32 dtype.
 * \note This pass is not used in default cases.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "../ir/data_type_rewriter.h"

namespace tvm {
namespace tirx {

class Int32DTypeNarrower : public IndexDataTypeNormalizer {
 public:
  using IndexDataTypeNormalizer::Mutate;
  using IndexDataTypeNormalizer::Mutate_;

  static Function RewriteDataType(Function func) {
    CheckTensorParams(func);
    auto narrower = ffi::make_object<Int32DTypeNarrower>(func);
    return narrower->Rewrite(func);
  }

  explicit Int32DTypeNarrower(Function func)
      : IndexDataTypeNormalizer(PrimType::Int(32)), func_(std::move(func)) {}

 private:
  /*! \brief Reject integer tensor parameters wider than int32. */
  static void CheckTensorParams(const Function& func) {
    for (const Var& param : func->params) {
      if (auto tensor = param.as<TensorVar>();
          tensor && tensor.value()->dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
          tensor.value()->dtype.bits() > 32) {
        TVM_FFI_THROW(InternalError) << "The tensor parameter " << tensor.value() << " has dtype "
                                     << tensor.value()->dtype << ". The function is " << func;
      }
    }
  }

  /*! \brief Reject allocated integer tensors wider than int32. */
  void CheckAllocatedTensor(const TensorVar& buf) const {
    // Scalar assignments in TVMScript use local scalar storage.  Keep its explicit
    // dtype (e.g. an int64 opaque call result) and cast at narrowed index uses.
    // IsScalar checks the scalar layout contract, not merely the allocation size.
    static const auto local = runtime::MemoryScope::Get("local");
    bool is_local_scalar = buf.scope() == local && buf.IsScalar();
    if (!is_local_scalar && buf->dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
        buf->dtype.bits() > 32) {
      TVM_FFI_THROW(InternalError)
          << "The tensor " << buf << " allocated in the function has dtype " << buf->dtype
          << ". The function is " << func_;
    }
  }

  bool ShouldClampShiftAmounts() const final { return true; }

  UnchangedOr<PrimExpr> Mutate_(const IntImmNode* op, InplaceMode inplace_mode) final {
    // ignore the enabled condition and always rewrite i64
    if (op->ty.as_or_throw<PrimType>() == PrimType::Int(64)) {
      TVM_FFI_ICHECK_LE(op->value, prim::max_value(target_data_type_).as_or_throw<IntImm>()->value);
      return IntImm::Int32(op->value);
    }
    return ffi::Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op()))
      return MutateAllocTensor(op, inplace_mode);
    return IndexDataTypeNormalizer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode* op, InplaceMode inplace_mode) {
    auto result = IndexDataTypeNormalizer::Mutate_(op, inplace_mode);
    auto alloc = std::move(result).ValueOrUnchanged(ffi::GetRef<Stmt>(op)).as_or_throw<Bind>();
    CheckAllocatedTensor(alloc->var.as_or_throw<TensorVar>());
    return alloc;
  }

  Function func_;
};

Function ForceNarrowIndexToInt32(Function func) {
  return Int32DTypeNarrower::RewriteDataType(func);
}

namespace transform {

Pass ForceNarrowIndexToInt32() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    return ForceNarrowIndexToInt32(f);
  };
  return CreateFunctionPass(pass_func, 0, "tirx.NarrowDataType");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.ForceNarrowIndexToInt32", ForceNarrowIndexToInt32);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
