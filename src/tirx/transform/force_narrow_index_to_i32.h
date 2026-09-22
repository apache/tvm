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
 * \file force_narrow_index_to_i32.h
 * \brief Narrowing rules shared by the TIRX and S-TIR ForceNarrowIndexToInt32 passes.
 */
#ifndef TVM_TIR_TRANSFORM_FORCE_NARROW_INDEX_TO_I32_H_
#define TVM_TIR_TRANSFORM_FORCE_NARROW_INDEX_TO_I32_H_

#include <tvm/tirx/op.h>

#include <utility>

#include "../ir/data_type_rewriter.h"

namespace tvm {
namespace tirx {

/*!
 * \brief Force index expressions and integer buffers to int32.
 * \tparam Normalizer The index normalizer that determines which statements are traversed:
 *         tirx::IndexDataTypeNormalizer for lowered TIR, s_tir::IndexDataTypeNormalizer for
 *         TIR that still contains S-TIR blocks.
 */
template <typename Normalizer>
class Int32DTypeNarrowerBase : public Normalizer {
 public:
  using Normalizer::Mutate;
  using Normalizer::Mutate_;

 protected:
  explicit Int32DTypeNarrowerBase(PrimFunc func)
      : Normalizer(PrimType::Int(32)), func_(std::move(func)) {}

  /*! \brief Reject integer buffer parameters wider than int32. */
  static void CheckBufferParams(const PrimFunc& func) {
    for (const Var& param : func->params) {
      if (auto buffer = param.as<BufferVar>();
          buffer && buffer.value()->dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
          buffer.value()->dtype.bits() > 32) {
        TVM_FFI_THROW(InternalError) << "The buffer parameter " << buffer.value() << " has dtype "
                                     << buffer.value()->dtype << ". The function is " << func;
      }
    }
  }

  /*! \brief Reject allocated integer buffers wider than int32. */
  void CheckAllocatedBuffer(const BufferVar& buf) const {
    // Scalar assignments in TVMScript use local scalar storage.  Keep its explicit
    // dtype (e.g. an int64 opaque call result) and cast at narrowed index uses.
    // IsScalar checks the scalar layout contract, not merely the allocation size.
    bool is_local_scalar = buf.scope() == "local" && buf.IsScalar();
    if (!is_local_scalar && buf->dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
        buf->dtype.bits() > 32) {
      TVM_FFI_THROW(InternalError)
          << "The buffer " << buf << " allocated in the function has dtype " << buf->dtype
          << ". The function is " << func_;
    }
  }

  bool ShouldClampShiftAmounts() const final { return true; }

  UnchangedOr<PrimExpr> Mutate_(const IntImmNode* op, InplaceMode inplace_mode) final {
    // ignore the enabled condition and always rewrite i64
    if (op->ty.as_or_throw<PrimType>() == PrimType::Int(64)) {
      TVM_FFI_ICHECK_LE(
          op->value,
          prim::max_value(this->target_data_type_).template as_or_throw<IntImm>()->value);
      return IntImm::Int32(op->value);
    }
    return ffi::Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    auto result = Normalizer::Mutate_(op, inplace_mode);
    auto alloc = std::move(result)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .template as_or_throw<AllocBuffer>();
    CheckAllocatedBuffer(alloc->buffer);
    return alloc;
  }

  PrimFunc func_;
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORM_FORCE_NARROW_INDEX_TO_I32_H_
