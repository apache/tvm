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

#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class Int32DTypeNarrower : public IndexDataTypeNormalizer {
 public:
  static PrimFunc RewriteDataType(PrimFunc func) {
    // Check if the integer parameter buffers have dtype other than int32.
    for (auto it : func->buffer_map) {
      if (it.second->dtype.is_int() && it.second->dtype.bits() > 32) {
        LOG(FATAL) << "The buffer " << it.second << " in the function buffer map has dtype "
                   << it.second->dtype << ". The function is " << func;
      }
    }

    Int32DTypeNarrower narrower(func);
    return narrower.Rewrite(func);
  }

 private:
  explicit Int32DTypeNarrower(PrimFunc func)
      : IndexDataTypeNormalizer(DataType::Int(32)), func_(std::move(func)) {}

  PrimExpr VisitExpr_(const IntImmNode* op) final {
    // ignore the enabled condition and always rewrite i64
    if (op->dtype == DataType::Int(64)) {
      ICHECK_LE(op->value, Downcast<Integer>(max_value(target_data_type_))->value);
      return IntImm(DataType::Int(32), op->value);
    }
    return GetRef<IntImm>(op);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block block_ = Downcast<Block>(IndexDataTypeNormalizer::VisitStmt_(block));
    // Check if the allocated integer buffers have dtype other than int32.
    for (const Buffer& buf : block_->alloc_buffers) {
      if (buf->dtype.is_int() && buf->dtype.bits() > 32) {
        LOG(FATAL) << "The buffer " << buf << " allocated in the function has dtype " << buf->dtype
                   << ". The function is " << func_;
      }
    }
    return block_;
  }

  PrimFunc func_;
};

PrimFunc ForceNarrowIndexToInt32(PrimFunc func) {
  return Int32DTypeNarrower::RewriteDataType(func);
}

namespace transform {

Pass ForceNarrowIndexToInt32() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return ForceNarrowIndexToInt32(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.NarrowDataType", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ForceNarrowIndexToInt32")
    .set_body_typed(ForceNarrowIndexToInt32);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
