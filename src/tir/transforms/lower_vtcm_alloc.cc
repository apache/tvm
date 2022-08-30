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

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tir {

inline bool IsVtcmStorage(std::string scope) {
  return scope.find("global.vtcm") != std::string::npos;
}

class VtcmAllocator : public StmtExprMutator {
 public:
  using StmtExprMutator::VisitStmt_;
  VtcmAllocator() {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    std::string storage_scope = GetStorageScope(op->buffer_var);
    if (IsVtcmStorage(storage_scope)) {
      Stmt body = this->VisitStmt(op->body);
      Array<PrimExpr> args;
      args.push_back(StringImm(storage_scope));
      args.push_back(IntImm(DataType::Int(64), op->extents.size()));
      args.push_back(Call(DataType::Handle(), builtin::tvm_stack_make_shape(), op->extents));
      return LetStmt(op->buffer_var,
                     Call(op->buffer_var.dtype(), builtin::nd_mem_alloc_with_scope(), args), body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

 protected:
  std::string GetStorageScope(const Var& var) {
    auto* ptr = var->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    return ptr->storage_scope;
  }
};

PrimFunc LowerVtcmAlloc(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = VtcmAllocator()(std::move(fptr->body));
  return func;
}

namespace transform {

Pass LowerVtcmAlloc() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerVtcmAlloc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerVtcmAlloc", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerVtcmAlloc").set_body_typed(LowerVtcmAlloc);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
