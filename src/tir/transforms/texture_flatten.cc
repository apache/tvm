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
 * \file texture_flatten.cc
 * \brief Flattens texture storage from multi-dimensional array
 * to 2D (width, height) buffer access
 */

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/texture.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {
using runtime::IsTextureStorage;

class TextureFlattener : public StmtExprMutator {
 public:
  using StmtExprMutator::VisitStmt_;
  TextureFlattener() {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt body = this->VisitStmt(op->body);
    std::string storage_scope = GetStorageScope(op->buffer_var);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();

    if (IsTextureStorage(storage_scope)) {
      Array<PrimExpr> args = {op->extents.back()};
      stmt = LetStmt(op->buffer_var,
                     Call(op->buffer_var.dtype(), builtin::texture2d_alloca(), args), body);
    }

    return stmt;
  }

 protected:
  std::string GetStorageScope(const Var& var) {
    auto* ptr = var->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    return ptr->storage_scope;
  }
};

PrimFunc TextureFlatten(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = TextureFlattener()(std::move(fptr->body));
  return func;
}

namespace transform {

Pass TextureFlatten() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TextureFlatten(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.TextureFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.TextureFlatten").set_body_typed(TextureFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
