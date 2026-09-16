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

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {

class AssertSkipper : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) override {
    if (input.as<ExprNode>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(input, inplace_mode);
  }
  UnchangedOr<Stmt> Mutate_(const AssertStmtNode* op, InplaceMode inplace_mode) final {
    // AssertStmt is a leaf — just remove it.
    return Evaluate(0);
  }
};

Stmt SkipAssert(Stmt stmt) {
  return ffi::make_object<AssertSkipper>()
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(stmt);
}

namespace transform {

Pass SkipAssert() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ffi::make_object<AssertSkipper>()
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.SkipAssert", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.SkipAssert", SkipAssert);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
