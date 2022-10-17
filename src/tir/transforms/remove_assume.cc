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
 * \file remove_store_undef.cc
 * \brief Remove stores of tir::builtin::undef
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

// Remove any builtin::assume calls
class AssumeRemover : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::assume())) {
        return Evaluate(0);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

namespace transform {
Pass RemoveAssumeInternal() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = AssumeRemover()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveAssumeInternal", {});
}

Pass RemoveAssume() {
  return Sequential({RemoveAssumeInternal(), RemoveNoOp()}, "tir.RemoveAssume");
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveAssume").set_body_typed(RemoveAssume);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
