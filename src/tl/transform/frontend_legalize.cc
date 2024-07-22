/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file frontend_legalize.cc
 * \brief Legalize the program from frontend
 */

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;

class FrontendLegalizer : public arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    FrontendLegalizer substituter(&analyzer);
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const ForNode* node) final {
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_++;
    }
    auto n = StmtExprMutator::VisitStmt_(node);
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_--;
    }
    return n;
  }

  PrimExpr VisitExpr_(const VarNode* node) final {
    if (let_bindings_.count(node)) {
      return arith::IRMutatorWithAnalyzer::VisitExpr(let_bindings_[node]);
    } else {
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }
  }

  Stmt VisitStmt_(const LetStmtNode* node) final {
    let_bindings_[node->var.get()] = node->value;
    return arith::IRMutatorWithAnalyzer::VisitStmt(node->body);
  }

  PrimExpr VisitExpr_(const LetNode* node) final {
    let_bindings_[node->var.get()] = node->value;
    return arith::IRMutatorWithAnalyzer::VisitExpr(node->body);
  }

  int parallel_for_scope_ = 0;
  std::unordered_map<const VarNode*, PrimExpr> let_bindings_;
};

using namespace tir::transform;

Pass FrontendLegalize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FrontendLegalizer::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.FrontendLegalize", {});
}

TVM_REGISTER_GLOBAL("tl.FrontendLegalize").set_body_typed(FrontendLegalize);

}  // namespace tl
}  // namespace tvm
