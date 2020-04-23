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
 *  SSA related checks and pass.
 *
 *  SSA requires each varaible to be only defined once.
 * \file verify_ssa.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/analysis.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tir {

class IRVerifySSA final : public StmtExprVisitor {
 public:
  bool is_ssa{true};

  void VisitExpr(const PrimExpr& n) final {
    if (!is_ssa) return;
    StmtExprVisitor::VisitExpr(n);
  }
  void VisitStmt(const Stmt& n) final {
    if (!is_ssa) return;
    StmtExprVisitor::VisitStmt(n);
  }
  void VisitExpr_(const LetNode* op) final {
    MarkDef(op->var.get());
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const LetStmtNode* op) final {
    MarkDef(op->var.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode* op) final {
    MarkDef(op->loop_var.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AllocateNode* op) final {
    MarkDef(op->buffer_var.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode* node) final {
    if (match_scope_) {
      MarkDef(node, true);
    }
  }

  void Run(const PrimFunc& func) {
    for (auto param : func->params) {
      MarkDef(param.get());
    }

    for (auto kv : func->buffer_map) {
      this->DefineBuffer(kv.second);
    }
    this->VisitStmt(func->body);
  }

  void DefineBuffer(const Buffer& buffer) {
    match_scope_ = true;
    this->VisitExpr(buffer->data);
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      this->VisitExpr(buffer->shape[i]);
    }

    if (buffer->strides.defined()) {
      for (size_t i = 0; i < buffer->strides.size(); ++i) {
        this->VisitExpr(buffer->strides[i]);
      }
    }
    this->VisitExpr(buffer->elem_offset);

    match_scope_ = false;
  }

 private:
  void MarkDef(const VarNode* v, bool allow_dup = false) {
    if (defined_.count(v) != 0) {
      if (!allow_dup) {
        is_ssa = false; return;
      }
    } else {
      defined_[v] = 1;
    }
  }
  // whether we are in match scope, where a var can occur multiple times.
  bool match_scope_{false};
  std::unordered_map<const VarNode*, int> defined_;
};


bool VerifySSA(const PrimFunc& func) {
  IRVerifySSA visitor;
  visitor.Run(func);
  return visitor.is_ssa;
}

TVM_REGISTER_GLOBAL("tir.analysis.verify_ssa")
.set_body_typed(VerifySSA);


namespace transform {

Pass VerifySSA() {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        auto func = GetRef<PrimFunc>(n);
        CHECK(VerifySSA(func))
            << "RuntimeError: IR is not in SSA form"
            << func;
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.VerifySSA", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VerifySSA")
.set_body_typed(VerifySSA);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
