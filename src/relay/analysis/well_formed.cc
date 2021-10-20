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
 * \file well_formed.cc
 * \brief check that expression is well formed.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/runtime/logging.h>

#include <unordered_set>

namespace tvm {
namespace relay {

//! brief make sure each Var is bound at most once in a scope.
class WellFormedChecker : private MixedModeVisitor, PatternVisitor {
 public:
  Optional<DiagnosticContext> diag_ctx;
  Span occurs_in;

  explicit WellFormedChecker(const Optional<DiagnosticContext>& ctx) : diag_ctx(ctx) {}

  bool well_formed = true;

  void Illformed(Diagnostic diag) {
    well_formed = false;
    if (diag_ctx) {
      diag_ctx.value().Emit(diag);
    } else {
      LOG(INFO) << "The IR is not well formed with: " << diag->message;
    }
  }

  std::vector<std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>> scope;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> current_bound;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> total_bound;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> free;

  struct Scope {
    WellFormedChecker* wfc;
    explicit Scope(WellFormedChecker* wfc) : wfc(wfc) { wfc->scope.push_back({{}}); }
    ~Scope() {
      ICHECK_GE(wfc->scope.size(), 0);
      for (const Var& v : wfc->scope.back()) {
        ICHECK_GE(wfc->current_bound.count(v), 0);
        wfc->current_bound.erase(v);
      }
      wfc->scope.pop_back();
    }
  };

  void Bound(const Var& v) {
    if (current_bound.count(v) != 0 || total_bound.count(v) != 0 || free.count(v) != 0) {
      Illformed(Diagnostic::Error(v->span) << "The variable " << v->name_hint()
                                           << " is bound more than once, this is not valid IR");
    }
    ICHECK_GE(scope.size(), 0);
    scope.back().insert(v);
    current_bound.insert(v);
    total_bound.insert(v);
  }

  using MixedModeVisitor::VisitExpr_;

  void VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    if (current_bound.count(v) == 0) {
      if (total_bound.count(v) != 0) {
        Illformed(Diagnostic::Error(v->span) << "the variable " << v->name_hint()
                                             << "is bound more then once, this is not valid IR");
      } else {
        free.insert(v);
      }
    }
  }

  void VisitExpr_(const LetNode* l) final {
    std::vector<Scope*> scopes;
    Expr let = GetRef<Let>(l);
    while (auto let_node = let.as<LetNode>()) {
      scopes.push_back(new Scope(this));
      // we do letrec only for FunctionNode,
      // but shadowing let in let binding is likely programming error, and we should forbidden it.
      Bound(let_node->var);
      CheckWellFormed(let_node->value);
      let = let_node->body;
    }
    CheckWellFormed(let);
    while (!scopes.empty()) {
      delete scopes.back();
      scopes.pop_back();
    }
  }

  void VisitExpr_(const FunctionNode* f) final {
    Scope s(this);
    for (const Var& param : f->params) {
      Bound(param);
    }
    CheckWellFormed(f->body);
  }

  void VisitExpr_(const CallNode* call) final {
    ICHECK(call->op.defined());

    for (auto arg : call->args) {
      ICHECK(arg.defined());
    }

    // ICHECK(call->attrs.defined());
    ICHECK(call->type_args.defined());
    MixedModeVisitor::VisitExpr_(call);
  }

  void VisitClause(const Clause& c) final {
    Scope s(this);
    VisitPattern(c->lhs);
    VisitExpr(c->rhs);
  }

  void VisitPattern(const Pattern& p) final { PatternVisitor::VisitPattern(p); }

  void VisitVar(const Var& v) final { Bound(v); }

 public:
  bool CheckWellFormed(const Expr& e) {
    if (auto v = e.as<VarNode>()) {
      VisitExpr_(v);
    } else {
      // this->occurs_in = e->span;
      VisitExpr(e);
    }
    return well_formed;
  }
};

bool WellFormed(const Expr& e, Optional<DiagnosticContext> diag_ctx) {
  return WellFormedChecker(diag_ctx).CheckWellFormed(e);
}

TVM_REGISTER_GLOBAL("relay.analysis.well_formed").set_body_typed([](Expr e) {
  return WellFormed(e);
});

}  // namespace relay
}  // namespace tvm
