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
 *  Copyright (c) 2018 by Contributors
 * \file well_formed.cc
 * \brief check that expression is well formed.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <unordered_set>

namespace tvm {
namespace relay {


//! brief make sure each Var is bind at most once.
class WellFormedChecker : private ExprVisitor, PatternVisitor {
  bool well_formed = true;

  std::vector<std::unordered_set<Var, NodeHash, NodeEqual>> scope;
  std::unordered_set<Var, NodeHash, NodeEqual> current_bound;
  std::unordered_set<Var, NodeHash, NodeEqual> total_bound;
  std::unordered_set<Var, NodeHash, NodeEqual> free;

  struct Scope {
    WellFormedChecker* wfc;
    explicit Scope(WellFormedChecker* wfc) : wfc(wfc) {
      wfc->scope.push_back({{}});
    }
    ~Scope() {
      CHECK_GE(wfc->scope.size(), 0);
      for (const Var& v : wfc->scope.back()) {
        CHECK_GE(wfc->current_bound.count(v), 0);
        wfc->current_bound.erase(v);
      }
      wfc->scope.pop_back();
    }
  };

  void Bound(const Var& v) {
    if (current_bound.count(v) != 0 || total_bound.count(v) != 0 || free.count(v) != 0) {
      well_formed = false;
    }
    CHECK_GE(scope.size(), 0);
    scope.back().insert(v);
    current_bound.insert(v);
    total_bound.insert(v);
  }

  void VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    if (current_bound.count(v) == 0) {
      if (total_bound.count(v) != 0) {
        well_formed = false;
      } else {
        free.insert(v);
      }
    }
  }

  void VisitExpr_(const LetNode* l) final {
    Scope s(this);
    // we do letrec only for FunctionNode,
    // but shadowing let in let binding is likely programming error, and we should forbidden it.
    Bound(l->var);
    CheckWellFormed(l->value);
    CheckWellFormed(l->body);
  }

  void VisitExpr_(const FunctionNode* f) final {
    Scope s(this);
    for (const Var& param : f->params) {
      Bound(param);
    }
    CheckWellFormed(f->body);
  }

  void VisitClause(const Clause& c) final {
    Scope s(this);
    VisitPattern(c->lhs);
    VisitExpr(c->rhs);
  }

  void VisitPattern(const Pattern& p) final {
    PatternVisitor::VisitPattern(p);
  }

  void VisitVar(const Var& v) final {
    Bound(v);
  }

  void VisitExpr(const Expr& e) final {
    if (auto v = e.as<VarNode>()) {
      VisitExpr_(v);
    } else {
      ExprVisitor::VisitExpr(e);
    }
  }

 public:
  bool CheckWellFormed(const Expr& e) {
    this->VisitExpr(e);
    return well_formed;
  }
};

bool WellFormed(const Expr& e) {
  return WellFormedChecker().CheckWellFormed(e);
}

TVM_REGISTER_API("relay._analysis.well_formed")
.set_body_typed(WellFormed);

}  // namespace relay
}  // namespace tvm
