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
 * Copyright (c) 2018 by Contributors
 *
 * \file dead_code.cc
 *
 * \brief Remove code that does not effect the program result.
 *
 * The algorithm is implemented by two visitor:
 * CalcDep turn an expr into a dependency graph of expr,
 * GenLet turn the dependency graph into a let list, taking only the used value.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include "let_list.h"

namespace tvm {
namespace relay {

// calculate the dependency graph from expression
class CalcDep : private ExprVisitor {
 public:
  static Expr Eliminate(const Expr& e) {
    CalcDep cd;
    cd.Calculate(e);
    Eliminator el(cd.expr_map_, cd.use_map_, cd.letrec_set_);
    return el(e);
  }

 private:
  template<typename X>
  using VarMap = std::unordered_map<Var, X, NodeHash, NodeEqual>;
  using VarSet = std::unordered_set<Var, NodeHash, NodeEqual>;
  VarMap<Expr> expr_map_;
  VarMap<size_t> use_map_;
  VarSet letrec_set_;
  bool count_ = true;
  VarSet dead_worklist_;
  VarSet current_letrec_;

  void LetRec(const std::function<void()>& func, const Var& v) {
    current_letrec_.insert(v);
    func();
    current_letrec_.erase(v);
  }

  void VisitExpr_(const LetNode* l) final {
    if (count_) {
      CHECK_EQ(expr_map_.count(l->var), 0);
      CHECK_EQ(use_map_.count(l->var), 0);
      expr_map_[l->var] = l->value;
      use_map_[l->var] = 0;
      dead_worklist_.insert(l->var);
      LetRec([&]() { VisitExpr(l->value); }, l->var);
    }
    VisitExpr(l->body);
  }

  void VisitExpr(const Expr& e) final {
    ExprFunctor<void(const Expr&)>::VisitExpr(e);
  }

  void VisitExpr_(const VarNode* v) final {
    Var var = GetRef<Var>(v);
    if (expr_map_.count(var) == 0) {
      return;
    }
    if (current_letrec_.count(var) == 0) {
      if (count_) {
        use_map_[var] += 1;
        dead_worklist_.erase(var);
      } else {
        CHECK_GT(use_map_[var], 0) << var;
        use_map_[var] -= 1;
        if (use_map_[var] == 0) {
          dead_worklist_.insert(var);
        }
      }
    } else {
      letrec_set_.insert(var);
    }
  }

  void Calculate(const Expr& v) {
    VisitExpr(v);
    count_ = false;
    while (!dead_worklist_.empty()) {
      Var dead = *(dead_worklist_.begin());
      dead_worklist_.erase(dead);
      CHECK_EQ(use_map_[dead], 0);
      if (expr_map_.count(dead) > 0) {
        LetRec([&]() { VisitExpr(expr_map_[dead]); }, dead);
      }
    }
  }

  class Eliminator : private ExprMutator {
   private:
    VarMap<Expr> expr_map_;
    VarMap<size_t> use_map_;
    VarSet letrec_set_;
    explicit Eliminator(const VarMap<Expr>& expr_map,
                        const VarMap<size_t>& use_map,
                        const VarSet& letrec_set) :
      expr_map_(expr_map), use_map_(use_map), letrec_set_(letrec_set) { }
    friend CalcDep;

    bool HasLet(const Var& v) {
      // TODO(@jroesch): MK fix me
      return (use_map_[v] > 0 || (use_map_[v] != 0 && letrec_set_.count(v) != 0));
    }

    Expr VisitExpr_(const VarNode* op) final {
      Var v = GetRef<Var>(op);
      return (expr_map_.count(v) == 0 || HasLet(v)) ? v : VisitExpr(expr_map_[v]);
    }

    Expr VisitExpr_(const LetNode* op) final {
      Var v = op->var;
      if (HasLet(v)) {
        return LetNode::make(v, VisitExpr(op->value), VisitExpr(op->body));
      } else {
        return VisitExpr(op->body);
      }
    }
  };
};

Expr DeadCodeElimination(const Expr& e) {
  return CalcDep::Eliminate(e);
}

TVM_REGISTER_API("relay._ir_pass.dead_code_elimination")
.set_body_typed(DeadCodeElimination);

namespace transform {

Pass DeadCodeElimination() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
    return Downcast<Function>(DeadCodeElimination(f));
  };
  return CreateFunctionPass(pass_func, 1, "dead_code_elimination", {});
}

}  // namespace transform

}  // namespace relay
}  // namespace tvm
