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
 *
 * \file dead_code.cc
 *
 * \brief Remove code that does not effect the program result.
 *
 * The algorithm is implemented by two visitor:
 * CalcDep turn an expr into a dependency graph of expr,
 * GenLet turn the dependency graph into a let list, taking only the used value.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "let_list.h"

namespace tvm {
namespace relay {

template <typename X>
using VarMap = std::unordered_map<Var, X, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

class CalcDep;
class FindDef : private ExprVisitor {
 private:
  VarMap<Expr> expr_map_;

  void VisitExpr_(const LetNode* l) final {
    auto pre_visit = [this](const LetNode* op) {
      ICHECK_EQ(expr_map_.count(op->var), 0);
      expr_map_[op->var] = op->value;
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(l, pre_visit, post_visit);
  }

  friend CalcDep;
};

class Eliminator : private ExprMutator {
 private:
  VarMap<Expr> expr_map_;
  VarMap<size_t> use_map_;
  bool inline_once_;
  explicit Eliminator(const VarMap<Expr>& expr_map, const VarMap<size_t>& use_map, bool inline_once)
      : expr_map_(expr_map), use_map_(use_map), inline_once_(inline_once) {}
  friend CalcDep;

  bool HasLet(const Var& v) {
    switch (use_map_[v]) {
      case 0:
        return false;
      case 1:
        return !inline_once_;
      default:
        return true;
    }
  }

  Expr VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    return (expr_map_.count(v) == 0 || HasLet(v)) ? v : VisitExpr(expr_map_[v]);
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      if (HasLet(op->var)) {
        Expr value = this->VisitExpr(op->value);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);
      Var v = op->var;
      if (HasLet(v)) {
        Expr value = this->VisitExpr(op->value);
        this->memo_[expr] = Let(v, value, body);
      } else {
        this->memo_[expr] = body;
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }
};

// calculate the dependency graph from expression
class CalcDep : protected MixedModeVisitor {
 public:
  static Expr Eliminate(const Expr& e, bool inline_once) {
    FindDef fd;
    fd(e);
    CalcDep cd(fd.expr_map_);
    cd(e);
    Eliminator el(fd.expr_map_, cd.use_map_, inline_once);
    return el(e);
  }

 private:
  explicit CalcDep(const VarMap<Expr>& expr_map) : MixedModeVisitor(2), expr_map_(expr_map) {}
  VarMap<Expr> expr_map_;
  VarMap<size_t> use_map_;

  using MixedModeVisitor::VisitExpr_;

  void VisitLeaf(const Expr& e) final {
    visit_counter_[e.get()]++;
    // The dce code seprate variable into three parts:
    // used 0 times (remove)
    // used 1 times (inline)
    // used 2 times (dont do anything).
    if (visit_counter_[e.get()] <= 2) {
      using TParent = ExprFunctor<void(const Expr&)>;
      TParent::VisitExpr(e);
    }
  }

  void VisitExpr_(const LetNode* l) final {
    Expr let_binding = GetRef<Expr>(l);
    const LetNode* let;
    while ((let = let_binding.as<LetNode>())) {
      let_binding = let->body;
      visit_counter_[l] += 1;
    }
    VisitExpr(let_binding);
  }

  void VisitExpr_(const VarNode* v) final {
    Var var = GetRef<Var>(v);
    ++use_map_[var];
    if (use_map_[var] == 1 && expr_map_.count(var) > 0) {
      VisitExpr(expr_map_[var]);
    }
  }
};

Expr DeadCodeElimination(const Expr& e, bool inline_once) {
  return CalcDep::Eliminate(e, inline_once);
}

namespace transform {

Pass DeadCodeElimination(bool inline_once) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DeadCodeElimination(f, inline_once));
      };
  return CreateFunctionPass(pass_func, 1, "DeadCodeElimination", {});
}

TVM_REGISTER_GLOBAL("relay._transform.DeadCodeElimination").set_body_typed(DeadCodeElimination);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
