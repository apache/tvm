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
 *  Copyright (c) 2019 by Contributors
 * \file stmt_simplify.cc
 * \brief Statement simplifier based on analyzer
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <tvm/arithmetic.h>
#include "arithmetic/Simplify.h"

namespace tvm {
namespace arith {
// statement simplifier
using namespace ir;

class StmtSimplifier : public IRMutator {
 public:
  Stmt Mutate_(const For* op, const Stmt& s) final {
    Var loop_var(op->loop_var.node_);
    analyzer_.Bind(loop_var, Range::make_by_min_extent(op->min, op->extent));
    return IRMutator::Mutate_(op, s);
  }

  // IfThenElse
  Stmt Mutate_(const IfThenElse* op, const Stmt& s) {
    Expr condition = this->Mutate(op->condition);
    Stmt then_case, else_case;
    {
      ConstraintContext ctx(&analyzer_, condition);
      then_case = this->Mutate(op->then_case);
    }
    if (op->else_case.defined()) {
      ConstraintContext ctx(&analyzer_, Mutate(Not::make(condition)));
      else_case = this->Mutate(op->else_case);
    }
    if (is_one(condition)) return then_case;
    if (is_zero(condition)) {
      if (else_case.defined()) {
        return else_case;
      }
      return Evaluate::make(0);
    }

    if (condition.same_as(op->condition) &&
        then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return s;
    } else {
      return IfThenElse::make(condition, then_case, else_case);
    }
  }

  // AttrStmt
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      IterVar iv(op->node.node_);
      CHECK_NE(iv->thread_tag.length(), 0U);
      if (!var_dom_.count(iv->var.get())) {
        Range dom = Range::make_by_min_extent(0, op->value);
        var_dom_[iv->var.get()] = dom;
        analyzer_.Bind(iv->var, dom);
      }
      Stmt stmt = IRMutator::Mutate_(op, s);
      return stmt;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  // AssertStmt
  Stmt Mutate_(const AssertStmt* op, const Stmt& s) final {
    Expr condition = this->Mutate(op->condition);
    Expr message = this->Mutate(op->message);
    ConstraintContext ctx(&analyzer_, condition);
    Stmt body = this->Mutate(op->body);

    if (condition.same_as(op->condition) &&
        message.same_as(op->message) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return AssertStmt::make(condition, message, body);
    }
  }

 protected:
  Analyzer analyzer_;
  // variable domain
  std::unordered_map<const Variable*, Range> var_dom_;
};


class CanonicalStmtSimplifier : public StmtSimplifier {
 public:
  using StmtSimplifier::Mutate;
  Expr Mutate(Expr expr) final {
    return analyzer_.canonical_simplify(expr);
  }

  Stmt CanonicalSimplify(Stmt stmt, Map<Var, Range> vrange) {
    for (auto kv : vrange) {
      analyzer_.Bind(kv.first, kv.second);
    }
    return Mutate(stmt);
  }
};

}  // namespace arith

namespace ir {

Stmt CanonicalSimplify(Stmt stmt, Map<Var, Range> vrange) {
  return arith::CanonicalStmtSimplifier().CanonicalSimplify(
      stmt, vrange);
}

Expr CanonicalSimplify(Expr expr, Map<Var, Range> vrange) {
  arith::Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }
  return analyzer.canonical_simplify(expr);
}

template<typename T>
T Simplify_(T a, Map<Var, Range> vrange) {
  using namespace HalideIR::Internal;
  Scope<Interval> rscope;
  for (auto kv : vrange) {
    Range r = kv.second;
    rscope.push(
        kv.first.get(),
        Interval(r->min,
                 simplify(r->min + r->extent - make_const(r->min.type(), 1))));
  }
  return HalideIR::Internal::simplify(a, true, rscope);
}


Expr Simplify(Expr a, Map<Var, Range> vrange) {
  // Simplify top level reduce.
  if (const Reduce* r = a.as<Reduce>()) {
    Array<Expr> new_source;
    for (auto& e : r->source) {
      new_source.push_back(Simplify_(e, vrange));
    }
    Expr new_condition = Simplify_(r->condition, vrange);
    if (r->source.same_as(new_source) &&
        r->condition.same_as(new_condition)) {
      return a;
    } else {
      return Reduce::make(
              r->combiner, new_source, r->axis, new_condition, r->value_index);
    }
  }
  return Simplify_(a, vrange);
}

Stmt Simplify(Stmt a, Map<Var, Range> vrange) {
  return Simplify_(a, vrange);
}
}  // namespace ir
}  // namespace tvm
