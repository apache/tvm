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
 * \file stmt_simplify.cc
 * \brief Statement simplifier based on analyzer
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include <tvm/arithmetic.h>
#include "ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace ir;

class StmtSimplifier : public IRMutatorWithAnalyzer {
 public:
  explicit StmtSimplifier(Analyzer* analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  using Parent = IRMutatorWithAnalyzer;
  using Parent::Mutate;
  using Parent::Mutate_;

  Expr Mutate(Expr expr) final {
    return analyzer_->Simplify(expr);
  }

  Stmt Simplify(Stmt stmt) {
    return Mutate(stmt);
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    analyzer_->Bind(op->loop_var, Range::make_by_min_extent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const LetStmt* op, const Stmt& s) {
    Expr value = this->Mutate(op->value);
    if (!ir::HasSideEffect(value)) {
      // it is fine to discard the let binding
      // because the call to simplify will always inline the var.
      analyzer_->Bind(op->var, value);
      return Mutate(op->body);
    }
    Stmt body = this->Mutate(op->body);
    if (value.same_as(op->value) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return LetStmt::make(op->var, value, body);
    }
  }

  // eliminate useless stores
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if (const Load* load = op->value.as<Load>()) {
      if (load->buffer_var.same_as(op->buffer_var) &&
          Equal(load->index, op->index)) {
        return Evaluate::make(0);
      }
    }
    return stmt;
  }
};

}  // namespace arith

namespace ir {

Stmt CanonicalSimplify(Stmt stmt, Map<Var, Range> vrange) {
  arith::Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }
  return arith::StmtSimplifier(&analyzer).Simplify(stmt);
}

Expr CanonicalSimplify(Expr expr, Map<Var, Range> vrange) {
  arith::Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }
  return analyzer.canonical_simplify(expr);
}

Expr Simplify(Expr expr, Map<Var, Range> vrange) {
  arith::Analyzer analyzer;
  for (auto kv : vrange) {
    analyzer.Bind(kv.first, kv.second);
  }
  expr = analyzer.Simplify(expr);
  return expr;
}

Stmt Simplify(Stmt stmt, Map<Var, Range> vrange) {
  return CanonicalSimplify(stmt, vrange);
}
}  // namespace ir
}  // namespace tvm
