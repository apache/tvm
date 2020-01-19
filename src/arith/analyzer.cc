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
 * \file tvm/arith/analyzer.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

Analyzer::Analyzer()
    : const_int_bound(this),
      modular_set(this),
      rewrite_simplify(this),
      canonical_simplify(this),
      int_set(this) {
}

void Analyzer::Bind(const Var& var, const PrimExpr& expr) {
  PrimExpr new_expr = expr;
  new_expr = this->canonical_simplify(new_expr);
  new_expr = this->rewrite_simplify(new_expr);

  this->const_int_bound.Update(var, this->const_int_bound(new_expr));
  this->modular_set.Update(var, this->modular_set(new_expr));
  this->rewrite_simplify.Update(var, new_expr);
  this->canonical_simplify.Update(var, new_expr);
}

void Analyzer::Bind(const Var& var, const Range& range) {
  CHECK(range.defined());
  if (tir::is_one(range->extent)) {
    this->Bind(var, range->min);
  } else {
    this->const_int_bound.Bind(var, range);
  }
  // skip modular_set
  // skip rewrite simplify
}


void ConstraintContext::EnterWithScope() {
  CHECK(exit_ == nullptr);
  // entering the scope.
  auto f0 = analyzer_->const_int_bound.EnterConstraint(constraint_);
  auto f1 = analyzer_->modular_set.EnterConstraint(constraint_);
  auto f2 = analyzer_->rewrite_simplify.EnterConstraint(constraint_);
  // recovery function.
  exit_ = [f0, f1, f2]() {
    if (f2 != nullptr) f2();
    if (f1 != nullptr) f1();
    if (f0 != nullptr) f0();
  };
}

void ConstraintContext::ExitWithScope() {
  CHECK(exit_ != nullptr);
  exit_();
}

bool Analyzer::CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound) {
  if (const auto* ptr = expr.as<tir::IntImmNode>()) {
    return ptr->value >= lower_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->min_value >= lower_bound) return true;
  return false;
}

bool Analyzer::CanProve(const PrimExpr& expr) {
  if (const auto* ptr = expr.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  auto res = this->rewrite_simplify(expr);
  if (const auto* ptr = res.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  res = this->canonical_simplify(expr);
  if (const auto* ptr = res.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  return false;
}

PrimExpr Analyzer::Simplify(const PrimExpr& expr) {
  if (tir::is_const(expr)) return expr;
  auto res = this->rewrite_simplify(expr);
  if (tir::is_const(res)) return res;
  res = this->canonical_simplify(res);
  return res;
}

}  // namespace arith
}  // namespace tvm
