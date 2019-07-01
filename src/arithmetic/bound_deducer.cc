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
 *  Copyright (c) 2017 by Contributors
 * \file bound_deducer.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/arithmetic.h>
#include <tvm/api_registry.h>

#include <unordered_set>
#include <unordered_map>
#include "int_set.h"

namespace tvm {
namespace arith {

using namespace ir;

// a visitor to find the path to the target variable
// from a expression.
class VariablePathFinder: public IRVisitor {
 public:
  explicit VariablePathFinder(Expr target) : target_(target) {}

  void Visit(const NodeRef& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());

    if (!found_) path_.push_back(node.get());
    if (node.same_as(target_)) found_ = true;
    IRVisitor::Visit(node);
    if (!found_) path_.pop_back();
  }

  std::vector<const Node*> path_;

 private:
  bool found_{false};
  Expr target_;
  std::unordered_set<const Node*> visited_;
};

// get the path to the variable,
// return empty vector to represent failure
std::vector<const Node*> GetPath(Expr target, Expr expr) {
  VariablePathFinder v(target);
  v.Visit(expr);
  return v.path_;
}

class BoundDeduceIntputChecker;

// a visitor to deduce the bound of a variable from a expression
class BoundDeducer: public IRVisitor {
 public:
  friend class BoundDeduceInputChecker;
  friend class Converter;
  BoundDeducer(Expr target, Expr expr,
               const std::unordered_map<const Variable*, IntSet>& hint_map,
               const std::unordered_map<const Variable*, IntSet>& relax_map)
  : target_(target), expr_(expr), hint_map_(hint_map), relax_map_(relax_map) {}

  void Deduce();

  void Visit(const NodeRef& e) final {
    if (!success_) return;
    if (e.get() == path_[iter_++]) {
      IRVisitor::Visit(e);
    } else {
      success_ = false;
      return;
    }
  }

  void Visit_(const LT* op) final {
    LOG(FATAL) << "unable to deduce due to multiple comparison operator";
  }

  void Visit_(const LE* op) final {
    LOG(FATAL) << "unable to deduce due to multiple comparison operator";
  }

  void Visit_(const GT* op) final {
    LOG(FATAL) << "unable to deduce due to multiple comparison operator";
  }

  void Visit_(const GE* op) final {
    LOG(FATAL) << "unable to deduce due to multiple comparison operator";
  }

  void Visit_(const Add* op) final {
    bool left = op->a.get() == path_[iter_];
    result_ -= left ? op->b : op->a;
    Visit(left ? op->a : op->b);
  }

  void Visit_(const Sub* op) final {
    bool left = op->a.get() == path_[iter_];
    if (left) {
      result_ += op->b;
    } else {
      result_ -= op->a;
      result_ = - result_;
      is_greater_ = !is_greater_;
    }
    Visit(left ? op->a : op->b);
  }

  void Visit_(const Mul* op) final {
    bool left = op->a.get() == path_[iter_];
    Expr operand = left ? op->b : op->a;
    Expr target_var = left ? op->a : op->b;

    SignType sign_operand;
    if (operand.type().is_uint()) {
      sign_operand = kPositive;
    } else {
      sign_operand = expr_map_[operand].sign_type();
    }

    if (sign_operand == SignType::kNegative) {
      is_greater_ = !is_greater_;
    } else if (sign_operand == SignType::kUnknown) {
      // unable to get the sign of operand
      success_ = false;
      return;
    }
    // always use relax bound
    bool divided = analyzer_.CanProve(result_ % operand == 0);

    result_ = result_ / operand;

    if (!divided) {
      // Handle non-divisible case
      // NOTE: this accounts for truc div behavior.
      bool target_is_non_neg = expr_map_[target_var].can_prove_non_negative();

      if (is_greater_) {
        result_ += 1;
      } else {
        // NOTE: this is a bit sutble hack.
        //
        // condition:
        // - x * operand <= result
        // - operand > 0
        // - x >= 0
        //
        // Then it is fine to deduce that x <= result / operand.
        // - if result > 0,  this division round down
        // - if result < 0, (result / operand) rounds up and may violate the constraint
        //   however, given that x is always non-negative,
        //   it is fine to have this relaxed bound, given that the user of deduce bound
        //   will respect the bound of x
        //
        // TODO(tvm-team): think about a better API to incorporate constraint of x.
        //                 e.g. specify an interval of x and return a bound
        //                 that is in the interval and satisfies the condition.
        if (target_is_non_neg && sign_operand == kPositive) {
          // do nothing
        } else {
          result_ -= 1;
        }
      }
    }
    Visit(left ? op->a : op->b);
  }

  Expr result_;
  bool is_greater_{true};
  bool success_{true};

 private:
  void Init();
  void Transform();
  void Relax();

  Expr target_;
  Expr expr_;
  const std::unordered_map<const Variable*, IntSet>& hint_map_;
  const std::unordered_map<const Variable*, IntSet>& relax_map_;
  ExprIntSetMap expr_map_;
  std::vector<const Node*> path_;
  size_t iter_{0};
  // internal analzyer
  Analyzer analyzer_;
};

class BoundDeduceInputChecker: public IRVisitor {
 public:
  bool Check(BoundDeducer* deducer) {
    deducer_ = deducer;
    Visit(deducer_->expr_);
    return target_count == 1;
  }

  void Visit(const NodeRef& e) final {
    if (e.same_as(deducer_->target_)) ++target_count;
    IRVisitor::Visit(e);
  }

 private:
  BoundDeducer* deducer_;
  size_t target_count{0};
};

void BoundDeducer::Init() {
  BoundDeduceInputChecker checker;
  if (!checker.Check(this)) success_ = false;
  Transform();
}

void BoundDeducer::Transform() {
  // We will ensure to set expr_ such that it contains target_
  if (const LT* op = expr_.as<LT>()) {
    if (GetPath(target_, op->a).empty()) {
      // a < b -> b >= a + 1
      is_greater_ = true;
      expr_ = op->b;
      result_ = op->a + 1;
    } else {
      // a < b -> a <= b - 1
      is_greater_ = false;
      expr_ = op->a;
      result_ = op->b - 1;
    }
  } else if (const LE* op = expr_.as<LE>()) {
    if (GetPath(target_, op->a).empty()) {
      // a <= b -> b >= a
      is_greater_ = true;
      expr_ = op->b;
      result_ = op->a;
    } else {
      is_greater_ = false;
      expr_ = op->a;
      result_ = op->b;
    }
  } else if (const GT* op = expr_.as<GT>()) {
    if (GetPath(target_, op->a).empty()) {
      // a > b -> b <= a - 1
      is_greater_ = false;
      expr_ = op->b;
      result_ = op->a - 1;
    } else {
      // a > b -> a >= b + 1
      is_greater_ = true;
      expr_ = op->a;
      result_ = op->b + 1;
    }
  } else if (const GE* op = expr_.as<GE>()) {
    if (GetPath(target_, op->a).empty()) {
      // a >= b -> b <= a
      is_greater_ = false;
      expr_ = op->b;
      result_ = op->a;
    } else {
      is_greater_ = true;
      expr_ = op->a;
      result_ = op->b;
    }
  } else {
    success_ = false;
  }
}

void BoundDeducer::Deduce() {
  Init();
  if (!success_) return;
  Relax();
  if (!success_) return;
  // get the path
  path_ = GetPath(target_, expr_);
  if (!path_.size()) {
    success_ = false;
    return;
  }
  expr_map_ = EvalSetForEachSubExpr(expr_, hint_map_);

  Visit(expr_);
}

void BoundDeducer::Relax() {
  IntSet a = EvalSet(expr_, relax_map_);
  IntSet b = EvalSet(result_, relax_map_);
  if (a.is_everything() || b.is_everything()) {
    success_ = false;
    return;
  }
  expr_  = is_greater_ ? a.min() : a.max();
  result_ = is_greater_ ? b.max() : b.min();
}

IntSet DeduceBound(Expr v, Expr e,
  const std::unordered_map<const Variable*, IntSet>& hint_map,
  const std::unordered_map<const Variable*, IntSet>& relax_map) {
  BoundDeducer d(v, e, hint_map, relax_map);
  d.Deduce();
  if (!d.success_) return IntSet::nothing();
  Expr min = neg_inf(), max = pos_inf();
  if (d.is_greater_) {
    min = d.result_;
  } else {
    max = d.result_;
  }
  return IntSet::interval(min, max);
}

// assuming e >= 0, deduce the bound of variable from it.
// return empty set to represent deduce failure.
IntSet DeduceBound(Expr v, Expr e,
                   const Map<Var, IntSet>& hint_map,
                   const Map<Var, IntSet>& relax_map) {
  std::unordered_map<const Variable*, IntSet> hmap;
  for (auto kv : hint_map) {
    hmap[kv.first.get()] = kv.second;
  }
  std::unordered_map<const Variable*, IntSet> rmap;
  for (auto kv : relax_map) {
    rmap[kv.first.get()] = kv.second;
  }
  return DeduceBound(v, e, hmap, rmap);
}

}  // namespace arith
}  // namespace tvm
