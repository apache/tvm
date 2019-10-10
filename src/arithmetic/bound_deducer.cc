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

enum CompareOp {kGreater, kLess, kEqual};

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
      comp_op = ReverseOp(comp_op);
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
      comp_op = ReverseOp(comp_op);
    } else if (sign_operand == SignType::kUnknown) {
      // unable to get the sign of operand
      success_ = false;
      return;
    }

    // always use relax bound
    bool divided = analyzer_.CanProve(floormod(result_, operand) == 0);

    result_ = floordiv(result_, operand);   // rounding down here

    if (!divided) {
      if (comp_op == kGreater) {
        // System will round down in all the cases, so add one for result_ for kGreater
        // (x >= 3/2 --> x >= 2)
        // (x >= -3/2 --> x >= -1)
        // (x >= 3/-2 --> x >= -1)
        // (x >= -3/-2 --> x >= 2)
        result_ += 1;
      } else if (comp_op == kEqual) {
        // condition unsatisfiable as with floor div, it will change the expression
        success_ = false;
        return;
      } else {
        // System rounds down in all cases, do nothing for kLess.
        // ( x <= 3/2 --> x <= 1)
        // ( x <= -3/2 --> x <= -2)
        // ( x <= 3/-2 --> x <= -2)
        // ( x <= -3/-2 --> x <= 1)
      }
    }
    Visit(left ? op->a : op->b);
  }

  Expr result_;
  CompareOp comp_op{kGreater};
  bool success_{true};

 private:
  void Init();
  void Transform();
  void Relax();
  CompareOp ReverseOp(CompareOp comp_op);
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

CompareOp BoundDeducer::ReverseOp(CompareOp comp_op) {
  switch (comp_op) {
    case kEqual: return kEqual;   // IntSet can not represent range for `NE
    case kGreater: return kLess;
    case kLess: return kGreater;
    default:
      LOG(FATAL) << "Not a valid compare op";
      return kGreater;  // return some default value
  }
}

void BoundDeducer::Transform() {
  // We will ensure to set expr_ such that it contains target_
  if (const LT* op = expr_.as<LT>()) {
    if (GetPath(target_, op->a).empty()) {
      // a < b -> b >= a + 1
      comp_op = kGreater;
      expr_ = op->b;
      result_ = op->a + 1;
    } else {
      // a < b -> a <= b - 1
      comp_op = kLess;
      expr_ = op->a;
      result_ = op->b - 1;
    }
  } else if (const LE* op = expr_.as<LE>()) {
    if (GetPath(target_, op->a).empty()) {
      // a <= b -> b >= a
      comp_op = kGreater;
      expr_ = op->b;
      result_ = op->a;
    } else {
      comp_op = kLess;
      expr_ = op->a;
      result_ = op->b;
    }
  } else if (const GT* op = expr_.as<GT>()) {
    if (GetPath(target_, op->a).empty()) {
      // a > b -> b <= a - 1
      comp_op = kLess;
      expr_ = op->b;
      result_ = op->a - 1;
    } else {
      // a > b -> a >= b + 1
      comp_op = kGreater;
      expr_ = op->a;
      result_ = op->b + 1;
    }
  } else if (const GE* op = expr_.as<GE>()) {
    if (GetPath(target_, op->a).empty()) {
      // a >= b -> b <= a
      comp_op = kLess;
      expr_ = op->b;
      result_ = op->a;
    } else {
      comp_op = kGreater;
      expr_ = op->a;
      result_ = op->b;
    }
  } else if (const EQ* op = expr_.as<EQ>()) {
    comp_op = kEqual;
    if (GetPath(target_, op->a).empty()) {
      // if the b == a -> a == b
      expr_ = op->b;
      result_ = op->a;
    } else {
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
  // Both LHS and RHS of the EQ should behave as constants e.g.  i == j,
  // can not be resolved when either `i` or `j`  or both are variables with
  // some Range OR `i` and `j` both should be a single point in IntSet
  if (comp_op == kEqual && (!analyzer_.CanProve(b.min() == b.max())
     || !analyzer_.CanProve(a.min() == a.max()))) {
    success_ = false;
    return;
  }
  expr_  = (comp_op == kGreater) ? a.min() : a.max();
  result_ = (comp_op == kGreater) ? b.max() : b.min();
}

IntSet DeduceBound(Expr v, Expr e,
  const std::unordered_map<const Variable*, IntSet>& hint_map,
  const std::unordered_map<const Variable*, IntSet>& relax_map) {
  BoundDeducer d(v, e, hint_map, relax_map);
  d.Deduce();
  if (!d.success_) return IntSet::nothing();
  Expr min = neg_inf(), max = pos_inf();
  if (d.comp_op == kEqual) {
    min = d.result_;
    max = d.result_;
  } else if (d.comp_op == kGreater) {
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
