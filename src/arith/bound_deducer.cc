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
 * \file bound_deducer.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <unordered_map>
#include <unordered_set>

#include "interval_set.h"

namespace tvm {
namespace arith {

using namespace tir;

// a visitor to find the path to the target variable
// from a expression.
class VariablePathFinder : public ExprVisitor {
 public:
  explicit VariablePathFinder(PrimExpr target) : target_(target) {}

  void VisitExpr(const PrimExpr& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());

    if (!found_) path_.push_back(node.get());
    if (node.same_as(target_)) found_ = true;
    ExprVisitor::VisitExpr(node);
    if (!found_) path_.pop_back();
  }

  std::vector<const Object*> path_;

 private:
  bool found_{false};
  PrimExpr target_;
  std::unordered_set<const Object*> visited_;
};

// get the path to the variable,
// return empty vector to represent failure
std::vector<const Object*> GetPath(PrimExpr target, PrimExpr expr) {
  VariablePathFinder v(target);
  v(expr);
  return v.path_;
}

enum CompareOp { kGreater, kLess, kEqual };

// a visitor to deduce the bound of a variable from a expression
class BoundDeducer : public ExprFunctor<void(const PrimExpr&)> {
 public:
  friend class BoundDeduceInputChecker;
  friend class Converter;
  BoundDeducer(PrimExpr target, PrimExpr expr,
               const std::unordered_map<const VarNode*, IntSet>& hint_map,
               const std::unordered_map<const VarNode*, IntSet>& relax_map)
      : target_(target), expr_(expr), hint_map_(hint_map), relax_map_(relax_map) {}

  void Deduce();

  void VisitExpr(const PrimExpr& e) final {
    if (!success_) return;
    if (iter_ < path_.size() && e.get() == path_[iter_++]) {
      ExprFunctor::VisitExpr(e);
    } else {
      success_ = false;
      return;
    }
  }

  void VisitExprDefault_(const Object* op) final { success_ = false; }

  SignType GetSignType(const PrimExpr& e) {
    if (e.dtype().is_uint()) {
      return kPositive;
    }
    return expr_map_[e].GetSignType();
  }

  void VisitExpr_(const VarNode* op) final {}

  void VisitExpr_(const AddNode* op) final {
    bool left = op->a.get() == path_[iter_];
    result_ -= left ? op->b : op->a;
    this->VisitExpr(left ? op->a : op->b);
  }

  void VisitExpr_(const SubNode* op) final {
    bool left = op->a.get() == path_[iter_];
    if (left) {
      result_ += op->b;
    } else {
      result_ -= op->a;
      result_ = -result_;
      comp_op = ReverseOp(comp_op);
    }
    this->VisitExpr(left ? op->a : op->b);
  }

  void VisitExpr_(const MulNode* op) final {
    bool left = op->a.get() == path_[iter_];
    PrimExpr operand = left ? op->b : op->a;
    PrimExpr target_var = left ? op->a : op->b;

    SignType sign_operand = GetSignType(operand);
    if (sign_operand == SignType::kNegative) {
      comp_op = ReverseOp(comp_op);
    } else if (sign_operand == SignType::kUnknown) {
      // unable to get the sign of operand
      success_ = false;
      return;
    }

    // always use relax bound
    bool divided = analyzer_.CanProve(floormod(result_, operand) == 0);

    result_ = floordiv(result_, operand);  // rounding down here

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
    this->VisitExpr(left ? op->a : op->b);
  }

  void VisitExpr_(const FloorDivNode* op) final {
    if (op->b.get() == path_[iter_]) {
      // Skip cases where the var is divisor.
      success_ = false;
      return;
    }
    PrimExpr divisor = op->b;
    if (analyzer_.CanProveEqual(divisor, 0)) {
      // Skip zero divisor
      success_ = false;
      return;
    }

    SignType sign_operand = GetSignType(divisor);
    if (sign_operand == SignType::kNegative) {
      comp_op = ReverseOp(comp_op);
      divisor = -divisor;
      result_ = -result_;
    } else if (sign_operand == SignType::kUnknown) {
      // unable to get the sign of operand
      success_ = false;
      return;
    }

    if (comp_op == kGreater) {
      // (x // 6 >= 4 --> x >= 4 * 6)
      result_ = result_ * divisor;
    } else if (comp_op == kEqual) {
      // The bound is not single directional
      // (x // 6 == 4 --> 30 > x >= 24)
      // TODO(@wrongtest): support bidirectional bound
      success_ = false;
      return;
    } else {
      // (x // 6 <= 4 --> x <= 4 * 6 + 5)
      result_ = result_ * divisor + divisor - 1;
    }
    if (sign_operand == SignType::kNegative) {
      // (x // -6 >= 4 --> -((x + 6 - 1) // 6) >= 4
      //               --> (x + 6 - 1) // 6 <= -4
      result_ = result_ - divisor + 1;
    }

    this->VisitExpr(op->a);
  }

  PrimExpr result_;
  CompareOp comp_op{kGreater};
  bool success_{true};

 private:
  void Init();
  void Transform();
  void Relax();
  CompareOp ReverseOp(CompareOp comp_op);
  PrimExpr target_;
  PrimExpr expr_;
  const std::unordered_map<const VarNode*, IntSet>& hint_map_;
  const std::unordered_map<const VarNode*, IntSet>& relax_map_;
  ExprIntSetMap expr_map_;
  std::vector<const Object*> path_;
  size_t iter_{0};
  // internal analzyer
  Analyzer analyzer_;
};

class BoundDeduceInputChecker : public ExprVisitor {
 public:
  bool Check(BoundDeducer* deducer) {
    deducer_ = deducer;
    this->VisitExpr(deducer_->expr_);
    return target_count == 1;
  }

  void VisitExpr(const PrimExpr& e) final {
    if (e.same_as(deducer_->target_)) ++target_count;
    ExprVisitor::VisitExpr(e);
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
    case kEqual:
      return kEqual;  // IntSet can not represent range for `NE
    case kGreater:
      return kLess;
    case kLess:
      return kGreater;
    default:
      LOG(FATAL) << "Not a valid compare op";
  }
}

void BoundDeducer::Transform() {
  // We will ensure to set expr_ such that it contains target_
  if (const LTNode* op = expr_.as<LTNode>()) {
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
  } else if (const LENode* op = expr_.as<LENode>()) {
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
  } else if (const GTNode* op = expr_.as<GTNode>()) {
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
  } else if (const GENode* op = expr_.as<GENode>()) {
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
  } else if (const EQNode* op = expr_.as<EQNode>()) {
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

  this->VisitExpr(expr_);

  if (success_) {
    result_ = analyzer_.Simplify(result_);
  }
}

void BoundDeducer::Relax() {
  IntSet a = EvalSet(expr_, relax_map_);
  IntSet b = EvalSet(result_, relax_map_);
  if (a.IsEverything() || b.IsEverything()) {
    success_ = false;
    return;
  }
  // Both LHS and RHS of the EQ should behave as constants e.g.  i == j,
  // can not be resolved when either `i` or `j`  or both are variables with
  // some Range OR `i` and `j` both should be a single point in IntSet
  if (comp_op == kEqual &&
      (!analyzer_.CanProve(b.min() == b.max()) || !analyzer_.CanProve(a.min() == a.max()))) {
    success_ = false;
    return;
  }
  expr_ = (comp_op == kGreater) ? a.min() : a.max();
  result_ = (comp_op == kGreater) ? b.max() : b.min();
}

IntSet DeduceBound(PrimExpr v, PrimExpr e,
                   const std::unordered_map<const VarNode*, IntSet>& hint_map,
                   const std::unordered_map<const VarNode*, IntSet>& relax_map) {
  BoundDeducer d(v, e, hint_map, relax_map);
  d.Deduce();
  if (!d.success_) return IntSet::Nothing();
  PrimExpr min = neg_inf(), max = pos_inf();
  if (d.comp_op == kEqual) {
    min = d.result_;
    max = d.result_;
  } else if (d.comp_op == kGreater) {
    min = d.result_;
  } else {
    max = d.result_;
  }
  return IntSet::Interval(min, max);
}

// assuming e >= 0, deduce the bound of variable from it.
// return empty set to represent deduce failure.
IntSet DeduceBound(PrimExpr v, PrimExpr e, const Map<Var, IntSet>& hint_map,
                   const Map<Var, IntSet>& relax_map) {
  std::unordered_map<const VarNode*, IntSet> hmap;
  for (auto kv : hint_map) {
    hmap[kv.first.get()] = kv.second;
  }
  std::unordered_map<const VarNode*, IntSet> rmap;
  for (auto kv : relax_map) {
    rmap[kv.first.get()] = kv.second;
  }
  return DeduceBound(v, e, hmap, rmap);
}

TVM_REGISTER_GLOBAL("arith.DeduceBound")
    .set_body_typed([](PrimExpr v, PrimExpr cond, const Map<Var, IntSet> hint_map,
                       const Map<Var, IntSet> relax_map) {
      return DeduceBound(v, cond, hint_map, relax_map);
    });

}  // namespace arith
}  // namespace tvm
