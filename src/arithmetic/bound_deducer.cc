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

namespace tvm {
namespace arith {

using namespace ir;
using HalideIR::Internal::Interval;

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
    if (!success) return;
    if (e.get() == path_[iter_++]) {
      IRVisitor::Visit(e);
    } else {
      success = false;
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
    result -= left ? op->b : op->a;
    Visit(left ? op->a : op->b);
  }

  void Visit_(const Sub* op) final {
    bool left = op->a.get() == path_[iter_];
    if (left) {
      result += op->b;
    } else {
      result -= op->a;
      result = - result;
      is_greater = !is_greater;
    }
    Visit(left ? op->a : op->b);
  }

  void Visit_(const Mul* op) final {
    bool left = op->a.get() == path_[iter_];
    Expr operand = left ? op->b : op->a;

    SignType sign;
    if (operand.type().is_uint()) {
      sign = kPositive;
    } else {
      sign = expr_map_[operand].sign_type();
    }

    if (sign == SignType::kNegative) {
      is_greater = !is_greater;
    } else if (sign == SignType::kUnknown) {
      // unable to get the sign of operand
      success = false;
      return;
    }

    // always use relax bound
    bool divided = can_prove(result % operand == 0);
    result = result / operand;
    // since system will round down when not divided
    // eg. 2/4 -> 0; -2/4 -> -1
    // no need fix for !is_greater:
    // eg. a <= 2/4 -> a <= 0
    // eg. a <= 0/4 -> a <= 0
    // so just fix for not divided and is_greater
    // eg. a >= 2/4 -> a >= 0 + 1
    // eg. a >= 0/4 -> a >= 0
    if (is_greater && !divided) {
       result += 1;
    }

    Visit(left ? op->a : op->b);
  }

  Expr result;
  bool is_greater{true};
  bool success{true};

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
  if (!checker.Check(this)) success = false;
  Transform();
}

void BoundDeducer::Transform() {
  if (const LT* op = expr_.as<LT>()) {
    is_greater = false;
    expr_      = op->a;
    // a < b -> a <= b - 1
    result     = op->b - 1;
  } else if (const LE* op = expr_.as<LE>()) {
    is_greater = false;
    expr_      = op->a;
    result     = op->b;
  } else if (const GT* op = expr_.as<GT>()) {
    is_greater = true;
    expr_      = op->a;
    // a > b -> a >= b + 1
    result     = op->b + 1;
  } else if (const GE* op = expr_.as<GE>()) {
    is_greater = true;
    expr_      = op->a;
    result     = op->b;
  } else {
    success = false;
  }
}

void BoundDeducer::Deduce() {
  Init();
  if (!success) return;
  Relax();
  if (!success) return;
  // get the path
  path_ = GetPath(target_, expr_);
  if (!path_.size()) {
    success = false;
    return;
  }

  expr_map_ = EvalSetForEachSubExpr(expr_, hint_map_);

  Visit(expr_);
}

void BoundDeducer::Relax() {
  IntSet a = EvalSet(expr_, relax_map_);
  IntSet b = EvalSet(result, relax_map_);
  if (a.is_everything() || b.is_everything()) {
    success = false;
    return;
  }
  expr_  = is_greater ? a.min() : a.max();
  result = is_greater ? b.max() : b.min();
}

IntSet DeduceBound(Expr v, Expr e,
  const std::unordered_map<const Variable*, IntSet>& hint_map,
  const std::unordered_map<const Variable*, IntSet>& relax_map) {
  BoundDeducer d(v, e, hint_map, relax_map);
  d.Deduce();
  if (!d.success) return IntSet::nothing();
  Expr min = Interval::neg_inf, max = Interval::pos_inf;
  if (d.is_greater) {
    min = d.result;
  } else {
    max = d.result;
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
