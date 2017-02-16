/*!
 *  Copyright (c) 2017 by Contributors
 * \file bound_deducer.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/api_registry.h>
#include <unordered_set>
#include <unordered_map>
#include "./int_set.h"

namespace tvm {
namespace arith {

using namespace ir;
using Halide::Internal::Interval;

// a visitor to find the path to the target variable
// from a expression.
class VariablePathFinder: public IRVisitor {
 public:
  explicit VariablePathFinder(Var target) : target_(target) {}

  void Visit(const NodeRef& node) final {
    if (!success) return;
    if (visited_.count(node.get()) != 0 &&
        !node.same_as(target_)) {
      return;
    }
    visited_.insert(node.get());

    if (!found_) path_.push_back(node.get());
    if (node.same_as(target_)) {
      if (!found_) {
        found_ = true;
      } else {
        // target variable appears at multiple location
        success = false;
        return;
      }
    }
    IRVisitor::Visit(node);
    if (!found_) path_.pop_back();
  }

  std::vector<const Node*> path_;
  bool success{true};

 private:
  bool found_{false};
  Var target_;
  std::unordered_set<const Node*> visited_;
};

// get the path to the variable,
// return empty vector to represent failure
std::vector<const Node*> GetPath(Var target, Expr expr) {
  VariablePathFinder v(target);
  v.Visit(expr);
  return v.success ? v.path_ : std::vector<const Node*>();
}

class Checker;

// a visitor to deduce the bound of a variable from a expression
class BoundDeducer: public IRVisitor {
 public:
  friend class Checker;
  BoundDeducer(Var target, Expr expr,
               const std::unordered_map<const Variable*, IntSet>& dom_map)
  : target_(target), expr_(expr), dom_map_(dom_map) {}

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
    is_greater = false;
    is_equal = false;
    result = op->b;
    Visit(op->a);
  }

  void Visit_(const LE* op) final {
    is_greater = false;
    is_equal = true;
    result = op->b;
    Visit(op->a);
  }

  void Visit_(const GT* op) final {
    is_greater = true;
    is_equal = false;
    result = op->b;
    Visit(op->a);
  }

  void Visit_(const GE* op) final {
    is_greater = true;
    is_equal = true;
    result = op->b;
    Visit(op->a);
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
    result = result / operand + (is_greater ? 1 : -1);
    Visit(left ? op->a : op->b);
  }

  Expr result;
  bool is_greater{true};
  bool is_equal{true};
  bool success{true};

 private:
  Var  target_;
  Expr expr_;
  const std::unordered_map<const Variable*, IntSet>& dom_map_;
  ExprIntSetMap expr_map_;
  std::vector<const Node*> path_;
  size_t iter_{0};
};

class Checker: public IRVisitor {
 public:
  bool Check(BoundDeducer* deducer) {
    deducer_ = deducer;
    Visit(deducer_->expr_);
    return target_count == 1 && cmp_count == 1;
  }

  void Visit(const NodeRef& e) final {
    if (e.same_as(deducer_->target_)) ++target_count;
    IRVisitor::Visit(e);
  }

  void Visit_(const LT* op) final {
    ++cmp_count;
    Visit(op->a);
    Visit(op->b);
  }

  void Visit_(const LE* op) final {
    ++cmp_count;
    Visit(op->a);
    Visit(op->b);
  }

  void Visit_(const GT* op) final {
    ++cmp_count;
    Visit(op->a);
    Visit(op->b);
  }

  void Visit_(const GE* op) final {
    ++cmp_count;
    Visit(op->a);
    Visit(op->b);
  }

 private:
  BoundDeducer* deducer_;
  size_t target_count{0};
  size_t cmp_count{0};
};

void BoundDeducer::Deduce() {
    result = make_zero(expr_.type());
    // get the path
    path_ = GetPath(target_, expr_);
    Checker checker;
    if (!checker.Check(this) || path_.empty()) {
      success = false;
      return;
    }
    // get the sign of every subexpr
    expr_map_ = EvalSetForEachSubExpr(expr_, dom_map_);

    Visit(expr_);
}

// assuming e >= 0, deduce the bound of variable from it.
// return empty set to represent deduce failure.
IntSet DeduceBound(Var v, Expr e,
                   const Map<Var, IntSet>& dom_map) {
  std::unordered_map<const Variable*, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap[kv.first.get()] = kv.second;
  }
  BoundDeducer d(v, e, dmap);
  d.Deduce();
  if (!d.success) return IntSet();
  Expr min = Interval::neg_inf, max = Interval::pos_inf;
  if (d.is_greater) {
    min = d.is_equal ? d.result : d.result + 1;
  } else {
    max = d.is_equal ? d.result : d.result - 1;
  }
  return IntSet::range(min, max);
}

}  // namespace arith
}  // namespace tvm
