/*!
 *  Copyright (c) 2017 by Contributors
 * \file bound_deducer.cc
 */
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>
#include "./int_set.h"
#include "./int_set_internal.h"

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
    if (visited_.count(node.get()) != 0) return;
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

// get the path to the variable
std::vector<const Node*> GetPath(Var target, Expr expr) {
  VariablePathFinder v(target);
  v.Visit(expr);
  return v.success ? v.path_ : std::vector<const Node*>();
}

// a visitor to deduce the bound of a variable from a expression
class BoundDeducer: public IRVisitor {
 public:
  void Deduce(Var target, Expr expr,
              const Map<IterVar, IntSet>& dom_map) {
    target_ = target;
    dom_map_ = dom_map;
    path_ = GetPath(target, expr);
    if (path_.empty()) {
      success = false;
      return;
    }
    iter_ = 0;
    result = make_zero(expr.type());

    Visit(expr);
  }

  void Visit(const NodeRef& e) final {
    if (!success) return;
    if (e.get() == path_[iter_++]) {
      IRVisitor::Visit(e);
    } else {
      success = false;
      return;
    }
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
      result = -1 * result;
      is_greater = !is_greater;
    }
    Visit(left ? op->a : op->b);
  }

  void Visit_(const Mul* op) final {
    bool left = op->a.get() == path_[iter_];
    Expr operand = left ? op->b : op->a;
    SignType sign = EvalSign(operand, dom_map_);
    if (sign == SignType::kNegative) {
      is_greater = !is_greater;
    } else if (sign == SignType::kUnknown) {
      // unable to get the sign of operand
      success = false;
      return;
    }
    // always use relax bound
    if (is_greater) {
      result = result / operand + 1;
    } else {
      result = result / operand - 1;
    }
    Visit(left ? op->a : op->b);
  }

  void Visit_(const Div* op) final {
    bool left = op->a.get() == path_[iter_];
    Expr operand = left ? op->b : op->a;
    if (is_negative_const(operand)) is_greater = !is_greater;
    result = left ? result * operand : operand / result;
    Visit(left ? op->a : op->b);
  }

  Expr result;
  bool is_greater{true};
  bool success{true};

 private:
  Var  target_;
  Map<IterVar, IntSet> dom_map_;
  std::vector<const Node*> path_;
  size_t iter_;
};

// Assuming e >= 0, deduce the bound of variable from it.
IntSet DeduceBound(Var v, Expr e,
                   const Map<IterVar, IntSet>& dom_map) {
    BoundDeducer deducer;
    deducer.Deduce(v, e, dom_map);
    if (!deducer.success) return IntSet();
    return deducer.is_greater ?
      IntervalSet::make(deducer.result, Interval::pos_inf) :
      IntervalSet::make(Interval::neg_inf, deducer.result);
}

} // namespace arith
} // namespace tvm
