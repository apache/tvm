/*!
 *  Copyright (c) 2017 by Contributors
 * \file bound_deducer.cc
 */
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/api_registry.h>
#include <unordered_set>
#include "./int_set.h"

namespace tvm {
namespace arith {

using namespace ir;
using Halide::Internal::Interval;

// a visitor to find the path to the target variable
// from a expression.
class VariableFinder: public IRVisitor {
 public:
  explicit VariableFinder(Var target) : target_(target) {}

  void Visit(const NodeRef& node) final {
    if (finded_) return;
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());

    path_.push_back(node.get());
    if (node.same_as(target_)) finded_ = true;
    IRVisitor::Visit(node);
    if (!finded_) path_.pop_back();
  }

  std::vector<const Node*> path_;

 private:
  bool finded_{false};
  Var target_;
  std::unordered_set<const Node*> visited_;
};


// get the path to the variable
std::vector<const Node*> GetPath(Var target, Expr expr) {
  VariableFinder v(target);
  v.Visit(expr);
  return v.path_;
}


// a visitor to deduce the bound of a variable from a expression
class BoundDeducer: public IRVisitor {
 public:
  Expr Deduce(Var target, Expr expr) {
    path_ = GetPath(target, expr);
    target_ = target;
    iter_ = 0;
    result = make_zero(expr.type());

    Visit(expr);
    return result;
  }

  void Visit(const NodeRef& e) final {
    if (e.get() == path_[iter_++]) {
      IRVisitor::Visit(e);
    } else {
      LOG(FATAL) << "the current node is not match with the deduced path";
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
    if (is_negative_const(operand)) is_greater = !is_greater;
    result /= operand;
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

 private:
  Var  target_;
  std::vector<const Node*> path_;
  size_t iter_;
};

// Assuming e >= 0, deduce the bound of variable from it.
IntSet DeduceBound(Var v, Expr e) {
    BoundDeducer deducer;
    deducer.Deduce(v, e);
    Type t = deducer.result.type();
    return deducer.is_greater ?
      IntSet::range(Range(deducer.result, Cast::make(t, Interval::pos_inf))) :
      IntSet::range(Range(Cast::make(t, Interval::neg_inf), deducer.result));
}

TVM_REGISTER_API(_pass_DeduceBound)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = DeduceBound(args[0].operator Var(), args[1].operator Expr());
  });


} // namespace arith
} // namespace tvm
