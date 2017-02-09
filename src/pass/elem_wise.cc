/*!
 *  Copyright (c) 2016 by Contributors
 * \file elem_wise.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

namespace tvm {
namespace ir {

class ElemWiseDetector : public IRVisitor {
 public:
  explicit ElemWiseDetector(Array<IterVar> axis) : axis_(axis) {}

  void Visit(const NodeRef& e) final {
    if (!is_elem_wise_)
        return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Call* op) final {
    Array<Expr> axis = op->args;
    if (axis_.size() != axis.size()) {
      is_elem_wise_ = false;
      return;
    }

    for (size_t i = 0; i < axis_.size(); ++i) {
      const Variable *v1 = axis_[i]->var.as<Variable>();
      const Variable *v2 = axis[i].as<Variable>();
      if (!(v1 && v2) || (v1 != v2)) {
        is_elem_wise_ = false;
        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  bool is_elem_wise_{true};

 private:
  Array<IterVar> axis_;
};


bool IsElemWise(const Operation& op) {
  if (const ComputeOpNode* compute = op.as<ComputeOpNode>()) {
    ElemWiseDetector v = ElemWiseDetector(compute->axis);
    v.Visit(compute->body);
    return v.is_elem_wise_;
  }
  return false;
}

}  // namespace ir
}  // namespace tvm
