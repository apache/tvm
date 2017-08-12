/*!
 *  Copyright (c) 2016 by Contributors
 * \file auto_inline_elem_wise.cc
 */
#include <tvm/schedule_pass.h>
#include <tvm/operation.h>
#include <tvm/ir_visitor.h>

namespace tvm {
namespace schedule {

using namespace ir;

class ElemWiseDetector : public ir::IRVisitor {
 public:
  explicit ElemWiseDetector(Array<IterVar> axis) : axis_(axis) {}

  void Visit(const NodeRef& e) final {
    if (!is_elem_wise_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Call* op) final {
    Array<Expr> axis = op->args;
    if (axis_.size() != axis.size()) {
      is_elem_wise_ = false;
      return;
    }

    for (size_t i = 0; i < axis_.size(); ++i) {
      if (!axis[i].same_as(axis_[i]->var)) {
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
    for (auto& e : compute->body) v.Visit(e);
    return v.is_elem_wise_;
  }
  return false;
}

void AutoInlineElemWise(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && IsElemWise(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

bool IsBroadcast(const Operation& op) {
  if (const ComputeOpNode* compute = op.as<ComputeOpNode>()) {
    if (compute->reduce_axis.size()) {
      return false;
    }
    // TODO(nicolasvasilache): Implement Me
  }
  return false;
}

void AutoInlineBroadcast(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && IsBroadcast(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

bool IsInjective(const Operation& op) {
  if (const ComputeOpNode* compute = op.as<ComputeOpNode>()) {
    return compute->reduce_axis.size() == 0;
  }
  return false;
}

void AutoInlineInjective(Schedule sch) {
  for (Stage s : sch->stages) {
    if (!s.is_scheduled() && IsInjective(s->op) && !s->is_output) {
      s.compute_inline();
    }
  }
}

}  // namespace schedule
}  // namespace tvm
