/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.cc
 */
#include <tvm/schedule_pass.h>
#include <tvm/ir.h>
#include "./graph.h"

namespace tvm {

namespace ir {

static bool check_index(std::vector<Expr> axis, Array<Expr> index) {
  if (axis.size() != index.size())
    return false;

  for (size_t i = 0; i < axis.size(); ++i) {
    const Variable *v1 = axis[i].as<Variable>();
    const Variable *v2 = index[i].as<Variable>();
    if (!(v1 && v2) || (v1 != v2))
      return false;
  }
  return true;
}

template<typename T>
static bool check_binary_op(const T *n, std::vector<Expr> axis) {
  const Call *ac = n->a.template as<Call>();
  const Call *bc = n->b.template as<Call>();
  if (!(ac && bc))
    return false;
  return (check_index(axis, ac->args) && check_index(axis, bc->args));
}

bool IsEwise(Expr e, std::vector<Expr> axis) {
  if (const Add *n = e.as<Add>()) {
    return check_binary_op(n, axis);
  } else if (const Sub *n = e.as<Sub>()) {
    return check_binary_op(n, axis);
  } else if (const Mul *n = e.as<Mul>()) {
    return check_binary_op(n, axis);
  } else if (const Div *n = e.as<Div>()) {
    return check_binary_op(n, axis);
  } else if (const Mod *n = e.as<Mod>()) {
    return check_binary_op(n, axis);
  } else if (const Min *n = e.as<Min>()) {
    return check_binary_op(n, axis);
  } else if (const Max *n = e.as<Max>()) {
    return check_binary_op(n, axis);
  }
  return false;
}

}  // namespace ir


namespace schedule {

Schedule Fusion(Schedule sch) {
  auto g = schedule::CreateReadGraph(sch->roots);
  Array<Operation> post_order = schedule::PostDFSOrder(sch->roots, g);
  for (Operation op : post_order) {
    if (const ComputeOpNode* compute = op.as<ComputeOpNode>()) {
      std::vector<Expr> axis;
      for (const auto& iter : compute->axis) {
        axis.push_back(iter->var);
      }
      if (ir::IsEwise(compute->body, axis)) {
        bool is_root = false;
        for (auto r : sch->roots) {
          if (r == op) {
            is_root = true;
            break;
          }
        }
        if (!is_root)
          sch[op].compute_inline();
      }
    }
  }
  return sch;
}

}  // namespace schedule
}  // namespace tvm
