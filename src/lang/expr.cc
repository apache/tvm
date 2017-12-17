/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr.cc
 */
#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <ir/IRPrinter.h>
#include <memory>

namespace tvm {

using HalideIR::IR::RangeNode;

Range::Range(Expr begin, Expr end)
    : Range(std::make_shared<RangeNode>(
          begin,
          is_zero(begin) ? end : (end - begin))) {
}

Range Range::make_by_min_extent(Expr min, Expr extent) {
  return Range(std::make_shared<HalideIR::IR::RangeNode>(min, extent));
}

IterVar IterVarNode::make(Range dom, Var var,
                          IterVarType t, std::string thread_tag) {
  std::shared_ptr<IterVarNode> n = std::make_shared<IterVarNode>();
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  return IterVar(n);
}

IterVar thread_axis(Range dom, std::string tag) {
  return IterVarNode::make(
      dom, Var(tag), kThreadIndex, tag);
}

IterVar reduce_axis(Range dom, std::string name) {
  return IterVarNode::make(
      dom, Var(name), kCommReduce);
}

std::ostream& operator<<(std::ostream& os, const NodeRef& n) {  // NOLINT(*)
  IRPrinter(os).print(n);
  return os;
}

Var var(const std::string& name_hint, Type t) {
  return Var(name_hint, t);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IterVarNode>([](const IterVarNode *op, IRPrinter *p) {
    p->stream << "iter_var(";
    if (op->var->name_hint.length() != 0) {
      p->stream  << op->var->name_hint << ", ";
    }
    if (op->dom.defined()) {
      p->stream << op->dom;
    }
    if (op->thread_tag.length() != 0) {
      p->stream << ", " << op->thread_tag;
    }
    p->stream << ")";
  });

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<RangeNode>([](const HalideIR::IR::RangeNode *op, IRPrinter *p) {
    p->stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
  });


TVM_REGISTER_NODE_TYPE(ArrayNode);
TVM_REGISTER_NODE_TYPE(MapNode);
TVM_REGISTER_NODE_TYPE(RangeNode);
TVM_REGISTER_NODE_TYPE(IterVarNode);

}  // namespace tvm
