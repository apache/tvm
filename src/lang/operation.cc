/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.cc
 */
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/ir.h>
#include <memory>

namespace tvm {

Tensor Operation::output(size_t i) const {
  auto node = std::make_shared<TensorNode>();
  node->op = *this;
  node->value_index = 0;
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  return Tensor(node);
}

// PlaceholderOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<PlaceholderOpNode>([](const PlaceholderOpNode *op, IRPrinter *p) {
    p->stream << "placeholder(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(PlaceholderOpNode);

Array<IterVar> PlaceholderOpNode::root_iter_vars() const {
  return {};
}

Type PlaceholderOpNode::output_dtype(size_t i) const {
  CHECK_EQ(i, 0U);
  return dtype;
}

Array<Expr> PlaceholderOpNode::output_shape(size_t i) const {
  CHECK_EQ(i, 0U);
  return shape;
}

Operation PlaceholderOpNode::make(std::string name,
                                  Array<Expr> shape,
                                  Type dtype) {
  auto n = std::make_shared<PlaceholderOpNode>();
  n->name = name;
  n->shape = shape;
  n->dtype = dtype;
  return Operation(n);
}



Tensor Placeholder(Array<Expr> shape, Type dtype, std::string name) {
  return PlaceholderOpNode::make(name, shape, dtype).output(0);
}

// ComputeOpNode
Array<IterVar> ComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

Type ComputeOpNode::output_dtype(size_t i) const {
  CHECK_EQ(i, 0U);
  return body.type();
}

Array<Expr> ComputeOpNode::output_shape(size_t i) const {
  CHECK_EQ(i, 0U);
  std::vector<Expr> shape;
  for (size_t i = 0; i < axis.size(); ++i) {
    const Range& r = axis[i]->dom;
    shape.push_back(r->extent);
  }
  return Array<Expr>(shape);
}

Tensor Compute(Array<Expr> shape, FCompute fcompute, std::string name) {
  auto op_node = std::make_shared<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(0, shape[i]), os.str()));
    args.push_back(axis.back()->var);
  }

  op_node->axis = Array<IterVar>(axis);
  op_node->body = fcompute(args);
  op_node->name = name;
  return Operation(op_node).output(0);
}

Operation ComputeOpNode::make(std::string name,
                              Array<IterVar> axis,
                              Expr body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->name = name;
  n->axis = axis;
  n->body = body;
  if (n->body->is_type<ir::Reduce>()) {
    n->reduce_axis = n->body.as<ir::Reduce>()->axis;
  }
  return Operation(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ComputeOpNode>([](const ComputeOpNode *op, IRPrinter *p) {
    p->stream << "compute(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

}  // namespace tvm
