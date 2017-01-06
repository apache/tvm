
/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.cc
 */
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <memory>

namespace tvm {

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ComputeOpNode>([](const ComputeOpNode *op, IRPrinter *p) {
    p->stream << "op(" << op << ")";
});

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
  return Operation(n);
}

Tensor Operation::output(size_t i) const {
  auto node = std::make_shared<TensorNode>();
  node->op = *this;
  node->value_index = 0;
  node->name =  (*this)->output_name(i);
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  return Tensor(node);
}

Array<IterVar> ComputeOpNode::root_iter_vars() const {
  return axis;
}

std::string ComputeOpNode::output_name(size_t i) const {
  CHECK_EQ(i, 0U);
  return name;
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

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

}  // namespace tvm
