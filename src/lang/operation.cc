
/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.cc
 */
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <memory>

namespace tvm {

Tensor Compute(Array<Expr> shape, FCompute fcompute, std::string name) {
  auto op_node = std::make_shared<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> dim_var;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "dim_var" << i;
    dim_var.push_back(IterVar(Range(0, shape[i]), os.str()));
    args.push_back(dim_var.back()->var);
  }

  op_node->dim_var = Array<IterVar>(dim_var);
  op_node->body = fcompute(args);
  op_node->name = name;
  return Operation(op_node).output(0);
}

Operation ComputeOpNode::make(std::string name,
                              Array<IterVar> dim_var,
                              Expr body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->name = name;
  n->dim_var = dim_var;
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
  return dim_var;
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
  for (size_t i = 0; i < dim_var.size(); ++i) {
    const Range& r = dim_var[i]->dom;
    shape.push_back(r->extent);
  }
  return Array<Expr>(shape);
}

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

}  // namespace tvm
