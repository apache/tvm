
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
  std::vector<Var> dim_index;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "dim_var" << i;
    dim_index.push_back(Var(os.str()));
  }

  std::vector<Range> dom;
  for (size_t i = 0; i < ndim; ++i) {
    dom.push_back(Range(0, shape[i]));
  }

  op_node->dim_var = Array<Var>(dim_index);
  op_node->domain = Domain(dom);
  op_node->body = fcompute(op_node->dim_var);
  op_node->name = name;

  return Operation(op_node).output(0);
}

Operation ComputeOpNode::make(Domain domain,
                              std::string name,
                              Array<Var> dim_var,
                              Expr body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->domain = domain;
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

std::string ComputeOpNode::output_name(size_t i) const {
  CHECK_EQ(i, 0);
  return name;
}

Type ComputeOpNode::output_dtype(size_t i) const {
  CHECK_EQ(i, 0);
  return body.type();
}

Array<Expr> ComputeOpNode::output_shape(size_t i) const {
  CHECK_EQ(i, 0);
  std::vector<Expr> shape;
  for (size_t i = 0; i < domain.size(); ++i) {
    shape.push_back(domain[i]->extent);
  }
  return Array<Expr>(shape);
}

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

}  // namespace tvm
