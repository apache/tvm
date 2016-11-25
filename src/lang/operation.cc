
/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.cc
 */
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <memory>

namespace tvm {

Tensor Compute(Array<Expr> shape, FCompute fcompute, std::string name) {
  auto node = std::make_shared<TensorNode>();
  auto op_node = std::make_shared<ComputeOpNode>();
  node->name = name;
  node->shape = shape;
  // compute dimension.
  size_t ndim = node->shape.size();
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

  op_node->iter_var = Array<Var>(dim_index);
  op_node->domain = Domain(dom);
  op_node->body = fcompute(op_node->iter_var);
  op_node->name = name;
  node->dtype = op_node->body.type();
  node->source_op = Operation(op_node);
  node->source_index = 0;
  return Tensor(node);
}

Operation ComputeOpNode::make(Domain domain,
                              std::string name,
                              Array<Var> iter_var,
                              Expr body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->domain = domain;
  n->name = name;
  n->iter_var = iter_var;
  n->body = body;
  return Operation(n);
}

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

}  // namespace tvm
