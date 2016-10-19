/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.cc
 */
#include <tvm/tensor.h>
#include <tvm/expr_node.h>
#include <tvm/expr_util.h>
#include <memory>

namespace tvm {

Tensor::Tensor(Array<Expr> shape, std::string name, DataType dtype) {
  auto node = std::make_shared<TensorNode>();
  node->name = std::move(name);
  node->dtype = dtype;
  node->shape = std::move(shape);
  node_ = std::move(node);
}

Tensor::Tensor(Array<Expr> shape, FCompute fcompute, std::string name) {
  auto node = std::make_shared<TensorNode>();
  node->name = std::move(name);
  node->shape = std::move(shape);
  size_t ndim = node->shape.size();
  std::vector<Var> dim_index;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "dim_index" << i;
    dim_index.push_back(Var(os.str()));
  }
  node->dim_index = Array<Var>(dim_index);
  node->source = fcompute(node->dim_index);
  node->dtype = node->source.dtype();
  node_ = std::move(node);
}

Expr Tensor::operator()(Array<Expr> indices) const {
  CHECK_EQ(ndim(), indices.size())
      << "Tensor dimension mismatch in read"
      << "ndim = " << ndim() << ", indices.size=" << indices.size();
  auto node = std::make_shared<TensorReadNode>();
  node->tensor = *this;
  node->indices = std::move(indices);
  return Expr(std::move(node));
}

std::vector<Tensor> Tensor::InputTensors() const {
  const TensorNode* n = static_cast<const TensorNode*>(node_.get());
  std::vector<Tensor> inputs;
  if (n->source.is_null()) return inputs;
  Visit(n->source, [&inputs](const Expr& e) {
      if (e.node_type() == kTensorReadNode) {
        inputs.push_back(e.Get<TensorReadNode>()->tensor);
      }
    });
  return inputs;
}

bool Tensor::IsRTensor() const {
  const TensorNode* n = static_cast<const TensorNode*>(node_.get());
  if (n->source.is_null()) return false;
  return n->source.node_type() == kReduceNode;
}

TVM_REGISTER_NODE_TYPE(TensorNode);

}  // namespace tvm
