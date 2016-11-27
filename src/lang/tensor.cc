/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.cc
 */
#include <tvm/tensor.h>
#include <ir/IR.h>
#include <memory>

namespace tvm {

Tensor::Tensor(Array<Expr> shape, std::string name, Type dtype) {
  auto node = std::make_shared<TensorNode>();
  node->name = std::move(name);
  node->dtype = dtype;
  node->shape = std::move(shape);
  node_ = std::move(node);
}

Expr Tensor::operator()(Array<Expr> indices) const {
  using Halide::Internal::Call;
  CHECK_EQ(ndim(), indices.size())
      << "Tensor dimension mismatch in read"
      << "ndim = " << ndim() << ", indices.size=" << indices.size();
  auto n = Call::make(
      (*this)->dtype, (*this)->name, indices, Call::Halide, *this);
  return n;
}


Tensor TensorNode::make(Array<Expr> shape,
                        std::string name,
                        Type dtype,
                        Operation source_op,
                        int source_index) {
  auto n = std::make_shared<TensorNode>();
  n->shape = shape;
  n->name = name;
  n->dtype = dtype;
  n->source_op = source_op;
  n->source_index = source_index;
  return Tensor(n);
}

TVM_REGISTER_NODE_TYPE(TensorNode);

}  // namespace tvm
