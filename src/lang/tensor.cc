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
  node->dim_var = Array<Var>(dim_index);
  node->source = fcompute(node->dim_var);
  node->dtype = node->source.type();
  node_ = std::move(node);
}

Expr Tensor::operator()(Array<Expr> indices) const {
  using Halide::Internal::Call;
  CHECK_EQ(ndim(), indices.size())
      << "Tensor dimension mismatch in read"
      << "ndim = " << ndim() << ", indices.size=" << indices.size();
  return Call::make(
      (*this)->dtype, (*this)->name, indices, Call::Halide, *this);
}

TVM_REGISTER_NODE_TYPE(TensorNode);

}  // namespace tvm
