/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.cc
 */
#include <tvm/tensor.h>
#include <tvm/operation.h>
#include <ir/IR.h>
#include <memory>

namespace tvm {

Expr Tensor::operator()(Array<Expr> indices) const {
  using Halide::Internal::Call;
  CHECK_EQ(ndim(), indices.size())
      << "Tensor dimension mismatch in read"
      << "ndim = " << ndim() << ", indices.size=" << indices.size();
  auto n = Call::make(
      (*this)->dtype, (*this)->op->name, indices, Call::Halide,
      (*this)->op, (*this)->value_index);
  return n;
}

Tensor TensorNode::make(Array<Expr> shape,
                        Type dtype,
                        Operation op,
                        int value_index) {
  auto n = std::make_shared<TensorNode>();
  n->shape = shape;
  n->dtype = dtype;
  n->op = op;
  n->value_index = value_index;
  return Tensor(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorNode>([](const TensorNode *t, IRPrinter *p) {
    p->stream << "Tensor(shape=" << t->shape
              << ", op.name=" << t->op->name << ')';
  });

TVM_REGISTER_NODE_TYPE(TensorNode);

Tensor Operation::output(size_t i) const {
  auto node = std::make_shared<TensorNode>();
  node->op = *this;
  node->value_index = i;
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  return Tensor(node);
}

}  // namespace tvm
