/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.cc
 */
#include <tvm/tensor.h>
#include <tvm/operation.h>
#include <tvm/tensor_intrin.h>
#include <ir/IR.h>
#include <memory>

namespace tvm {

Expr Tensor::operator()(Array<Var> indices) const {
  Array<Expr> arr(indices.begin(), indices.end());
  return operator()(arr);
}

Expr Tensor::operator()(Array<Expr> indices) const {
  using HalideIR::Internal::Call;
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
  n->shape = std::move(shape);
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

TensorIntrin TensorIntrinNode::make(std::string name,
                                    Operation op,
                                    Array<Tensor> inputs,
                                    Array<Buffer> buffers,
                                    Stmt body,
                                    Stmt reduce_init,
                                    Stmt reduce_update) {
  auto n = std::make_shared<TensorIntrinNode>();
  n->name = std::move(name);
  n->op = std::move(op);
  n->inputs = std::move(inputs);
  n->buffers = std::move(buffers);
  n->body = std::move(body);
  n->reduce_init = std::move(reduce_init);
  n->reduce_update = std::move(reduce_update);
  return TensorIntrin(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorIntrinNode>([](const TensorIntrinNode *n, IRPrinter *p) {
    p->stream << "TensorIntrin(name=" << n->name << ", " << n << ")";
  });

TVM_REGISTER_NODE_TYPE(TensorIntrinNode);
}  // namespace tvm
