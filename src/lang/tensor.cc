/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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

// Tensor

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

SparseFormat OperationNode::output_sformat(size_t i) const {
  CHECK_EQ(i, 0);
  return DeclDenseFormat(this->output_shape(i).size());
}

Tensor Operation::output(size_t i) const {
  auto node = make_node<TensorNode>();
  node->op = *this;
  node->value_index = i;
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  node->sformat = (*this)->output_sformat(i);
  return Tensor(node);
}

Tensor TensorNode::make(Array<Expr> shape,
                        Type dtype,
                        Operation op,
                        int value_index) {
  auto n = make_node<TensorNode>();
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->op = op;
  n->value_index = value_index;
  n->sformat = DeclDenseFormat(n->shape.size());
  return Tensor(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorNode>([](const TensorNode *t, IRPrinter *p) {
    p->stream << "Tensor(shape=" << t->shape
              << ", sformat=" << t->sformat
              << ", op.name=" << t->op->name << ')';
  });

TVM_REGISTER_NODE_TYPE(TensorNode);


// TensorIntrin

TensorIntrin TensorIntrinNode::make(std::string name,
                                    Operation op,
                                    Array<Tensor> inputs,
                                    Array<Buffer> buffers,
                                    Stmt body,
                                    Stmt reduce_init,
                                    Stmt reduce_update) {
  auto n = make_node<TensorIntrinNode>();
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


// TensorIntrinCall

TensorIntrinCall TensorIntrinCallNode::make(TensorIntrin intrin,
                                            Array<Tensor> tensors,
                                            Array<Region> regions,
                                            Array<IterVar> reduce_axis) {
  auto n = make_node<TensorIntrinCallNode>();
  n->intrin = std::move(intrin);
  n->tensors = std::move(tensors);
  n->regions = std::move(regions);
  n->reduce_axis = std::move(reduce_axis);
  return TensorIntrinCall(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<TensorIntrinCallNode>([](const TensorIntrinCallNode *n, IRPrinter *p) {
    p->stream << "TensorIntrinCall(intrin=" << n->intrin << ", " << n << ")";
  });

TVM_REGISTER_NODE_TYPE(TensorIntrinCallNode);

}  // namespace tvm
