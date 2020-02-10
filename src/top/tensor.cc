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
 * \file tensor.cc
 */
#include <tvm/top/tensor.h>
#include <tvm/top/operation.h>
#include <tvm/top/tensor_intrin.h>
#include <memory>

namespace tvm {
namespace top {
// Tensor
PrimExpr Tensor::operator()(Array<Var> indices) const {
  Array<PrimExpr> arr(indices.begin(), indices.end());
  return operator()(arr);
}

PrimExpr Tensor::operator()(Array<PrimExpr> indices) const {
  using ir::CallNode;
  if (ndim() != 0) {
    CHECK_EQ(ndim(), indices.size())
        << "Tensor dimension mismatch in read"
        << "ndim = " << ndim() << ", indices.size=" << indices.size();
  }
  auto n = CallNode::make(
      (*this)->dtype, (*this)->op->name, indices, CallNode::Halide,
      (*this)->op, (*this)->value_index);
  return n;
}

Tensor Operation::output(size_t i) const {
  auto node = make_object<TensorNode>();
  node->op = *this;
  node->value_index = i;
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  return Tensor(node);
}

Tensor TensorNode::make(Array<PrimExpr> shape,
                        DataType dtype,
                        Operation op,
                        int value_index) {
  auto n = make_object<TensorNode>();
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->op = op;
  n->value_index = value_index;
  return Tensor(n);
}

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<TensorNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* t = static_cast<const TensorNode*>(node.get());
    p->stream << "Tensor(shape=" << t->shape
              << ", op.name=" << t->op->name << ')';
  });

TVM_REGISTER_NODE_TYPE(TensorNode);


// TensorIntrin

TensorIntrin TensorIntrinNode::make(std::string name,
                                    Operation op,
                                    Array<Tensor> inputs,
                                    Array<Buffer> buffers,
                                    Array<Var> scalar_params,
                                    Stmt body,
                                    Stmt reduce_init,
                                    Stmt reduce_update) {
  auto n = make_object<TensorIntrinNode>();
  n->name = std::move(name);
  n->op = std::move(op);
  n->inputs = std::move(inputs);
  n->buffers = std::move(buffers);
  n->scalar_params = std::move(scalar_params);
  n->body = std::move(body);
  n->reduce_init = std::move(reduce_init);
  n->reduce_update = std::move(reduce_update);
  return TensorIntrin(n);
}

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<TensorIntrinNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* op = static_cast<const TensorIntrinNode*>(node.get());
    p->stream << "TensorIntrin(name=" << op->name << ", " << op << ")";
  });

TVM_REGISTER_NODE_TYPE(TensorIntrinNode);


// TensorIntrinCall

TensorIntrinCall TensorIntrinCallNode::make(TensorIntrin intrin,
                                            Array<Tensor> tensors,
                                            Array<Region> regions,
                                            Array<IterVar> reduce_axis,
                                            Array<PrimExpr> scalar_inputs) {
  auto n = make_object<TensorIntrinCallNode>();
  n->intrin = std::move(intrin);
  n->tensors = std::move(tensors);
  n->regions = std::move(regions);
  n->reduce_axis = std::move(reduce_axis);
  n->scalar_inputs = std::move(scalar_inputs);
  return TensorIntrinCall(n);
}

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<TensorIntrinCallNode>([](const ObjectRef& node, NodePrinter* p) {
    auto* n = static_cast<const TensorIntrinCallNode*>(node.get());
    p->stream << "TensorIntrinCall(intrin=" << n->intrin << ", " << n << ")";
  });

TVM_REGISTER_NODE_TYPE(TensorIntrinCallNode);

}  // namespace top
}  // namespace tvm
