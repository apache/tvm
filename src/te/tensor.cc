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
#include <tvm/ffi/function.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>

namespace tvm {
namespace te {

void TensorNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TensorNode>()
      .def_ro("shape", &TensorNode::shape)
      .def_ro("dtype", &TensorNode::dtype)
      .def_ro("op", &TensorNode::op)
      .def_ro("value_index", &TensorNode::value_index);
}

TVM_FFI_STATIC_INIT_BLOCK({ TensorNode::RegisterReflection(); });

IterVar thread_axis(Range dom, std::string tag) {
  return IterVar(dom, Var(tag, dom.defined() ? dom->extent.dtype() : DataType::Int(32)),
                 kThreadIndex, tag);
}

IterVar reduce_axis(Range dom, std::string name) {
  return IterVar(dom, Var(name, dom->extent.dtype()), kCommReduce);
}

Var var(std::string name_hint, DataType t) { return Var(name_hint, t); }

// Tensor
inline PrimExpr Tensor::IndexTensor(Array<PrimExpr> indices, bool support_negative_indices) const {
  Array<PrimExpr> shape = (*this)->shape;

  if (shape.size() != 0) {
    ICHECK_EQ(shape.size(), indices.size())
        << "Tensor dimension mismatch in read "
        << "ndim = " << ndim() << ", indices.size=" << indices.size();
  }

  if (support_negative_indices) {
    for (size_t i = 0; i < shape.size(); i++) {
      PrimExpr new_index =
          Select(indices[i] < make_const(indices[i]->dtype, 0), indices[i] + shape[i], indices[i]);
      indices.Set(i, new_index);
    }
  }
  return ProducerLoad((*this), indices);
}

PrimExpr Tensor::operator()(Array<Var> indices) const {
  Array<PrimExpr> arr(indices.begin(), indices.end());
  return operator()(arr);
}

PrimExpr Tensor::operator()(Array<PrimExpr> indices) const { return IndexTensor(indices, false); }

PrimExpr Tensor::IndexWithNegativeIndices(Array<Var> indices) const {
  Array<PrimExpr> arr(indices.begin(), indices.end());
  return IndexWithNegativeIndices(arr);
}

PrimExpr Tensor::IndexWithNegativeIndices(Array<PrimExpr> indices) const {
  return IndexTensor(indices, true);
}

String TensorNode::GetNameHint() const {
  return op->num_outputs() == 1 ? op->name : (op->name + ".v" + std::to_string(value_index));
}

PrimExpr TensorNode::ToPrimExpr() const { return GetRef<Tensor>(this)(); }

Tensor Operation::output(size_t i) const {
  auto node = make_object<TensorNode>();
  node->op = *this;
  node->value_index = i;
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  return Tensor(node);
}

Tensor::Tensor(Array<PrimExpr> shape, DataType dtype, Operation op, int value_index) {
  auto n = make_object<TensorNode>();
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->op = op;
  n->value_index = value_index;
  data_ = std::move(n);
}

TVM_FFI_REGISTER_GLOBAL("te.Tensor")
    .set_body_typed([](Array<PrimExpr> shape, DataType dtype, Operation op, int value_index) {
      return Tensor(shape, dtype, op, value_index);
    });

TVM_REGISTER_NODE_TYPE(TensorNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* t = static_cast<const TensorNode*>(node.get());
      p->stream << "Tensor(shape=" << t->shape << ", op.name=" << t->op->name << ')';
    });

// Other tensor ops.
TVM_FFI_REGISTER_GLOBAL("te.TensorEqual").set_body_method(&Tensor::operator==);

TVM_FFI_REGISTER_GLOBAL("te.TensorHash").set_body_typed([](Tensor tensor) -> int64_t {
  return static_cast<int64_t>(std::hash<Tensor>()(tensor));
});

TVM_FFI_REGISTER_GLOBAL("te.OpGetOutput").set_body_typed([](Operation op, int64_t output) {
  return op.output(static_cast<size_t>(output));
});

TVM_FFI_REGISTER_GLOBAL("te.OpNumOutputs").set_body_method(&OperationNode::num_outputs);

TVM_FFI_REGISTER_GLOBAL("te.OpInputTensors").set_body_method(&OperationNode::InputTensors);

}  // namespace te
}  // namespace tvm
