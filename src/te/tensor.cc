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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
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

TVM_FFI_STATIC_INIT_BLOCK() { TensorNode::RegisterReflection(); }

IterVar thread_axis(Range dom, std::string tag) {
  return IterVar(dom, PrimVar(tag, dom.defined() ? dom->extent.ty() : PrimType::Int(32)),
                 kThreadIndex, tag);
}

IterVar reduce_axis(Range dom, std::string name) {
  return IterVar(dom, PrimVar(name, dom->extent.ty()), kCommReduce);
}

PrimVar var(std::string name_hint, PrimType t) { return PrimVar(name_hint, t); }

// Tensor
inline PrimExpr Tensor::IndexTensor(ffi::Array<PrimExpr> indices,
                                    bool support_negative_indices) const {
  ffi::Array<PrimExpr> shape = (*this)->shape;

  TVM_FFI_ICHECK_EQ(shape.size(), indices.size())
      << "Tensor dimension mismatch in read "
      << "ndim = " << ndim() << ", indices.size=" << indices.size();

  if (support_negative_indices) {
    for (size_t i = 0; i < shape.size(); i++) {
      PrimExpr new_index =
          Select(indices[i] < IntImm(indices[i].ty(), 0), indices[i] + shape[i], indices[i]);
      indices.Set(i, new_index);
    }
  }
  ffi::Array<Expr> args;
  args.reserve(indices.size());
  for (const PrimExpr& index : indices) {
    args.push_back(index);
  }
  return PrimExpr(Call((*this)->dtype, *this, args));
}

PrimExpr Tensor::operator()(ffi::Array<PrimVar> indices) const {
  ffi::Array<PrimExpr> arr =
      indices.Map([](const PrimVar& var) { return static_cast<PrimExpr>(var); });
  return operator()(arr);
}

PrimExpr Tensor::operator()(ffi::Array<PrimExpr> indices) const {
  return IndexTensor(indices, false);
}

PrimExpr Tensor::IndexWithNegativeIndices(ffi::Array<PrimVar> indices) const {
  ffi::Array<PrimExpr> arr =
      indices.Map([](const PrimVar& var) { return static_cast<PrimExpr>(var); });
  return IndexWithNegativeIndices(arr);
}

PrimExpr Tensor::IndexWithNegativeIndices(ffi::Array<PrimExpr> indices) const {
  return IndexTensor(indices, true);
}

ffi::String TensorNode::GetNameHint() const {
  return op->num_outputs() == 1 ? op->name : (op->name + ".v" + std::to_string(value_index));
}

PrimExpr TensorNode::ToPrimExpr() const { return ffi::GetRef<Tensor>(this)(); }

Tensor Operation::output(size_t i) const {
  return Tensor((*this)->output_shape(i), (*this)->output_dtype(i), *this, static_cast<int>(i));
}

Tensor::Tensor(ffi::Array<PrimExpr> shape, PrimType dtype, Operation op, int value_index) {
  auto n = ffi::make_object<TensorNode>();
  n->ExprNode::ty = OpaqueType();
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->op = op;
  n->value_index = value_index;
  data_ = std::move(n);
}

bool IsTensorLoad(const Expr& expr) {
  const auto* call = expr.as<CallNode>();
  return call != nullptr && call->op.as<TensorNode>() != nullptr;
}

namespace {

ffi::Array<PrimExpr> ValidateTensorLoad(const Call& call, Tensor* tensor_out) {
  const auto* tensor_node = call->op.as<TensorNode>();
  TVM_FFI_ICHECK(tensor_node != nullptr) << "Expected a Call whose callee is a TE Tensor";
  Tensor tensor = ffi::GetRef<Tensor>(tensor_node);
  TVM_FFI_ICHECK_EQ(call->args.size(), tensor->shape.size())
      << "Tensor-load index count must match tensor rank";
  TVM_FFI_ICHECK(call->ty.as<PrimTypeNode>() != nullptr && call->ty == tensor->dtype)
      << "Tensor-load result type must match the tensor element type";

  ffi::Array<PrimExpr> indices;
  indices.reserve(call->args.size());
  for (const Expr& arg : call->args) {
    auto index = arg.as<PrimExpr>();
    TVM_FFI_ICHECK(index.has_value()) << "Tensor-load indices must have primitive type";
    indices.push_back(index.value());
  }
  if (tensor_out != nullptr) {
    *tensor_out = std::move(tensor);
  }
  return indices;
}

}  // namespace

Tensor GetTensorFromLoad(const Call& call) {
  Tensor tensor;
  ValidateTensorLoad(call, &tensor);
  return tensor;
}

ffi::Array<PrimExpr> GetTensorLoadIndices(const Call& call) {
  return ValidateTensorLoad(call, nullptr);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "te.Tensor", [](ffi::Array<PrimExpr> shape, PrimType dtype, Operation op, int value_index) {
        return Tensor(shape, dtype, op, value_index);
      });
}

// Pattern A (RM): auto-default repr from reflection.

// Other tensor ops.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("te.TensorEqual", &Tensor::operator==)
      .def("te.TensorDType", [](Tensor tensor) -> PrimType { return tensor->dtype; })
      .def("te.TensorLoad",
           [](Tensor tensor, ffi::Array<PrimExpr> indices) { return tensor(indices); })
      .def("te.TensorHash",
           [](Tensor tensor) -> int64_t {
             return static_cast<int64_t>(std::hash<Tensor>()(tensor));
           })
      .def("te.OpGetOutput",
           [](Operation op, int64_t output) { return op.output(static_cast<size_t>(output)); })
      .def_method("te.OpNumOutputs", &OperationNode::num_outputs)
      .def_method("te.OpInputTensors", &OperationNode::InputTensors);
}

}  // namespace te
}  // namespace tvm
