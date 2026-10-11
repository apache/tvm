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
 * \file sorting.cc
 * \brief sorting operators.
 */

#include "sorting.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr_types.h>

#include <vector>

namespace tvm {
namespace relax {

void SortAttrsNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SortAttrsNode>()
      .def_ro("axis", &SortAttrsNode::axis,
              "Axis along which the sort is computed."
              "The default the last axis is used.",
              refl::DefaultValue(-1))
      .def_ro("descending", &SortAttrsNode::descending,
              "Whether to sort in descending order."
              "If it is not specified, it defaults to the ascending order.",
              refl::DefaultValue(false));
}

void ArgsortAttrsNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ArgsortAttrsNode>()
      .def_ro("axis", &ArgsortAttrsNode::axis,
              "Axis along which the argsort is computed."
              "The default the last axis is used.",
              refl::DefaultValue(-1))
      .def_ro("descending", &ArgsortAttrsNode::descending,
              "Whether to argsort in descending order."
              "If it is not specified, it defaults to the ascending order.",
              refl::DefaultValue(false))
      .def_ro("dtype", &ArgsortAttrsNode::dtype, "DType of the output indices.",
              refl::DefaultValue((DLDataType{kDLInt, 32, 1})));
}

void TopKAttrsNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TopKAttrsNode>()
      .def_ro("k", &TopKAttrsNode::k, "Number of top elements to select", refl::DefaultValue(1))
      .def_ro("axis", &TopKAttrsNode::axis, "Axis along which to sort the input tensor.",
              refl::DefaultValue(-1))
      .def_ro("ret_type", &TopKAttrsNode::ret_type,
              "The return type [both, values, indices]."
              "both - return both top k data and indices."
              "values - return top k data only."
              "indices - return top k indices only.",
              refl::DefaultValue("both"))
      .def_ro("largest", &TopKAttrsNode::largest,
              "Whether to return largest or smallest elements."
              "By default, return the largest k elements.",
              refl::DefaultValue(true))
      .def_ro("dtype", &TopKAttrsNode::dtype, "Data type of the output indices.",
              refl::DefaultValue((DLDataType{kDLInt, 32, 1})));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  SortAttrsNode::RegisterReflection();
  ArgsortAttrsNode::RegisterReflection();
  TopKAttrsNode::RegisterReflection();
}

/* relax.sort */

Expr sort(Expr data, int axis, bool descending) {
  auto attrs = ffi::make_object<SortAttrsNode>();
  attrs->axis = std::move(axis);
  attrs->descending = std::move(descending);

  static const Op op = Op::Get("relax.sort");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.sort", sort);
}

Type InferTypeSort(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  return GetUnaryInputTensorType(call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.sort")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<SortAttrsNode>())
      .set_attr<FInferType>(tvm::op_attr::kInferType, FInferType::FromNative<&InferTypeSort>())
      .set_attr<bool>(tvm::relax::op_attr::kPurity, true);
}

/* relax.argsort */

Expr argsort(Expr data, int axis, bool descending, ffi::Optional<DLDataType> dtype) {
  auto attrs = ffi::make_object<ArgsortAttrsNode>();
  attrs->axis = std::move(axis);
  attrs->descending = std::move(descending);
  attrs->dtype = std::move(dtype);

  static const Op op = Op::Get("relax.argsort");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.argsort", argsort);
}

Type InferTypeArgsort(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<ArgsortAttrsNode>();
  ffi::Optional<PrimType> out_type = attrs->dtype.has_value()
                                         ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                                         : data_ty->dtype;
  if (data_ty->shape.has_value()) {
    return TensorType(data_ty->shape.value(), out_type, data_ty->vdevice);
  }
  return TensorType(out_type, data_ty->ndim, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.argsort")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<ArgsortAttrsNode>())
      .set_attr<FInferType>(tvm::op_attr::kInferType, FInferType::FromNative<&InferTypeArgsort>())
      .set_attr<bool>(tvm::relax::op_attr::kPurity, true);
}

/* relax.topk */

Expr topk(Expr data, int k, int axis, ffi::String ret_type, bool largest,
          ffi::Optional<DLDataType> dtype) {
  auto attrs = ffi::make_object<TopKAttrsNode>();
  attrs->k = std::move(k);
  attrs->axis = std::move(axis);
  attrs->ret_type = std::move(ret_type);
  attrs->largest = std::move(largest);
  attrs->dtype = std::move(dtype);

  static const Op op = Op::Get("relax.topk");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.topk", topk);
}

Type InferTypeTopK(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* data_shape = data_ty->shape.as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<TopKAttrsNode>();
  ffi::Optional<PrimType> indices_type =
      attrs->dtype.has_value() ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                               : data_ty->dtype;
  int ndim = data_ty->ndim;
  int k = attrs->k;
  ffi::String ret_type = attrs->ret_type;
  int axis = attrs->axis;
  if (axis < 0 && ndim > 0) {
    axis += ndim;
  }

  std::vector<Type> output_tys;
  output_tys.reserve(2);
  if (data_shape == nullptr) {
    output_tys.push_back(TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice));
    output_tys.push_back(TensorType(indices_type, data_ty->ndim, data_ty->vdevice));
  } else {
    ffi::Array<PrimExpr> out_shape = data_shape->values;
    const auto* int_dim = out_shape[axis].as<IntImmNode>();
    if (k > 0 && (int_dim == nullptr || k < int_dim->value)) {
      out_shape.Set(axis, k);
    }
    output_tys.push_back(TensorType(ShapeExpr(out_shape), data_ty->dtype, data_ty->vdevice));
    output_tys.push_back(TensorType(ShapeExpr(out_shape), indices_type, data_ty->vdevice));
  }

  if (ret_type == "both") {
    return TupleType(output_tys);
  } else if (ret_type == "values") {
    return output_tys[0];
  } else if (ret_type == "indices") {
    return output_tys[1];
  }
  TVM_FFI_THROW(InternalError) << "Unsupported ret type: " << ret_type;
  TVM_FFI_UNREACHABLE();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.topk")
      .signature(sig::arg("data", "The input tensor."), sig::call_attrs<TopKAttrsNode>())
      .set_attr<FInferType>(tvm::op_attr::kInferType, FInferType::FromNative<&InferTypeTopK>())
      .set_attr<bool>(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
