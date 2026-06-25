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

#include <vector>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  SortAttrs::RegisterReflection();
  ArgsortAttrs::RegisterReflection();
  TopKAttrs::RegisterReflection();
}

/* relax.sort */

Expr sort(Expr data, int axis, bool descending) {
  auto attrs = ffi::make_object<SortAttrs>();
  attrs->axis = std::move(axis);
  attrs->descending = std::move(descending);

  static const Op& op = Op::Get("relax.sort");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.sort", sort);
}

Type InferTypeSort(const Call& call, const BlockBuilder& ctx) {
  return GetUnaryInputTensorType(call, ctx);
}

TVM_REGISTER_OP("relax.sort")
    .set_attrs_type<SortAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeSort)
    .set_attr<bool>("FPurity", true);

/* relax.argsort */

Expr argsort(Expr data, int axis, bool descending, DLDataType dtype) {
  auto attrs = ffi::make_object<ArgsortAttrs>();
  attrs->axis = std::move(axis);
  attrs->descending = std::move(descending);
  attrs->dtype = std::move(dtype);

  static const Op& op = Op::Get("relax.argsort");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.argsort", argsort);
}

Type InferTypeArgsort(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<ArgsortAttrs>();
  PrimType out_type =
      attrs->dtype == DLDataType{kDLOpaqueHandle, 0, 0} ? data_ty->dtype : PrimType(attrs->dtype);
  if (data_ty->shape.defined()) {
    return TensorType(data_ty->shape.value(), out_type, data_ty->vdevice);
  }
  return TensorType(out_type, data_ty->ndim, data_ty->vdevice);
}

TVM_REGISTER_OP("relax.argsort")
    .set_attrs_type<ArgsortAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeArgsort)
    .set_attr<bool>("FPurity", true);

/* relax.topk */

Expr topk(Expr data, int k, int axis, ffi::String ret_type, bool largest, DLDataType dtype) {
  auto attrs = ffi::make_object<TopKAttrs>();
  attrs->k = std::move(k);
  attrs->axis = std::move(axis);
  attrs->ret_type = std::move(ret_type);
  attrs->largest = std::move(largest);
  attrs->dtype = std::move(dtype);

  static const Op& op = Op::Get("relax.topk");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.topk", topk);
}

Type InferTypeTopK(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* data_shape = data_ty->shape.as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<TopKAttrs>();
  PrimType indices_type =
      attrs->dtype == DLDataType{kDLOpaqueHandle, 0, 0} ? data_ty->dtype : PrimType(attrs->dtype);
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

TVM_REGISTER_OP("relax.topk")
    .set_attrs_type<TopKAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeTopK)
    .set_attr<bool>("FPurity", true);

}  // namespace relax
}  // namespace tvm
