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
 * \file datatype.cc
 * \brief Datatype operators.
 */

#include "datatype.h"

#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  AstypeAttrs::RegisterReflection();
  WrapParamAttrs::RegisterReflection();
}

/* relax.astype */

Expr astype(Expr x, DLDataType dtype) {
  ffi::ObjectPtr<AstypeAttrs> attrs = ffi::make_object<AstypeAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.astype");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.astype", astype);
}

Type InferTypeAstype(const Call& call, const BlockBuilder& ctx) {
  TensorType ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<AstypeAttrs>();
  ffi::ObjectPtr<TensorTypeNode> new_ty = ffi::make_object<TensorTypeNode>(*ty.get());
  new_ty->dtype = PrimType(attrs->dtype);
  return TensorType(new_ty);
}

TVM_REGISTER_OP("relax.astype")
    .set_attrs_type<AstypeAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor")
    .set_attr<FInferType>("FInferType", InferTypeAstype)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

/* relax.wrap_param */

Expr MakeWrapParam(Expr data, DLDataType dtype) {
  ffi::ObjectPtr<WrapParamAttrs> attrs = ffi::make_object<WrapParamAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.wrap_param");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.wrap_param", MakeWrapParam);
}

Type InferTypeWrapParam(const Call& call, const BlockBuilder& ctx) {
  TensorType ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<WrapParamAttrs>();
  ffi::ObjectPtr<TensorTypeNode> new_ty = ffi::make_object<TensorTypeNode>(*ty.get());
  new_ty->dtype = PrimType(attrs->dtype);
  return TensorType(new_ty);
}

TVM_REGISTER_OP("relax.wrap_param")
    .set_attrs_type<WrapParamAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferType>("FInferType", InferTypeWrapParam)
    .set_attr<bool>("FPurity", true);

}  // namespace relax
}  // namespace tvm
