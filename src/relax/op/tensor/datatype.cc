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

TVM_FFI_STATIC_INIT_BLOCK({
  AstypeAttrs::RegisterReflection();
  WrapParamAttrs::RegisterReflection();
});

/* relax.astype */

Expr astype(Expr x, DataType dtype) {
  ObjectPtr<AstypeAttrs> attrs = ffi::make_object<AstypeAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.astype");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.astype", astype);
});

StructInfo InferStructInfoAstype(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<AstypeAttrs>();
  ObjectPtr<TensorStructInfoNode> new_sinfo = ffi::make_object<TensorStructInfoNode>(*sinfo.get());
  new_sinfo->dtype = attrs->dtype;
  return TensorStructInfo(new_sinfo);
}

TVM_REGISTER_OP("relax.astype")
    .set_attrs_type<AstypeAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAstype)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.wrap_param */

Expr MakeWrapParam(Expr data, DataType dtype) {
  ObjectPtr<WrapParamAttrs> attrs = ffi::make_object<WrapParamAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.wrap_param");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.wrap_param", MakeWrapParam);
});

StructInfo InferStructInfoWrapParam(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<WrapParamAttrs>();
  ObjectPtr<TensorStructInfoNode> new_sinfo = ffi::make_object<TensorStructInfoNode>(*sinfo.get());
  new_sinfo->dtype = attrs->dtype;
  return TensorStructInfo(new_sinfo);
}

TVM_REGISTER_OP("relax.wrap_param")
    .set_attrs_type<WrapParamAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoWrapParam)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
