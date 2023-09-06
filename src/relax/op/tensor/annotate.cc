/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file annotate.cc
 * \brief operators for statistics collection in SmoothQuant.
 */

#include "annotate.h"

#include <utility>

namespace tvm {
namespace relax {

/* relax.annotate.smooth */

Expr smooth(Expr x, Expr m, int k, String md) {
  ObjectPtr<AnnotateSmoothAttrs> attrs = make_object<AnnotateSmoothAttrs>();
  attrs->kind = k;
  attrs->mode = std::move(md);
  static const Op& op = Op::Get("relax.annotate.smooth");
  return Call(op, {std::move(x), std::move(m)}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.smooth").set_body_typed(smooth);

StructInfo InferStructInfoSmooth(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<AnnotateSmoothAttrs>();
  if (attrs->mode != "identity" && attrs->mode != "multiply" && attrs->mode != "quantize")
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Unsupported mode attribute for operation: '" << attrs->mode
                     << "'. Possible variants are: \"identity\", \"multiply\" or \"quantize\"");
  if (attrs->kind != kSQActivation && attrs->kind != kSQWeight)
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Unsupported kind attribute for operation: '" << attrs->kind);

  TensorStructInfo input_sinfo = GetInputTensorStructInfo(call, ctx)[0];
  auto output_sinfo = make_object<TensorStructInfoNode>(*input_sinfo.get());
  output_sinfo->dtype = input_sinfo->dtype;
  return TensorStructInfo(output_sinfo);
}

TVM_REGISTER_NODE_TYPE(AnnotateSmoothAttrs);

TVM_REGISTER_OP("relax.annotate.smooth")
    .set_attrs_type<AnnotateSmoothAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("multiplier", "Tensor", "Tensor smooth multiplier.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSmooth);

/* relax.annotate.absmax */

Expr absmax(Expr x, int k) {
  ObjectPtr<AnnotateAbsMaxAttrs> attrs = make_object<AnnotateAbsMaxAttrs>();
  attrs->kind = k;
  static const Op& op = Op::Get("relax.annotate.absmax");
  return Call(op, {std::move(x)}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.absmax").set_body_typed(absmax);

StructInfo InferStructInfoAbsMax(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo input_sinfo = GetInputTensorStructInfo(call, ctx)[0];

  const auto* attrs = call->attrs.as<AnnotateAbsMaxAttrs>();
  if (attrs->kind == kSQNone) {
    return TensorStructInfo(ShapeExpr({1}), input_sinfo->dtype);
  }

  ICHECK(input_sinfo->shape.defined());
  const auto* input_shape = input_sinfo->shape.as<ShapeExprNode>();
  int ndim = input_sinfo->ndim;
  PrimExpr size = input_shape->values[ndim - 1];
  return TensorStructInfo(ShapeExpr({size}), input_sinfo->dtype);
}

TVM_REGISTER_NODE_TYPE(AnnotateAbsMaxAttrs);

TVM_REGISTER_OP("relax.annotate.absmax")
    .set_attrs_type<AnnotateAbsMaxAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAbsMax);

}  // namespace relax
}  // namespace tvm
