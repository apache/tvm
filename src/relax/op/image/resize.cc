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
 * \file resize.cc
 * \brief Image resize operators.
 */

#include "resize.h"

#include <utility>

namespace tvm {
namespace relax {

/* relax.resize2d */
TVM_REGISTER_NODE_TYPE(Resize2DAttrs);

Expr resize2d(Expr data, Expr size, Array<FloatImm> roi, String layout, String method,
              String coordinate_transformation_mode, String rounding_method, double cubic_alpha,
              int cubic_exclude, double extrapolation_value, DataType out_dtype) {
  ObjectPtr<Resize2DAttrs> attrs = make_object<Resize2DAttrs>();
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = std::move(coordinate_transformation_mode);
  attrs->rounding_method = std::move(rounding_method);
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.image.resize2d");
  return Call(op, {std::move(data), std::move(size)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.image.resize2d").set_body_typed(resize2d);

StructInfo InferStructInfoResize2D(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1 && call->args.size() != 2) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Resize2D expects either one or two arguments, while the given number of arguments is "
        << call->args.size());
  }

  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* size_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  const auto* size_value = call->args[1].as<ShapeExprNode>();
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Resize2D expects the input data to be a Tensor, while the given data is "
                     << call->args[0]->GetTypeKey());
  }
  if (size_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Resize2D expects the given output image size to be a Shape, while the given one is "
        << call->args[1]->GetTypeKey());
  }
  if (size_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Resize2D expects the given output image size to "
                                                "be a 2-dim shape, while the given one has ndim "
                                             << size_sinfo->ndim);
  }

  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->layout,  //
                                                    /*tgt_layout=*/"NCHW",     //
                                                    /*tensor_name=*/"data");

  DataType out_dtype = attrs->out_dtype.is_void() ? data_sinfo->dtype : attrs->out_dtype;

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, GetRef<TensorStructInfo>(data_sinfo), data_layout);
  if (!data_shape.defined() || size_value == nullptr) {
    return TensorStructInfo(out_dtype, data_layout.ndim(), data_sinfo->vdevice);
  }

  Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> out_NCHW_shape(data_NCHW_shape);
  out_NCHW_shape.Set(2, size_value->values[0]);
  out_NCHW_shape.Set(3, size_value->values[1]);

  Array<PrimExpr> out_shape = data2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, data_sinfo->vdevice);
}

InferLayoutOutput InferLayoutResize2d(const Call& call,
                                      const Map<String, Array<String>>& desired_layouts,
                                      const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ObjectPtr<Resize2DAttrs> new_attrs = make_object<Resize2DAttrs>(*attrs);
  new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(4), layout->layout).name();
  return InferLayoutOutput({layout, InitialNLayout(call->args[1])}, {layout}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.image.resize2d")
    .set_attrs_type<Resize2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("size", "Shape", "The output image shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoResize2D)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutResize2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
