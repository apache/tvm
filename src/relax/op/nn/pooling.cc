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

#include "pooling.h"

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.nn.max_pool2d and relax.nn.avg_pool2d */
TVM_REGISTER_NODE_TYPE(Pool2DAttrs);

Expr MakePool2d(String op_name, Expr data, Array<IntImm> pool_size, Array<IntImm> strides,
                Array<IntImm> padding, Array<IntImm> dilation, bool ceil_mode, String layout,
                Optional<String> out_layout) {
  padding = GetCompletePadding2D(std::move(padding));
  if (pool_size.size() == 1) {
    pool_size.push_back(pool_size[0]);
  }
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  CHECK_EQ(pool_size.size(), 2)
      << "The input pool_size length is expected to be 2. However, the given pool_size is "
      << pool_size;
  CHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;

  auto attrs = make_object<Pool2DAttrs>();
  attrs->pool_size = ConvertIntImmToInt64(pool_size);
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->ceil_mode = ceil_mode;
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  const Op& op = Op::Get(op_name);
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

Expr max_pool2d(Expr data, Array<IntImm> pool_size, Array<IntImm> strides, Array<IntImm> padding,
                Array<IntImm> dilation, bool ceil_mode, String layout,
                Optional<String> out_layout) {
  return MakePool2d("relax.nn.max_pool2d", data, pool_size, strides, padding, dilation, ceil_mode,
                    layout, out_layout);
}

TVM_REGISTER_GLOBAL("relax.op.nn.max_pool2d").set_body_typed(max_pool2d);

StructInfo InferStructInfoPool2D(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<Pool2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->layout,  //
                                                    /*tgt_layout=*/"NCHW",     //
                                                    /*tensor_name=*/"data");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  if (!data_shape.defined()) {
    return TensorStructInfo(data_sinfo->dtype, out_layout.ndim(), data_sinfo->vdevice);
  }

  Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = attrs->pool_size[0];
  PrimExpr kernel_w = attrs->pool_size[1];
  PrimExpr padding_h = attrs->padding[0] + attrs->padding[2];
  PrimExpr padding_w = attrs->padding[1] + attrs->padding[3];

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.resize(4);
  out_NCHW_shape[0] = data_NCHW_shape[0];
  out_NCHW_shape[1] = data_NCHW_shape[1];

  PrimExpr numerator_h = input_h + padding_h - attrs->dilation[0] * (kernel_h - 1) - 1;
  PrimExpr numerator_w = input_w + padding_w - attrs->dilation[1] * (kernel_w - 1) - 1;
  if (attrs->ceil_mode) {
    numerator_h += attrs->strides[0] - 1;
    numerator_w += attrs->strides[1] - 1;
  }
  out_NCHW_shape[2] = analyzer->Simplify(floordiv(numerator_h, attrs->strides[0]) + 1);
  out_NCHW_shape[3] = analyzer->Simplify(floordiv(numerator_w, attrs->strides[1]) + 1);

  Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

InferLayoutOutput InferLayoutPool2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK_EQ(tensor_sinfo->ndim, 4) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<Pool2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ObjectPtr<Pool2DAttrs> new_attrs = make_object<Pool2DAttrs>(*attrs);
  new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(4), layout->layout).name();
  new_attrs->out_layout = TransposeLike(attrs->out_layout, InitialLayout(4), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.nn.max_pool2d")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<Pool2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoPool2D)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutPool2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

Expr avg_pool2d(Expr data, Array<IntImm> pool_size, Array<IntImm> strides, Array<IntImm> padding,
                Array<IntImm> dilation, bool ceil_mode, String layout,
                Optional<String> out_layout) {
  return MakePool2d("relax.nn.avg_pool2d", data, pool_size, strides, padding, dilation, ceil_mode,
                    layout, out_layout);
}

TVM_REGISTER_GLOBAL("relax.op.nn.avg_pool2d").set_body_typed(avg_pool2d);

TVM_REGISTER_OP("relax.nn.avg_pool2d")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<Pool2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoPool2D)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutPool2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.adaptive_avg_pool2d */
TVM_REGISTER_NODE_TYPE(AdaptivePool2DAttrs);

Expr adaptive_avg_pool2d(Expr data, Optional<Array<IntImm>> output_size, String layout,
                         Optional<String> out_layout) {
  ObjectPtr<AdaptivePool2DAttrs> attrs = make_object<AdaptivePool2DAttrs>();
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  if (output_size.defined()) {
    Array<IntImm> _output_size = output_size.value();
    if (_output_size.size() == 1) {
      _output_size.push_back(_output_size[0]);
    }
    CHECK_EQ(_output_size.size(), 2)
        << "The output_size length is expected to be 2. However, the given output_size is "
        << _output_size;
    attrs->output_size = std::move(_output_size);
  }

  static const Op& op = Op::Get("relax.nn.adaptive_avg_pool2d");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.adaptive_avg_pool2d").set_body_typed(adaptive_avg_pool2d);

StructInfo InferStructInfoAdaptiveAvgPool2D(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->layout,  //
                                                    /*tgt_layout=*/"NCHW",     //
                                                    /*tensor_name=*/"data");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  if (!data_shape.defined()) {
    if (data_sinfo->shape.defined() && attrs->out_layout == attrs->layout &&
        !attrs->output_size.defined()) {
      return data_sinfo;
    } else {
      return TensorStructInfo(data_sinfo->dtype, out_layout.ndim(), data_sinfo->vdevice);
    }
  }

  Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> out_NCHW_shape(data_NCHW_shape);
  if (attrs->output_size.defined()) {
    out_NCHW_shape.Set(2, attrs->output_size.value()[0]);
    out_NCHW_shape.Set(3, attrs->output_size.value()[1]);
  }

  Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

InferLayoutOutput InferLayoutAdaptiveAvgPool2D(const Call& call,
                                               const Map<String, Array<String>>& desired_layouts,
                                               const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call);
  ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  ICHECK_EQ(tensor_sinfo->ndim, 4) << "Unsupported initial layout";
  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  ObjectPtr<AdaptivePool2DAttrs> new_attrs = make_object<AdaptivePool2DAttrs>(*attrs);
  new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(4), layout->layout).name();
  new_attrs->out_layout = TransposeLike(attrs->out_layout, InitialLayout(4), layout->layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

TVM_REGISTER_OP("relax.nn.adaptive_avg_pool2d")
    .set_attrs_type<AdaptivePool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAdaptiveAvgPool2D)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutAdaptiveAvgPool2D)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
