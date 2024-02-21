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
 * \file src/relax/op/nn/convolution.cc
 * \brief Convolution operators
 */

#include "convolution.h"

#include <vector>

namespace tvm {
namespace relax {

/* relax.nn.conv1d */
TVM_REGISTER_NODE_TYPE(Conv1DAttrs);

Expr conv1d(Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
            Array<IntImm> dilation, int groups, String data_layout, String kernel_layout,
            Optional<String> out_layout, DataType out_dtype) {
  padding = GetCompletePadding1D(std::move(padding));

  CHECK_GT(groups, 0) << "The number of groups in convolution is expected to be positive. However, "
                         "the given number of groups is "
                      << groups;
  CHECK_EQ(strides.size(), 1)
      << "The input strides length is expected to be 1. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 1)
      << "The input dilation length is expected to be 1. However, the given dilation is "
      << dilation;
  return MakeConv<Conv1DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), groups, data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype, /*op_name=*/"relax.nn.conv1d");
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv1d").set_body_typed(conv1d);

StructInfo InferStructInfoConv1d(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo weight_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<Conv1DAttrs>();
  auto [data_layout, data2NCW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                   /*tgt_layout=*/"NCW",           //
                                                   /*tensor_name=*/"data");
  auto [weight_layout, weight2OIW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                       /*tgt_layout=*/"OIW",             //
                                                       /*tensor_name=*/"kernel");
  auto [out_layout, out2NCW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                 /*tgt_layout=*/"NCW",          //
                                                 /*tensor_name=*/"output");

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);

  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, data_sinfo, weight_sinfo)
                           : attrs->out_dtype;
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, weight_sinfo);
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, out_layout.ndim(), vdevice);
  }

  Array<PrimExpr> data_NCW_shape = data2NCW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> weight_OIW_shape = weight2OIW.ForwardShape(weight_shape.value()->values);

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCW_shape[1];
  PrimExpr input_channel_kernel = weight_OIW_shape[1];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups);
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIW_shape[0], attrs->groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv1d expects the number of output channels to be divisible by the "
                        "number of groups. However, the number of output channels is "
                     << weight_OIW_shape[0] << " while the number of groups is " << attrs->groups);
  } else if (!analyzer->CanProveEqual(floormod(weight_OIW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_w = data_NCW_shape[2];
  PrimExpr kernel_w = weight_OIW_shape[2];
  PrimExpr padding_w = attrs->padding[0] + attrs->padding[1];

  std::vector<PrimExpr> out_NCW_shape;
  out_NCW_shape.resize(3);
  out_NCW_shape[0] = data_NCW_shape[0];
  out_NCW_shape[1] = weight_OIW_shape[0];

  PrimExpr numerator_w = input_w + padding_w - attrs->dilation[0] * (kernel_w - 1) - 1;
  out_NCW_shape[2] = analyzer->Simplify(floordiv(numerator_w, attrs->strides[0]) + 1);

  Array<PrimExpr> out_shape = out2NCW.BackwardShape(out_NCW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv1d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.nn.conv1d");
  const auto* attrs = call->attrs.as<Conv1DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout, weight_layout, output_layout;
  ObjectPtr<Conv1DAttrs> new_attrs = make_object<Conv1DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv1d.
    Layout desired_data_layout = (*it).second[0];
    Layout desired_weight_layout = (*it).second[1];
    Layout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal()) << "Axis swap only";
    ICHECK_EQ(desired_weight_layout.ndim(), desired_weight_layout.ndim_primal())
        << "Axis swap only";
    ICHECK_EQ(desired_output_layout.ndim(), desired_output_layout.ndim_primal())
        << "Axis swap only";
    data_layout = TransposeLike(InitialLayout(3), attrs->data_layout, desired_data_layout);
    weight_layout = TransposeLike(InitialLayout(3), attrs->kernel_layout, desired_weight_layout);
    output_layout = TransposeLike(InitialLayout(3), attrs->out_layout, desired_output_layout);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    // We don't have a desired layout for conv1d.
    // We can just propagate the layout from the input.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(attrs->data_layout, InitialLayout(3), data_layout->layout).name();
    new_attrs->kernel_layout =
        TransposeLike(attrs->kernel_layout, InitialLayout(3), weight_layout->layout).name();
    new_attrs->out_layout =
        TransposeLike(attrs->out_layout, InitialLayout(3), output_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv1d(const Call& call, const DataType& out_dtype) {
  const auto* conv1d_attrs = call->attrs.as<Conv1DAttrs>();
  return Downcast<Call>(conv1d(call->args[0], call->args[1], conv1d_attrs->strides,
                               conv1d_attrs->padding, conv1d_attrs->dilation, conv1d_attrs->groups,
                               conv1d_attrs->data_layout, conv1d_attrs->kernel_layout,
                               conv1d_attrs->out_layout, out_dtype));
}

TVM_REGISTER_OP("relax.nn.conv1d")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv1DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv1d)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutConv1d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionConv1d)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.conv2d */
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

Expr conv2d(Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
            Array<IntImm> dilation, int groups, String data_layout, String kernel_layout,
            Optional<String> out_layout, DataType out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  CHECK_GT(groups, 0) << "The number of groups in convolution is expected to be positive. However, "
                         "the given number of groups is "
                      << groups;
  CHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;
  return MakeConv<Conv2DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), groups, data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype, /*op_name=*/"relax.nn.conv2d");
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv2d").set_body_typed(conv2d);

StructInfo InferStructInfoConv2d(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo weight_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                    /*tgt_layout=*/"NCHW",          //
                                                    /*tensor_name=*/"data");
  auto [weight_layout, weight2OIHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                        /*tgt_layout=*/"OIHW",            //
                                                        /*tensor_name=*/"kernel");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);

  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, data_sinfo, weight_sinfo)
                           : attrs->out_dtype;
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, weight_sinfo);
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, out_layout.ndim(), vdevice);
  }

  Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> weight_OIHW_shape = weight2OIHW.ForwardShape(weight_shape.value()->values);

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCHW_shape[1];
  PrimExpr input_channel_kernel = weight_OIHW_shape[1];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups);
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIHW_shape[0], attrs->groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv2d expects the number of output channels to be divisible by the "
                        "number of groups. However, the number of output channels is "
                     << weight_OIHW_shape[0] << " while the number of groups is " << attrs->groups);
  } else if (!analyzer->CanProveEqual(floormod(weight_OIHW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = weight_OIHW_shape[2];
  PrimExpr kernel_w = weight_OIHW_shape[3];
  PrimExpr padding_h = attrs->padding[0] + attrs->padding[2];
  PrimExpr padding_w = attrs->padding[1] + attrs->padding[3];

  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.resize(4);
  out_NCHW_shape[0] = data_NCHW_shape[0];
  out_NCHW_shape[1] = weight_OIHW_shape[0];

  PrimExpr numerator_h = input_h + padding_h - attrs->dilation[0] * (kernel_h - 1) - 1;
  PrimExpr numerator_w = input_w + padding_w - attrs->dilation[1] * (kernel_w - 1) - 1;
  out_NCHW_shape[2] = analyzer->Simplify(floordiv(numerator_h, attrs->strides[0]) + 1);
  out_NCHW_shape[3] = analyzer->Simplify(floordiv(numerator_w, attrs->strides[1]) + 1);

  Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.nn.conv2d");
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout, weight_layout, output_layout;
  ObjectPtr<Conv2DAttrs> new_attrs = make_object<Conv2DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv2d.
    Layout desired_data_layout = (*it).second[0];
    Layout desired_weight_layout = (*it).second[1];
    Layout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal()) << "Axis swap only";
    ICHECK_EQ(desired_weight_layout.ndim(), desired_weight_layout.ndim_primal())
        << "Axis swap only";
    ICHECK_EQ(desired_output_layout.ndim(), desired_output_layout.ndim_primal())
        << "Axis swap only";
    data_layout = TransposeLike(InitialLayout(4), attrs->data_layout, desired_data_layout);
    weight_layout = TransposeLike(InitialLayout(4), attrs->kernel_layout, desired_weight_layout);
    output_layout = TransposeLike(InitialLayout(4), attrs->out_layout, desired_output_layout);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    // We don't have a desired layout for conv2d.
    // We can just propagate the layout from the input.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(attrs->data_layout, InitialLayout(4), data_layout->layout).name();
    new_attrs->kernel_layout =
        TransposeLike(attrs->kernel_layout, InitialLayout(4), weight_layout->layout).name();
    new_attrs->out_layout =
        TransposeLike(attrs->out_layout, InitialLayout(4), output_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv2d(const Call& call, const DataType& out_dtype) {
  const auto* conv2d_attrs = call->attrs.as<Conv2DAttrs>();
  return Downcast<Call>(conv2d(call->args[0], call->args[1], conv2d_attrs->strides,
                               conv2d_attrs->padding, conv2d_attrs->dilation, conv2d_attrs->groups,
                               conv2d_attrs->data_layout, conv2d_attrs->kernel_layout,
                               conv2d_attrs->out_layout, out_dtype));
}

TVM_REGISTER_OP("relax.nn.conv2d")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2d)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutConv2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionConv2d)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.conv3d */
TVM_REGISTER_NODE_TYPE(Conv3DAttrs);

Expr conv3d(Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
            Array<IntImm> dilation, int groups, String data_layout, String kernel_layout,
            Optional<String> out_layout, DataType out_dtype) {
  padding = GetCompletePadding3D(std::move(padding));
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
    dilation.push_back(dilation[0]);
  }

  CHECK_GT(groups, 0) << "The number of groups in convolution is expected to be positive. However, "
                         "the given number of groups is "
                      << groups;
  CHECK_EQ(strides.size(), 3)
      << "The input strides length is expected to be 3. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 3)
      << "The input dilation length is expected to be 3. However, the given dilation is "
      << dilation;
  return MakeConv<Conv3DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), groups, data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype, /*op_name=*/"relax.nn.conv3d");
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv3d").set_body_typed(conv3d);

StructInfo InferStructInfoConv3d(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo weight_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<Conv3DAttrs>();
  auto [data_layout, data2NCDHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                     /*tgt_layout=*/"NCDHW",         //
                                                     /*tensor_name=*/"data");
  auto [weight_layout, weight2OIDHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                         /*tgt_layout=*/"OIDHW",           //
                                                         /*tensor_name=*/"kernel");
  auto [out_layout, out2NCDHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                   /*tgt_layout=*/"NCDHW",        //
                                                   /*tensor_name=*/"output");

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);

  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, data_sinfo, weight_sinfo)
                           : attrs->out_dtype;
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, weight_sinfo);
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, out_layout.ndim(), vdevice);
  }

  Array<PrimExpr> data_NCDHW_shape = data2NCDHW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> weight_OIDHW_shape = weight2OIDHW.ForwardShape(weight_shape.value()->values);

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCDHW_shape[1];
  PrimExpr input_channel_kernel = weight_OIDHW_shape[1];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups);
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIDHW_shape[0], attrs->groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv3d expects the number of output channels to be divisible by the "
                        "number of groups. However, the number of output channels is "
                     << weight_OIDHW_shape[0] << " while the number of groups is "
                     << attrs->groups);
  } else if (!analyzer->CanProveEqual(floormod(weight_OIDHW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_d = data_NCDHW_shape[2];
  PrimExpr input_h = data_NCDHW_shape[3];
  PrimExpr input_w = data_NCDHW_shape[4];
  PrimExpr kernel_d = weight_OIDHW_shape[2];
  PrimExpr kernel_h = weight_OIDHW_shape[3];
  PrimExpr kernel_w = weight_OIDHW_shape[4];
  PrimExpr padding_d = attrs->padding[0] + attrs->padding[3];
  PrimExpr padding_h = attrs->padding[1] + attrs->padding[4];
  PrimExpr padding_w = attrs->padding[2] + attrs->padding[5];

  std::vector<PrimExpr> out_NCDHW_shape;
  out_NCDHW_shape.resize(5);
  out_NCDHW_shape[0] = data_NCDHW_shape[0];
  out_NCDHW_shape[1] = weight_OIDHW_shape[0];

  PrimExpr numerator_d = input_d + padding_d - attrs->dilation[0] * (kernel_d - 1) - 1;
  PrimExpr numerator_h = input_h + padding_h - attrs->dilation[1] * (kernel_h - 1) - 1;
  PrimExpr numerator_w = input_w + padding_w - attrs->dilation[2] * (kernel_w - 1) - 1;
  out_NCDHW_shape[2] = analyzer->Simplify(floordiv(numerator_d, attrs->strides[0]) + 1);
  out_NCDHW_shape[3] = analyzer->Simplify(floordiv(numerator_h, attrs->strides[1]) + 1);
  out_NCDHW_shape[4] = analyzer->Simplify(floordiv(numerator_w, attrs->strides[2]) + 1);

  Array<PrimExpr> out_shape = out2NCDHW.BackwardShape(out_NCDHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, vdevice);
}

InferLayoutOutput InferLayoutConv3d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.nn.conv3d");
  const auto* attrs = call->attrs.as<Conv3DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout, weight_layout, output_layout;
  ObjectPtr<Conv3DAttrs> new_attrs = make_object<Conv3DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv3d.
    Layout desired_data_layout = (*it).second[0];
    Layout desired_weight_layout = (*it).second[1];
    Layout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal()) << "Axis swap only";
    ICHECK_EQ(desired_weight_layout.ndim(), desired_weight_layout.ndim_primal())
        << "Axis swap only";
    ICHECK_EQ(desired_output_layout.ndim(), desired_output_layout.ndim_primal())
        << "Axis swap only";
    data_layout = TransposeLike(InitialLayout(5), attrs->data_layout, desired_data_layout);
    weight_layout = TransposeLike(InitialLayout(5), attrs->kernel_layout, desired_weight_layout);
    output_layout = TransposeLike(InitialLayout(5), attrs->out_layout, desired_output_layout);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    // We don't have a desired layout for conv2d.
    // We can just propagate the layout from the input.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(attrs->data_layout, InitialLayout(5), data_layout->layout).name();
    new_attrs->kernel_layout =
        TransposeLike(attrs->kernel_layout, InitialLayout(5), weight_layout->layout).name();
    new_attrs->out_layout =
        TransposeLike(attrs->out_layout, InitialLayout(5), output_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv3d(const Call& call, const DataType& out_dtype) {
  const auto* conv3d_attrs = call->attrs.as<Conv3DAttrs>();
  return Downcast<Call>(conv3d(call->args[0], call->args[1], conv3d_attrs->strides,
                               conv3d_attrs->padding, conv3d_attrs->dilation, conv3d_attrs->groups,
                               conv3d_attrs->data_layout, conv3d_attrs->kernel_layout,
                               conv3d_attrs->out_layout, out_dtype));
}

TVM_REGISTER_OP("relax.nn.conv3d")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv3DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv3d)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutConv3d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionConv3d)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_NODE_TYPE(Conv1DTransposeAttrs);

Expr conv1d_transpose(Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
                      Array<IntImm> output_padding, Array<IntImm> dilation, int groups,
                      String data_layout, String kernel_layout, Optional<String> out_layout,
                      DataType out_dtype) {
  padding = GetCompletePadding1D(std::move(padding));

  CHECK_GT(groups, 0) << "The number of groups in convolution is expected to be positive. However, "
                         "the given number of groups is "
                      << groups;
  CHECK_EQ(output_padding.size(), 1) << "The input output_padding length is expected to be 1. "
                                        "However, the given output_padding is "
                                     << output_padding;
  CHECK_EQ(strides.size(), 1)
      << "The input strides length is expected to be 1. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 1)
      << "The input dilation length is expected to be 1. However, the given dilation is "
      << dilation;

  auto attrs = make_object<Conv1DTransposeAttrs>();
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->output_padding = ConvertIntImmToInt64(output_padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->groups = groups;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout.value_or(data_layout));
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get("relax.nn.conv1d_transpose");
  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv1d_transpose").set_body_typed(conv1d_transpose);

StructInfo InferStructInfoConv1dTranspose(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo weight_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<Conv1DTransposeAttrs>();
  auto [data_layout, data2NCW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                   /*tgt_layout=*/"NCW",           //
                                                   /*tensor_name=*/"data");
  auto [weight_layout, weight2IOW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                       /*tgt_layout=*/"IOW",             //
                                                       /*tensor_name=*/"kernel");
  auto [out_layout, out2NCW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                 /*tgt_layout=*/"NCW",          //
                                                 /*tensor_name=*/"output");
  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);

  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, data_sinfo, weight_sinfo)
                           : attrs->out_dtype;
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, weight_sinfo);
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, out_layout.ndim(), vdevice);
  }

  Array<PrimExpr> data_NCW_shape = data2NCW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> weight_IOW_shape = weight2IOW.ForwardShape(weight_shape.value()->values);

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCW_shape[1];
  PrimExpr input_channel_kernel = weight_IOW_shape[0];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel)) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Conv1dTranspose expects the channel size of the data should equal to the input channel "
           "size of the weight. However, the data channel size is "
        << input_channel_data << " while the weight input channel size is "
        << input_channel_kernel);
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(input_channel_kernel, attrs->groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv1dTranspose expects the number of input channels to be divisible by "
                        "the number of groups. However, the number of input channels is "
                     << input_channel_kernel << " while the number of groups is " << attrs->groups);
  } else if (!analyzer->CanProveEqual(floormod(input_channel_kernel, attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(attrs->output_padding[0]->value >= attrs->strides[0]->value)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv1dTranspose expects the output padding less than the strides, but the "
                        "output padding is"
                     << attrs->output_padding << " while the strides are" << attrs->strides);
  } else if (!analyzer->CanProve(attrs->output_padding[0]->value < attrs->strides[0]->value)) {
    // Todo(relax-team): Trust the input padding at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_w = data_NCW_shape[2];
  PrimExpr kernel_w = weight_IOW_shape[2];
  PrimExpr padding_w = attrs->padding[0] + attrs->padding[1];

  std::vector<PrimExpr> out_NCW_shape;
  out_NCW_shape.resize(3);
  out_NCW_shape[0] = data_NCW_shape[0];
  out_NCW_shape[1] = weight_IOW_shape[1] * attrs->groups;

  PrimExpr out_w = (input_w - 1) * attrs->strides[0] - padding_w +
                   attrs->dilation[0] * (kernel_w - 1) + attrs->output_padding[0] + 1;
  out_NCW_shape[2] = analyzer->Simplify(out_w);

  Array<PrimExpr> out_shape = out2NCW.BackwardShape(out_NCW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, vdevice);
}

// TODO(relax-team): implement FInferMixedPrecision and FRelaxInferLayout for conv1d_transpose
// and unit test for mixed_precision
TVM_REGISTER_OP("relax.nn.conv1d_transpose")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv1DTransposeAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv1dTranspose)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.nn.conv2d_transpose */
TVM_REGISTER_NODE_TYPE(Conv2DTransposeAttrs);

Expr conv2d_transpose(Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
                      Array<IntImm> output_padding, Array<IntImm> dilation, int groups,
                      String data_layout, String kernel_layout, Optional<String> out_layout,
                      DataType out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));
  if (output_padding.size() == 1) {
    output_padding.push_back(output_padding[0]);
  }
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  CHECK_GT(groups, 0) << "The number of groups in convolution is expected to be positive. However, "
                         "the given number of groups is "
                      << groups;
  CHECK_EQ(output_padding.size(), 2) << "The input output_padding length is expected to be 4. "
                                        "However, the given output_padding is "
                                     << output_padding;
  CHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;

  auto attrs = make_object<Conv2DTransposeAttrs>();
  attrs->strides = ConvertIntImmToInt64(strides);
  attrs->padding = ConvertIntImmToInt64(padding);
  attrs->output_padding = ConvertIntImmToInt64(output_padding);
  attrs->dilation = ConvertIntImmToInt64(dilation);
  attrs->groups = groups;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout.value_or(data_layout));
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get("relax.nn.conv2d_transpose");
  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv2d_transpose").set_body_typed(conv2d_transpose);

StructInfo InferStructInfoConv2dTranspose(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo weight_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<Conv2DTransposeAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                    /*tgt_layout=*/"NCHW",          //
                                                    /*tensor_name=*/"data");
  auto [weight_layout, weight2IOHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                        /*tgt_layout=*/"IOHW",            //
                                                        /*tensor_name=*/"kernel");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);

  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, data_sinfo, weight_sinfo)
                           : attrs->out_dtype;
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, data_sinfo, weight_sinfo);
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, out_layout.ndim(), vdevice);
  }

  Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> weight_IOHW_shape = weight2IOHW.ForwardShape(weight_shape.value()->values);

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCHW_shape[1];
  PrimExpr input_channel_kernel = weight_IOHW_shape[0];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel)) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Conv2dTranspose expects the channel size of the data should equal to the input channel "
           "size of the weight. However, the data channel size is "
        << input_channel_data << " while the weight input channel size is "
        << input_channel_kernel);
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(input_channel_kernel, attrs->groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv2dTranspose expects the number of input channels to be divisible by "
                        "the number of groups. However, the number of input channels is "
                     << input_channel_kernel << " while the number of groups is " << attrs->groups);
  } else if (!analyzer->CanProveEqual(floormod(input_channel_kernel, attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(attrs->output_padding[0]->value >= attrs->strides[0]->value ||
                         attrs->output_padding[1]->value >= attrs->strides[1]->value)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv2dTranspose expects the output padding less than the strides, but the "
                        "output padding is"
                     << attrs->output_padding << " while the strides are" << attrs->strides);
  } else if (!analyzer->CanProve(attrs->output_padding[0]->value < attrs->strides[0]->value &&
                                 attrs->output_padding[1]->value < attrs->strides[1]->value)) {
    // Todo(relax-team): Trust the input padding at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = weight_IOHW_shape[2];
  PrimExpr kernel_w = weight_IOHW_shape[3];
  PrimExpr padding_h = attrs->padding[0] + attrs->padding[2];
  PrimExpr padding_w = attrs->padding[1] + attrs->padding[3];

  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.resize(4);
  out_NCHW_shape[0] = data_NCHW_shape[0];
  out_NCHW_shape[1] = weight_IOHW_shape[1] * attrs->groups;

  PrimExpr out_h = (input_h - 1) * attrs->strides[0] - padding_h +
                   attrs->dilation[0] * (kernel_h - 1) + attrs->output_padding[0] + 1;
  PrimExpr out_w = (input_w - 1) * attrs->strides[1] - padding_w +
                   attrs->dilation[1] * (kernel_w - 1) + attrs->output_padding[1] + 1;
  out_NCHW_shape[2] = analyzer->Simplify(out_h);
  out_NCHW_shape[3] = analyzer->Simplify(out_w);

  Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, vdevice);
}

// TODO(relax-team): implement FInferMixedPrecision and FRelaxInferLayout for conv2d_transpose
// and unit test for mixed_precision
TVM_REGISTER_OP("relax.nn.conv2d_transpose")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DTransposeAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2dTranspose)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
