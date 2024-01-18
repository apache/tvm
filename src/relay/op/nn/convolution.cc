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
 * \file convolution.cc
 * \brief Convolution operators
 */
#include "convolution.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../op_common.h"
#include "convolution_make.h"

namespace tvm {
namespace relay {

Expr MakeConvWinogradWeightTransform(Expr weight, int tile_size, std::string op_name) {
  auto attrs = make_object<ConvWinogradWeightTransformAttrs>();
  attrs->tile_size = tile_size;
  const Op& op = Op::Get(op_name);
  return Call(op, {weight}, Attrs(attrs), {});
}

Expr MakeConvGemmWeightTransform(Expr weight, int tile_N, int tile_K, std::string op_name) {
  auto attrs = make_object<ConvGemmWeightTransformAttrs>();
  attrs->tile_N = tile_N;
  attrs->tile_K = tile_K;
  const Op& op = Op::Get(op_name);
  return Call(op, {weight}, Attrs(attrs), {});
}

// relay.nn.conv1d
TVM_REGISTER_NODE_TYPE(Conv1DAttrs);

bool Conv1DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCW("NCW");
  static const Layout kOIW("OIW");

  const auto* param = attrs.as<Conv1DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCW);
  ICHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCW);
  ICHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_ncw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    Array<IndexExpr> wshape;

    wshape = {{param->channels, indexdiv(dshape_ncw[1], param->groups), param->kernel_size[0]}};

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]))
          << "Conv1D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv1D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }
    if (!dshape_ncw[1].as<tir::AnyNode>() && !wshape[1].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(dshape_ncw[1], wshape[1]));
    }
    channels = wshape[0];
    dilated_ksize = 1 + (wshape[2] - 1) * param->dilation[0];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_ncw[0], channels, 0});

  if (!dshape_ncw[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(dshape_ncw[2] + param->padding[0] + param->padding[1] - dilated_ksize,
                           param->strides[0]) +
                      1);
  } else {
    oshape.Set(2, dshape_ncw[2]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv1d")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, DataType out_dtype) {
      return MakeConv<Conv1DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                                   kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                                   "nn.conv1d");
    });

RELAY_REGISTER_OP("nn.conv1d")
    .describe(R"code(1D convolution layer (e.g. spatial convolution over sequences).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 3D array of shape
            (batch_size, in_channels, width) if `layout` is `NCW`.
- **weight**: (channels, in_channels, kernel_size)
- **out**:  This depends on the `layout` parameter. Output is 3D array of shape
            (batch_size, channels, out_width) if `layout` is `NCW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv1DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("Conv1D", Conv1DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv1DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.conv2d
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

bool Conv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCHW("NCHW");
  Layout kOIHW("OIHW");

  const auto* param = attrs.as<Conv2DAttrs>();
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
    if (out_dtype.bits() == 0 && weight != nullptr) {
      out_dtype = weight->dtype;
    }
  }
  TensorType meta_schedule_weight{nullptr};
  if (param->meta_schedule_original_shape.size() != 0) {
    meta_schedule_weight = TensorType(param->meta_schedule_original_shape, out_dtype);
    weight = meta_schedule_weight.get();
  }
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  bool is_dnnl_group_conv = false;
  if (param->groups > 1 && kernel_layout.name().find("G") != std::string::npos) {
    kOIHW = Layout("GOIHW");
    is_dnnl_group_conv = true;
  }

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  if (!trans_in_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "conv2d only support input layouts that are convertible from NCHW."
        << " The provided layout is: " << in_layout);
    return false;
  }

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIHW);
  if (!trans_kernel_layout.defined()) {
    reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                << "conv2d only support kernel layouts that are convertible from "
                                << kOIHW << "."
                                << " The provided layout is: " << kernel_layout);
    return false;
  }

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCHW);
  if (!trans_out_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "conv2d only support output layouts that are convertible from NCHW."
        << "The provided layout is: " << out_layout);
    return false;
  }

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);
  bool is_depthwise = false;
  if (param->groups > 1) {
    if (!(weight && weight->shape.defined())) {
      reporter->GetDiagCtx().Emit(
          Diagnostic::Error(reporter->GetSpan())
          << "Weight shape must be specified when groups is greater than 1.");
      return false;
    }

    Array<IndexExpr> wshape_oihw = trans_kernel_layout.ForwardShape(weight->shape);
    if (tvm::tir::ExprDeepEqual()(param->groups, dshape_nchw[1]) &&
        tvm::tir::ExprDeepEqual()(param->groups, wshape_oihw[0])) {
      is_depthwise = true;
    }
  }

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 2);
    ICHECK_EQ(param->dilation.size(), 2);
    Array<IndexExpr> wshape;

    if (is_dnnl_group_conv) {
      // infer weight's shape for group convolution
      wshape = {{param->groups, indexdiv(param->channels, param->groups),
                 indexdiv(dshape_nchw[1], param->groups), param->kernel_size[0],
                 param->kernel_size[1]}};
    } else if (is_depthwise) {
      // infer weight's shape for depthwise convolution
      wshape = {{dshape_nchw[1], indexdiv(param->channels, dshape_nchw[1]), param->kernel_size[0],
                 param->kernel_size[1]}};
    } else {
      wshape = {{param->channels, indexdiv(dshape_nchw[1], param->groups), param->kernel_size[0],
                 param->kernel_size[1]}};
    }

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }

    if (param->auto_scheduler_rewritten_layout.size() != 0) {
      // If the layout is rewritten by auto-scheduler,
      // we just forcly apply the layout provided by auto-scheduler and
      // skip the normal inference logic.
      {}  // do nothing
    } else if (param->meta_schedule_original_shape.size() == 0) {
      // Normal case: assign result to reporter
      reporter->Assign(types[1], TensorType(wshape, weight_dtype));
    }
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;

    Array<PrimExpr> wshape;
    if (param->auto_scheduler_rewritten_layout.size() != 0) {
      // works for the default kernel layout "HWIO"
      ICHECK_EQ(param->kernel_layout, "HWIO");
      wshape = auto_scheduler::GetShapeFromRewrittenLayout(param->auto_scheduler_rewritten_layout,
                                                           {"ry", "rx", "rc", "ff"});
    } else {
      wshape = weight->shape;
    }

    wshape = trans_kernel_layout.ForwardShape(wshape);
    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 2);

      if (!reporter->AssertEQ(param->kernel_size[0], wshape[2])) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                    << "Conv2D: shape of weight is inconsistent with kernel_size,"
                                    << " kernel_size=" << param->kernel_size
                                    << " wshape=" << wshape);
      }

      if (!reporter->AssertEQ(param->kernel_size[1], wshape[3])) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                    << "Conv2D: shape of weight is inconsistent with kernel_size,"
                                    << " kernel_size=" << param->kernel_size
                                    << " wshape=" << wshape);
        return false;
      }
    }

    if (param->channels.defined() && !reporter->AssertEQ(param->channels, wshape[0])) {
      reporter->GetDiagCtx().Emit(
          Diagnostic::Error(reporter->GetSpan())
          << "conv2D: the first dimensions of the weight tensor (" << wshape << ")"
          << "does not match the number of channels (" << param->channels << ").");
      return false;
    }

    if (!dshape_nchw[1].as<tir::AnyNode>() && !wshape[1].as<tir::AnyNode>()) {
      if (!reporter->AssertEQ(indexdiv(dshape_nchw[1], param->groups), wshape[1])) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(reporter->GetSpan())
                                    << "conv2d: requires that `"
                                    << indexdiv(dshape_nchw[1], param->groups) << "`,"
                                    << " the input channels (" << dshape_nchw[1] << ")"
                                    << " divided by groups (" << param->groups << ")"
                                    << ",\n must match the input channels"
                                    << " of the weight `" << wshape[1]
                                    << "`, where the weight shape is (" << wshape << ").");
        return false;
      }
    }
    channels = wshape[0];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  if (!dshape_nchw[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(dshape_nchw[2] + pad_h - dilated_ksize_y, param->strides[0]) + 1);
  } else {
    oshape.Set(2, dshape_nchw[2]);
  }

  if (!dshape_nchw[3].as<tir::AnyNode>()) {
    oshape.Set(3, indexdiv(dshape_nchw[3] + pad_w - dilated_ksize_x, param->strides[1]) + 1);
  } else {
    oshape.Set(3, dshape_nchw[3]);
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv2d")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, DataType out_dtype) {
      return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                                   kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                                   "nn.conv2d");
    });

RELAY_REGISTER_OP("nn.conv2d")
    .describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("Conv2D", Conv2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.conv3d
TVM_REGISTER_NODE_TYPE(Conv3DAttrs);

bool Conv3DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCDHW("NCDHW");
  static const Layout kOIDHW("OIDHW");

  const auto* param = attrs.as<Conv3DAttrs>();
  ICHECK(param != nullptr);
  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
    if (out_dtype.bits() == 0 && weight != nullptr) {
      out_dtype = weight->dtype;
    }
  }
  TensorType meta_schedule_weight{nullptr};
  if (param->meta_schedule_original_shape.size() != 0) {
    meta_schedule_weight = TensorType(param->meta_schedule_original_shape, out_dtype);
    weight = meta_schedule_weight.get();
  }
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCDHW);
  ICHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCDHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIDHW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIDHW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCDHW);
  ICHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCDHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_ncdhw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_z, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 3);
    ICHECK_EQ(param->dilation.size(), 3);

    bool is_depthwise = false;
    if (param->groups > 1) {
      if (!(weight && weight->shape.defined())) {
        reporter->GetDiagCtx().Emit(
            Diagnostic::Error(reporter->GetSpan())
            << "Weight shape must be specified when groups is greater than 1.");
        return false;
      }

      Array<IndexExpr> wshape_oidhw = trans_kernel_layout.ForwardShape(weight->shape);
      if (tvm::tir::ExprDeepEqual()(param->groups, dshape_ncdhw[1]) &&
          tvm::tir::ExprDeepEqual()(param->groups, wshape_oidhw[0])) {
        is_depthwise = true;
      }
    }

    Array<IndexExpr> wshape;
    if (is_depthwise) {
      auto channel_multiplier = indexdiv(param->channels, dshape_ncdhw[1]);
      wshape = {dshape_ncdhw[1], channel_multiplier, param->kernel_size[0], param->kernel_size[1],
                param->kernel_size[2]};
    } else {
      wshape = {param->channels, indexdiv(dshape_ncdhw[1], param->groups), param->kernel_size[0],
                param->kernel_size[1], param->kernel_size[2]};
    }

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize_z = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_y = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    dilated_ksize_x = 1 + (param->kernel_size[2] - 1) * param->dilation[2];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }

    if (param->auto_scheduler_rewritten_layout.size() != 0) {
      // If the layout is rewritten by auto-scheduler,
      // we just forcly apply the layout provided by auto-scheduler and
      // skip the normal inference logic.
      {}  // do nothing
    } else if (param->meta_schedule_original_shape.size() == 0) {
      // Normal case: assign result to reporter
      reporter->Assign(types[1], TensorType(wshape, weight_dtype));
    }

  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;

    Array<PrimExpr> wshape;
    if (param->auto_scheduler_rewritten_layout.size() != 0) {
      // works for the default kernel layout "DHWIO"
      ICHECK_EQ(param->kernel_layout, "DHWIO");
      wshape = auto_scheduler::GetShapeFromRewrittenLayout(param->auto_scheduler_rewritten_layout,
                                                           {"rd", "rh", "rw", "rc", "cc"});
    } else {
      wshape = weight->shape;
    }

    wshape = trans_kernel_layout.ForwardShape(wshape);
    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 3);
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
             reporter->AssertEQ(param->kernel_size[1], wshape[3]) &&
             reporter->AssertEQ(param->kernel_size[2], wshape[4]))
          << "Conv3D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }

    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv3D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }

    if (!dshape_ncdhw[1].as<tir::AnyNode>() && !wshape[1].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(indexdiv(dshape_ncdhw[1], param->groups), wshape[1]));
    }
    channels = wshape[0];
    dilated_ksize_z = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_y = 1 + (wshape[3] - 1) * param->dilation[1];
    dilated_ksize_x = 1 + (wshape[4] - 1) * param->dilation[2];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_ncdhw[0], channels, 0, 0, 0});

  IndexExpr pad_d, pad_h, pad_w;
  GetPaddingDepthHeightWidth(param->padding, &pad_d, &pad_h, &pad_w);
  if (!dshape_ncdhw[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(dshape_ncdhw[2] + pad_d - dilated_ksize_z, param->strides[0]) + 1);
  } else {
    oshape.Set(2, dshape_ncdhw[2]);
  }

  if (!dshape_ncdhw[3].as<tir::AnyNode>()) {
    oshape.Set(3, indexdiv(dshape_ncdhw[3] + pad_h - dilated_ksize_y, param->strides[1]) + 1);
  } else {
    oshape.Set(3, dshape_ncdhw[3]);
  }

  if (!dshape_ncdhw[4].as<tir::AnyNode>()) {
    oshape.Set(4, indexdiv(dshape_ncdhw[4] + pad_w - dilated_ksize_x, param->strides[2]) + 1);
  } else {
    oshape.Set(4, dshape_ncdhw[4]);
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv3d")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, DataType out_dtype) {
      return MakeConv<Conv3DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                                   kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                                   "nn.conv3d");
    });

RELAY_REGISTER_OP("nn.conv3d")
    .describe(R"code(3D convolution layer (e.g. convolution over 3D image data,
like Magnetic Resonance Imaging (MRI) data in medicine).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 5D array of shape
            (batch_size, in_channels, depth, height, width) if `layout` is `NCDHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
- **out**:  This depends on the `layout` parameter. Output is 5D array of shape
            (batch_size, channels, out_depth, out_height, out_width) if `layout` is `NCDHW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv3DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("Conv3D", Conv3DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv3DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.conv3d_transpose
TVM_REGISTER_NODE_TYPE(Conv3DTransposeAttrs);

template <typename AttrType>
bool Conv3DTransposeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCDHW("NCDHW");
  static const Layout kIODHW("IODHW");

  const Conv3DTransposeAttrs* param = attrs.as<Conv3DTransposeAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCDHW);
  ICHECK(trans_in_layout.defined())
      << "Conv3d_transpose only support input layouts that are convertible from NCDHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kIODHW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv3d_transpose only support kernel layouts that are convertible from IODHW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCDHW);
  ICHECK(trans_out_layout.defined())
      << "Conv3d_transpose only support output layouts that are convertible from NCDHW."
      << " But got " << out_layout;

  IndexExpr channels, dilated_ksize_d, dilated_ksize_y, dilated_ksize_x;

  auto dshape_ncdhw = trans_in_layout.ForwardShape(data->shape);

  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 3);
    ICHECK_EQ(param->dilation.size(), 3);

    Array<IndexExpr> wshape({dshape_ncdhw[1], indexdiv(param->channels, param->groups),
                             param->kernel_size[0], param->kernel_size[1], param->kernel_size[2]});

    wshape = trans_kernel_layout.BackwardShape(wshape);
    dilated_ksize_d = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_y = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    dilated_ksize_x = 1 + (param->kernel_size[2] - 1) * param->dilation[2];
    channels = param->channels;

    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 3);
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
             reporter->AssertEQ(param->kernel_size[1], wshape[3]) &&
             reporter->AssertEQ(param->kernel_size[2], wshape[4]))
          << "Conv3DTransposed: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << Array<IndexExpr>(wshape);
    }
    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(indexdiv(param->channels, param->groups), wshape[1]))
          << "Conv3DTransposed: shape of weight is inconsistent out_channels, "
          << " out_channels // groups != weight.shape[1] "
          << " out_channels=" << param->channels << " groups=" << param->groups
          << " wshape=" << Array<IndexExpr>(wshape);
    }
    if (!dshape_ncdhw[1].as<tir::AnyNode>() && !wshape[0].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(dshape_ncdhw[1], wshape[0]));
    }
    channels = wshape[1];
    dilated_ksize_d = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
    dilated_ksize_y = 1 + (wshape[4] - 1) * param->dilation[2];
  }

  // dilation
  Array<IndexExpr> oshape({dshape_ncdhw[0], channels, 0, 0, 0});
  IndexExpr pad_d, pad_h, pad_w;
  GetPaddingDepthHeightWidth(param->padding, &pad_d, &pad_h, &pad_w);

  if (!dshape_ncdhw[2].as<tir::AnyNode>()) {
    oshape.Set(2, (param->strides[0] * (dshape_ncdhw[2] - 1) + dilated_ksize_d - pad_d +
                   param->output_padding[0]));
  } else {
    oshape.Set(2, dshape_ncdhw[2]);
  }
  if (!dshape_ncdhw[3].as<tir::AnyNode>()) {
    oshape.Set(3, (param->strides[1] * (dshape_ncdhw[3] - 1) + dilated_ksize_y - pad_h +
                   param->output_padding[1]));
  } else {
    oshape.Set(3, dshape_ncdhw[3]);
  }
  if (!dshape_ncdhw[4].as<tir::AnyNode>()) {
    oshape.Set(4, (param->strides[2] * (dshape_ncdhw[4] - 1) + dilated_ksize_x - pad_w +
                   param->output_padding[2]));
  } else {
    oshape.Set(4, dshape_ncdhw[4]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv3d_transpose")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, Array<IndexExpr> output_padding, DataType out_dtype) {
      return MakeConvTranspose<Conv3DTransposeAttrs>(
          data, weight, strides, padding, dilation, groups, channels, kernel_size, data_layout,
          kernel_layout, out_layout, output_padding, out_dtype, "nn.conv3d_transpose");
    });

RELAY_REGISTER_OP("nn.conv3d_transpose")
    .describe(R"code(Transposed 3D convolution layer (sometimes called Deconvolution 3D).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

- **data**: This depends on the `layout` parameter. Input is 5D array of shape
            (batch_size, in_channels, depth, height, width) if `layout` is `NCDHW`.
- **weight**: (in_channels, channels, kernel_size[0], kernel_size[1], kernel_size[2])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 5D array of shape
            (batch_size, channels, out_depth, out_height, out_width) if `layout` is `NCDHW`.

            out_depth and out_height and out_width are calculated as::
                out_depth = (depth-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
                out_height = (height-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]
                out_width = (width-1)*strides[2]-2*padding[2]+kernel_size[2]+output_padding[2]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv3DTransposeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   ConvInferCorrectLayout<Conv3DTransposeAttrs>)
    .add_type_rel("Conv3DTranspose", Conv3DTransposeRel<Conv3DTransposeAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.conv2d_transpose
TVM_REGISTER_NODE_TYPE(Conv2DTransposeAttrs);

bool Conv2DTransposeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");
  Layout kIOHW("IOHW");

  const Conv2DTransposeAttrs* param = attrs.as<Conv2DTransposeAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  bool is_dnnl_group_conv = false;
  if (param->groups > 1 && kernel_layout.name().find("G") != std::string::npos) {
    kIOHW = Layout("GIOHW");
    is_dnnl_group_conv = true;
  }

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(trans_in_layout.defined())
      << "Conv2DTransposed only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kIOHW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv2DTransposed only support kernel layouts that are convertible from " << kIOHW << "."
      << " But got " << kernel_layout << " " << kIOHW;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCHW);
  ICHECK(trans_out_layout.defined())
      << "Conv2DTransposed only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;

  auto dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 2);
    ICHECK_EQ(param->dilation.size(), 2);

    Array<IndexExpr> wshape;
    if (is_dnnl_group_conv) {
      // infer weight's shape for group convolution
      wshape = {{param->groups, indexdiv(dshape_nchw[1], param->groups),
                 indexdiv(param->channels, param->groups), param->kernel_size[0],
                 param->kernel_size[1]}};
    } else {
      // infer weight's shape for depthwise convolution
      wshape = {{dshape_nchw[1], indexdiv(param->channels, param->groups), param->kernel_size[0],
                 param->kernel_size[1]}};
    }

    wshape = trans_kernel_layout.BackwardShape(wshape);
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    channels = param->channels;

    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 2);
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
             reporter->AssertEQ(param->kernel_size[1], wshape[3]))
          << "Conv2DTransposed: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << Array<IndexExpr>(wshape);
    }
    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(indexdiv(param->channels, param->groups), wshape[1]))
          << "Conv2DTransposed: shape of weight is inconsistent with out_channels, "
          << " out_channels // groups != weight.shape[1] "
          << " out_channels=" << param->channels << " groups=" << param->groups
          << " weight.shape=" << Array<IndexExpr>(wshape);
    }
    if (!dshape_nchw[1].as<tir::AnyNode>() && !wshape[0].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(dshape_nchw[1], wshape[0]))
          << "Conv2DTransposed: shape of weight is inconsistent with in_channels."
          << " data.shape= " << Array<IndexExpr>(dshape_nchw) << " groups= " << param->groups
          << " weight.shape= " << Array<IndexExpr>(wshape);
    }
    channels = wshape[1];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});
  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  if (!dshape_nchw[2].as<tir::AnyNode>()) {
    oshape.Set(2, (param->strides[0] * (dshape_nchw[2] - 1) + dilated_ksize_y - pad_h +
                   param->output_padding[0]));
  } else {
    oshape.Set(2, dshape_nchw[2]);
  }
  if (!dshape_nchw[3].as<tir::AnyNode>()) {
    oshape.Set(3, (param->strides[1] * (dshape_nchw[3] - 1) + dilated_ksize_x - pad_w +
                   param->output_padding[1]));
  } else {
    oshape.Set(3, dshape_nchw[3]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv2d_transpose")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, Array<IndexExpr> output_padding, DataType out_dtype) {
      return MakeConvTranspose<Conv2DTransposeAttrs>(
          data, weight, strides, padding, dilation, groups, channels, kernel_size, data_layout,
          kernel_layout, out_layout, output_padding, out_dtype, "nn.conv2d_transpose");
    });

RELAY_REGISTER_OP("nn.conv2d_transpose")
    .describe(R"code(Transposed 2D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (in_channels, channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
v            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

            out_height and out_width are calculated as::
                out_height = (height-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]
                out_width = (width-1)*strides[1]-2*padding[1]+kernel_size[1]+output_padding[1]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DTransposeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   ConvInferCorrectLayout<Conv2DTransposeAttrs>)
    .add_type_rel("Conv2DTranspose", Conv2DTransposeRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.conv1d_transpose
TVM_REGISTER_NODE_TYPE(Conv1DTransposeAttrs);

bool Conv1DTransposeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCW("NCW");
  static const Layout kIOW("IOW");

  const Conv1DTransposeAttrs* param = attrs.as<Conv1DTransposeAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCW);
  ICHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kIOW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from IOW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCW);
  ICHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCW."
      << " But got " << out_layout;

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;

  auto dshape_ncw = trans_in_layout.ForwardShape(data->shape);

  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 1);
    ICHECK_EQ(param->dilation.size(), 1);

    Array<IndexExpr> wshape(
        {dshape_ncw[1], indexdiv(param->channels, param->groups), param->kernel_size[0]});

    wshape = trans_kernel_layout.BackwardShape(wshape);
    dilated_ksize_x = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    channels = param->channels;

    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 1);
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]))
          << "Conv1DTraspose: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << Array<IndexExpr>(wshape);
    }
    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(indexdiv(param->channels, param->groups), wshape[1]))
          << "Conv1DTraspose: shape of weight is inconsistent with channels, "
          << " out_channels // groups != weight.shape[1] "
          << " out_channels=" << param->channels << " groups=" << param->groups
          << " wshape=" << Array<IndexExpr>(wshape);
    }
    if (!dshape_ncw[1].as<tir::AnyNode>() && !wshape[0].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(dshape_ncw[1], wshape[0]));
    }
    channels = wshape[1];
    dilated_ksize_x = 1 + (wshape[2] - 1) * param->dilation[0];
  }
  // dilation
  IndexExpr pad_w;
  GetPaddingWidth(param->padding, &pad_w);
  Array<IndexExpr> oshape({dshape_ncw[0], channels, 0});
  if (!dshape_ncw[2].as<tir::AnyNode>()) {
    oshape.Set(2, (param->strides[0] * (dshape_ncw[2] - 1) + dilated_ksize_x - pad_w +
                   param->output_padding[0]));
  } else {
    oshape.Set(2, dshape_ncw[2]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv1d_transpose")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, Array<IndexExpr> output_padding, DataType out_dtype) {
      return MakeConvTranspose<Conv1DTransposeAttrs>(
          data, weight, strides, padding, dilation, groups, channels, kernel_size, data_layout,
          kernel_layout, out_layout, output_padding, out_dtype, "nn.conv1d_transpose");
    });

RELAY_REGISTER_OP("nn.conv1d_transpose")
    .describe(R"code(Transposed 1D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

- **data**: This depends on the `layout` parameter. Input is 3D array of shape
            (batch_size, in_channels, width) if `layout` is `NCW`.
- **weight**: (in_channels, channels, kernel_size[0])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 3D array of shape
            (batch_size, channels, out_width) if `layout` is `NCW`.

            out_width is calculated as::
                out_width = (width-1)*strides[0]-2*padding[0]+kernel_size[0]+output_padding[0]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv1DTransposeAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(2)
    .add_type_rel("Conv1DTranspose", Conv1DTransposeRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.contrib_conv2d_winograd_without_weight_transform
TVM_REGISTER_NODE_TYPE(Conv2DWinogradAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform")
    .set_body_typed([](Expr data, Expr weight, int tile_size, Array<IndexExpr> strides,
                       Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
                       IndexExpr channels, Array<IndexExpr> kernel_size, String data_layout,
                       String kernel_layout, String out_layout, DataType out_dtype) {
      return MakeConvWinograd<Conv2DWinogradAttrs>(
          data, weight, tile_size, strides, padding, dilation, groups, channels, kernel_size,
          data_layout, kernel_layout, out_layout, out_dtype,
          "nn.contrib_conv2d_winograd_without_weight_transform");
    });

RELAY_REGISTER_OP("nn.contrib_conv2d_winograd_without_weight_transform")
    .describe(R"code(Compute conv2d with winograd algorithm. Only supports NCHW layout.
                 This operator assumes the weight tensor is already pre-transformed by
                 nn.contrib_conv2d_winograd_weight_transform.

- **data**: Input is 4D array of shape  (batch_size, in_channels, height, width)
- **weight**: Any shape
            We do not check the shape for this input tensor. Since different backend
            has different layout strategy.

- **out**:  Output is 4D array of shape (batch_size, channels, out_height, out_width)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DWinogradAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv2DWinograd", Conv2DWinogradRel<Conv2DWinogradAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   ConvInferCorrectLayout<Conv2DWinogradAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.contrib_conv2d_winograd_weight_transform
TVM_REGISTER_NODE_TYPE(ConvWinogradWeightTransformAttrs);

bool Conv2DWinogradWeightTransformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const ConvWinogradWeightTransformAttrs* param = attrs.as<ConvWinogradWeightTransformAttrs>();
  ICHECK(param != nullptr);

  ICHECK_EQ(data->shape.size(), 4) << "Only support NCHW normal kernel layout";

  std::vector<IndexExpr> oshape{
      param->tile_size + data->shape[2] - 1,
      param->tile_size + data->shape[3] - 1,
      data->shape[0],
      data->shape[1],
  };

  reporter->Assign(types[1], TensorType(Array<IndexExpr>(oshape), data->dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_winograd_weight_transform")
    .set_body_typed([](Expr weight, int tile_size) {
      return MakeConvWinogradWeightTransform(weight, tile_size,
                                             "nn.contrib_conv2d_winograd_weight_transform");
    });

RELAY_REGISTER_OP("nn.contrib_conv2d_winograd_weight_transform")
    .describe(R"code(Weight transformation of winograd fast convolution algorithm.

Separate this into another operator in order to enable Precompute Pass to compute the
weight transformation in advance.

- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
)code" TVM_ADD_FILELINE)
    .set_attrs_type<ConvWinogradWeightTransformAttrs>()
    .set_num_inputs(1)
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv2DWinogradWeightTransform", Conv2DWinogradWeightTransformRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.contrib_conv3d_winograd_without_weight_transform
TVM_REGISTER_NODE_TYPE(Conv3DWinogradAttrs);

bool Conv3DWinogradRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCDHW("NCDHW");
  static const Layout kOIDHW("OIDHW");

  const auto* param = attrs.as<Conv3DWinogradAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCDHW);
  ICHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCDHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIDHW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIDHW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCDHW);
  ICHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCDHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_ncdhw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_d, dilated_ksize_y, dilated_ksize_x;

  ICHECK(param->kernel_size.defined() && param->channels.defined())
      << "The kernel size and channels of a Conv must be set or inferred by previous pass";

  ICHECK_EQ(param->kernel_size.size(), 3);
  ICHECK_EQ(param->dilation.size(), 3);

  channels = param->channels;
  dilated_ksize_d = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
  dilated_ksize_y = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
  dilated_ksize_x = 1 + (param->kernel_size[2] - 1) * param->dilation[2];

  // NOTE: Do not check weight shape here!
  // Different backend requires different layout to compute
  // the batch gemm stage in winograd efficiently, but we want to
  // make this op work for all backends.
  // So we accept all weight shapes, and assume the TOPI developers
  // can handle this correctly in alter_op_layout.

  // dilation
  Array<IndexExpr> oshape({dshape_ncdhw[0], channels, 0, 0, 0});

  IndexExpr pad_d, pad_h, pad_w;
  GetPaddingDepthHeightWidth(param->padding, &pad_d, &pad_h, &pad_w);
  if (!dshape_ncdhw[2].as<tir::AnyNode>()) {
    oshape.Set(2, (dshape_ncdhw[2] + pad_d - dilated_ksize_d) / param->strides[0] + 1);
  } else {
    oshape.Set(2, dshape_ncdhw[2]);
  }
  if (!dshape_ncdhw[2].as<tir::AnyNode>()) {
    oshape.Set(3, (dshape_ncdhw[3] + pad_h - dilated_ksize_y) / param->strides[1] + 1);
  } else {
    oshape.Set(3, dshape_ncdhw[3]);
  }
  if (!dshape_ncdhw[4].as<tir::AnyNode>()) {
    oshape.Set(4, (dshape_ncdhw[4] + pad_w - dilated_ksize_x) / param->strides[2] + 1);
  } else {
    oshape.Set(4, dshape_ncdhw[4]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform")
    .set_body_typed([](Expr data, Expr weight, int tile_size, Array<IndexExpr> strides,
                       Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
                       IndexExpr channels, Array<IndexExpr> kernel_size, String data_layout,
                       String kernel_layout, String out_layout, DataType out_dtype) {
      return MakeConvWinograd<Conv3DWinogradAttrs>(
          data, weight, tile_size, strides, padding, dilation, groups, channels, kernel_size,
          data_layout, kernel_layout, out_layout, out_dtype,
          "nn.contrib_conv3d_winograd_without_weight_transform");
    });

RELAY_REGISTER_OP("nn.contrib_conv3d_winograd_without_weight_transform")
    .describe(R"code(Compute conv3d with winograd algorithm. Only supports NCDHW layout.
              This operator assumes the weight tensor is already pre-transformed by
              nn.contrib_conv3d_winograd_weight_transform.

- **data**: Input is 5D array of shape  (batch_size, in_channels, depth, height, width)
- **weight**: Any shape
            We do not check the shape for this input tensor. Since different backend
            has different layout strategy.

- **out**:  Output is 5D array of shape (batch_size, channels, depth, out_height, out_width)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv3DWinogradAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv3DWinograd", Conv3DWinogradRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   ConvInferCorrectLayout<Conv3DWinogradAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.contrib_conv3d_winograd_weight_transform
TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv3d_winograd_weight_transform")
    .set_body_typed([](Expr weight, int tile_size) {
      return MakeConvWinogradWeightTransform(weight, tile_size,
                                             "nn.contrib_conv3d_winograd_weight_transform");
    });

bool Conv3DWinogradWeightTransformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const ConvWinogradWeightTransformAttrs* param = attrs.as<ConvWinogradWeightTransformAttrs>();
  ICHECK(param != nullptr);

  ICHECK_EQ(data->shape.size(), 5) << "Only support NCDHW normal kernel layout";

  // Shape of packed weights depends on whether depth is being transformed or not.
  Array<IndexExpr> oshape({0, 0, 0, data->shape[0], data->shape[1]});
  auto* depth_imm = data->shape[2].as<IntImmNode>();
  bool transform_depth = (depth_imm->value > 2) && (depth_imm->value < 8);
  if (transform_depth) {
    oshape.Set(0, param->tile_size + data->shape[2] - 1);
    oshape.Set(1, param->tile_size + data->shape[3] - 1);
    oshape.Set(2, param->tile_size + data->shape[4] - 1);
  } else {
    oshape.Set(0, param->tile_size + data->shape[3] - 1);
    oshape.Set(1, param->tile_size + data->shape[4] - 1);
    oshape.Set(2, data->shape[2]);
  }

  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

RELAY_REGISTER_OP("nn.contrib_conv3d_winograd_weight_transform")
    .describe(R"code(Weight transformation of winograd fast 3d convolution algorithm.

Separate this into another operator in order to enable Precompute Pass to compute the
weight transformation in advance.

- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
)code" TVM_ADD_FILELINE)
    .set_attrs_type<ConvWinogradWeightTransformAttrs>()
    .set_num_inputs(1)
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv3DWinogradWeightTransform", Conv3DWinogradWeightTransformRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.contrib_conv2d_winograd_nnpack_weight_transform
TVM_REGISTER_NODE_TYPE(Conv2DWinogradNNPACKWeightTransformAttrs);

bool Conv2DWinogradNNPACKWeightTransformRel(const Array<Type>& types, int num_inputs,
                                            const Attrs& attrs, const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }

  const Conv2DWinogradNNPACKWeightTransformAttrs* param =
      attrs.as<Conv2DWinogradNNPACKWeightTransformAttrs>();
  ICHECK(param != nullptr);

  ICHECK_EQ(data->shape.size(), 4) << "Only support NCHW normal kernel layout";

  std::vector<IndexExpr> oshape{
      data->shape[0],
      data->shape[1],
      8,
      8,
  };

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  reporter->Assign(types[1], TensorType(Array<IndexExpr>(oshape), out_dtype));
  return true;
}

Expr MakeConv2DWinogradNNPACKWeightTransform(Expr weight, int convolution_algorithm,
                                             DataType out_dtype) {
  auto attrs = make_object<Conv2DWinogradNNPACKWeightTransformAttrs>();
  attrs->convolution_algorithm = convolution_algorithm;
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("nn.contrib_conv2d_winograd_nnpack_weight_transform");
  return Call(op, {weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_winograd_nnpack_weight_transform")
    .set_body_typed(MakeConv2DWinogradNNPACKWeightTransform);

RELAY_REGISTER_OP("nn.contrib_conv2d_winograd_nnpack_weight_transform")
    .describe(R"code(Weight transformation of winograd fast convolution algorithm with NNPACK.
Separate this into another symbol in order to enable Precompute Pass to compute the
weight transformation in advance.

- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])

)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DWinogradNNPACKWeightTransformAttrs>()
    .set_num_inputs(1)
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv2DWinogradNNPACKWeightTransform", Conv2DWinogradNNPACKWeightTransformRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

// relay.nn.contrib_conv2d_gemm_without_weight_transform
TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_gemm_without_weight_transform")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, tvm::String data_layout,
                       tvm::String kernel_layout, tvm::String out_layout, DataType out_dtype) {
      return MakeConvGemm<Conv2DAttrs>(
          data, weight, strides, padding, dilation, groups, channels, kernel_size, data_layout,
          kernel_layout, out_layout, out_dtype, "nn.contrib_conv2d_gemm_without_weight_transform");
    });

bool Conv2DGemmRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNHWC("NHWC");
  static const Layout kHWIO("HWIO");

  const auto* param = attrs.as<Conv2DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNHWC);
  ICHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NHWC."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kHWIO);
  ICHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from HWIO."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNHWC);
  ICHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NHWC."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_nhwc = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;

  ICHECK(param->kernel_size.defined() && param->channels.defined())
      << "The kernel size and channels of a Conv must be set or inferred by previous pass";

  ICHECK_EQ(param->kernel_size.size(), 2);
  ICHECK_EQ(param->dilation.size(), 2);

  channels = param->channels;
  dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
  dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];

  // NOTE: Do not check weight shape here!

  // dilation
  Array<IndexExpr> oshape({dshape_nhwc[0], 0, 0, channels});

  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  if (!dshape_nhwc[2].as<tir::AnyNode>()) {
    oshape.Set(1, (dshape_nhwc[1] + pad_h - dilated_ksize_y) / param->strides[0] + 1);
  } else {
    oshape.Set(1, dshape_nhwc[1]);
  }
  if (!dshape_nhwc[3].as<tir::AnyNode>()) {
    oshape.Set(2, (dshape_nhwc[2] + pad_w - dilated_ksize_x) / param->strides[1] + 1);
  } else {
    oshape.Set(2, dshape_nhwc[2]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

RELAY_REGISTER_OP("nn.contrib_conv2d_gemm_without_weight_transform")
    .describe(R"code(Compute conv2d with gemm algorithm. Only supports NHWC layout.
                 This operator assumes the weight tensor is already pre-transformed by
                 nn.contrib_conv2d_gemm_weight_transform.

- **data**: Input is 4D array of shape  (batch_size, height, width, in_channels)
- **weight**: Any shape
            We do not check the shape for this input tensor. Since different backend
            has different layout strategy.

- **out**:  Output is 4D array of shape (batch_size, channels, out_height, out_width)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv2DGemm", Conv2DGemmRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// relay.nn.contrib_conv2d_gemm_weight_transform

TVM_REGISTER_NODE_TYPE(ConvGemmWeightTransformAttrs);

// Gemm convolution shape relations
// In order to run GEMM we need to transform the K x N weights matrix W.
//
// For integer datatypes, the high level idea is to subdivide W in tiles of tile_K x tile_N, and
// transpose and interleave them. The final output is a [N//tile_N, K//tile_K, tile_N, tile_K]
// matrix that we call W_interleaved_t.
//
// In the following picture, we show how the first [tile_K,tile_N] block of W is transformed
// for tile_N = 4 and tile_K = 16
//
//              W[0,0,:,:]                        W_interleaved_t[0,0,:,:]
//  +-------------------------------+     +----------------------------------- +
//  |W[0,0]  W[0,1]  W[0,2]  W[0,3] |     |W[0,0]  W[1,0]  W[2,0]  ...  W[15,0]|
//  |W[1,0]  W[1,1]  W[1,2]  W[1,3] | --\ |W[0,1]  W[1,1]  W[2,1]  ...  W[15,1]|
//  |W[2,0]  W[2,1]  W[2,2]  W[2,3] | --/ |W[0,2]  W[1,2]  W[2,2]  ...  W[15,2]|
//  |  ...     ...    ...      ...  |     |W[0,3]  W[1,3]  W[2,3]  ...  W[15,3]|
//  |  ...     ...    ...      ...  |     +------------------------------------+
//  |W[15,0] W[15,1] W[15,2] W[15,3]|
//  +-------------------------------+
//
// Alternatively, for floating point datatypes, we subdivide W in tiles of tile_K x tile_N size,
// then interleave these tiles, without transposing. The final output is a [N//tile_N, K//tile_K,
// tile_K, tile_N] matrix called W_interleaved.
//
// In the following illustration, we show how the tiles are interleaved.
// Note that the inside of each tile is kept unchanged during this tranformation.
//
//           W[:,:,:,:]               W_interleaved[:,:,:,:]
//  +--------+--------+--------+       +--------+--------+
//  |        |        |        |       |        |        |
//  | tile_1 | tile_2 | tile_3 |       | tile_1 | tile_4 |
//  |        |        |        |  --\  |        |        |
//  +--------+--------+--------+  --/  +--------+--------+
//  |        |        |        |       |        |        |
//  | tile_4 | tile_5 | tile_6 |       | tile_2 | tile_5 |
//  |        |        |        |       |        |        |
//  +--------+--------+--------+       +--------+--------+
//                                     |        |        |
//                                     | tile_3 | tile_6 |
//                                     |        |        |
//                                     +--------+--------+
//
// Tile K is the direction of the reduction in both cases. So, if our target can reduce k elements
// at the time, we should set tile_K = k.
// Tile N is connected with the number of registers available for the given target.
//
bool Conv2DGemmWeightTransformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* weight = types[0].as<TensorTypeNode>();
  if (weight == nullptr) return false;

  const ConvGemmWeightTransformAttrs* param = attrs.as<ConvGemmWeightTransformAttrs>();
  ICHECK(param != nullptr);
  int n = param->tile_N;
  int k = param->tile_K;

  ICHECK_EQ(weight->shape.size(), 4) << "Only support HWIO kernel layout";

  const auto K = weight->shape[0] * weight->shape[1] * weight->shape[2];
  const auto N = weight->shape[3];

  auto K_mod_k = indexmod(K, k * 4);
  auto N_mod_n = indexmod(N, n);

  auto pad_K = tvm::if_then_else(K_mod_k != 0, k * 4 - K_mod_k, tir::make_zero(DataType::Int(32)));
  auto pad_N = tvm::if_then_else(N_mod_n != 0, n - N_mod_n, tir::make_zero(DataType::Int(32)));

  const auto N_padded = N + pad_N;
  const auto K_padded = K + pad_K;

  Array<IndexExpr> oshape;
  if (weight->dtype.bits() == 8 && (weight->dtype.is_int() || weight->dtype.is_uint()))
    oshape = {
        indexdiv(N_padded, n),
        indexdiv(K_padded, k),
        n,
        k,
    };
  else
    oshape = {
        indexdiv(N_padded, n),
        indexdiv(K_padded, k),
        k,
        n,
    };

  reporter->Assign(types[1], TensorType(oshape, weight->dtype));
  return true;
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_gemm_weight_transform")
    .set_body_typed([](Expr weights, int tile_rows, int tile_cols) {
      return MakeConvGemmWeightTransform(weights, tile_rows, tile_cols,
                                         "nn.contrib_conv2d_gemm_weight_transform");
    });

RELAY_REGISTER_OP("nn.contrib_conv2d_gemm_weight_transform")
    .describe(R"code(Weight transformation of GEMM convolution algorithm.

Separate this into another operator in order to enable Precompute Pass to compute the
weight transformation in advance.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ConvGemmWeightTransformAttrs>()
    .set_num_inputs(1)
    .add_argument("weights", "Tensor", "The weights tensor.")
    .set_support_level(10)
    .add_type_rel("Conv2DGemmWeightTransform", Conv2DGemmWeightTransformRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// Positional relay function to create conv2d NCHWc operator
// used by frontend FFI.
TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_NCHWc")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, DataType out_dtype) {
      return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                                   kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                                   "nn.contrib_conv2d_NCHWc");
    });

RELAY_REGISTER_OP("nn.contrib_conv2d_NCHWc")
    .describe(R"code(Compute conv2d with NCHWc data layout. Only supports NCHW layout.
- **data**: Input is 5D packed tensor.
- **weight**: 6D packed tensor.

- **out**:  Output is 5D packed tensor
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv2DNCHWc", Conv2DWinogradRel<Conv2DAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// Positional relay function to create depthwise conv2d NCHWc operator
// used by frontend FFI.
TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_depthwise_conv2d_NCHWc")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String data_layout, String kernel_layout,
                       String out_layout, DataType out_dtype) {
      return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                                   kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                                   "nn.contrib_depthwise_conv2d_NCHWc");
    });

RELAY_REGISTER_OP("nn.contrib_depthwise_conv2d_NCHWc")
    .describe(R"code(Compute conv2d with NCHWc data layout. Only supports NCHW layout.
- **data**: Input is 5D packed tensor.
- **weight**: 6D packed tensor.

- **out**:  Output is 5D packed tensor
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(10)
    .add_type_rel("Conv2D", Conv2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

TVM_REGISTER_NODE_TYPE(DeformableConv2DAttrs);

// Deformable Convolution shape relations.
bool DeformableConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[2].as<TensorTypeNode>();

  ICHECK(data);
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  auto* param = attrs.as<DeformableConv2DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  if (!trans_in_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "deformable_conv2d only support input layouts that are convertible from NCHW."
        << " The provided layout is: " << in_layout);
    return false;
  }

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIHW);
  if (!trans_kernel_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "deformable_conv2d only support kernel layouts that are convertible from OIHW."
        << " The provided layout is: " << kernel_layout);
    return false;
  }

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCHW);
  if (!trans_out_layout.defined()) {
    reporter->GetDiagCtx().Emit(
        Diagnostic::Error(reporter->GetSpan())
        << "deformable_conv2d only support output layouts that are convertible from NCHW."
        << "The provided layout is: " << out_layout);
    return false;
  }

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x, ksize_y, ksize_x;

  // infer weight shape if kernel_size and channels are defiend
  if (param->kernel_size.defined() && param->channels.defined()) {
    ICHECK_EQ(param->kernel_size.size(), 2);
    ICHECK_EQ(param->dilation.size(), 2);
    Array<IndexExpr> wshape({param->channels, indexdiv(dshape_nchw[1], param->groups),
                             param->kernel_size[0], param->kernel_size[1]});

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    ksize_y = param->kernel_size[0];
    ksize_x = param->kernel_size[1];
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    // assign result to reporter
    reporter->Assign(types[2], TensorType(wshape, data->dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);

    if (param->kernel_size.defined()) {
      ICHECK_EQ(param->kernel_size.size(), 2);
      // check the size
      ICHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
             reporter->AssertEQ(param->kernel_size[1], wshape[3]))
          << "DeformableConv2D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      ICHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "DeformableConv2D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }
    if (!dshape_nchw[1].as<tir::AnyNode>() && !wshape[1].as<tir::AnyNode>()) {
      ICHECK(reporter->AssertEQ(indexdiv(dshape_nchw[1], param->groups), wshape[1]));
    }
    channels = wshape[0];
    ksize_y = wshape[2];
    ksize_x = wshape[3];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  oshape.Set(2, indexdiv(dshape_nchw[2] + pad_h - dilated_ksize_y, param->strides[0]) + 1);
  oshape.Set(3, indexdiv(dshape_nchw[3] + pad_w - dilated_ksize_x, param->strides[1]) + 1);
  DataType out_dtype = param->out_dtype;

  // infer offset shape
  Array<IndexExpr> offset_shape(
      {dshape_nchw[0], 2 * ksize_y * ksize_x * param->deformable_groups, oshape[2], oshape[3]});
  offset_shape = trans_in_layout.BackwardShape(offset_shape);
  reporter->Assign(types[1], TensorType(offset_shape, data->dtype));
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  oshape = trans_out_layout.BackwardShape(oshape);
  reporter->Assign(types[3], TensorType(oshape, out_dtype));
  return true;
}

InferCorrectLayoutOutput DeformableConvInferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  const auto* params = attrs.as<DeformableConv2DAttrs>();
  return InferCorrectLayoutOutput(
      {params->data_layout, params->data_layout, params->kernel_layout},
      {params->out_layout == "" ? params->data_layout : params->out_layout}, attrs);
}

RELAY_REGISTER_OP("nn.deformable_conv2d")
    .describe(R"code(Compute 2-D deformable convolution on 4-D input.
The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

For 2-D deformable convolution, the shapes are
- **data**: (batch_size, channel, height, width)
- **offset**: (batch_size, deformable_groups * kernel[0] * kernel[1] * 2, out_height, out_width)
- **weight**: (num_filter, channel, kernel[0], kernel[1])
- **out**: (batch_size, num_filter, out_height, out_width).

If `deformable_groups` is larger than 1, denoted by *dg*, then split the
input `offset` evenly into *dg* parts along the channel axis, and also evenly split `out`
evenly into *dg* parts along the channel axis. Next compute the deformable convolution, apply the
*i*-th part of the offset part on the *i*-th out.

If `groups` is larger than 1, denoted by *g*, then split the input `data` evenly into *g* parts
along the channel axis, and also evenly split `weight` along the first dimension. Next compute
the convolution on the *i*-th part of the data with the *i*-th weight part. The output is obtained
by concating all the *g* results.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<DeformableConv2DAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("offset", "Tensor", "The offset tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_support_level(5)
    .add_type_rel("DeformableConv2D", DeformableConv2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", DeformableConvInferCorrectLayout)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

// Positional relay function to create deformable_conv2d operator
// used by frontend FFI.
TVM_REGISTER_GLOBAL("relay.op.nn._make.deformable_conv2d")
    .set_body_typed([](Expr data, Expr offset, Expr weight, Array<IndexExpr> strides,
                       Array<IndexExpr> padding, Array<IndexExpr> dilation, int deformable_groups,
                       int groups, int channels, Array<IndexExpr> kernel_size, String data_layout,
                       String kernel_layout, String out_layout, DataType out_dtype) {
      return MakeDeformableConv<DeformableConv2DAttrs>(
          data, offset, weight, strides, padding, dilation, deformable_groups, groups, channels,
          kernel_size, data_layout, kernel_layout, out_layout, out_dtype, "nn.deformable_conv2d");
    });

inline Expr MakeConv2dBackwardWeight(Expr grad, Expr data, Array<IndexExpr> strides,
                                     Array<IndexExpr> padding, Array<IndexExpr> dilation,
                                     int groups, IndexExpr channels, Array<IndexExpr> kernel_size,
                                     std::string grad_layout, std::string data_layout,
                                     std::string kernel_layout, DataType out_dtype) {
  auto attrs = make_object<Conv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->out_dtype = std::move(out_dtype);
  attrs->data_layout = std::move(grad_layout);
  attrs->kernel_layout = std::move(data_layout);
  attrs->out_layout = std::move(kernel_layout);
  const Op& op = Op::Get("nn.conv2d_backward_weight");
  return Call(op, {grad, data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv2d_backward_weight")
    .set_body_typed([](Expr grad, Expr data, Array<IndexExpr> strides, Array<IndexExpr> padding,
                       Array<IndexExpr> dilation, int groups, IndexExpr channels,
                       Array<IndexExpr> kernel_size, String grad_layout, String data_layout,
                       String kernel_layout, DataType out_dtype) {
      return MakeConv2dBackwardWeight(grad, data, strides, padding, dilation, groups, channels,
                                      kernel_size, grad_layout, data_layout, kernel_layout,
                                      out_dtype);
    });

bool Conv2DBackwardWeightRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                             const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* grad = types[0].as<TensorTypeNode>();
  const auto* data = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const auto* param = attrs.as<Conv2DAttrs>();
  ICHECK(param != nullptr);
  // Require kernel_size to be passed, to simplify the output shape determination.
  ICHECK(param->kernel_size.defined()) << "kernel_size attribute needs to be specified";

  // We repurpose Conv2dAttrs for Conv2DBackwardWeight, note the meanings of layouts.
  const Layout grad_layout(param->data_layout);
  const Layout in_layout(param->kernel_layout);
  const Layout kernel_layout(param->out_layout);

  const auto trans_grad_layout = tir::BijectiveLayout(grad_layout, kNCHW);
  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIHW);

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);
  Array<IndexExpr> grad_shape_nchw = trans_grad_layout.ForwardShape(grad->shape);

  auto in_channels = dshape_nchw[1];
  auto out_channels = grad_shape_nchw[1];

  auto in_channels_intimm = in_channels.as<IntImmNode>();
  auto out_channels_intimm = out_channels.as<IntImmNode>();
  ICHECK(in_channels_intimm);
  ICHECK(out_channels_intimm);

  IndexExpr weight_dim_i;
  if (in_channels_intimm->value == out_channels_intimm->value &&
      in_channels_intimm->value == param->groups) {
    // depthwise
    ICHECK(param->channels.defined())
        << "out_channels attribute not specified for depth wise conv2d.";
    weight_dim_i = indexdiv(param->channels, param->groups);
  } else {
    weight_dim_i = indexdiv(in_channels, param->groups);
  }

  Array<IndexExpr> wshape_oihw{out_channels, weight_dim_i, param->kernel_size[0],
                               param->kernel_size[1]};
  auto wshape = trans_kernel_layout.BackwardShape(wshape_oihw);

  const auto dw_dtype = (param->out_dtype == DataType() || param->out_dtype.is_void())
                            ? grad->dtype
                            : param->out_dtype;

  reporter->Assign(types[2], TensorType(wshape, dw_dtype));
  return true;
}

RELAY_REGISTER_OP("nn.conv2d_backward_weight")
    .describe(R"code(The gradient of the 2D convolution layer with respect to the weight.

This layer computes the gradient of the conv2d op with respect to weight,
given the original input data and the output gradient.

- **grad**: (batch, channels, out_height, out_width) if `layout` is `NCHW`.
- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (channels, in_channels, kernel_size[0], kernel_size[1]) if `layout` is `NCHW`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DAttrs>()
    .set_num_inputs(2)
    .add_argument("grad", "Tensor", "The gradient tensor.")
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("Conv2DBackwardWeight", Conv2DBackwardWeightRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

}  // namespace relay
}  // namespace tvm
