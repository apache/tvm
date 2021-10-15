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
 * \file src/relay/op/contrib/ethosu/depthwise.cc
 * \brief Depthwise convolution 2D operator definition for the Arm(R) Ethos(TM)-U NPU
 */
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_layout.h>

#include "../../../qnn/utils.h"
#include "../../nn/convolution.h"
#include "common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

/*! \brief Attributes used by the Ethos(TM)-U NPU depthwise operator */
struct EthosuDepthwiseConv2DAttrs : public tvm::AttrsNode<EthosuDepthwiseConv2DAttrs> {
  double ifm_scale;
  int ifm_zero_point;
  int weight_zero_point;
  double ofm_scale;
  int ofm_zero_point;
  Array<IndexExpr> kernel_shape;
  IndexExpr ofm_channels;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  String activation;
  int clip_min;
  int clip_max;
  String upscale;
  String ifm_layout;
  String ofm_layout;

  TVM_DECLARE_ATTRS(EthosuDepthwiseConv2DAttrs, "relay.attrs.EthosuDepthwiseConv2DAttrs") {
    TVM_ATTR_FIELD(ifm_scale).describe("The quantization scale for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(weight_zero_point)
        .describe("The quantization zero point for the weight tensor.");
    TVM_ATTR_FIELD(ofm_scale).describe("The quantization scale for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(kernel_shape)
        .describe("The 2 dimensional kernel shape as (kernel_height, kernel_width).")
        .set_default(NullValue<Array<IndexExpr> >());
    TVM_ATTR_FIELD(ofm_channels)
        .describe("The number of OFM channels.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(strides)
        .describe("The 2 dimensional strides as (stride_height, stride_width).")
        .set_default(Array<IndexExpr>({1, 1}));
    TVM_ATTR_FIELD(padding)
        .describe("The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right)")
        .set_default(Array<IndexExpr>({0, 0, 0, 0}));
    TVM_ATTR_FIELD(dilation)
        .describe("The 2 dimensional dilation as (dilation_height, dilation_width).")
        .set_default(Array<IndexExpr>({1, 1}));
    TVM_ATTR_FIELD(activation)
        .describe(
            "Description: The activation function to use."
            "'NONE' - no activation function."
            "'CLIP' - clip the output between clip_min and clip_max."
            "'TANH - tanh activation function."
            "'SIGMOID' - sigmoid activation function."
            "'LUT' - use a look-up table to perform the activation function.")
        .set_default("NONE");
    TVM_ATTR_FIELD(clip_min)
        .describe("The minimum clipping value if activation = CLIP.")
        .set_default(0);
    TVM_ATTR_FIELD(clip_max)
        .describe("The maximum clipping value if activation = CLIP.")
        .set_default(0);
    TVM_ATTR_FIELD(upscale)
        .describe(
            "The 2x2 upscaling mode to apply to the Input Feature Map tensor. "
            "'NONE' - no upscaling. "
            "'NEAREST' - upscale using nearest neighbour. "
            "'ZEROS' - upscale using zeros.")
        .set_default("NONE");
    TVM_ATTR_FIELD(ifm_layout)
        .set_default("NHWC")
        .describe("The layout of the Input Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.");
    TVM_ATTR_FIELD(ofm_layout)
        .set_default("NHWC")
        .describe("The layout of the Output Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.");
  }
};

TVM_REGISTER_NODE_TYPE(EthosuDepthwiseConv2DAttrs);

bool EthosuDepthwiseConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                              const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 5);
  const auto* ifm = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  const auto* scale_bias = types[2].as<TensorTypeNode>();
  if (ifm == nullptr || weight == nullptr) return false;

  const auto* param = attrs.as<EthosuDepthwiseConv2DAttrs>();
  ICHECK(param != nullptr) << "EthosuDepthwiseConv2DAttrs cannot be nullptr.";
  ICHECK(ifm->dtype == DataType::UInt(8) || ifm->dtype == DataType::Int(8))
      << "Expected ethosu_depthwise_conv2d type(uint8) or type(int8) for ifm but was "
      << ifm->dtype;
  ICHECK(weight->dtype == DataType::UInt(8) || weight->dtype == DataType::Int(8))
      << "Expected ethosu_depthwise_conv2d type(uint8) or type(int8) for weight but was "
      << weight->dtype;
  ICHECK(scale_bias->dtype == DataType::UInt(8))
      << "Expected ethosu_depthwise_conv2d type(uint8) for scale_bias but was "
      << scale_bias->dtype;

  // Collect the ifm, weight and ofm tensors for using in the inference function
  Array<Type> tensor_types = {types[0], types[1], types[4]};

  // Assign weight type {ofm_channels, kernel_height, kernel_width, 1}
  reporter->Assign(types[1], TensorType({param->ofm_channels, param->kernel_shape[0],
                                         param->kernel_shape[1], weight->shape[3]},
                                        weight->dtype));

  // Assign ofm type
  auto ofm_shape =
      EthosuInferKernelOutput(ifm->shape, param->ifm_layout, param->ofm_layout, param->kernel_shape,
                              param->ofm_channels, param->dilation, param->strides, param->padding);

  reporter->Assign(types[4], TensorType(ofm_shape, ifm->dtype));

  return true;
}

Expr MakeEthosuDepthwiseConv2D(Expr ifm, Expr weight, Expr scale_bias, Expr lut, double ifm_scale,
                               int ifm_zero_point, int weight_zero_point, double ofm_scale,
                               int ofm_zero_point, Array<IndexExpr> kernel_shape,
                               IndexExpr ofm_channels, Array<IndexExpr> strides,
                               Array<IndexExpr> padding, Array<IndexExpr> dilation,
                               String activation, int clip_min, int clip_max, String upscale,
                               String ifm_layout, String ofm_layout) {
  auto attrs = make_object<EthosuDepthwiseConv2DAttrs>();
  attrs->ifm_scale = ifm_scale;
  attrs->ifm_zero_point = ifm_zero_point;
  attrs->weight_zero_point = weight_zero_point;
  attrs->ofm_scale = ofm_scale;
  attrs->ofm_zero_point = ofm_zero_point;
  attrs->kernel_shape = std::move(kernel_shape);
  attrs->ofm_channels = std::move(ofm_channels);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->activation = std::move(activation);
  attrs->clip_min = clip_min;
  attrs->clip_max = clip_max;
  attrs->upscale = std::move(upscale);
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  static const Op& op = Op::Get("contrib.ethosu.depthwise_conv2d");
  return Call(op, {ifm, weight, scale_bias, lut}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ethosu_depthwise_conv2d")
    .set_body_typed(MakeEthosuDepthwiseConv2D);

RELAY_REGISTER_OP("contrib.ethosu.depthwise_conv2d")
    .describe(R"code(Arm(R) Ethos(TM)-U NPU 2D quantized depthwise operator.

This Relay operator corresponds to the hardware-implemented quantized
depthwise operation found on Ethos(TM)-U NPUs. It accepts either NHWC or NHCWB16 format
for the input data (input feature map, or IFM) and OHWI format for the kernel weights.

- **ifm**: NHWC - (1, ifm_height, ifm_width, ifm_channels)
           NHCWB16 - (1, ifm_height, ifm_channels // 16, ifm_width, 16)
- **weight**: (ofm_channels, kernel_shape[0], kernel_shape[1], 1 (depth multiplier))
- **scale_bias**: (ofm_channels, 10)
- **ofm**: (1, ofm_height, ofm_width, ofm_channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EthosuDepthwiseConv2DAttrs>()
    .set_num_inputs(4)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .add_argument("scale_bias", "Tensor", "The packed per-channel weight scale and bias tensor.")
    .add_argument("lut", "Tensor", "The look-up table values to use if activation = 'LUT'")
    .set_support_level(11)
    .add_type_rel("EthosuDepthwiseConv2D", EthosuDepthwiseConv2DRel);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
