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
 * \file src/relay/op/contrib/ethosu/op_attrs.h
 * \brief Attributes for the Arm(R) Ethos(TM)-U NPU operators.
 */

#ifndef TVM_RELAY_OP_CONTRIB_ETHOSU_OP_ATTRS_H_
#define TVM_RELAY_OP_CONTRIB_ETHOSU_OP_ATTRS_H_

#include <tvm/relay/op.h>

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

/*! \brief Attributes used by the Ethos(TM)-U NPU binary elementwise operators */
struct EthosuBinaryElementwiseAttrs : public tvm::AttrsNode<EthosuBinaryElementwiseAttrs> {
  String operator_type;
  double ifm_scale;
  int ifm_zero_point;
  double ifm2_scale;
  int ifm2_zero_point;
  double ofm_scale;
  int ofm_zero_point;
  IndexExpr ifm_channels;
  IndexExpr ifm2_channels;
  bool reversed_operands;
  String activation;
  int clip_min;
  int clip_max;
  String rounding_mode;
  String ifm_layout;
  String ifm2_layout;
  String ofm_layout;
  String ofm_dtype;
  bool use_rescale;
  int rescale_scale;
  int rescale_shift;

  TVM_DECLARE_ATTRS(EthosuBinaryElementwiseAttrs, "relay.attrs.EthosuBinaryElementwiseAttrs") {
    TVM_ATTR_FIELD(operator_type)
        .describe(
            "The type of the binary elementwise operator."
            "'ADD'"
            "'SUB'"
            "'MUL'"
            "'MIN'"
            "'MAX'"
            "'SHR'"
            "'SHL'");
    TVM_ATTR_FIELD(ifm_scale).describe("The quantization scale for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_zero_point)
        .describe("The quantization zero point for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm2_scale)
        .describe("The quantization scale for the Input Feature Map tensor 2.");
    TVM_ATTR_FIELD(ifm2_zero_point)
        .describe("The quantization zero point for the Input Feature Map tensor 2.");
    TVM_ATTR_FIELD(ofm_scale).describe("The quantization scale for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_channels).describe("The number of the Input Feature Map channels.");
    TVM_ATTR_FIELD(ifm2_channels).describe("The number of the Input Feature Map 2 channels.");
    TVM_ATTR_FIELD(reversed_operands)
        .describe("True if IFM2 is the first operand and IFM is the second operand.")
        .set_default(false);
    TVM_ATTR_FIELD(activation)
        .describe(
            "The activation function to use. "
            "'NONE' - no activation function. "
            "'CLIP' - clip the output between clip_min and clip_max. "
            "'TANH' - tanh activation function. "
            "'SIGMOID' - sigmoid activation function. "
            "'LUT' - use a look-up table to perform the activation function."
            "Available activations for activation type:"
            "{int8, uint8}: 'NONE', 'CLIP', 'TANH', 'SIGMOID', 'LUT'"
            "{int32}: 'NONE'")
        .set_default("NONE");
    TVM_ATTR_FIELD(clip_min)
        .describe("The minimum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(clip_max)
        .describe("The maximum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(rounding_mode)
        .describe(
            "The rounding mode to apply to the Output Feature Map tensor. "
            "'TFL' - Tensorflow Lite rounding scheme. "
            "'TRUNCATE' - Truncate towards zero."
            "'NATURAL' - Round to nearest value, with x.5 rounded up towards +infinity.")
        .set_default("TFL");
    TVM_ATTR_FIELD(ifm_layout)
        .describe("The layout of the Input Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
    TVM_ATTR_FIELD(ifm2_layout)
        .describe("The layout of the Input Feature Map tensor 2. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
    TVM_ATTR_FIELD(ofm_layout)
        .describe("The layout of the Output Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
    TVM_ATTR_FIELD(ofm_dtype).describe(
        "The Output Feature Map tensor type."
        "MUL, ADD, SUB {IFM}->{OFM}:"
        "  {uint8, int8 int32} -> {uint8, int8, int32}, any pairing"
        "MAX, MIN:"
        "  IFM and OFM must be of the same type, one of:"
        "  {int8, uint8}"
        "SHR {IFM}->{OFM}:"
        "  {int32}->{int8, uint8, int32}, any pairing"
        "SHL:"
        "  {int32}->{int32} only");
    TVM_ATTR_FIELD(use_rescale).describe("Use explicit scaling if True.").set_default(false);
    TVM_ATTR_FIELD(rescale_scale)
        .describe(
            "Scale value for rescale. "
            "For 32-bit operations scale is not applied but shift is.")
        .set_default(0);
    TVM_ATTR_FIELD(rescale_shift).describe("Shift value for rescale.").set_default(0);
  }
};

TVM_REGISTER_NODE_TYPE(EthosuBinaryElementwiseAttrs);

/*! \brief Attributes used by the Ethos(TM)-U NPU convolution operator */
struct EthosuConv2DAttrs : public tvm::AttrsNode<EthosuConv2DAttrs> {
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
  String rounding_mode;
  String upscale;
  String ifm_layout;
  String ofm_layout;

  TVM_DECLARE_ATTRS(EthosuConv2DAttrs, "relay.attrs.EthosuConv2DAttrs") {
    TVM_ATTR_FIELD(ifm_scale).describe("The quantization scale for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_zero_point)
        .describe("The quantization zero point for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(weight_zero_point)
        .describe("The quantization zero point for the weight tensor.");
    TVM_ATTR_FIELD(ofm_scale).describe("The quantization scale for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(kernel_shape)
        .describe("The 2 dimensional kernel shape as (kernel_height, kernel_width).")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(ofm_channels)
        .describe("The number of the Output Feature Map channels.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The 2 dimensional strides as (stride_height, stride_width).");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0, 0}))
        .describe("The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The 2 dimensional dilation as (dilation_height, dilation_width).");
    TVM_ATTR_FIELD(activation)
        .describe(
            "The activation function to use. "
            "'NONE' - no activation function. "
            "'CLIP' - clip the output between clip_min and clip_max. "
            "'TANH' - tanh activation function. "
            "'SIGMOID' - sigmoid activation function. "
            "'LUT' - use a look-up table to perform the activation function.")
        .set_default("NONE");
    TVM_ATTR_FIELD(clip_min)
        .describe("The minimum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(clip_max)
        .describe("The maximum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(rounding_mode)
        .describe(
            "The rounding mode to apply to the Output Feature Map tensor. "
            "'TFL' - Tensorflow Lite rounding scheme. "
            "'TRUNCATE' - Truncate towards zero."
            "'NATURAL' - Round to nearest value, with x.5 rounded up towards +infinity.")
        .set_default("TFL");
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

TVM_REGISTER_NODE_TYPE(EthosuConv2DAttrs);

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
  String rounding_mode;
  String upscale;
  String ifm_layout;
  String ofm_layout;
  String ofm_dtype;

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
        .set_default(NullValue<Array<IndexExpr>>());
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
    TVM_ATTR_FIELD(rounding_mode)
        .describe(
            "The rounding mode to apply to the Output Feature Map tensor. "
            "'TFL' - Tensorflow Lite rounding scheme. "
            "'TRUNCATE' - Truncate towards zero."
            "'NATURAL' - Round to nearest value, with x.5 rounded up towards +infinity.")
        .set_default("TFL");
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
    TVM_ATTR_FIELD(ofm_dtype)
        .describe("The Output Feature Map tensor data type. Can be 'int8', 'uint8' or 'int16'.")
        .set_default("int8");
  }
};

TVM_REGISTER_NODE_TYPE(EthosuDepthwiseConv2DAttrs);

/*! \brief Attributes used by the NPU identity operator */
struct EthosuIdentityAttrs : public tvm::AttrsNode<EthosuIdentityAttrs> {
  double ifm_scale;
  int ifm_zero_point;
  double ofm_scale;
  int ofm_zero_point;
  String activation;
  String rounding_mode;

  TVM_DECLARE_ATTRS(EthosuIdentityAttrs, "relay.attrs.EthosuIdentityAttrs") {
    TVM_ATTR_FIELD(ifm_scale).describe("The quantization scale for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_zero_point)
        .describe("The quantization zero point for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_scale).describe("The quantization scale for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(activation)
        .describe(
            "The activation function to use. "
            "'NONE' - no activation function. "
            "'TANH' - tanh activation function. "
            "'SIGMOID' - sigmoid activation function. "
            "'LUT' - use a look-up table to perform the activation function.")
        .set_default("NONE");
    TVM_ATTR_FIELD(rounding_mode)
        .describe(
            "The rounding mode to apply to the Output Feature Map tensor. "
            "'TFL' - Tensorflow Lite rounding scheme. "
            "'TRUNCATE' - Truncate towards zero."
            "'NATURAL' - Round to nearest value, with x.5 rounded up towards +infinity.")
        .set_default("TFL");
  }
};

TVM_REGISTER_NODE_TYPE(EthosuIdentityAttrs);

/*! \brief Attributes used by the Ethos(TM)-U NPU pooling operator */
struct EthosuPoolingAttrs : public tvm::AttrsNode<EthosuPoolingAttrs> {
  String pooling_type;
  double ifm_scale;
  int ifm_zero_point;
  double ofm_scale;
  int ofm_zero_point;
  Array<IndexExpr> pool_shape;
  IndexExpr ofm_channels;
  String ofm_dtype;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  String activation;
  int clip_min;
  int clip_max;
  String rounding_mode;
  String upscale;
  String ifm_layout;
  String ofm_layout;

  TVM_DECLARE_ATTRS(EthosuPoolingAttrs, "relay.attrs.EthosuPoolingAttrs") {
    TVM_ATTR_FIELD(pooling_type)
        .describe(
            "The type of the pooling. 'AVG' - average pool, 'MAX' - max pool, "
            "'SUM' - reduce sum pool.");
    TVM_ATTR_FIELD(ifm_scale).describe("The quantization scale for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_zero_point)
        .describe("The quantization zero point for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_scale).describe("The quantization scale for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(pool_shape)
        .describe("The 2 dimensional pool shape as (pool_shape_height, pool_shape_width).")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(ofm_channels)
        .describe(" The number of the Output Feature Map channels.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(ofm_dtype).describe(
        "The Output Feature Map tensor data type. "
        "'AVG' or 'MAX' pooling - can be 'int8', 'uint8', or 'int16'. "
        "'SUM' pooling - can be 'int32'.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The 2 dimensional strides as (stride_height, stride_width).");
    TVM_ATTR_FIELD(padding)
        .describe("The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).")
        .set_default(Array<IndexExpr>({0, 0, 0, 0}));
    TVM_ATTR_FIELD(activation)
        .describe(
            "The activation function to use. "
            "'NONE' - no activation function. "
            "'CLIP' - clip the output between clip_min and clip_max. "
            "'TANH' - tanh activation function. "
            "'SIGMOID' - sigmoid activation function. "
            "'LUT' - use a look-up table to perform the activation function.")
        .set_default("NONE");
    TVM_ATTR_FIELD(clip_min)
        .describe("The minimum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(clip_max)
        .describe("The maximum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(rounding_mode)
        .describe(
            "The rounding mode to apply to the Output Feature Map tensor. "
            "'TFL' - Tensorflow Lite rounding scheme. "
            "'TRUNCATE' - Truncate towards zero."
            "'NATURAL' - Round to nearest value, with x.5 rounded up towards +infinity.")
        .set_default("TFL");
    TVM_ATTR_FIELD(upscale)
        .describe(
            "The 2x2 upscaling mode to apply to the Input Feature Map tensor. "
            "'NONE' - no upscaling. "
            "'NEAREST' - upscale using nearest neighbour. "
            "'ZEROS' - upscale using zeros.")
        .set_default("NONE");
    TVM_ATTR_FIELD(ifm_layout)
        .describe("The layout of the Input Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
    TVM_ATTR_FIELD(ofm_layout)
        .describe("The layout of the Output Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
  }
};

TVM_REGISTER_NODE_TYPE(EthosuPoolingAttrs);

/*! \brief Attributes used by the NPU unary elementwise operator */
struct EthosuUnaryElementwiseAttrs : public tvm::AttrsNode<EthosuUnaryElementwiseAttrs> {
  String operator_type;
  double ifm_scale;
  int ifm_zero_point;
  double ofm_scale;
  int ofm_zero_point;
  IndexExpr ofm_channels;
  String activation;
  int clip_min;
  int clip_max;
  String rounding_mode;
  String ifm_layout;
  String ofm_layout;

  TVM_DECLARE_ATTRS(EthosuUnaryElementwiseAttrs, "relay.attrs.EthosuUnaryElementwiseAttrs") {
    TVM_ATTR_FIELD(operator_type)
        .describe(
            "The type of the unary elementwise operator."
            "'ABS'"
            "'CLZ'");
    TVM_ATTR_FIELD(ifm_scale).describe("The quantization scale for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_zero_point)
        .describe("The quantization zero point for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_scale).describe("The quantization scale for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_channels).describe("The number of OFM channels.");
    TVM_ATTR_FIELD(activation)
        .describe(
            "The activation function to use. "
            "'NONE' - no activation function. "
            "'CLIP' - clip the output between clip_min and clip_max. "
            "'TANH' - tanh activation function. "
            "'SIGMOID' - sigmoid activation function. "
            "'LUT' - use a look-up table to perform the activation function.")
        .set_default("NONE");
    TVM_ATTR_FIELD(clip_min)
        .describe("The minimum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(clip_max)
        .describe("The maximum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(rounding_mode)
        .describe(
            "The rounding mode to apply to the Output Feature Map tensor. "
            "'TFL' - Tensorflow Lite rounding scheme. "
            "'TRUNCATE' - Truncate towards zero."
            "'NATURAL' - Round to nearest value, with x.5 rounded up towards +infinity.")
        .set_default("TFL");
    TVM_ATTR_FIELD(ifm_layout)
        .describe("The layout of the Input Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
    TVM_ATTR_FIELD(ofm_layout)
        .describe("The layout of the Output Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
  }
};

TVM_REGISTER_NODE_TYPE(EthosuUnaryElementwiseAttrs);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_CONTRIB_ETHOSU_OP_ATTRS_H_
