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
 * \file tvm/relay/qnn/attrs.h
 * \brief Auxiliary attributes for qnn operators.
 */
#ifndef TVM_RELAY_QNN_ATTRS_H_
#define TVM_RELAY_QNN_ATTRS_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  double input_scale;
  int32_t input_zero_point;
  double output_scale;
  int32_t output_zero_point;
  std::string rounding;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.RequantizeAttrs") {
    TVM_ATTR_FIELD(input_scale)
        .describe("The scale of the input tensor.");
    TVM_ATTR_FIELD(input_zero_point)
        .describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_scale)
        .describe("The scale of the output tensor.");
    TVM_ATTR_FIELD(output_zero_point)
        .describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(rounding).set_default("TONEAREST")
        .describe("Defines the rounding direction when the value is midway between"
                  "two representable values. There are two supported modes - UPWARD"
                  "or TONEAREST. Both modes behave exactly same except at the"
                  "midpoints between the two representable values. At the midpoint,"
                  "UPWARD rounds towards positive infinity (for example -1.5 will be"
                  "rounded to -1). TONEAREST is the standard rounding where the"
                  "value is rounded away from zero at midpoints (for example, -1.5"
                  "rounds to -2). More context can be found at following gblic manual"
                  "https://www.gnu.org/software/libc/manual/html_node/Rounding.html.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attribute for quantize operator */
struct QuantizeAttrs : public tvm::AttrsNode<QuantizeAttrs> {
  int32_t output_zero_point;
  double output_scale;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relay.attrs.QuantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype)
      .describe("Output data type, can be one of [int8 or uint8].");

    TVM_ATTR_FIELD(output_zero_point)
      .describe("The zero_point for the activation of this op.");

    TVM_ATTR_FIELD(output_scale)
      .describe("The scale for the activation of this op.");
  }
};

/*! \brief Attribute for dequantize operator */
struct DequantizeAttrs : public tvm::AttrsNode<DequantizeAttrs> {
  int32_t input_zero_point;
  double input_scale;

  TVM_DECLARE_ATTRS(DequantizeAttrs, "relay.attrs.DequantizeAttrs") {
    TVM_ATTR_FIELD(input_zero_point)
      .describe("The zero_point for the input tensor of this op.");

    TVM_ATTR_FIELD(input_scale)
      .describe("The scale for the input tensor of this op.");
  }
};

/*! \brief Attributes used in QNN concatenate operator */
struct QnnConcatenateAttrs : public tvm::AttrsNode<QnnConcatenateAttrs> {
  Array<tvm::Expr> input_scales;
  Array<tvm::Expr> input_zero_points;
  double output_scale;
  int32_t output_zero_point;
  int axis;

  TVM_DECLARE_ATTRS(QnnConcatenateAttrs, "relay.attrs.QnnConcatenateAttrs") {
    TVM_ATTR_FIELD(input_scales)
        .describe("The list of scales of input quantized tensors.");

    TVM_ATTR_FIELD(input_zero_points)
        .describe("The list of zero points of input quantized tensors.");

    TVM_ATTR_FIELD(output_zero_point)
      .describe("The zero_point for the output tensor.");

    TVM_ATTR_FIELD(output_scale)
      .describe("The scale for the output tensor.");

    TVM_ATTR_FIELD(axis)
        .describe("The axis at which the input arrays are concatenated."
                  "Should lie in range `[-ndim, ndim)`.")
        .set_default(0);
  }
};  // struct QnnConcatenateAttrs

/*! \brief Attribute for QNN Conv2d operator */
struct QnnConv2DAttrs : public tvm::AttrsNode<QnnConv2DAttrs> {
  // Traditional conv2d attributes.
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;

  TVM_DECLARE_ATTRS(QnnConv2DAttrs, "relay.attrs.QnnConv2DAttrs") {
    TVM_ATTR_FIELD(strides).set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).set_default(Array<IndexExpr>({0, 0}))
        .describe("If padding is non-zero, then the input is implicitly zero-padded"
                  "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation).set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1)
        .describe("Controls the connections between inputs and outputs."
                  "At groups=1, all inputs are convolved to all outputs."
                  "At groups=2, the operation becomes equivalent to having two convolution"
                  "layers side by side, each seeing half the input channels, and producing"
                  "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe("The number of output channels in the convolution."
                  " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr> >());
    TVM_ATTR_FIELD(data_layout).set_default("NCHW")
        .describe("Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                  "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                  "dimensions respectively. Convolution is applied on the 'H' and"
                  "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout).set_default("OIHW")
        .describe("Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
                  "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
                  "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout).set_default("")
        .describe("Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
                  "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                  "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(input_zero_point)
        .describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point)
        .describe("The zero point of the kernel tensor.");
  }
};

/*! \brief Attribute for QNN binary operator */
struct QnnBinaryOpAttrs : public tvm::AttrsNode<QnnBinaryOpAttrs> {
  int32_t lhs_zero_point;
  double lhs_scale;
  int32_t rhs_zero_point;
  double rhs_scale;
  int32_t output_zero_point;
  double output_scale;

  TVM_DECLARE_ATTRS(QnnBinaryOpAttrs, "relay.attrs.QnnBinaryOpAttrs") {
    TVM_ATTR_FIELD(lhs_zero_point)
      .describe("The zero_point for the lhs input tensor of this op.");

    TVM_ATTR_FIELD(lhs_scale)
      .describe("The scale for the lhs input tensor of this op.");

    TVM_ATTR_FIELD(rhs_zero_point)
      .describe("The zero_point for the rhs input tensor of this op.");

    TVM_ATTR_FIELD(rhs_scale)
      .describe("The scale for the rhs input tensor of this op.");

    TVM_ATTR_FIELD(output_zero_point)
      .describe("The zero_point for the activation of this op.");

    TVM_ATTR_FIELD(output_scale)
      .describe("The scale for the activation of this op.");
  }
};

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_ATTRS_H_
