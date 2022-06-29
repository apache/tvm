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

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>

#include <string>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  int axis;
  std::string rounding;
  std::string compute_dtype;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.RequantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rounding).set_default("None").describe(
        "Defines the rounding direction when the value is midway between"
        "two representable values. There are two supported modes - UPWARD"
        "or TONEAREST. Both modes behave exactly same except at the"
        "midpoints between the two representable values. At the midpoint,"
        "UPWARD rounds towards positive infinity (for example -1.5 will be"
        "rounded to -1). TONEAREST is the standard rounding where the"
        "value is rounded away from zero at midpoints (for example, -1.5"
        "rounds to -2). More context can be found at following gblic manual"
        "https://www.gnu.org/software/libc/manual/html_node/Rounding.html.");
    TVM_ATTR_FIELD(compute_dtype)
        .set_default("None")
        .describe(
            "Specifies the data type used during requantize. Supported "
            "options: \"int64\", \"float32\", \"float64\"");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attribute for quantize operator */
struct QuantizeAttrs : public tvm::AttrsNode<QuantizeAttrs> {
  DataType out_dtype;
  int axis;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relay.attrs.QuantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type, can be one of [int8 or uint8].");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attribute for dequantize operator */
struct DequantizeAttrs : public tvm::AttrsNode<DequantizeAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(DequantizeAttrs, "relay.attrs.DequantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The channel axis for channel wise dequantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attribute for broadcast operator */
struct BroadcastAttrs : public tvm::AttrsNode<BroadcastAttrs> {
  int lhs_axis;
  int rhs_axis;

  TVM_DECLARE_ATTRS(BroadcastAttrs, "relay.attrs.BroadcastAttrs") {
    TVM_ATTR_FIELD(lhs_axis)
        .describe(
            "The channel axis for channel wise broadcast. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rhs_axis)
        .describe(
            "The channel axis for channel wise broadcast. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attributes used in QNN convolution operator */
struct QConv2DAttrs : public tvm::AttrsNode<QConv2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  tvm::String data_layout;
  tvm::String kernel_layout;
  tvm::String out_layout;
  tvm::String auto_scheduler_rewritten_layout;  // The layout after auto-scheduler's layout rewrite
  DataType out_dtype;

  // Optional extra attributes for Hexagon target. Describes requantization parameters.
  // Note, It is not set up explicitly through qnn._make.conv2d.
  int axis;
  DataType rq_out_dtype;

  TVM_DECLARE_ATTRS(QConv2DAttrs, "relay.attrs.QConv2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(axis)
        .describe(
            "The channel axis for channel wise requantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rq_out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Requantized output data type");
  }
};

/*! \brief Attributes for QNN dense operator */
struct QDenseAttrs : public tvm::AttrsNode<QDenseAttrs> {
  IndexExpr units;
  tvm::String auto_scheduler_rewritten_layout;  // The layout after auto-scheduler's layout rewrite
  DataType out_dtype;

  // Optional extra attributes for Hexagon target. Describes requantization parameters.
  // Note, It is not set up explicitly through qnn._make.dense.
  int axis;
  DataType rq_out_dtype;

  TVM_DECLARE_ATTRS(QDenseAttrs, "relay.attrs.QDenseAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(axis)
        .describe(
            "The channel axis for channel wise requantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rq_out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Requantized output data type");
  }
};

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_ATTRS_H_
