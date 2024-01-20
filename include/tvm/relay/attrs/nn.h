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
 * \file tvm/relay/attrs/nn.h
 * \brief Auxiliary attributes for nn operators.
 */
#ifndef TVM_RELAY_ATTRS_NN_H_
#define TVM_RELAY_ATTRS_NN_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Add a 1D Tensor to an axis of a data.
 *
 * \note bias_add is a special add operator that is in nn
 *   and enables automatic derivation of bias's shape.
 *   You can directly use add for more generalized case.
 */
struct BiasAddAttrs : public tvm::AttrsNode<BiasAddAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(BiasAddAttrs, "relay.attrs.BiasAddAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis to add the bias").set_default(1);
  }
};

/*! \brief Attributes used in 1D convolution operators */
struct Conv1DAttrs : public tvm::AttrsNode<Conv1DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  tvm::String data_layout;
  tvm::String kernel_layout;
  tvm::String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv1DAttrs, "relay.attrs.Conv1DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({
            1,
        }))
        .describe("Specifies the stride of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({
            1,
        }))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Currently unused but may be added in the future.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCW")
        .describe(
            "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Convolution is applied on the 'W'"
            "dimension.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIW")
        .describe(
            "Dimension ordering of weight. Can be 'OIW', or 'WIO', etc."
            "'O', 'I', 'W' stands for num_filter, input_channel, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Default to be same as input layout.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in convolution operators */
struct Conv2DAttrs : public tvm::AttrsNode<Conv2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  tvm::String data_layout;
  tvm::String kernel_layout;
  tvm::String out_layout;
  tvm::String auto_scheduler_rewritten_layout;   // The layout after auto-scheduler's layout rewrite
  Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DAttrs, "relay.attrs.Conv2DAttrs") {
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
  }
};

/*! \brief Attributes used in winograd weight transformation operators */
struct ConvWinogradWeightTransformAttrs : public tvm::AttrsNode<ConvWinogradWeightTransformAttrs> {
  int tile_size;

  TVM_DECLARE_ATTRS(ConvWinogradWeightTransformAttrs,
                    "relay.attrs.ConvWinogradWeightTransformAttrs") {
    TVM_ATTR_FIELD(tile_size).describe(
        "Tile size of winograd. E.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)");
  }
};

/*! \brief Attributes used in gemm weight transformation operators */
struct ConvGemmWeightTransformAttrs : public tvm::AttrsNode<ConvGemmWeightTransformAttrs> {
  int tile_N;
  int tile_K;

  TVM_DECLARE_ATTRS(ConvGemmWeightTransformAttrs, "relay.attrs.ConvGemmWeightTransformAttrs") {
    TVM_ATTR_FIELD(tile_N).describe(
        "Tile size across N axis of the weight transformation for ConvGemm. (N = OC)");
    TVM_ATTR_FIELD(tile_K).describe(
        "Tile size across K axis of the weight transformation for ConvGemm. (K = KW * KH * IC)");
  }
};

/*! \brief Attributes used in convolution operators with winograd algorithm */
struct Conv2DWinogradAttrs : public tvm::AttrsNode<Conv2DWinogradAttrs> {
  int tile_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  tvm::String data_layout;
  tvm::String kernel_layout;
  tvm::String out_layout;
  tvm::String auto_scheduler_rewritten_layout;   // The layout after auto-scheduler's layout rewrite
  Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DWinogradAttrs, "relay.attrs.Conv2DWinogradAttrs") {
    TVM_ATTR_FIELD(tile_size).describe(
        "The tile size of winograd. E.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)");
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
  }
};

/*! \brief Attributes used in winograd weight transformation operators */
struct Conv2DWinogradNNPACKWeightTransformAttrs
    : public tvm::AttrsNode<Conv2DWinogradNNPACKWeightTransformAttrs> {
  int convolution_algorithm;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DWinogradNNPACKWeightTransformAttrs,
                    "relay.attrs.Conv2DWinogradNNPACKWeightTransformAttrs") {
    TVM_ATTR_FIELD(convolution_algorithm)
        .describe(
            "The convolution algorithm for Winograd NNPACK. "
            "E.g. tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8 for WT_8x8, "
            "tvm.contrib.nnpack.ConvolutionAlgorithm.WT_8x8_FP16 for WT_8x8_FP16");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in convolution operators */
struct Conv3DAttrs : public tvm::AttrsNode<Conv3DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  tvm::String data_layout;
  tvm::String kernel_layout;
  tvm::String out_layout;
  tvm::String auto_scheduler_rewritten_layout;   // The layout after auto-scheduler's layout rewrite
  Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv3DAttrs, "relay.attrs.Conv3DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : back, bottom, right will use same padding as front, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom,"
            "right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1, 1}))
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
        .set_default("NCDHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Convolution is applied on the 'D', 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIDHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIDHW', 'OIDHW16o16i', etc."
            "'O', 'I', 'D', 'H', 'W' stands for num_filter, input_channel, depth, height,"
            "and width dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Default to be same as input layout.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in transposed convolution operator */
struct Conv3DTransposeAttrs : public tvm::AttrsNode<Conv3DTransposeAttrs> {
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> output_padding;
  Array<IndexExpr> dilation;
  int groups;
  tvm::String data_layout;
  tvm::String kernel_layout;
  tvm::String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv3DTransposeAttrs, "relay.attrs.Conv3DTransposeAttrs") {
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe(
            "The dimensionality of the output space"
            "i.e. the number of output channels in the convolution.");
    TVM_ATTR_FIELD(kernel_size)
        .describe("The dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("The strides of the convolution.");
    TVM_ATTR_FIELD(output_padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "Zero-padding added to one side of the output."
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : front, bottom, right will use same padding as back, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : front, bottom, right will use same padding as back, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCDHW")
        .describe(
            "Dimension ordering of data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Convolution is applied on the 'D', 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("IODHW")
        .describe(
            "Dimension ordering of data and weight. Can be 'IODHW', 'IODHW16i16o', etc."
            "'I', 'O', 'D', 'H', 'W' stands for input_channel, num_filter, depth, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in 3d winograd convolution operators */
struct Conv3DWinogradAttrs : public tvm::AttrsNode<Conv3DWinogradAttrs> {
  int tile_size;
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

  TVM_DECLARE_ATTRS(Conv3DWinogradAttrs, "relay.attrs.Conv3DWinogradAttrs") {
    TVM_ATTR_FIELD(tile_size).describe(
        "The tile size of winograd. E.g. 2 for F(2x2x2, 3x3x3) and 4 for F(4x4x4, 3x3x3)");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : back, bottom, right will use same padding as front, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom,"
            "right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1, 1}))
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
        .set_default("NCDHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Convolution is applied on the 'D', 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIDHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIDHW', 'OIDHW16o16i', etc."
            "'O', 'I', 'D', 'H', 'W' stands for num_filter, input_channel, depth, height,"
            "and width dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Default to be same as input layout.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public tvm::AttrsNode<SoftmaxAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SoftmaxAttrs, "relay.attrs.SoftmaxAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe("The axis to sum over when computing softmax.");
  }
};

/*! \brief Attributes used in transposed convolution operator */
struct Conv2DTransposeAttrs : public tvm::AttrsNode<Conv2DTransposeAttrs> {
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> output_padding;
  Array<IndexExpr> dilation;
  int groups;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DTransposeAttrs, "relay.attrs.Conv2DTransposeAttrs") {
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe(
            "The dimensionality of the output space"
            "i.e. the number of output channels in the convolution.");
    TVM_ATTR_FIELD(kernel_size)
        .describe("The dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The strides of the convolution.");
    TVM_ATTR_FIELD(output_padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "Zero-padding added to one side of the output."
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
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
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("IOHW")
        .describe(
            "Dimension ordering of data and weight. Can be 'IOHW', 'OIHW16o16i', etc."
            "'I', 'O', 'H', 'W' stands for input_channel, num_filter, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in dilate operator */
struct DilateAttrs : public tvm::AttrsNode<DilateAttrs> {
  Array<IndexExpr> strides;
  double dilation_value;

  TVM_DECLARE_ATTRS(DilateAttrs, "relay.attrs.DilateAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Dilation stride on each dimension, 1 means no dilation.");
    TVM_ATTR_FIELD(dilation_value).set_default(0.0).describe("Value used to dilate the input.");
  }
};

/*! \brief Attributes used in 1D transposed convolution operator */
struct Conv1DTransposeAttrs : public tvm::AttrsNode<Conv1DTransposeAttrs> {
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> output_padding;
  Array<IndexExpr> dilation;
  int groups;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv1DTransposeAttrs, "relay.attrs.Conv1DTransposeAttrs") {
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe(
            "The dimensionality of the output space"
            "i.e. the number of output channels in the convolution.");
    TVM_ATTR_FIELD(kernel_size)
        .describe("The dimensions of the convolution window.")
        .set_default(NullValue<Array<IndexExpr>>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1}))
        .describe("The strides of the convolution.");
    TVM_ATTR_FIELD(output_padding)
        .set_default(Array<IndexExpr>({0}))
        .describe("Zero-padding added to one side of the output.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0}))
        .describe(
            "Symmetric or asymmetric padding."
            "Single value: the input is implicitly zero-padded on both sides."
            "Two values: padding[0] is used for left input padding, "
            "padding[1] is used for right input padding,");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCW")
        .describe(
            "Dimension ordering of data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Convolution is applied on the"
            "'W' dimension.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("IOW")
        .describe(
            "Dimension ordering of data and weight. Can be 'IOW', 'IOW16o16i', etc."
            "'I', 'O', 'W' stands for input_channel, num_filter and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes for max pool operator */
struct MaxPool2DAttrs : public tvm::AttrsNode<MaxPool2DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  tvm::String layout;
  tvm::String out_layout;
  bool ceil_mode;

  TVM_DECLARE_ATTRS(MaxPool2DAttrs, "relay.attrs.MaxPool2DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
  }
};

/*! \brief Attributes for avg pool operator */
struct AvgPool2DAttrs : public tvm::AttrsNode<AvgPool2DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  tvm::String layout;
  tvm::String out_layout;
  bool ceil_mode;
  bool count_include_pad;

  TVM_DECLARE_ATTRS(AvgPool2DAttrs, "relay.attrs.AvgPool2DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
    TVM_ATTR_FIELD(count_include_pad)
        .set_default(false)
        .describe("When true, will include padding to compute the average");
  }
};

/*! \brief Attributes for global pool operator */
struct GlobalPool2DAttrs : public tvm::AttrsNode<GlobalPool2DAttrs> {
  tvm::String layout;
  tvm::String out_layout;

  TVM_DECLARE_ATTRS(GlobalPool2DAttrs, "relay.attrs.GlobalPool2DAttrs") {
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
  }
};

/*! \brief Attributes for 1d adaptive pool operator */
struct AdaptivePool1DAttrs : public tvm::AttrsNode<AdaptivePool1DAttrs> {
  Array<IndexExpr> output_size;
  std::string layout;
  tvm::String out_layout;

  TVM_DECLARE_ATTRS(AdaptivePool1DAttrs, "relay.attrs.AdaptivePool1DAttrs") {
    TVM_ATTR_FIELD(output_size).set_default(Array<IndexExpr>({})).describe("Output width.");
    TVM_ATTR_FIELD(layout).set_default("NCW").describe(
        "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
        "'N', 'C', 'W' stands for batch, channel, and width"
        "dimensions respectively. Pooling is applied on the"
        "'W' dimension.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Pooling is applied on the"
            "'W' dimension.");
  }
};

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrs : public tvm::AttrsNode<AdaptivePool2DAttrs> {
  Array<IndexExpr> output_size;
  std::string layout;
  tvm::String out_layout;

  TVM_DECLARE_ATTRS(AdaptivePool2DAttrs, "relay.attrs.AdaptivePool2DAttrs") {
    TVM_ATTR_FIELD(output_size)
        .set_default(Array<IndexExpr>({}))
        .describe("Output height and width.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
  }
};

/*! \brief Attributes for 3d adaptive pool operator */
struct AdaptivePool3DAttrs : public tvm::AttrsNode<AdaptivePool3DAttrs> {
  Array<IndexExpr> output_size;
  std::string layout;
  tvm::String out_layout;

  TVM_DECLARE_ATTRS(AdaptivePool3DAttrs, "relay.attrs.AdaptivePool3DAttrs") {
    TVM_ATTR_FIELD(output_size)
        .set_default(Array<IndexExpr>({}))
        .describe("Output depth, height and width.");
    TVM_ATTR_FIELD(layout).set_default("NCDHW").describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Pooling is applied on 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Pooling is applied on 'D', 'H' and"
            "'W' dimensions.");
  }
};

/*! \brief Attributes for 1D max pool operator */
struct MaxPool1DAttrs : public tvm::AttrsNode<MaxPool1DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> dilation;
  Array<IndexExpr> padding;
  std::string layout;
  tvm::String out_layout;
  bool ceil_mode;

  TVM_DECLARE_ATTRS(MaxPool1DAttrs, "relay.attrs.MaxPool1DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1}))
        .describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding supports both symmetric and asymmetric as"
            "one int : same padding used on each side"
            "two int : indicates left padding, right padding");
    TVM_ATTR_FIELD(layout).set_default("NCW").describe(
        "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
        "'N', 'C', 'W' stands for batch, channel, and width"
        "dimensions respectively. Pooling is applied on the 'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Pooling is applied on the 'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
  }
};

/*! \brief Attributes for 1D avg pool operator */
struct AvgPool1DAttrs : public tvm::AttrsNode<AvgPool1DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> dilation;
  Array<IndexExpr> padding;
  std::string layout;
  tvm::String out_layout;
  bool ceil_mode;
  bool count_include_pad;

  TVM_DECLARE_ATTRS(AvgPool1DAttrs, "relay.attrs.AvgPool1DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1}))
        .describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding supports both symmetric and asymmetric as"
            "one int : same padding used on each side"
            "two int : indicates left padding, right padding");
    TVM_ATTR_FIELD(layout).set_default("NCW").describe(
        "Dimension ordering of input data. Can be 'NCW', 'NHC', etc."
        "'N', 'C', 'W' stands for batch, channel, and width"
        "dimensions respectively. Pooling is applied on the 'W' dimension.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCW', 'NHC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Pooling is applied on the 'W' dimension.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
    TVM_ATTR_FIELD(count_include_pad)
        .set_default(false)
        .describe("When true, will include padding to compute the average");
  }
};

/*! \brief Attributes for 3D max pool operator */
struct MaxPool3DAttrs : public tvm::AttrsNode<MaxPool3DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> dilation;
  Array<IndexExpr> padding;
  std::string layout;
  tvm::String out_layout;
  bool ceil_mode;

  TVM_DECLARE_ATTRS(MaxPool3DAttrs, "relay.attrs.MaxPool3DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : back, bottom, right will use same padding as front, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCDHW").describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Pooling is applied on the 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Pooling is applied on the 'D', 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
  }
};

/*! \brief Attributes for 3D avg pool operator */
struct AvgPool3DAttrs : public tvm::AttrsNode<AvgPool3DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> dilation;
  Array<IndexExpr> padding;
  std::string layout;
  tvm::String out_layout;
  bool ceil_mode;
  bool count_include_pad;

  TVM_DECLARE_ATTRS(AvgPool3DAttrs, "relay.attrs.AvgPool3DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<IndexExpr>({1, 1, 1}))
        .describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "three int : back, bottom, right will use same padding as front, top, left"
            "six int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCDHW").describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Pooling is applied on the 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Pooling is applied on the 'D', 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
    TVM_ATTR_FIELD(count_include_pad)
        .set_default(false)
        .describe("When true, will include padding to compute the average");
  }
};

/*! \brief Attributes for matmul operator */
struct MatmulAttrs : public tvm::AttrsNode<MatmulAttrs> {
  IndexExpr units;
  DataType out_dtype;
  bool transpose_a;
  bool transpose_b;
  // layout of B after auto-scheduler's layout rewrite
  tvm::String auto_scheduler_rewritten_layout;
  Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights

  TVM_DECLARE_ATTRS(MatmulAttrs, "relay.attrs.MatmulAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(transpose_a)
        .set_default(false)
        .describe("Whether the first input tensor is in transposed format.");

    TVM_ATTR_FIELD(transpose_b)
        .set_default(false)
        .describe("Whether the second input tensor is in transposed format.");
  }
};

/*! \brief Attributes for dense operator */
struct DenseAttrs : public tvm::AttrsNode<DenseAttrs> {
  IndexExpr units;
  // layout of B after auto-scheduler's layout rewrite
  tvm::String auto_scheduler_rewritten_layout;
  Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights
  DataType out_dtype;

  TVM_DECLARE_ATTRS(DenseAttrs, "relay.attrs.DenseAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes for dense_pack operator */
struct DensePackAttrs : public tvm::AttrsNode<DensePackAttrs> {
  IndexExpr units;
  DataType out_dtype;
  tvm::String weight_layout;

  TVM_DECLARE_ATTRS(DensePackAttrs, "relay.attrs.DensePackAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(weight_layout)
        .set_default("NC")
        .describe("Dimension ordering of weight. Packed layouts, such as NC8n, are possible.");
  }
};

/*! \brief Attributes for batch matmul operator. */
struct BatchMatmulAttrs : public tvm::AttrsNode<BatchMatmulAttrs> {
  DataType out_dtype;
  bool transpose_a;
  bool transpose_b;
  tvm::String auto_scheduler_rewritten_layout;   // The layout after auto-scheduler's layout rewrite
  Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights

  TVM_DECLARE_ATTRS(BatchMatmulAttrs, "relay.attrs.BatchMatmulAttrs") {
    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(transpose_a)
        .set_default(false)
        .describe("Whether the first input tensor is in transposed format.");

    TVM_ATTR_FIELD(transpose_b)
        .set_default(false)
        .describe("Whether the second input tensor is in transposed format.");
  }
};

/*! \brief Attributes for sparse_dense operator */
struct SparseDenseAttrs : public tvm::AttrsNode<SparseDenseAttrs> {
  bool sparse_lhs;

  TVM_DECLARE_ATTRS(SparseDenseAttrs, "relay.attrs.SparseDenseAttrs") {
    TVM_ATTR_FIELD(sparse_lhs)
        .set_default(false)
        .describe(
            "Indicate whether sparse matrix is multiplied on the right or the left. If true, then "
            "the operation is S * D^T (D dense, S sparse). If false, the operation is D * S^T");
  }
};

/*! \brief Attributes for sparse_transpose operator */
struct SparseTransposeAttrs : public tvm::AttrsNode<SparseTransposeAttrs> {
  TVM_DECLARE_ATTRS(SparseTransposeAttrs, "relay.attrs.SparseTransposeAttrs") {}
};

/*! \brief Attributes for sparse_dense operator */
struct SparseConv2DAttrs : public tvm::AttrsNode<SparseConv2DAttrs> {
  std::string layout;
  Array<IndexExpr> kernel_size;

  TVM_DECLARE_ATTRS(SparseConv2DAttrs, "relay.attrs.SparseConv2DAttrs") {
    TVM_ATTR_FIELD(layout).set_default("NHWC").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC'"
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively.");
    TVM_ATTR_FIELD(kernel_size)
        .set_default(Array<IndexExpr>{1, 1})
        .describe("Kernel size for SparseConv2D, 1x1 or 3x3. ");
  }
};

/*! \brief Attributes for FIFO buffer operator */
struct FIFOBufferAttrs : public tvm::AttrsNode<FIFOBufferAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(FIFOBufferAttrs, "relay.attrs.FIFOBufferAttrs") {
    TVM_ATTR_FIELD(axis).set_default(0);
  }
};

/*! \brief Attributes for upsampling operator */
struct UpSamplingAttrs : public tvm::AttrsNode<UpSamplingAttrs> {
  double scale_h;
  double scale_w;
  tvm::String layout;
  tvm::String method;
  bool align_corners;

  TVM_DECLARE_ATTRS(UpSamplingAttrs, "relay.attrs.UpSamplingAttrs") {
    TVM_ATTR_FIELD(scale_h).describe("The upsampling factor for height");
    TVM_ATTR_FIELD(scale_w).describe("The upsampling factor for width");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Upsampling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method)
        .set_default("nearest_neighbor")
        .describe(
            "Specify the mode to use for scaling."
            "nearest_neighbor -  Nearest Neighbor"
            "bilinear - Bilinear Interpolation"
            "bicubic - Bicubic Interpolation");
    TVM_ATTR_FIELD(align_corners)
        .set_default(false)
        .describe("Should be true to preserve the values at the corner pixels");
  }
};

/*! \brief Attributes for upsampling3d operator */
struct UpSampling3DAttrs : public tvm::AttrsNode<UpSampling3DAttrs> {
  double scale_d;
  double scale_h;
  double scale_w;
  std::string layout;
  std::string method;
  std::string coordinate_transformation_mode;

  TVM_DECLARE_ATTRS(UpSampling3DAttrs, "relay.attrs.UpSampling3DAttrs") {
    TVM_ATTR_FIELD(scale_d).describe("The upsampling factor for depth");
    TVM_ATTR_FIELD(scale_h).describe("The upsampling factor for height");
    TVM_ATTR_FIELD(scale_w).describe("The upsampling factor for width");
    TVM_ATTR_FIELD(layout).set_default("NCDHW").describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Upsampling is applied on the 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method)
        .set_default("nearest_neighbor")
        .describe(
            "Specify the mode to use for scaling."
            "nearest_neighbor -  Nearest Neighbor"
            "trilinear - Trilinear Interpolation");
    TVM_ATTR_FIELD(coordinate_transformation_mode)
        .set_default("half_pixel")
        .describe(
            "Describes how to transform the coordinate in the resized tensor"
            "to the coordinate in the original tensor."
            "Refer to the ONNX Resize operator specification for details"
            "Available options are half_pixel, align_corners and asymmetric");
  }
};

/*! \brief Attributes used for the padding operator */
struct PadAttrs : public tvm::AttrsNode<PadAttrs> {
  Array<Array<Integer>> pad_width;
  tvm::String pad_mode;

  TVM_DECLARE_ATTRS(PadAttrs, "relay.attrs.PadAttrs") {
    TVM_ATTR_FIELD(pad_width).describe(
        "Number of values padded to the edges of each axis, "
        "in the format of ((before_1, after_1), ..., (before_N, after_N))");
    TVM_ATTR_FIELD(pad_mode)
        .set_default("constant")
        .describe(
            "Padding type to use. \"constant\" pads with constant_value, "
            "\"edge\" pads using the edge values of the input array, "
            "\"reflect\" pads by reflecting values with respect to the edges.");
  }
};

/*! \brief Attributes used for the MirrorPadding operator */
struct MirrorPadAttrs : public tvm::AttrsNode<MirrorPadAttrs> {
  std::string mode;
  Array<Array<IndexExpr>> pad_width;

  TVM_DECLARE_ATTRS(MirrorPadAttrs, "relay.attrs.MirrorPadAttrs") {
    TVM_ATTR_FIELD(mode)
        .set_default("SYMMETRIC")
        .describe("Specifies how mirroring should be performed.");
    TVM_ATTR_FIELD(pad_width).describe(
        "Number of values padded to the edges of each axis, "
        "in the format of ((before_1, after_1), ..., (before_N, after_N))");
  }
};

/*! \brief Attributes for leaky relu operator */
struct LeakyReluAttrs : public tvm::AttrsNode<LeakyReluAttrs> {
  double alpha;

  TVM_DECLARE_ATTRS(LeakyReluAttrs, "relay.attrs.LeakyReluAttrs") {
    TVM_ATTR_FIELD(alpha).set_lower_bound(0.0).set_default(0.25).describe(
        "Slope coefficient for the negative half axis.");
  }
};

/*! \brief Attributes for prelu operator */
struct PReluAttrs : public tvm::AttrsNode<PReluAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(PReluAttrs, "relay.attrs.PReluAttrs") {
    TVM_ATTR_FIELD(axis).set_default(1).describe(
        "Specify which shape axis the channel is specified.");
  }
};

/*! \brief Attributes used in dropout operator */
struct DropoutAttrs : public tvm::AttrsNode<DropoutAttrs> {
  double rate;
  TVM_DECLARE_ATTRS(DropoutAttrs, "relay.attrs.DropoutAttrs") {
    TVM_ATTR_FIELD(rate)
        .describe("Fraction of the input that gets dropped out during training time")
        .set_default(0.5);
  }
};  // struct DropoutAttrs

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public tvm::AttrsNode<BatchNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(BatchNormAttrs, "relay.attrs.BatchNormAttrs") {
    TVM_ATTR_FIELD(axis).describe("Specify which shape axis denotes the channel.").set_default(1);
    TVM_ATTR_FIELD(epsilon)
        .describe("Small float added to variance to avoid dividing by zero")
        .set_default(1e-5);
    TVM_ATTR_FIELD(center)
        .describe("If True, add offset of beta to normalized tensor. If False, beta is ignored")
        .set_default(true);
    TVM_ATTR_FIELD(scale)
        .describe(
            "If True, multiply by gamma. If False, gamma is not used. "
            "When the next layer is piecewise linear (also, e.g., nn.relu), "
            "this can be disabled since the scaling will be done by the next layer.")
        .set_default(true);
  }
};  // struct BatchNormAttrs

/*! \brief Attributes used in instance_norm operator */
struct InstanceNormAttrs : public tvm::AttrsNode<InstanceNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(InstanceNormAttrs, "relay.attrs.InstanceNormAttrs") {
    TVM_ATTR_FIELD(axis).describe("Specify which shape axis denotes the channel.").set_default(1);
    TVM_ATTR_FIELD(epsilon)
        .describe("Small float added to variance to avoid dividing by zero")
        .set_default(1e-5);
    TVM_ATTR_FIELD(center).set_default(true).describe(
        "If true, add offset of beta to normalized tensor; "
        "otherwise, beta is ignored.");
    TVM_ATTR_FIELD(scale).set_default(true).describe(
        "If true, multiply by gamma; otherwise, gamma is ignored.");
  }
};  // struct InstanceNormAttrs

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public tvm::AttrsNode<LayerNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(LayerNormAttrs, "relay.attrs.LayerNormAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Specify which shape axis denotes the channel.");
    TVM_ATTR_FIELD(epsilon).set_default(1e-5).describe(
        "Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).set_default(true).describe(
        "If true, add offset of beta to normalized tensor; "
        "otherwise, beta is ignored.");
    TVM_ATTR_FIELD(scale).set_default(true).describe(
        "If true, multiply by gamma; otherwise, gamma is ignored.");
  }
};  // struct LayerNormAttrs

/*! \brief Attributes used in group_norm operator */
struct GroupNormAttrs : public tvm::AttrsNode<GroupNormAttrs> {
  int num_groups;
  int axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(GroupNormAttrs, "relay.attrs.GroupNormAttrs") {
    TVM_ATTR_FIELD(num_groups)
        .set_default(0)
        .describe("Specify number of groups to separate the channels into.");
    TVM_ATTR_FIELD(axis).set_default(1).describe("Specify which shape axis denotes the channel.");
    TVM_ATTR_FIELD(epsilon).set_default(1e-5).describe(
        "Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).set_default(true).describe(
        "If true, add offset of beta to normalized tensor; "
        "otherwise, beta is ignored.");
    TVM_ATTR_FIELD(scale).set_default(true).describe(
        "If true, multiply by gamma; otherwise, gamma is ignored.");
  }
};  // struct GroupNormAttrs

/*! \brief Attributes for LRN operator */
struct LRNAttrs : public tvm::AttrsNode<LRNAttrs> {
  int size;
  int axis;
  double bias;
  double alpha;
  double beta;

  TVM_DECLARE_ATTRS(LRNAttrs, "relay.attrs.LRNAttrs") {
    TVM_ATTR_FIELD(size).set_default(5).describe(
        "The size of the local region to be considered for normalization.");
    TVM_ATTR_FIELD(axis).set_default(1).describe("Axis of input data layout channel.");
    TVM_ATTR_FIELD(bias).set_default(2).describe("The offset parameter to avoid division by 0.");
    TVM_ATTR_FIELD(alpha).set_default(0.0001).describe("The scaling parameter.");
    TVM_ATTR_FIELD(beta).set_default(0.75).describe("The exponent parameter.");
  }
};

/*! \brief Attributes for L2Normalize operator */
struct L2NormalizeAttrs : public tvm::AttrsNode<L2NormalizeAttrs> {
  double eps;
  Array<Integer> axis;

  TVM_DECLARE_ATTRS(L2NormalizeAttrs, "relay.attrs.L2NormalizeAttrs") {
    TVM_ATTR_FIELD(eps).describe("A lower bound value for the norm, to avoid division by 0.");
    TVM_ATTR_FIELD(axis).describe("Axis over the normalization applied.");
  }
};

/*! \brief Attributes for DeformableConv2D operator */
struct DeformableConv2DAttrs : public tvm::AttrsNode<DeformableConv2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int deformable_groups;
  int groups;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(DeformableConv2DAttrs, "relay.attrs.DeformableConv2DAttrs") {
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
    TVM_ATTR_FIELD(deformable_groups)
        .set_default(1)
        .describe(
            "Controls the connections between inputs and offsets."
            "Input channels are partitioned into multiple deformable groups. Offsets"
            "are shared across input channels in the same deformable group.");
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
  }
};

/*! \brief Attributes used in subpixel operators */
struct SubPixelAttrs : public tvm::AttrsNode<SubPixelAttrs> {
  int block_size;
  std::string layout;
  std::string mode;

  TVM_DECLARE_ATTRS(SubPixelAttrs, "relay.attrs.SubPixelAttrs") {
    TVM_ATTR_FIELD(block_size)
        .describe("The size of subpixel blocks to compose or decompose.")
        .set_default(1);
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively.");
    TVM_ATTR_FIELD(mode).set_default("DCR").describe(
        "Indicates order in which channels are accessed. Must be one of"
        "DCR or CDR.");
  }
};  // struct SubPixelAttrs

/*! \brief Attributes used in correlation operators */
struct CorrelationAttrs : public tvm::AttrsNode<CorrelationAttrs> {
  int kernel_size;
  int max_displacement;
  int stride1;
  int stride2;
  Array<IndexExpr> padding;
  bool is_multiply;
  String layout;

  TVM_DECLARE_ATTRS(CorrelationAttrs, "relay.attrs.CorrelationAttrs") {
    TVM_ATTR_FIELD(kernel_size)
        .describe("Kernel size for correlation, must be an odd number.")
        .set_default(1);
    TVM_ATTR_FIELD(max_displacement).describe("Max displacement of Correlation.").set_default(1);
    TVM_ATTR_FIELD(stride1).describe("Stride for data1.").set_default(1);
    TVM_ATTR_FIELD(stride2).describe("Stride for data2.").set_default(1);
    TVM_ATTR_FIELD(padding)
        .describe("Padding for data1 and data2.")
        .set_default(Array<IndexExpr>{0, 0});
    TVM_ATTR_FIELD(is_multiply)
        .describe("Operation type is either multiplication or substraction.")
        .set_default(true);
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively.");
  }
};  // struct CorrelationAttrs

/*! \brief Attributes used in SpaceToBatchND operator */
struct SpaceToBatchNDAttrs : public tvm::AttrsNode<SpaceToBatchNDAttrs> {
  Array<Integer> block_shape;
  Array<Array<IndexExpr>> paddings;
  double pad_value;

  TVM_DECLARE_ATTRS(SpaceToBatchNDAttrs, "relay.attrs.SpaceToBatchNDAttrs") {
    TVM_ATTR_FIELD(block_shape)
        .set_default(Array<Integer>({1, 1}))
        .describe("1-D containing block size for each spatial dimension.");
    TVM_ATTR_FIELD(paddings).describe("2-D containing paddings for each spatial dimension.");
    TVM_ATTR_FIELD(pad_value).set_default(0.0).describe("The value used for padding.");
  }
};  // struct SpaceToBatchNDAttrs

/*! \brief Attributes used in BatchToSpaceND operator */
struct BatchToSpaceNDAttrs : public tvm::AttrsNode<BatchToSpaceNDAttrs> {
  Array<Integer> block_shape;
  Array<Array<IndexExpr>> crops;

  TVM_DECLARE_ATTRS(BatchToSpaceNDAttrs, "relay.attrs.BatchToSpaceNDAttrs") {
    TVM_ATTR_FIELD(block_shape)
        .set_default(Array<Integer>({1, 1}))
        .describe("1-D containing block size for each spatial dimension.");
    TVM_ATTR_FIELD(crops).describe("2-D containing amount to crop from spatial dimension.");
  }
};  // struct BatchToSpaceNDAttrs

/*! \brief Attributes used in NLLLoss operator */
struct NLLLossAttrs : public tvm::AttrsNode<NLLLossAttrs> {
  std::string reduction;
  int ignore_index;

  TVM_DECLARE_ATTRS(NLLLossAttrs, "relay.attrs.NLLLossAttrs") {
    TVM_ATTR_FIELD(reduction).set_default("mean").describe(
        "The reduction method to apply to the output. Can be"
        "'none', 'mean' or 'sum'.");
    TVM_ATTR_FIELD(ignore_index).describe("The target value to ignore.");
  }
};  // struct NLLLossAttrs

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_NN_H_
