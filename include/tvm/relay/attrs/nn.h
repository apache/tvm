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

#include <tvm/attrs.h>
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
    TVM_ATTR_FIELD(axis)
        .describe("The axis to add the bias")
        .set_default(1);
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
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DAttrs, "relay.attrs.Conv2DAttrs") {
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

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};


/*! \brief Attributes used in winograd weight transformation operators */
struct Conv2DWinogradWeightTransformAttrs :
    public tvm::AttrsNode<Conv2DWinogradWeightTransformAttrs> {
  int tile_size;

  TVM_DECLARE_ATTRS(Conv2DWinogradWeightTransformAttrs,
      "relay.attrs.Conv2DWinogradWeightTransformAttrs") {
    TVM_ATTR_FIELD(tile_size)
      .describe("Tile size of winograd. E.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)");
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
  std::string data_layout;
  std::string kernel_layout;
  std::string out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DWinogradAttrs, "relay.attrs.Conv2DWinogradAttrs") {
    TVM_ATTR_FIELD(tile_size)
      .describe("The tile size of winograd. E.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)");
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

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public tvm::AttrsNode<SoftmaxAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SoftmaxAttrs, "relay.attrs.SoftmaxAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1)
      .describe("The axis to sum over when computing softmax.");
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
      .describe("The dimensionality of the output space"
                "i.e. the number of output channels in the convolution.");
    TVM_ATTR_FIELD(kernel_size)
      .describe("The dimensions of the convolution window.")
      .set_default(NullValue<Array<IndexExpr> >());
    TVM_ATTR_FIELD(strides).set_default(Array<IndexExpr>({1, 1}))
      .describe("The strides of the convolution.");
    TVM_ATTR_FIELD(output_padding).set_default(Array<IndexExpr>({0, 0}))
      .describe("Zero-padding added to one side of the output.");
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
    TVM_ATTR_FIELD(data_layout).set_default("NCHW")
      .describe("Dimension ordering of data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout).set_default("OIHW")
      .describe("Dimension ordering of data and weight. Can be 'OIHW', 'OIHW16o16i', etc."
                "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
                "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout).set_default("")
        .describe("Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
                      "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
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
  std::string layout;
  bool ceil_mode;

  TVM_DECLARE_ATTRS(MaxPool2DAttrs, "relay.attrs.MaxPool2DAttrs") {
    TVM_ATTR_FIELD(pool_size)
      .describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides).set_default(Array<IndexExpr>({1, 1}))
      .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).set_default(Array<IndexExpr>({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : bottom, right will use same padding as top, left"
                "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false)
      .describe("When true, will use ceil instead of floor to compute the output shape.");
  }
};

/*! \brief Attributes for avg pool operator */
struct AvgPool2DAttrs : public tvm::AttrsNode<AvgPool2DAttrs> {
  Array<IndexExpr> pool_size;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  std::string layout;
  bool ceil_mode;
  bool count_include_pad;

  TVM_DECLARE_ATTRS(AvgPool2DAttrs, "relay.attrs.AvgPool2DAttrs") {
    TVM_ATTR_FIELD(pool_size)
      .describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides).set_default(Array<IndexExpr>({1, 1}))
      .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).set_default(Array<IndexExpr>({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : bottom, right will use same padding as top, left"
                "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false)
      .describe("When true, will use ceil instead of floor to compute the output shape.");
    TVM_ATTR_FIELD(count_include_pad).set_default(false)
      .describe("When true, will include padding to compute the average");
  }
};

/*! \brief Attributes for global pool operator */
struct GlobalPool2DAttrs : public tvm::AttrsNode<GlobalPool2DAttrs> {
  std::string layout;

  TVM_DECLARE_ATTRS(GlobalPool2DAttrs, "relay.attrs.GlobalPool2DAttrs") {
    TVM_ATTR_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
  }
};

/*! \brief Attributes for adaptive pool operator */
struct AdaptivePool2DAttrs : public tvm::AttrsNode<AdaptivePool2DAttrs> {
  Array<IndexExpr> output_size;
  std::string layout;

  TVM_DECLARE_ATTRS(AdaptivePool2DAttrs, "relay.attrs.AdaptivePool2DAttrs") {
    TVM_ATTR_FIELD(output_size).set_default(Array<IndexExpr>({}))
      .describe("Output height and width.");
    TVM_ATTR_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                  "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                  "dimensions respectively. Convolution is applied on the 'H' and"
                  "'W' dimensions.");
  }
};


/*! \brief Attributes for dense operator */
struct DenseAttrs : public tvm::AttrsNode<DenseAttrs> {
  IndexExpr units;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(DenseAttrs, "relay.attrs.DenseAttrs") {
    TVM_ATTR_FIELD(units)
        .describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes for sparse_dense operator */
struct SparseDenseAttrs : public tvm::AttrsNode<SparseDenseAttrs> {
  TVM_DECLARE_ATTRS(SparseDenseAttrs, "relay.attrs.SparseDenseAttrs") {}
};

/*! \brief Attributes for sparse_transpose operator */
struct SparseTransposeAttrs : public tvm::AttrsNode<SparseTransposeAttrs> {
  TVM_DECLARE_ATTRS(SparseTransposeAttrs, "relay.attrs.SparseTransposeAttrs") {}
};

/*! \brief Attributes for upsampling operator */
struct UpSamplingAttrs : public tvm::AttrsNode<UpSamplingAttrs> {
  int scale;
  std::string layout;
  std::string method;

  TVM_DECLARE_ATTRS(UpSamplingAttrs, "relay.attrs.UpSamplingAttrs") {
    TVM_ATTR_FIELD(scale)
        .describe("Should be true to preserve the values at the corner pixels");
    TVM_ATTR_FIELD(layout).set_default("NCHW")
        .describe("Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                  "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                  "dimensions respectively. Upsampling is applied on the 'H' and"
                  "'W' dimensions.");
    TVM_ATTR_FIELD(method).set_default("NEAREST_NEIGHBOR")
        .describe("Specify the mode to use for scaling."
                  "NEAREST_NEIGHBOR -  Nearest Neighbor"
                  "BILINEAR - Bilinear Interpolation");
  }
};

/*! \brief Attributes used for the padding operator */
struct PadAttrs : public tvm::AttrsNode<PadAttrs> {
  double pad_value;
  Array<Array<IndexExpr> > pad_width;

  TVM_DECLARE_ATTRS(PadAttrs, "relay.attrs.PadAttrs") {
    TVM_ATTR_FIELD(pad_value).set_default(0.0)
      .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(pad_width)
      .describe("Number of values padded to the edges of each axis, "
                "in the format of ((before_1, after_1), ..., (before_N, after_N))");
  }
};

/*! \brief Attributes used for the MirrorPadding operator */
struct MirrorPadAttrs : public tvm::AttrsNode<MirrorPadAttrs> {
  std::string mode;
  Array<Array<IndexExpr> > pad_width;

  TVM_DECLARE_ATTRS(MirrorPadAttrs, "relay.attrs.MirrorPadAttrs") {
    TVM_ATTR_FIELD(mode).set_default("SYMMETRIC")
      .describe("Specifies how mirroring should be performed.");
    TVM_ATTR_FIELD(pad_width)
      .describe("Number of values padded to the edges of each axis, "
                "in the format of ((before_1, after_1), ..., (before_N, after_N))");
  }
};

/*! \brief Attributes for leaky relu operator */
struct LeakyReluAttrs : public tvm::AttrsNode<LeakyReluAttrs> {
  double alpha;

  TVM_DECLARE_ATTRS(LeakyReluAttrs, "relay.attrs.LeakyReluAttrs") {
    TVM_ATTR_FIELD(alpha).set_lower_bound(0.0).set_default(0.25)
        .describe("Slope coefficient for the negative half axis.");
  }
};


/*! \brief Attributes for prelu operator */
struct PReluAttrs : public tvm::AttrsNode<PReluAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(PReluAttrs, "relay.attrs.PReluAttrs") {
    TVM_ATTR_FIELD(axis).set_default(1)
        .describe("Specify which shape axis the channel is specified.");
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
    TVM_ATTR_FIELD(axis)
      .describe("Specify which shape axis denotes the channel.")
      .set_default(1);
    TVM_ATTR_FIELD(epsilon)
      .describe("Small float added to variance to avoid dividing by zero")
      .set_default(1e-5);
    TVM_ATTR_FIELD(center)
      .describe("If True, add offset of beta to normalized tensor. If False, beta is ignored")
      .set_default(true);
    TVM_ATTR_FIELD(scale)
      .describe("If True, multiply by gamma. If False, gamma is not used. "
                "When the next layer is piecewise linear (also, e.g., nn.relu), "
                "this can be disabled since the scaling will be done by the next layer.")
      .set_default(true);
  }
};  // struct BatchNormAttrs


/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public tvm::AttrsNode<LayerNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(LayerNormAttrs, "relay.attrs.LayerNormAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1)
      .describe("Specify which shape axis denotes the channel.");
    TVM_ATTR_FIELD(epsilon).set_default(1e-5)
      .describe("Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).set_default(true)
      .describe("If true, add offset of beta to normalized tensor; "
                "otherwise, beta is ignored.");
    TVM_ATTR_FIELD(scale).set_default(true)
      .describe("If true, multiply by gamma; otherwise, gamma is ignored.");
  }
};  // struct LayerNormAttrs


/*! \brief Attributes for LRN operator */
struct LRNAttrs : public tvm::AttrsNode<LRNAttrs> {
  int size;
  int axis;
  double bias;
  double alpha;
  double beta;

  TVM_DECLARE_ATTRS(LRNAttrs, "relay.attrs.LRNAttrs") {
    TVM_ATTR_FIELD(size).set_default(5)
      .describe("The size of the local region to be considered for normalization.");
    TVM_ATTR_FIELD(axis).set_default(1)
      .describe("Axis of input data layout channel.");
    TVM_ATTR_FIELD(bias).set_default(2)
      .describe("The offset parameter to avoid division by 0.");
    TVM_ATTR_FIELD(alpha).set_default(0.0001)
      .describe("The scaling parameter.");
    TVM_ATTR_FIELD(beta).set_default(0.75)
      .describe("The exponent parameter.");
  }
};


/*! \brief Attributes for L2Normalize operator */
struct L2NormalizeAttrs : public tvm::AttrsNode<L2NormalizeAttrs> {
  double eps;
  Array<Integer> axis;

  TVM_DECLARE_ATTRS(L2NormalizeAttrs, "relay.attrs.L2NormalizeAttrs") {
    TVM_ATTR_FIELD(eps)
      .describe("A lower bound value for the norm, to avoid division by 0.");
    TVM_ATTR_FIELD(axis)
      .describe("Axis over the normalization applied.");
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
    TVM_ATTR_FIELD(strides).set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).set_default(Array<IndexExpr>({0, 0}))
        .describe("If padding is non-zero, then the input is implicitly zero-padded"
                  "on both sides for padding number of points");
    TVM_ATTR_FIELD(dilation).set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(deformable_groups).set_default(1)
        .describe("Controls the connections between inputs and offsets."
                  "Input channels are partitioned into multiple deformable groups. Offsets"
                  "are shared across input channels in the same deformable group.");
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

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_NN_H_
