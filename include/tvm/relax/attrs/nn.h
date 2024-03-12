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
 * \file tvm/relax/attrs/nn.h
 * \brief Attributes for neural network operators.
 */
#ifndef TVM_RELAX_ATTRS_NN_H_
#define TVM_RELAX_ATTRS_NN_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in Conv1d operator */
struct Conv1DAttrs : public tvm::AttrsNode<Conv1DAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv1DAttrs, "relax.attrs.Conv1DAttrs") {
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on both sides"
        "two int : padding width in the order of (left, right)");
    TVM_ATTR_FIELD(dilation).describe(
        "Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).describe(
        "Number of groups to split the input into for grouped convolution. The number of input and "
        "output channels should be divisible by the number of groups.");
    TVM_ATTR_FIELD(data_layout)
        .describe(
            "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, width"
            "dimensions respectively. Convolution is applied on the 'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .describe(
            "Dimension ordering of weight. Can be 'OIW', 'IOW', etc."
            "'O', 'I', 'W' stands for num_filter, input_channel, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
  }
};  // struct Conv1dAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DAttrs : public tvm::AttrsNode<Conv2DAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DAttrs, "relax.attrs.Conv2DAttrs") {
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "two int : bottom, right will use same padding as top, left"
        "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(dilation).describe(
        "Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).describe(
        "Number of groups to split the input into for grouped convolution. The number of input and "
        "output channels should be divisible by the number of groups.");
    TVM_ATTR_FIELD(data_layout)
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
  }
};  // struct Conv2dAttrs

/*! \brief Attributes used in Conv3d operator */
struct Conv3DAttrs : public tvm::AttrsNode<Conv3DAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv3DAttrs, "relax.attrs.Conv3DAttrs") {
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "two int : bottom, right will use same padding as top, left"
        "four int : padding width in the order of (forward, back, top, left, bottom, right)");
    TVM_ATTR_FIELD(dilation).describe(
        "Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).describe(
        "Number of groups to split the input into for grouped convolution. The number of input and "
        "output channels should be divisible by the number of groups.");
    TVM_ATTR_FIELD(data_layout)
        .describe(
            "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Convolution is applied on the 'D', 'H', and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .describe(
            "Dimension ordering of weight. Can be 'OIDHW', 'OIDHW16o16i', etc."
            "'O', 'I', 'D', 'H', 'W' stands for num_filter, input_channel, depth, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
  }
};  // struct Conv3dAttrs

/*! \brief Attributes used in Conv1DTranspose operator */
struct Conv1DTransposeAttrs : public tvm::AttrsNode<Conv1DTransposeAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> output_padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv1DTransposeAttrs, "relax.attrs.Conv1DTransposeAttrs") {
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on both sides"
        "two int : padding width in the order of (left, right)");
    TVM_ATTR_FIELD(output_padding).describe("Used to disambiguate the output shape.");
    TVM_ATTR_FIELD(dilation).describe(
        "Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).describe(
        "Number of groups to split the input into for grouped convolution. The number of input and "
        "output channels should be divisible by the number of groups.");
    TVM_ATTR_FIELD(data_layout)
        .describe(
            "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, width"
            "dimensions respectively. Convolution is applied on the 'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .describe(
            "Dimension ordering of weight. Can be 'OIW', 'IOW', etc."
            "'O', 'I', 'W' stands for num_filter, input_channel, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
  }
};  // struct Conv1DTransposeAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DTransposeAttrs : public tvm::AttrsNode<Conv2DTransposeAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> output_padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DTransposeAttrs, "relax.attrs.Conv2DTransposeAttrs") {
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "two int : bottom, right will use same padding as top, left"
        "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(output_padding).describe("Used to disambiguate the output shape.");
    TVM_ATTR_FIELD(dilation).describe(
        "Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).describe(
        "Number of groups to split the input into for grouped convolution. The number of input and "
        "output channels should be divisible by the number of groups.");
    TVM_ATTR_FIELD(data_layout)
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
  }
};  // struct Conv2DTransposeAttrs

/*! \brief Attributes used in max_pool1d and avg_pool1d operator */
struct Pool1DAttrs : public tvm::AttrsNode<Pool1DAttrs> {
  Array<IntImm> pool_size;
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  bool ceil_mode;
  bool count_include_pad;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(Pool1DAttrs, "relax.attrs.Pool1DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation).describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "two int : padding width in the order of (left, right)");
    TVM_ATTR_FIELD(ceil_mode).describe(
        "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
        "every element in the input tensor will be covered by a sliding window.");
    TVM_ATTR_FIELD(count_include_pad)
        .describe("When true, will include padding to compute the average");
    TVM_ATTR_FIELD(layout).set_default("NCW").describe(
        "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
        "'N', 'C', 'W' stands for batch, channel, and width"
        "dimensions respectively. Pooling is applied on the 'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel, and width"
            "dimensions respectively. Pooling is applied on the 'W' dimensions.");
  }
};  // struct Pool1dAttrs

/*! \brief Attributes used in max_pool2d and avg_pool2d operator */
struct Pool2DAttrs : public tvm::AttrsNode<Pool2DAttrs> {
  Array<IntImm> pool_size;
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  bool ceil_mode;
  bool count_include_pad;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(Pool2DAttrs, "relax.attrs.Pool2DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation).describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "two int : bottom, right will use same padding as top, left"
        "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(ceil_mode).describe(
        "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
        "every element in the input tensor will be covered by a sliding window.");
    TVM_ATTR_FIELD(count_include_pad)
        .describe("When true, will include padding to compute the average");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
  }
};  // struct Pool2dAttrs

/*! \brief Attributes used in max_pool3d and avg_pool3d operator */
struct Pool3DAttrs : public tvm::AttrsNode<Pool3DAttrs> {
  Array<IntImm> pool_size;
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  bool ceil_mode;
  bool count_include_pad;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(Pool3DAttrs, "relax.attrs.Pool3DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation).describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "three int : back, bottom, right will use same padding as front, top, left"
        "four int : padding width in the order of (front, top, left, back, bottom, right)");
    TVM_ATTR_FIELD(ceil_mode).describe(
        "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
        "every element in the input tensor will be covered by a sliding window.");
    TVM_ATTR_FIELD(count_include_pad)
        .describe("When true, will include padding to compute the average");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Pooling is applied on the 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Pooling is applied on the 'D', 'H' and"
            "'W' dimensions.");
  }
};  // struct Pool3dAttrs

/*! \brief Attributes for 1d adaptive pool operator */
struct AdaptivePool1DAttrs : public tvm::AttrsNode<AdaptivePool1DAttrs> {
  Optional<Array<IntImm>> output_size;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(AdaptivePool1DAttrs, "relax.attrs.AdaptivePool1DAttrs") {
    TVM_ATTR_FIELD(output_size).describe("Output width.");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
        "'N', 'C', 'W' stands for batch, channel and width"
        "dimensions respectively. Pooling is applied on the"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
            "'N', 'C', 'W' stands for batch, channel and width"
            "dimensions respectively. Pooling is applied on the"
            "'W' dimensions.");
  }
};  // struct AdaptivePool1DAttrs

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrs : public tvm::AttrsNode<AdaptivePool2DAttrs> {
  Optional<Array<IntImm>> output_size;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(AdaptivePool2DAttrs, "relax.attrs.AdaptivePool2DAttrs") {
    TVM_ATTR_FIELD(output_size).describe("Output height and width.");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
  }
};  // struct AdaptivePool2DAttrs

/*! \brief Attributes for 3d adaptive pool operator */
struct AdaptivePool3DAttrs : public tvm::AttrsNode<AdaptivePool3DAttrs> {
  Optional<Array<IntImm>> output_size;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(AdaptivePool3DAttrs, "relax.attrs.AdaptivePool3DAttrs") {
    TVM_ATTR_FIELD(output_size).describe("Output depth, height and width.");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Pooling is applied on 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
            "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
            "dimensions respectively. Pooling is applied on 'D', 'H' and"
            "'W' dimensions.");
  }
};  // struct AdaptivePool3DAttrs

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public tvm::AttrsNode<SoftmaxAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SoftmaxAttrs, "relax.attrs.SoftmaxAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis to sum over when computing softmax.");
  }
};

/*! \brief Attributes used in softmax operators */
struct LeakyReluAttrs : public tvm::AttrsNode<LeakyReluAttrs> {
  double alpha;

  TVM_DECLARE_ATTRS(LeakyReluAttrs, "relax.attrs.LeakyReluAttrs") {
    TVM_ATTR_FIELD(alpha).describe("The slope of the negative part.");
  }
};

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public tvm::AttrsNode<BatchNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;
  double momentum;

  TVM_DECLARE_ATTRS(BatchNormAttrs, "relax.attrs.BatchNormAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis along which the normalization is applied.");
    TVM_ATTR_FIELD(epsilon).describe("Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).describe(
        "Indicating if the beta offset will be added to the normalized tensor.");
    TVM_ATTR_FIELD(scale).describe("Indicating if the gamma scale will be multiplied.");
    TVM_ATTR_FIELD(momentum).describe("The value used for the moving_mean and moving_var update.");
  }
};  // struct BatchNormAttrs

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public tvm::AttrsNode<LayerNormAttrs> {
  Array<Integer> axes;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(LayerNormAttrs, "relax.attrs.LayerNormAttrs") {
    TVM_ATTR_FIELD(axes).describe("The axes that along which the normalization is applied.");
    TVM_ATTR_FIELD(epsilon).describe("Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).describe(
        "Indicating if the beta offset will be added to the normalized tensor.");
    TVM_ATTR_FIELD(scale).describe("Indicating if the gamma scale will be multiplied.");
  }
};  // struct LayerNormAttrs

/*! \brief Attributes used in group_norm operator */
struct GroupNormAttrs : public tvm::AttrsNode<GroupNormAttrs> {
  int num_groups;
  int channel_axis;
  Array<Integer> axes;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(GroupNormAttrs, "relax.attrs.GroupNormAttrs") {
    TVM_ATTR_FIELD(num_groups).describe("The number of groups to separate the channels into.");
    TVM_ATTR_FIELD(channel_axis).describe("The axis that represents the channel.");
    TVM_ATTR_FIELD(axes).describe(
        "The axes that along which the normalization is applied (excluding the channel axis).");
    TVM_ATTR_FIELD(epsilon).describe("Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).describe(
        "Indicating if the beta offset will be added to the normalized tensor.");
    TVM_ATTR_FIELD(scale).describe("Indicating if the gamma scale will be multiplied.");
  }
};  // struct GroupNormAttrs

/*! \brief Attributes used in rms_norm operator */
struct RMSNormAttrs : public tvm::AttrsNode<RMSNormAttrs> {
  Array<Integer> axes;
  double epsilon;

  TVM_DECLARE_ATTRS(RMSNormAttrs, "relax.attrs.RMSNormAttrs") {
    TVM_ATTR_FIELD(axes).describe("The axes that along which the normalization is applied.");
    TVM_ATTR_FIELD(epsilon).describe("Small float added to variance to avoid dividing by zero");
  }
};  // struct RMSNormAttrs

/*! \brief Attributes used in nll_loss operator */
struct NLLLossAttrs : public tvm::AttrsNode<NLLLossAttrs> {
  String reduction;
  int ignore_index;

  TVM_DECLARE_ATTRS(NLLLossAttrs, "relax.attrs.NLLLossAttrs") {
    TVM_ATTR_FIELD(reduction).set_default("mean").describe(
        "The reduction method to apply to the output. Can be"
        "'none', 'mean' or 'sum'.");
    TVM_ATTR_FIELD(ignore_index).describe("The target value to ignore.");
  }
};  // struct NLLLossAttrs

/*! \brief Attributes used in dropout operator */
struct DropoutAttrs : public tvm::AttrsNode<DropoutAttrs> {
  double rate;

  TVM_DECLARE_ATTRS(DropoutAttrs, "relax.attrs.DropoutAttrs") {
    TVM_ATTR_FIELD(rate).describe(
        "Fraction of the input that gets dropped out during training time");
  }
};  // struct DropoutAttrs

/*! \brief Attributes used in Attention operator */
struct AttentionAttrs : public tvm::AttrsNode<AttentionAttrs> {
  Optional<FloatImm> scale;
  Optional<String> causal_mask;
  Optional<IntImm> window_size;

  TVM_DECLARE_ATTRS(AttentionAttrs, "relax.attrs.AttentionAttrs") {
    TVM_ATTR_FIELD(scale).describe(
        "The custom scale applied before the softmax. The default value is 1 / sqrt(head_dim).");
    TVM_ATTR_FIELD(causal_mask)
        .describe("The type of the causal mask, i.e. 'TopLeft' and 'BottomRight'.");
    TVM_ATTR_FIELD(window_size).describe("The size of the window for sliding-window attention.");
  }
};  // struct AttentionAttrs

/*! \brief Attributes used for the padding operator */
struct PadAttrs : public tvm::AttrsNode<PadAttrs> {
  Array<Integer> pad_width;
  tvm::String pad_mode;

  TVM_DECLARE_ATTRS(PadAttrs, "relay.attrs.PadAttrs") {
    TVM_ATTR_FIELD(pad_width).describe(
        "Number of values padded to the edges of each axis, "
        "in the format of (before_1, after_1, ..., before_N, after_N)");
    TVM_ATTR_FIELD(pad_mode)
        .set_default("constant")
        .describe(
            "Padding type to use. \"constant\" pads with constant_value, "
            "\"edge\" pads using the edge values of the input array, "
            "\"reflect\" pads by reflecting values with respect to the edges.");
  }
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_NN_H_
