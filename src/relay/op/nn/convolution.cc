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
#include <tvm/tir/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../../transforms/infer_layout_util.h"
#include "../op_common.h"
#include "convolution.h"

namespace tvm {
namespace relay {

template <typename T>
Expr MakeConv(Expr data,
              Expr weight,
              Array<IndexExpr> strides,
              Array<IndexExpr> padding,
              Array<IndexExpr> dilation,
              int groups,
              IndexExpr channels,
              Array<IndexExpr> kernel_size,
              std::string data_layout,
              std::string kernel_layout,
              std::string out_layout,
              DataType out_dtype,
              std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

template <typename T>
Expr MakeConvWinograd(Expr data,
                      Expr weight,
                      int tile_size,
                      Array<IndexExpr> strides,
                      Array<IndexExpr> padding,
                      Array<IndexExpr> dilation,
                      int groups,
                      IndexExpr channels,
                      Array<IndexExpr> kernel_size,
                      std::string data_layout,
                      std::string kernel_layout,
                      std::string out_layout,
                      DataType out_dtype,
                      std::string op_name) {
  auto attrs = make_object<T>();
  attrs->tile_size = tile_size;
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

Expr MakeConvWinogradWeightTransform(Expr weight,
                                     int tile_size,
                                     std::string op_name) {
  auto attrs = make_object<ConvWinogradWeightTransformAttrs>();
  attrs->tile_size = tile_size;
  const Op& op = Op::Get(op_name);
  return Call(op, {weight}, Attrs(attrs), {});
}

template <typename T>
Expr MakeConvTranspose(Expr data,
                       Expr weight,
                       Array<IndexExpr> strides,
                       Array<IndexExpr> padding,
                       Array<IndexExpr> dilation,
                       int groups,
                       IndexExpr channels,
                       Array<IndexExpr> kernel_size,
                       std::string data_layout,
                       std::string kernel_layout,
                       std::string out_layout,
                       Array<IndexExpr> output_padding,
                       DataType out_dtype,
                       std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->output_padding = std::move(output_padding);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

template <typename T>
Expr MakeDeformableConv(Expr data,
                        Expr offset,
                        Expr weight,
                        Array<IndexExpr> strides,
                        Array<IndexExpr> padding,
                        Array<IndexExpr> dilation,
                        int deformable_groups,
                        int groups,
                        int channels,
                        Array<IndexExpr> kernel_size,
                        std::string data_layout,
                        std::string kernel_layout,
                        std::string out_layout,
                        DataType out_dtype,
                        std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = strides;
  attrs->padding = padding;
  attrs->dilation = dilation;
  attrs->deformable_groups = deformable_groups;
  attrs->groups = groups;
  attrs->channels = channels;
  attrs->kernel_size = kernel_size;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = kernel_layout;
  attrs->out_layout = out_layout;
  attrs->out_dtype = out_dtype;
  const Op& op = Op::Get(op_name);
  return Call(op, {data, offset, weight}, Attrs{attrs}, {});
}


// relay.nn.conv1d
TVM_REGISTER_NODE_TYPE(Conv1DAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv1d")
.set_body_typed([](Expr data,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeConv<Conv1DAttrs>(
    data, weight, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.conv1d");
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
.add_type_rel("Conv1D", Conv1DRel<Conv1DAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv1DAttrs>);


// relay.nn.conv2d
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv2d")
.set_body_typed([](Expr data,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeConv<Conv2DAttrs>(
    data, weight, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.conv2d");
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
.add_type_rel("Conv2D", Conv2DRel<Conv2DAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv2DAttrs>);


// relay.nn.conv3d
TVM_REGISTER_NODE_TYPE(Conv3DAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv3d")
.set_body_typed([](Expr data,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeConv<Conv3DAttrs>(
    data, weight, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.conv3d");
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
.add_type_rel("Conv3D", Conv3DRel<Conv3DAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<Conv3DAttrs>);


// relay.nn.conv2d_transpose
TVM_REGISTER_NODE_TYPE(Conv2DTransposeAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv2d_transpose")
.set_body_typed([](Expr data,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   Array<IndexExpr> output_padding,
                   DataType out_dtype) {
  return MakeConvTranspose<Conv2DTransposeAttrs>(
    data, weight, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
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
.add_type_rel("Conv2DTranspose", Conv2DTransposeRel<Conv2DTransposeAttrs>);

// relay.nn.conv1d_transpose
TVM_REGISTER_NODE_TYPE(Conv1DTransposeAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.conv1d_transpose")
.set_body_typed([](Expr data,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   Array<IndexExpr> output_padding,
                   DataType out_dtype) {
  return MakeConvTranspose<Conv1DTransposeAttrs>(
    data, weight, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
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
.add_type_rel("Conv1DTranspose", Conv1DTransposeRel<Conv1DTransposeAttrs>);

// relay.nn.contrib_conv2d_winograd_without_weight_transform
TVM_REGISTER_NODE_TYPE(Conv2DWinogradAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform")
.set_body_typed([](Expr data,
                   Expr weight,
                   int tile_size,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeConvWinograd<Conv2DWinogradAttrs>(
    data, weight, tile_size, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.contrib_conv2d_winograd_without_weight_transform");
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
        ConvInferCorrectLayout<Conv2DWinogradAttrs>);

// relay.nn.contrib_conv2d_winograd_weight_transform
TVM_REGISTER_NODE_TYPE(ConvWinogradWeightTransformAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_winograd_weight_transform")
.set_body_typed([](Expr weight,
                   int tile_size) {
  return MakeConvWinogradWeightTransform(
    weight, tile_size, "nn.contrib_conv2d_winograd_weight_transform");
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
.add_type_rel("Conv2DWinogradWeightTransform", Conv2DWinogradWeightTransformRel);

// relay.nn.contrib_conv3d_winograd_without_weight_transform
TVM_REGISTER_NODE_TYPE(Conv3DWinogradAttrs);

TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform")
.set_body_typed([](Expr data,
                   Expr weight,
                   int tile_size,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeConvWinograd<Conv3DWinogradAttrs>(
    data, weight, tile_size, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.contrib_conv3d_winograd_without_weight_transform");
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
.add_type_rel("Conv3DWinograd", Conv3DWinogradRel<Conv3DWinogradAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                ConvInferCorrectLayout<Conv3DWinogradAttrs>);

// relay.nn.contrib_conv3d_winograd_weight_transform
TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv3d_winograd_weight_transform")
.set_body_typed([](Expr weight,
                   int tile_size) {
  return MakeConvWinogradWeightTransform(
    weight, tile_size, "nn.contrib_conv3d_winograd_weight_transform");
});

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
.add_type_rel("Conv3DWinogradWeightTransform", Conv3DWinogradWeightTransformRel);


// relay.nn.contrib_conv2d_winograd_nnpack_weight_transform
TVM_REGISTER_NODE_TYPE(Conv2DWinogradNNPACKWeightTransformAttrs);

Expr MakeConv2DWinogradNNPACKWeightTransform(Expr weight,
                                             int convolution_algorithm,
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
.add_type_rel("Conv2DWinogradNNPACKWeightTransform", Conv2DWinogradNNPACKWeightTransformRel);


// Positional relay function to create conv2d NCHWc operator
// used by frontend FFI.
TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_conv2d_NCHWc")
.set_body_typed([](Expr data,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeConv<Conv2DAttrs>(
    data, weight, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.contrib_conv2d_NCHWc");
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
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
        ConvInferCorrectLayout<Conv2DAttrs>);


// Positional relay function to create depthwise conv2d NCHWc operator
// used by frontend FFI.
TVM_REGISTER_GLOBAL("relay.op.nn._make.contrib_depthwise_conv2d_NCHWc")
.set_body_typed([](Expr data,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int groups,
                   IndexExpr channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeConv<Conv2DAttrs>(
    data, weight, strides, padding, dilation,
    groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.contrib_depthwise_conv2d_NCHWc");
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
.add_type_rel("Conv2D", Conv2DRel<Conv2DAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
        ConvInferCorrectLayout<Conv2DAttrs>);


TVM_REGISTER_NODE_TYPE(DeformableConv2DAttrs);

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
.add_type_rel("DeformableConv2D", DeformableConv2DRel<DeformableConv2DAttrs>);

// Positional relay function to create deformable_conv2d operator
// used by frontend FFI.
TVM_REGISTER_GLOBAL("relay.op.nn._make.deformable_conv2d")
.set_body_typed([](Expr data,
                   Expr offset,
                   Expr weight,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   Array<IndexExpr> dilation,
                   int deformable_groups,
                   int groups,
                   int channels,
                   Array<IndexExpr> kernel_size,
                   std::string data_layout,
                   std::string kernel_layout,
                   std::string out_layout,
                   DataType out_dtype) {
  return MakeDeformableConv<DeformableConv2DAttrs>(
    data, offset, weight, strides, padding, dilation,
    deformable_groups, groups, channels, kernel_size, data_layout,
    kernel_layout, out_layout, out_dtype, "nn.deformable_conv2d");
});

}  // namespace relay
}  // namespace tvm
