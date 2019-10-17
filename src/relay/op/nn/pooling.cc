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
 *  Copyright (c) 2018 by Contributors
 * \file pooling.cc
 * \brief Pooling operators
 */
#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/nn.h>
#include <topi/nn/pooling.h>
#include <vector>
#include "../../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

// relay.nn.max_pool2d & relay.nn.avg_pool2d
TVM_REGISTER_NODE_TYPE(MaxPool2DAttrs);
TVM_REGISTER_NODE_TYPE(AvgPool2DAttrs);

template <typename T>
Array<Array<Layout> > Pool2DInferCorrectLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr>> &old_in_shapes) {
  // NOTE: Discard "const" qualifier here.
  T *params = const_cast<T*>(attrs.as<T>());

  if (new_in_layouts.defined()) {
    // Set the pool with the new layout.
    CHECK_EQ(new_in_layouts.size(), 1);
    params->layout = new_in_layouts[0].name();
  }

  Layout inferred_layout(params->layout);
  return Array<Array<Layout> >{{inferred_layout}, {inferred_layout}};
}

template <typename AttrType>
bool Pool2DRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) return false;

  const auto dshape = data->shape;
  CHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto param = attrs.as<AttrType>();
  CHECK(param != nullptr);

  Layout layout(param->layout);
  CHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
        !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
    << "Invalid layout " << layout
    << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));

  IndexExpr pad_h, pad_w;
  if (param->padding.size() == 1) {
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[0] * 2;
  } else if (param->padding.size() == 2) {
    // (top, left)
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[1] * 2;
  } else if (param->padding.size() == 4) {
    // (top, left, bottom, right)
    pad_h = param->padding[0] + param->padding[2];
    pad_w = param->padding[1] + param->padding[3];
  } else {
    return false;
  }

  std::vector<IndexExpr> oshape;
  for (const auto& e : dshape) {
    oshape.push_back(e);
  }

  if (param->ceil_mode) {
    oshape[hidx] = ((dshape[hidx] + pad_h - param->pool_size[0] +
                    param->strides[0] - 1) / param->strides[0]) + 1;
    oshape[widx] = ((dshape[widx] + pad_w - param->pool_size[1] +
                    param->strides[1] - 1) / param->strides[1]) + 1;
  } else {
    oshape[hidx] = ((dshape[hidx] + pad_h - param->pool_size[0]) / param->strides[0]) + 1;
    oshape[widx] = ((dshape[widx] + pad_w - param->pool_size[1]) / param->strides[1]) + 1;
  }

  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

// MaxPool2D
Expr MakeMaxPool2D(Expr data,
                   Array<IndexExpr> pool_size,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   std::string layout,
                   bool ceil_mode) {
  auto attrs = make_node<MaxPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->ceil_mode = ceil_mode;
  static const Op& op = Op::Get("nn.max_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

template<typename AttrType, topi::nn::PoolType mode>
Array<Tensor> Pool2DCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type,
                            const Target& target) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  Layout layout(param->layout);

  CHECK(BijectiveLayoutNode::make(layout, kNCHW).defined())
      << "max_pool2d currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "max_pool2d does not support input split on height";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "max_pool2d does not support input split on width";

  CHECK(inputs[0].ndim() == 4U ||
        inputs[0].ndim() == 5U ||
        inputs[0].ndim() == 6U)
      << "Pool2D only support 4-D input (e.g., NCHW)"
      << " or 5-D input (e.g. NCHWc on for vector instructions)"
      << " or 6-D input (e.g. NCHWnc for tensor accelerators)";

  if (param->padding.size() == 1) {
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
  } else if (param->padding.size() == 2) {
    padding.push_back(padding[0]);
    padding.push_back(padding[1]);
  }
  if (mode == topi::nn::kAvgPool) {
    bool count_include_pad = reinterpret_cast<const AvgPool2DAttrs*>(param)->count_include_pad;
    return Array<Tensor>{
      topi::nn::pool(inputs[0], pool_size, strides, padding,
                     mode, ceil_mode, layout.name(), count_include_pad)};
  } else {
    return Array<Tensor>{
      topi::nn::pool(inputs[0], pool_size, strides, padding,
                     mode, ceil_mode, layout.name())};
  }
}

TVM_REGISTER_API("relay.op.nn._make.max_pool2d")
.set_body_typed(MakeMaxPool2D);


RELAY_REGISTER_OP("nn.max_pool2d")
.describe(R"code(Max pooling operation for two dimensional data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.
           out_height and out_width are calculated as::

               out_height = floor((height+padding[0]+padding[2]-pool_size[0])/strides[0])+1
               out_width = floor((width+padding[1]+padding[3]-pool_size[1])/strides[1])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               two int : bottom, right use same as top and left.
               four int: padding width in the order of (top, left, bottom, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.MaxPool2DAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("MaxPool2D", Pool2DRel<MaxPool2DAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", Pool2DInferCorrectLayout<MaxPool2DAttrs>)
.set_attr<FTVMCompute>("FTVMCompute", Pool2DCompute<MaxPool2DAttrs, topi::nn::kMaxPool>);


// AvgPool2D
Expr MakeAvgPool2D(Expr data,
                   Array<IndexExpr> pool_size,
                   Array<IndexExpr> strides,
                   Array<IndexExpr> padding,
                   std::string layout,
                   bool ceil_mode,
                   bool count_include_pad) {
  auto attrs = make_node<AvgPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  static const Op& op = Op::Get("nn.avg_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.avg_pool2d")
.set_body_typed(MakeAvgPool2D);


RELAY_REGISTER_OP("nn.avg_pool2d")
.describe(R"code(
Average pooling operation for one dimensional data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.
           out_height and out_width are calculated as::

               out_height = floor((height+padding[0]+padding[2]-pool_size[0])/strides[0])+1
               out_width = floor((width+padding[1]+padding[3]-pool_size[1])/strides[1])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               two int : bottom, right use same as top and left.
               four int: padding width in the order of (top, left, bottom, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.AvgPool2DAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("AvgPool2D", Pool2DRel<AvgPool2DAttrs>)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", Pool2DInferCorrectLayout<AvgPool2DAttrs>)
.set_attr<FTVMCompute>("FTVMCompute", Pool2DCompute<AvgPool2DAttrs, topi::nn::kAvgPool>);

// relay.nn.global_pool_2d & relay.nn.max_pool_2d
TVM_REGISTER_NODE_TYPE(GlobalPool2DAttrs);

bool GlobalPool2DRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) { return false; }
  const auto dshape = data->shape;
  CHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto param = attrs.as<GlobalPool2DAttrs>();
  CHECK(param != nullptr);

  Layout layout(param->layout);
  CHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
        !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
    << "Invalid layout " << layout
    << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  Array<IndexExpr> oshape(dshape);
  oshape.Set(hidx, 1);
  oshape.Set(widx, 1);

  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}


template<topi::nn::PoolType mode>
Array<Tensor> GlobalPool2DCompute(const Attrs& attrs,
                                  const Array<Tensor>& inputs,
                                  const Type& out_type,
                                  const Target& target) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<GlobalPool2DAttrs>();
  CHECK(param != nullptr);
  Layout layout(param->layout);
  CHECK(BijectiveLayoutNode::make(layout, kNCHW).defined())
    << "global_avg_pool2d currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
    << "global_avg_pool2d does not support input split on height";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
    << "global_avg_pool2d does not support input split on width";

  CHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
    << "Pool2D only support 4-D input (e.g., NCHW)"
    << " or 5-D input (last dimension is a split of channel)";
  return Array<Tensor>{
    topi::nn::global_pool(inputs[0], mode, layout.name()) };
}

Expr MakeGlobalAvgPool2D(Expr data,
                         std::string layout) {
  auto attrs = make_node<GlobalPool2DAttrs>();
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("nn.global_avg_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.nn._make.global_avg_pool2d")
.set_body_typed(MakeGlobalAvgPool2D);

// GlobalAvgPool
RELAY_REGISTER_OP("nn.global_avg_pool2d")
.describe(R"code(Global average pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.GlobalPool2DAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("GlobalAvgPool2D", GlobalPool2DRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               Pool2DInferCorrectLayout<GlobalPool2DAttrs>)
.set_attr<FTVMCompute>("FTVMCompute", GlobalPool2DCompute<topi::nn::kAvgPool>);

// GlobalMaxPool
Expr MakeGlobalMaxPool2D(Expr data,
                         std::string layout) {
  auto attrs = make_node<GlobalPool2DAttrs>();
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("nn.global_max_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.global_max_pool2d")
.set_body_typed(MakeGlobalMaxPool2D);


RELAY_REGISTER_OP("nn.global_max_pool2d")
.describe(R"code(Global max pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.GlobalPool2DAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("GlobalMaxPool2D", GlobalPool2DRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               Pool2DInferCorrectLayout<GlobalPool2DAttrs>)
.set_attr<FTVMCompute>("FTVMCompute", GlobalPool2DCompute<topi::nn::kMaxPool>);


// relay.nn.adaptive_pool_2d
TVM_REGISTER_NODE_TYPE(AdaptivePool2DAttrs);

bool AdaptivePool2DRel(const Array<Type>& types,
                       int num_inputs,
                       const Attrs& attrs,
                       const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) { return false; }
  const auto dshape = data->shape;
  CHECK_GE(dshape.size(), 2U)
    << "Pool2D only support input >= 2-D: input must have height and width";
  const auto* param = attrs.as<AdaptivePool2DAttrs>();
  CHECK(param != nullptr);

  Layout layout(param->layout);
  CHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
        !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
    << "Invalid layout " << layout
    << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  Array<IndexExpr> oshape(dshape);
  auto output_size = param->output_size;
  CHECK_LE(output_size.size(), 2U)
    << "output_size can have up to 2 elements.";
  IndexExpr output_height, output_width;
  if (output_size.empty()) {
    output_height = dshape[hidx];
    output_width = dshape[widx];
  } else if (output_size.size() == 1) {
    output_height = output_size[0];
    output_width = output_size[0];
  } else {
    output_height = output_size[0];
    output_width = output_size[1];
  }

  oshape.Set(hidx, output_height);
  oshape.Set(widx, output_width);

  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

template<topi::nn::PoolType mode>
Array<Tensor> AdaptivePool2DCompute(const Attrs& attrs,
                                    const Array<Tensor>& inputs,
                                    const Type& out_type,
                                    const Target& target) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<AdaptivePool2DAttrs>();
  CHECK(param != nullptr);
  Layout layout(param->layout);
  CHECK(BijectiveLayoutNode::make(layout, kNCHW).defined())
    << "Adaptive pool2d currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
    << "Adaptive pool2d does not support input split on height";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
    << "Adaptive pool2d does not support input split on width";

  CHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
    << "Pool2D only support 4-D input (e.g., NCHW)"
    << " or 5-D input (last dimension is a split of channel)";

  auto output_size = param->output_size;
  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  IndexExpr output_height, output_width;
  if (output_size.empty()) {
    output_height = inputs[0]->shape[hidx];
    output_width = inputs[0]->shape[widx];
  } else if (output_size.size() == 1) {
    output_height = output_size[0];
    output_width = output_size[0];
  } else {
    output_height = output_size[0];
    output_width = output_size[1];
  }
  return Array<Tensor>{
    topi::nn::adaptive_pool(inputs[0], Array<IndexExpr>{ output_height, output_width },
                            mode, layout.name()) };
}

// relay.contrib.adaptive_avg_pool2d
Expr MakeAdaptiveAvgPool2D(Expr data,
                           Array<IndexExpr> output_size,
                           std::string layout) {
  auto attrs = make_node<AdaptivePool2DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("contrib.adaptive_avg_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.contrib._make.adaptive_avg_pool2d")
.set_body_typed(MakeAdaptiveAvgPool2D);

RELAY_REGISTER_OP("contrib.adaptive_avg_pool2d")
  .describe(R"code(Adaptive average pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **output_size**: If this argument is not provided, input height and width will be used
                   as output height and width.
                   If a single integer is provided for output_size, the output size is
                   (N x C x output_size x output_size) for any input (NCHW).
                   If a tuple of integers (height, width) are provided for output_size,
                   the output size is (N x C x height x width) for any input (NCHW).
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, output_height, output_width)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.AdaptivePool2DAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(10)
.add_type_rel("AdaptiveAvgPool2D", AdaptivePool2DRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               Pool2DInferCorrectLayout<AdaptivePool2DAttrs>)
.set_attr<FTVMCompute>("FTVMCompute", AdaptivePool2DCompute<topi::nn::kAvgPool>);


// relay.contrib.adaptive_max_pool2d
Expr MakeAdaptiveMaxPool2D(Expr data,
                           Array<IndexExpr> output_size,
                           std::string layout) {
  auto attrs = make_node<AdaptivePool2DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("contrib.adaptive_max_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.contrib._make.adaptive_max_pool2d")
.set_body_typed(MakeAdaptiveMaxPool2D);

RELAY_REGISTER_OP("contrib.adaptive_max_pool2d")
  .describe(R"code(Adaptive max pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **output_size**: If this argument is not provided, input height and width will be used
                   as output height and width.
                   If a single integer is provided for output_size, the output size is
                   (N x C x output_size x output_size) for any input (NCHW).
                   If a tuple of integers (height, width) are provided for output_size,
                   the output size is (N x C x height x width) for any input (NCHW).
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, output_height, output_width)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.AdaptivePool2DAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(10)
.add_type_rel("AdaptiveMaxPool2D", AdaptivePool2DRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               Pool2DInferCorrectLayout<AdaptivePool2DAttrs>)
.set_attr<FTVMCompute>("FTVMCompute", AdaptivePool2DCompute<topi::nn::kMaxPool>);


bool Pool2DGradRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[1].as<TensorTypeNode>();

  if (data == nullptr) return false;

  // assign output type
  reporter->Assign(types[2], types[1]);
  return true;
}

template <typename AttrType, topi::nn::PoolType mode>
Array<Tensor> Pool2DGradCompute(const Attrs& attrs, const Array<Tensor>& inputs,
                                const Type& out_type, const Target& target) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  CHECK_EQ(inputs.size(), 2);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  Layout layout(param->layout);

  CHECK(BijectiveLayoutNode::make(layout, kNCHW).defined())
      << "pool2d_grad currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "pool2d_grad does not support input split on height";
  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "pool2d_grad does not support input split on width";

  CHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
      << "Pool2DGrad only support 4-D output gradient (e.g., NCHW)"
      << " or 5-D output gradient (last dimension is a split of channel)";

  CHECK(inputs[1].ndim() == 4U || inputs[1].ndim() == 5U)
      << "Pool2DGrad only support 4-D input (e.g., NCHW)"
      << " or 5-D input (last dimension is a split of channel)";

  if (param->padding.size() == 1) {
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
  } else if (param->padding.size() == 2) {
    padding.push_back(padding[0]);
    padding.push_back(padding[1]);
  }
  if (mode == topi::nn::kAvgPool) {
    bool count_include_pad = reinterpret_cast<const AvgPool2DAttrs*>(param)->count_include_pad;
    return Array<Tensor>{topi::nn::pool_grad(inputs[0], inputs[1], pool_size, strides, padding,
        mode, ceil_mode, layout.name(), count_include_pad)};
  } else {
    return Array<Tensor>{topi::nn::pool_grad(inputs[0], inputs[1], pool_size, strides, padding,
        mode, ceil_mode, layout.name())};
  }
}


// MaxPool2DGrad
Expr MakeMaxPool2DGrad(Expr out_grad, Expr data, Array<IndexExpr> pool_size,
    Array<IndexExpr> strides, Array<IndexExpr> padding, std::string layout, bool ceil_mode) {
  auto attrs = make_node<MaxPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->ceil_mode = ceil_mode;
  static const Op& op = Op::Get("nn.max_pool2d_grad");
  return CallNode::make(op, {out_grad, data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.max_pool2d_grad").set_body_typed(MakeMaxPool2DGrad);


RELAY_REGISTER_OP("nn.max_pool2d_grad")
    .describe(R"code(Gradient of max pooling operation for two dimensional data.

- **out_grad**: This depends on the `layout` parameter. Output gradient is 4D array of
                shape (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
                out_height and out_width are are the output size of the pooling operation,
                which are calculated as::
                    out_height = floor((height+padding[0]+padding[2]-pool_size[0])/strides[0])+1
                    out_width = floor((width+padding[1]+padding[3]-pool_size[1])/strides[1])+1

                where padding will be an expanded array based on number of values passed as::
                    one int : all sides same padding used.
                    two int : bottom, right use same as top and left.
                    four int: padding width in the order of (top, left, bottom, right).

                When `ceil_mode` is `True`, ceil will be used instead of floor in this
                equation.
- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **grad**: This depends on the `layout` parameter. Grad is 4D array of shape
           (batch_size, channels, height, width)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.MaxPool2DAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("MaxPool2DGrad", Pool2DGradRel)
.set_attr<FTVMCompute>("FTVMCompute", Pool2DGradCompute<MaxPool2DAttrs, topi::nn::kMaxPool>);


// AvgPool2DGrad
Expr MakeAvgPool2DGrad(Expr out_grad, Expr data, Array<IndexExpr> pool_size,
    Array<IndexExpr> strides, Array<IndexExpr> padding, std::string layout, bool ceil_mode,
    bool count_include_pad) {
  auto attrs = make_node<AvgPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  static const Op& op = Op::Get("nn.avg_pool2d_grad");
  return CallNode::make(op, {out_grad, data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.avg_pool2d_grad").set_body_typed(MakeAvgPool2DGrad);


RELAY_REGISTER_OP("nn.avg_pool2d_grad")
    .describe(R"code(Gradient of average pooling operation for two dimensional data.

- **out_grad**: This depends on the `layout` parameter. Output gradient is 4D array of
                shape (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
                out_height and out_width are are the output size of the pooling operation,
                which are calculated as::
                    out_height = floor((height+padding[0]+padding[2]-pool_size[0])/strides[0])+1
                    out_width = floor((width+padding[1]+padding[3]-pool_size[1])/strides[1])+1

                where padding will be an expanded array based on number of values passed as::
                    one int : all sides same padding used.
                    two int : bottom, right use same as top and left.
                    four int: padding width in the order of (top, left, bottom, right).

                When `ceil_mode` is `True`, ceil will be used instead of floor in this
                equation.
- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **grad**: This depends on the `layout` parameter. Grad is 4D array of shape
           (batch_size, channels, height, width)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.MaxPool2DAttrs")
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("MaxPool2DGrad", Pool2DGradRel)
.set_attr<FTVMCompute>("FTVMCompute", Pool2DGradCompute<AvgPool2DAttrs, topi::nn::kAvgPool>);


}  // namespace relay
}  // namespace tvm
