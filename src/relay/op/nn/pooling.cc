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
 * \file pooling.cc
 * \brief Pooling operators
 */
#include "pooling.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/nn/pooling.h>

#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "pooling_common.h"

namespace tvm {
namespace relay {

// relay.nn.max_pool2d & relay.nn.avg_pool2d
TVM_REGISTER_NODE_TYPE(MaxPool2DAttrs);
TVM_REGISTER_NODE_TYPE(AvgPool2DAttrs);

template <typename AttrType>
bool Pool2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) return false;

  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto param = attrs.as<AttrType>();
  ICHECK(param != nullptr);

  Layout layout(param->layout);
  ICHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
         !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout << ". Pool2D layout must have H and W, which cannot be split";

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

  std::vector<IndexExpr> oshape(dshape.begin(), dshape.end());

  if (dshape[hidx].as<tir::AnyNode>()) {
    oshape[hidx] = dshape[hidx];
  } else {
    oshape[hidx] =
        calculate_pool_dimension(dshape[hidx], pad_h, param->pool_size[0], param->dilation[0],
                                 param->strides[0], param->ceil_mode);
  }
  if (dshape[widx].as<tir::AnyNode>()) {
    oshape[widx] = dshape[widx];
  } else {
    oshape[widx] =
        calculate_pool_dimension(dshape[widx], pad_w, param->pool_size[1], param->dilation[1],
                                 param->strides[1], param->ceil_mode);
  }

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

template <typename AttrType, topi::nn::PoolType mode>
Array<te::Tensor> Pool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto dilation = param->dilation;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  Layout layout(param->layout);
  Layout out_layout(param->out_layout);

  ICHECK(tir::BijectiveLayout(layout, kNCHW).defined())
      << "max_pool2d currently only supports layouts that are convertible from NCHW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "max_pool2d does not support input split on height";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "max_pool2d does not support input split on width";

  ICHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U || inputs[0].ndim() == 6U)
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
    return Array<te::Tensor>{topi::nn::pool2d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name(), count_include_pad)};
  } else {
    return Array<te::Tensor>{topi::nn::pool2d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name())};
  }
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.max_pool2d")
    .set_body_typed([](Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                       Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode) {
      return MakeMaxPool<MaxPool2DAttrs>(data, pool_size, strides, dilation, padding, layout,
                                         out_layout, ceil_mode, "nn.max_pool2d");
    });

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
    .set_attrs_type<MaxPool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("MaxPool2D", Pool2DRel<MaxPool2DAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<MaxPool2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool2DCompute<MaxPool2DAttrs, topi::nn::kMaxPool>);

// AvgPool2D
TVM_REGISTER_GLOBAL("relay.op.nn._make.avg_pool2d")
    .set_body_typed([](Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                       Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode, bool count_include_pad) {
      return MakeAvgPool<AvgPool2DAttrs>(data, pool_size, strides, dilation, padding, layout,
                                         out_layout, ceil_mode, count_include_pad, "nn.avg_pool2d");
    });

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
    .set_attrs_type<AvgPool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("AvgPool2D", Pool2DRel<AvgPool2DAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<AvgPool2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool2DCompute<AvgPool2DAttrs, topi::nn::kAvgPool>);

// relay.nn.global_pool_2d & relay.nn.max_pool_2d
TVM_REGISTER_NODE_TYPE(GlobalPool2DAttrs);

bool GlobalPool2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto param = attrs.as<GlobalPool2DAttrs>();
  ICHECK(param != nullptr);

  Layout layout(param->layout);
  ICHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
         !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  Array<IndexExpr> oshape(dshape);
  oshape.Set(hidx, 1);
  oshape.Set(widx, 1);

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

template <topi::nn::PoolType mode>
Array<te::Tensor> GlobalPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                      const Type& out_type) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<GlobalPool2DAttrs>();
  ICHECK(param != nullptr);
  Layout layout(param->layout);
  ICHECK(tir::BijectiveLayout(layout, kNCHW).defined())
      << "global_avg_pool2d currently only supports layouts that are convertible from NCHW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "global_avg_pool2d does not support input split on height";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "global_avg_pool2d does not support input split on width";

  ICHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
      << "Pool2D only support 4-D input (e.g., NCHW)"
      << " or 5-D input (last dimension is a split of channel)";
  return Array<te::Tensor>{topi::nn::global_pool(inputs[0], mode, layout.name())};
}

Expr MakeGlobalAvgPool2D(Expr data, String layout, String out_layout) {
  auto attrs = make_object<GlobalPool2DAttrs>();
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.global_avg_pool2d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.global_avg_pool2d").set_body_typed(MakeGlobalAvgPool2D);

// GlobalAvgPool
RELAY_REGISTER_OP("nn.global_avg_pool2d")
    .describe(R"code(Global average pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<GlobalPool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("GlobalAvgPool2D", GlobalPool2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<GlobalPool2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", GlobalPool2DCompute<topi::nn::kAvgPool>);

// GlobalMaxPool
Expr MakeGlobalMaxPool2D(Expr data, String layout, String out_layout) {
  auto attrs = make_object<GlobalPool2DAttrs>();
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.global_max_pool2d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.global_max_pool2d").set_body_typed(MakeGlobalMaxPool2D);

RELAY_REGISTER_OP("nn.global_max_pool2d")
    .describe(R"code(Global max pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<GlobalPool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("GlobalMaxPool2D", GlobalPool2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<GlobalPool2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", GlobalPool2DCompute<topi::nn::kMaxPool>);

// relay.nn.adaptive_pool_1d
TVM_REGISTER_NODE_TYPE(AdaptivePool1DAttrs);

bool AdaptivePool1DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 1U) << "Pool2D only support input >= 1-D: input must have width";
  const auto* param = attrs.as<AdaptivePool1DAttrs>();
  ICHECK(param != nullptr);

  Layout layout(param->layout);
  ICHECK(layout.Contains(LayoutAxis::Get('W')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout << ". Pool1D layout must have W, which cannot be split";

  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  Array<IndexExpr> oshape(dshape);
  auto output_size = param->output_size;
  ICHECK_LE(output_size.size(), 1U) << "output_size must have 1 element.";
  IndexExpr output_width;
  if (output_size.empty()) {
    output_width = dshape[widx];
  } else {
    output_width = output_size[0];
  }

  oshape.Set(widx, output_width);

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

template <topi::nn::PoolType mode>
Array<te::Tensor> AdaptivePool1DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                        const Type& out_type) {
  static const Layout kNCW("NCW");
  const auto* param = attrs.as<AdaptivePool1DAttrs>();
  ICHECK(param != nullptr);
  Layout layout(param->layout);
  ICHECK(tir::BijectiveLayout(layout, kNCW).defined())
      << "Adaptive pool1d currently only supports layouts that are convertible from NCW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "Adaptive pool2d does not support input split on width";

  ICHECK(inputs[0].ndim() == 3U || inputs[0].ndim() == 4U)
      << "Pool1D only support 3-D input (e.g., NCW)"
      << " or 4-D input (last dimension is a split of channel)";

  auto output_size = param->output_size;
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  IndexExpr output_width;
  if (output_size.empty()) {
    output_width = inputs[0]->shape[widx];
  } else {
    output_width = output_size[0];
  }
  return Array<te::Tensor>{
      topi::nn::adaptive_pool1d(inputs[0], Array<IndexExpr>{output_width}, mode, layout.name())};
}

// relay.nn.adaptive_avg_pool1d
Expr MakeAdaptiveAvgPool1D(Expr data, Array<IndexExpr> output_size, String layout,
                           String out_layout) {
  auto attrs = make_object<AdaptivePool1DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.adaptive_avg_pool1d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.adaptive_avg_pool1d").set_body_typed(MakeAdaptiveAvgPool1D);

RELAY_REGISTER_OP("nn.adaptive_avg_pool1d")
    .describe(R"code(Adaptive average pooling operation for 1D data.

- **data**: This depends on the `layout` parameter. Input is 3D array of shape
            (batch_size, channels, width) if `layout` is `NCW`.
- **output_size**: If this argument is not provided, input width will be used
                   as output width.
                   If an integer is provided for output_size, the output size is
                   (N x C x output_size) for any input (NCW).
- **out**: This depends on the `layout` parameter. Output is 3D array of shape
           (batch_size, channels, output_width)  if `layout` is `NCW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AdaptivePool1DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("AdaptiveAvgPool1D", AdaptivePool1DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   PoolInferCorrectLayout<AdaptivePool1DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", AdaptivePool1DCompute<topi::nn::kAvgPool>);

// relay.nn.adaptive_max_pool1d
Expr MakeAdaptiveMaxPool1D(Expr data, Array<IndexExpr> output_size, String layout,
                           String out_layout) {
  auto attrs = make_object<AdaptivePool1DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.adaptive_max_pool1d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.adaptive_max_pool1d").set_body_typed(MakeAdaptiveMaxPool1D);

RELAY_REGISTER_OP("nn.adaptive_max_pool1d")
    .describe(R"code(Adaptive max pooling operation for 1D data.

- **data**: This depends on the `layout` parameter. Input is 3D array of shape
            (batch_size, channels, width) if `layout` is `NCW`.
- **output_size**: If this argument is not provided, input width will be used
                   as output width.
                   If an integer is provided for output_size, the output size is
                   (N x C x output_size) for any input (NCW).
- **out**: This depends on the `layout` parameter. Output is 3D array of shape
           (batch_size, channels, output_width)  if `layout` is `NCW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AdaptivePool1DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("AdaptiveMaxPool1D", AdaptivePool1DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   PoolInferCorrectLayout<AdaptivePool1DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", AdaptivePool1DCompute<topi::nn::kMaxPool>);

// relay.nn.adaptive_pool_2d
TVM_REGISTER_NODE_TYPE(AdaptivePool2DAttrs);

bool AdaptivePool2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto* param = attrs.as<AdaptivePool2DAttrs>();
  ICHECK(param != nullptr);

  Layout layout(param->layout);
  ICHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
         !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  Array<IndexExpr> oshape(dshape);
  auto output_size = param->output_size;
  ICHECK_LE(output_size.size(), 2U) << "output_size can have up to 2 elements.";
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
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

template <topi::nn::PoolType mode>
Array<te::Tensor> AdaptivePool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                        const Type& out_type) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<AdaptivePool2DAttrs>();
  ICHECK(param != nullptr);
  Layout layout(param->layout);
  ICHECK(tir::BijectiveLayout(layout, kNCHW).defined())
      << "Adaptive pool2d currently only supports layouts that are convertible from NCHW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "Adaptive pool2d does not support input split on height";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "Adaptive pool2d does not support input split on width";

  ICHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
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
  return Array<te::Tensor>{topi::nn::adaptive_pool(
      inputs[0], Array<IndexExpr>{output_height, output_width}, mode, layout.name())};
}

// relay.nn.adaptive_avg_pool2d
Expr MakeAdaptiveAvgPool2D(Expr data, Array<IndexExpr> output_size, String layout,
                           String out_layout) {
  auto attrs = make_object<AdaptivePool2DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.adaptive_avg_pool2d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.adaptive_avg_pool2d").set_body_typed(MakeAdaptiveAvgPool2D);

RELAY_REGISTER_OP("nn.adaptive_avg_pool2d")
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
    .set_attrs_type<AdaptivePool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("AdaptiveAvgPool2D", AdaptivePool2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   PoolInferCorrectLayout<AdaptivePool2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", AdaptivePool2DCompute<topi::nn::kAvgPool>);

// relay.nn.adaptive_max_pool2d
Expr MakeAdaptiveMaxPool2D(Expr data, Array<IndexExpr> output_size, String layout,
                           String out_layout) {
  auto attrs = make_object<AdaptivePool2DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.adaptive_max_pool2d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.adaptive_max_pool2d").set_body_typed(MakeAdaptiveMaxPool2D);

RELAY_REGISTER_OP("nn.adaptive_max_pool2d")
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
    .set_attrs_type<AdaptivePool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("AdaptiveMaxPool2D", AdaptivePool2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   PoolInferCorrectLayout<AdaptivePool2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", AdaptivePool2DCompute<topi::nn::kMaxPool>);

// relay.nn.adaptive_pool3d
TVM_REGISTER_NODE_TYPE(AdaptivePool3DAttrs);

bool AdaptivePool3DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 3U)
      << "Pool3D only support input >= 3-D: input must have depth, height and width";
  const auto* param = attrs.as<AdaptivePool3DAttrs>();
  ICHECK(param != nullptr);

  Layout layout(param->layout);
  ICHECK(layout.Contains(LayoutAxis::Get('D')) && layout.Contains(LayoutAxis::Get('H')) &&
         layout.Contains(LayoutAxis::Get('W')) && !layout.Contains(LayoutAxis::Get('d')) &&
         !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout
      << ". Pool3D layout must have D, H and W, which cannot be split";

  const auto didx = layout.IndexOf(LayoutAxis::Get('D'));
  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  Array<IndexExpr> oshape(dshape);
  auto output_size = param->output_size;
  ICHECK_LE(output_size.size(), 3U) << "output_size can have up to 3 elements.";
  IndexExpr output_depth, output_height, output_width;
  if (output_size.empty()) {
    output_depth = dshape[didx];
    output_height = dshape[hidx];
    output_width = dshape[widx];
  } else if (output_size.size() == 1) {
    output_depth = output_size[0];
    output_height = output_size[0];
    output_width = output_size[0];
  } else {
    output_depth = output_size[0];
    output_height = output_size[1];
    output_width = output_size[2];
  }

  oshape.Set(didx, output_depth);
  oshape.Set(hidx, output_height);
  oshape.Set(widx, output_width);

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

template <topi::nn::PoolType mode>
Array<te::Tensor> AdaptivePool3DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                        const Type& out_type) {
  static const Layout kNCDHW("NCDHW");
  const auto* param = attrs.as<AdaptivePool3DAttrs>();
  ICHECK(param != nullptr);
  Layout layout(param->layout);
  Layout out_layout(param->out_layout);
  ICHECK(tir::BijectiveLayout(layout, kNCDHW).defined())
      << "Adaptive pool3d currently only supports layouts that are convertible from NCDHW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('d')), -1)
      << "Adaptive pool3d does not support input split on depth";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "Adaptive pool3d does not support input split on height";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "Adaptive pool3d does not support input split on width";

  ICHECK(inputs[0].ndim() == 5U || inputs[0].ndim() == 6U)
      << "Pool3D only support 5-D input (e.g., NCDHW)"
      << " or 6-D input (last dimension is a split of channel)";

  auto output_size = param->output_size;
  const auto didx = layout.IndexOf(LayoutAxis::Get('D'));
  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  IndexExpr output_depth, output_height, output_width;
  if (output_size.empty()) {
    output_depth = inputs[0]->shape[didx];
    output_height = inputs[0]->shape[hidx];
    output_width = inputs[0]->shape[widx];
  } else if (output_size.size() == 1) {
    output_depth = output_size[0];
    output_height = output_size[0];
    output_width = output_size[0];
  } else {
    output_depth = output_size[0];
    output_height = output_size[1];
    output_width = output_size[2];
  }

  auto osize = Array<IndexExpr>{output_depth, output_height, output_width};
  return Array<te::Tensor>{topi::nn::adaptive_pool3d(inputs[0], osize, mode, layout.name())};
}

// relay.nn.adaptive_max_pool3d
Expr MakeAdaptiveMaxPool3D(Expr data, Array<IndexExpr> output_size, String layout,
                           String out_layout) {
  auto attrs = make_object<AdaptivePool3DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.adaptive_max_pool3d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.adaptive_max_pool3d").set_body_typed(MakeAdaptiveMaxPool3D);

RELAY_REGISTER_OP("nn.adaptive_max_pool3d")
    .describe(R"code(Adaptive max pooling operation for 3D data.

- **data**: This depends on the `layout` parameter. Input is 5D array of shape
            (batch_size, channels, depth, height, width) if `layout` is `NCDHW`.
- **output_size**: If this argument is not provided, input depth, height and width will be used
                   as output depth, height and width.
                   If a single integer is provided for output_size, the output size is
                   (N x C x output_size x output_size x output_size) for any input (NCDHW).
                   If a tuple of integers (depth, height, width) are provided for output_size,
                   the output size is (N x C x depth x height x width) for any input (NCDHW).
- **out**: This depends on the `layout` parameter. Output is 5D array of shape
           (batch_size, channels, output_depth, output_height, output_width)  if `layout` is `NCDHW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AdaptivePool3DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("AdaptiveMaxPool3D", AdaptivePool3DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   PoolInferCorrectLayout<AdaptivePool3DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", AdaptivePool3DCompute<topi::nn::kMaxPool>);

// relay.nn.adaptive_max_pool3d
Expr MakeAdaptiveAvgPool3D(Expr data, Array<IndexExpr> output_size, String layout,
                           String out_layout) {
  auto attrs = make_object<AdaptivePool3DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  static const Op& op = Op::Get("nn.adaptive_avg_pool3d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.adaptive_avg_pool3d").set_body_typed(MakeAdaptiveAvgPool3D);

RELAY_REGISTER_OP("nn.adaptive_avg_pool3d")
    .describe(R"code(Adaptive avg pooling operation for 3D data.
- **data**: This depends on the `layout` parameter. Input is 5D array of shape
            (batch_size, channels, depth, height, width) if `layout` is `NCDHW`.
- **output_size**: If this argument is not provided, input depth, height and width will be used
                   as output depth, height and width.
                   If a single integer is provided for output_size, the output size is
                   (N x C x output_size x output_size x output_size) for any input (NCDHW).
                   If a tuple of integers (depth, height, width) are provided for output_size,
                   the output size is (N x C x depth x height x width) for any input (NCDHW).
- **out**: This depends on the `layout` parameter. Output is 5D array of shape
           (batch_size, channels, output_depth, output_height, output_width)  if `layout` is `NCDHW`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<AdaptivePool3DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(10)
    .add_type_rel("AdaptiveAvgPool3D", AdaptivePool3DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   PoolInferCorrectLayout<AdaptivePool3DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", AdaptivePool3DCompute<topi::nn::kAvgPool>);

bool Pool2DGradRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[1].as<TensorTypeNode>();

  if (data == nullptr) return false;

  // assign output type
  reporter->Assign(types[2], types[1]);
  return true;
}

template <typename AttrType, topi::nn::PoolType mode>
Array<te::Tensor> Pool2DGradCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                    const Type& out_type) {
  static const Layout kNCHW("NCHW");
  const auto* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  ICHECK_EQ(inputs.size(), 2);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  Layout layout(param->layout);

  ICHECK(tir::BijectiveLayout(layout, kNCHW).defined())
      << "pool2d_grad currently only supports layouts that are convertible from NCHW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "pool2d_grad does not support input split on height";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "pool2d_grad does not support input split on width";

  ICHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
      << "Pool2DGrad only support 4-D output gradient (e.g., NCHW)"
      << " or 5-D output gradient (last dimension is a split of channel)";

  ICHECK(inputs[1].ndim() == 4U || inputs[1].ndim() == 5U)
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
    return Array<te::Tensor>{topi::nn::pool_grad(inputs[0], inputs[1], pool_size, strides, padding,
                                                 mode, ceil_mode, layout.name(),
                                                 count_include_pad)};
  } else {
    return Array<te::Tensor>{topi::nn::pool_grad(inputs[0], inputs[1], pool_size, strides, padding,
                                                 mode, ceil_mode, layout.name())};
  }
}

// MaxPool2DGrad
Expr MakeMaxPool2DGrad(Expr out_grad, Expr data, Array<IndexExpr> pool_size,
                       Array<IndexExpr> strides, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode) {
  auto attrs = make_object<MaxPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  attrs->ceil_mode = ceil_mode;
  static const Op& op = Op::Get("nn.max_pool2d_grad");
  return Call(op, {out_grad, data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.max_pool2d_grad").set_body_typed(MakeMaxPool2DGrad);

RELAY_REGISTER_OP("nn.max_pool2d_grad")
    .describe(R"code(Gradient of max pooling operation for two dimensional data.

- **out_grad**: This depends on the `layout` parameter. Output gradient is 4D array of
                shape (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
                out_height and out_width are the output size of the pooling operation,
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
    .set_attrs_type<MaxPool2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("grad", "Tensor", "The grad tensor.")
    .set_support_level(2)
    .add_type_rel("MaxPool2DGrad", Pool2DGradRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool2DGradCompute<MaxPool2DAttrs, topi::nn::kMaxPool>);

// AvgPool2DGrad
Expr MakeAvgPool2DGrad(Expr out_grad, Expr data, Array<IndexExpr> pool_size,
                       Array<IndexExpr> strides, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode, bool count_include_pad) {
  auto attrs = make_object<AvgPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  static const Op& op = Op::Get("nn.avg_pool2d_grad");
  return Call(op, {out_grad, data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.avg_pool2d_grad").set_body_typed(MakeAvgPool2DGrad);

RELAY_REGISTER_OP("nn.avg_pool2d_grad")
    .describe(R"code(Gradient of average pooling operation for two dimensional data.

- **out_grad**: This depends on the `layout` parameter. Output gradient is 4D array of
                shape (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
                out_height and out_width are the output size of the pooling operation,
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
    .set_attrs_type<MaxPool2DAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("grad", "Tensor", "The grad tensor.")
    .set_support_level(2)
    .add_type_rel("MaxPool2DGrad", Pool2DGradRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool2DGradCompute<AvgPool2DAttrs, topi::nn::kAvgPool>);

// relay.nn.max_pool1d & relay.nn.avg_pool1d
TVM_REGISTER_NODE_TYPE(MaxPool1DAttrs);
TVM_REGISTER_NODE_TYPE(AvgPool1DAttrs);

template <typename AttrType>
bool Pool1DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) return false;

  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 1U) << "Pool1D only support input >= 1-D: input must have width";
  const auto param = attrs.as<AttrType>();
  ICHECK(param != nullptr);

  Layout layout(param->layout);
  Layout out_layout(param->out_layout);
  ICHECK(layout.Contains(LayoutAxis::Get('W')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout << ". Pool1D layout must have W, which cannot be split";

  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));

  IndexExpr pad_w;
  if (param->padding.size() == 1) {
    pad_w = param->padding[0] * 2;
  } else if (param->padding.size() == 2) {
    // (left, right)
    pad_w = param->padding[0] + param->padding[1];
  } else {
    return false;
  }

  std::vector<IndexExpr> oshape(dshape.begin(), dshape.end());

  if (dshape[widx].as<tir::AnyNode>()) {
    oshape[widx] = dshape[widx];
  } else {
    oshape[widx] =
        calculate_pool_dimension(dshape[widx], pad_w, param->pool_size[0], param->dilation[0],
                                 param->strides[0], param->ceil_mode);
  }

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

template <typename AttrType, topi::nn::PoolType mode>
Array<te::Tensor> Pool1DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  static const Layout kNCW("NCW");
  const auto* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto dilation = param->dilation;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  Layout layout(param->layout);
  Layout out_layout(param->out_layout);

  ICHECK(tir::BijectiveLayout(layout, kNCW).defined())
      << "max_pool1d currently only supports layouts that are convertible from NCW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "max_pool1d does not support input split on width";

  ICHECK(inputs[0].ndim() == 3U || inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
      << "Pool1D only support 3-D input (e.g., NCW)"
      << " or 4-D input (e.g. NCWc on for vector instructions)"
      << " or 5-D input (e.g. NCWnc for tensor accelerators)";

  if (param->padding.size() == 1) {
    padding.push_back(padding[0]);
  }

  if (mode == topi::nn::kAvgPool) {
    bool count_include_pad = reinterpret_cast<const AvgPool1DAttrs*>(param)->count_include_pad;
    return Array<te::Tensor>{topi::nn::pool1d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name(), count_include_pad)};
  } else {
    return Array<te::Tensor>{topi::nn::pool1d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name())};
  }
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.max_pool1d")
    .set_body_typed([](Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                       Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode) {
      return MakeMaxPool<MaxPool1DAttrs>(data, pool_size, strides, dilation, padding, layout,
                                         out_layout, ceil_mode, "nn.max_pool1d");
    });

RELAY_REGISTER_OP("nn.max_pool1d")
    .describe(R"code(Max pooling operation for one dimensional data.

- **data**: This depends on the `layout` parameter. Input is 3D array of shape
            (batch_size, channels, width) if `layout` is `NCW`.
- **out**: This depends on the `layout` parameter. Output is 3D array of shape
           (batch_size, channels, , out_width)  if `layout` is `NCW`.
           out_width is calculated as::

               out_width = floor((width+padding[0]+padding[1]-pool_size[0])/strides[0])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               two int: padding width in the order of (left, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<MaxPool1DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("MaxPool1D", Pool1DRel<MaxPool1DAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<MaxPool1DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool1DCompute<MaxPool1DAttrs, topi::nn::kMaxPool>);

// AvgPool1D
TVM_REGISTER_GLOBAL("relay.op.nn._make.avg_pool1d")
    .set_body_typed([](Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                       Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode, bool count_include_pad) {
      return MakeAvgPool<AvgPool1DAttrs>(data, pool_size, strides, dilation, padding, layout,
                                         out_layout, ceil_mode, count_include_pad, "nn.avg_pool1d");
    });

RELAY_REGISTER_OP("nn.avg_pool1d")
    .describe(R"code(
Average pooling operation for one dimensional data.

- **data**: This depends on the `layout` parameter. Input is 3D array of shape
            (batch_size, channels, width) if `layout` is `NCW`.
- **out**: This depends on the `layout` parameter. Output is 3D array of shape
           (batch_size, channels, out_width)  if `layout` is `NCW`.
           out_width is calculated as::

               out_width = floor((width+padding[0]+padding[1]-pool_size[0])/strides[0])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               two int: padding width in the order of (left, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AvgPool1DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("AvgPool1D", Pool1DRel<AvgPool1DAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<AvgPool1DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool1DCompute<AvgPool1DAttrs, topi::nn::kAvgPool>);

// relay.nn.max_pool3d & relay.nn.avg_pool3d
TVM_REGISTER_NODE_TYPE(MaxPool3DAttrs);
TVM_REGISTER_NODE_TYPE(AvgPool3DAttrs);

template <typename AttrType>
bool Pool3DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) return false;

  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 3U)
      << "Pool3D only support input >= 3-D: input must have depth, height and width";
  const auto param = attrs.as<AttrType>();
  ICHECK(param != nullptr);

  Layout layout(param->layout);
  Layout out_layout(param->out_layout);
  ICHECK(layout.Contains(LayoutAxis::Get('D')) && layout.Contains(LayoutAxis::Get('H')) &&
         layout.Contains(LayoutAxis::Get('W')) && !layout.Contains(LayoutAxis::Get('d')) &&
         !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout
      << ". Pool3D layout must have D, H and W, which cannot be split";

  const auto didx = layout.IndexOf(LayoutAxis::Get('D'));
  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));

  IndexExpr pad[3];
  if (param->padding.size() == 1) {
    pad[0] = param->padding[0] * 2;
    pad[1] = param->padding[0] * 2;
    pad[2] = param->padding[0] * 2;
  } else if (param->padding.size() == 3) {
    // (front, top, left)
    pad[0] = param->padding[0] * 2;
    pad[1] = param->padding[1] * 2;
    pad[2] = param->padding[2] * 2;
  } else if (param->padding.size() == 6) {
    // (front, top, left, back, bottom, right)
    pad[0] = param->padding[0] + param->padding[3];
    pad[1] = param->padding[1] + param->padding[4];
    pad[2] = param->padding[2] + param->padding[5];
  } else {
    return false;
  }

  std::vector<IndexExpr> oshape(dshape.begin(), dshape.end());

  int idxes[3] = {didx, hidx, widx};
  for (int i = 0; i < 3; i++) {
    int ii = idxes[i];
    if (dshape[ii].as<tir::AnyNode>()) {
      oshape[ii] = dshape[ii];
    } else {
      oshape[ii] =
          calculate_pool_dimension(dshape[ii], pad[i], param->pool_size[i], param->dilation[i],
                                   param->strides[i], param->ceil_mode);
    }
  }

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

template <typename AttrType, topi::nn::PoolType mode>
Array<te::Tensor> Pool3DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  static const Layout kNCDHW("NCDHW");
  const auto* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto dilation = param->dilation;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  Layout layout(param->layout);
  Layout out_layout(param->out_layout);

  ICHECK(tir::BijectiveLayout(layout, kNCDHW).defined())
      << "max_pool3d currently only supports layouts that are convertible from NCDHW";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('d')), -1)
      << "max_pool3d does not support input split on depth";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
      << "max_pool3d does not support input split on height";
  ICHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
      << "max_pool3d does not support input split on width";

  ICHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U || inputs[0].ndim() == 6U)
      << "Pool3D only support 5-D input (e.g., NCDHW)"
      << " or 6-D input (e.g. NCDHWc on for vector instructions)"
      << " or 7-D input (e.g. NCDHWnc for tensor accelerators)";

  if (param->padding.size() == 1) {
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
  } else if (param->padding.size() == 3) {
    padding.push_back(padding[0]);
    padding.push_back(padding[1]);
    padding.push_back(padding[2]);
  }
  if (mode == topi::nn::kAvgPool) {
    bool count_include_pad = reinterpret_cast<const AvgPool3DAttrs*>(param)->count_include_pad;
    return Array<te::Tensor>{topi::nn::pool3d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name(), count_include_pad)};
  } else {
    return Array<te::Tensor>{topi::nn::pool3d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name())};
  }
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.max_pool3d")
    .set_body_typed([](Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                       Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode) {
      return MakeMaxPool<MaxPool3DAttrs>(data, pool_size, strides, dilation, padding, layout,
                                         out_layout, ceil_mode, "nn.max_pool3d");
    });

RELAY_REGISTER_OP("nn.max_pool3d")
    .describe(R"code(Max pooling operation for three dimensional data.

- **data**: This depends on the `layout` parameter. Input is 5D array of shape
            (batch_size, channels, depth, height, width) if `layout` is `NCDHW`.
- **out**: This depends on the `layout` parameter. Output is 5D array of shape
           (batch_size, channels, out_depth, out_height, out_width)  if `layout` is `NCDHW`.
           out_depth, out_height and out_width are calculated as::

               out_depth = floor((depth+padding[0]+padding[3]-pool_size[0])/strides[0])+1
               out_height = floor((height+padding[1]+padding[4]-pool_size[1])/strides[1])+1
               out_width = floor((width+padding[2]+padding[5]-pool_size[2])/strides[2])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               three int : front, bottom, right use same as back, top and left.
               six int: padding width in the order of (front, top, left, back, bottom, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<MaxPool3DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("MaxPool3D", Pool3DRel<MaxPool3DAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<MaxPool3DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool3DCompute<MaxPool3DAttrs, topi::nn::kMaxPool>);

// AvgPool3D
TVM_REGISTER_GLOBAL("relay.op.nn._make.avg_pool3d")
    .set_body_typed([](Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                       Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                       String out_layout, bool ceil_mode, bool count_include_pad) {
      return MakeAvgPool<AvgPool3DAttrs>(data, pool_size, strides, dilation, padding, layout,
                                         out_layout, ceil_mode, count_include_pad, "nn.avg_pool3d");
    });

RELAY_REGISTER_OP("nn.avg_pool3d")
    .describe(R"code(
Average pooling operation for three dimensional data.

- **data**: This depends on the `layout` parameter. Input is 5D array of shape
            (batch_size, channels, depth, height, width) if `layout` is `NCDHW`.
- **out**: This depends on the `layout` parameter. Output is 5D array of shape
           (batch_size, channels, out_depth, out_height, out_width)  if `layout` is `NCDHW`.
           out_depth, out_height and out_width are calculated as::

               out_depth = floor((depth+padding[0]+padding[3]-pool_size[0])/strides[0])+1
               out_height = floor((height+padding[1]+padding[4]-pool_size[1])/strides[1])+1
               out_width = floor((width+padding[2]+padding[5]-pool_size[2])/strides[2])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               three int : front, bottom, right use same as back, top and left.
               six int: padding width in the order of (front, top, left, back, bottom, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AvgPool3DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(2)
    .add_type_rel("AvgPool3D", Pool3DRel<AvgPool3DAttrs>)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", PoolInferCorrectLayout<AvgPool3DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FTVMCompute>("FTVMCompute", Pool3DCompute<AvgPool3DAttrs, topi::nn::kAvgPool>);

}  // namespace relay
}  // namespace tvm
