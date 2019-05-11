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
 *  Copyright (c) 2017 by Contributors
 * \file pooling.cc
 * \brief Property def of pooling operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/nn.h>
#include "nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/nn/pooling.h"

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

DMLC_REGISTER_PARAMETER(MaxPool2DParam);

template <typename T>
inline bool Pool2DInferShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  const T& param = nnvm::get<T>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;

  CHECK_GE(dshape.ndim(), 2U)
    << "Pool2D only support input >= 2-D: input must have height and width";

  Layout layout(param.layout);
  CHECK(layout.contains('H') && layout.contains('W') &&
        !layout.contains('h') && !layout.contains('w'))
    << "Invalid layout " << layout
    << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.indexof('H');
  const auto widx = layout.indexof('W');

  dim_t pad_h, pad_w;
  if (param.padding.ndim() == 1) {
    pad_h = param.padding[0] * 2;
    pad_w = param.padding[0] * 2;
  } else if (param.padding.ndim() == 2) {
    // (top, left)
    pad_h = param.padding[0] * 2;
    pad_w = param.padding[1] * 2;
  } else if (param.padding.ndim() == 4) {
    // (top, left, bottom, right)
    pad_h = param.padding[0] + param.padding[2];
    pad_w = param.padding[1] + param.padding[3];
  } else {
    return false;
  }

  TShape oshape = dshape;
  CHECK(param.pool_size[0] <= dshape[hidx] + pad_h)
      << "pool size (" << param.pool_size[0] << ") exceeds input (" << dshape[hidx]
      << " padded to " << (dshape[hidx] + pad_h) << ")";
  CHECK(param.pool_size[1] <= dshape[widx] + pad_w)
      << "pool size (" << param.pool_size[1] << ") exceeds input (" << dshape[widx]
      << " padded to " << (dshape[widx] + pad_w) << ")";

  if (!param.ceil_mode) {
    oshape[hidx] = ((dshape[hidx] + pad_h - param.pool_size[0]) /
                    param.strides[0]) + 1;
    oshape[widx] = ((dshape[widx] + pad_w - param.pool_size[1]) /
                    param.strides[1]) + 1;
  } else {
    oshape[hidx] = ((dshape[hidx] + pad_h - param.pool_size[0] +
                    param.strides[0] - 1) / param.strides[0]) + 1;
    oshape[widx] = ((dshape[widx] + pad_w - param.pool_size[1] +
                    param.strides[1] - 1) / param.strides[1]) + 1;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

template <typename T>
inline bool Pool2DCorrectLayout(const NodeAttrs& attrs,
                                std::vector<Layout> *ilayouts,
                                const std::vector<Layout> *last_ilayouts,
                                std::vector<Layout> *olayouts) {
  const T &param = nnvm::get<T>(attrs.parsed);
  CHECK_EQ(ilayouts->size(), 1);
  CHECK_EQ(last_ilayouts->size(), 1);
  CHECK_EQ(olayouts->size(), 1);

  Layout input = (*ilayouts)[0];
  const Layout layout(param.layout);

  if (input.defined()) {
    CHECK(input.convertible(layout)) << "Invalid input layout " << input;
    if (input.indexof('W') != layout.indexof('W') ||
        input.indexof('H') != layout.indexof('H') ||
        input.contains('w') || input.contains('h')) {
      // as long as the index doesn't change for width and height
      // pool2d can keep the input layout.
      input = layout;
    }
  } else {
    input = layout;
  }

  NNVM_ASSIGN_LAYOUT(*ilayouts, 0, input);
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, input);

  return true;
}

NNVM_REGISTER_OP(max_pool2d)
.describe(R"code(Max pooling operation for one dimensional data.

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

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(MaxPool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<MaxPool2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<MaxPool2DParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", Pool2DInferShape<MaxPool2DParam>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", Pool2DCorrectLayout<MaxPool2DParam>)
.set_attr<FTVMCompute>("FTVMCompute", [](const NodeAttrs& attrs,
                                         const Array<Tensor>& inputs,
                                         const Array<Tensor>& out_info) {
  const MaxPool2DParam& param = nnvm::get<MaxPool2DParam>(attrs.parsed);
  auto pool_size = ShapeToArray(param.pool_size);
  auto strides = ShapeToArray(param.strides);
  auto padding = ShapeToArray(param.padding);
  auto ceil_mode = param.ceil_mode;

  Layout layout(param.layout);
  CHECK(layout.convertible(Layout("NCHW")))
    << "max_pool2d currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.indexof('h'), -1) << "max_pool2d does not support input split on height";
  CHECK_EQ(layout.indexof('w'), -1) << "max_pool2d does not support input split on width";

  CHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
    << "Pool2D only support 4-D input (e.g., NCHW)"
    << " or 5-D input (last dimension is a split of channel)";

  if (param.padding.ndim() == 1) {
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
  } else if (param.padding.ndim() == 2) {
    padding.push_back(padding[0]);
    padding.push_back(padding[1]);
  }

  return Array<Tensor>{
    topi::nn::pool(inputs[0], pool_size, strides, padding,
                   topi::nn::kMaxPool, ceil_mode, layout.name())};
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return MakeGradNode("_max_pool2d_grad", n,
                        {ograds[0], n->inputs[0], NodeEntry{n, 0, 0}},
                        n->attrs.dict);
})
.set_support_level(2);

NNVM_REGISTER_OP(_max_pool2d_grad)
  .describe(R"code(Max pooling 2D grad.

)code" NNVM_ADD_FILELINE)
.add_argument("ograd", "4D Tensor", "Output grad.")
.add_argument("input", "4D Tensor", "Input data of max_pool2d grad.")
.add_argument("output", "4D Tensor", "Output data of max_pool2d grad.")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<MaxPool2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<MaxPool2DParam>)
.set_attr<FInferShape>("FInferShape", AssignOutputAttr<TShape, 1, 0>)
.set_attr<FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<TIsBackward>("TIsBackward", true);

DMLC_REGISTER_PARAMETER(AvgPool2DParam);

NNVM_REGISTER_OP(avg_pool2d)
.describe(R"code(Average pooling operation for one dimensional data.

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

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(AvgPool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<AvgPool2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<AvgPool2DParam>)
.set_attr<FInferShape>("FInferShape", Pool2DInferShape<AvgPool2DParam>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", Pool2DCorrectLayout<AvgPool2DParam>)
.set_attr<FTVMCompute>("FTVMCompute", [](const NodeAttrs& attrs,
                                         const Array<Tensor>& inputs,
                                         const Array<Tensor>& out_info) {
  const AvgPool2DParam& param = nnvm::get<AvgPool2DParam>(attrs.parsed);
  auto pool_size = ShapeToArray(param.pool_size);
  auto strides = ShapeToArray(param.strides);
  auto padding = ShapeToArray(param.padding);
  auto ceil_mode = param.ceil_mode;
  auto count_include_pad = param.count_include_pad;

  Layout layout(param.layout);
  CHECK(layout.convertible(Layout("NCHW")))
    << "avg_pool2d currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.indexof('h'), -1) << "avg_pool2d does not support input split on height";
  CHECK_EQ(layout.indexof('w'), -1) << "avg_pool2d does not support input split on width";

  CHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
    << "Pool2D only support 4-D input (e.g., NCHW)"
    << " or 5-D input (last dimension is a split of channel)";

  if (param.padding.ndim() == 1) {
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
  } else if (param.padding.ndim() == 2) {
    padding.push_back(padding[0]);
    padding.push_back(padding[1]);
  }

  return Array<Tensor>{
    topi::nn::pool(inputs[0], pool_size, strides, padding,
                   topi::nn::kAvgPool, ceil_mode, layout.name(), count_include_pad)};
})
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(2);


DMLC_REGISTER_PARAMETER(GlobalPool2DParam);

inline bool GlobalPool2DInferShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_shape,
                                   std::vector<TShape>* out_shape) {
  static const Layout kNCHW("NCHW");
  const GlobalPool2DParam& param = nnvm::get<GlobalPool2DParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 1U);
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;

  CHECK_GE(dshape.ndim(), 2U)
    << "Pool2D only support input >= 2-D: input must have height and width";

  Layout layout(param.layout);
  CHECK(layout.contains('H') && layout.contains('W') &&
        !layout.contains('h') && !layout.contains('w'))
    << "Invalid layout " << layout
    << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.indexof('H');
  const auto widx = layout.indexof('W');

  TShape oshape = dshape;
  oshape[hidx] = oshape[widx] = 1;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

inline bool GlobalPool2DCorrectLayout(const NodeAttrs& attrs,
                                      std::vector<Layout> *ilayouts,
                                      const std::vector<Layout> *last_ilayouts,
                                      std::vector<Layout> *olayouts) {
  const GlobalPool2DParam &param = nnvm::get<GlobalPool2DParam>(attrs.parsed);
  CHECK_EQ(ilayouts->size(), 1);
  CHECK_EQ(last_ilayouts->size(), 1);
  CHECK_EQ(olayouts->size(), 1);

  Layout input = (*ilayouts)[0];
  const Layout layout(param.layout);

  if (input.defined()) {
    CHECK(input.convertible(layout)) << "Invalid input layout " << input;
    if (input.indexof('W') != layout.indexof('W') ||
        input.indexof('H') != layout.indexof('H') ||
        input.contains('w') || input.contains('h')) {
      // as long as the index doesn't change for width and height
      // pool2d can keep the input layout.
      input = layout;
    }
  } else {
    input = layout;
  }

  NNVM_ASSIGN_LAYOUT(*ilayouts, 0, input);
  NNVM_ASSIGN_LAYOUT(*olayouts, 0, input);

  return true;
}

NNVM_REGISTER_OP(global_max_pool2d)
.describe(R"code(Global max pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(GlobalPool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<GlobalPool2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<GlobalPool2DParam>)
.set_attr<FInferShape>("FInferShape", GlobalPool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", GlobalPool2DCorrectLayout)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
  const GlobalPool2DParam& param = nnvm::get<GlobalPool2DParam>(attrs.parsed);
  Layout layout(param.layout);
  CHECK(layout.convertible(Layout("NCHW")))
    << "global_max_pool2d currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.indexof('h'), -1)
    << "global_max_pool2d does not support input split on height";
  CHECK_EQ(layout.indexof('w'), -1)
    << "global_max_pool2d does not support input split on width";

  CHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
    << "Pool2D only support 4-D input (e.g., NCHW)"
    << " or 5-D input (last dimension is a split of channel)";

  return Array<Tensor>{
    topi::nn::global_pool(inputs[0], topi::nn::kMaxPool, layout.name()) };
})
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(2);


NNVM_REGISTER_OP(global_avg_pool2d)
.describe(R"code(Global average pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(GlobalPool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<GlobalPool2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<GlobalPool2DParam>)
.set_attr<FInferShape>("FInferShape", GlobalPool2DInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", GlobalPool2DCorrectLayout)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
  const GlobalPool2DParam& param = nnvm::get<GlobalPool2DParam>(attrs.parsed);
  Layout layout(param.layout);
  CHECK(layout.convertible(Layout("NCHW")))
    << "global_avg_pool2d currently only supports layouts that are convertible from NCHW";
  CHECK_EQ(layout.indexof('h'), -1)
    << "global_avg_pool2d does not support input split on height";
  CHECK_EQ(layout.indexof('w'), -1)
    << "global_avg_pool2d does not support input split on width";

  CHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U)
    << "Pool2D only support 4-D input (e.g., NCHW)"
    << " or 5-D input (last dimension is a split of channel)";

  return Array<Tensor>{
    topi::nn::global_pool(inputs[0], topi::nn::kAvgPool, layout.name()) };
})
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(2);

}  // namespace top
}  // namespace nnvm
