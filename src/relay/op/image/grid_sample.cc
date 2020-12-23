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
 * \file grid_sample.cc
 * \brief affine_grid and grid_sample operator
 */
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.image.affine_grid
TVM_REGISTER_NODE_TYPE(AffineGridAttrs);

bool AffineGridRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  auto batch_size = data->shape[0];

  const AffineGridAttrs* param = attrs.as<AffineGridAttrs>();
  ICHECK(param != nullptr);

  Array<IndexExpr> oshape;

  ICHECK(data->shape.size() == 3U && reporter->AssertEQ(data->shape[1], 2) &&
         reporter->AssertEQ(data->shape[2], 3))
      << "data should be an"
         "affine matrix with shape [batch_size, 2, 3]";
  ICHECK(param->target_shape.defined() && param->target_shape.size() == 2)
      << "target_shape should be 2D";
  oshape.push_back(batch_size);
  oshape.push_back(2);
  oshape.push_back(param->target_shape[0]);
  oshape.push_back(param->target_shape[1]);

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

// Positional relay function to create affine_grid operator
// used by frontend FFI.
Expr MakeAffineGrid(Expr data, Array<IndexExpr> target_shape) {
  auto attrs = make_object<AffineGridAttrs>();
  attrs->target_shape = std::move(target_shape);
  static const Op& op = Op::Get("image.affine_grid");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.image._make.affine_grid").set_body_typed(MakeAffineGrid);

RELAY_REGISTER_OP("image.affine_grid")
    .describe(R"code(affine_grid operator that generates 2D sampling grid.

This operation is described in https://arxiv.org/pdf/1506.02025.pdf. It generates a uniform
sampling grid within the target shape and normalizes it to [-1, 1]. The provided affine
transformation is then applied on the sampling grid.

- **data**: data is 3D array of shape [batch, 2, 3], which defines an affine transformation.

- **out**: out is 4D array of shape [batch, 2, height, width], where each vector
           :math:`out[b, :, h, w]` represents the coordinate :math:`(x, y)`

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AffineGridAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The affine matrix.")
    .set_support_level(5)
    .add_type_rel("AffineGrid", AffineGridRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// relay.image.grid_sample
TVM_REGISTER_NODE_TYPE(GridSampleAttrs);

bool GridSampleRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* grid = types[1].as<TensorTypeNode>();
  if (!data || !grid) return false;
  const auto* param = attrs.as<GridSampleAttrs>();
  ICHECK(param);
  static const Layout kNCHW("NCHW");
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, grid->shape[2]);
  oshape.Set(3, grid->shape[3]);
  // assign output type
  reporter->Assign(types[2], TensorType(layout_converter.BackwardShape(oshape), data->dtype));
  return true;
}

// Positional relay function to create affine_grid operator
// used by frontend FFI.
Expr MakeGridSample(Expr data, Expr grid, String method, String layout) {
  auto attrs = make_object<GridSampleAttrs>();
  attrs->method = std::move(method);
  attrs->layout = std::move(layout);
  static const Op& op = Op::Get("image.grid_sample");
  return Call(op, {data, grid}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.image._make.grid_sample").set_body_typed(MakeGridSample);

RELAY_REGISTER_OP("image.grid_sample")
    .describe(R"code(Applies grid sampling to input feature map.

Given :math:`data` and :math:`grid`, then the output is computed by

.. math::
  x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
  y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
  output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})

:math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and
:math:`G()` denotes the interpolation function.
The out-boundary points will be padded with zeros. The shape of the output will be
(data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).

The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].

grid_sample often cooperates with affine_grid which generates sampling grids for grid_sample.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **grid**: grid is 4D array of shape [batch, 2, out_height, out_width], where each vector
           :math:`out[b, :, h, w]` represents the coordinate :math:`(x, y)`

- **out**: out is 4D array of shape
           (batch, in_channel, out_height, out_width) for NCHW
           (batch_size, in_height, in_width, channels) for NHWC

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<GridSampleAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("grid", "Tensor", "The grid tensor.")
    .set_support_level(5)
    .add_type_rel("GridSample", GridSampleRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
