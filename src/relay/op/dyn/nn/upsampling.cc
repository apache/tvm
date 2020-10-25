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
 * \file upsampling.cc
 * \brief upsampling operator
 */

#include "upsampling.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <utility>
#include <vector>

#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace dyn {

bool UpSamplingRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  // types = [data_type, scale_h_type, scale_w_type, ret_type]
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* scale_h = types[1].as<TensorTypeNode>();
  const auto* scale_w = types[2].as<TensorTypeNode>();
  if (data == nullptr) return false;
  if (scale_h == nullptr) return false;
  if (scale_w == nullptr) return false;

  ICHECK_EQ(scale_h->shape.size(), 0);
  ICHECK_EQ(scale_w->shape.size(), 0);
  static const Layout kNCHW("NCHW");

  const UpSamplingAttrs* param = attrs.as<UpSamplingAttrs>();
  ICHECK(param);
  const Layout in_layout(param->layout);

  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(layout_converter.defined())
      << "UpSampling only supports input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto nchw_oshape = layout_converter.ForwardShape(data->shape);

  nchw_oshape.Set(2, Any());
  nchw_oshape.Set(3, Any());
  auto oshape = layout_converter.BackwardShape(nchw_oshape);

  reporter->Assign(types[3], TensorType(oshape, data->dtype));
  return true;
}

// Positional relay function to create upsampling operator
// used by frontend FFI.
Expr MakeUpSampling(Expr data, Expr scale_h, Expr scale_w, String layout, String method,
                    bool align_corners) {
  auto attrs = make_object<UpSamplingAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->align_corners = align_corners;

  static const Op& op = Op::Get("dyn.nn.upsampling");
  return Call(op, {data, scale_h, scale_w}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn.nn._make.upsampling").set_body_typed(MakeUpSampling);

RELAY_REGISTER_OP("dyn.nn.upsampling")
    .describe(
        R"code(Perform upsampling on input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **scale_h**: scale_h is a double of the amount to scale height by

- **scale_w**: scale_w is a double of the amount to scale width by

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, in_height*scale_h, in_width*scale_w)

           for layout NHWC
           (batch_size, in_height*scale_h, in_width*scale_w, channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<UpSamplingAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale_h", "double", "The scale for the height.")
    .add_argument("scale_w", "double", "The scale for the width.")
    .set_support_level(2)
    .add_type_rel("DynamicUpSampling", UpSamplingRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   UpsamplingInferCorrectLayout<UpSamplingAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

// UpSampling3D
bool UpSampling3DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  // types = [data_type, scale_d_type, scale_h_type, scale_w_type, ret_type]
  ICHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCDHW("NCDHW");

  const UpSampling3DAttrs* param = attrs.as<UpSampling3DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->layout);

  auto layout_converter = tir::BijectiveLayout(in_layout, kNCDHW);
  ICHECK(layout_converter.defined())
      << "UpSampling3D only support input layouts that are convertible from NCDHW."
      << " But got " << in_layout;

  auto ncdhw_oshape = layout_converter.ForwardShape(data->shape);

  ncdhw_oshape.Set(2, Any());
  ncdhw_oshape.Set(3, Any());
  ncdhw_oshape.Set(4, Any());

  auto oshape = layout_converter.BackwardShape(ncdhw_oshape);

  reporter->Assign(types[4], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeUpSampling3D(Expr data, Expr scale_d, Expr scale_h, Expr scale_w, String layout,
                      String method, String coordinate_transformation_mode) {
  auto attrs = make_object<UpSampling3DAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;

  static const Op& op = Op::Get("dyn.nn.upsampling3d");
  return Call(op, {data, scale_d, scale_h, scale_w}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn.nn._make.upsampling3d").set_body_typed(MakeUpSampling3D);

RELAY_REGISTER_OP("dyn.nn.upsampling3d")
    .describe(R"code(Perform upsampling on input array with nearest neighbour or
bilinear interpolation.

- **data**: data is 5D array of shape
            (batch_size, channels, in_depth, in_height, in_width) for NCDHW
            (batch_size, in_depth, in_height, in_width, channels) for NDHWC

- **scale_d**: scale_d is a double of the amount to scale depth by

- **scale_h**: scale_h is a double of the amount to scale height by

- **scale_w**: scale_w is a double of the amount to scale width by

- **out**: Output is 5D array of shape
           for layout NCDHW
           (batch_size, channels, in_depth*scale_d, in_height*scale_h, in_width*scale_w)

           for layout NDHWC
           (batch_size, in_depth*scale_d, in_height*scale_h, in_width*scale_w, channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<UpSampling3DAttrs>()
    .set_num_inputs(4)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("scale_d", "double", "The scale for the depth.")
    .add_argument("scale_h", "double", "The scale for the height.")
    .add_argument("scale_w", "double", "The scale for the width.")
    .set_support_level(2)
    .add_type_rel("DynamicUpSampling3D", UpSampling3DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   UpsamplingInferCorrectLayout<UpSampling3DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm
