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
#include <tvm/tir/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op_attr_types.h>
#include <vector>
#include "../op_common.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(UpSamplingAttrs);
TVM_REGISTER_NODE_TYPE(UpSampling3DAttrs);

template <typename T>
Array<Array<Layout> > UpsamplingInferCorrectLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr>> &old_in_shapes) {
  // NOTE: Discard "const" qualifier here.
  T *params = const_cast<T*>(attrs.as<T>());

  if (new_in_layouts.defined()) {
    CHECK_EQ(new_in_layouts.size(), 1);

    Layout raw_layout(params->layout);
    Layout input = new_in_layouts[0];
    if (input.IndexOf(LayoutAxis::Get('W')) == raw_layout.IndexOf(LayoutAxis::Get('W')) &&
      input.IndexOf(LayoutAxis::Get('H')) == raw_layout.IndexOf(LayoutAxis::Get('H')) &&
        !input.Contains(LayoutAxis::Get('w')) && !input.Contains(LayoutAxis::Get('h'))&&
        (input.IndexOf(LayoutAxis::Get('D')) == -1 ||
        (input.IndexOf(LayoutAxis::Get('D')) == raw_layout.IndexOf(LayoutAxis::Get('D')) &&
        !input.Contains(LayoutAxis::Get('d'))))) {
        params->layout = input.name();  // modify self to follow the input layout
    }
  }

  Layout inferred_layout(params->layout);
  return Array<Array<Layout> >{{inferred_layout}, {inferred_layout}};
}

bool UpSamplingRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const UpSamplingAttrs* param = attrs.as<UpSamplingAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->layout);

  auto layout_converter = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(layout_converter.defined())
    << "UpSampling only support input layouts that are convertible from NCHW."
    << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, tir::CastNode::make(oshape[2].dtype(), tvm::round(oshape[2] * param->scale_h)));
  oshape.Set(3, tir::CastNode::make(oshape[3].dtype(), tvm::round(oshape[3] * param->scale_w)));

  // assign output type
  reporter->Assign(types[1],
                   TensorType(layout_converter.BackwardShape(oshape),
                                        data->dtype));
  return true;
}


// Positional relay function to create upsampling operator
// used by frontend FFI.
Expr MakeUpSampling(Expr data,
                    double scale_h,
                    double scale_w,
                    std::string layout,
                    std::string method,
                    bool align_corners) {
  auto attrs = make_object<UpSamplingAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->scale_h = scale_h;
  attrs->scale_w = scale_w;
  attrs->align_corners = align_corners;
  static const Op& op = Op::Get("nn.upsampling");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.upsampling")
.set_body_typed(MakeUpSampling);


RELAY_REGISTER_OP("nn.upsampling")
.describe(R"code(Perform upsampling on input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, in_height*scale, in_width*scale)

           for layout NHWC
           (batch_size, in_height*scale, in_width*scale, channels)

)code" TVM_ADD_FILELINE)
.set_attrs_type<UpSamplingAttrs>()
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("UpSampling", UpSamplingRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
  UpsamplingInferCorrectLayout<UpSamplingAttrs>)
.set_attr<TOpPattern>("TOpPattern", kInjective);


// UpSampling3D
bool UpSampling3DRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCDHW("NCDHW");

  const UpSampling3DAttrs* param = attrs.as<UpSampling3DAttrs>();
  CHECK(param != nullptr);
  const Layout in_layout(param->layout);

  auto layout_converter = BijectiveLayoutNode::make(in_layout, kNCDHW);
  CHECK(layout_converter.defined())
    << "UpSampling3D only support input layouts that are convertible from NCDHW."
    << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, tir::CastNode::make(oshape[2].dtype(), tvm::round(oshape[2] * param->scale_d)));
  oshape.Set(3, tir::CastNode::make(oshape[3].dtype(), tvm::round(oshape[3] * param->scale_h)));
  oshape.Set(4, tir::CastNode::make(oshape[4].dtype(), tvm::round(oshape[4] * param->scale_w)));

  // assign output type
  reporter->Assign(types[1],
                   TensorType(layout_converter.BackwardShape(oshape),
                                        data->dtype));
  return true;
}

// Positional relay function to create upsampling3d operator
// used by frontend FFI.
Expr MakeUpSampling3D(Expr data,
                      double scale_d,
                      double scale_h,
                      double scale_w,
                      std::string layout,
                      std::string method,
                      std::string coordinate_transformation_mode) {
  auto attrs = make_object<UpSampling3DAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->scale_d = scale_d;
  attrs->scale_h = scale_h;
  attrs->scale_w = scale_w;
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;
  static const Op& op = Op::Get("nn.upsampling3d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.nn._make.upsampling3d")
.set_body_typed(MakeUpSampling3D);


RELAY_REGISTER_OP("nn.upsampling3d")
.describe(R"code(Perform upsampling on input array with nearest neighbour or
bilinear interpolation.

- **data**: data is 5D array of shape
            (batch_size, channels, in_depth, in_height, in_width) for NCDHW
            (batch_size, in_depth, in_height, in_width, channels) for NDHWC

- **out**: Output is 5D array of shape
           for layout NCDHW
           (batch_size, channels, in_depth*scale, in_height*scale, in_width*scale)

           for layout NDHWC
           (batch_size, in_depth*scale, in_height*scale, in_width*scale, channels)

)code" TVM_ADD_FILELINE)
.set_attrs_type<UpSampling3DAttrs>()
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("UpSampling3D", UpSampling3DRel)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
  UpsamplingInferCorrectLayout<UpSampling3DAttrs>)
.set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
