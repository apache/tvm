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
 * \file resize.cc
 * \brief Image resize operators
 */
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include "../make_op.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

template <typename T>
InferCorrectLayoutOutput ResizeInferCorrectLayout(const Attrs& attrs,
                                                  const Array<Layout>& new_in_layouts,
                                                  const Array<Layout>& old_in_layouts,
                                                  const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<T>();
  CHECK(attrs_ptr);
  ObjectPtr<T> params = make_object<T>(*attrs_ptr);

  if (new_in_layouts.defined()) {
    ICHECK_EQ(new_in_layouts.size(), 1);

    Layout raw_layout(params->layout);
    Layout new_layout = new_in_layouts[0];
    Layout old_layout = old_in_layouts[0];
    if (!new_layout.Equals(old_layout) && raw_layout.Equals(old_layout) &&
        new_layout->axes.size() == old_layout->axes.size()) {
      // Follow input layout
      params->layout = new_layout.name();
    }
  }

  return InferCorrectLayoutOutput({params->layout}, {params->layout}, Attrs(params));
}

TVM_REGISTER_NODE_TYPE(Resize1DAttrs);

bool Resize1DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCW("NCW");

  const Resize1DAttrs* param = attrs.as<Resize1DAttrs>();
  ICHECK(param != nullptr);
  ICHECK(param->size.size() == 1);
  ICHECK(param->roi.size() == 2);
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCW);
  ICHECK(layout_converter.defined())
      << "Resize only support input layouts that are convertible from NCW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, param->size[0]);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // assign output type
  reporter->Assign(types[1], TensorType(layout_converter.BackwardShape(oshape), out_dtype));
  return true;
}

// Positional relay function to create image operator
// used by frontend FFI.
Expr MakeResize1D(Expr data, Array<IndexExpr> size, Array<FloatImm> roi, String layout,
                  String method, String coordinate_transformation_mode, String rounding_method,
                  double cubic_alpha, int cubic_exclude, double extrapolation_value,
                  DataType out_dtype) {
  auto attrs = make_object<Resize1DAttrs>();
  attrs->size = std::move(size);
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;
  attrs->rounding_method = rounding_method;
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("image.resize1d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.image._make.resize1d").set_body_typed(MakeResize1D);

RELAY_REGISTER_OP("image.resize1d")
    .describe(R"code(Perform resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 3D array of shape
            (batch_size, channels, in_width) for NCW
            (batch_size, in_width, channels) for NWC

- **out**: Output is 3D array of shape
           for layout NCW
           (batch_size, channels, size[0])

           for layout NWC
           (batch_size, size[0], channels)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Resize1DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(5)
    .add_type_rel("Resize1D", Resize1DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ResizeInferCorrectLayout<Resize1DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

TVM_REGISTER_NODE_TYPE(Resize2DAttrs);

bool Resize2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const Resize2DAttrs* param = attrs.as<Resize2DAttrs>();
  ICHECK(param != nullptr);
  ICHECK(param->size.size() == 2);
  ICHECK(param->roi.size() == 4);
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(layout_converter.defined())
      << "Resize only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, param->size[0]);
  oshape.Set(3, param->size[1]);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // assign output type
  reporter->Assign(types[1], TensorType(layout_converter.BackwardShape(oshape), out_dtype));
  return true;
}

// Positional relay function to create image operator
// used by frontend FFI.
Expr MakeResize2D(Expr data, Array<IndexExpr> size, Array<FloatImm> roi, String layout,
                  String method, String coordinate_transformation_mode, String rounding_method,
                  double cubic_alpha, int cubic_exclude, double extrapolation_value,
                  DataType out_dtype) {
  auto attrs = make_object<Resize2DAttrs>();
  attrs->size = std::move(size);
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;
  attrs->rounding_method = rounding_method;
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("image.resize2d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.image._make.resize2d").set_body_typed(MakeResize2D);

RELAY_REGISTER_OP("image.resize2d")
    .describe(R"code(Perform resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, size[0], size[1])

           for layout NHWC
           (batch_size, size[0], size[1], channels)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Resize2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(5)
    .add_type_rel("Resize2D", Resize2DRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ResizeInferCorrectLayout<Resize2DAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

TVM_REGISTER_NODE_TYPE(Resize3DAttrs);

bool Resize3DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCDHW("NCDHW");

  const Resize3DAttrs* param = attrs.as<Resize3DAttrs>();
  ICHECK(param != nullptr);
  ICHECK(param->size.size() == 3);
  ICHECK(param->roi.size() == 6);
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCDHW);
  ICHECK(layout_converter.defined())
      << "Resize3d only support input layouts that are convertible from NCDHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, param->size[0]);
  oshape.Set(3, param->size[1]);
  oshape.Set(4, param->size[2]);

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // assign output type
  reporter->Assign(types[1], TensorType(layout_converter.BackwardShape(oshape), out_dtype));
  return true;
}

// Positional relay function to create image operator
// used by frontend FFI.
Expr MakeResize3D(Expr data, Array<IndexExpr> size, Array<FloatImm> roi, String layout,
                  String method, String coordinate_transformation_mode, String rounding_method,
                  double cubic_alpha, int cubic_exclude, double extrapolation_value,
                  DataType out_dtype) {
  auto attrs = make_object<Resize3DAttrs>();
  attrs->size = std::move(size);
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;
  attrs->rounding_method = rounding_method;
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("image.resize3d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.image._make.resize3d").set_body_typed(MakeResize3D);

RELAY_REGISTER_OP("image.resize3d")
    .describe(R"code(
Perform resize3d to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 5D array of shape
            (batch_size, channels, in_depth, in_height, in_width) for NCDHW
            (batch_size, in_depth, in_height, in_width, channels) for NDHWC

- **out**: Output is 5D array of shape
           for layout NCDHW
           (batch_size, channels, size[0], size[1], size[2])

           for layout NDHWC
           (batch_size, size[0], size[1], size[2], channels)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Resize3DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(5)
    .add_type_rel("Resize3d", Resize3DRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

TVM_REGISTER_NODE_TYPE(CropAndResizeAttrs);

bool CropAndResizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* boxes = types[1].as<TensorTypeNode>();
  const auto* box_indices = types[2].as<TensorTypeNode>();
  if (data == nullptr || boxes == nullptr || box_indices == nullptr) return false;

  const CropAndResizeAttrs* param = attrs.as<CropAndResizeAttrs>();
  ICHECK(param != nullptr);
  auto crop_size = param->crop_size;

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // 4-D tensor of shape [num_boxes, crop_height, crop_width, depth]
  static const Layout kNCHW("NCHW");
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(0, boxes->shape[0]);
  oshape.Set(2, crop_size[0]);
  oshape.Set(3, crop_size[1]);
  auto bshape = layout_converter.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[3], TensorType(bshape, out_dtype));
  return true;
}

Expr MakeCropAndResize(Expr data, Expr boxes, Expr box_indices, Array<IndexExpr> crop_size,
                       String layout, String method, double extrapolation_value,
                       DataType out_dtype) {
  auto attrs = make_object<CropAndResizeAttrs>();
  attrs->crop_size = std::move(crop_size);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->extrapolation_value = std::move(extrapolation_value);
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("image.crop_and_resize");
  return Call(op, {data, boxes, box_indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.image._make.crop_and_resize").set_body_typed(MakeCropAndResize);

RELAY_REGISTER_OP("image.crop_and_resize")
    .describe(
        R"code(Perform crop and resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, crop_size[0], crop_size[1])

           for layout NHWC
           (batch_size, crop_size[0], crop_size[1], channels)
)code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("boxes", "Tensor", "The boxes tensor.")
    .add_argument("box_indices", "Tensor", "The box indices tensor.")
    .set_attrs_type<CropAndResizeAttrs>()
    .set_support_level(5)
    .add_type_rel("CropAndResize", CropAndResizeRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace relay
}  // namespace tvm
