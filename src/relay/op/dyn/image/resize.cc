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

#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace dyn {

TVM_REGISTER_NODE_TYPE(Resize2DAttrs);

bool Resize2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // {data, size, roi, out}
  ICHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  static const Layout kNCHW("NCHW");

  const Resize2DAttrs* param = attrs.as<Resize2DAttrs>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->layout);
  auto layout_converter = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(layout_converter.defined())
      << "Resize only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  auto oshape = layout_converter.ForwardShape(data->shape);
  oshape.Set(2, Any());
  oshape.Set(3, Any());

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }

  // assign output type
  reporter->Assign(types[3], TensorType(layout_converter.BackwardShape(oshape), out_dtype));
  return true;
}

// Positional relay function to create image operator
// used by frontend FFI.
Expr MakeResize2D(Expr data, Expr size, Expr roi, String layout, String method,
                  String coordinate_transformation_mode, String rounding_method, double cubic_alpha,
                  double cubic_exclude, double extrapolation_value, DataType out_dtype) {
  auto attrs = make_object<Resize2DAttrs>();
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = coordinate_transformation_mode;
  attrs->rounding_method = rounding_method;
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("dyn.image.resize2d");
  return Call(op, {data, size, roi}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn.image._make.resize2d").set_body_typed(MakeResize2D);

RELAY_REGISTER_OP("dyn.image.resize2d")
    .describe(R"code(Perform resize to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **size**: data is 2D array of shape (2,) with values
            (new_height, new_width)

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, size[0], size[1])

           for layout NHWC
           (batch_size, size[0], size[1], channels)
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Resize2DAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("size", "Tensor", "The output size tensor.")
    .add_argument("roi", "Tensor", "The region of interest for tf_crop_and_resize.")
    .set_support_level(5)
    .add_type_rel("DynResize2D", Resize2DRel)
    .set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm
