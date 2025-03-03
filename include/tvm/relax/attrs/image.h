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
 * \file tvm/relax/attrs/image.h
 * \brief Attributes for image operators.
 */
#ifndef TVM_RELAX_ATTRS_IMAGE_H_
#define TVM_RELAX_ATTRS_IMAGE_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in image resize2d operator */
struct Resize2DAttrs : public tvm::AttrsNode<Resize2DAttrs> {
  Array<FloatImm> roi;
  String layout;
  String method;
  String coordinate_transformation_mode;
  String rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Resize2DAttrs, "relax.attrs.Resize2DAttrs") {
    TVM_ATTR_FIELD(roi).describe(
        "Region of Interest for coordinate transformation mode 'tf_crop_and_resize'");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Resize is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method).describe(
        "Specify the mode to use for scaling."
        "nearest_neighbor -  Nearest Neighbor"
        "linear - Bilinear Interpolation"
        "cubic - Bicubic Interpolation");
    TVM_ATTR_FIELD(coordinate_transformation_mode)
        .describe(
            "Describes how to transform the coordinate in the resized tensor"
            "to the coordinate in the original tensor."
            "Refer to the ONNX Resize operator specification for details"
            "Available options are half_pixel, align_corners and asymmetric");
    TVM_ATTR_FIELD(rounding_method)
        .describe(
            "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
            "Available options are round, floor, and ceil.");
    TVM_ATTR_FIELD(cubic_alpha).describe("Spline Coefficient for Bicubic Interpolation");
    TVM_ATTR_FIELD(cubic_exclude)
        .describe("Flag to exclude exterior of the image during bicubic interpolation");
    TVM_ATTR_FIELD(extrapolation_value)
        .describe("Value to return when roi is outside of the image");
    TVM_ATTR_FIELD(out_dtype).describe(
        "The dtype of the output tensor. It it is not specified, the output will have the same "
        "dtype as input if not specified.");
  }
};  // struct Resize2dAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_IMAGE_H_
