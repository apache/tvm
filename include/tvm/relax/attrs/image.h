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
struct Resize2DAttrs : public AttrsNodeReflAdapter<Resize2DAttrs> {
  Array<FloatImm> roi;
  String layout;
  String method;
  String coordinate_transformation_mode;
  String rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  DataType out_dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Resize2DAttrs>()
        .def_ro("roi", &Resize2DAttrs::roi,
                "Region of Interest for coordinate transformation mode 'tf_crop_and_resize'")
        .def_ro("layout", &Resize2DAttrs::layout,
                "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Resize is applied on the 'H' and"
                "'W' dimensions.")
        .def_ro("method", &Resize2DAttrs::method,
                "Specify the mode to use for scaling."
                "nearest_neighbor -  Nearest Neighbor"
                "linear - Bilinear Interpolation"
                "cubic - Bicubic Interpolation")
        .def_ro("coordinate_transformation_mode", &Resize2DAttrs::coordinate_transformation_mode,
                "Describes how to transform the coordinate in the resized tensor"
                "to the coordinate in the original tensor."
                "Refer to the ONNX Resize operator specification for details"
                "Available options are half_pixel, align_corners and asymmetric")
        .def_ro("rounding_method", &Resize2DAttrs::rounding_method,
                "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
                "Available options are round, floor, and ceil.")
        .def_ro("cubic_alpha", &Resize2DAttrs::cubic_alpha,
                "Spline Coefficient for Bicubic Interpolation")
        .def_ro("cubic_exclude", &Resize2DAttrs::cubic_exclude,
                "Flag to exclude exterior of the image during bicubic interpolation")
        .def_ro("extrapolation_value", &Resize2DAttrs::extrapolation_value,
                "Value to return when roi is outside of the image")
        .def_ro(
            "out_dtype", &Resize2DAttrs::out_dtype,
            "The dtype of the output tensor. It it is not specified, the output will have the same "
            "dtype as input if not specified.");
  }

  static constexpr const char* _type_key = "relax.attrs.Resize2DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Resize2DAttrs, BaseAttrsNode);
};  // struct Resize2dAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_IMAGE_H_
