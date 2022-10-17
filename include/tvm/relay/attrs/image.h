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
 * \file tvm/relay/attrs/image.h
 * \brief Auxiliary attributes for image operators.
 */
#ifndef TVM_RELAY_ATTRS_IMAGE_H_
#define TVM_RELAY_ATTRS_IMAGE_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>

#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in image resize1d operator */
struct Resize1DAttrs : public tvm::AttrsNode<Resize1DAttrs> {
  Array<IndexExpr> size;
  Array<FloatImm> roi;
  std::string layout;
  std::string method;
  std::string coordinate_transformation_mode;
  std::string rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Resize1DAttrs, "relay.attrs.Resize1DAttrs") {
    TVM_ATTR_FIELD(size).set_default(NullValue<Array<IndexExpr>>()).describe("Output Size.");
    TVM_ATTR_FIELD(roi)
        .set_default(NullValue<Array<FloatImm>>())
        .describe("Region of Interest for coordinate transformation mode 'tf_crop_and_resize'");
    TVM_ATTR_FIELD(layout).set_default("NCW").describe(
        "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
        "'N', 'C', 'W' stands for batch, channel and width"
        "dimensions respectively. Resize is applied on the"
        "'W' dimension.");
    TVM_ATTR_FIELD(method).set_default("linear").describe(
        "Specify the mode to use for scaling."
        "nearest_neighbor -  Nearest Neighbor"
        "linear - Linear Interpolation"
        "cubic - Cubic Interpolation");
    TVM_ATTR_FIELD(coordinate_transformation_mode)
        .set_default("half_pixel")
        .describe(
            "Describes how to transform the coordinate in the resized tensor"
            "to the coordinate in the original tensor."
            "Refer to the ONNX Resize operator specification for details"
            "Available options are half_pixel, align_corners and asymmetric");
    TVM_ATTR_FIELD(rounding_method)
        .set_default("round")
        .describe(
            "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
            "Available options are round, floor, and ceil.");
    TVM_ATTR_FIELD(cubic_alpha)
        .set_default(-0.5)
        .describe("Spline Coefficient for cubic interpolation");
    TVM_ATTR_FIELD(cubic_exclude)
        .set_default(0)
        .describe("Flag to exclude exterior of the image during cubic interpolation");
    TVM_ATTR_FIELD(extrapolation_value)
        .set_default(0.0)
        .describe("Value to return when roi is outside of the image");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
  }
};

/*! \brief Attributes used in image resize2d operator */
struct Resize2DAttrs : public tvm::AttrsNode<Resize2DAttrs> {
  Array<IndexExpr> size;
  Array<FloatImm> roi;
  std::string layout;
  std::string method;
  std::string coordinate_transformation_mode;
  std::string rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Resize2DAttrs, "relay.attrs.Resize2DAttrs") {
    TVM_ATTR_FIELD(size).set_default(NullValue<Array<IndexExpr>>()).describe("Output Size.");
    TVM_ATTR_FIELD(roi)
        .set_default(NullValue<Array<FloatImm>>())
        .describe("Region of Interest for coordinate transformation mode 'tf_crop_and_resize'");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Resize is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method).set_default("linear").describe(
        "Specify the mode to use for scaling."
        "nearest_neighbor -  Nearest Neighbor"
        "linear - Bilinear Interpolation"
        "cubic - Bicubic Interpolation");
    TVM_ATTR_FIELD(coordinate_transformation_mode)
        .set_default("half_pixel")
        .describe(
            "Describes how to transform the coordinate in the resized tensor"
            "to the coordinate in the original tensor."
            "Refer to the ONNX Resize operator specification for details"
            "Available options are half_pixel, align_corners and asymmetric");
    TVM_ATTR_FIELD(rounding_method)
        .set_default("round")
        .describe(
            "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
            "Available options are round, floor, and ceil.");
    TVM_ATTR_FIELD(cubic_alpha)
        .set_default(-0.5)
        .describe("Spline Coefficient for Bicubic Interpolation");
    TVM_ATTR_FIELD(cubic_exclude)
        .set_default(0)
        .describe("Flag to exclude exterior of the image during bicubic interpolation");
    TVM_ATTR_FIELD(extrapolation_value)
        .set_default(0.0)
        .describe("Value to return when roi is outside of the image");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
  }
};

/*! \brief Attributes used in image resize3d operator */
struct Resize3DAttrs : public tvm::AttrsNode<Resize3DAttrs> {
  Array<IndexExpr> size;
  Array<FloatImm> roi;
  std::string layout;
  std::string method;
  std::string coordinate_transformation_mode;
  std::string rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Resize3DAttrs, "relay.attrs.Resize3DAttrs") {
    TVM_ATTR_FIELD(size).set_default(NullValue<Array<IndexExpr>>()).describe("Output Size.");
    TVM_ATTR_FIELD(roi)
        .set_default(NullValue<Array<FloatImm>>())
        .describe("Region of Interest for coordinate transformation mode 'tf_crop_and_resize'");
    TVM_ATTR_FIELD(layout).set_default("NCDHW").describe(
        "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively. Resize3d is applied on the 'D', 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method).set_default("linear").describe(
        "Specify the mode to use for scaling."
        "nearest_neighbor -  Nearest Neighbor"
        "linear - Trilinear Interpolation"
        "cubic - Tricubic Interpolation");
    TVM_ATTR_FIELD(coordinate_transformation_mode)
        .set_default("half_pixel")
        .describe(
            "Describes how to transform the coordinate in the resized tensor"
            "to the coordinate in the original tensor."
            "Refer to the ONNX Resize operator specification for details"
            "Available options are half_pixel, align_corners and asymmetric");
    TVM_ATTR_FIELD(rounding_method)
        .set_default("round")
        .describe(
            "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
            "Available options are round, floor, and ceil.");
    TVM_ATTR_FIELD(cubic_alpha)
        .set_default(-0.5)
        .describe("Spline Coefficient for Tricubic Interpolation");
    TVM_ATTR_FIELD(cubic_exclude)
        .set_default(0)
        .describe("Flag to exclude exterior of the image during tricubic interpolation");
    TVM_ATTR_FIELD(extrapolation_value)
        .set_default(0.0)
        .describe("Value to return when roi is outside of the image");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
  }
};

/*! \brief Attributes used in image crop_and_resize operator */
struct CropAndResizeAttrs : public tvm::AttrsNode<CropAndResizeAttrs> {
  Array<IndexExpr> crop_size;
  std::string layout;
  std::string method;
  double extrapolation_value;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(CropAndResizeAttrs, "relay.attrs.CropAndResizeAttrs") {
    TVM_ATTR_FIELD(crop_size).set_default(NullValue<Array<IndexExpr>>()).describe("Target Size.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Resize is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method)
        .set_default("bilinear")
        .describe(
            "Specify the mode to use for scaling."
            "nearest_neighbor -  Nearest Neighbor"
            "bilinear - Bilinear Interpolation");
    TVM_ATTR_FIELD(extrapolation_value)
        .set_default(0.0)
        .describe("Specify value for extrapolation.");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
  }
};

/*! \brief Attributes used in dilation operators */
struct Dilation2DAttrs : public tvm::AttrsNode<Dilation2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilations;
  std::string data_layout;
  std::string kernel_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Dilation2DAttrs, "relay.attrs.Dilation2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the sliding window. [stride_height, stride_width].");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(dilations)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the dilation rate to use. [dilation_height, dilation_width]");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("IHW")
        .describe(
            "Dimension ordering of weight. Can be 'IHW', 'HWI', etc."
            "'I', 'H', 'W' stands for input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in image affine_grid operator */
struct AffineGridAttrs : public tvm::AttrsNode<AffineGridAttrs> {
  Array<IndexExpr> target_shape;

  TVM_DECLARE_ATTRS(AffineGridAttrs, "relay.attrs.AffineGridAttrs") {
    TVM_ATTR_FIELD(target_shape).describe("Specifies the output shape (H, W).");
  }
};

/*! \brief Attributes used in image grid_sample operator */
struct GridSampleAttrs : public tvm::AttrsNode<GridSampleAttrs> {
  String method;
  String layout;
  String padding_mode;
  bool align_corners;

  TVM_DECLARE_ATTRS(GridSampleAttrs, "relay.attrs.GridSampleAttrs") {
    TVM_ATTR_FIELD(method)
        .set_default("bilinear")
        .describe(
            "Specify the mode to use for scaling."
            "nearest - 2D or 3D Nearest Interpolation."
            "bilinear - '2D Bilinear' or '3D Trilinear' Interpolation."
            "bicubic - 2D Bicubic Interpolation.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NCDHW', etc."
        "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
        "dimensions respectively."
        "2D Resize is applied on the 'H' and 'W' dimensions."
        "3D Resize is applied on the 'D' and 'H' and 'W' dimensions.");
    TVM_ATTR_FIELD(padding_mode)
        .set_default("zeros")
        .describe(
            "If :attr:'grid' has values outside the range of '[-1, 1]', the corresponding"
            "outputs are handled as defined by padding_mode. Options are"
            "padding_mode='zeros': use '0' for out-of-bound grid locations,"
            "padding_mode='border': use border values for out-of-bound grid locations"
            "padding_mode='reflection': use values at locations reflected by"
            "the border for out-of-bound grid locations. For location far away"
            "from the border, it will keep being reflected until becoming in bound,"
            "e.g., (normalized) pixel location 'x = -3.5' reflects by border '-1'"
            "and becomes 'x' = 1.5, then reflects by border '1' and becomes"
            "'x' = -0.5");
    TVM_ATTR_FIELD(align_corners)
        .set_default(true)
        .describe(
            "Geometrically, we consider the pixels of the"
            "input as squares rather than points."
            "If set to True, the extrema (-1 and 1) are considered as referring"
            "to the center points of the input's corner pixels. If set to False, they"
            "are instead considered as referring to the corner points of the input's corner"
            "pixels, making the sampling more resolution agnostic.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_IMAGE_H_
