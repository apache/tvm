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
 * \file tvm/relax/attrs/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAX_ATTRS_VISION_H_
#define TVM_RELAX_ATTRS_VISION_H_

#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/type.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in AllClassNonMaximumSuppression operator */
struct AllClassNonMaximumSuppressionAttrs
    : public AttrsNodeReflAdapter<AllClassNonMaximumSuppressionAttrs> {
  ffi::String output_format;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllClassNonMaximumSuppressionAttrs>().def_ro(
        "output_format", &AllClassNonMaximumSuppressionAttrs::output_format,
        "Output format, onnx or tensorflow. Returns outputs in a way that can be easily "
        "consumed by each frontend.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllClassNonMaximumSuppressionAttrs",
                                    AllClassNonMaximumSuppressionAttrs, BaseAttrsNode);
};  // struct AllClassNonMaximumSuppressionAttrs

/*! \brief Attributes used in ROIAlign operator */
struct ROIAlignAttrs : public AttrsNodeReflAdapter<ROIAlignAttrs> {
  ffi::Array<int64_t> pooled_size;
  double spatial_scale;
  int sample_ratio;
  bool aligned;
  ffi::String layout;
  ffi::String mode;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ROIAlignAttrs>()
        .def_ro("pooled_size", &ROIAlignAttrs::pooled_size, "Output size of roi align.")
        .def_ro("spatial_scale", &ROIAlignAttrs::spatial_scale,
                "Ratio of input feature map height (or width) to raw image height (or width).")
        .def_ro("sample_ratio", &ROIAlignAttrs::sample_ratio,
                "Optional sampling ratio of ROI align, using adaptive size by default.")
        .def_ro("aligned", &ROIAlignAttrs::aligned,
                "Whether to use the aligned ROIAlign semantics without the legacy 1-pixel clamp.")
        .def_ro("layout", &ROIAlignAttrs::layout, "Dimension ordering of the input data.")
        .def_ro("mode", &ROIAlignAttrs::mode, "Mode for ROI Align. Can be 'avg' or 'max'.");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ROIAlignAttrs", ROIAlignAttrs, BaseAttrsNode);
};  // struct ROIAlignAttrs

/*! \brief Attributes for multibox_transform_loc (SSD / TFLite-style box decode). */
struct MultiboxTransformLocAttrs : public AttrsNodeReflAdapter<MultiboxTransformLocAttrs> {
  bool clip;
  double threshold;
  ffi::Array<double> variances;
  bool keep_background;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MultiboxTransformLocAttrs>()
        .def_ro("clip", &MultiboxTransformLocAttrs::clip, "Clip decoded ymin,xmin,ymax,xmax to [0,1].")
        .def_ro("threshold", &MultiboxTransformLocAttrs::threshold,
                "After softmax, zero scores strictly below this value.")
        .def_ro("variances", &MultiboxTransformLocAttrs::variances,
                "(x,y,w,h) scales = TFLite 1/x_scale,1/y_scale,1/w_scale,1/h_scale on encodings.")
        .def_ro("keep_background", &MultiboxTransformLocAttrs::keep_background,
                "If false, force output scores[:,0,:] to 0 (background class).");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MultiboxTransformLocAttrs",
                                    MultiboxTransformLocAttrs, BaseAttrsNode);
};  // struct MultiboxTransformLocAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_VISION_H_
