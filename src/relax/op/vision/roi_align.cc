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
 * \file roi_align.cc
 * \brief ROI Align operators.
 */

#include "roi_align.h"

#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() { ROIAlignAttrs::RegisterReflection(); }

Expr roi_align(Expr data, Expr rois, ffi::Array<int64_t> pooled_size, double spatial_scale,
               int sample_ratio, bool aligned, ffi::String layout, ffi::String mode) {
  if (pooled_size.size() == 1) {
    pooled_size.push_back(pooled_size[0]);
  }
  TVM_FFI_ICHECK_EQ(pooled_size.size(), 2)
      << "The input pooled_size length is expected to be 2. However, the given pooled_size is "
      << pooled_size;

  auto attrs = ffi::make_object<ROIAlignAttrs>();
  attrs->pooled_size = std::move(pooled_size);
  attrs->spatial_scale = spatial_scale;
  attrs->sample_ratio = sample_ratio;
  attrs->aligned = aligned;
  attrs->layout = layout;
  attrs->mode = mode;

  static const Op& op = Op::Get("relax.vision.roi_align");
  return Call(op, {std::move(data), std::move(rois)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.roi_align", roi_align);
}

StructInfo InferStructInfoROIAlign(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ROIAlign expects two arguments, while the given number of arguments is "
                     << call->args.size());
  }

  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* rois_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ROIAlign expects the input data to be a Tensor, while the given data is "
                     << call->args[0]->GetTypeKey());
  }
  if (rois_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ROIAlign expects the rois to be a Tensor, while the given rois is "
                     << call->args[1]->GetTypeKey());
  }
  if (!data_sinfo->IsUnknownNdim() && data_sinfo->ndim != 4) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ROIAlign expects the input data to be 4-D, while the given data has ndim "
                     << data_sinfo->ndim);
  }
  if (!rois_sinfo->IsUnknownNdim() && rois_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ROIAlign expects the rois tensor to be 2-D, while the given rois has ndim "
                     << rois_sinfo->ndim);
  }

  const auto* attrs = call->attrs.as<ROIAlignAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid ROIAlign attrs";
  if (attrs->layout != "NCHW" && attrs->layout != "NHWC") {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ROIAlign only supports NCHW and NHWC layout, but got " << attrs->layout);
  }
  if (attrs->mode != "avg" && attrs->mode != "max") {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "ROIAlign only supports avg and max mode, but got " << attrs->mode);
  }

  const auto* rois_shape = rois_sinfo->shape.as<ShapeExprNode>();
  if (rois_shape != nullptr) {
    const auto* last_dim = rois_shape->values[1].as<IntImmNode>();
    if (last_dim != nullptr && last_dim->value != 5) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "ROIAlign expects rois to have shape (num_roi, 5), but got last "
                          "dimension "
                       << last_dim->value);
    }
  }

  if (data_sinfo->shape.as<ShapeExprNode>() == nullptr || rois_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, 4, data_sinfo->vdevice);
  }

  ffi::Array<PrimExpr> data_shape = data_sinfo->shape.as<ShapeExprNode>()->values;
  ffi::Array<PrimExpr> out_shape;
  if (attrs->layout == "NCHW") {
    out_shape = {rois_shape->values[0], data_shape[1], Integer(attrs->pooled_size[0]),
                 Integer(attrs->pooled_size[1])};
  } else {
    out_shape = {rois_shape->values[0], Integer(attrs->pooled_size[0]),
                 Integer(attrs->pooled_size[1]), data_shape[3]};
  }
  return TensorStructInfo(ShapeExpr(out_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.vision.roi_align")
    .set_attrs_type<ROIAlignAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("rois", "Tensor",
                  "The input rois with shape (num_roi, 5) in [batch_idx, x1, y1, x2, y2] format.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoROIAlign)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
