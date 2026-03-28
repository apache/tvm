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
 * \file multibox_transform_loc.cc
 * \brief Multibox transform (location decode) for object detection.
 */

#include "multibox_transform_loc.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/struct_info.h>

#include <utility>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() { MultiboxTransformLocAttrs::RegisterReflection(); }

Expr multibox_transform_loc(Expr cls_pred, Expr loc_pred, Expr anchor, bool clip, double threshold,
                            ffi::Array<double> variances, bool keep_background) {
  TVM_FFI_ICHECK_EQ(variances.size(), 4)
      << "multibox_transform_loc: variances must be length 4 (x,y,w,h), got " << variances.size();

  auto attrs = ffi::make_object<MultiboxTransformLocAttrs>();
  attrs->clip = clip;
  attrs->threshold = threshold;
  attrs->variances = std::move(variances);
  attrs->keep_background = keep_background;

  static const Op& op = Op::Get("relax.vision.multibox_transform_loc");
  return Call(op, {std::move(cls_pred), std::move(loc_pred), std::move(anchor)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.multibox_transform_loc", multibox_transform_loc);
}

/*!
 * \brief Infer struct info for relax.vision.multibox_transform_loc.
 *
 * \note Shape cross-checks that need the anchor count N (e.g. loc_pred.shape[1] == 4*N,
 * anchor.shape[1] == N with N = cls_pred.shape[2]) run only when cls_pred has a known
 * static shape. If cls_pred shape is unknown, inference returns generic rank-3 outputs and
 * skips those N-based relations; other checks (ndim, dtype, loc dim divisible by 4, etc.)
 * still apply when their inputs are known.
 */
StructInfo InferStructInfoMultiboxTransformLoc(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "multibox_transform_loc: expected 3 inputs (cls_pred, loc_pred, anchor), "
                        "got "
                     << call->args.size());
  }

  ffi::Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  const auto cls_sinfo = input_sinfo[0];
  const auto loc_sinfo = input_sinfo[1];
  const auto anchor_sinfo = input_sinfo[2];

  if (!cls_sinfo->IsUnknownNdim() && cls_sinfo->ndim != 3) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "multibox_transform_loc: cls_pred must be 3-D [B, num_classes, N], got "
                        "ndim "
                     << cls_sinfo->ndim);
  }
  if (!loc_sinfo->IsUnknownNdim() && loc_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "multibox_transform_loc: loc_pred must be 2-D [B, 4*N], got ndim "
                     << loc_sinfo->ndim);
  }
  if (!anchor_sinfo->IsUnknownNdim() && anchor_sinfo->ndim != 3) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "multibox_transform_loc: anchor must be 3-D [1, N, 4] ltrb, got ndim "
                     << anchor_sinfo->ndim);
  }

  if (!cls_sinfo->IsUnknownDtype() && !loc_sinfo->IsUnknownDtype() &&
      cls_sinfo->dtype != loc_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "multibox_transform_loc: cls_pred and loc_pred dtype must match, got "
                     << cls_sinfo->dtype << " vs " << loc_sinfo->dtype);
  }
  if (!cls_sinfo->IsUnknownDtype() && !anchor_sinfo->IsUnknownDtype() &&
      cls_sinfo->dtype != anchor_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "multibox_transform_loc: cls_pred and anchor dtype must match, got "
                     << cls_sinfo->dtype << " vs " << anchor_sinfo->dtype);
  }

  auto vdev = cls_sinfo->vdevice;
  const auto* cls_shape = cls_sinfo->shape.as<ShapeExprNode>();
  const auto* loc_shape = loc_sinfo->shape.as<ShapeExprNode>();
  const auto* anchor_shape = anchor_sinfo->shape.as<ShapeExprNode>();

  if (loc_shape != nullptr) {
    const auto* loc_dim1 = loc_shape->values[1].as<IntImmNode>();
    if (loc_dim1 != nullptr && loc_dim1->value % 4 != 0) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "multibox_transform_loc: loc_pred.shape[1] must be divisible by 4, got "
                       << loc_dim1->value);
    }
  }

  if (cls_shape != nullptr && loc_shape != nullptr) {
    const auto* cls_b = cls_shape->values[0].as<IntImmNode>();
    const auto* loc_b = loc_shape->values[0].as<IntImmNode>();
    if (cls_b != nullptr && loc_b != nullptr && cls_b->value != loc_b->value) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "multibox_transform_loc: cls_pred.shape[0] must match loc_pred.shape[0], "
                          "got B="
                       << cls_b->value << " vs " << loc_b->value);
    }
  }

  if (anchor_shape != nullptr) {
    const auto* anchor_batch = anchor_shape->values[0].as<IntImmNode>();
    if (anchor_batch != nullptr && anchor_batch->value != 1) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "multibox_transform_loc: anchor.shape[0] must be 1, got "
                       << anchor_batch->value);
    }
    const auto* anchor_last = anchor_shape->values[2].as<IntImmNode>();
    if (anchor_last != nullptr && anchor_last->value != 4) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "multibox_transform_loc: anchor.shape[2] must be 4 (ltrb), got "
                       << anchor_last->value);
    }
  }

  if (cls_shape == nullptr) {
    ffi::Array<StructInfo> fields = {TensorStructInfo(cls_sinfo->dtype, 3, vdev),
                                     TensorStructInfo(cls_sinfo->dtype, 3, vdev)};
    return TupleStructInfo(fields);
  }

  const auto& batch = cls_shape->values[0];
  const auto& num_classes = cls_shape->values[1];
  const auto& num_anchors = cls_shape->values[2];

  if (loc_shape != nullptr) {
    const auto* num_anchors_imm = num_anchors.as<IntImmNode>();
    const auto* loc_dim1 = loc_shape->values[1].as<IntImmNode>();
    if (num_anchors_imm != nullptr && loc_dim1 != nullptr &&
        loc_dim1->value != num_anchors_imm->value * 4) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "multibox_transform_loc: loc_pred.shape[1] must equal 4*N with "
                          "N=cls_pred.shape[2], got loc_dim="
                       << loc_dim1->value << ", N=" << num_anchors_imm->value);
    }
  }
  if (anchor_shape != nullptr) {
    const auto* num_anchors_imm = num_anchors.as<IntImmNode>();
    const auto* anchor_num_anchors = anchor_shape->values[1].as<IntImmNode>();
    if (num_anchors_imm != nullptr && anchor_num_anchors != nullptr &&
        anchor_num_anchors->value != num_anchors_imm->value) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "multibox_transform_loc: anchor.shape[1] must equal N=cls_pred.shape[2], "
                          "got anchor_N="
                       << anchor_num_anchors->value << ", N=" << num_anchors_imm->value);
    }
  }

  ffi::Array<PrimExpr> boxes_shape = {batch, num_anchors, Integer(4)};
  ffi::Array<PrimExpr> scores_shape = {batch, num_classes, num_anchors};
  ffi::Array<StructInfo> fields = {
      TensorStructInfo(ShapeExpr(boxes_shape), cls_sinfo->dtype, vdev),
      TensorStructInfo(ShapeExpr(scores_shape), cls_sinfo->dtype, vdev)};
  return TupleStructInfo(fields);
}

TVM_REGISTER_OP("relax.vision.multibox_transform_loc")
    .describe("Decode SSD/TFLite-style priors and offsets into boxes and softmax scores. If "
              "cls_pred shape is unknown, N-based loc/anchor shape checks are skipped in "
              "inference. Very large variances (w,h) can overflow exp in half box sizes.")
    .set_attrs_type<MultiboxTransformLocAttrs>()
    .set_num_inputs(3)
    .add_argument("cls_pred", "Tensor", "[B,C,N] class logits (pre-softmax).")
    .add_argument("loc_pred", "Tensor",
                  "[B,4*N] box encodings (x,y,w,h); TFLite yxhw order remapped to xywh.")
    .add_argument("anchor", "Tensor", "[1,N,4] priors as ltrb (left,top,right,bottom).")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMultiboxTransformLoc)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
