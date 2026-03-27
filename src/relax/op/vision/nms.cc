/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "nms.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/relax/attrs/vision.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/object.h>

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  AllClassNonMaximumSuppressionAttrs::RegisterReflection();
  GetValidCountsAttrs::RegisterReflection();
  NonMaximumSuppressionAttrs::RegisterReflection();
}

/* relax.vision.all_class_non_max_suppression */

Expr all_class_non_max_suppression(Expr boxes, Expr scores, Expr max_output_boxes_per_class,
                                   Expr iou_threshold, Expr score_threshold,
                                   ffi::String output_format) {
  auto attrs = tvm::ffi::make_object<AllClassNonMaximumSuppressionAttrs>();
  attrs->output_format = output_format;

  static const Op& op = Op::Get("relax.vision.all_class_non_max_suppression");
  return Call(op,
              {std::move(boxes), std::move(scores), std::move(max_output_boxes_per_class),
               std::move(iou_threshold), std::move(score_threshold)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.all_class_non_max_suppression",
                        all_class_non_max_suppression);
}

StructInfo InferStructInfoAllClassNMS(const Call& call, const BlockBuilder& ctx) {
  tvm::ffi::Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  const auto boxes_sinfo = input_sinfo[0];
  const auto scores_sinfo = input_sinfo[1];
  TVM_FFI_ICHECK(!boxes_sinfo->IsUnknownNdim()) << "Only support known ndim";
  TVM_FFI_ICHECK(!scores_sinfo->IsUnknownNdim()) << "Only support known ndim";
  TVM_FFI_ICHECK_EQ(boxes_sinfo->ndim, 3) << "AllClassNMS input boxes should be 3-D.";
  TVM_FFI_ICHECK_EQ(scores_sinfo->ndim, 3) << "AllClassNMS input scores count should be 3-D.";

  const auto batch = boxes_sinfo->shape.as<ShapeExprNode>()->values[0];
  const auto num_classes = scores_sinfo->shape.as<ShapeExprNode>()->values[1];
  const auto num_boxes = boxes_sinfo->shape.as<ShapeExprNode>()->values[1];

  auto vdev = input_sinfo[0]->vdevice;
  const auto* attrs = call->attrs.as<AllClassNonMaximumSuppressionAttrs>();
  if (attrs->output_format == "onnx") {
    auto vdev = input_sinfo[0]->vdevice;
    auto num_total_boxes = batch * num_classes * num_boxes;
    tvm::ffi::Array<PrimExpr> oshape_values = {num_total_boxes, 3};
    ShapeExpr oshape(oshape_values);
    tvm::ffi::Array<PrimExpr> counts_values = {1};
    ShapeExpr counts_shape(counts_values);
    tvm::ffi::Array<StructInfo> fields = {TensorStructInfo(oshape, DataType::Int(64), vdev),
                                          TensorStructInfo(counts_shape, DataType::Int(64), vdev)};
    return TupleStructInfo(fields);
  }

  auto num_total_boxes_per_batch = num_classes * num_boxes;
  tvm::ffi::Array<PrimExpr> indices_values = {batch, num_total_boxes_per_batch, 2};
  ShapeExpr indices_shape(indices_values);
  tvm::ffi::Array<PrimExpr> scores_values = {batch, num_total_boxes_per_batch};
  ShapeExpr scores_shape(scores_values);
  tvm::ffi::Array<PrimExpr> counts_values = {batch};
  ShapeExpr counts_shape(counts_values);
  tvm::ffi::Array<StructInfo> fields = {TensorStructInfo(indices_shape, DataType::Int(64), vdev),
                                        TensorStructInfo(scores_shape, DataType::Float(32), vdev),
                                        TensorStructInfo(counts_shape, DataType::Int(64), vdev)};
  return TupleStructInfo(fields);
}

TVM_REGISTER_OP("relax.vision.all_class_non_max_suppression")
    .set_attrs_type<AllClassNonMaximumSuppressionAttrs>()
    .set_num_inputs(5)
    .add_argument("boxes", "Tensor", "The input boxes in the format [batch, num_boxes, 4].")
    .add_argument("scores", "Tensor",
                  "Scores for each box and class in the format [batch, num_classes, num_boxes].")
    .add_argument("max_output_boxes_per_class", "Tensor",
                  "The maximum number of output boxes per class.")
    .add_argument("iou_threshold", "Tensor", "The IoU threshold for box the overlap test.")
    .add_argument("score_threshold", "Tensor",
                  "The score threshold to filter out low score boxes early.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAllClassNMS)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.vision.get_valid_counts */

Expr get_valid_counts(Expr data, double score_threshold, int id_index, int score_index) {
  auto attrs = tvm::ffi::make_object<GetValidCountsAttrs>();
  attrs->score_threshold = score_threshold;
  attrs->id_index = id_index;
  attrs->score_index = score_index;

  static const Op& op = Op::Get("relax.vision.get_valid_counts");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.get_valid_counts", get_valid_counts);
}

StructInfo InferStructInfoGetValidCounts(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "get_valid_counts expects 1 argument, got " << call->args.size());
  }

  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "get_valid_counts expects input data to be a Tensor.");
  }
  if (data_sinfo->ndim != -1 && data_sinfo->ndim != 3) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "get_valid_counts expects 3-D input, got ndim " << data_sinfo->ndim);
  }

  const auto* attrs = call->attrs.as<GetValidCountsAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid get_valid_counts attrs";
  auto vdev = data_sinfo->vdevice;
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    tvm::ffi::Array<StructInfo> fields = {
        TensorStructInfo(DataType::Int(32), /*ndim=*/1, vdev),
        TensorStructInfo(data_sinfo->dtype, /*ndim=*/3, vdev),
        TensorStructInfo(DataType::Int(32), /*ndim=*/2, vdev)};
    return TupleStructInfo(fields);
  }

  auto batch = data_shape->values[0];
  auto num_anchors = data_shape->values[1];
  auto elem_length = data_shape->values[2];
  const auto* elem_length_imm = elem_length.as<IntImmNode>();
  if (elem_length_imm != nullptr) {
    if (attrs->score_index < 0 || attrs->score_index >= elem_length_imm->value) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "get_valid_counts expects score_index to be in range [0, "
                       << elem_length_imm->value << "), but got " << attrs->score_index);
    }
    if (attrs->id_index >= elem_length_imm->value) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "get_valid_counts expects id_index to be smaller than elem_length "
                       << elem_length_imm->value << ", but got " << attrs->id_index);
    }
  }

  tvm::ffi::Array<StructInfo> fields = {
      TensorStructInfo(ShapeExpr({batch}), DataType::Int(32), vdev),
      TensorStructInfo(ShapeExpr({batch, num_anchors, elem_length}), data_sinfo->dtype, vdev),
      TensorStructInfo(ShapeExpr({batch, num_anchors}), DataType::Int(32), vdev)};
  return TupleStructInfo(fields);
}

TVM_REGISTER_OP("relax.vision.get_valid_counts")
    .set_attrs_type<GetValidCountsAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor",
                  "Input data, 3-D tensor [batch_size, num_anchors, elem_length].")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoGetValidCounts)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.vision.non_max_suppression */

Expr non_max_suppression(Expr data, Expr valid_count, Expr indices, int max_output_size,
                         double iou_threshold, bool force_suppress, int top_k, int coord_start,
                         int score_index, int id_index, bool return_indices,
                         bool invalid_to_bottom) {
  auto attrs = tvm::ffi::make_object<NonMaximumSuppressionAttrs>();
  attrs->max_output_size = max_output_size;
  attrs->iou_threshold = iou_threshold;
  attrs->force_suppress = force_suppress;
  attrs->top_k = top_k;
  attrs->coord_start = coord_start;
  attrs->score_index = score_index;
  attrs->id_index = id_index;
  attrs->return_indices = return_indices;
  attrs->invalid_to_bottom = invalid_to_bottom;

  static const Op& op = Op::Get("relax.vision.non_max_suppression");
  return Call(op, {std::move(data), std::move(valid_count), std::move(indices)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vision.non_max_suppression", non_max_suppression);
}

StructInfo InferStructInfoNMS(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects 3 arguments, got " << call->args.size());
  }

  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* valid_count_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const auto* indices_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[2]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects input data to be a Tensor.");
  }
  if (valid_count_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects valid_count to be a Tensor.");
  }
  if (indices_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects indices to be a Tensor.");
  }
  if (data_sinfo->ndim != -1 && data_sinfo->ndim != 3) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects 3-D input, got ndim " << data_sinfo->ndim);
  }
  if (valid_count_sinfo->ndim != -1 && valid_count_sinfo->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects valid_count to be 1-D, got ndim "
                     << valid_count_sinfo->ndim);
  }
  if (indices_sinfo->ndim != -1 && indices_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects indices to be 2-D, got ndim "
                     << indices_sinfo->ndim);
  }
  if (!valid_count_sinfo->IsUnknownDtype() && valid_count_sinfo->dtype != DataType::Int(32)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects valid_count to have dtype int32, got "
                     << valid_count_sinfo->dtype);
  }
  if (!indices_sinfo->IsUnknownDtype() && indices_sinfo->dtype != DataType::Int(32)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "non_max_suppression expects indices to have dtype int32, got "
                     << indices_sinfo->dtype);
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  const auto* valid_count_shape = valid_count_sinfo->shape.as<ShapeExprNode>();
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  if (data_shape != nullptr) {
    arith::Analyzer* analyzer = ctx->GetAnalyzer();
    PrimExpr batch = data_shape->values[0];
    PrimExpr num_anchors = data_shape->values[1];
    if (valid_count_shape != nullptr &&
        !analyzer->CanProveEqual(valid_count_shape->values[0], batch)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "non_max_suppression expects valid_count to have shape [batch_size]. "
                          "However, the given data tensor has batch size `"
                       << batch << "` and the given valid_count tensor has shape "
                       << valid_count_sinfo->shape);
    }
    if (indices_shape != nullptr) {
      if (!analyzer->CanProveEqual(indices_shape->values[0], batch) ||
          !analyzer->CanProveEqual(indices_shape->values[1], num_anchors)) {
        ctx->ReportFatal(
            Diagnostic::Error(call)
            << "non_max_suppression expects indices to have shape [batch_size, num_anchors]. "
               "However, the given data tensor has shape "
            << data_sinfo->shape << " and the given indices tensor has shape "
            << indices_sinfo->shape);
      }
    }
  }

  const auto* attrs = call->attrs.as<NonMaximumSuppressionAttrs>();
  TVM_FFI_ICHECK(attrs != nullptr) << "Invalid non_max_suppression attrs";
  auto vdev = data_sinfo->vdevice;
  if (data_shape != nullptr) {
    const auto* elem_length_imm = data_shape->values[2].as<IntImmNode>();
    if (elem_length_imm != nullptr) {
      int64_t elem_length = elem_length_imm->value;
      if (attrs->score_index < 0 || attrs->score_index >= elem_length) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "non_max_suppression expects score_index to be in range [0, "
                         << elem_length << "), but got " << attrs->score_index);
      }
      if (attrs->coord_start < 0 || attrs->coord_start + 3 >= elem_length) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "non_max_suppression expects coord_start to reference four "
                            "consecutive box coordinates within elem_length "
                         << elem_length << ", but got " << attrs->coord_start);
      }
      if (attrs->id_index >= elem_length) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "non_max_suppression expects id_index to be smaller than "
                            "elem_length "
                         << elem_length << ", but got " << attrs->id_index);
      }
    }
  }

  if (attrs->return_indices) {
    // Returns (box_indices[batch, num_anchors], valid_box_count[batch, 1])
    if (data_shape == nullptr) {
      tvm::ffi::Array<StructInfo> fields = {
          TensorStructInfo(DataType::Int(32), /*ndim=*/2, vdev),
          TensorStructInfo(DataType::Int(32), /*ndim=*/2, vdev)};
      return TupleStructInfo(fields);
    }
    auto batch = data_shape->values[0];
    auto num_anchors = data_shape->values[1];
    tvm::ffi::Array<StructInfo> fields = {
        TensorStructInfo(ShapeExpr({batch, num_anchors}), DataType::Int(32), vdev),
        TensorStructInfo(ShapeExpr({batch, IntImm(DataType::Int(64), 1)}), DataType::Int(32),
                         vdev)};
    return TupleStructInfo(fields);
  }

  // Returns modified data tensor with the same shape as input.
  if (const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>()) {
    return TensorStructInfo(ffi::GetRef<ShapeExpr>(data_shape), data_sinfo->dtype, vdev);
  }
  return TensorStructInfo(data_sinfo->dtype, /*ndim=*/3, vdev);
}

TVM_REGISTER_OP("relax.vision.non_max_suppression")
    .set_attrs_type<NonMaximumSuppressionAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor",
                  "Input data, 3-D tensor [batch_size, num_anchors, elem_length].")
    .add_argument("valid_count", "Tensor", "1-D tensor for valid number of boxes.")
    .add_argument("indices", "Tensor", "2-D tensor with shape [batch_size, num_anchors].")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoNMS)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
