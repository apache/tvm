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

TVM_FFI_STATIC_INIT_BLOCK() { AllClassNonMaximumSuppressionAttrs::RegisterReflection(); }

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
  ICHECK(!boxes_sinfo->IsUnknownNdim()) << "Only support known ndim";
  ICHECK(!scores_sinfo->IsUnknownNdim()) << "Only support known ndim";
  ICHECK_EQ(boxes_sinfo->ndim, 3) << "AllClassNMS input boxes should be 3-D.";
  ICHECK_EQ(scores_sinfo->ndim, 3) << "AllClassNMS input scores count should be 3-D.";

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

}  // namespace relax
}  // namespace tvm
