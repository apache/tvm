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
 * \file nms.cc
 * \brief Non-maximum suppression operators
 */
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/op.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(GetValidCountsAttrs);

bool GetValidCountRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const auto& dshape = data->shape;
  ICHECK_EQ(dshape.size(), 3) << "Input data should be 3-D.";

  std::vector<IndexExpr> oshape({data->shape[0]});
  std::vector<IndexExpr> oshape_indices({data->shape[0], data->shape[1]});
  std::vector<Type> fields;
  fields.push_back(TensorType(oshape, DataType::Int(32)));
  fields.push_back(TensorType(data->shape, data->dtype));
  fields.push_back(TensorType(oshape_indices, DataType::Int(32)));

  // assign output type
  reporter->Assign(types[2], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeGetValidCounts(Expr data, Expr score_threshold, int id_index, int score_index) {
  auto attrs = make_object<GetValidCountsAttrs>();
  attrs->id_index = id_index;
  attrs->score_index = score_index;
  static const Op& op = Op::Get("vision.get_valid_counts");
  return Call(op, {data, score_threshold}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.get_valid_counts").set_body_typed(MakeGetValidCounts);

RELAY_REGISTER_OP("vision.get_valid_counts")
    .describe(R"doc(Get valid count of bounding boxes given
a score threshold. Also moves valid boxes to the top of
input data.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("score_threshold", "Tensor", "Minimum Score.")
    .set_support_level(5)
    .add_type_rel("GetValidCount", GetValidCountRel);

TVM_REGISTER_NODE_TYPE(NonMaximumSuppressionAttrs);

bool NMSRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
            const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 6);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const auto* valid_count = types[1].as<TensorTypeNode>();
  if (valid_count == nullptr) return false;
  const NonMaximumSuppressionAttrs* param = attrs.as<NonMaximumSuppressionAttrs>();
  const auto& dshape = data->shape;
  const auto& vshape = valid_count->shape;
  ICHECK_EQ(dshape.size(), 3) << "Input data should be 3-D.";
  ICHECK_EQ(vshape.size(), 1) << "Input valid count should be 1-D.";

  // assign output type
  if (param->return_indices) {
    std::vector<Type> fields;
    // dynamic happens for return_indices in TensorFlow & ONNX
    std::vector<IndexExpr> oshape({dshape[0], dshape[1]});
    fields.push_back(TensorType(oshape, DataType::Int(32)));
    std::vector<IndexExpr> countshape({dshape[0], 1});
    fields.push_back(TensorType(countshape, DataType::Int(32)));
    reporter->Assign(types[5], TupleType(Array<Type>(fields)));
  } else {
    reporter->Assign(types[5], TensorType(dshape, data->dtype));
  }
  return true;
}

Expr MakeNMS(Expr data, Expr valid_count, Expr indices, Expr max_output_size, Expr iou_threshold,
             bool force_suppress, int top_k, int coord_start, int score_index, int id_index,
             bool return_indices, bool invalid_to_bottom) {
  auto attrs = make_object<NonMaximumSuppressionAttrs>();
  attrs->force_suppress = force_suppress;
  attrs->top_k = top_k;
  attrs->coord_start = coord_start;
  attrs->score_index = score_index;
  attrs->id_index = id_index;
  attrs->return_indices = return_indices;
  attrs->invalid_to_bottom = invalid_to_bottom;
  static const Op& op = Op::Get("vision.non_max_suppression");
  return Call(op, {data, valid_count, indices, max_output_size, iou_threshold}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.non_max_suppression").set_body_typed(MakeNMS);

RELAY_REGISTER_OP("vision.non_max_suppression")
    .describe(R"doc(Non-maximum suppression. The input boxes should
be in the format of [class_id, score, left, top, right, bottom]
or [score, left, top, right, bottom]. Set id_index to be -1 to
ignore class_id axis.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("valid_count", "Tensor", "Number of valid anchor boxes.")
    .add_argument("indices", "Tensor", "Corresponding indices in original input tensor.")
    .add_argument("max_output_size", "Tensor", "Max number of output valid boxes.")
    .add_argument("iou_threshold", "Tensor", "Threshold for box overlap.")
    .set_support_level(5)
    .add_type_rel("NMS", NMSRel);

TVM_REGISTER_NODE_TYPE(AllClassNonMaximumSuppressionAttrs);

bool AllClassNMSRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 6);
  const auto* boxes = types[0].as<TensorTypeNode>();
  if (boxes == nullptr) return false;
  const auto* scores = types[1].as<TensorTypeNode>();
  if (scores == nullptr) return false;

  const auto& boxes_shape = boxes->shape;
  const auto& scores_shape = scores->shape;
  ICHECK_EQ(boxes_shape.size(), 3) << "Input boxes should be 3-D.";
  ICHECK_EQ(scores_shape.size(), 3) << "Input scores count should be 3-D.";

  IndexExpr batch = boxes_shape[0];
  IndexExpr num_classes = scores_shape[1];
  IndexExpr num_boxes = boxes_shape[1];

  const auto* param = attrs.as<AllClassNonMaximumSuppressionAttrs>();
  CHECK(param);

  std::vector<Type> fields;
  if (param->output_format == "onnx") {
    IndexExpr num_total_boxes = Any();
    if (!batch.as<AnyNode>() && !num_boxes.as<AnyNode>()) {
      num_total_boxes = batch * num_classes * num_boxes;
    }
    std::vector<IndexExpr> oshape{num_total_boxes, 3};
    std::vector<IndexExpr> counts_shape{1};
    fields.push_back(TensorType(oshape, DataType::Int(64)));
    fields.push_back(TensorType(counts_shape, DataType::Int(64)));
  } else {
    IndexExpr num_total_boxes_per_batch = Any();
    if (!num_boxes.as<AnyNode>()) {
      num_total_boxes_per_batch = num_classes * num_boxes;
    }
    std::vector<IndexExpr> indices_shape{batch, num_total_boxes_per_batch, 2};
    std::vector<IndexExpr> scores_shape{batch, num_total_boxes_per_batch};
    std::vector<IndexExpr> counts_shape{batch};
    fields.push_back(TensorType(indices_shape, DataType::Int(64)));
    fields.push_back(TensorType(scores_shape, DataType::Float(32)));
    fields.push_back(TensorType(counts_shape, DataType::Int(64)));
  }
  reporter->Assign(types[5], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeAllClassNMS(Expr boxes, Expr scores, Expr max_output_boxes_per_class, Expr iou_threshold,
                     Expr score_threshold, std::string output_format = "onnx") {
  auto attrs = make_object<AllClassNonMaximumSuppressionAttrs>();
  attrs->output_format = std::move(output_format);
  static const Op& op = Op::Get("vision.all_class_non_max_suppression");
  return Call(op, {boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold},
              Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.all_class_non_max_suppression")
    .set_body_typed(MakeAllClassNMS);

RELAY_REGISTER_OP("vision.all_class_non_max_suppression")
    .describe(R"doc(Non-maximum suppression operator for object detection, corresponding to ONNX
    NonMaxSuppression and TensorFlow combined_non_max_suppression.
    NMS is performed for each class separately
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(5)
    .add_argument("boxes", "Tensor", "The input boxes in the format [batch, num_boxes, 4].")
    .add_argument("scores", "Tensor",
                  "Scores for each box and class in the format [batch, num_classes, num_boxes].")
    .add_argument("max_output_boxes_per_class", "Tensor",
                  "The maximum number of output boxes per class.")
    .add_argument("iou_threshold", "Tensor", "The IoU threshold for box the overlap test.")
    .add_argument("score_threshold", "Tensor",
                  "The score threshold to filter out low score boxes early.")
    .set_support_level(5)
    .add_type_rel("AllClassNMS", AllClassNMSRel);

TVM_REGISTER_NODE_TYPE(RegularNonMaximumSuppressionAttrs);

bool RegularNMSRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* boxes = types[0].as<TensorTypeNode>();
  if (boxes == nullptr) return false;
  const auto* scores = types[1].as<TensorTypeNode>();
  if (scores == nullptr) return false;

  const auto& boxes_shape = boxes->shape;
  const auto& scores_shape = scores->shape;
  ICHECK_EQ(boxes_shape.size(), 3) << "Input boxes should be 3-D.";
  ICHECK_EQ(scores_shape.size(), 3) << "Input scores count should be 3-D.";

  IndexExpr num_batches = boxes_shape[0];

  const auto* param = attrs.as<RegularNonMaximumSuppressionAttrs>();
  CHECK(param);

  std::vector<Type> fields;
  std::vector<IndexExpr> nmsed_boxes_shape{num_batches, param->max_detections, 4};
  std::vector<IndexExpr> nmsed_classes_shape{num_batches, param->max_detections};
  std::vector<IndexExpr> nmsed_scores_shape{num_batches, param->max_detections};
  std::vector<IndexExpr> nmsed_detections_number_shape{num_batches};
  fields.push_back(TensorType(nmsed_boxes_shape, DataType::Float(32)));
  fields.push_back(TensorType(nmsed_classes_shape, DataType::Float(32)));
  fields.push_back(TensorType(nmsed_scores_shape, DataType::Float(32)));
  fields.push_back(TensorType(nmsed_detections_number_shape, DataType::Int(32)));

  reporter->Assign(types[2], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeRegularNMS(Expr boxes, Expr scores, int32_t max_detections_per_class,
                    int32_t max_detections, int32_t num_classes, double iou_threshold,
                    double score_threshold) {
  auto attrs = make_object<RegularNonMaximumSuppressionAttrs>();
  attrs->max_detections_per_class = max_detections_per_class;
  attrs->max_detections = max_detections;
  attrs->num_classes = num_classes;
  attrs->iou_threshold = iou_threshold;
  attrs->score_threshold = score_threshold;
  static const Op& op = Op::Get("vision.regular_non_max_suppression");
  return Call(op, {boxes, scores}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.regular_non_max_suppression")
    .set_body_typed(MakeRegularNMS);

RELAY_REGISTER_OP("vision.regular_non_max_suppression")
    .describe(R"doc(TBD)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("boxes", "Tensor",
                  "3-D tensor with shape (batch_size, num_boxes, 4). The four values in boxes "
                  "encode (ymin, xmin, ymax, xmax) coordinates of a box.")
    .add_argument("scores", "Tensor",
                  "3-D tensor with shape (batch_size, num_boxes, num_classes_with_background).")
    .set_support_level(5)
    .add_type_rel("RegularNMS", RegularNMSRel);

}  // namespace relay
}  // namespace tvm
