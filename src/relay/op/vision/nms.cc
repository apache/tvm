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
 *  Copyright (c) 2018 by Contributors
 * \file nms.cc
 * \brief Non-maximum suppression operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/vision.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(GetValidCountsAttrs);

bool GetValidCountRel(const Array<Type>& types,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  CHECK_EQ(dshape.size(), 3) << "Input data should be 3-D.";

  std::vector<IndexExpr> oshape({data->shape[0]});
  std::vector<Type> fields;
  fields.push_back(TensorTypeNode::make(oshape, Int(32)));
  fields.push_back(TensorTypeNode::make(data->shape, data->dtype));

  // assign output type
  reporter->Assign(types[1], TupleTypeNode::make(Array<Type>(fields)));
  return true;
}

Expr MakeGetValidCounts(Expr data,
                        double score_threshold,
                        int id_index,
                        int score_index) {
  auto attrs = make_node<GetValidCountsAttrs>();
  attrs->score_threshold = score_threshold;
  attrs->id_index = id_index;
  attrs->score_index = score_index;
  static const Op& op = Op::Get("vision.get_valid_counts");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.vision._make.get_valid_counts")
.set_body_typed(MakeGetValidCounts);


RELAY_REGISTER_OP("vision.get_valid_counts")
.describe(R"doc(Get valid count of bounding boxes given
a score threshold. Also moves valid boxes to the top of
input data.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "Input data.")
.set_support_level(5)
.add_type_rel("GetValidCount", GetValidCountRel);


TVM_REGISTER_NODE_TYPE(NonMaximumSuppressionAttrs);

bool NMSRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* valid_count = types[1].as<TensorTypeNode>();
  const NonMaximumSuppressionAttrs* param =
    attrs.as<NonMaximumSuppressionAttrs>();
  const auto& dshape = data->shape;
  const auto& vshape = valid_count->shape;
  CHECK_EQ(dshape.size(), 3) << "Input data should be 3-D.";
  CHECK_EQ(vshape.size(), 1) << "Input valid count should be 1-D.";

  // assign output type
  if (param->return_indices) {
    std::vector<IndexExpr> oshape({dshape[0], dshape[1]});
    reporter->Assign(types[2], TensorTypeNode::make(oshape, Int(32)));
  } else {
    reporter->Assign(types[2], TensorTypeNode::make(dshape, data->dtype));
  }
  return true;
}


Expr MakeNMS(Expr data,
             Expr valid_count,
             int max_output_size,
             double iou_threshold,
             bool force_suppress,
             int top_k,
             int coord_start,
             int score_index,
             int id_index,
             bool return_indices,
             bool invalid_to_bottom) {
  auto attrs = make_node<NonMaximumSuppressionAttrs>();
  attrs->max_output_size = max_output_size;
  attrs->iou_threshold = iou_threshold;
  attrs->force_suppress = force_suppress;
  attrs->top_k = top_k;
  attrs->coord_start = coord_start;
  attrs->score_index = score_index;
  attrs->id_index = id_index;
  attrs->return_indices = return_indices;
  attrs->invalid_to_bottom = invalid_to_bottom;
  static const Op& op = Op::Get("vision.non_max_suppression");
  return CallNode::make(op, {data, valid_count}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.vision._make.non_max_suppression")
.set_body_typed(MakeNMS);


RELAY_REGISTER_OP("vision.non_max_suppression")
.describe(R"doc(Non-maximum suppression. The input boxes should
be in the format of [class_id, score, left, top, right, bottom].
Set id_index to be -1 to ignore class_id axis.
)doc" TVM_ADD_FILELINE)
.set_num_inputs(2)
.add_argument("data", "Tensor", "Input data.")
.add_argument("valid_count", "Tensor", "Number of valid anchor boxes.")
.set_support_level(5)
.add_type_rel("NMS", NMSRel);

}  // namespace relay
}  // namespace tvm
