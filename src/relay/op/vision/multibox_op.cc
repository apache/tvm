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
 * \file multibox_op.cc
 * \brief Multibox related operators
 */
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/op.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(MultiBoxPriorAttrs);

bool MultiboxPriorRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const MultiBoxPriorAttrs* param = attrs.as<MultiBoxPriorAttrs>();
  const auto& dshape = data->shape;
  ICHECK_EQ(dshape.size(), 4) << "Input data should be 4D: "
                                 "[batch, channel, height, width]";
  IndexExpr in_height = dshape[2];
  IndexExpr in_width = dshape[3];
  int num_sizes = static_cast<int>(param->sizes.size());
  int num_ratios = static_cast<int>(param->ratios.size());

  // since input sizes are same in each batch, we could share MultiBoxPrior
  std::vector<IndexExpr> oshape({1, in_height * in_width * (num_sizes + num_ratios - 1), 4});

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeMultiBoxPrior(Expr data, Array<IndexExpr> sizes, Array<IndexExpr> ratios,
                       Array<IndexExpr> steps, Array<IndexExpr> offsets, bool clip) {
  auto attrs = make_object<MultiBoxPriorAttrs>();
  attrs->sizes = std::move(sizes);
  attrs->ratios = std::move(ratios);
  attrs->steps = std::move(steps);
  attrs->offsets = std::move(offsets);
  attrs->clip = clip;
  static const Op& op = Op::Get("vision.multibox_prior");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.multibox_prior").set_body_typed(MakeMultiBoxPrior);

RELAY_REGISTER_OP("vision.multibox_prior")
    .describe(R"doc("Generate prior(anchor) boxes from data, sizes and ratios."
)doc" TVM_ADD_FILELINE)
    .set_attrs_type<MultiBoxPriorAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(5)
    .add_type_rel("MultiBoxPrior", MultiboxPriorRel);

TVM_REGISTER_NODE_TYPE(MultiBoxTransformLocAttrs);

bool MultiBoxTransformLocRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                             const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);

  const auto* cls_prob = types[0].as<TensorTypeNode>();
  const auto* loc_pred = types[1].as<TensorTypeNode>();
  const auto* anchor = types[2].as<TensorTypeNode>();

  if (cls_prob == nullptr || loc_pred == nullptr || anchor == nullptr) {
    return false;
  }

  const auto& cls_shape = cls_prob->shape;
  const auto& loc_shape = loc_pred->shape;
  const auto& anchor_shape = anchor->shape;

  ICHECK_EQ(cls_shape.size(), 3U) << "The dimension of class probability should be 3, but received "
                                  << cls_shape.size();
  ICHECK_EQ(loc_shape.size(), 2U)
      << "The dimension of location prediction should be 2, but received " << loc_shape.size();
  ICHECK_EQ(anchor_shape.size(), 3U)
      << "The dimension of anchor should be 3, but received " << anchor_shape.size();

  ICHECK(reporter->AssertEQ(cls_shape[2], anchor_shape[1])) << "Number of anchors mismatch found";
  ICHECK(reporter->AssertEQ(cls_shape[2] * 4, loc_shape[1])) << "# anchors mismatch with # loc.";
  ICHECK(reporter->Assert(anchor_shape[1] > 0)) << "Number of anchors must > 0.";
  ICHECK(reporter->AssertEQ(anchor_shape[2], 4));

  std::vector<IndexExpr> oshape0({cls_shape[0], anchor_shape[1], 6});
  std::vector<IndexExpr> oshape1({cls_shape[0]});
  std::vector<Type> fields;
  fields.push_back(TensorType(oshape0, cls_prob->dtype));
  fields.push_back(TensorType(oshape1, DataType::Int(32)));

  // assign output type
  reporter->Assign(types[3], TupleType(Array<Type>(fields)));
  return true;
}

Expr MakeMultiBoxTransformLoc(Expr cls_prob, Expr loc_pred, Expr anchor, bool clip,
                              double threshold, Array<IndexExpr> variances, bool keep_background) {
  auto attrs = make_object<MultiBoxTransformLocAttrs>();
  attrs->clip = std::move(clip);
  attrs->threshold = std::move(threshold);
  attrs->variances = std::move(variances);
  attrs->keep_background = std::move(keep_background);
  static const Op& op = Op::Get("vision.multibox_transform_loc");
  return Call(op, {cls_prob, loc_pred, anchor}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.multibox_transform_loc")
    .set_body_typed(MakeMultiBoxTransformLoc);

RELAY_REGISTER_OP("vision.multibox_transform_loc")
    .describe(R"doc("Location transformation for multibox detection."
)doc" TVM_ADD_FILELINE)
    .set_attrs_type<MultiBoxTransformLocAttrs>()
    .set_num_inputs(3)
    .add_argument("cls_prob", "Tensor", "Class probabilities.")
    .add_argument("loc_pred", "Tensor", "Location regression predictions.")
    .add_argument("anchor", "Tensor", "Multibox prior anchor boxes")
    .add_type_rel("MultiBoxTransformLoc", MultiBoxTransformLocRel)
    .set_support_level(5);

}  // namespace relay
}  // namespace tvm
