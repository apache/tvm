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
 * \file rcnn_op.cc
 * \brief Faster RCNN and Mask RCNN operators
 */
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "../../transforms/infer_layout_utils.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ROIAlignAttrs);

bool ROIAlignRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  auto roi_align_attrs = attrs.as<ROIAlignAttrs>();
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* rois = types[1].as<TensorTypeNode>();
  ICHECK(data);
  ICHECK(rois);
  const auto& dshape = data->shape;
  const auto& rshape = rois->shape;
  ICHECK(roi_align_attrs);
  ICHECK_EQ(dshape.size(), 4) << "Input data should be 4-D.";
  ICHECK_EQ(rshape.size(), 2) << "Input rois should be 2-D.";
  // assign output type
  std::vector<IndexExpr> oshape;
  if (roi_align_attrs->layout == "NCHW") {
    oshape = {rshape[0], dshape[1], roi_align_attrs->pooled_size[0],
              roi_align_attrs->pooled_size[1]};
  } else {
    ICHECK_EQ(roi_align_attrs->layout, "NHWC") << "Unexpected ROI Align layout";
    oshape = {rshape[0], roi_align_attrs->pooled_size[0], roi_align_attrs->pooled_size[1],
              dshape[3]};
  }

  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

template <typename T>
InferCorrectLayoutOutput ROIAlignInferCorrectLayout(const Attrs& attrs,
                                                    const Array<Layout>& new_in_layouts,
                                                    const Array<Layout>& old_in_layouts,
                                                    const Array<tvm::relay::Type>& old_in_types) {
  const T* params = attrs.as<T>();
  Layout data_layout = params->layout;

  // Layout inference needs to define the layout for all inputs and output data layouts.
  // For roi_align, the second inputs is 2-D tensor with shape [num_roi, 5].
  // So, we set the layout as "N5".
  return InferCorrectLayoutOutput({data_layout, Layout("N5")}, {data_layout}, attrs);
}

Expr MakeROIAlign(Expr data, Expr rois, Array<IndexExpr> pooled_size, double spatial_scale,
                  int sample_ratio, String layout, String mode) {
  auto attrs = make_object<ROIAlignAttrs>();
  attrs->pooled_size = pooled_size;
  attrs->spatial_scale = spatial_scale;
  attrs->sample_ratio = sample_ratio;
  attrs->layout = layout;
  attrs->mode = mode;
  static const Op& op = Op::Get("vision.roi_align");
  return Call(op, {data, rois}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.roi_align").set_body_typed(MakeROIAlign);

RELAY_REGISTER_OP("vision.roi_align")
    .describe(R"doc(ROI Align operator.

 - **data**: This depends on the `layout` parameter. Input is 4D array of shape
             (batch_size, channels, height, width) if `layout` is `NCHW`.
 - **rois**: 2D array of shape (num_roi, 5). The last dimension should be in format of
             [batch_index, w_start, h_start, w_end, h_end].
 - **out**: This depends on the `layout` parameter. Output is 4D array of shape
            (num_roi, channels, pooled_height, pooled_width) if `layout` is `NCHW`.
 )doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("rois", "Tensor", "The input rois")
    .set_support_level(5)
    .add_type_rel("ROIAlign", ROIAlignRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                                   ROIAlignInferCorrectLayout<ROIAlignAttrs>);

TVM_REGISTER_NODE_TYPE(ROIPoolAttrs);

bool ROIPoolRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  auto roi_pool_attrs = attrs.as<ROIPoolAttrs>();
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* rois = types[1].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  const auto& rshape = rois->shape;
  ICHECK(roi_pool_attrs);
  ICHECK_EQ(dshape.size(), 4) << "Input data should be 4-D.";
  ICHECK_EQ(rshape.size(), 2) << "Input rois should be 2-D.";
  // assign output type
  std::vector<IndexExpr> oshape;
  if (roi_pool_attrs->layout == "NCHW") {
    oshape = {rshape[0], dshape[1], roi_pool_attrs->pooled_size[0], roi_pool_attrs->pooled_size[1]};
  } else if (roi_pool_attrs->layout == "NHWC") {
    oshape = {rshape[0], roi_pool_attrs->pooled_size[0], roi_pool_attrs->pooled_size[1], dshape[3]};
  } else {
    LOG(FATAL) << "vision.roi_pool does not support " << roi_pool_attrs->layout << " layout";
  }

  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

template <typename T>
InferCorrectLayoutOutput ROIPoolInferCorrectLayout(const Attrs& attrs,
                                                   const Array<Layout>& new_in_layouts,
                                                   const Array<Layout>& old_in_layouts,
                                                   const Array<tvm::relay::Type>& old_in_types) {
  const T* params = attrs.as<T>();
  Layout data_layout = params->layout;

  // Layout inference needs to define the layout for all inputs and output data layouts.
  // For roi_pool, the second inputs is 2-D tensor with shape [num_roi, 5].
  // So, we set the layout as "N5".
  return InferCorrectLayoutOutput({data_layout, Layout("N5")}, {data_layout}, attrs);
}

Expr MakeROIPool(Expr data, Expr rois, Array<IndexExpr> pooled_size, double spatial_scale,
                 String layout) {
  auto attrs = make_object<ROIPoolAttrs>();
  attrs->pooled_size = pooled_size;
  attrs->spatial_scale = spatial_scale;
  attrs->layout = layout;
  static const Op& op = Op::Get("vision.roi_pool");
  return Call(op, {data, rois}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.roi_pool").set_body_typed(MakeROIPool);

RELAY_REGISTER_OP("vision.roi_pool")
    .describe(R"doc(ROI Pool operator.

 - **data**: This depends on the `layout` parameter. Input is 4D array of shape
             (batch_size, channels, height, width) if `layout` is `NCHW`.
 - **rois**: 2D array of shape (num_roi, 5). The last dimension should be in format of
             [batch_index, w_start, h_start, w_end, h_end].
 - **out**: This depends on the `layout` parameter. Output is 4D array of shape
            (num_roi, channels, pooled_height, pooled_width) if `layout` is `NCHW`.
 )doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("rois", "Tensor", "The input rois")
    .set_support_level(5)
    .add_type_rel("ROIPool", ROIPoolRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ROIPoolInferCorrectLayout<ROIPoolAttrs>);

TVM_REGISTER_NODE_TYPE(ProposalAttrs);

bool ProposalRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  auto proposal_attrs = attrs.as<ProposalAttrs>();
  ICHECK_EQ(types.size(), 4);
  const auto* cls_prob = types[0].as<TensorTypeNode>();
  const auto* bbox_pred = types[1].as<TensorTypeNode>();
  const auto* im_info = types[2].as<TensorTypeNode>();

  if (!cls_prob || !bbox_pred || !im_info) {
    return false;
  }

  ICHECK_EQ(cls_prob->shape.size(), 4U)
      << "The dimension of class probability should be 4, but received " << cls_prob->shape.size();
  ICHECK_EQ(bbox_pred->shape.size(), 4U)
      << "The dimension of box prediction should be 4, but received " << bbox_pred->shape.size();
  ICHECK_EQ(im_info->shape.size(), 2U)
      << "The dimension of image info should be 2, but received " << im_info->shape.size();
  ICHECK(reporter->AssertEQ(im_info->shape[1], 3));

  auto batch = cls_prob->shape[0];

  std::vector<IndexExpr> oshape({batch * proposal_attrs->rpn_post_nms_top_n, 5});
  reporter->Assign(types[3], TensorType(oshape, cls_prob->dtype));
  return true;
}

Expr MakeProposal(Expr cls_prob, Expr bbox_pred, Expr im_info, Array<IndexExpr> scales,
                  Array<IndexExpr> ratios, int feature_stride, double threshold,
                  int rpn_pre_nms_top_n, int rpn_post_nms_top_n, int rpn_min_size, bool iou_loss) {
  auto attrs = make_object<ProposalAttrs>();
  attrs->scales = scales;
  attrs->ratios = ratios;
  attrs->feature_stride = feature_stride;
  attrs->threshold = threshold;
  attrs->rpn_pre_nms_top_n = rpn_pre_nms_top_n;
  attrs->rpn_post_nms_top_n = rpn_post_nms_top_n;
  attrs->rpn_min_size = rpn_min_size;
  attrs->iou_loss = iou_loss;
  static const Op& op = Op::Get("vision.proposal");
  return Call(op, {cls_prob, bbox_pred, im_info}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.proposal").set_body_typed(MakeProposal);

RELAY_REGISTER_OP("vision.proposal")
    .describe(R"code(Generate region proposals via RPN.

 - **cls_prob**: 4-D with shape [batch, 2 * num_anchors, height, width].
 - **bbox_pred**: 4-D with shape [batch, 4 * num_anchors, height, width].
 - **im_info**: 2-D with shape [batch, 3].
 - **out**: 2-D with shape [batch * rpn_post_nms_top_n, 5].
 )code" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .add_argument("cls_prob", "Tensor", "Score of how likely proposal is object")
    .add_argument("bbox_pred", "Tensor", "BBox predicted deltas from anchors for proposals")
    .add_argument("im_info", "Tensor", "Image size and scale")
    .set_support_level(5)
    .add_type_rel("Proposal", ProposalRel);

}  // namespace relay
}  // namespace tvm
