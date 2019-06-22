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
 * \file tvm/relay/attrs/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAY_ATTRS_VISION_H_
#define TVM_RELAY_ATTRS_VISION_H_

#include <tvm/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in multibox_prior operators */
struct MultiBoxPriorAttrs : public tvm::AttrsNode<MultiBoxPriorAttrs> {
  Array<IndexExpr> sizes;
  Array<IndexExpr> ratios;
  Array<IndexExpr> steps;
  Array<IndexExpr> offsets;
  bool clip;

  TVM_DECLARE_ATTRS(MultiBoxPriorAttrs, "relay.attrs.MultiBoxPriorAttrs") {
    TVM_ATTR_FIELD(sizes)
      .set_default(Array<IndexExpr>({static_cast<float>(1.0)}))
      .describe("List of sizes of generated MultiBoxPriores.");
    TVM_ATTR_FIELD(ratios)
      .set_default(Array<IndexExpr>({static_cast<float>(1.0)}))
      .describe("List of aspect ratios of generated MultiBoxPriores.");
    TVM_ATTR_FIELD(steps)
      .set_default(Array<IndexExpr>({static_cast<float>(-1.0),
                                     static_cast<float>(-1.0)}))
      .describe("Priorbox step across y and x, -1 for auto calculation.");
    TVM_ATTR_FIELD(offsets)
      .set_default(Array<IndexExpr>({static_cast<float>(0.5),
                                     static_cast<float>(0.5)}))
      .describe("Priorbox center offsets, y and x respectively.");
    TVM_ATTR_FIELD(clip).set_default(false)
      .describe("Whether to clip out-of-boundary boxes.");
  }
};

struct MultiBoxTransformLocAttrs
    : public tvm::AttrsNode<MultiBoxTransformLocAttrs> {
  bool clip;
  double threshold;
  Array<IndexExpr> variances;

  TVM_DECLARE_ATTRS(MultiBoxTransformLocAttrs,
                    "relay.attrs.MultiBoxTransformLocAttrs") {
    TVM_ATTR_FIELD(clip).set_default(true)
      .describe("Clip out-of-boundary boxes.");
    TVM_ATTR_FIELD(threshold).set_default(0.01)
      .describe("Threshold to be a positive prediction.");
    TVM_ATTR_FIELD(variances)
      .set_default(Array<IndexExpr>({0.1f, 0.1f , 0.2f, 0.2f}))
      .describe("Variances to be decoded from box regression output.");
  }
};

/*! \brief Attributes used in get_valid_counts operator */
struct GetValidCountsAttrs : public tvm::AttrsNode<GetValidCountsAttrs> {
  double score_threshold;
  int id_index;
  int score_index;

  TVM_DECLARE_ATTRS(GetValidCountsAttrs, "relay.attrs.GetValidCountsAttrs") {
    TVM_ATTR_FIELD(score_threshold).set_default(0.0)
      .describe("Lower limit of score for valid bounding boxes.");
    TVM_ATTR_FIELD(id_index).set_default(0)
      .describe("Axis index of id.");
    TVM_ATTR_FIELD(score_index).set_default(1)
      .describe("Index of the scores/confidence of boxes.");
  }
};

/*! \brief Attributes used in non_maximum_suppression operator */
struct NonMaximumSuppressionAttrs : public tvm::AttrsNode<NonMaximumSuppressionAttrs> {
  int max_output_size;
  double iou_threshold;
  bool force_suppress;
  int top_k;
  int coord_start;
  int score_index;
  int id_index;
  bool return_indices;
  bool invalid_to_bottom;

  TVM_DECLARE_ATTRS(NonMaximumSuppressionAttrs, "relay.attrs.NonMaximumSuppressionAttrs") {
    TVM_ATTR_FIELD(max_output_size).set_default(-1)
      .describe("Max number of output valid boxes for each instance."
                "By default all valid boxes are returned.");
    TVM_ATTR_FIELD(iou_threshold).set_default(0.5)
      .describe("Non-maximum suppression threshold.");
    TVM_ATTR_FIELD(force_suppress).set_default(false)
      .describe("Suppress all detections regardless of class_id.");
    TVM_ATTR_FIELD(top_k).set_default(-1)
      .describe("Keep maximum top k detections before nms, -1 for no limit.");
    TVM_ATTR_FIELD(coord_start).set_default(2)
      .describe("Start index of the consecutive 4 coordinates.");
    TVM_ATTR_FIELD(score_index).set_default(1)
      .describe("Index of the scores/confidence of boxes.");
    TVM_ATTR_FIELD(id_index).set_default(0)
      .describe("Axis index of id.");
    TVM_ATTR_FIELD(return_indices).set_default(true)
      .describe("Whether to return box indices in input data.");
    TVM_ATTR_FIELD(invalid_to_bottom).set_default(false)
      .describe("Whether to move all invalid bounding boxes to the bottom.");
  }
};

/*! \brief Attributes used in roi_align operators */
struct ROIAlignAttrs : public tvm::AttrsNode<ROIAlignAttrs> {
  Array<IndexExpr> pooled_size;
  double spatial_scale;
  int sample_ratio;
  std::string layout;
  TVM_DECLARE_ATTRS(ROIAlignAttrs, "relay.attrs.ROIAlignAttrs") {
    TVM_ATTR_FIELD(pooled_size).describe("Output size of roi align.");
    TVM_ATTR_FIELD(spatial_scale)
        .describe(
            "Ratio of input feature map height (or w) to raw image height (or w). "
            "Equals the reciprocal of total stride in convolutional layers, which should be "
            "in range (0.0, 1.0]");
    TVM_ATTR_FIELD(sample_ratio)
        .set_default(-1)
        .describe("Optional sampling ratio of ROI align, using adaptive size by default.");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
  }
};

/*! \brief Attributes used in roi_pool operators */
struct ROIPoolAttrs : public tvm::AttrsNode<ROIPoolAttrs> {
  Array<IndexExpr> pooled_size;
  double spatial_scale;
  std::string layout;
  TVM_DECLARE_ATTRS(ROIPoolAttrs, "relay.attrs.ROIPoolAttrs") {
    TVM_ATTR_FIELD(pooled_size).describe("Output size of roi pool.");
    TVM_ATTR_FIELD(spatial_scale)
        .describe(
            "Ratio of input feature map height (or w) to raw image height (or w). "
            "Equals the reciprocal of total stride in convolutional layers, which should be "
            "in range (0.0, 1.0]");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Convolution is applied on the 'H' and"
        "'W' dimensions.");
  }
};

/*! \brief Attributes used in yolo reorg operators */
struct YoloReorgAttrs : public tvm::AttrsNode<YoloReorgAttrs> {
  Integer stride;

  TVM_DECLARE_ATTRS(YoloReorgAttrs, "relay.attrs.YoloReorgAttrs") {
    TVM_ATTR_FIELD(stride)
      .set_default(1)
      .describe("Stride value for yolo reorg");
  }
};

/*! \brief Attributes used in proposal operators */
struct ProposalAttrs : public tvm::AttrsNode<ProposalAttrs> {
  Array<IndexExpr> scales;
  Array<IndexExpr> ratios;
  int feature_stride;
  double threshold;
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  int rpn_min_size;
  bool iou_loss;

  TVM_DECLARE_ATTRS(ProposalAttrs, "relay.attrs.ProposalAttrs") {
    TVM_ATTR_FIELD(scales)
        .set_default(Array<IndexExpr>({4.0f, 8.0f, 16.0f, 32.0f}))
        .describe("Used to generate anchor windows by enumerating scales");
    TVM_ATTR_FIELD(ratios)
        .set_default(Array<IndexExpr>({0.5f, 1.0f, 2.0f}))
        .describe("Used to generate anchor windows by enumerating ratios");
    TVM_ATTR_FIELD(feature_stride)
        .set_default(16)
        .describe(
            "The size of the receptive field each unit in the convolution layer of the rpn,"
            "for example the product of all stride's prior to this layer.");
    TVM_ATTR_FIELD(threshold)
        .set_default(0.7)
        .describe(
            "IoU threshold of non-maximum suppresion (suppress boxes with IoU >= this threshold)");
    TVM_ATTR_FIELD(rpn_pre_nms_top_n)
        .set_default(6000)
        .describe("Number of top scoring boxes to apply NMS. -1 to use all boxes");
    TVM_ATTR_FIELD(rpn_post_nms_top_n)
        .set_default(300)
        .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    TVM_ATTR_FIELD(rpn_min_size).set_default(16).describe("Minimum height or width in proposal");
    TVM_ATTR_FIELD(iou_loss).set_default(false).describe("Usage of IoU Loss");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_VISION_H_
