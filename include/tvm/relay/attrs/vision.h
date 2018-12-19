/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/vision.h
 * \brief Auxiliary attributes for vision operators.
 */
#ifndef TVM_RELAY_ATTRS_VISION_H_
#define TVM_RELAY_ATTRS_VISION_H_

#include <tvm/attrs.h>
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

/*! \brief Attributes used in non_maximum_suppression operators */
struct NMSAttrs : public tvm::AttrsNode<NMSAttrs>{
  double overlap_threshold;
  bool force_suppress;
  int topk;

  TVM_DECLARE_ATTRS(NMSAttrs, "relay.attrs.NMSAttrs") {
      TVM_ATTR_FIELD(overlap_threshold).set_default(0.5)
        .describe("Non-maximum suppression threshold.");
      TVM_ATTR_FIELD(force_suppress).set_default(false)
        .describe("Suppress all detections regardless of class_id.");
      TVM_ATTR_FIELD(topk).set_default(-1)
        .describe("Keep maximum top k detections before nms, -1 for no limit.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_VISION_H_
