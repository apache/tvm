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

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_VISION_H_
