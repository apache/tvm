/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/attrs/image.h
 * \brief Auxiliary attributes for image operators.
 */
#ifndef TVM_RELAY_ATTRS_IMAGE_H_
#define TVM_RELAY_ATTRS_IMAGE_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in image resize operator */
struct ResizeAttrs : public tvm::AttrsNode<ResizeAttrs> {
  Array<IndexExpr> size;
  std::string layout;
  std::string method;
  bool align_corners;

  TVM_DECLARE_ATTRS(ResizeAttrs, "relay.attrs.ResizeAttrs") {
    TVM_ATTR_FIELD(size).set_default(NullValue<Array<IndexExpr> >())
        .describe("Output Size.");
    TVM_ATTR_FIELD(layout).set_default("NCHW")
        .describe("Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                  "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                  "dimensions respectively. Resize is applied on the 'H' and"
                  "'W' dimensions.");
    TVM_ATTR_FIELD(method).set_default("BILINEAR")
        .describe("Specify the mode to use for scaling."
                  "NEAREST_NEIGHBOR -  Nearest Neighbor"
                  "BILINEAR - Bilinear Interpolation");
    TVM_ATTR_FIELD(align_corners).set_default(false)
        .describe("Should be true to preserve the values at the corner pixels");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_IMAGE_H_
