/*!
 *  Copyright (c) 2018 by Contributors
 * \file resize.h
 */
#ifndef NNVM_TOP_VISION_RESIZE_H_
#define NNVM_TOP_VISION_RESIZE_H_

#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <sstream>

namespace nnvm {
namespace top {

struct ResizeParam : public dmlc::Parameter<ResizeParam> {
  TShape out_size;
  std::string layout;
  std::string mode;
  bool align_corners;

  DMLC_DECLARE_PARAMETER(ResizeParam) {
    DMLC_DECLARE_FIELD(out_size)
      .describe("Scale output size."
                "actual out size if is_factor false"
                "or a multiply factor if is_factor is true");
    DMLC_DECLARE_FIELD(layout)
      .set_default("NCHW")
      .describe("Dimension ordering of data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Resize is applied on the 'H' and"
                "'W' dimensions.");
    DMLC_DECLARE_FIELD(mode)
      .set_default("NN")
      .describe("Specify the mode to use for scaling."
                "NN -  Nearest Neighbour"
                "BILINEAR - Bilinear Interpolation");
    DMLC_DECLARE_FIELD(align_corners)
      .set_default(false)
      .describe("Should be true to preserve the values at the corner pixels");
  }
};

}  // namespace top
}  // namespace nnvm
#endif  // NNVM_TOP_VISION_RESIZE_H_
