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
 * \file src/relay/op/contrib/ethosu/common.cc
 * \brief A set of utilities and common functionality for Arm(R) Ethos(TM)-U NPU QNN ops.
 */

#include "common.h"

#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

Array<IndexExpr> EthosuInferKernelOutput(Array<IndexExpr> ifm_shape, String ifm_layout,
                                         String ofm_layout, Array<IndexExpr> kernel_shape,
                                         IndexExpr ofm_channels, Array<IndexExpr> dilation,
                                         Array<IndexExpr> strides, Array<IndexExpr> padding) {
  // In the case of NHCWB16, convert the ifm shape to NHW (C not required for this function)
  if (ifm_layout == "NHCWB16") {
    ifm_shape = {ifm_shape[0], ifm_shape[1], ifm_shape[3]};
  }
  Array<IndexExpr> output_shape({ifm_shape[0], 0, 0, ofm_channels});

  IndexExpr dilated_ksize_y = 1 + (kernel_shape[0] - 1) * dilation[0];
  IndexExpr dilated_ksize_x = 1 + (kernel_shape[1] - 1) * dilation[1];
  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(padding, &pad_h, &pad_w);
  output_shape.Set(1, indexdiv(ifm_shape[1] + pad_h - dilated_ksize_y, strides[0]) + 1);
  output_shape.Set(2, indexdiv(ifm_shape[2] + pad_w - dilated_ksize_x, strides[1]) + 1);

  // If the ofm is NHCWB16, convert the layout
  if (ofm_layout == "NHCWB16") {
    int channel_bricks = 1 + (output_shape[3].as<IntImmNode>()->value - 1) / 16;
    output_shape = {output_shape[0], output_shape[1], channel_bricks, output_shape[2], 16};
  }

  return output_shape;
}

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
