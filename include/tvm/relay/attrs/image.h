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
 * \file tvm/relay/attrs/image.h
 * \brief Auxiliary attributes for image operators.
 */
#ifndef TVM_RELAY_ATTRS_IMAGE_H_
#define TVM_RELAY_ATTRS_IMAGE_H_

#include <tvm/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in image resize operator */
struct ResizeAttrs : public tvm::AttrsNode<ResizeAttrs> {
  Array<IndexExpr> size;
  std::string layout;
  std::string method;
  bool align_corners;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(ResizeAttrs, "relay.attrs.ResizeAttrs") {
    TVM_ATTR_FIELD(size).set_default(NullValue<Array<IndexExpr> >())
        .describe("Output Size.");
    TVM_ATTR_FIELD(layout).set_default("NCHW")
        .describe("Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                  "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                  "dimensions respectively. Resize is applied on the 'H' and"
                  "'W' dimensions.");
    TVM_ATTR_FIELD(method).set_default("bilinear")
        .describe("Specify the mode to use for scaling."
                  "nearest_neighbor -  Nearest Neighbor"
                  "bilinear - Bilinear Interpolation"
                  "bicubic - Bicubic Interpolation");
    TVM_ATTR_FIELD(align_corners).set_default(true)
        .describe("Should be true to preserve the values at the corner pixels");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_IMAGE_H_
