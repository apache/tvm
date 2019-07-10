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
 * \file tvm/relay/attrs/nn.h
 * \brief Auxiliary attributes for nn operators.
 */
#ifndef TVM_RELAY_ATTRS_QNN_H_
#define TVM_RELAY_ATTRS_QNN_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  double input_scale;
  int32_t input_zero_point;
  double output_scale;
  int32_t output_zero_point;
  bool use_int_compute;
  std::string rounding_mode;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.RequantizeAttrs") {
    TVM_ATTR_FIELD(input_zero_point)
        .describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_zero_point)
        .describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(input_scale)
        .describe("The scale of the input tensor.");
    TVM_ATTR_FIELD(output_scale)
        .describe("The scale of the output tensor.");
    TVM_ATTR_FIELD(use_int_compute).set_default(true)
      .describe("When true, the integer computation is used to handle output scale."
                "The float compuation can be used as reference implementation or in"
                "cases where FP32 computation for requantize is not expensive");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
    TVM_ATTR_FIELD(rounding_mode).set_default("FE_UPWARD")
        .describe("Defines the rounding direction when the value is midway between"
                  "two representable values. There are two supported modes - FE_UPWARD"
                  "or FE_AWAY_FROM_ZERO. More context can be found at"
                  "https://www.gnu.org/software/libc/manual/html_node/Rounding.html");
  }
};


}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_QNN_H_
