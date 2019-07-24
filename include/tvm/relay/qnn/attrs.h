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
 * \file tvm/relay/qnn/attrs.h
 * \brief Auxiliary attributes for qnn operators.
 */

#ifndef TVM_RELAY_QNN_ATTRS_H_
#define TVM_RELAY_QNN_ATTRS_H_

#include <tvm/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  double input_scale;
  int32_t input_zero_point;
  double output_scale;
  int32_t output_zero_point;
  std::string rounding;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.RequantizeAttrs") {
    TVM_ATTR_FIELD(input_scale)
        .describe("The scale of the input tensor.");
    TVM_ATTR_FIELD(input_zero_point)
        .describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(output_scale)
        .describe("The scale of the output tensor.");
    TVM_ATTR_FIELD(output_zero_point)
        .describe("The zero point of the output tensor.");
    TVM_ATTR_FIELD(rounding).set_default("AWAY_FROM_ZERO")
        .describe("Defines the rounding direction when the value is midway between"
                  "two representable values. There are two supported modes - UPWARD"
                  "or AWAY_FROM_ZERO. Both modes behave exactly same except at the"
                  "midpoints between the two representable values. At the midpoint,"
                  "UPWARD rounds towards positive infinity (for example -1.5 will be"
                  "rounded to -1). AWAY_FROM_ZERO is the standard rounding where the"
                  "value is rounded away from zero at midpoints (for example, -1.5"
                  "rounds to -2). More context can be found at following gblic manual"
                  "https://www.gnu.org/software/libc/manual/html_node/Rounding.html."
                  "FE_UPWARD corresponds to UPWARD here and FE_TONEAREST corresponds"
                  "to AWAY_FROM_ZERO rounding mode.");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes for quantized dense operator */
struct QDenseAttrs : public tvm::AttrsNode<QDenseAttrs> {
  IndexExpr units;
  DataType out_dtype;
  // Quantization related attributes.
  int32_t input_zero_point;
  int32_t kernel_zero_point;

  TVM_DECLARE_ATTRS(QDenseAttrs, "relay.attrs.QDenseAttrs") {
    TVM_ATTR_FIELD(units)
        .describe("Number of hidden units of the dense transformation.");

    TVM_ATTR_FIELD(out_dtype)
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(input_zero_point)
        .describe("The zero point of the input tensor.");
    TVM_ATTR_FIELD(kernel_zero_point)
        .describe("The zero point of the kernel tensor.");
  }
};

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_QNN_H_
