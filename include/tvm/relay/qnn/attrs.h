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

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  int axis;
  std::string rounding;
  std::string compute_dtype;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.RequantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rounding).set_default("None").describe(
        "Defines the rounding direction when the value is midway between"
        "two representable values. There are two supported modes - UPWARD"
        "or TONEAREST. Both modes behave exactly same except at the"
        "midpoints between the two representable values. At the midpoint,"
        "UPWARD rounds towards positive infinity (for example -1.5 will be"
        "rounded to -1). TONEAREST is the standard rounding where the"
        "value is rounded away from zero at midpoints (for example, -1.5"
        "rounds to -2). More context can be found at following gblic manual"
        "https://www.gnu.org/software/libc/manual/html_node/Rounding.html.");
    TVM_ATTR_FIELD(compute_dtype)
        .set_default("None")
        .describe(
            "Specifies the data type used during requantize. Supported "
            "options: \"int64\", \"float32\", \"float64\"");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attribute for quantize operator */
struct QuantizeAttrs : public tvm::AttrsNode<QuantizeAttrs> {
  DataType out_dtype;
  int axis;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relay.attrs.QuantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type, can be one of [int8 or uint8].");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attribute for dequantize operator */
struct DequantizeAttrs : public tvm::AttrsNode<DequantizeAttrs> {
  DataType out_dtype;
  int axis;

  TVM_DECLARE_ATTRS(DequantizeAttrs, "relay.attrs.DequantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type, can be one of [float16, float32].");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The channel axis for channel wise dequantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attribute for broadcast operator */
struct BroadcastAttrs : public tvm::AttrsNode<BroadcastAttrs> {
  int lhs_axis;
  int rhs_axis;

  TVM_DECLARE_ATTRS(BroadcastAttrs, "relay.attrs.BroadcastAttrs") {
    TVM_ATTR_FIELD(lhs_axis)
        .describe(
            "The channel axis for channel wise broadcast. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rhs_axis)
        .describe(
            "The channel axis for channel wise broadcast. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_ATTRS_H_
