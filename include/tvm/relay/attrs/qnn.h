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
#ifndef TVM_RELAY_ATTRS_NN_QUANTIZE_H_
#define TVM_RELAY_ATTRS_NN_QUANTIZE_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

struct QuantizeAttrs : public tvm::AttrsNode<QuantizeAttrs> {
  int32_t output_zero_point;
  double output_scale;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relay.attrs.QuantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype)
        .describe("Output data type, can be one of [int8 or uint8].");

    TVM_ATTR_FIELD(output_zero_point)
        .describe("The zero_point for the activation of this op.");

    TVM_ATTR_FIELD(output_scale)
        .describe("The scale for the activation of this op.");
  }
};

struct DequantizeAttrs : public tvm::AttrsNode<DequantizeAttrs> {
  int32_t input_zero_point;
  double input_scale;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relay.attrs.QuantizeAttrs") {

    TVM_ATTR_FIELD(input_zero_point)
        .describe("The zero_point for the input tensor of this op.");

    TVM_ATTR_FIELD(input_scale)
        .describe("The scale for the input tensor of this op.");
  }
};



}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_NN_QUANTIZE_H_
