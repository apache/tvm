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

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  double input_scale;
  int32_t input_zero_point;
  double output_scale;
  int32_t output_zero_point;
  bool use_int_compute;
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
    TVM_ATTR_FIELD(use_int_compute).set_default(false)
        .describe("When true, the integer computation is used to handle output scale");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_NN_QUANTIZE_H_
