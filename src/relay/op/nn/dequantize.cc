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
 *  Copyright (c) 2018 by Contributors
 * \file quantize.cpp
 * \brief Quantize and requantize operator
 */

#include <tvm/relay/op.h>
#include <tvm/relay/attrs/qnn.h>
#include <tvm/relay/quantize_util.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(DequantizeAttrs);

bool DequantizeRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto input_dtype = data->dtype;
  CHECK(is_valid_quantized_op_input_type(QuantizeOpType::Dequantize, input_dtype))
    << "Input type should be one of the quantized types [unit8, int8] but was " <<  input_dtype;
  const Array<tvm::Expr> oshape = data->shape;
  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, Float(32)));
  return true;
}

Expr MakeDequantize(Expr data,
                  int32_t input_zero_point,
                  double input_scale) {
  auto attrs = make_node<DequantizeAttrs>();
  attrs->input_scale = input_scale;
  attrs->input_zero_point = input_zero_point;
  static const Op& op = Op::Get("qnn.dequantize");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.dequantize")
    .describe(R"code(Quantizes the input and produces quantized output.

The input is always quantized (int8, uint8) and will be converted to float32 given input scale and shift.
- **data**: Quantized tensor of any shape to dequantize. The input data can be of floating point
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.DequantizeAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The tensor to dequantize.")
.set_support_level(10)
.add_type_rel("Dequantize", DequantizeRel);

TVM_REGISTER_API("relay.op.qnn._make.dequantize")
.set_body_typed(MakeDequantize);

}  // namespace relay
}  // namespace tvm
