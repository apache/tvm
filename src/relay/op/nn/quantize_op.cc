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

TVM_REGISTER_NODE_TYPE(QuantizeAttrs);

bool QuantizeRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto input_dtype = data->dtype;
  CHECK(is_valid_quantized_op_input_type(QuantizeOpType::Quantize, input_dtype))
      << "Input type should be one of float32 but was " <<  input_dtype;
  const auto* param = attrs.as<QuantizeAttrs>();
  const Array<tvm::Expr> oshape = data->shape;
  const DataType out_dtype = param->out_dtype;
  CHECK(is_valid_quantized_op_output_type(QuantizeOpType::Quantize, out_dtype))
      << "Output type should be one of [int8, unit8 ] but was " << out_dtype;
  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, out_dtype));
  return true;
}

Expr MakeQuantize(Expr data,
                  int32_t output_zero_point,
                  double output_scale,
                  DataType out_dtype) {
  auto attrs = make_node<QuantizeAttrs>();
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("qnn.quantize");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.quantize")
.describe(R"code(Quantizes the input and produces quantized output.

The input can be either float or quantized(int8, unit8). If the input is float,
this op takes scale and zero point and quantize the float value to
quantized output, in int8 or uint8 format. If the input is quantized value,
the op requantize the input (of a certain type, with a given scale and zero
point) to the output of the same or different type with a same or different
scale and zero point.

- **data**: Tensor of any shape to quantize. The input data can be of floating point
            or quantized.
)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.QuantizeAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The tensor to quantize.")
.set_support_level(10)
.add_type_rel("Quantize", QuantizeRel);

TVM_REGISTER_API("relay.op.qnn._make.quantize")
.set_body_typed(MakeQuantize);

}  // namespace relay
}  // namespace tvm