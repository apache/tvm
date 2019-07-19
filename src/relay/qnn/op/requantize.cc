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
 *  Copyright (c) 2019 by Contributors
 * \file requantize.cc
 * \brief Quantized convolution operators
 */

#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/qnn/attrs.h>
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(RequantizeAttrs);


bool RequantizeRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto input_dtype = data->dtype;
  CHECK(IsValidOpInputType(QuantizeOpType::Requantize, input_dtype))
    << "Input type should be a quantized type (u)int8 or (u)int16 but was " <<  input_dtype;

  const Array<tvm::Expr> oshape = data->shape;
  // assign output type
  const RequantizeAttrs* param = attrs.as<RequantizeAttrs>();
  reporter->Assign(types[1], TensorTypeNode::make(oshape, param->out_dtype));
  return true;
}

// Positional relay function to create quantized conv2d operator
// used by frontend FFI.
Expr MakeRequantize(Expr data,
                    double input_scale,
                    int32_t input_zero_point,
                    double output_scale,
                    int32_t output_zero_point,
                    std::string rounding,
                    DataType out_dtype) {
  auto attrs = make_node<RequantizeAttrs>();
  attrs->input_scale = std::move(input_scale);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->output_scale = std::move(output_scale);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->rounding = std::move(rounding);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("qnn.requantize");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.requantize")
.describe(R"code(Requantize operator.
The requantize operator converts one quantized tensor to another quantized
tensor. For the output tensor, we are provided with output scale and zero
point. The computation looks like this

Q_output = zp_output +  (scale_input)/(scale_ouptut) * (Q_input - zp_input)

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.RequantizeAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The quantized input tensor.")
.set_support_level(11)
.add_type_rel("Requantize", RequantizeRel);

TVM_REGISTER_API("relay.qnn.op._make.requantize")
.set_body_typed(MakeRequantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
