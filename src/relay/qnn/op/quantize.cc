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
 * \file src/relay/qnn/op/quantize.cc
 * \brief QNN dequantize operator. Dequantize operator converts from quantized
 * domain to unquantized domain.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../../pass/pattern_util.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(QuantizeAttrs);

bool QuantizeRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto input_dtype = data->dtype;
  CHECK(input_dtype == DataType::Float(32))
    << "Input type should be one of float32 but was " <<  input_dtype;
  const auto* quantize_attrs = attrs.as<QuantizeAttrs>();
  const Array<tvm::Expr> oshape = data->shape;
  const DataType out_dtype = quantize_attrs->out_dtype;
  CHECK(out_dtype == DataType::Int(8) ||
        out_dtype == DataType::UInt(8) ||
        out_dtype == DataType::Int(32))
    << "Output type should be one of [int8, unit8, int32] but was " << out_dtype;
  // assign output type
  reporter->Assign(types[1], TensorTypeNode::make(oshape, out_dtype));
  return true;
}

Expr MakeQuantize(Expr data,
                  double output_scale,
                  int32_t output_zero_point,
                  DataType out_dtype) {
  auto attrs = make_object<QuantizeAttrs>();
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = std::move(out_dtype);
  // result_quantized_value = result_zero_point + result_real_value / result_scale.
  // A more detailed explanation can be found here - https://github.com/google/gemmlowp/blob/master/doc/quantization.md
  static const Op& op = Op::Get("qnn.quantize");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

Expr QuantizeLower(const Expr& input_tensor,
                   const QuantizeAttrs* attrs) {
  const auto out_dtype = attrs->out_dtype;
  const auto output_zero_point = MakeConstantScalar(DataType::Float(32), attrs->output_zero_point);
  const auto scale = MakeConstantScalar(DataType::Float(32), attrs->output_scale);
  const int32_t min_val = GetQmin(out_dtype);
  const int32_t max_val = GetQmax(out_dtype);
  auto scale_data = Divide(input_tensor, scale);
  auto add_zero_point = Cast(Round(Add(scale_data, output_zero_point)), DataType::Int(32));
  auto clamped_output = Clip(add_zero_point, min_val, max_val);
  auto clamp_out_dtype = Cast(clamped_output, out_dtype);
  return clamp_out_dtype;
}

Expr QuantizeQnnCanonicalize(const Attrs& attrs,
                             const Array<Expr>& new_args,
                             const Array<tvm::relay::Type>& types) {
  CHECK_EQ(new_args.size(), 1);
  auto& data = new_args[0];
  const auto* quantize_attrs = attrs.as<QuantizeAttrs>();
  CHECK(quantize_attrs != nullptr);

  CHECK_EQ(types.size(), 2);
  return QuantizeLower(data, quantize_attrs);
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
.set_attrs_type<QuantizeAttrs>()
.set_num_inputs(1)
.add_argument("data", "Tensor", "The tensor to quantize.")
.set_support_level(11)
.add_type_rel("Quantize", QuantizeRel)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QuantizeQnnCanonicalize);

TVM_REGISTER_API("relay.qnn.op._make.quantize")
.set_body_typed(MakeQuantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
