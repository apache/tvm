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
 * \file src/relay/qnn/op/requantize.cc
 * \brief QNN requantize operator.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../../pass/pattern_util.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(RequantizeAttrs);

// Lowering of qnn.requantize op



/*
 * \brief Lower requantize to a sequence of ops.
 * \param input_tensor The input tensor to requantize op.
 * \param param The requantize op attrs.
 * \param input_shape The input tensor shape of the requantize op.
 * \return The sequence of existing Relay ops.
 * \note Requantization using only integer computation. Here, the computation is
 *       converted to a fixed point computation by computing output multiplier
 *       and shift. This is useful, if the target device does not support/have
 *       very expensive floating point computations.
 *
 *       The whole computation this can be broken down into following steps
 *       1) Calculate the integer multiplier and integer shift.
 *       2) Subtract the input integer zero point.
 *       3) Perform fixed point multiplication.
 *       4) Add the output zero point.
 *       5) Cast to the out_dtype.
 */
Expr RequantizeLower(const Expr& input_tensor, const RequantizeAttrs* param,
                     const Array<IndexExpr>& input_shape, const DataType& out_dtype) {
  double double_multiplier = param->input_scale / param->output_scale;

  DataType hp_dtype = Int(64);

  auto tensor = Cast(input_tensor, hp_dtype);
  // 1) Subtract the input_zero_point
  if (param->input_zero_point != 0) {
    auto input_zp = MakeConstantScalar(hp_dtype, param->input_zero_point);
    tensor = Subtract(tensor, input_zp);
  }

  // 2) If the input and output scales are same, we can skip the fixed point multiplication.
  auto scaled_int64_t = tensor;
  if (param->input_scale != param->output_scale) {
    scaled_int64_t = FixedPointMuliply(scaled_int64_t, double_multiplier, input_shape,
                                       param->rounding);
  }

  // 3) Add the output zero point.
  auto shifted_int64_t = scaled_int64_t;
  if (param->output_zero_point != 0) {
    auto output_zp = MakeConstantScalar(hp_dtype, param->output_zero_point);
    shifted_int64_t = Add(output_zp, scaled_int64_t);
  }

  // 4) Clip to the out_dtype min/max.
  auto q_min = GetQmin(out_dtype);
  auto q_max = GetQmax(out_dtype);
  auto clipped_t = Clip(shifted_int64_t, q_min, q_max);
  return Cast(clipped_t, out_dtype);
}

/*
 * \brief Forward rewrite the requantize op.
 * \param ref_call The original call that will be lowered.
 * \param new_args The new mutated args to the call node.
 * \param ctx The node context.
 * \return The sequence of Relay ops for requantize op.
 * \note Lowering of the requantize operation. The requantize operator converts
 *       one quantized tensor to another quantized tensor. For the output
 *       tensor, we are provided with output scale and zero point. The
 *       computation looks like this
 *
 * Q_output = zp_output +  (scale_input)/(scale_ouptut) * (Q_input - zp_input)
 */
Expr RequantizeQnnCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                               const Array<tvm::relay::Type>& types) {
  CHECK_EQ(new_args.size(), 1);
  auto& quantized_data = new_args[0];
  const auto* param = attrs.as<RequantizeAttrs>();
  CHECK(param != nullptr);

  // Find input shape.
  CHECK_EQ(types.size(), 2);
  auto in_type = types[0];
  auto in_tensor_type = in_type.as<TensorTypeNode>();
  CHECK(in_tensor_type != nullptr) << "Type information missing."
                                   << " Please run infer_type pass.";
  Array<IndexExpr> input_shape = in_tensor_type->shape;

  // Find the output dtype.
  auto out_type = types[1];
  auto out_tensor_type = out_type.as<TensorTypeNode>();
  CHECK(out_tensor_type != nullptr) << "Type information missing."
                                    << " Please run infer_type pass.";
  auto out_dtype = out_tensor_type->dtype;

  // Check rounding validity.
  CHECK(param->rounding == "UPWARD" || param->rounding == "TONEAREST")
      << "QNN requantize supports two rounding modes - UPWARD and "
      << "TONEAREST";
  return RequantizeLower(quantized_data, param, input_shape, out_dtype);
}

/*
 * \brief Infer shape function of Requantize op.
 * \param types The types of input args.
 * \param num_inputs The number of inputs.
 * \param attrs The op attributes.
 * \param reporter The type reporter that sets the dtype and shapes.
 * \return True if the infer shape succeeded.
 */
bool RequantizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto in_dtype = data->dtype;
  CHECK(in_dtype == Int(8) || in_dtype == UInt(8) || in_dtype == Int(32))
      << "Input type should be one of [int8, uint8, int32] but was " << in_dtype;

  const Array<tvm::Expr> oshape = data->shape;
  // assign output type
  const RequantizeAttrs* param = attrs.as<RequantizeAttrs>();
  auto out_dtype = param->out_dtype;
  CHECK(out_dtype == Int(8) || out_dtype == UInt(8) || out_dtype == Int(32))
      << "Output type should be one of [int8, uint8, int32] but was " << out_dtype;
  reporter->Assign(types[1], TensorTypeNode::make(oshape, out_dtype));
  return true;
}

// Positional relay function to create qnn requantize operator
// used by frontend FFI.
Expr MakeRequantize(Expr data, double input_scale, int32_t input_zero_point, double output_scale,
                    int32_t output_zero_point, std::string rounding, DataType out_dtype) {
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

Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.RequantizeAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The quantized input tensor.")
.set_support_level(11)
.add_type_rel("Requantize", RequantizeRel)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", RequantizeQnnCanonicalize);

TVM_REGISTER_API("relay.qnn.op._make.requantize")
.set_body_typed(MakeRequantize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
