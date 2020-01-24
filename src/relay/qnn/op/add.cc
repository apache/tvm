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
 * \file src/relay/qnn/op/add.cc
 * \brief QNN add operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../../pass/pattern_util.h"
#include "../util.h"
#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

/*
 * \brief Canonicalizes the QNN add op.
 * \param attrs The QNN concatenate attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for add op.
 */
Expr QnnAddCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                        const Array<tvm::relay::Type>& arg_types) {
  // Get the attrs.
  CHECK_EQ(new_args.size(), 8);
  auto& lhs = new_args[0];
  auto& rhs = new_args[1];
  auto& lhs_scale = new_args[2];
  auto& lhs_zero_point = new_args[3];
  auto& rhs_scale = new_args[4];
  auto& rhs_zero_point = new_args[5];
  auto& output_scale = new_args[6];
  auto& output_zero_point = new_args[7];

  // Get the input dtype and shape.
  CHECK_EQ(arg_types.size(), 9);
  auto tensor_type = arg_types[0].as<TensorTypeNode>();
  CHECK(tensor_type != nullptr);
  auto input_dtype = tensor_type->dtype;
  auto input_shape = tensor_type->shape;

  // FIXME (anijain2305) - The lowering can be further optimized. Instead of inserting requantize in
  // the start, we can insert requantize at the end if both input tensors have same qnn params. In
  // that case, we can first add the tensors, subtract the zero point, and requantize at the end.
  // This can be done in future.

  // Since the input qnn params can be different than output qnn params, we first requantize the
  // input tensors to the output qnn params. Then we call relay.add on the requantized inputs. This
  // addition results in extra addition of the output zero point. We futher subtract the zero
  // point. The whole process can be represented using following equations
  //
  //          scale_c * (Q_c - zp_c) = scale_a * (Q_a - zp_a) + scale_b * (Q_b - zp_b)
  //
  // After requantizing Q_a and Q_b, equation becomes,
  //          scale_c * (Q_c - zp_c) = scale_c * (Q_a' - zp_c) + scale_c * (Q_b' - zp_c)
  //          scale_c * (Q_c - zp_c) = scale_c * (Q_a' + Q_b' - zp_c - zp_c)
  //
  // Comparing the LHS and RHS, it results in
  //          Q_c = Q_a' + Q_b' - zp_c
  // The add op is done in int32 precision.

  // Requantize LHS if necessary.
  auto requantized_lhs = lhs;
  if (!IsEqualScalar(lhs_scale, output_scale) ||
      !IsEqualScalar(lhs_zero_point, output_zero_point)) {
    requantized_lhs = Requantize(lhs, input_shape, lhs_scale, lhs_zero_point, output_scale,
                                 output_zero_point, DataType::Int(32));
  } else {
    requantized_lhs = Cast(requantized_lhs, DataType::Int(32));
  }

  // Requantize RHS if necessary.
  auto requantized_rhs = rhs;
  if (!IsEqualScalar(rhs_scale, output_scale) ||
      !IsEqualScalar(rhs_zero_point, output_zero_point)) {
    requantized_rhs = Requantize(rhs, input_shape, rhs_scale, rhs_zero_point, output_scale,
                                 output_zero_point, DataType::Int(32));
  } else {
    requantized_rhs = Cast(requantized_rhs, DataType::Int(32));
  }

  auto output = Add(requantized_lhs, requantized_rhs);

  // Subtract zero point.
  auto zero_scalar = MakeConstantScalar(DataType::Int(32), 0);
  if (!IsEqualScalar(output_zero_point, zero_scalar)) {
    output = Subtract(output, output_zero_point);
  }

  // Go back to lower precision.
  auto q_min = GetQmin(input_dtype);
  auto q_max = GetQmax(input_dtype);
  output = Clip(output, q_min, q_max);
  return Cast(output, input_dtype);
}

// QNN Addition operator.
QNN_REGISTER_BINARY_OP("add")
.describe("Elementwise add with with broadcasting for quantized tensors.")
.set_support_level(11)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnAddCanonicalize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
