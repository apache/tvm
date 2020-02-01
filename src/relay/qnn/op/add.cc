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

#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(QnnAddAttrs);

/*
 * \brief Canonicalizes the QNN add op.
 * \param attrs The QNN add attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for add op.
 */
Expr QnnAddCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                        const Array<tvm::relay::Type>& arg_types) {
  // Get the args.
  QnnBinaryOpArguments args(new_args);

  // Get the attrs.
  const QnnAddAttrs* add_attrs = attrs.as<QnnAddAttrs>();
  CHECK(add_attrs != nullptr);
  auto& rounding = add_attrs->rounding;

  // Get the input dtype and shape.
  QnnBinaryOpTensorType input_type(arg_types, 0);

  if (rounding == "TFLITE") {
    double lhs_scale_val = GetScalarFromConstant<float>(args.lhs_scale);
    double rhs_scale_val = GetScalarFromConstant<float>(args.rhs_scale);
    double out_scale_val = GetScalarFromConstant<float>(args.output_scale);
    double twice_max_input_scale = 2 * std::max(lhs_scale_val, rhs_scale_val);
    double real_lhs_scale_val = lhs_scale_val / twice_max_input_scale;
    double real_rhs_scale_val = rhs_scale_val / twice_max_input_scale;
    double real_out_scale_val = twice_max_input_scale / ((1 << 20) * out_scale_val);

    auto real_lhs_scale = MakeConstantScalar<double>(DataType::Float(64), real_lhs_scale_val);
    auto real_rhs_scale = MakeConstantScalar<double>(DataType::Float(64), real_rhs_scale_val);
    auto real_out_scale = MakeConstantScalar<double>(DataType::Float(64), real_out_scale_val);
    auto one_scalar = MakeConstantScalar<double>(DataType::Float(64), 1);
    auto zero_scalar = MakeConstantScalar<int>(DataType::Int(32), 0);
    auto left_shift_scalar = MakeConstantScalar<int>(DataType::Int(32), 1 << 20);

    Expr adapted_lhs = Cast(args.lhs, DataType::Int(32));
    if (!IsEqualScalar(args.lhs_zero_point, zero_scalar)) {
      adapted_lhs = Subtract(adapted_lhs, Cast(args.lhs_zero_point, DataType::Int(32)));
    }
    adapted_lhs = Multiply(adapted_lhs, left_shift_scalar);

    Expr adapted_rhs = Cast(args.rhs, DataType::Int(32));
    if (!IsEqualScalar(args.rhs_zero_point, zero_scalar)) {
      adapted_rhs = Subtract(adapted_rhs, Cast(args.rhs_zero_point, DataType::Int(32)));
    }
    adapted_rhs = Multiply(adapted_rhs, left_shift_scalar);

    auto requantized_lhs = Requantize(adapted_lhs, input_type.shape, real_lhs_scale, zero_scalar,
                                      one_scalar, zero_scalar, DataType::Int(32), rounding);

    auto requantized_rhs = Requantize(adapted_rhs, input_type.shape, real_rhs_scale, zero_scalar,
                                      one_scalar, zero_scalar, DataType::Int(32), rounding);

    auto output = Add(requantized_lhs, requantized_rhs);
    output = Requantize(output, input_type.shape, real_out_scale, zero_scalar, one_scalar,
                        args.output_zero_point, DataType::Int(32), rounding);
    // Go back to lower precision.
    return ConvertDtype(output, input_type.dtype);
  }

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

  // Requantize LHS if necessary. Computes Q_a'
  auto requantized_lhs =
      RequantizeOrUpcast(args.lhs, args.lhs_scale, args.lhs_zero_point, args.output_scale,
                         args.output_zero_point, input_type.shape);
  // Requantize RHS if necessary. Computes Q_b'
  auto requantized_rhs =
      RequantizeOrUpcast(args.rhs, args.rhs_scale, args.rhs_zero_point, args.output_scale,
                         args.output_zero_point, input_type.shape);
  // Computes Q_a' + Q_b'
  auto output = Add(requantized_lhs, requantized_rhs);

  // Subtract zero point. Computes (Q_a' + Q_b') - zp_c
  auto zero_scalar = MakeConstantScalar(DataType::Int(32), 0);
  if (!IsEqualScalar(args.output_zero_point, zero_scalar)) {
    output = Subtract(output, args.output_zero_point);
  }

  // Go back to lower precision.
  return ConvertDtype(output, input_type.dtype);
}

Expr MakeQnnAdd(Expr lhs, Expr rhs, Expr lhs_scale, Expr lhs_zero_point, Expr rhs_scale,
                Expr rhs_zero_point, Expr output_scale, Expr output_zero_point,
                std::string rounding) {
  auto attrs = make_object<QnnAddAttrs>();
  attrs->rounding = std::move(rounding);

  static const Op& op = Op::Get("qnn.add");
  return Call(op,
              {lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale,
               output_zero_point},
              Attrs(attrs), {});
}

// QNN Addition operator.
QNN_REGISTER_BINARY_OP_WITH_BODY("add", MakeQnnAdd)
    .describe("Elementwise add with with broadcasting for quantized tensors.")
    .set_attrs_type<QnnAddAttrs>()
    .set_support_level(11)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnAddCanonicalize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
