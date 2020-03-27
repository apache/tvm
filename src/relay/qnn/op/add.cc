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

/*
 * \brief Canonicalizes the QNN add op.
 * \param attrs The empty attribute.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for add op.
 */
Expr QnnAddCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                        const Array<tvm::relay::Type>& arg_types) {
  // Get the args.
  QnnBinaryOpArguments args(new_args);

  // Get the input dtype and shape.
  QnnBinaryOpTensorType input_type(arg_types, 0);


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
  auto requantized_lhs = RequantizeOrUpcast(args.lhs, args.lhs_scale,
                                            args.lhs_zero_point,
                                            args.output_scale, args.output_zero_point,
                                            input_type.shape);
  // Requantize RHS if necessary. Computes Q_b'
  auto requantized_rhs = RequantizeOrUpcast(args.rhs, args.rhs_scale,
                                            args.rhs_zero_point,
                                            args.output_scale, args.output_zero_point,
                                            input_type.shape);
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

// QNN Addition operator.
QNN_REGISTER_BINARY_OP("add")
.describe("Elementwise add with with broadcasting for quantized tensors.")
.set_support_level(11)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnAddCanonicalize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
