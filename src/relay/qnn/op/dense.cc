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
 * \file src/relay/qnn/op/dense.cc
 * \brief Property def of qnn dense operator.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../op/nn/nn.h"
#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.dense

bool QnnDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // Expected Types: data, weight, input_zero_point, weight_zero_point, input_scale, weight_scale,
  // out_type
  ICHECK_EQ(types.size(), 7);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) return false;
  const auto* param = attrs.as<DenseAttrs>();
  ICHECK(param != nullptr) << "DenseAttrs cannot be nullptr.";
  ICHECK(data->dtype == DataType::Int(8) || data->dtype == DataType::UInt(8) ||
         data->dtype == DataType::Int(16) || data->dtype == DataType::UInt(16))
      << "Expected quantized dense type(int8, uint8, int16, uint16) for input but was "
      << data->dtype;
  ICHECK(weight->dtype == DataType::Int(8) || weight->dtype == DataType::UInt(8))
      << "Expected quantized dense type(int8, uint8) for weight but was " << weight->dtype;
  ICHECK(param->out_dtype == DataType::Int(32) || param->out_dtype == DataType::Int(64))
      << "Expected quantized dense type(int32, int64) for output but was " << param->out_dtype;

  // Check the types of scale and zero points.
  for (size_t i = 2; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // input_zero_point
  ICHECK(IsScalarType(types[4], DataType::Float(32)));  // input_scale
  // weight_zero_point can be a scalar or a vector of the same shape as the weight_scale
  AssignType(types[5], DataType::Float(32), param->units, reporter);  // weight_scale

  ICHECK(param->out_dtype.bits() > 0) << "Output dtype bits should be greater than 0.";

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // Dense infer type function.
  Array<Type> tensor_types = {types[0], types[1], types[6]};
  return MatmulRel<DenseAttrs>(tensor_types, 3, attrs, reporter);
}

// Positional relay function to create quantized dense operator used by frontend FFI.
Expr MakeQuantizedDense(Expr data, Expr weight, Expr input_zero_point, Expr kernel_zero_point,
                        Expr input_scale, Expr kernel_scale, IndexExpr units, DataType out_dtype) {
  auto attrs = make_object<DenseAttrs>();
  attrs->units = std::move(units);
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("qnn.dense");
  return Call(op, {data, weight, input_zero_point, kernel_zero_point, input_scale, kernel_scale},
              Attrs(attrs), {});
}

Expr DenseFirstTerm(const Expr& quantized_data, const Expr& quantized_kernel,
                    const DenseAttrs* attrs) {
  return Dense(quantized_data, quantized_kernel, attrs->units, attrs->out_dtype);
}

Expr DenseSecondTerm(const Expr& quantized_data, const Expr& kernel_zero_point,
                     const int out_dim_size) {
  Array<Integer> axes = {1};
  Expr reduced_t2 = Sum(Cast(quantized_data, DataType::Int(32)), axes, true, false);
  Expr multiplied_t2;
  if (!IsConstScalar(kernel_zero_point)) {
    multiplied_t2 = Multiply(kernel_zero_point, MakeRepeat(reduced_t2, out_dim_size, 1));
  } else {
    multiplied_t2 = Multiply(kernel_zero_point, reduced_t2);
  }
  return multiplied_t2;
}

Expr DenseThirdTerm(const Expr& quantized_kernel, const Expr& input_zero_point) {
  Array<Integer> axes = {1};
  return Multiply(input_zero_point,
                  Sum(Cast(quantized_kernel, DataType::Int(32)), axes, false, false));
}

Expr DenseFourthTerm(int input_zero_point_int, int kernel_zero_point_int, int reduction_dim_size) {
  int32_t scalar_term = input_zero_point_int * kernel_zero_point_int * reduction_dim_size;
  return MakeConstantScalar(DataType::Int(32), scalar_term);
}

Expr DenseFourthTerm(const Expr& input_zero_point, const Expr& kernel_zero_point,
                     int reduction_dim_size) {
  auto reduction_dim = MakeConstantScalar(DataType::Int(32), reduction_dim_size);
  return Multiply(Multiply(input_zero_point, kernel_zero_point), reduction_dim);
}

Expr DenseCombineTerms(const Expr& term1, const Expr& term2, const Expr& term3, const Expr& term4) {
  auto data_term = Subtract(term1, term2);
  // Putting constant terms together, so that constant folding can fold it.
  auto const_term = Subtract(term4, term3);
  return Add(data_term, const_term);
}
/*
 * \brief Forward rewrite the qnn dense op.
 * \param attrs The QNN dense attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for qnn cov2d op.
 * \note Lowering of the qnn.dense operator
 *       A quantized tensor is represented in following manner
 *          A = scale_a x (QA - zp_A)
 *       where QA is quantized tensor, scale_a and zp_A are quantization
 *       params.
 *
 *       Quantized dense multiplies two quantized tensors and returns a
 *       quantized tensor of default dtype of int32, with scale equaling to the
 *       product of scales of input tensors, and a zero point of zero.
 *
 *       The lowering for asymmetric quantized dense looks as follows. More details at
 *       https://discuss.tvm.ai/t/tf-lite-quantized-conv2d-operator-conversion/2651/8
 *       The computation gets unrolled into following 4 terms
 *          C(m, n) = Sigma(k) (A(m, k) * W(n, k))
 *
 *          RHS becomes
 *            Sigma(k) ([QA(m, k) - zp_a] * [QW(n, k) - zp_w])
 *
 *          Unrolling leads to following sequence
 *            Sigma(k) QA(m, k) * QW(n, k)                         // Term1
 *          - Sigma(k) zp_w * QA(m, k)                             // Term2
 *          - Sigma(k) zp_a * QW(n, k)                             // Term3
 *          - Sigma(k) * zp_a * zp_w                               // Term4
 *
 *       Term3 and Term4 can be computed at compile time.
 */
Expr QnnDenseCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                          const Array<tvm::relay::Type>& arg_types) {
  ICHECK_EQ(new_args.size(), 6);
  Expr quantized_data = new_args[0];
  Expr quantized_kernel = new_args[1];
  Expr input_zero_point = new_args[2];
  Expr kernel_zero_point = new_args[3];

  const auto in_shape = get_shape(arg_types[0]);
  const auto w_shape = get_shape(arg_types[1]);
  const int reduction_dim_size = get_const_int(in_shape[1]);
  const int out_dim_size = get_const_int(w_shape[0]);

  const auto* qnn_dense_attrs = attrs.as<DenseAttrs>();

  auto term1 = DenseFirstTerm(quantized_data, quantized_kernel, qnn_dense_attrs);
  auto term2 = DenseSecondTerm(quantized_data, kernel_zero_point, out_dim_size);
  auto term3 = DenseThirdTerm(quantized_kernel, input_zero_point);

  // Extract the integer zero points.

  if (!IsConstScalar(input_zero_point) || !IsConstScalar(kernel_zero_point)) {
    auto term4 = DenseFourthTerm(input_zero_point, kernel_zero_point, reduction_dim_size);
    return DenseCombineTerms(term1, term2, term3, term4);
  }

  auto kernel_zero_point_int = GetScalarFromConstant<int>(kernel_zero_point);
  auto input_zero_point_int = GetScalarFromConstant<int>(input_zero_point);

  // Get all the terms as described in the comments.
  auto term4 = DenseFourthTerm(input_zero_point_int, kernel_zero_point_int, reduction_dim_size);

  // Combine those 4 terms depending on the zero points to get the best lowering.
  if (input_zero_point_int == 0 && kernel_zero_point_int == 0) {
    // term 2, 3 and 4 become zero.
    return term1;
  } else if (input_zero_point_int == 0 && kernel_zero_point_int != 0) {
    // term 3 and term 4 become zero.
    return Subtract(term1, term2);
  } else if (input_zero_point_int != 0 && kernel_zero_point_int == 0) {
    // term 2 and term 4 become zero.
    return Subtract(term1, term3);
  } else {
    return DenseCombineTerms(term1, term2, term3, term4);
  }
}

RELAY_REGISTER_OP("qnn.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.
- **data**: quantized(int8, unit8) `(x1, x2, ..., xn, input_dim)`
- **weight**: quantized(int8, unit8) `(units, input_dim)`
- **out**: quantized(int32) `(x1, x2, ..., xn, units)`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<DenseAttrs>()
    .set_num_inputs(6)
    .add_argument("data", "quantized nD Tensor", "Input data.")
    .add_argument("weight", "quantized 2D Tensor", "Weight matrix.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("weight_scale", "Tensor", "The quantization scale of the weight tensor.")
    .add_argument("weight_zero_point", "Tensor",
                  "The quantization zero_point of the weight tensor.")
    .set_support_level(11)
    .add_type_rel("QDense", QnnDenseRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnDenseCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.dense").set_body_typed(MakeQuantizedDense);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
