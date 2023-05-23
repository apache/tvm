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
 * \file src/relay/qnn/op/batch_matmul.cc
 * \brief Property def of qnn batch_matmul operator.
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

// relay.op.qnn.batch_matmul

bool QnnBatchMatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  // Expected Types: x, y, x_zero_point, y_zero_point, x_scale, y_scale,
  // out_type
  ICHECK_EQ(types.size(), 7);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;
  const auto* param = attrs.as<BatchMatmulAttrs>();
  ICHECK(param != nullptr) << "BatchMatmulAttrs cannot be nullptr.";
  ICHECK(x->dtype == DataType::Int(8) || x->dtype == DataType::UInt(8))
      << "Expected quantized batch_matmul type(int8, uint8) for input but was " << x->dtype;
  ICHECK(y->dtype == DataType::Int(8) || y->dtype == DataType::UInt(8))
      << "Expected quantized batch_matmul type(int8, uint8) for weight but was " << y->dtype;
  ICHECK(param->out_dtype == DataType::Int(32))
      << "Expected quantized batch_matmul type(int32) for output but was " << param->out_dtype;

  // Check the types of scale and zero points.
  for (size_t i = 2; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // x_zero_point
  ICHECK(IsScalarType(types[3], DataType::Int(32)));    // y_zero_point
  ICHECK(IsScalarType(types[4], DataType::Float(32)));  // x_scale
  ICHECK(IsScalarType(types[5], DataType::Float(32)));  // y_scale

  ICHECK(param->out_dtype.bits() > 0) << "Output dtype bits should be greater than 0.";

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // BatchMatmul infer type function.
  Array<Type> tensor_types = {types[0], types[1], types[6]};
  return BatchMatmulRel<BatchMatmulAttrs>(tensor_types, 3, attrs, reporter);
}

// Positional relay function to create quantized batch_matmul operator used by frontend FFI.
Expr MakeQuantizedBatchMatmul(Expr x, Expr y, Expr x_zero_point, Expr y_zero_point, Expr x_scale,
                              Expr y_scale, DataType out_dtype) {
  auto attrs = make_object<BatchMatmulAttrs>();
  attrs->out_dtype = out_dtype;
  // For legacy reason, currently `qnn.batch_matmul` only supports
  // (transpose_a=false, transpose_b=true)
  // TODO(jcf94): extent to support all tensor format
  attrs->transpose_a = false;
  attrs->transpose_b = true;
  static const Op& op = Op::Get("qnn.batch_matmul");
  return Call(op, {x, y, x_zero_point, y_zero_point, x_scale, y_scale}, Attrs(attrs), {});
}

Expr BatchMatmulFirstTerm(const Expr& quantized_x, const Expr& quantized_y,
                          const BatchMatmulAttrs* attrs) {
  ICHECK(attrs->transpose_a == false && attrs->transpose_b == true)
      << "Currently qnn.batch_matmul only supports (transpose_a=false, transpose_b=true).";
  return MakeBatchMatmul(quantized_x, quantized_y, attrs->out_dtype, attrs->transpose_a,
                         attrs->transpose_b);
}

Expr BatchMatmulSecondTerm(const Expr& x_quantized_data, const Expr& y_zero_point) {
  if (IsScalar(y_zero_point)) {
    Array<Integer> axes = {2};
    return Multiply(y_zero_point,
                    Sum(Cast(x_quantized_data, DataType::Int(32)), axes, true, false));
  } else {
    LOG(FATAL) << "Tensor zero point (non-scalar) is not supported";
    return Expr();
  }
}

Expr BatchMatmulThirdTerm(const Expr& y_quantized_data, const Expr& x_zero_point,
                          int broadcast_dim_size) {
  if (IsScalar(x_zero_point)) {
    Array<Integer> axes = {2};
    auto reducemult =
        Multiply(x_zero_point, Sum(Cast(y_quantized_data, DataType::Int(32)), axes, true, false));
    Array<Integer> newshape;

    // dimension of 0 in reshape copies old dimension size
    newshape = {0, 1, broadcast_dim_size};
    return Reshape(reducemult, newshape);
  } else {
    LOG(FATAL) << "Tensor zero point (non-scalar) is not supported";
    return Expr();
  }
}

Expr BatchMatmulFourthTerm(Expr x_zero_point, Expr y_zero_point, int reduction_dim_size) {
  if (IsScalar(x_zero_point) && IsScalar(y_zero_point)) {
    auto zero_point_mul = Multiply(x_zero_point, y_zero_point);
    auto const_scale = MakeConstantScalar(DataType::Int(32), reduction_dim_size);
    return Multiply(zero_point_mul, const_scale);
  } else {
    LOG(FATAL) << "Tensor zero point (non-scalar) is not supported";
    return Expr();
  }
}

Expr BatchMatmulFourthTerm(int x_zero_point_int, int y_zero_point_int, int reduction_dim_size) {
  int32_t scalar_term = x_zero_point_int * y_zero_point_int * reduction_dim_size;
  return MakeConstantScalar(DataType::Int(32), scalar_term);
}

Expr BatchMatmulCombineTerms(const Expr& term1, const Expr& term2, const Expr& term3,
                             const Expr& term4) {
  auto data1_term = Subtract(term1, term2);
  auto data2_term = Subtract(term4, term3);
  return Add(data1_term, data2_term);
}

/*
 * \brief Forward rewrite the qnn batch_matmul op.
 * \param attrs The QNN batch_matmul attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for qnn batch_matmul op.
 * \note Lowering of the qnn.batch_matmul operator
 *       A quantized tensor is represented in following manner
 *          A = scale_a x (QA - zp_A)
 *       where QA is quantized tensor, scale_a and zp_A are quantization
 *       params.
 *
 *       Quantized batch_matmul multiplies two quantized tensors and returns a
 *       quantized tensor of default dtype of int32, with scale equaling to the
 *       product of scales of input tensors, and a zero point of zero.
 *
 *       The lowering for asymmetric quantized batch_matmul looks similar to
 *       quantized conv2d and dense and originally was discussed here:
 *       https://discuss.tvm.apache.org/t/tf-lite-quantized-conv2d-operator-conversion/2651/7
 *
 *       The computation gets unrolled into following 4 terms
 *          C(m, n) = Sigma(k) (X(m, k) * Y(n, k))
 *
 *          RHS becomes
 *            Sigma(k) ([QX(m, k) - zp_x] * [QY(n, k) - zp_y])
 *
 *          Unrolling leads to following sequence
 *            Sigma(k) QX(m, k) * QX(n, k)                         // Term1
 *          - Sigma(k) zp_y * QX(m, k)                             // Term2
 *          - Sigma(k) zp_x * QY(n, k)                             // Term3
 *          - Sigma(k) * zp_x * zp_y                               // Term4
 *
 *       Term4 can be computed at compile time, everything else depending on the
 *       input type.
 */
Expr QnnBatchMatmulCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                                const Array<tvm::relay::Type>& arg_types) {
  ICHECK_EQ(new_args.size(), 6);
  Expr quantized_x = new_args[0];
  Expr quantized_y = new_args[1];
  Expr x_zero_point = new_args[2];
  Expr y_zero_point = new_args[3];

  const auto in_shape = get_shape(arg_types[0]);
  const int reduction_dim_size = get_const_int(in_shape[2]);

  const auto y_shape = get_shape(arg_types[1]);
  const int broadcast_dim_size = get_const_int(y_shape[1]);

  const auto* qnn_batch_matmul_attrs = attrs.as<BatchMatmulAttrs>();

  // Get all the terms as described in the comments.
  auto term1 = BatchMatmulFirstTerm(quantized_x, quantized_y, qnn_batch_matmul_attrs);
  auto term2 = BatchMatmulSecondTerm(quantized_x, y_zero_point);
  auto term3 = BatchMatmulThirdTerm(quantized_y, x_zero_point, broadcast_dim_size);

  if (IsConstScalar(x_zero_point) && IsConstScalar(y_zero_point)) {
    // Extract the integer zero points.
    auto y_zero_point_int = GetScalarFromConstant<int>(y_zero_point);
    auto x_zero_point_int = GetScalarFromConstant<int>(x_zero_point);
    auto term4 = BatchMatmulFourthTerm(x_zero_point_int, y_zero_point_int, reduction_dim_size);
    // Combine those 4 terms depending on the zero points to get the best lowering.
    if (x_zero_point_int == 0 && y_zero_point_int == 0) {
      // term 2, 3 and 4 become zero.
      return term1;
    } else if (x_zero_point_int == 0 && y_zero_point_int != 0) {
      // term 3 and term 4 become zero.
      return Subtract(term1, term2);
    } else if (x_zero_point_int != 0 && y_zero_point_int == 0) {
      // term 2 and term 4 become zero.
      return Subtract(term1, term3);
    } else {
      return BatchMatmulCombineTerms(term1, term2, term3, term4);
    }
  } else {
    auto term4 = BatchMatmulFourthTerm(x_zero_point, y_zero_point, reduction_dim_size);
    return BatchMatmulCombineTerms(term1, term2, term3, term4);
  }
}

RELAY_REGISTER_OP("qnn.batch_matmul")
    .describe(R"code(Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

Note we expect tensor_b to be transposed to copy the standard nn.batch_matmul conventions.

.. math::

  batch\_matmul(A, B)[i, :, :] = matmul(A[i, :, :], B[i, :, :]^T)

- **data**: quantized(int8, unit8) `(i, m, k)`
- **weight**: quantized(int8, unit8) `(i, n, k)`
- **out**: quantized(int32) `(i, m, n)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<BatchMatmulAttrs>()
    .set_num_inputs(6)
    .add_argument("x", "quantized 2D Tensor", "First input data.")
    .add_argument("y", "quantized 2D Tensor", "Second input data.")
    .add_argument("x_scale", "Tensor", "The quantization scale of the x input tensor.")
    .add_argument("x_zero_point", "Tensor", "The quantization zero_point of the x input tensor.")
    .add_argument("y_scale", "Tensor", "The quantization scale of the y input tensor.")
    .add_argument("y_zero_point", "Tensor", "The quantization zero_point of the y input tensor.")
    .set_support_level(11)
    .add_type_rel("QBatchMatmul", QnnBatchMatmulRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnBatchMatmulCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.batch_matmul").set_body_typed(MakeQuantizedBatchMatmul);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
