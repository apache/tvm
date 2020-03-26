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
 * \file src/relay/qnn/util.h
 * \brief Utility methods needs for quantized ops that can be shared
 */

#ifndef TVM_RELAY_QNN_UTIL_H_
#define TVM_RELAY_QNN_UTIL_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/attrs.h>
#include <limits>
#include <string>
#include <vector>
#include <utility>

namespace tvm {
namespace relay {
namespace qnn {

static inline Array<IndexExpr> get_shape(const Type& type) {
  auto input_tt = type.as<TensorTypeNode>();
  CHECK(input_tt != nullptr) << "Type information missing."
                             << " Please run infer_type pass.";
  return input_tt->shape;
}

static inline int32_t GetQmin(const DataType& dtype) {
  CHECK_LE(dtype.bits(), 32)
      << "QNN ops support int32 or lower precision";
  if (dtype.is_int() || dtype.is_uint()) {
    auto* min_value = tir::as_const_int(tvm::min_value(dtype));
    CHECK(min_value != nullptr);
    return static_cast<int32_t>(min_value[0]);
  } else {
    LOG(FATAL) << "Type not supported " << dtype;
    return -1;  // To hide the warning
  }
}

static inline int32_t GetQmax(const DataType& dtype) {
  CHECK_LE(dtype.bits(), 32)
      << "QNN ops support int32 or lower precision";
  if (dtype.is_int() || dtype.is_uint()) {
    auto* max_value = tir::as_const_int(tvm::max_value(dtype));
    CHECK(max_value != nullptr);
    return static_cast<int32_t>(max_value[0]);
  } else {
    LOG(FATAL) << "Type not supported " << dtype;
    return -1;  // To hide the warning
  }
}

Expr RequantizeLower(const Expr& input_tensor, const Expr& input_scale,
                     const Expr& input_zero_point, const Expr& output_scale,
                     const Expr& output_zero_point, const RequantizeAttrs* param,
                     const Array<IndexExpr>& input_shape, const DataType& out_dtype);

static inline Expr Requantize(const Expr& data, const Array<IndexExpr>& input_shape,
                              const Expr& input_scale, const Expr& input_zero_point,
                              const Expr& output_scale, const Expr& output_zero_point,
                              const DataType& out_dtype, const std::string& rounding = "UPWARD") {
  auto attrs = make_object<RequantizeAttrs>();
  attrs->rounding = std::move(rounding);
  attrs->out_dtype = std::move(out_dtype);
  return RequantizeLower(data, input_scale, input_zero_point, output_scale, output_zero_point,
                         attrs.operator->(), input_shape, attrs->out_dtype);
}

static inline int64_t get_const_int(const tvm::PrimExpr& x) {
  auto* value_ptr = tir::as_const_int(x);
  CHECK(value_ptr) << "Expr is not a constant int";
  return value_ptr[0];
}

/*
 * \brief Fixed point multiplication between integer tensor with floating point
 scalar.
 * \param tensor The quantized input tensor of dtype int64.
 * \param multiplier The scalar multiplier.
 * \param input_shape Shape of the input tensor.
 * \param rounding "UPWARD" or "TONEAREST". The rounding direction when the value
 is midway between" "two representable values.
 * \return The sequence of Relay ops for fixed point multiplication.

 * \note Original compuation is scale_fp32 * quantized_tensor.  To convert into
 *       integer computation, the multiplication with fp32 scalar can be
 *       replaced by multiplication with an int value and then right shifting
 *       the result. This approximates the floating point computation with a
 *       fixed point computation.
 *
 *       Computation of fixed point multiplication is consist of following
 steps:
 *       1) Multiply the fixed point multiplier with quantized tensor.
 *       2) Round the result.
 *       3) Right shift the result
 */
Expr FixedPointMultiply(Expr tensor, double multiplier, const Array<IndexExpr>& input_shape,
                        const std::string& rounding);

/*
 * \brief Fixed point multiplication between integer tensor with floating point
 scalar where the input tensor is per-axis/per-channel quantized..
 * \param tensor The quantized input tensor of dtype int64.
 * \param multiplier The scalar multiplier.
 * \param input_shape Shape of the input tensor.
 * \param channel_axis The channel_axis along which the input tensor is quantized. Default value is
 -1 which corresponds to the last channel_axis.
 * \param rounding "UPWARD" or "TONEAREST". The rounding direction when the value
 is midway between" "two representable values.
 * \return The sequence of Relay ops for fixed point multiplication.

 * \note Original compuation is scale_fp32 * quantized_tensor.  To convert into
 *       integer computation, the multiplication with fp32 vector can be
 *       replaced by multiplication with an int vector and then right shifting
 *       the result. This approximates the floating point computation with a
 *       fixed point computation.
 *
 *       Computation of fixed point multiplication is consist of following
 steps:
 *       1) Multiply the fixed point multiplier with quantized tensor.
 *       2) Round the result.
 *       3) Right shift the result
 */
Expr FixedPointMultiplyPerChannel(Expr tensor, std::vector<double> multiplier,
                                  const Array<IndexExpr>& input_shape, int channel_axis,
                                  const std::string& rounding);
/*
 * \brief Checks whether an expr type is scalar of a given data type.
 * \param expr_type The type of expr to be checked.
 * \param dtype The expected dtype.
 * \return True if the type is a scalar of given dtype
 */
static inline bool IsScalarType(const Type& expr_type, const DataType& dtype) {
  const auto* tensor_type = expr_type.as<TensorTypeNode>();
  CHECK(tensor_type) << "Only tensor type can be checked for scalar values. But got"
                     << AsText(expr_type, false);
  CHECK_EQ(tensor_type->shape.size(), 0);
  CHECK(tensor_type->dtype == dtype) << "Expected " << dtype << " but got " << tensor_type->dtype;
  return true;
}

/*
 * \brief Checks and assigns types to scale and zero points.
 * \param expr_type The type of expr to be checked.
 * \param dtype The expected dtype.
 * \param shape The shape at C dim of original tensor.
 * \param reporter The type reported of original InferType call.
 */
static inline void AssignType(const Type& expr_type, const DataType& dtype, const IndexExpr& shape,
                              const TypeReporter& reporter) {
  // Scale/Zero_points can be either const scalar or a vector with C axis num elems.
  const auto* tensor_type = expr_type.as<TensorTypeNode>();
  CHECK(tensor_type) << "Can assign type to Tensor type only. But got "
                     << AsText(expr_type, false);
  const auto tensor_dtype = tensor_type->dtype;
  CHECK(tensor_dtype == dtype) << "Expected type is " << dtype << " but received " << tensor_dtype;
  if (tensor_type->shape.size() != 0) {
    reporter->Assign(expr_type, TensorType({shape}, tensor_type->dtype));
  }
}

static inline std::vector<float> GetFloatVectorFromConstant(const Expr& expr) {
  const auto* n = expr.as<ConstantNode>();
  std::vector<float> vals;
  CHECK(n) << "Expr must be a constant expr - " << AsText(expr, false);
  int64_t num_elems = 1;
  auto shape = n->data.Shape();
  for (size_t i = 0; i < shape.size(); i++) {
    num_elems *= shape[i];
  }
  for (int64_t i = 0; i < num_elems; i++) {
    vals.push_back(static_cast<float*>(n->data->data)[i]);
  }
  return vals;
}

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_UTIL_H_
