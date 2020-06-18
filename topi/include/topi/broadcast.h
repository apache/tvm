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
 * \brief Broadcast op constructions
 * \file topi/broadcast.h
 */
#ifndef TOPI_BROADCAST_H_
#define TOPI_BROADCAST_H_

#include <topi/detail/broadcast.h>
#include <topi/detail/constant_utils.h>
#include <topi/tags.h>

#include <string>
#include <algorithm>

namespace topi {

/*!
 * \brief Creates an operation that broadcasts a tensor into a compatible
 * shape according to numpy's rules
 *
 * \param t The input tensor
 * \param output_shape The target output shape, must be compatible
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a broadcast operation
 */
inline tvm::te::Tensor broadcast_to(const tvm::te::Tensor& t,
                                    const tvm::Array<tvm::PrimExpr>& output_shape,
                                    std::string name = "T_broadcast_to",
                                    std::string tag = kBroadcast) {
  CHECK_GE(output_shape.size(), t->shape.size())
      << "Not a broadcast, output dimensionality smaller than input.\noutput: "
      << output_shape << "\nvs\ninput: " << t;
  auto bh = detail::BroadcastShape(output_shape, t->shape);
  CHECK_EQ(output_shape.size(), bh.common_shape.size());
  for (size_t i = 0; i < output_shape.size(); ++i) {
    CHECK(topi::detail::EqualCheck(output_shape[i], bh.common_shape[i]));
  }
  auto l = [&](tvm::Array<tvm::tir::Var> ovars) {
    return t(detail::InputIndexFromBroadcast(ovars, t, bh.vars2, bh.all_vars));
  };
  return tvm::te::compute(
      tvm::Array<tvm::PrimExpr>(bh.common_shape.begin(), bh.common_shape.end()),
      l,
      name,
      tag);
}

#define TOPI_DEFINE_BCAST_OP(Name, ComputeRule)                       \
  inline tvm::PrimExpr Name(const tvm::PrimExpr& a,                   \
                            const tvm::PrimExpr& b) {                 \
    ComputeRule;                                                      \
  }                                                                   \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A,               \
                              const tvm::te::Tensor& B,               \
                              std::string name = "T_" #Name,          \
                              std::string tag = kBroadcast) {         \
    auto l = [](tvm::PrimExpr a, tvm::PrimExpr b) { ComputeRule; };   \
    return detail::WithBroadcast(l, A, B, name, tag);                 \
  }                                                                   \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A,               \
                              const tvm::PrimExpr& B,                 \
                              std::string name = "T_" #Name,          \
                              std::string tag = kElementWise) {       \
    auto l = [](tvm::PrimExpr a, tvm::PrimExpr b) { ComputeRule; };   \
    return tvm::te::compute(A->shape, [&](const ::tvm::Array<::tvm::tir::Var>& i) { \
        return l(A(i), B);                                            \
      }, name, tag);                                                  \
  }                                                                   \
  inline tvm::te::Tensor Name(const tvm::PrimExpr& A,                 \
                              const tvm::te::Tensor& B,               \
                              std::string name = "T_" #Name,          \
                              std::string tag = kElementWise) {       \
    auto l = [&](tvm::PrimExpr a, tvm::PrimExpr b) { ComputeRule; };  \
    return tvm::te::compute(B->shape, [&](const ::tvm::Array<::tvm::tir::Var>& i) { \
        return l(A, B(i));                                            \
      }, name, tag);                                                  \
  }


#define TOPI_DEFINE_OP_OVERLOAD(Name, OpName)                       \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A,             \
                              const tvm::te::Tensor& B) {           \
    return topi::OpName(A, B);                                      \
  }                                                                 \
  inline tvm::te::Tensor Name(const tvm::PrimExpr& A,               \
                              const tvm::te::Tensor& B) {           \
    return topi::OpName(A, B);                                      \
  }                                                                 \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A,             \
                              const tvm::PrimExpr& B) {             \
    return topi::OpName(A, B);                                      \
  }

/*!
 * \fn logical_and
 * \brief Compute A && B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(logical_and, { return a && b; });
TOPI_DEFINE_OP_OVERLOAD(operator&&, logical_and);

/*!
 * \fn logical_or
 * \brief Compute A || B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(logical_or, { return a || b; });
TOPI_DEFINE_OP_OVERLOAD(operator||, logical_or);

/*!
 * \fn logical_xor
 * \brief Compute A ^ B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(logical_xor, { return a ^ b; });

/*!
 * \fn bitwise_and
 * \brief Compute A & B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(bitwise_and, { return a & b; });
TOPI_DEFINE_OP_OVERLOAD(operator&, bitwise_and);

/*!
 * \fn bitwise_or
 * \brief Compute A | B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(bitwise_or, { return a | b; });
TOPI_DEFINE_OP_OVERLOAD(operator|, bitwise_or);

/*!
 * \fn bitwise_xor
 * \brief Compute A ^ B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(bitwise_xor, { return a ^ b; });
TOPI_DEFINE_OP_OVERLOAD(operator^, bitwise_xor);

/*!
 * \fn add
 * \brief Compute A + B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(add, { return a + b; });
TOPI_DEFINE_OP_OVERLOAD(operator+, add);

/*!
 * \fn subtract
 * \brief Compute A - B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(subtract, { return a - b; });
TOPI_DEFINE_OP_OVERLOAD(operator-, subtract);

/*!
 * \fn multiply
 * \brief Compute A * B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(multiply, { return a * b; });
TOPI_DEFINE_OP_OVERLOAD(operator*, multiply);

/*!
 * \fn divide
 * \brief Compute A / B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(divide, { return div(a, b); });

/*!
 * \fn floor divide
 * \brief Compute floor(A / B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(floor_divide, {
  if (a.dtype().is_int() || a.dtype().is_uint()) {
    return floordiv(a, b);
  } else {
    return floor(div(a, b));
  }
});

/*!
 * \fn mod
 * \brief Compute A % B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(mod, { return truncmod(a, b); });

/*!
 * \fn floor mod
 * \brief Compute A - floor_div(A, B) * B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(floor_mod, {
  if (a.dtype().is_int() || a.dtype().is_uint()) {
    return floormod(a, b);
  } else {
    return a - floor_divide(a, b) * b;
  }
});

/*!
 * \fn maximum
 * \brief Compute maximum(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(maximum, { return tvm::max(a, b); });

/*!
 * \fn minimum
 * \brief Compute minimum(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(minimum, { return tvm::min(a, b); });

/*!
 * \fn power
 * \brief Compute power(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(power, { return tvm::pow(a, b); });

/*!
 * \fn left_shift
 * \brief Compute A << B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(left_shift, { return a << b; });
TOPI_DEFINE_OP_OVERLOAD(operator<<, left_shift);

/*!
 * \fn right_shift
 * \brief Compute A >> B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(right_shift, { return a >> b; });
TOPI_DEFINE_OP_OVERLOAD(operator>>, right_shift);

/*!
 * \fn greater
 * \brief Compute (A > B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(greater, { return (a > b); });

/*!
 * \fn less
 * \brief Compute (A < B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(less, { return (a < b); });

/*!
 * \fn equal
 * \brief Compute (A == B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(equal, { return (a == b); });

/*!
 * \fn not_equal
 * \brief Compute (A != B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(not_equal, { return (a != b); });

/*!
 * \fn greater_equal
 * \brief Compute (A >= B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(greater_equal, { return (a >= b); });

/*!
 * \fn less_equal
 * \brief Compute (A <= B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(less_equal, { return (a <= b); });

}  // namespace topi

#endif  // TOPI_BROADCAST_H_
