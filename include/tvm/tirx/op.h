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
 * \file tvm/tirx/op.h
 * \brief Common operators defined for Expr.
 *
 * \note Most of the operator defined here perform simple constant folding
 *   when the type is int32 or int64 for simplifying the index expressions.
 */
// Acknowledgement: Most operator APIs originate from Halide.
#ifndef TVM_TIRX_OP_H_
#define TVM_TIRX_OP_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

namespace tvm::prim {

#define TVM_TIR_REGISTER_OP(OpName)                               \
  TVM_REGISTER_OP("tirx." OpName)                                 \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", OpName) \
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"), /*plevel=*/1)

#define TVM_TIRX_REGISTER_OP(OpName) TVM_TIR_REGISTER_OP(OpName)

// Shared primitive construction and constants are declared in ir/prim/op.h.

/*!
 * \brief Get the type of the expression under the unified type system.
 *
 * This function could return a more refined type than the runtime dtype
 * implied by PrimExpr::ty().
 *
 * \param expr The input parameter.
 * \return The result type.
 *
 * \sa tvm/ir/type.h for discussion about the relation between Type and DLPack dtype.
 */
TVM_DLL Type GetType(const PrimExpr& expr);

/*!
 * \brief Get the type corresponding to a runtime DLPack dtype.
 * \param dtype The runtime dtype.
 * \return The result type
 *
 * \sa tvm/ir/type.h for discussion about the relation between Type and DLPack dtype.
 */
TVM_DLL Type GetTypeFromRuntimeDataType(DLDataType dtype);

/*!
 * Get the value of infinity.
 * \param dtype The primitive type.
 * \param span The location of this operation in the source.
 * \return the infinity value in this format.
 */
TVM_DLL PrimExpr infinity(PrimType dtype, Span span = Span());

/*!
 * \brief perform reinterpret cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
TVM_DLL PrimExpr reinterpret(PrimType t, PrimExpr value, Span span = Span());
TVM_DLL PrimExpr reinterpret(DLDataType t, PrimExpr value, Span span = Span());
/*! \brief Perform a reinterpret cast involving an exact primitive or pointer type. */
TVM_DLL Expr reinterpret(Type target_ty, Expr value, Span span = Span());

/*!
 * \brief Compute log(exp(a) + exp(b)).
 *
 * \param a Left operand.
 * \param b Right operand.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr logaddexp(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief Calculate power(x, y)
 * \param x The left operand.
 * \param y The right operand.
 * \param span The location of this operation in the source.
 */
TVM_DLL PrimExpr pow(PrimExpr x, PrimExpr y, Span span = Span());
/*!
 * \brief Calculate absolute value of x.
 * \param x The input data
 * \param span The location of this operation in the source.
 *
 * \return The absolute value of input data x
 */
TVM_DLL PrimExpr abs(PrimExpr x, Span span = Span());
/*!
 * \brief Check if x is NaN.
 * \param x The input data
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isnan(PrimExpr x, Span span = Span());

/*!
 * \brief Check if x is finite.
 * \param x The input data
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isfinite(PrimExpr x, Span span = Span());

/*!
 * \brief Check if x is infinite.
 * \param x The input data
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isinf(PrimExpr x, Span span = Span());

/*!
 * \brief sum of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param span The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr sum(PrimExpr source, ffi::Array<tirx::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Span span = Span());

/*!
 * \brief logical And of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param span The location of this operation in the source.
 */
TVM_DLL PrimExpr all(PrimExpr source, ffi::Array<tirx::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Span span = Span());

/*!
 * \brief logical Or of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param span The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr any(PrimExpr source, ffi::Array<tirx::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Span span = Span());

/*!
 * \brief product of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param span The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr prod(PrimExpr source, ffi::Array<tirx::IterVar> axis,
                      ffi::Array<PrimExpr> init = {}, Span span = Span());

/*!
 * \brief Calculate floor(x)
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr floor(PrimExpr x, Span span = Span());

/*!
 * \brief Round x to the nearest integer, ties to even.
 *
 * Uses IEEE 754 default rounding mode (ties-to-even / banker's rounding).
 * Constant-folding and all backends consistently use std::nearbyint semantics.
 *
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr round(PrimExpr x, Span span = Span());

/*!
 * \brief Round x to the nearest integer, ties to even.
 *
 * Equivalent to round(). Both use IEEE 754 default rounding mode (ties-to-even).
 *
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr nearbyint(PrimExpr x, Span span = Span());

/*!
 * \brief Calculate trunc(x)
 * \param x The input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr trunc(PrimExpr x, Span span = Span());

/*!
 * \brief Execute a multiplication between two Q-numbers x and y
 * followed by a right shift s. The mathematical expression is:
 *
 *    out = round(x*y*2^-s)
 *
 * Please note that the two Q-numbers x and y are supposed to have
 * the same number of fractional bits q.
 *
 * More about Q-numbers here: https://en.wikipedia.org/wiki/Q_(number_format)
 *
 * The rounding rule is to the nearest value, rounding half up
 * (i.e., round(x.1) = x and round (x.5) = x+1)
 * \param x first Q-number
 * \param y second Q-number
 * \param q number of fractional bits in x and y. Needs to be > 0
 * \param s integer right shift
 * \param span The location of this operation in the source.
 * \return The constructed expression.
 */
TVM_DLL PrimExpr q_multiply_shift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr s,
                                  Span span = Span());

/*!
 * \brief Fast_erf_float expression from Eigen
 *
 * \param arg The input expression.
 * \param bits The number of bits in the type.
 * \return The constructed expression.
 */
TVM_DLL PrimExpr fast_erf_float_expr(PrimExpr arg, int bits);

inline void CheckMathUnaryOpInputDType(const char* op_name, const PrimType& dtype) {
  TVM_FFI_CHECK(dtype.code() == DLDataTypeCode::kDLFloat ||
                    dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16),
                TypeError)
      << "tirx." << op_name << " only supports floating-point inputs, but got " << dtype;
}

// Intrinsic operators
#define TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, CheckInputDType)                           \
  inline PrimExpr OpName(PrimExpr x, Span span = Span()) {                                     \
    static const Op op = Op::Get("tirx." #OpName);                                             \
    PrimType x_ty = x.ty();                                                                    \
    CheckInputDType(#OpName, x_ty);                                                            \
    if (x_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {                              \
      PrimType bf16_ty = x_ty;                                                                 \
      PrimType f32_ty =                                                                        \
          x_ty.IsScalableVector()                                                              \
              ? PrimType::ScalableVector(DLDataTypeCode::kDLFloat, 32, x_ty.VScaleFactor())    \
              : PrimType::Float(32, x_ty.lanes());                                             \
      PrimExpr x_fp32 = prim::Cast(f32_ty, x, span);                                           \
      PrimExpr result_fp32 = Call(f32_ty, op, {x_fp32}, {}, {}, span).as_or_throw<PrimExpr>(); \
      return prim::Cast(bf16_ty, result_fp32, span);                                           \
    } else {                                                                                   \
      return Call(x_ty, op, {x}, {}, {}, span).as_or_throw<PrimExpr>();                        \
    }                                                                                          \
  }

#define TVM_DECLARE_INTRIN_UNARY(OpName) \
  TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, [](const char*, const PrimType&) {})

#define TVM_DECLARE_FLOAT_INTRIN_UNARY(OpName) \
  TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, CheckMathUnaryOpInputDType)

TVM_DECLARE_INTRIN_UNARY(exp);
TVM_DECLARE_INTRIN_UNARY(exp2);
TVM_DECLARE_INTRIN_UNARY(exp10);
TVM_DECLARE_INTRIN_UNARY(erf);
TVM_DECLARE_FLOAT_INTRIN_UNARY(tanh);
TVM_DECLARE_INTRIN_UNARY(sigmoid);
TVM_DECLARE_INTRIN_UNARY(sqrt);
TVM_DECLARE_INTRIN_UNARY(rsqrt);
TVM_DECLARE_INTRIN_UNARY(log);
TVM_DECLARE_INTRIN_UNARY(log10);
TVM_DECLARE_INTRIN_UNARY(log1p);
TVM_DECLARE_INTRIN_UNARY(popcount);
TVM_DECLARE_FLOAT_INTRIN_UNARY(tan);
TVM_DECLARE_FLOAT_INTRIN_UNARY(cos);
TVM_DECLARE_FLOAT_INTRIN_UNARY(cosh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(sin);
TVM_DECLARE_FLOAT_INTRIN_UNARY(sinh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(asin);
TVM_DECLARE_FLOAT_INTRIN_UNARY(acos);
TVM_DECLARE_FLOAT_INTRIN_UNARY(atan);
TVM_DECLARE_FLOAT_INTRIN_UNARY(acosh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(asinh);
TVM_DECLARE_FLOAT_INTRIN_UNARY(atanh);

#define TVM_DECLARE_INTRIN_BINARY(OpName)                                  \
  inline PrimExpr OpName(PrimExpr x, PrimExpr y, Span span = Span()) {     \
    static const Op op = Op::Get("tirx." #OpName);                         \
    return Call(x.ty(), op, {x, y}, {}, {}, span).as_or_throw<PrimExpr>(); \
  }

TVM_DECLARE_INTRIN_BINARY(atan2);
TVM_DECLARE_INTRIN_BINARY(nextafter);
TVM_DECLARE_INTRIN_BINARY(copysign);
TVM_DECLARE_INTRIN_BINARY(hypot);
TVM_DECLARE_INTRIN_BINARY(ldexp);

/*!
 * \brief Check if type is a pointer to a runtime element type.
 * \param type The type to be checked.
 * \param element_type The corresponding element type.
 * \return The check results
 */
inline bool IsPointerType(const Type& type, DLDataType element_type) {
  if (type.IsMissing()) return false;
  if (const auto* ptr_type = type.as<PointerTypeNode>()) {
    if (const auto* prim_type = ptr_type->element_type.as<PrimTypeNode>()) {
      return prim_type->dtype == element_type;
    }
  }
  return false;
}

/*!
 * \brief Make a constant opaque-pointer value.
 * \param value The integer payload to reinterpret as a handle.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
inline Expr ConstHandle(int64_t value, Span span = Span());

/*!
 * \brief Check whether stmt is nop.
 * \param stmt The input statement
 * \return whether stmt is nop
 */
inline bool is_no_op(const tirx::Stmt& stmt);

/*!
 * \brief Left fold.
 * \param freduce The reduction function.
 * \param init_value The initial value.
 * \param values The values to be folded.
 * \param span The location of the fold in the source.
 * \return The result.
 * \tparam FReduce The type of the reduction.
 */
template <typename FReduce>
inline PrimExpr foldl(FReduce freduce, PrimExpr init_value, const ffi::Array<PrimExpr>& values,
                      Span span = Span()) {
  for (PrimExpr val : values) {
    init_value = freduce(init_value, val, span);
  }
  return init_value;
}

inline bool is_no_op(const tirx::Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (const auto* op = stmt.as<tirx::EvaluateNode>()) {
    auto value = op->value.as<PrimExpr>();
    return value && is_const_int(value.value());
  }
  if (const auto* op = stmt.as<tirx::SeqStmtNode>()) {
    return op->seq.size() == 0;
  }
  return false;
}

inline Expr ConstHandle(int64_t value, Span span) {
  return reinterpret(PointerType::VoidPointerTy(), IntImm(PrimType::UInt(64), value, span), span);
}

TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(logaddexp);
}  // namespace tvm::prim

namespace tvm {
/*!
 * \brief max of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param span The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr max(PrimExpr source, ffi::Array<tirx::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Span span = Span());

/*!
 * \brief max of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 * \param init The value with which to initialize the output.
 * \param span The location of this operation in the source.
 * \return The result.
 */
TVM_DLL PrimExpr min(PrimExpr source, ffi::Array<tirx::IterVar> axis,
                     ffi::Array<PrimExpr> init = {}, Span span = Span());

}  // namespace tvm
#endif  // TVM_TIR_OP_H_
