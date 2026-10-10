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

/*! \file tvm/ir/prim/op.h
 * \brief Shared primitive construction and constant helpers.
 */
#ifndef TVM_IR_PRIM_OP_H_
#define TVM_IR_PRIM_OP_H_
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>
namespace tvm {
namespace prim {

/*!
 * \brief Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
 * (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
 */
TVM_DLL const Op& vscale_op();

/*! \brief Round up to the nearest integral value. */
TVM_DLL const Op& ceil_op();

/*! \brief Base-two logarithm. */
TVM_DLL const Op& log2_op();

/*! \brief Count leading zero bits. */
TVM_DLL const Op& clz_op();

/*!
 * \brief Same as select, used for unsafe memory access.
 *
 *  Type tvm_if_then_else(cond, a, b) {
 *    return cond ? a : b;
 *  }
 */
TVM_DLL const Op& if_then_else_op();

/*! \brief Marks a condition is likely going to happen. */
TVM_DLL const Op& likely_op();

}  // namespace prim

/*!
 * \brief add operator
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr add(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief subtraction operator
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr sub(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief negation.
 *
 * \param a input.
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr neg(PrimExpr a, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief multiplication operator
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr mul(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief left shift operator
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr left_shift(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief right shift operator
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr right_shift(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief greater
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr greater(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief greater_equal
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr greater_equal(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief less
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr less(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief less_equal
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr less_equal(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief equal
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr equal(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief not_equal
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr not_equal(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief and
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr logical_and(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief or
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr logical_or(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief not
 *
 * \param a left operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr logical_not(PrimExpr a, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute division in C semantics.
 *
 * a / b as in C/C++.
 *
 * When operands are integers, it directly corresponds to truncdiv.
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr div(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute trunc(a / b)
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr truncdiv(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute the remainder of truncdiv
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr truncmod(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute floor(a / b) where a and b are non-negative.
 *
 * Use this function for index split calculation.
 *
 * This function might take advantage of the fact
 * that a and b are non-negative.
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr indexdiv(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute ceil(a / b) where a and b are non-negative.
 *
 * Use this function for shape split calculation.
 *
 * This function might take advantage of the fact
 * that a and b are non-negative.
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       shape types(int32, int64) when possible.
 */
TVM_DLL PrimExpr shapediv(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute the remainder floor(a / b) where a and b are non-negative.
 *
 * Use this function for index split calculation.
 * This function might take advantage of the fact
 * that a and b are non-negative.
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr indexmod(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute floor(a / b)
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr floordiv(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute ceil(a / b)
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */

TVM_DLL PrimExpr ceildiv(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief compute the remainder of floordiv
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr floormod(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief take maximum of two values
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr max(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief take minimum of two values
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr min(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief take bitwise and of two values
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_and(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief take bitwise or of two values
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_or(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief take bitwise xor of two values
 *
 * \param a left operand
 * \param b right operand
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief take bitwise negation of two values
 *
 * \param a the input expression.
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_neg(PrimExpr a, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Conditional expression.
 *
 * \param cond The condition
 * \param true_value The value when results are true.
 * \param false_value The value when results are false.
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value,
                              ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Mark condition as likely.
 * \param cond The condition
 * \param loc The location of this operation in the source.
 * \return The marked expression.
 */
TVM_DLL PrimExpr likely(PrimExpr cond, ffi::Optional<Location> loc = std::nullopt);
/*! \brief Round up to the nearest integral value, preserving the input type. */
TVM_DLL PrimExpr ceil(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);
/*! \brief Construct a base-two logarithm with the input's primitive type. */
TVM_DLL PrimExpr log2(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);
namespace prim {
/*! \brief Count leading zero bits, preserving the input primitive type. */
TVM_DLL PrimExpr clz(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);
/*!
 * Query the minimum possible value of dtype.
 * \param dtype The primitive type.
 * \param loc The location of this operation in the source.
 * \return the minimum possible value in this format.
 */
TVM_DLL PrimExpr min_value(PrimType dtype, ffi::Optional<Location> loc = std::nullopt);

/*!
 * Query the maximum possible value of dtype.
 * \param dtype The primitive type.
 * \param loc The location of this operation in the source.
 * \return the maximum possible value in this format.
 */
TVM_DLL PrimExpr max_value(PrimType dtype, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \param loc The location of this operation in the source.
 * \return The result expression.
 * \note This function may return value if the type is the same. Constant folding
 * uses MakeConst for scalar and vector constants.
 */
TVM_DLL PrimExpr cast(PrimType t, PrimExpr value, ffi::Optional<Location> loc = std::nullopt);
TVM_DLL PrimExpr cast(DLDataType t, PrimExpr value, ffi::Optional<Location> loc = std::nullopt);
/*! \brief Construct integer absolute value; floating absolute value belongs to TIRX. */
TVM_DLL PrimExpr IntegerAbs(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);
/*!
 * \brief Make a const value with certain data type.
 *
 * Prefer direct IntImm or FloatImm construction when dtype is known to be
 * scalar integer or floating point. This makes the compiled code more compact
 * and efficient. Keep MakeConst for generic overload cases where dtype can be
 * integer, floating point, or vector-valued and the caller needs its
 * scalar/vector dispatch. Integer payloads use the exact BigInt representation.
 *
 * \param dtype The target type.
 * \param value The input value
 * \return the result expression.
 * \tparam ValueType The constant value type
 * \param loc The location of this operation in the source.
 */
template <typename ValueType,
          typename = typename std::enable_if<(std::is_standard_layout<ValueType>::value &&
                                              std::is_trivial<ValueType>::value) ||
                                             std::is_same<ValueType, ffi::BigInt>::value>::type>
inline PrimExpr MakeConst(PrimType dtype, ValueType value,
                          ffi::Optional<Location> loc = std::nullopt);

template <typename ValueType>
inline PrimExpr MakeConstScalar(PrimType dtype, ValueType value,
                                ffi::Optional<Location> loc = std::nullopt) {
  if constexpr (std::is_enum_v<ValueType>) {
    return MakeConstScalar(dtype, static_cast<std::underlying_type_t<ValueType>>(value), loc);
  } else {
    DLDataTypeCode code = dtype.code();
    if (code == DLDataTypeCode::kDLInt || code == DLDataTypeCode::kDLBool) {
      return IntImm(dtype, ffi::BigInt(value), loc);
    }
    if (code == DLDataTypeCode::kDLUInt) {
      TVM_FFI_ICHECK(value >= 0) << "cannot make uint from negative value " << value;
      return IntImm(dtype, ffi::BigInt(value), loc);
    }
    if (dtype.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLFloat8_e3m4,
                          DLDataTypeCode::kDLFloat8_e4m3, DLDataTypeCode::kDLFloat8_e4m3b11fnuz,
                          DLDataTypeCode::kDLFloat8_e4m3fn, DLDataTypeCode::kDLFloat8_e4m3fnuz,
                          DLDataTypeCode::kDLFloat8_e5m2, DLDataTypeCode::kDLFloat8_e5m2fnuz,
                          DLDataTypeCode::kDLFloat8_e8m0fnu, DLDataTypeCode::kDLFloat6_e2m3fn,
                          DLDataTypeCode::kDLFloat6_e3m2fn, DLDataTypeCode::kDLFloat4_e2m1fn) ||
        dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
      return FloatImm(dtype, static_cast<double>(value), loc);
    }
    TVM_FFI_THROW(InternalError) << "cannot make const for type " << dtype;
    throw;
  }
}

template <>
inline PrimExpr MakeConstScalar(PrimType dtype, bool value, ffi::Optional<Location> loc) {
  return MakeConstScalar(dtype, static_cast<int>(value), loc);
}

template <typename ValueType, typename>
inline PrimExpr MakeConst(PrimType dtype, ValueType value, ffi::Optional<Location> loc) {
  if (!dtype.IsScalableVector() && !dtype.IsFixedLengthVector()) {
    return MakeConstScalar(dtype, value, loc);
  }
  PrimType elem_ty = dtype.WithLanes(1);
  if (dtype.IsFixedLengthVector()) {
    return prim::Broadcast(MakeConstScalar(elem_ty, value, loc), dtype.lanes(), loc);
  }
  PrimExpr lanes = prim::Mul(Call(PrimType::Int(32), prim::vscale_op(), {}).as_or_throw<PrimExpr>(),
                             dtype.VScaleFactor());
  return prim::Broadcast(MakeConstScalar(elem_ty, value, loc), lanes, loc);
}

}  // namespace prim

// additional const expression overloading
#define TVM_DEFINE_ASSIGN_OP_OVERLOAD(Name, OpFunc) \
  inline PrimExpr Name(PrimExpr& a, PrimExpr b) {   \
    a = OpFunc(a, b);                               \
    return a;                                       \
  }

#define TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(Name)                                                \
  inline PrimExpr Name(const PrimExpr& a, float b) { return Name(a, PrimExpr(b)); }              \
  inline PrimExpr Name(float a, const PrimExpr& b) { return Name(PrimExpr(a), b); }              \
  inline PrimExpr Name(int a, const PrimExpr& b) { return Name(prim::MakeConst(b.ty(), a), b); } \
  inline PrimExpr Name(const PrimExpr& a, int b) { return Name(a, prim::MakeConst(a.ty(), b)); } \
  inline PrimExpr Name(const PrimExpr& a, double b) {                                            \
    return Name(a, FloatImm(PrimType::Float(64), b));                                            \
  }

#define TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(Name)                                         \
  inline PrimExpr Name(const PrimExpr& a, float b, ffi::Optional<Location> loc = std::nullopt) {  \
    return Name(a, PrimExpr(b), loc);                                                             \
  }                                                                                               \
  inline PrimExpr Name(float a, const PrimExpr& b, ffi::Optional<Location> loc = std::nullopt) {  \
    return Name(PrimExpr(a), b, loc);                                                             \
  }                                                                                               \
  inline PrimExpr Name(int a, const PrimExpr& b, ffi::Optional<Location> loc = std::nullopt) {    \
    return Name(prim::MakeConst(b.ty(), a), b, loc);                                              \
  }                                                                                               \
  inline PrimExpr Name(const PrimExpr& a, int b, ffi::Optional<Location> loc = std::nullopt) {    \
    return Name(a, prim::MakeConst(a.ty(), b), loc);                                              \
  }                                                                                               \
  inline PrimExpr Name(const PrimExpr& a, double b, ffi::Optional<Location> loc = std::nullopt) { \
    return Name(a, FloatImm(PrimType::Float(64), b), loc);                                        \
  }

#define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(Name)                             \
  inline PrimExpr Name(const PrimExpr& a, bool b) { return Name(a, PrimExpr(b)); } \
  inline PrimExpr Name(bool a, const PrimExpr& b) { return Name(PrimExpr(a), b); }

#define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD_SPANNED(Name)                                  \
  inline PrimExpr Name(const PrimExpr& a, bool b, ffi::Optional<Location> loc = std::nullopt) { \
    return Name(a, PrimExpr(b), loc);                                                           \
  }                                                                                             \
  inline PrimExpr Name(bool a, const PrimExpr& b, ffi::Optional<Location> loc = std::nullopt) { \
    return Name(PrimExpr(a), b, loc);                                                           \
  }

#define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(Name)                                               \
  inline PrimExpr Name(const PrimExpr& a, int b) { return Name(a, prim::MakeConst(a.ty(), b)); } \
  inline PrimExpr Name(int a, const PrimExpr& b) { return Name(prim::MakeConst(b.ty(), a), b); }

#define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(Name)                                     \
  inline PrimExpr Name(const PrimExpr& a, int b, ffi::Optional<Location> loc = std::nullopt) { \
    return Name(a, prim::MakeConst(a.ty(), b), loc);                                           \
  }                                                                                            \
  inline PrimExpr Name(int a, const PrimExpr& b, ffi::Optional<Location> loc = std::nullopt) { \
    return Name(prim::MakeConst(b.ty(), a), b, loc);                                           \
  }

TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator+=, operator+);
TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator-=, operator-);
TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator*=, operator*);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator+);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator-);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator*);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>);  // NOLINT(*)
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>=);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<);  // NOLINT(*)
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<=);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(max);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(min);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(div);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(add);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(sub);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(mul);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(greater);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(greater_equal);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(less);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(less_equal);
// integer related ops
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(indexdiv);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(indexmod);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(truncdiv);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(truncmod);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(floordiv);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(floormod);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(right_shift);  // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(left_shift);   // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(bitwise_and);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(bitwise_or);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(bitwise_xor);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator>>);  // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator<<);  // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator&);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator|);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator^);
// logical ops
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator&&);
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator||);
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD_SPANNED(logical_and);
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD_SPANNED(logical_or);

/*!
 * \brief Helper function to raise a compiler error about division ambiguity.
 * \note The call to this function will always results in a compiler error.
 * \tparam TA Any class type.
 */
template <typename TA>
inline void DivAmbiguityError(const TA& a) {
  constexpr bool div_ambiguity = !std::is_class<TA>::value;
  static_assert(div_ambiguity,
                "TVM supports multiple types of integer divisions, "
                "please call div, indexdiv/indexmod, "
                "floordiv/floormod or truncdiv/truncmod directly "
                "to avoid ambiguity in the code. "
                "Checkout these functions in ir/prim/op.h.");
}

// The following code are not intended to be used in the codebase.
// Instead, they generate clear compiler errors that ask developers
// to use the specific division function.
// The second template argument is necessary to make sure the
// code compiles lazily by the compiler during invocation.
template <typename TB>
inline PrimExpr operator/(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

template <typename TB>
inline PrimExpr operator/=(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

template <typename TB>
inline PrimExpr operator%(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

// Constant predicates
namespace prim {

/*!
 * \brief Check whether x is a constant integer expression.
 * \param x The input argument
 * \param value the value to be compared against.
 * \return whether x is constant expression.
 */
inline bool IsConstInt(const PrimExpr& x, int64_t value);

/*!
 * \brief Check whether x is a constant integer 1
 * \param x The input argument.
 * \note This only returns true for integer types.
 * \return whether x is constant 1
 */
inline bool IsOne(const PrimExpr& x);

/*!
 * \brief Check whether x is a constant integer 0
 * \param x The input argument
 * \return whether x is constant 0
 * \note This only returns true for integer types.
 */
inline bool IsZero(const PrimExpr& x);

/*!
 * \brief Check whether x is an integer constant, including wide IntImm values.
 * \return whether x is constant
 */
inline bool IsConstInt(const PrimExpr& x);

/*!
 * \brief Check whether x is an integer or floating-point constant.
 * \note Also recognizes a Broadcast containing an integer or floating-point constant.
 * \return whether x is constant
 */
inline bool IsConstNumber(const PrimExpr& x);

/*!
 * \brief Check whether x is a constant power of two
 * If x is power of two, write the power to the shift.
 *
 * \param x The input expression.
 * \param shift The output shift if x is power of two.
 * \return whether x is constant power of two
 */
TVM_DLL bool IsPowerOfTwoInt(const PrimExpr& x, int* shift);

/*! \brief Check whether a is a positive integer constant. */
inline bool IsPositiveConst(const PrimExpr& a);

/*! \brief Check whether a is a negative integer constant. */
inline bool IsNegativeConst(const PrimExpr& a);

// Inline constant predicate implementations.
inline bool IsConstInt(const PrimExpr& x) { return x.as<IntImmNode>() != nullptr; }

inline bool IsConstNumber(const PrimExpr& x) {
  if (x.as<IntImmNode>()) {
    return true;
  } else if (x.as<FloatImmNode>()) {
    return true;
  } else if (const auto* op = x.as<prim::BroadcastNode>()) {
    return (op->value->IsInstance<IntImmNode>() || op->value->IsInstance<FloatImmNode>());
  }
  return false;
}

inline bool IsPositiveConst(const PrimExpr& a) {
  const auto* as_int = a.as<IntImmNode>();
  return as_int && as_int->value > 0;
}

inline bool IsNegativeConst(const PrimExpr& a) {
  const auto* as_int = a.as<IntImmNode>();
  return as_int && as_int->value < 0;
}

inline bool IsConstInt(const PrimExpr& x, int64_t value) {
  const auto* as_int = x.as<IntImmNode>();
  return as_int && as_int->value == value;
}

inline bool IsOne(const PrimExpr& x) { return IsConstInt(x, 1); }

inline bool IsZero(const PrimExpr& x) { return IsConstInt(x, 0); }

/*!
 * Get the value of infinity.
 * \param dtype The primitive type.
 * \param loc The location of this operation in the source.
 * \return the infinity value in this format.
 */
TVM_DLL PrimExpr infinity(PrimType dtype, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Calculate power(x, y)
 * \param x The left operand.
 * \param y The right operand.
 * \param loc The location of this operation in the source.
 */
TVM_DLL PrimExpr pow(PrimExpr x, PrimExpr y, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Calculate absolute value of x.
 * \param x The input data
 * \param loc The location of this operation in the source.
 *
 * \return The absolute value of input data x
 */
TVM_DLL PrimExpr abs(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Check if x is NaN.
 * \param x The input data
 * \param loc The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isnan(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Check if x is finite.
 * \param x The input data
 * \param loc The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isfinite(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Check if x is infinite.
 * \param x The input data
 * \param loc The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr isinf(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Calculate floor(x)
 * \param x The input expression.
 * \param loc The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr floor(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Round x to the nearest integer, ties to even.
 *
 * Uses IEEE 754 default rounding mode (ties-to-even / banker's rounding).
 * Constant-folding and all backends consistently use std::nearbyint semantics.
 *
 * \param x The input expression.
 * \param loc The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr round(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Round x to the nearest integer, ties to even.
 *
 * Equivalent to round(). Both use IEEE 754 default rounding mode (ties-to-even).
 *
 * \param x The input expression.
 * \param loc The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr nearbyint(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*!
 * \brief Calculate trunc(x)
 * \param x The input expression.
 * \param loc The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr trunc(PrimExpr x, ffi::Optional<Location> loc = std::nullopt);

/*! \brief Floating-point remainder of x divided by y. */
TVM_DLL PrimExpr fmod(PrimExpr x, PrimExpr y, ffi::Optional<Location> loc = std::nullopt);
/*! \brief Raise x to the power y. Arguments: x, y. */
TVM_DLL const Op& pow_op();
/*! \brief Absolute value. Argument: x. */
TVM_DLL const Op& fabs_op();
/*! \brief Floating-point remainder. Arguments: x, y. */
TVM_DLL const Op& fmod_op();
/*! \brief Round toward negative infinity. Argument: x. */
TVM_DLL const Op& floor_op();
/*! \brief Round to nearest integer, ties to even. Argument: x. */
TVM_DLL const Op& round_op();
/*! \brief Round to nearest integer, ties to even. Argument: x. */
TVM_DLL const Op& nearbyint_op();
/*! \brief Round toward zero. Argument: x. */
TVM_DLL const Op& trunc_op();
/*! \brief Natural exponential. Argument: x. */
TVM_DLL const Op& exp_op();
/*! \brief Base-two exponential. Argument: x. */
TVM_DLL const Op& exp2_op();
/*! \brief Base-ten exponential. Argument: x. */
TVM_DLL const Op& exp10_op();
/*! \brief Error function. Argument: x. */
TVM_DLL const Op& erf_op();
/*! \brief Hyperbolic tangent. Argument: x. */
TVM_DLL const Op& tanh_op();
/*! \brief Logistic sigmoid. Argument: x. */
TVM_DLL const Op& sigmoid_op();
/*! \brief Square root. Argument: x. */
TVM_DLL const Op& sqrt_op();
/*! \brief Reciprocal square root. Argument: x. */
TVM_DLL const Op& rsqrt_op();
/*! \brief Natural logarithm. Argument: x. */
TVM_DLL const Op& log_op();
/*! \brief Natural logarithm of one plus x. Argument: x. */
TVM_DLL const Op& log1p_op();
/*! \brief Base-ten logarithm. Argument: x. */
TVM_DLL const Op& log10_op();
/*! \brief Tangent. Argument: x. */
TVM_DLL const Op& tan_op();
/*! \brief Cosine. Argument: x. */
TVM_DLL const Op& cos_op();
/*! \brief Hyperbolic cosine. Argument: x. */
TVM_DLL const Op& cosh_op();
/*! \brief Sine. Argument: x. */
TVM_DLL const Op& sin_op();
/*! \brief Hyperbolic sine. Argument: x. */
TVM_DLL const Op& sinh_op();
/*! \brief Inverse sine. Argument: x. */
TVM_DLL const Op& asin_op();
/*! \brief Inverse cosine. Argument: x. */
TVM_DLL const Op& acos_op();
/*! \brief Inverse tangent. Argument: x. */
TVM_DLL const Op& atan_op();
/*! \brief Inverse hyperbolic cosine. Argument: x. */
TVM_DLL const Op& acosh_op();
/*! \brief Inverse hyperbolic sine. Argument: x. */
TVM_DLL const Op& asinh_op();
/*! \brief Inverse hyperbolic tangent. Argument: x. */
TVM_DLL const Op& atanh_op();
/*! \brief Inverse tangent of y/x with quadrant selection. Arguments: y, x. */
TVM_DLL const Op& atan2_op();
/*! \brief Next representable value from x toward y. Arguments: x, y. */
TVM_DLL const Op& nextafter_op();
/*! \brief Euclidean length of x and y. Arguments: x, y. */
TVM_DLL const Op& hypot_op();
/*! \brief Magnitude of x with the sign of y. Arguments: x, y. */
TVM_DLL const Op& copysign_op();
/*! \brief Multiply x by two raised to exponent. Arguments: x, exponent. */
TVM_DLL const Op& ldexp_op();
/*! \brief Check for NaN, preserving the lane count. Argument: x. */
TVM_DLL const Op& isnan_op();
/*! \brief Count set bits. Argument: x. */
TVM_DLL const Op& popcount_op();
/*! \brief Fused multiply-add, x * y + z. Arguments: x, y, z. */
TVM_DLL const Op& fma_op();
/*! \brief Record a known condition for simplification. Argument: condition. */
TVM_DLL const Op& assume_op();
/*! \brief Record a known condition for compile-time simplification. */
TVM_DLL PrimExpr assume(PrimExpr condition, ffi::Optional<Location> loc = std::nullopt);

/*! \brief Fused multiply-add of x, y and z, in that order. */
inline PrimExpr fma(PrimExpr x, PrimExpr y, PrimExpr z,
                    ffi::Optional<Location> loc = std::nullopt) {
  return Call(x.ty(), fma_op(), {x, y, z}, {}, {}, loc).as_or_throw<PrimExpr>();
}

inline void CheckMathUnaryOpInputDType(const char* op_name, const PrimType& dtype) {
  TVM_FFI_CHECK(dtype.code() == DLDataTypeCode::kDLFloat ||
                    dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16),
                TypeError)
      << "prim." << op_name << " only supports floating-point inputs, but got " << dtype;
}

// Intrinsic operators
#define TVM_DECLARE_INTRIN_UNARY_WITH_CHECK(OpName, CheckInputDType)                          \
  inline PrimExpr OpName(PrimExpr x, ffi::Optional<Location> loc = std::nullopt) {            \
    static const Op op = Op::Get("prim." #OpName);                                            \
    PrimType x_ty = x.ty();                                                                   \
    CheckInputDType(#OpName, x_ty);                                                           \
    if (x_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {                             \
      PrimType bf16_ty = x_ty;                                                                \
      PrimType f32_ty =                                                                       \
          x_ty.IsScalableVector()                                                             \
              ? PrimType::ScalableVector(DLDataTypeCode::kDLFloat, 32, x_ty.VScaleFactor())   \
              : PrimType::Float(32, x_ty.lanes());                                            \
      PrimExpr x_fp32 = prim::Cast(f32_ty, x, loc);                                           \
      PrimExpr result_fp32 = Call(f32_ty, op, {x_fp32}, {}, {}, loc).as_or_throw<PrimExpr>(); \
      return prim::Cast(bf16_ty, result_fp32, loc);                                           \
    } else {                                                                                  \
      return Call(x_ty, op, {x}, {}, {}, loc).as_or_throw<PrimExpr>();                        \
    }                                                                                         \
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

#define TVM_DECLARE_INTRIN_BINARY(OpName)                                                      \
  inline PrimExpr OpName(PrimExpr x, PrimExpr y, ffi::Optional<Location> loc = std::nullopt) { \
    static const Op op = Op::Get("prim." #OpName);                                             \
    return Call(x.ty(), op, {x, y}, {}, {}, loc).as_or_throw<PrimExpr>();                      \
  }

TVM_DECLARE_INTRIN_BINARY(atan2);
TVM_DECLARE_INTRIN_BINARY(nextafter);
TVM_DECLARE_INTRIN_BINARY(copysign);
TVM_DECLARE_INTRIN_BINARY(hypot);
TVM_DECLARE_INTRIN_BINARY(ldexp);

template <typename FReduce>
inline PrimExpr foldl(FReduce freduce, PrimExpr init_value, const ffi::Array<PrimExpr>& values,
                      ffi::Optional<Location> loc = std::nullopt) {
  for (PrimExpr val : values) {
    init_value = freduce(init_value, val, loc.value_or(Location()));
  }
  return init_value;
}

}  // namespace prim

}  // namespace tvm
#endif  // TVM_IR_PRIM_OP_H_
