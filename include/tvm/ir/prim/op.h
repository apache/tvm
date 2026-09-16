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
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>
namespace tvm {
/*!
 * \brief add operator
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr add(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief subtraction operator
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr sub(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief negation.
 *
 * \param a input.
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr neg(PrimExpr a, Span span = Span());

/*!
 * \brief multiplication operator
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr mul(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief left shift operator
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr left_shift(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief right shift operator
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr right_shift(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief greater
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr greater(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief greater_equal
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr greater_equal(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief less
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr less(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief less_equal
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr less_equal(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief equal
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr equal(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief not_equal
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr not_equal(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief and
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr logical_and(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief or
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr logical_or(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief not
 *
 * \param a left operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr logical_not(PrimExpr a, Span span = Span());

/*!
 * \brief compute division in C semantics.
 *
 * a / b as in C/C++.
 *
 * When operands are integers, it directly corresponds to truncdiv.
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr div(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief compute trunc(a / b)
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr truncdiv(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief compute the remainder of truncdiv
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr truncmod(PrimExpr a, PrimExpr b, Span span = Span());

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
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr indexdiv(PrimExpr a, PrimExpr b, Span span = Span());

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
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       shape types(int32, int64) when possible.
 */
TVM_DLL PrimExpr shapediv(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief compute the remainder floor(a / b) where a and b are non-negative.
 *
 * Use this function for index split calculation.
 * This function might take advantage of the fact
 * that a and b are non-negative.
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr indexmod(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief compute floor(a / b)
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr floordiv(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief compute ceil(a / b)
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */

TVM_DLL PrimExpr ceildiv(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief compute the remainder of floordiv
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr floormod(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief take maximum of two values
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr max(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief take minimum of two values
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr min(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief take bitwise and of two values
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_and(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief take bitwise or of two values
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_or(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief take bitwise xor of two values
 *
 * \param a left operand
 * \param b right operand
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, Span span = Span());

/*!
 * \brief take bitwise negation of two values
 *
 * \param a the input expression.
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr bitwise_neg(PrimExpr a, Span span = Span());

/*!
 * \brief Conditional expression.
 *
 * \param cond The condition
 * \param true_value The value when results are true.
 * \param false_value The value when results are false.
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value,
                              Span span = Span());

/*!
 * \brief Mark condition as likely.
 * \param cond The condition
 * \param span The location of this operation in the source.
 * \return The marked expression.
 */
TVM_DLL PrimExpr likely(PrimExpr cond, Span span = Span());
/*! \brief Round up to the nearest integral value, preserving the input type. */
TVM_DLL PrimExpr ceil(PrimExpr x, Span span = Span());
/*! \brief Construct a base-two logarithm with the input's primitive type. */
TVM_DLL PrimExpr log2(PrimExpr x, Span span = Span());
namespace prim {
/*!
 * Query the minimum possible value of dtype.
 * \param dtype The primitive type.
 * \param span The location of this operation in the source.
 * \return the minimum possible value in this format.
 */
TVM_DLL PrimExpr min_value(PrimType dtype, Span span = Span());

/*!
 * Query the maximum possible value of dtype.
 * \param dtype The primitive type.
 * \param span The location of this operation in the source.
 * \return the maximum possible value in this format.
 */
TVM_DLL PrimExpr max_value(PrimType dtype, Span span = Span());

/*!
 * \brief cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \param span The location of this operation in the source.
 * \return The result expression.
 * \note This function may return value if the type is the same. Constant folding
 * uses MakeConst for scalar and vector constants.
 */
TVM_DLL PrimExpr cast(PrimType t, PrimExpr value, Span span = Span());
TVM_DLL PrimExpr cast(DLDataType t, PrimExpr value, Span span = Span());
/*! \brief Construct integer absolute value; floating absolute value belongs to TIRX. */
TVM_DLL PrimExpr IntegerAbs(PrimExpr x, Span span = Span());
/*!
 * \brief Make a const value with certain data type.
 *
 * Prefer direct IntImm or FloatImm construction when dtype is known to be
 * scalar integer or floating point. This makes the compiled code more compact
 * and efficient. Keep MakeConst for generic overload cases where dtype can be
 * integer, floating point, or vector-valued and the caller needs its
 * scalar/vector dispatch.
 *
 * \param dtype The target type.
 * \param value The input value
 * \return the result expression.
 * \tparam ValueType The constant value type
 * \param span The location of this operation in the source.
 */
template <typename ValueType,
          typename = typename std::enable_if<std::is_standard_layout<ValueType>::value &&
                                             std::is_trivial<ValueType>::value>::type>
inline PrimExpr MakeConst(PrimType dtype, ValueType value, Span span = Span());

/*!
 * \brief Get x as constant int expression.
 * \param x The expression
 * \return the address to the int expression,
 *         return nullptr, if x is not IntImm.
 */
inline const int64_t* as_const_int(const PrimExpr& x) {
  if (!x.defined()) return nullptr;
  if (const IntImmNode* op = x.as<IntImmNode>()) {
    return &(op->value);
  }

  return nullptr;
}

/*!
 * \brief Check whether x is a constant integer expression.
 * \param x The input argument
 * \param value the value to be compared against.
 * \return whether x is constant expression.
 */
inline bool is_const_int(const PrimExpr& x, int64_t value);

/*!
 * \brief Check whether x is a constant integer 1
 * \param x The input argument.
 * \note This only return true for integer types.
 * \return whether x is constant 1
 */
inline bool is_one(const PrimExpr& x) { return is_const_int(x, 1); }

/*!
 * \brief Check whether x is a constant integer 0
 * \param x The input argument
 * \return whether x is constant 0
 * \note This only return true for integer types.
 */
inline bool is_zero(const PrimExpr& x) { return is_const_int(x, 0); }

/*!
 * \brief Check whether x is an integer constant.
 * \note This only return true for integer types.
 * \return whether x is constant
 */
inline bool is_const_int(const PrimExpr& x);

/*!
 * \brief Check whether x is an integer/float constant.
 * \note This only return true for integer types.
 * \return whether x is constant
 */
inline bool is_const_number(const PrimExpr& x);

/*!
 * \brief Check whether x is a constant power of two
 * If x is power of two, write the power to the shift.
 *
 * \param x The input expression.
 * \param shift The output shift if x is power of two.
 * \return whether x is constant power of two
 */
TVM_DLL bool is_const_power_of_two_integer(const PrimExpr& x, int* shift);

// Implementation details after this
inline bool is_const_int(const PrimExpr& x) { return as_const_int(x); }

inline bool is_const_number(const PrimExpr& x) {
  if (x.as<IntImmNode>()) {
    return true;
  } else if (x.as<FloatImmNode>()) {
    return true;
  } else if (const auto* op = x.as<prim::BroadcastNode>()) {
    return (op->value->IsInstance<IntImmNode>() || op->value->IsInstance<FloatImmNode>());
  }
  return false;
}

inline bool is_positive_const(const PrimExpr& a) {
  const int64_t* as_int = as_const_int(a);
  return as_int && (*as_int > 0);
}

inline bool is_negative_const(const PrimExpr& a) {
  const int64_t* as_int = as_const_int(a);
  return as_int && (*as_int < 0);
}

inline bool is_const_int(const PrimExpr& x, int64_t value) {
  const int64_t* as_int = as_const_int(x);
  return as_int && (*as_int == value);
}

/*!
 * \brief Construct a large uint constant by its low 32 bits and high 32bits.
 * \param value_ty The final primitive type.
 * \param low The lower 32 bits.
 * \param high The higher 32 bits.
 * \param span The location of this operation in the source.
 * \return The constructed expression.
 */
TVM_DLL PrimExpr LargeUIntImm(PrimType value_ty, int64_t low, int64_t high, Span span = Span());

template <typename ValueType>
inline PrimExpr MakeConstScalar(PrimType dtype, ValueType value, Span span = Span()) {
  DLDataTypeCode code = dtype.code();
  if (code == DLDataTypeCode::kDLInt || code == DLDataTypeCode::kDLBool) {
    return IntImm(dtype, static_cast<int64_t>(value), span);
  }
  if (code == DLDataTypeCode::kDLUInt) {
    // Use IntImm if it is a small integer
    uint64_t uval = static_cast<uint64_t>(value);
    if (value < static_cast<ValueType>(0)) {
      TVM_FFI_THROW(InternalError) << "cannot make uint from negative value " << value;
    } else if (uval <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return IntImm(dtype, static_cast<int64_t>(value), span);
    } else {
      return LargeUIntImm(dtype, static_cast<int64_t>(uval & 0xffffffffULL),
                          static_cast<int64_t>(uval >> 32U), span);
    }
  }
  if (dtype.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLFloat8_e3m4,
                        DLDataTypeCode::kDLFloat8_e4m3, DLDataTypeCode::kDLFloat8_e4m3b11fnuz,
                        DLDataTypeCode::kDLFloat8_e4m3fn, DLDataTypeCode::kDLFloat8_e4m3fnuz,
                        DLDataTypeCode::kDLFloat8_e5m2, DLDataTypeCode::kDLFloat8_e5m2fnuz,
                        DLDataTypeCode::kDLFloat8_e8m0fnu, DLDataTypeCode::kDLFloat6_e2m3fn,
                        DLDataTypeCode::kDLFloat6_e3m2fn, DLDataTypeCode::kDLFloat4_e2m1fn) ||
      dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    return FloatImm(dtype, static_cast<double>(value), span);
  }
  TVM_FFI_THROW(InternalError) << "cannot make const for type " << dtype;
  throw;
}

template <>
inline PrimExpr MakeConstScalar(PrimType dtype, bool value, Span span) {
  return MakeConstScalar(dtype, static_cast<int>(value), span);
}

template <typename ValueType, typename>
inline PrimExpr MakeConst(PrimType dtype, ValueType value, Span span) {
  if (!dtype.IsScalableVector() && !dtype.IsFixedLengthVector()) {
    return MakeConstScalar(dtype, value, span);
  }
  PrimType elem_ty = dtype.WithLanes(1);
  if (dtype.IsFixedLengthVector()) {
    return prim::Broadcast(MakeConstScalar(elem_ty, value, span), dtype.lanes(), span);
  }
  PrimExpr lanes =
      prim::Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                dtype.VScaleFactor());
  return prim::Broadcast(MakeConstScalar(elem_ty, value, span), lanes, span);
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

#define TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD_SPANNED(Name)                 \
  inline PrimExpr Name(const PrimExpr& a, float b, Span span = Span()) {  \
    return Name(a, PrimExpr(b), span);                                    \
  }                                                                       \
  inline PrimExpr Name(float a, const PrimExpr& b, Span span = Span()) {  \
    return Name(PrimExpr(a), b, span);                                    \
  }                                                                       \
  inline PrimExpr Name(int a, const PrimExpr& b, Span span = Span()) {    \
    return Name(prim::MakeConst(b.ty(), a), b, span);                     \
  }                                                                       \
  inline PrimExpr Name(const PrimExpr& a, int b, Span span = Span()) {    \
    return Name(a, prim::MakeConst(a.ty(), b), span);                     \
  }                                                                       \
  inline PrimExpr Name(const PrimExpr& a, double b, Span span = Span()) { \
    return Name(a, FloatImm(PrimType::Float(64), b), span);               \
  }

#define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(Name)                             \
  inline PrimExpr Name(const PrimExpr& a, bool b) { return Name(a, PrimExpr(b)); } \
  inline PrimExpr Name(bool a, const PrimExpr& b) { return Name(PrimExpr(a), b); }

#define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD_SPANNED(Name)          \
  inline PrimExpr Name(const PrimExpr& a, bool b, Span span = Span()) { \
    return Name(a, PrimExpr(b), span);                                  \
  }                                                                     \
  inline PrimExpr Name(bool a, const PrimExpr& b, Span span = Span()) { \
    return Name(PrimExpr(a), b, span);                                  \
  }

#define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(Name)                                               \
  inline PrimExpr Name(const PrimExpr& a, int b) { return Name(a, prim::MakeConst(a.ty(), b)); } \
  inline PrimExpr Name(int a, const PrimExpr& b) { return Name(prim::MakeConst(b.ty(), a), b); }

#define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(Name)             \
  inline PrimExpr Name(const PrimExpr& a, int b, Span span = Span()) { \
    return Name(a, prim::MakeConst(a.ty(), b), span);                  \
  }                                                                    \
  inline PrimExpr Name(int a, const PrimExpr& b, Span span = Span()) { \
    return Name(prim::MakeConst(b.ty(), a), b, span);                  \
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

}  // namespace tvm
#endif  // TVM_IR_PRIM_OP_H_
