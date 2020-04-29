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
 * \file tvm/tir/op.h
 * \brief Common operators defined for Expr.
 *
 * \note Most of the operator defined here perform simple constant folding
 *   when the type is int32 or int64 for simplifying the index expressions.
 */
// Acknowledgement: Most operator APIs originate from Halide.
#ifndef TVM_TIR_OP_H_
#define TVM_TIR_OP_H_

#include <tvm/ir/type.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <algorithm>
#include <type_traits>
#include <limits>


namespace tvm {

// Most common operators can be overloaded by argument type(PrimExpr).
// So we put them under the root namespace.
// It is also necessary to overload operators for PrimExpr.
//
// We put more developer oriented APIs -- make_const and is_const under tir
// as they are more specific to the tir namespace.

/*!
 * \brief Get the type of the expression under the unified type system.
 *
 * This function could return a more refined type than
 * the runtime type provided by expr->dtype
 *
 * \param expr The input parameter.
 * \return The result type.
 *
 * \sa tvm/ir/type.h for discussion about the relation between Type and runtime::DataType.
 */
TVM_DLL Type GetType(const PrimExpr& expr);

/*!
 * \brief Get the implied DataType for storing values with type during runtime.
 *
 * \param type The input type.
 * \return The result runtime::DataType.
 *
 * \sa tvm/ir/type.h for discussion about the relation between Type and runtime::DataType.
 */
TVM_DLL runtime::DataType GetRuntimeDataType(const Type& type);

/*!
 * Query the maximum possible value of dtype.
 * \param dtype The data type.
 * \return the maximum possible value in this format.
 */
TVM_DLL PrimExpr max_value(const DataType& dtype);

/*!
 * Query the minimum possible value of dtype.
 * \param dtype The data type.
 * \return the minimum possible value in this format.
 */
TVM_DLL PrimExpr min_value(const DataType& dtype);

/*!
 * Get the value of infinity.
 * \param dtype The data type.
 * \return the infinity value in this format.
 */
TVM_DLL PrimExpr infinity(const DataType& dtype);

/*!
 * \brief cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
TVM_DLL PrimExpr cast(const DataType& t, PrimExpr value);
/*!
 * \brief perform reinterpret cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
TVM_DLL PrimExpr reinterpret(const DataType& t, PrimExpr value);
/*!
 * \brief add operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator+(PrimExpr a, PrimExpr b);
/*!
 * \brief subtraction operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator-(PrimExpr a, PrimExpr b);
/*!
 * \brief negation.
 *
 * \param a input.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator-(PrimExpr a);
/*!
 * \brief multiplication operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator*(PrimExpr a, PrimExpr b);
/*!
 * \brief division operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator/(PrimExpr a, PrimExpr b);
/*!
 * \brief left shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<<(PrimExpr a, PrimExpr b);
/*!
 * \brief right shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>>(PrimExpr a, PrimExpr b);
/*!
 * \brief greater
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>(PrimExpr a, PrimExpr b);
/*!
 * \brief greater_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>=(PrimExpr a, PrimExpr b);
/*!
 * \brief less
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<(PrimExpr a, PrimExpr b);
/*!
 * \brief less_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<=(PrimExpr a, PrimExpr b);
/*!
 * \brief equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator==(PrimExpr a, PrimExpr b);
/*!
 * \brief not_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator!=(PrimExpr a, PrimExpr b);
/*!
 * \brief and
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator&&(PrimExpr a, PrimExpr b);
/*!
 * \brief or
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator||(PrimExpr a, PrimExpr b);
/*!
 * \brief not
 *
 * \param a left operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator!(PrimExpr a);
/*!
 * \brief compute division in C semantics.
 *
 * a / b as in C/C++.
 *
 * When operands are integers, it directly corresponds to truncdiv.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr div(PrimExpr a, PrimExpr b);
/*!
 * \brief compute trunc(a / b)
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr truncdiv(PrimExpr a, PrimExpr b);
/*!
 * \brief compute the remainder of truncdiv
 *
 * This is the default integer division behavior in C.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr truncmod(PrimExpr a, PrimExpr b);
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
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr indexdiv(PrimExpr a, PrimExpr b);
/*!
 * \brief compute the remainder floor(a / b) where a and b are non-negative.
 *
 * Use this function for index split calculation.
 * This function might take advantage of the fact
 * that a and b are non-negative.
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr indexmod(PrimExpr a, PrimExpr b);
/*!
 * \brief compute floor(a / b)
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr floordiv(PrimExpr a, PrimExpr b);
/*!
 * \brief compute the remainder of floordiv
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr floormod(PrimExpr a, PrimExpr b);
/*!
 * \brief take maximum of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr max(PrimExpr a, PrimExpr b);
/*!
 * \brief take minimum of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr min(PrimExpr a, PrimExpr b);
/*!
 * \brief take bitwise and of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator&(PrimExpr a, PrimExpr b);
/*!
 * \brief take bitwise or of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator|(PrimExpr a, PrimExpr b);
/*!
 * \brief take bitwise xor of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator^(PrimExpr a, PrimExpr b);
/*!
 * \brief take bitwise negation of two values
 *
 * \param a the input expression.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator~(PrimExpr a);
/*!
 * \brief Conditional expression.
 *
 * \param cond The condition
 * \param true_value The value when results are true.
 * \param false_value The value when results are false.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value);
/*!
 * \brief Mark condition as likely.
 * \param cond The condition
 * \return The marked expression.
 */
TVM_DLL PrimExpr likely(PrimExpr cond);
/*!
 * \brief Calculate power(x, y)
 * \param x The left operand.
 * \param y The right operand.
 */
TVM_DLL PrimExpr pow(PrimExpr x, PrimExpr y);
/*!
 * \brief Calculate absolute value of x.
 * \param x The input data
 *
 * \return The aboslute value of input data x
 */
TVM_DLL PrimExpr abs(PrimExpr x);
/*!
 * \brief Check if x is NaN.
 * \param x The input data
 * \return The result expression.
 */
TVM_DLL PrimExpr isnan(PrimExpr x);

/*!
 * \brief Check if x is finite.
 * \param x The input data
 * \return The result expression.
 */
TVM_DLL PrimExpr isfinite(PrimExpr x);

/*!
 * \brief Check if x is infinite.
 * \param x The input data
 * \return The result expression.
 */
TVM_DLL PrimExpr isinf(PrimExpr x);

/*!
 * \brief sum of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL PrimExpr sum(PrimExpr source, Array<tir::IterVar> axis);

/*!
 * \brief logical And of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL PrimExpr all(PrimExpr source, Array<tir::IterVar> axis);

/*!
 * \brief logical Or of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL PrimExpr any(PrimExpr source, Array<tir::IterVar> axis);

/*!
 * \brief max of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL PrimExpr max(PrimExpr source, Array<tir::IterVar> axis);

/*!
 * \brief max of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL PrimExpr min(PrimExpr source, Array<tir::IterVar> axis);

/*!
 * \brief product of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL PrimExpr prod(PrimExpr source, Array<tir::IterVar> axis);

/*!
 * \brief Calculate floor(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL PrimExpr floor(PrimExpr x);

/*!
 * \brief Calculate ceil(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL PrimExpr ceil(PrimExpr x);

/*!
 * \brief Calculate round(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL PrimExpr round(PrimExpr x);

/*!
 * \brief Calculates std::nearbyint(x)
 * \param x The input expression.
 * \return The result expression.
 * This is a faster alternate to round.
 */
TVM_DLL PrimExpr nearbyint(PrimExpr x);

/*!
 * \brief Calculate trunc(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL PrimExpr trunc(PrimExpr x);

/*!
 * \brief Construct a large uint constant by its low 32 bits and high 32bits.
 * \param dtype The final data type.
 * \param low The lower 32 bits.
 * \param high The higher 32 bits.
 * \return The constructed expression.
 */
TVM_DLL PrimExpr LargeUIntImm(DataType dtype, int64_t low, int64_t high);

// Intrinsic operators
#define TVM_DECLARE_INTRIN_UNARY(OpName)                                               \
  inline PrimExpr OpName(PrimExpr x) {                                                 \
    return tir::CallNode::make(x.dtype(), #OpName, {x}, tir::CallNode::PureIntrinsic); \
  }                                                                                    \

TVM_DECLARE_INTRIN_UNARY(exp);
TVM_DECLARE_INTRIN_UNARY(exp2);
TVM_DECLARE_INTRIN_UNARY(exp10);
TVM_DECLARE_INTRIN_UNARY(erf);
TVM_DECLARE_INTRIN_UNARY(tanh);
TVM_DECLARE_INTRIN_UNARY(sigmoid);
TVM_DECLARE_INTRIN_UNARY(sqrt);
TVM_DECLARE_INTRIN_UNARY(rsqrt);
TVM_DECLARE_INTRIN_UNARY(log);
TVM_DECLARE_INTRIN_UNARY(log2);
TVM_DECLARE_INTRIN_UNARY(log10);
TVM_DECLARE_INTRIN_UNARY(popcount);
TVM_DECLARE_INTRIN_UNARY(tan);
TVM_DECLARE_INTRIN_UNARY(cos);
TVM_DECLARE_INTRIN_UNARY(cosh);
TVM_DECLARE_INTRIN_UNARY(sin);
TVM_DECLARE_INTRIN_UNARY(sinh);
TVM_DECLARE_INTRIN_UNARY(asin);
TVM_DECLARE_INTRIN_UNARY(acos);
TVM_DECLARE_INTRIN_UNARY(atan);
TVM_DECLARE_INTRIN_UNARY(acosh);
TVM_DECLARE_INTRIN_UNARY(asinh);
TVM_DECLARE_INTRIN_UNARY(atanh);


namespace tir {
/*!
 * \brief Make a const value with certain data type.
 * \param t The target type.
 * \param value The input value
 * \return the result expression.
 * \tparam ValueType The constant value type
 */
template<typename ValueType,
         typename = typename std::enable_if<std::is_pod<ValueType>::value>::type>
inline PrimExpr make_const(DataType t, ValueType value);
/*!
 * \brief Make a const zero expr.
 * \param t The target type.
 * \return the result expression.
 */
inline PrimExpr make_zero(DataType t);
/*!
 * \brief Make a constant true expression.
 * \param lanes The number of lanes in the bool
 * \return The result expression.
 */
inline PrimExpr const_true(int lanes = 1) {
  return make_const(DataType::UInt(1, lanes), 1);
}
/*!
 * \brief Make a constant false expression.
 * \param lanes The number of lanes in the bool
 * \return The result expression.
 */
inline PrimExpr const_false(int lanes = 1) {
  return make_const(DataType::UInt(1, lanes), 0);
}
/*!
 * \brief Get x as constant int expression.
 * \param x The expression
 * \return the address to the int expression,
 *         return nullptr, if x is not IntImm.
 */
inline const int64_t* as_const_int(const PrimExpr& x) {
  if (!x.defined()) return nullptr;
  if (const tir::IntImmNode* op = x.as<tir::IntImmNode>()) {
    return &(op->value);
  } else {
    return nullptr;
  }
}

/*!
 * \brief Check whether x is a constant integer expression.
 * \param x The input argument
 * \param value the value to be compared against.
 * \return whether x is constant expression.
 */
inline bool is_const_int(const PrimExpr& x, int64_t value);

/*!
 * \brief Check whether stmt is nop.
 * \param stmt The input statement
 * \return whether stmt is nop
 */
inline bool is_no_op(const tir::Stmt& stmt);

/*!
 * \brief Check whether x is a constant integer 1
 * \param x The input argument.
 * \note This only return true for integer types.
 * \return whether x is constant 1
 */
inline bool is_one(const PrimExpr& x) {
  return is_const_int(x, 1);
}

/*!
 * \brief Check whether x is a constant integer 0
 * \param x The input argument
 * \return whether x is constant 0
 * \note This only return true for integer types.
 */
inline bool is_zero(const PrimExpr& x) {
  return is_const_int(x, 0);
}

/*!
 * \brief Check whether x is a constant.
 * \note This only return true for integer types.
 * \return whether x is constant
 */
inline bool is_const(const PrimExpr& x);

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
inline bool is_const(const PrimExpr& x) {
  if (x.as<tir::IntImmNode>()) {
    return true;
  } else if (const auto* op = x.as<tir::BroadcastNode>()) {
    const PrimExpr& val = op->value;
    if (val.as<tir::IntImmNode>()) {
      return true;
    }
  }
  return false;
}

inline bool is_positive_const(const PrimExpr& a) {
  if (const tir::IntImmNode* op = a.as<tir::IntImmNode>()) {
    return op->value > 0;
  } else {
    return false;
  }
}

inline bool is_negative_const(const PrimExpr& a) {
  if (const tir::IntImmNode* op = a.as<tir::IntImmNode>()) {
    return op->value < 0;
  } else {
    return false;
  }
}

inline bool is_const_int(const PrimExpr& x, int64_t value) {
  if (const auto* op = x.as<tir::IntImmNode>()) {
    return op->value == value;
  } else if (const auto* op = x.as<tir::BroadcastNode>()) {
    const PrimExpr& val = op->value;
    if (const auto* opv = val.as<tir::IntImmNode>()) {
      return opv->value == value;
    }
  }
  return false;
}

inline bool is_no_op(const tir::Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (const auto* op = stmt.as<tir::EvaluateNode>()) {
    return is_const(op->value);
  }
  if (const auto* op = stmt.as<tir::SeqStmtNode>()) {
    return op->seq.size() == 0;
  }
  return false;
}

template<typename ValueType>
inline PrimExpr MakeConstScalar(DataType t, ValueType value) {
  if (t.is_int()) return IntImm(t, static_cast<int64_t>(value));
  if (t.is_uint()) {
    // Use IntImm if it is a small integer
    uint64_t uval = static_cast<uint64_t>(value);
    if (uval <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return IntImm(t, static_cast<int64_t>(value));
    } else {
      uint64_t mask = (static_cast<uint64_t>(1) << 32U) - 1U;
      uint64_t low = uval & mask;
      uint64_t high = uval >> 32U;
      return LargeUIntImm(t, static_cast<int64_t>(low), static_cast<int64_t>(high));
    }
  }
  if (t.is_float()) return FloatImm(t, static_cast<double>(value));
  // For now, we store const scalar values of custom datatypes within doubles; later, during the
  // datatypes lowering pass, we will lower the value to its true representation in the format
  // specified by the datatype.
  // TODO(gus) when do we need to start worrying about doubles not being precise enough?
  if (static_cast<uint8_t>(t.code()) >= static_cast<uint8_t>(kTVMCustomBegin)) {
    return FloatImm(t, static_cast<double>(value));
  }
  LOG(FATAL) << "cannot make const for type " << t;
  return PrimExpr();
}

template<typename ValueType, typename>
inline PrimExpr make_const(DataType t, ValueType value) {
  if (t.lanes() == 1) {
    return MakeConstScalar(t, value);
  } else {
    return tir::BroadcastNode::make(
        MakeConstScalar(t.element_of(), value), t.lanes());
  }
}

inline PrimExpr make_zero(DataType t) {
  if (t.is_handle()) {
    return reinterpret(t, make_const(DataType::UInt(64), 0));
  }
  return make_const(t, 0);
}
}  // namespace tir

// additional const expression overloading
#define TVM_DEFINE_ASSIGN_OP_OVERLOAD(Name, OpFunc)             \
  inline PrimExpr Name(PrimExpr& a, PrimExpr b) {\
    a = OpFunc(a, b);                                           \
    return a;                                                   \
  }

#define TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(Name)              \
  inline PrimExpr Name(const PrimExpr& a, float b) {           \
    return Name(a, PrimExpr(b));                               \
  }                                                            \
  inline PrimExpr Name(float a, const PrimExpr& b) {           \
    return Name(PrimExpr(a), b);                               \
  }                                                            \
  inline PrimExpr Name(int a, const PrimExpr& b) {             \
    return Name(tir::make_const(b.dtype(), a), b);             \
  }                                                            \
  inline PrimExpr Name(const PrimExpr& a, int b) {             \
    return Name(a, tir::make_const(a.dtype(), b));             \
  }                                                            \
  inline PrimExpr Name(const PrimExpr& a, double b) {          \
    return Name(a, tir::make_const(DataType::Float(64), b));   \
  }

#define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(Name)         \
  inline PrimExpr Name(const PrimExpr& a, bool b) {            \
    return Name(a, PrimExpr(b));                               \
  }                                                            \
  inline PrimExpr Name(bool a, const PrimExpr& b) {            \
    return Name(PrimExpr(a), b);                               \
  }

#define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(Name)            \
  inline PrimExpr Name(const PrimExpr& a, int b) {            \
    return Name(a, tir::make_const(a.dtype(), b));            \
  }                                                           \
  inline PrimExpr Name(int a, const PrimExpr& b) {            \
    return Name(tir::make_const(b.dtype(), a), b);            \
  }

TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator+=, operator+);
TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator-=, operator-);
TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator*=, operator*);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator+);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator-);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator*);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(max);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(min);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(div);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>);  // NOLINT(*)
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>=);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<);  // NOLINT(*)
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<=);
// integer related ops
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(indexdiv);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(indexmod);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(truncdiv);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(truncmod);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(floordiv);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(floormod);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator>>); // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator<<); // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator&);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator|);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator^);
// logical ops
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator&&);
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator||);

/*!
 * \brief Helper function to raise a compiler error about division ambiguity.
 * \note The call to this function will always results in a compiler error.
 * \tparam TA Any class type.
 */
template<typename TA>
inline void DivAmbiguityError(const TA& a) {
  constexpr bool div_ambiguity = !std::is_class<TA>::value;
  static_assert(div_ambiguity,
                "TVM supports multiple types of integer divisions, "
                "please call div, indexdiv/indexmod, "
                "floordiv/floormod or truncdiv/truncmod directly "
                "to avoid ambiguity in the code. "
                "Checkout these functions in expr_operator.h.");
}

// The following code are not intended to be used in the codebase.
// Instead, they generate clear compiler errors that ask developers
// to use the specific division function.
// The second template argument is necessary to make sure the
// code compiles lazily by the compiler during invocation.
template<typename TB>
inline PrimExpr operator/(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

template<typename TB>
inline PrimExpr operator/=(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}

template<typename TB>
inline PrimExpr operator%(const PrimExpr& a, const TB& b) {
  DivAmbiguityError(a);
  return a;
}
}  // namespace tvm
#endif  // TVM_TIR_OP_H_
