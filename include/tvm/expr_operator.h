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
 * \file tvm/expr_operator.h
 * \brief Common operators defined for Expr.
 *
 * \note Most of the operator defined here perform simple constant folding
 *   when the type is int32 or int64 for simplifying the index expressions.
 */
// Acknowledgement: Most operator APIs originate from Halide.
#ifndef TVM_EXPR_OPERATOR_H_
#define TVM_EXPR_OPERATOR_H_

#include <algorithm>
#include <type_traits>
#include "expr.h"
#include "ir.h"

namespace tvm {

/*!
 * \brief Make a const value with certain data type.
 * \param t The target type.
 * \param value The input value
 * \return the result expression.
 * \tparam ValueType The constant value type
 */
template<typename ValueType,
         typename = typename std::enable_if<std::is_pod<ValueType>::value>::type>
inline Expr make_const(Type t, ValueType value);
/*!
 * \brief Make a const zero expr.
 * \param t The target type.
 * \return the result expression.
 */
inline Expr make_zero(Type t);
/*!
 * \brief Make a constant true expression.
 * \param lanes The number of lanes in the bool
 * \return The result expression.
 */
inline Expr const_true(int lanes = 1) {
  return make_const(UInt(1, lanes), 1);
}
/*!
 * \brief Make a constant false expression.
 * \param lanes The number of lanes in the bool
 * \return The result expression.
 */
inline Expr const_false(int lanes = 1) {
  return make_const(UInt(1, lanes), 0);
}
/*!
 * \brief Get x as constant int expression.
 * \param x The expression
 * \return the address to the int expression,
 *         return nullptr, if x is not IntImm.
 */
inline const int64_t* as_const_int(const Expr& x) {
  if (!x.defined()) return nullptr;
  if (const ir::IntImm* op = x.as<ir::IntImm>()) {
    return &(op->value);
  } else {
    return nullptr;
  }
}

/*!
 * \brief Get x as constant uint expression.
 * \param x The expression
 * \return the address to the int expression,
 *         return nullptr, if x is not UIntImm.
 */
inline const uint64_t* as_const_uint(const Expr& x) {
  if (!x.defined()) return nullptr;
  if (const ir::UIntImm* op = x.as<ir::UIntImm>()) {
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
inline bool is_const_int(const Expr& x, int64_t value);

/*!
 * \brief Check whether stmt is nop.
 * \param stmt The input statement
 * \return whether stmt is nop
 */
inline bool is_no_op(const Stmt& stmt);

/*!
 * \brief Check whether x is a constant integer 1
 * \param x The input argument.
 * \note This only return true for integer types.
 * \return whether x is constant 1
 */
inline bool is_one(const Expr& x) {
  return is_const_int(x, 1);
}

/*!
 * \brief Check whether x is a constant integer 0
 * \param x The input argument
 * \return whether x is constant 0
 * \note This only return true for integer types.
 */
inline bool is_zero(const Expr& x) {
  return is_const_int(x, 0);
}

/*!
 * \brief Check whether x is a constant.
 * \note This only return true for integer types.
 * \return whether x is constant
 */
inline bool is_const(const Expr& x);

/*!
 * \brief Check whether x is a constant power of two
 * If x is power of two, write the power to the shift.
 *
 * \param x The input expression.
 * \param shift The output shift if x is power of two.
 * \return whether x is constant power of two
 */
TVM_DLL bool is_const_power_of_two_integer(const Expr& x, int* shift);

/*!
 * \brief cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
TVM_DLL Expr cast(const Type& t, Expr value);
/*!
 * \brief perform reinterpret cast value to type.
 *
 * \param t the target type.
 * \param value The value
 * \return The result expression.
 * \note This function may return value if the type is the same.
 */
TVM_DLL Expr reinterpret(const Type& t, Expr value);
/*!
 * \brief add operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator+(Expr a, Expr b);
/*!
 * \brief subtraction operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator-(Expr a, Expr b);
/*!
 * \brief negation.
 *
 * \param a input.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator-(Expr a);
/*!
 * \brief multiplication operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator*(Expr a, Expr b);
/*!
 * \brief division operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator/(Expr a, Expr b);
/*!
 * \brief mod operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator%(Expr a, Expr b);
/*!
 * \brief left shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator<<(Expr a, Expr b);
/*!
 * \brief right shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator>>(Expr a, Expr b);
/*!
 * \brief greater
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator>(Expr a, Expr b);
/*!
 * \brief greater_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator>=(Expr a, Expr b);
/*!
 * \brief less
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator<(Expr a, Expr b);
/*!
 * \brief less_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator<=(Expr a, Expr b);
/*!
 * \brief equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator==(Expr a, Expr b);
/*!
 * \brief not_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator!=(Expr a, Expr b);
/*!
 * \brief and
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL Expr operator&&(Expr a, Expr b);
/*!
 * \brief or
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL Expr operator||(Expr a, Expr b);
/*!
 * \brief not
 *
 * \param a left operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL Expr operator!(Expr a);
/*!
 * \brief compute floor(a / b)
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr floordiv(Expr a, Expr b);
/*!
 * \brief compute the remainder of floordiv
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr floormod(Expr a, Expr b);
/*!
 * \brief take maximum of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr max(Expr a, Expr b);
/*!
 * \brief take minimum of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr min(Expr a, Expr b);
/*!
 * \brief take bitwise and of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator&(Expr a, Expr b);
/*!
 * \brief take bitwise or of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator|(Expr a, Expr b);
/*!
 * \brief take bitwise xor of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator^(Expr a, Expr b);
/*!
 * \brief take bitwise negation of two values
 *
 * \param a the input expression.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL Expr operator~(Expr a);
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
TVM_DLL Expr if_then_else(Expr cond, Expr true_value, Expr false_value);
/*!
 * \brief Mark condition as likely.
 * \param cond The condition
 * \return The marked expression.
 */
TVM_DLL Expr likely(Expr cond);
/*!
 * \brief Calculate power(x, y)
 * \param x The left operand.
 * \param y The right operand.
 */
TVM_DLL Expr pow(Expr x, Expr y);
/*!
 * \brief Calculate absolute value of x.
 * \param x The input data
 *
 * \return The aboslute value of input data x
 */
TVM_DLL Expr abs(Expr x);

/*!
 * \brief sum of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL Expr sum(Expr source, Array<IterVar> axis);

/*!
 * \brief logical And of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL Expr all(Expr source, Array<IterVar> axis);

/*!
 * \brief max of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL Expr max(Expr source, Array<IterVar> axis);

/*!
 * \brief max of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL Expr min(Expr source, Array<IterVar> axis);

/*!
 * \brief product of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL Expr prod(Expr source, Array<IterVar> axis);

/*!
 * \brief Calculate floor(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL Expr floor(Expr x);

/*!
 * \brief Calculate ceil(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL Expr ceil(Expr x);

/*!
 * \brief Calculate round(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL Expr round(Expr x);

/*!
 * \brief Calculate trunc(x)
 * \param x The input expression.
 * \return The result expression.
 */
TVM_DLL Expr trunc(Expr x);

// Intrinsic operators
#define TVM_DECLARE_INTRIN_UNARY(OpName)                                \
  inline Expr OpName(Expr x) {                                          \
    return ir::Call::make(x.type(), #OpName, {x}, ir::Call::PureIntrinsic); \
  }                                                                     \

TVM_DECLARE_INTRIN_UNARY(exp);
TVM_DECLARE_INTRIN_UNARY(tanh);
TVM_DECLARE_INTRIN_UNARY(sigmoid);
TVM_DECLARE_INTRIN_UNARY(sqrt);
TVM_DECLARE_INTRIN_UNARY(rsqrt);
TVM_DECLARE_INTRIN_UNARY(log);
TVM_DECLARE_INTRIN_UNARY(popcount);
TVM_DECLARE_INTRIN_UNARY(cos);
TVM_DECLARE_INTRIN_UNARY(sin);

// Implementation details after this
inline bool is_const(const Expr& x) {
  if (x.as<ir::IntImm>() || x.as<ir::UIntImm>()) {
    return true;
  } else if (const auto* op = x.as<ir::Broadcast>()) {
    const Expr& val = op->value;
    if (val.as<ir::IntImm>() || val.as<ir::UIntImm>()) {
      return true;
    }
  }
  return false;
}

inline bool is_positive_const(const Expr& a) {
  if (const ir::IntImm* op = a.as<ir::IntImm>()) {
    return op->value > 0;
  } else if (const ir::UIntImm* op = a.as<ir::UIntImm>()) {
    return op->value > 0;
  } else {
    return false;
  }
}

inline bool is_negative_const(const Expr& a) {
  if (const ir::IntImm* op = a.as<ir::IntImm>()) {
    return op->value < 0;
  } else {
    return false;
  }
}

inline bool is_const_int(const Expr& x, int64_t value) {
  if (const auto* op = x.as<ir::IntImm>()) {
    return op->value == value;
  } else if (const auto* op = x.as<ir::UIntImm>()) {
    return op->value == static_cast<uint64_t>(value);
  } else if (const auto* op = x.as<ir::Broadcast>()) {
    const Expr& val = op->value;
    if (const auto* opv = val.as<ir::IntImm>()) {
      return opv->value == value;
    } else if (const auto* opv = val.as<ir::UIntImm>()) {
      return opv->value == static_cast<uint64_t>(value);
    }
  }
  return false;
}

inline bool is_no_op(const Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (const auto* op = stmt.as<ir::Evaluate>()) {
    return is_const(op->value);
  }
  return false;
}

template<typename ValueType>
inline Expr MakeConstScalar(Type t, ValueType value) {
  if (t.is_int()) return ir::IntImm::make(t, static_cast<int64_t>(value));
  if (t.is_uint()) return ir::UIntImm::make(t, static_cast<uint64_t>(value));
  if (t.is_float()) return ir::FloatImm::make(t, static_cast<double>(value));
  // For now, we store const scalar values of custom datatypes within doubles; later, during the
  // datatypes lowering pass, we will lower the value to its true representation in the format
  // specified by the datatype.
  // TODO(gus) when do we need to start worrying about doubles not being precise enough?
  if (static_cast<uint8_t>(t.code()) >= static_cast<uint8_t>(kCustomBegin))
    return ir::FloatImm::make(t, static_cast<double>(value));
  LOG(FATAL) << "cannot make const for type " << t;
  return Expr();
}

template<typename ValueType, typename>
inline Expr make_const(Type t, ValueType value) {
  if (t.lanes() == 1) {
    return MakeConstScalar(t, value);
  } else {
    return ir::Broadcast::make(
        MakeConstScalar(t.element_of(), value), t.lanes());
  }
}

inline Expr make_zero(Type t) {
  if (t.is_handle()) {
    return reinterpret(t, make_const(UInt(64), 0));
  }
  return make_const(t, 0);
}

// additional const expression overloading
#define TVM_DEFINE_ASSIGN_OP_OVERLOAD(Name, OpFunc)            \
  inline Expr Name(Expr& a, Expr b) {                          \
    a = OpFunc(a, b);                                          \
    return a;                                                  \
  }

#define TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(Name)              \
  inline Expr Name(const Expr& a, float b) {                   \
    return Name(a, Expr(b));                                   \
  }                                                            \
  inline Expr Name(float a, const Expr& b) {                   \
    return Name(Expr(a), b);                                   \
  }                                                            \
  inline Expr Name(int a, const Expr& b) {                     \
    return Name(make_const(b.type(), a), b);                   \
  }                                                            \
  inline Expr Name(const Expr& a, int b) {                     \
    return Name(a, make_const(a.type(), b));                   \
  }

#define TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(Name)                  \
  inline Expr Name(const Expr& a, bool b) {                             \
    return Name(a, Expr(b));                                            \
  }                                                                     \
  inline Expr Name(bool a, const Expr& b) {                             \
    return Name(Expr(a), b);                                            \
  }

#define TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(Name)                      \
  inline Expr Name(const Expr& a, int b) {                              \
    return Name(a, make_const(a.type(), b));                            \
  }                                                                     \
  inline Expr Name(int a, const Expr& b) {                              \
    return Name(make_const(b.type(), a), b);                            \
  }


TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator+=, operator+);
TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator-=, operator-);
TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator*=, operator*);
TVM_DEFINE_ASSIGN_OP_OVERLOAD(operator/=, operator/);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator+);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator-);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator*);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator/);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(max);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(min);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>);  // NOLINT(*)
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator>=);
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<);  // NOLINT(*)
TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD(operator<=);
// integer related ops
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator%);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator>>); // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator<<); // NOLINT(*)
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator&);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator|);
TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD(operator^);
// logical ops
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator&&);
TVM_DEFINE_LOGICAL_OP_CONST_VAL_OVERLOAD(operator||);

}  // namespace tvm
#endif  // TVM_EXPR_OPERATOR_H_
