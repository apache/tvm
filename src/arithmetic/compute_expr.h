/*!
 *  Copyright (c) 2017 by Contributors
 * \file compute_expr.h
 * \brief Utility integer expression with quick eager simplification.
 *  This is weaker than Simplify but can be done Eagerly.
 */
#ifndef TVM_ARITHMETIC_COMPUTE_EXPR_H_
#define TVM_ARITHMETIC_COMPUTE_EXPR_H_

#include <tvm/ir.h>
#include <arithmetic/Interval.h>
#include <limits>

namespace tvm {
namespace arith {

using HalideIR::Internal::add_would_overflow;
using HalideIR::Internal::sub_would_overflow;
using HalideIR::Internal::mul_would_overflow;

/*!
 * \brief Compute the expression with the given binary op.
 * \param lhs The left operand
 * \param rhs The right operand
 * \tparam Op the computation operator
 * \return The result.
 */
template<typename OP>
inline Expr ComputeExpr(Expr lhs, Expr rhs) {
  return OP::make(lhs, rhs);
}

/*!
 * \brief Compute an reduction with Op
 * \param values The input values.
 * \param empty_value The value when return if it is empty, can be Expr()
 *        which will cause an error to be rasied.
 * \tparam Op The computation operator
 * \return The result.
 */
template<typename Op>
inline Expr ComputeReduce(
    const Array<Expr>& values, Expr empty_value);

template<typename T>
inline bool GetConst(Expr e, T* out);

template<>
inline bool GetConst<int64_t>(Expr e, int64_t *out) {
  if (e.type().is_vector()) return false;
  const int64_t *v = as_const_int(e);
  if (v) {
    *out = *v; return true;
  } else {
    return false;
  }
}
template<>
inline bool GetConst<uint64_t>(Expr e, uint64_t *out) {
  if (e.type().is_vector()) return false;
  const uint64_t *v = as_const_uint(e);
  if (v) {
    *out = *v; return true;
  } else {
    return false;
  }
}

// get a small constant int
inline bool GetConstInt(Expr e, int* out) {
  int64_t v1 = 0;
  uint64_t v2 = 0;
  if (GetConst(e, &v1)) {
    if (v1 > static_cast<int64_t>(
            std::numeric_limits<int>::max())) return false;
    *out = static_cast<int>(v1); return true;
  }
  if (GetConst(e, &v2)) {
    if (v2 > static_cast<uint64_t>(
            std::numeric_limits<int>::max())) return false;
    *out = static_cast<int>(v2); return true;
  }
  return false;
}

#define TVM_CONST_PROPAGATION(OP_NAME, OP)                       \
  int64_t ia = 0, ib = 0;                                        \
  if (GetConst(a, &ia) && GetConst(b, &ib)) {                    \
    if (OP_NAME ## _would_overflow(a.type().bits(), ia, ib)) {   \
      LOG(FATAL) << "signed int overflow";                       \
    }                                                            \
    return ir::IntImm::make(a.type(), ia OP ib);                 \
  }                                                              \
  uint64_t ua = 0, ub = 0;                                       \
  if (GetConst(a, &ua) && GetConst(b, &ub)) {                    \
    return ir::UIntImm::make(a.type(), ua OP ub);                \
  }                                                              \

template<>
inline Expr ComputeExpr<ir::Add>(Expr a, Expr b) {
  if (is_zero(a)) return b;
  if (is_zero(b)) return a;
  TVM_CONST_PROPAGATION(add, +);
  return ir::Add::make(a, b);
}

template<>
inline Expr ComputeExpr<ir::Sub>(Expr a, Expr b) {
  if (is_zero(b)) return a;
  TVM_CONST_PROPAGATION(sub, -);
  return ir::Sub::make(a, b);
}

template<>
inline Expr ComputeExpr<ir::Mul>(Expr a, Expr b) {
  if (is_one(a)) return b;
  if (is_one(b)) return a;
  TVM_CONST_PROPAGATION(mul, *);
  return ir::Mul::make(a, b);
}

template<>
inline Expr ComputeExpr<ir::Div>(Expr a, Expr b) {
  if (is_one(b)) return a;
  return ir::Div::make(a, b);
}

template<>
inline Expr ComputeExpr<ir::Mod>(Expr a, Expr b) {
  if (is_zero(a)) return make_zero(a.type());
  return ir::Mod::make(a, b);
}

template<>
inline Expr ComputeExpr<ir::Max>(Expr a, Expr b) {
  return HalideIR::Internal::Interval::make_max(a, b);
}

template<>
inline Expr ComputeExpr<ir::Min>(Expr a, Expr b) {
  return HalideIR::Internal::Interval::make_min(a, b);
}

template<typename Op>
inline Expr ComputeReduce(const Array<Expr>& values, Expr empty_value) {
  if (values.size() == 0U) {
    CHECK(empty_value.defined());
    return empty_value;
  }
  Expr res = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    res = ComputeExpr<Op>(res, values[i]);
  }
  return res;
}

}  // namespace arith
}  // namespace tvm
#endif   // TVM_ARITHMETIC_COMPUTE_EXPR_H_
