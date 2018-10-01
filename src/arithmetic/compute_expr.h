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

inline bool GetConst(Expr e, int64_t* out) {
  if (e.type().is_vector()) return false;
  const int64_t* v = as_const_int(e);
  if (v) {
    *out = *v; return true;
  } else {
    return false;
  }
}

// get a small constant int
inline bool GetConstInt(Expr e, int* out) {
  int64_t v1 = 0;
  if (GetConst(e, &v1)) {
    if (v1 > static_cast<int64_t>(
            std::numeric_limits<int>::max())) return false;
    *out = static_cast<int>(v1); return true;
  }
  return false;
}

template<>
inline Expr ComputeExpr<ir::Add>(Expr a, Expr b) {
  return a + b;
}

template<>
inline Expr ComputeExpr<ir::Sub>(Expr a, Expr b) {
  return a - b;
}

template<>
inline Expr ComputeExpr<ir::Mul>(Expr a, Expr b) {
  return a * b;
}

template<>
inline Expr ComputeExpr<ir::Div>(Expr a, Expr b) {
  return a / b;
}

template<>
inline Expr ComputeExpr<ir::Mod>(Expr a, Expr b) {
  return a % b;
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
