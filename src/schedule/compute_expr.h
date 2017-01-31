/*!
 *  Copyright (c) 2017 by Contributors
 * \file compute_expr.h
 * \brief Utility integer expression with quick eager simplification.
 *  This is weaker than Simplify but can be done Eagerly.
 */
#ifndef TVM_SCHEDULE_COMPUTE_EXPR_H_
#define TVM_SCHEDULE_COMPUTE_EXPR_H_

#include <tvm/ir.h>
#include <pass/Interval.h>

namespace tvm {
namespace schedule {

using Halide::Internal::add_would_overflow;
using Halide::Internal::sub_would_overflow;
using Halide::Internal::mul_would_overflow;

/*!
 * \brief Compute the expression with the given binary op.
 * \param lhs The left operand
 * \param rhs The right operand
 * \return The result.
 */
template<typename OP>
inline Expr ComputeExpr(Expr lhs, Expr rhs) {
  return OP::make(lhs, rhs);
}

template<typename T>
inline bool GetConst(Expr e, T* out);

template<>
bool GetConst<int64_t>(Expr e, int64_t *out) {
  if (e.type().is_vector()) return false;
  const int64_t *v = as_const_int(e);
  if (v) {
    *out = *v; return true;
  } else {
    return false;
  }
}
template<>
bool GetConst<uint64_t>(Expr e, uint64_t *out) {
  if (e.type().is_vector()) return false;
  const uint64_t *v = as_const_uint(e);
  if (v) {
    *out = *v; return true;
  } else {
    return false;
  }
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
    return ir::UIntImm::make(a.type(), ua + ub);                 \
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
  return ir::Add::make(a, b);
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
  return ir::Mul::make(a, b);
}

template<>
inline Expr ComputeExpr<ir::Max>(Expr a, Expr b) {
  return Halide::Internal::Interval::make_max(a, b);
}

template<>
inline Expr ComputeExpr<ir::Min>(Expr a, Expr b) {
  return Halide::Internal::Interval::make_min(a, b);
}

}  // namespace schedule
}  // namespace tvm
#endif   // TVM_SCHEDULE_COMPUTE_EXPR_H_
