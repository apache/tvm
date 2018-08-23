/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/ir_operator.h
 * \brief Common operators of Expr
 */
#ifndef TVM_IR_OPERATOR_H_
#define TVM_IR_OPERATOR_H_

#include <algorithm>
#include "expr.h"
#include "ir.h"

namespace tvm {

using HalideIR::likely;
using HalideIR::likely_if_innermost;
// functions
using HalideIR::cast;
using HalideIR::min;
using HalideIR::max;
using HalideIR::select;

/*!
 * \brief sum of of source expression over axis
 * \param source The source expression.
 * \param axis List of iteration variables that will be used for reduction.
 */
TVM_DLL Expr sum(Expr source, Array<IterVar> axis);

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


// Unary intrinsic operators
#define TVM_DECLARE_INTRIN_UNARY(OpName)                                \
  inline Expr OpName(Expr x) {                                          \
    return ir::Call::make(x.type(), #OpName, {x}, ir::Call::PureIntrinsic); \
  }                                                                     \


TVM_DECLARE_INTRIN_UNARY(exp);
TVM_DECLARE_INTRIN_UNARY(tanh);
TVM_DECLARE_INTRIN_UNARY(sigmoid);
TVM_DECLARE_INTRIN_UNARY(sqrt);
TVM_DECLARE_INTRIN_UNARY(log);
TVM_DECLARE_INTRIN_UNARY(floor);
TVM_DECLARE_INTRIN_UNARY(ceil);
TVM_DECLARE_INTRIN_UNARY(round);
TVM_DECLARE_INTRIN_UNARY(trunc);

/*!
 * \brief Calculate power(x, y)
 * \param x The left operand.
 * \param y The right operand.
 */
inline Expr pow(Expr x, Expr y) {
  match_types(x, y);
  CHECK(x.type().is_float()) << "power only applies to float";
  return ir::Call::make(x.type(), "pow", { x, y }, ir::Call::PureIntrinsic);
}

/*!
 * \brief Calculate absolute value of x, elementwise
 * \param x The input data
 *
 * \return The aboslute value of input data x
 */
inline Expr abs(Expr x) {
  if (x.type().is_int()) {
    return select(x >= make_zero(x.type()), x, -x);
  } else if (x.type().is_float()) {
    return ir::Call::make(x.type(), "fabs", {x}, ir::Call::PureIntrinsic);
  } else if (x.type().is_uint()) {
    return x;
  } else {
    LOG(WARNING) << "Warning: Data type " << x.type()
      <<" not supported for absolute op. Skipping absolute op...";
    return x;
  }
}

}  // namespace tvm

#endif  // TVM_IR_OPERATOR_H_
