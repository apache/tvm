/*!
 *  Copyright (c) 2017 by Contributors
 * \file ir_operator.h
 * \brief Common operators of Expr
 */
#ifndef TVM_IR_OPERATOR_H_
#define TVM_IR_OPERATOR_H_

#include <algorithm>
#include "./expr.h"
#include "./ir.h"

namespace tvm {

using HalideIR::likely;
using HalideIR::likely_if_innermost;
// functions
using HalideIR::cast;
using HalideIR::min;
using HalideIR::max;
using HalideIR::abs;
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

inline Expr pow(Expr x, Expr y) {
  return ir::Call::make(x.type(), "pow", { x, y }, ir::Call::PureIntrinsic);
}

}  // namespace tvm

#endif  // TVM_IR_OPERATOR_H_
