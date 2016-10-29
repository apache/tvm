/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr.h
 * \brief Defines the expressions in AST.
 */
#ifndef TVM_EXPR_H_
#define TVM_EXPR_H_

#include <ir/Expr.h>
#include <ir/IROperator.h>
#include <type_traits>
#include "./base.h"

namespace tvm {

using Halide::Type;
using Halide::Float;
using Halide::Int;
using Halide::UInt;
using Halide::Handle;

// functions
using Halide::cast;
using Halide::min;
using Halide::max;
using Halide::abs;
using Halide::select;

using Halide::Expr;
using Var = Halide::VarExpr;

}  // namespace tvm
#endif  // TVM_EXPR_H_
