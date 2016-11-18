/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr.h
 * \brief Defines the expressions in AST.
 */
#ifndef TVM_EXPR_H_
#define TVM_EXPR_H_

#include <ir/Expr.h>
#include <ir/IROperator.h>
#include <string>
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
using Halide::IR::FunctionBaseNode;
using Halide::Internal::Stmt;

class Var : public Halide::VarExpr {
 public:
  explicit Var(const std::string& name_hint = "v",
               Type t = Int(32)) : VarExpr(name_hint, t) {}
};

}  // namespace tvm
#endif  // TVM_EXPR_H_
