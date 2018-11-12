/*!
 *  Copyright (c) 2018 by Contributors
 * \file expr_subst.h
 * \brief Utility functions for substituting expressions.
 */
#ifndef TVM_RELAY_PASS_EXPR_SUBST_H_
#define TVM_RELAY_PASS_EXPR_SUBST_H_
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

    Expr ExprSubst(const Expr& expr, tvm::Map<Expr, Expr> subst_map);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_EXPR_SUBST_H_
