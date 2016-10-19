/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr_util.h
 * \brief Expression util
 */
#ifndef TVM_EXPR_UTIL_H_
#define TVM_EXPR_UTIL_H_

#include "./expr.h"
#include "./expr_node.h"

namespace tvm {

/*!
 * \brief simplify the expression src
 * \param src The source expression
 * \return the simplified expression.
 */
inline Expr Simplify(Expr src) {
  return src;
}

/*!
 * \brief visit the exression node in expr tree in post DFS order.
 * \param expr The expression tree
 * \param fvisit The visit function.
 */
template<typename FVisit>
inline void Visit(const Expr& expr, FVisit fvisit) {
  // TODO(tqchen) change to stack based impl.
  switch (expr.node_type()) {
    case kBinaryOpNode: {
      const auto* n = expr.Get<BinaryOpNode>();
      Visit(n->lhs, fvisit);
      Visit(n->rhs, fvisit);
      break;
    }
    case kUnaryOpNode: {
      const auto* n = expr.Get<UnaryOpNode>();
      Visit(n->src, fvisit);
      break;
    }
    default: break;
  }
  fvisit(expr);
}

}  // namespace tvm

#endif  // TVM_EXPR_UTIL_H_
