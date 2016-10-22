/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr_util.h
 * \brief Expression util
 */
#ifndef TVM_EXPR_UTIL_H_
#define TVM_EXPR_UTIL_H_

#include <vector>

#include "./expr.h"
#include "./expr_node.h"

namespace tvm {

/*!
 * \brief simplify the expression src
 * \param src The source expression
 * \return the simplified expression.
 */
Expr Simplify(Expr src);

/*!
 * \brief replace the variables in expression src by specification from dict
 * \param src The source expression
 * \param dict The specification for variable replacement
 * \return the new expression with variable replaced
 */
Expr Bind(Expr src, std::unordered_map<Expr, Expr> dict);

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
    case kReduceNode: {
      const auto* n = expr.Get<ReduceNode>();
      Visit(n->src, fvisit);
      break;
    }
    case kTensorReadNode: {
      const auto* n = expr.Get<TensorReadNode>();
      for (size_t i = 0; i < n->indices.size(); ++i) {
        Visit(n->indices[i], fvisit);
      }
      break;
    }
    default: break;
  }
  fvisit(expr);
}

/*!
 * \brief transform the exression node in expr tree in post DFS order.
 * \param expr The expression tree
 * \param fvisit The visit function.
 * \return the new expression after transformation
 */
template<typename FVisit>
inline Expr Transform(const Expr& expr, FVisit fvisit) {
  // TODO(tqchen) change to stack based impl.
  std::vector<Expr> children;
  switch (expr.node_type()) {
    case kBinaryOpNode: {
      const auto* n = expr.Get<BinaryOpNode>();
      Expr e = Transform(n->lhs, fvisit);
      children.push_back(e);
      children.push_back(Transform(n->rhs, fvisit));
      break;
    }
    case kUnaryOpNode: {
      const auto* n = expr.Get<UnaryOpNode>();
      children.push_back(Transform(n->src, fvisit));
      break;
    }
    case kReduceNode: {
      const auto* n = expr.Get<ReduceNode>();
      children.push_back(Transform(n->src, fvisit));
      break;
    }
    case kTensorReadNode: {
      const auto* n = expr.Get<TensorReadNode>();
      for (size_t i = 0; i < n->indices.size(); ++i) {
        children.push_back(Transform(n->indices[i], fvisit));
      }
      break;
    }
    default: break;
  }
  Expr ret = fvisit(expr, children);
  return ret;
}

}  // namespace tvm

#endif  // TVM_EXPR_UTIL_H_
