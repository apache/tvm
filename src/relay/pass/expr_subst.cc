/*!
 *  Copyright (c) 2018 by Contributors
 * \file expr_subst.h
 * \brief Utility functions for substituting expressions.
 */

#include <tvm/relay/expr_functor.h>
#include "./expr_subst.h"

namespace tvm {
namespace relay {

class ExprSubstituter : public ExprMutator {
 public:
  explicit ExprSubstituter(std::unordered_map<Expr, Expr, NodeHash, NodeEqual> subst_map)
      : subst_map_(subst_map) {}

  Expr VisitExpr(const Expr& expr) final {
    auto it = subst_map_.find(expr);
    if (it != subst_map_.end()) {
      return ExprMutator::VisitExpr((*it).second);
    }
    return ExprMutator::VisitExpr(expr);
  }

 private:
  tvm::Map<Expr, Expr> subst_map_;
};

Expr ExprSubst(const Expr& expr, std::unordered_map<Expr, Expr, NodeHash, NodeEqual> subst_map) {
  return ExprSubstituter(std::move(subst_map)).Mutate(expr);
}

}  // namespace relay
}  // namespace tvm
