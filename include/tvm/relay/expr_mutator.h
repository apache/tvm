/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/expr_mutator.h
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */
#ifndef TVM_RELAY_EXPR_MUTATOR_H_
#define TVM_RELAY_EXPR_MUTATOR_H_

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class ExprMutator
    : public ::tvm::relay::ExprFunctor<Expr(const Expr&, const Expr&)> {
 public:
  Expr Mutate(const Expr& expr);
  Expr VisitExpr_(const VarNode* op, const Expr& e) override;
  Expr VisitExpr_(const ConstantNode* op, const Expr& e) override;
  Expr VisitExpr_(const GlobalVarNode* op, const Expr& e) override;
  Expr VisitExpr_(const OpNode* op, const Expr& expr) override;
  Expr VisitExpr_(const TupleNode* op, const Expr& e) override;
  Expr VisitExpr_(const ParamNode* op, const Expr& e) override;
  Expr VisitExpr_(const FunctionNode* op, const Expr& e) override;
  Expr VisitExpr_(const CallNode* call_node, const Expr& e) override;
  Expr VisitExpr_(const LetNode* op, const Expr& e) override;
  Expr VisitExpr_(const IfNode* op, const Expr& e) override;
  virtual Type VisitType(const Type& t);

 private:
  tvm::Map<Expr, Expr> memo_;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_MUTATOR_H_
