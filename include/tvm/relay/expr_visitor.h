/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/expr_visitor.h
 * \brief A simple visitor wrapper around ExprFunctor.
 *
 * Exposes two visitors with default traversal strategies, one
 * which doesn't compute a result but can mutate internal state,
 * and another which functionally builds a new Expr.
 */
#ifndef TVM_RELAY_EXPR_VISITOR_H_
#define TVM_RELAY_EXPR_VISITOR_H_

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class ExprVisitor : public ::tvm::relay::ExprFunctor<void(const Expr& n)> {
 public:
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const GlobalVarNode* op) override;
  void VisitExpr_(const ConstantNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const ParamNode* op) override;
  void VisitExpr_(const FunctionNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const LetNode* op) override;
  void VisitExpr_(const IfNode* op) override;
  void VisitExpr_(const OpNode* op) override;
  virtual void VisitType(const Type& t);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_VISITOR_H_
