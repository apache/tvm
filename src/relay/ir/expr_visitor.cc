/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/expr_visitor.h
 * \brief A simple visitor wrapper around ExprFunctor.
 *
 * Exposes two visitors with default traversal strategies, one
 * which doesn't compute a result but can mutate internal state,
 * and another which functionally builds a new Expr.
 */
#include <tvm/relay/expr_visitor.h>

namespace tvm {
namespace relay {

void ExprVisitor::ExprVisitor::VisitExpr_(const VarNode* op) { return; }

void ExprVisitor::ExprVisitor::VisitExpr_(const GlobalVarNode* op) { return; }

void ExprVisitor::ExprVisitor::VisitExpr_(const ConstantNode* op) { return; }

void ExprVisitor::ExprVisitor::VisitExpr_(const TupleNode* op) {
  for (auto field : op->fields) {
    this->VisitExpr(field);
  }
}

void ExprVisitor::ExprVisitor::VisitExpr_(const ParamNode* op) {
  this->VisitExpr(op->var);
}

void ExprVisitor::ExprVisitor::VisitExpr_(const FunctionNode* op) {
  for (auto param : op->params) {
    this->VisitExpr(param);
  }

  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitExpr(op->op);
  for (auto ty_arg : op->type_args) {
    this->VisitType(ty_arg);
  }

  for (auto arg : op->args) {
    this->VisitExpr(arg);
  }
}

void ExprVisitor::VisitExpr_(const LetNode* op) {
  this->VisitExpr(op->var);
  this->VisitExpr(op->value);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}

void ExprVisitor::VisitExpr_(const OpNode* op) { return; }

void ExprVisitor::VisitType(const Type& t) { return; }

}  // namespace relay
}  // namespace tvm
