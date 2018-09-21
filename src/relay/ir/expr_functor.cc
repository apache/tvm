/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/expr_mutator.cc
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

Expr ExprMutator::Mutate(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = ExprMutator::VisitExpr(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

Expr ExprMutator::VisitExpr_(const VarNode* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const ConstantNode* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const GlobalVarNode* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const OpNode* op) {
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const TupleNode* op) {
  tvm::Array<Expr> fields;
  bool all_fields_unchanged = true;
  for (auto field : op->fields) {
    auto new_field = this->Mutate(field);
    fields.push_back(new_field);
    all_fields_unchanged &= new_field.same_as(field);
  }

  if (all_fields_unchanged) {
    return GetRef<Expr>(op);
  } else {
    return TupleNode::make(fields);
  }
}

Expr ExprMutator::VisitExpr_(const ParamNode* op) {
  Var var = Downcast<Var>(this->Mutate(op->var));
  auto type = this->VisitType(op->type);
  if (op->var.same_as(var) && op->type.same_as(type)) {
    return GetRef<Expr>(op);
  } else {
    return ParamNode::make(var, type);
  }
}

Expr ExprMutator::VisitExpr_(const FunctionNode* op) {
  tvm::Array<TypeParam> ty_params;
  bool all_ty_params_changed = true;

  for (auto ty_param : op->type_params) {
    TypeParam new_ty_param = Downcast<TypeParam>(VisitType(ty_param));
    ty_params.push_back(new_ty_param);
    all_ty_params_changed &= new_ty_param.same_as(ty_param);
  }

  tvm::Array<Param> params;
  bool all_params_changed = true;
  for (auto param : op->params) {
    Param new_param = Downcast<Param>(this->Mutate(param));
    params.push_back(new_param);
    all_params_changed &= param.same_as(new_param);
  }

  auto ret_type = this->VisitType(op->ret_type);
  auto body = this->Mutate(op->body);

  if (ty_params.same_as(op->type_params) &&
      params.same_as(op->params) &&
      ret_type.same_as(op->ret_type) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return FunctionNode::make(params, ret_type, body, ty_params);
  }
}

Expr ExprMutator::VisitExpr_(const CallNode* call_node) {
  auto new_op = this->Mutate(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  tvm::Array<Type> ty_args;
  for (auto ty_arg : call_node->type_args) {
    auto new_ty_arg = this->VisitType(ty_arg);
    ty_args.push_back(new_ty_arg);
    unchanged &= new_ty_arg.same_as(ty_arg);
  }

  tvm::Array<Expr> call_args;
  for (auto arg : call_node->args) {
    auto new_arg = this->Mutate(arg);
    call_args.push_back(new_arg);
    unchanged &= new_arg.same_as(arg);
  }

  if (unchanged) {
    return GetRef<Expr>(call_node);
  } else {
    return CallNode::make(new_op, call_args, call_node->attrs, ty_args);
  }
}

Expr ExprMutator::VisitExpr_(const LetNode* op) {
  Var var = Downcast<Var>(this->Mutate(op->var));
  auto type = this->VisitType(op->value_type);
  auto value = this->Mutate(op->value);
  auto body = this->Mutate(op->body);

  if (var.same_as(op->var) &&
      type.same_as(op->value_type) &&
      value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return LetNode::make(var, value, body, type);
  }
}

Expr ExprMutator::VisitExpr_(const IfNode* op) {
  auto guard = this->Mutate(op->cond);
  auto true_b = this->Mutate(op->true_branch);
  auto false_b = this->Mutate(op->false_branch);
  if (op->cond.same_as(guard) &&
      op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    return GetRef<Expr>(op);;
  } else {
    return IfNode::make(guard, true_b, false_b);
  }
}

Type ExprMutator::VisitType(const Type& t) {
  return t;
}

void ExprVisitor::ExprVisitor::VisitExpr_(const VarNode* op) {
}

void ExprVisitor::ExprVisitor::VisitExpr_(const GlobalVarNode* op) {
}

void ExprVisitor::ExprVisitor::VisitExpr_(const ConstantNode* op) {
}

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
