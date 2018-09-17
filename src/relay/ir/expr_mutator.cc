/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/expr_mutator.cc
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */

#include <tvm/relay/expr_mutator.h>

namespace tvm {
namespace relay {

Expr ExprMutator::Mutate(const Expr& expr) {
  auto cached_expr = this->memo_.find(expr);
  if (cached_expr != this->memo_.end()) {
    return (*cached_expr).second;
  } else {
    auto new_expr = this->ExprMutator::VisitExpr(expr, expr);
    this->memo_.Set(expr, new_expr);
    return new_expr;
  }
}

Expr ExprMutator::VisitExpr_(const VarNode* op, const Expr& expr) {
  return expr;
}

Expr ExprMutator::VisitExpr_(const ConstantNode* op, const Expr& expr) {
  return expr;
}

Expr ExprMutator::VisitExpr_(const GlobalVarNode* op, const Expr& expr) {
  return expr;
}

Expr ExprMutator::VisitExpr_(const OpNode* op, const Expr& expr) {
  return expr;
}

Expr ExprMutator::VisitExpr_(const TupleNode* op, const Expr& e) {
  tvm::Array<Expr> fields;
  bool all_fields_unchanged = true;
  for (auto field : op->fields) {
    auto new_field = this->Mutate(field);
    fields.push_back(new_field);
    all_fields_unchanged &= new_field.same_as(field);
  }

  if (all_fields_unchanged) {
    return e;
  } else {
    return TupleNode::make(fields);
  }
}

Expr ExprMutator::VisitExpr_(const ParamNode* op, const Expr& e) {
  Var var = Downcast<Var>(this->Mutate(op->var));
  auto type = this->VisitType(op->type);
  if (var == op->var && type == op->type) {
    return e;
  } else {
    return ParamNode::make(var, type);
  }
}

Expr ExprMutator::VisitExpr_(const FunctionNode* op, const Expr& e) {
  tvm::Array<TypeParam> ty_params;

  for (auto ty : op->type_params) {
    Type ty_param_type = VisitType(ty);
    if (auto ty_param = ty_param_type.as<TypeParamNode>()) {
      auto ty_param_ref = GetRef<TypeParam>(ty_param);
      ty_params.push_back(ty_param_ref);
    } else {
      LOG(FATAL) << "the default function visitor expected a TypeParam found: "
                 << ty_param_type << std::endl;
      return Expr();
    }
  }

  tvm::Array<Param> params;
  for (auto param : op->params) {
    Expr param_expr = this->Mutate(param);
    if (const ParamNode* param_node = param_expr.as<ParamNode>()) {
      auto param = GetRef<Param>(param_node);
      params.push_back(param);
    } else {
      CHECK(false) << "the default function visitor expected a Param found: "
                   << param_expr << std::endl;
      return Expr();
    }
  }

  auto ret_type = this->VisitType(op->ret_type);
  auto body = this->Mutate(op->body);
  return FunctionNode::make(params, ret_type, body, ty_params);
}

Expr ExprMutator::VisitExpr_(const CallNode* call_node, const Expr& e) {
  auto fn = this->Mutate(call_node->op);

  tvm::Array<Type> ty_args;
  for (auto ty_arg : call_node->type_args) {
    auto new_ty_arg = this->VisitType(ty_arg);
    ty_args.push_back(new_ty_arg);
  }

  tvm::Array<Expr> call_args;
  for (auto arg : call_node->args) {
    call_args.push_back(this->Mutate(arg));
  }

  auto call = CallNode::make(fn, call_args, call_node->attrs, ty_args);

  return call;
}

Expr ExprMutator::VisitExpr_(const LetNode* op, const Expr& e) {
  Expr var_expr = this->Mutate(op->var);
  if (const VarNode* var_node = var_expr.as<VarNode>()) {
    auto var = GetRef<Var>(var_node);
    auto type = this->VisitType(op->value_type);
    auto value = this->Mutate(op->value);
    auto body = this->Mutate(op->body);
    return LetNode::make(var, value, body, type);
  } else {
    LOG(FATAL) << "the default let visitor expected a Var found: " << var_expr
               << std::endl;
    return Expr();
  }
}

Expr ExprMutator::VisitExpr_(const IfNode* op, const Expr& e) {
  auto guard = this->Mutate(op->cond);
  auto true_b = this->Mutate(op->true_branch);
  auto false_b = this->Mutate(op->false_branch);
  if (op->cond == guard && true_b == op->true_branch &&
      false_b == op->false_branch) {
    return e;
  } else {
    return IfNode::make(guard, true_b, false_b);
  }
}

Type ExprMutator::VisitType(const Type& t) { return t; }

}  // namespace relay
}  // namespace tvm
