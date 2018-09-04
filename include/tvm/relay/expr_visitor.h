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
  void VisitExpr_(const LocalVarNode* op) override { return; }

  void VisitExpr_(const GlobalVarNode* op) override { return; }

  void VisitExpr_(const ConstantNode* op) override { return; }

  void VisitExpr_(const TupleNode* op) override {
    for (auto field : op->fields) {
      this->VisitExpr(field);
    }
  }

  void VisitExpr_(const ParamNode* op) override {
    this->VisitExpr(op->var);
  }

  void VisitExpr_(const FunctionNode* op) override {
    for (auto param : op->params) {
      this->VisitExpr(param);
    }

    this->VisitExpr(op->body);
  }

  void VisitExpr_(const CallNode* op) override {
    this->VisitExpr(op->op);
    for (auto ty_arg : op->type_args) {
      this->VisitType(ty_arg);
    }

    for (auto arg : op->args) {
      this->VisitExpr(arg);
    }
  }

  void VisitExpr_(const LetNode* op) override {
    this->VisitExpr(op->var);
    this->VisitExpr(op->value);
    this->VisitExpr(op->body);
  }

  void VisitExpr_(const IfNode* op) override {
    this->VisitExpr(op->cond);
    this->VisitExpr(op->true_value);
    this->VisitExpr(op->false_value);
  }

  void VisitExpr_(const OpNode* op) override { return; }

  virtual void VisitType(const Type& t) {}
};

class ExprFVisitor : public ::tvm::relay::ExprFunctor<Expr(const Expr& n)> {
 public:
  Expr VisitExpr_(const LocalVarNode* op) override {
    return GetRef<LocalVar>(op);
  }

  Expr VisitExpr_(const GlobalVarNode* op) override {
    return GetRef<GlobalVar>(op);
  }

  Expr VisitExpr_(const OpNode* op) override {
    return GetRef<Op>(op);
  }

  Expr VisitExpr_(const TupleNode* op) override {
    tvm::Array<Expr> fields;
    for (auto field : op->fields) {
      fields.push_back(this->VisitExpr(field));
    }

    return TupleNode::make(fields);
  }

  Expr VisitExpr_(const ParamNode* op) override {
    Expr var_expr = this->VisitExpr(op->var);
    if (const LocalVarNode* var_node = var_expr.as<LocalVarNode>()) {
      auto var = GetRef<LocalVar>(var_node);
      auto type = this->VisitType(op->type);
      return ParamNode::make(var, type);
    } else {
      throw dmlc::Error("the default param visitor has bug");
    }
  }

  Expr VisitExpr_(const FunctionNode* op) override {
    tvm::Array<TypeParam> ty_params;

    for (auto ty : op->type_params) {
      Type ty_param_type = VisitType(ty);
      if (auto ty_param = ty_param_type.as<TypeParamNode>()) {
        auto ty_param_ref = GetRef<TypeParam>(ty_param);
        ty_params.push_back(ty_param_ref);
      } else {
        throw dmlc::Error("the default func visitor has bug");
      }
    }

    tvm::Array<Param> params;
    for (auto param : op->params) {
      Expr param_expr = this->VisitExpr(param);
      if (const ParamNode* param_node = param_expr.as<ParamNode>()) {
        auto param = GetRef<Param>(param_node);
        params.push_back(param);
      } else {
        throw dmlc::Error("the default func visitor has bug");
      }
    }

    auto ret_type = this->VisitType(op->ret_type);
    auto body = this->VisitExpr(op->body);
    return FunctionNode::make(params, ret_type, body, ty_params);
  }

  Expr VisitExpr_(const CallNode* call_node) override {
    auto fn = this->VisitExpr(call_node->op);

    tvm::Array<Type> ty_args;
    for (auto ty_arg : call_node->type_args) {
      auto new_ty_arg = this->VisitType(ty_arg);
      ty_args.push_back(new_ty_arg);
    }

    tvm::Array<Expr> call_args;
    for (auto arg : call_node->args) {
      call_args.push_back(this->VisitExpr(arg));
    }

    auto call = CallNode::make(fn, call_args, call_node->attrs, ty_args);

    return call;
  }

  Expr VisitExpr_(const LetNode* op) override {
    Expr var_expr = this->VisitExpr(op->var);
    if (const LocalVarNode* var_node = var_expr.as<LocalVarNode>()) {
      auto var = GetRef<LocalVar>(var_node);
      auto type = this->VisitType(op->value_type);
      auto value = this->VisitExpr(op->value);
      auto body = this->VisitExpr(op->body);
      return LetNode::make(var, value, body, type);
    } else {
      throw dmlc::Error("the default let visitor has error");
    }
  }

  Expr VisitExpr_(const IfNode* op) override {
    auto guard = this->VisitExpr(op->cond);
    auto true_b = this->VisitExpr(op->true_value);
    auto false_b = this->VisitExpr(op->false_value);
    return IfNode::make(guard, true_b, false_b);
  }

  virtual Type VisitType(const Type& t) { return t; }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_VISITOR_H_
