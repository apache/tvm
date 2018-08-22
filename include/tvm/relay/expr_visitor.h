/*!
 *  Copyright (c) 2018 by Contributors
 * \file expr_visitor.h
 * \brief A simple visitor wrapper around ExprFunctor designed for visitors which
 * maintain mutable state.
 */
#ifndef TVM_RELAY_EXPR_VISITOR_H_
#define TVM_RELAY_EXPR_VISITOR_H_

#include "tvm/relay/expr_functor.h"

namespace tvm {
namespace relay {

template <typename... Args>
class ExprVisitor : public ::tvm::relay::ExprFunctor<void(const Expr& n, Args...)> {
 public:
  void VisitExpr_(const LocalVarNode* op, Args... args) override { return; }

  void VisitExpr_(const GlobalVarNode* op, Args... args) override { return; }

  void VisitExpr_(const ConstantNode* op, Args... args) override { return; }

  void VisitExpr_(const TupleNode* op, Args... args) override {
    for (auto field : op->fields) {
      this->VisitExpr(field, args...);
    }
  }

  void VisitExpr_(const ParamNode* op, Args... args) override {
    this->VisitExpr(op->var, args...);
  }

  void VisitExpr_(const FunctionNode* op, Args... args) override {
    for (auto param : op->params) {
      this->VisitExpr(param, args...);
    }

    this->VisitExpr(op->body, args...);
  }

  void VisitExpr_(const CallNode* op, Args... args) override {
    this->VisitExpr(op->op, args...);
    for (auto arg : op->args) {
      this->VisitExpr(arg, args...);
    }
  }

  void VisitExpr_(const LetNode* op, Args... args) override {
    this->VisitExpr(op->var, args...);
    this->VisitExpr(op->value, args...);
    this->VisitExpr(op->body, args...);
  }

  void VisitExpr_(const IfNode* op, Args... args) override {
    this->VisitExpr(op->cond, args...);
    this->VisitExpr(op->true_value, args...);
    this->VisitExpr(op->false_value, args...);
  }

  void VisitExpr_(const OperatorNode* op, Args... args) override { return; }
};

template <typename... Args>
class ExprFVisitor : public ::tvm::relay::ExprFunctor<Expr(const Expr& n, Args...)> {
 public:
  Expr VisitExpr_(const LocalVarNode* op, Args... args) override {
    return GetRef<LocalVar>(op);
  }

  Expr VisitExpr_(const GlobalVarNode* op, Args... args) override {
    return GetRef<GlobalVar>(op);
  }

  Expr VisitExpr_(const OperatorNode* op, Args... args) override {
    return GetRef<Operator>(op);
  }

  Expr VisitExpr_(const TupleNode* op, Args... args) override {
    tvm::Array<Expr> fields;
    for (auto field : op->fields) {
      fields.push_back(this->VisitExpr(field, args...));
    }

    return TupleNode::make(fields);
  }

  Expr VisitExpr_(const ParamNode* op, Args... args) override {
    Expr var_expr = this->VisitExpr(op->var, args...);
    if (const LocalVarNode* var_node = var_expr.as<LocalVarNode>()) {
      auto var = GetRef<LocalVar>(var_node);
      auto type = this->VisitType(op->type, args...);
      return ParamNode::make(var, type);
    } else {
      throw dmlc::Error("the default param visitor has bug");
    }
  }

  Expr VisitExpr_(const FunctionNode* op, Args... args) override {
    tvm::Array<Type> ty_params;
    for (auto ty : op->type_params) {
      ty_params.push_back(this->VisitType(ty, args...));
    }

    tvm::Array<Param> params;
    for (auto param : op->params) {
      Expr param_expr = this->VisitExpr(param, args...);
      if (const ParamNode* param_node = param_expr.as<ParamNode>()) {
        auto param = GetRef<Param>(param_node);
        params.push_back(param);
      } else {
        throw dmlc::Error("the default func visitor has bug");
      }
    }

    auto ret_type = this->VisitType(op->ret_type, args...);
    auto body = this->VisitExpr(op->body, args...);
    return FunctionNode::make(ty_params, params, ret_type, body);
  }

  Expr VisitExpr_(const CallNode* call_node, Args... args) override {
    auto fn = this->VisitExpr(call_node->op, args...);

    tvm::Array<Type> ty_args;
    for (auto ty_arg : call_node->type_args) {
      auto new_ty_arg = this->VisitType(ty_arg, args...);
      ty_args.push_back(new_ty_arg);
    }

    tvm::Array<Expr> call_args;
    for (auto arg : call_node->args) {
      call_args.push_back(this->VisitExpr(arg, args...));
    }

    auto call = CallNode::make(fn, call_args, call_node->attrs);
    call->ty_args = ty_args;

    return call;
  }

  Expr VisitExpr_(const LetNode* op, Args... args) override {
    Expr var_expr = this->VisitExpr(op->var, args...);
    if (const LocalVarNode* var_node = var_expr.as<LocalVarNode>()) {
      auto var = GetRef<LocalVar>(var_node);
      auto type = this->VisitType(op->value_type, args...);
      auto value = this->VisitExpr(op->value, args...);
      auto body = this->VisitExpr(op->body, args...);
      return LetNode::make(var, type, value, body);
    } else {
      throw dmlc::Error("the default let visitor has error");
    }
  }

  Expr VisitExpr_(const IfNode* op, Args... args) override {
    auto guard = this->VisitExpr(op->cond, args...);
    auto true_b = this->VisitExpr(op->true_value, args...);
    auto false_b = this->VisitExpr(op->false_value, args...);
    return IfNode::make(guard, true_b, false_b);
  }

  virtual Type VisitType(const Type& t, Args... args) { return t; }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_VISITOR_H_
