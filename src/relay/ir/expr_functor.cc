/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file src/tvm/relay/expr_mutator.cc
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include "type_functor.h"

namespace tvm {
namespace relay {

Expr ExprMutator::VisitExpr(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = ExprFunctor::VisitExpr(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

Expr ExprMutator::VisitExpr_(const VarNode* op) {
  if (op->type_annotation.defined()) {
    auto type = this->VisitType(op->type_annotation);
    if (!op->type_annotation.same_as(type)) {
      return VarNode::make(op->vid, type);
    }
  }
  // default case return self.
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

Expr ExprMutator::VisitExpr_(const FunctionNode* op) {
  tvm::Array<TypeVar> ty_params;
  bool all_ty_params_unchanged = true;

  for (auto ty_param : op->type_params) {
    TypeVar new_ty_param = Downcast<TypeVar>(VisitType(ty_param));
    ty_params.push_back(new_ty_param);
    all_ty_params_unchanged &= new_ty_param.same_as(ty_param);
  }

  tvm::Array<Var> params;
  bool all_params_unchanged = true;
  for (auto param : op->params) {
    Var new_param = Downcast<Var>(this->Mutate(param));
    params.push_back(new_param);
    all_params_unchanged &= param.same_as(new_param);
  }

  auto ret_type = this->VisitType(op->ret_type);
  auto body = this->Mutate(op->body);

  if (all_ty_params_unchanged &&
      all_params_unchanged &&
      ret_type.same_as(op->ret_type) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return FunctionNode::make(params, body, ret_type, ty_params, op->attrs);
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
  auto value = this->Mutate(op->value);
  auto body = this->Mutate(op->body);

  if (var.same_as(op->var) &&
      value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return LetNode::make(var, value, body);
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

Expr ExprMutator::VisitExpr_(const TupleGetItemNode* g) {
  auto t = this->Mutate(g->tuple);
  if (g->tuple == t) {
    return GetRef<Expr>(g);
  } else {
    return TupleGetItemNode::make(t, g->index);
  }
}

Expr ExprMutator::VisitExpr_(const RefCreateNode* op) {
  Expr value = this->Mutate(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Expr>(op);
  } else {
    return RefCreateNode::make(value);
  }
}

Expr ExprMutator::VisitExpr_(const RefReadNode* op) {
  Expr ref = this->Mutate(op->ref);
  if (ref.same_as(op->ref)) {
    return GetRef<Expr>(op);
  } else {
    return RefReadNode::make(ref);
  }
}

Expr ExprMutator::VisitExpr_(const RefWriteNode* op) {
  Expr ref = this->Mutate(op->ref);
  Expr value = this->Mutate(op->value);
  if (ref.same_as(op->ref) && value.same_as(op->value)) {
    return GetRef<Expr>(op);
  } else {
    return RefWriteNode::make(ref, value);
  }
}

Expr ExprMutator::VisitExpr_(const ConstructorNode* c) {
  return GetRef<Expr>(c);
}

Expr ExprMutator::VisitExpr_(const MatchNode* m) {
  std::vector<Clause> clauses;
  for (const Clause& p : m->clauses) {
    clauses.push_back(VisitClause(p));
  }
  return MatchNode::make(VisitExpr(m->data), clauses);
}

Clause ExprMutator::VisitClause(const Clause& c) {
  return ClauseNode::make(VisitPattern(c->lhs), VisitExpr(c->rhs));
}

Pattern ExprMutator::VisitPattern(const Pattern& p) { return p; }

Type ExprMutator::VisitType(const Type& t) { return t; }

void ExprVisitor::VisitExpr(const Expr& expr) {
  auto it = visit_counter_.find(expr.get());
  if (it != visit_counter_.end()) {
    ++it->second;
  } else {
    using TParent = ExprFunctor<void(const Expr&)>;
    TParent::VisitExpr(expr);
    visit_counter_.insert({expr.get(), 1});
  }
}

void ExprVisitor::ExprVisitor::VisitExpr_(const VarNode* op) {
  if (op->type_annotation.defined()) {
    this->VisitType(op->type_annotation);
  }
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
  this->VisitExpr(op->value);
  this->VisitExpr(op->var);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}

void ExprVisitor::VisitExpr_(const OpNode* op) { return; }

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) {
  this->VisitExpr(op->tuple);
}

void ExprVisitor::ExprVisitor::VisitExpr_(const RefCreateNode* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::ExprVisitor::VisitExpr_(const RefReadNode* op) {
  this->VisitExpr(op->ref);
}

void ExprVisitor::ExprVisitor::VisitExpr_(const RefWriteNode* op) {
  this->VisitExpr(op->ref);
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const ConstructorNode* op) {
  for (const Type& t : op->inputs) {
    this->VisitType(t);
  }
  this->VisitType(op->belong_to);
}

void ExprVisitor::VisitExpr_(const MatchNode* op) {
  this->VisitExpr(op->data);
  for (const Clause& c : op->clauses) {
    this->VisitClause(c);
  }
}

void ExprVisitor::VisitClause(const Clause& op) {
  this->VisitPattern(op->lhs);
  this->VisitExpr(op->rhs);
}

void ExprVisitor::VisitPattern(const Pattern& p) { return; }

void ExprVisitor::VisitType(const Type& t) { return; }

// visitor to implement apply
class ExprApplyVisit : public ExprVisitor {
 public:
  explicit ExprApplyVisit(std::function<void(const Expr&)> f) : f_(f) {}

  void VisitExpr(const Expr& e) final {
    if (visited_.count(e.get()) != 0) return;
    visited_.insert(e.get());
    ExprVisitor::VisitExpr(e);
    f_(e);
  }

 private:
  std::function<void(const Expr&)> f_;
  std::unordered_set<const Node*> visited_;
};

void PostOrderVisit(const Expr& e, std::function<void(const Expr&)> fvisit) {
  ExprApplyVisit(fvisit).VisitExpr(e);
}

TVM_REGISTER_API("relay._analysis.post_order_visit")
.set_body_typed<void(Expr, PackedFunc)>([](Expr expr, PackedFunc f) {
    PostOrderVisit(expr, [f](const Expr& n) {
        f(n);
      });
  });

// Implement bind.
class ExprBinder : public ExprMutator, PatternMutator {
 public:
  explicit ExprBinder(const tvm::Map<Var, Expr>& args_map)
    : args_map_(args_map) {
  }

  Expr VisitExpr_(const LetNode* op) final {
    CHECK(!args_map_.count(op->var))
        << "Cannot bind an internel variable in let";
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    for (Var param : op->params) {
      CHECK(!args_map_.count(param))
          << "Cannnot bind an internal function parameter";
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const VarNode* op) final {
    auto id = GetRef<Var>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return std::move(id);
    }
  }

  Pattern VisitPattern(const Pattern& p) final {
    return PatternMutator::VisitPattern(p);
  }

  Clause VisitClause(const Clause& c) final {
    Pattern pat = VisitPattern(c->lhs);
    return ClauseNode::make(pat, VisitExpr(c->rhs));
  }

  Var VisitVar(const Var& v) final {
    return Downcast<Var>(VisitExpr(v));
  }

 private:
  const tvm::Map<Var, Expr>& args_map_;
};

Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& args_map) {
  if (const FunctionNode* func = expr.as<FunctionNode>()) {
    Expr new_body = ExprBinder(args_map).VisitExpr(func->body);
    Array<Var> new_params;
    for (Var param : func->params) {
      if (!args_map.count(param)) {
        new_params.push_back(param);
      }
    }
    if (new_body.same_as(func->body) &&
        new_params.size() == func->params.size()) {
      return expr;
    }
    auto ret = FunctionNode::make(new_params,
                                  new_body,
                                  func->ret_type,
                                  func->type_params,
                                  func->attrs);
    std::unordered_set<Var, NodeHash, NodeEqual> set;
    for (const auto& v : FreeVars(expr)) {
      set.insert(v);
    }
    for (const auto& v : FreeVars(ret)) {
      if (set.count(v) == 0) {
        new_params.push_back(v);
      }
    }
    ret = FunctionNode::make(new_params,
                             new_body,
                             func->ret_type,
                             func->type_params,
                             func->attrs);
    CHECK_EQ(FreeVars(expr).size(), FreeVars(ret).size());
    return ret;
  } else {
    return ExprBinder(args_map).VisitExpr(expr);
  }
}


TVM_REGISTER_API("relay._expr.Bind")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    NodeRef input = args[0];
    if (input->derived_from<ExprNode>()) {
      *ret = Bind(Downcast<Expr>(input), args[1]);
    } else {
      CHECK(input->derived_from<TypeNode>());
      *ret = Bind(Downcast<Type>(input), args[1]);
    }
  });
}  // namespace relay
}  // namespace tvm
