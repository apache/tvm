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
 * \file src/relay/expr_functor.cc
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>

#include <stack>

#include "../op/annotation/annotation.h"
#include "../op/memory/on_device.h"

namespace tvm {
namespace relay {
MixedModeVisitor::MixedModeVisitor(int visit_limit) {
  ICHECK(visit_limit > 0) << "Dataflow visit limit must be greater than 0";
  ICHECK(visit_limit < 10) << "Dataflow visit limit must be less than 10";
  visit_limit_ = visit_limit;
}

void MixedModeVisitor::VisitLeaf(const Expr& expr) {
  if (visit_counter_[expr.get()] < visit_limit_) {
    ExprFunctor::VisitExpr(expr);
  }
  visit_counter_[expr.get()]++;
}

bool MixedModeVisitor::CheckVisited(const Expr& expr) {
  if (visit_counter_[expr.get()] < visit_limit_) {
    return false;
  } else {
    visit_counter_[expr.get()]++;
    return true;
  }
}

void MixedModeVisitor::VisitExpr(const Expr& expr) {
  auto fcheck_visited = [this](const Expr& expr) { return this->CheckVisited(expr); };
  auto fvisit_leaf = [this](const Expr& expr) { return this->VisitLeaf(expr); };
  if (visit_counter_[expr.get()] < visit_limit_) {
    ExpandDataflow(expr, fcheck_visited, fvisit_leaf);
  }
}

// Overwrite the VisitExpr so we don't recurse for dataflow nodes
void MixedModeVisitor::VisitExpr_(const CallNode* op) {}

// Overwrite the VisitExpr so we don't recurse for dataflow nodes
void MixedModeVisitor::VisitExpr_(const TupleNode* op) {}

// Overwrite the VisitExpr so we don't recurse for dataflow nodes
void MixedModeVisitor::VisitExpr_(const TupleGetItemNode* op) {}

void MixedModeMutator::VisitLeaf(const Expr& expr) {
  if (!memo_.count(expr)) {
    Expr ret = this->DispatchVisitExpr(expr);
    memo_[expr] = ret;
  }
}

bool MixedModeMutator::CheckVisited(const Expr& expr) {
  if (memo_.count(expr)) {
    return true;
  } else {
    return false;
  }
}

Expr MixedModeMutator::DispatchVisitExpr(const Expr& expr) { return ExprMutator::VisitExpr(expr); }

Expr MixedModeMutator::VisitExpr(const Expr& expr) {
  auto fcheck_visited = [this](const Expr& expr) { return this->CheckVisited(expr); };
  auto fvisit_leaf = [this](const Expr& expr) { return this->VisitLeaf(expr); };
  if (memo_.count(expr)) {
    return memo_[expr];
  } else {
    ExpandDataflow(expr, fcheck_visited, fvisit_leaf);
    return memo_[expr];
  }
}

class PostOrderRewriter : public MixedModeMutator {
 public:
  explicit PostOrderRewriter(ExprRewriter* rewriter) : rewriter_(rewriter) {}

  Expr DispatchVisitExpr(const Expr& expr) final {
    auto post = ExprFunctor::VisitExpr(expr);
    return rewriter_->Rewrite(expr, post);
  }

  using MixedModeMutator::VisitExpr_;

  Expr VisitExpr_(const LetNode* node) final {
    auto pre_visit = [this](const LetNode* op) {
      Expr var = this->Mutate(op->var);
      Expr value = this->Mutate(op->value);
    };
    auto post_visit = [this, node](const LetNode* op) {
      Var var = Downcast<Var>(this->Mutate(op->var));
      Expr value = this->Mutate(op->value);
      Expr body = this->Mutate(op->body);
      Expr expr = GetRef<Expr>(op);
      Expr post;
      if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
        post = expr;
      } else {
        post = Let(var, value, body);
      }
      //  avoid rewriting the first LetNode twice
      if (op == node) {
        this->memo_[expr] = post;
      } else {
        this->memo_[expr] = this->rewriter_->Rewrite(expr, post);
      }
    };
    ExpandANormalForm(node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(node)];
  }

 protected:
  ExprRewriter* rewriter_;
};

Expr PostOrderRewrite(const Expr& expr, ExprRewriter* rewriter) {
  return PostOrderRewriter(rewriter).VisitExpr(expr);
}

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

Expr ExprMutator::VisitExpr_(const VarNode* var_node) {
  Type type_annotation = var_node->type_annotation;
  if (var_node->type_annotation.defined()) {
    type_annotation = this->VisitType(var_node->type_annotation);
  }
  return WithFields(GetRef<Var>(var_node), var_node->vid, type_annotation);
}

Expr ExprMutator::VisitExpr_(const ConstantNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const GlobalVarNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const OpNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const TupleNode* tuple_node) {
  tvm::Array<Expr> fields;
  fields.reserve(tuple_node->fields.size());

  for (auto field : tuple_node->fields) {
    auto new_field = this->Mutate(field);
    fields.push_back(new_field);
  }
  return WithFields(GetRef<Tuple>(tuple_node), fields);
}

Expr ExprMutator::VisitExpr_(const FunctionNode* func_node) {
  tvm::Array<TypeVar> ty_params;

  for (auto ty_param : func_node->type_params) {
    TypeVar new_ty_param = Downcast<TypeVar>(VisitType(ty_param));
    ty_params.push_back(new_ty_param);
  }

  tvm::Array<Var> params;
  for (auto param : func_node->params) {
    Var new_param = Downcast<Var>(this->Mutate(param));
    params.push_back(new_param);
  }

  auto ret_type = this->VisitType(func_node->ret_type);
  auto body = this->Mutate(func_node->body);

  return WithFields(GetRef<Function>(func_node), params, body, ret_type, ty_params);
}

Expr ExprMutator::VisitExpr_(const CallNode* call_node) {
  auto new_op = this->Mutate(call_node->op);

  tvm::Array<Type> ty_args;
  ty_args.reserve(call_node->type_args.size());

  for (auto ty_arg : call_node->type_args) {
    auto new_ty_arg = this->VisitType(ty_arg);
    ty_args.push_back(new_ty_arg);
  }

  tvm::Array<Expr> call_args;
  call_args.reserve(call_node->args.size());
  for (auto arg : call_node->args) {
    auto new_arg = this->Mutate(arg);
    call_args.push_back(new_arg);
  }

  return WithFields(GetRef<Call>(call_node), new_op, call_args, {}, ty_args);
}

Expr ExprMutator::VisitExpr_(const LetNode* let_node) {
  Var var = Downcast<Var>(this->Mutate(let_node->var));
  auto value = this->Mutate(let_node->value);
  auto body = this->Mutate(let_node->body);

  return WithFields(GetRef<Let>(let_node), var, value, body);
}

Expr ExprMutator::VisitExpr_(const IfNode* if_node) {
  auto cond = this->Mutate(if_node->cond);
  auto true_b = this->Mutate(if_node->true_branch);
  auto false_b = this->Mutate(if_node->false_branch);

  return WithFields(GetRef<If>(if_node), cond, true_b, false_b);
}

Expr ExprMutator::VisitExpr_(const TupleGetItemNode* get_item) {
  Expr tuple = this->Mutate(get_item->tuple);
  return WithFields(GetRef<TupleGetItem>(get_item), tuple);
}

Expr ExprMutator::VisitExpr_(const RefCreateNode* ref_create) {
  Expr value = this->Mutate(ref_create->value);
  return WithFields(GetRef<RefCreate>(ref_create), value);
}

Expr ExprMutator::VisitExpr_(const RefReadNode* ref_read) {
  Expr ref = this->Mutate(ref_read->ref);
  return WithFields(GetRef<RefRead>(ref_read), ref);
}

Expr ExprMutator::VisitExpr_(const RefWriteNode* ref_write) {
  Expr ref = this->Mutate(ref_write->ref);
  Expr value = this->Mutate(ref_write->value);
  return WithFields(GetRef<RefWrite>(ref_write), ref, value);
}

Expr ExprMutator::VisitExpr_(const ConstructorNode* c) { return GetRef<Expr>(c); }

Expr ExprMutator::VisitExpr_(const MatchNode* match_node) {
  Array<Clause> clauses;
  for (const Clause& p : match_node->clauses) {
    clauses.push_back(VisitClause(p));
  }
  Expr data = Mutate(match_node->data);

  return WithFields(GetRef<Match>(match_node), data, clauses);
}

Clause ExprMutator::VisitClause(const Clause& clause) {
  Pattern lhs = VisitPattern(clause->lhs);
  Expr rhs = Mutate(clause->rhs);
  return WithFields(clause, lhs, rhs);
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

void ExprVisitor::VisitExpr_(const VarNode* op) {
  this->VisitSpan(op->span);
  if (op->type_annotation.defined()) {
    this->VisitType(op->type_annotation);
  }
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const ConstantNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  this->VisitSpan(op->span);
  for (auto field : op->fields) {
    this->VisitExpr(field);
  }
}

void ExprVisitor::VisitExpr_(const FunctionNode* op) {
  this->VisitSpan(op->span);
  for (auto param : op->params) {
    this->VisitExpr(param);
  }

  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);

  for (auto ty_arg : op->type_args) {
    this->VisitType(ty_arg);
  }

  for (auto arg : op->args) {
    this->VisitExpr(arg);
  }
}

void ExprVisitor::VisitExpr_(const LetNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->value);
  this->VisitExpr(op->var);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}

void ExprVisitor::VisitExpr_(const OpNode* op) { return; }

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->tuple);
}

void ExprVisitor::VisitExpr_(const RefCreateNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const RefReadNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->ref);
}

void ExprVisitor::VisitExpr_(const RefWriteNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->ref);
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const ConstructorNode* op) {
  // TODO(@jroesch): visit spans
  for (const Type& t : op->inputs) {
    this->VisitType(t);
  }
  this->VisitType(op->belong_to);
}

void ExprVisitor::VisitExpr_(const MatchNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->data);
  for (const Clause& c : op->clauses) {
    this->VisitClause(c);
  }
}

void ExprVisitor::VisitClause(const Clause& op) {
  // TODO(@jroesch): visit spans
  this->VisitPattern(op->lhs);
  this->VisitExpr(op->rhs);
}

void ExprVisitor::VisitPattern(const Pattern& p) { return; }

void ExprVisitor::VisitType(const Type& t) { return; }

void ExprVisitor::VisitSpan(const Span& span) { return; }

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
  std::unordered_set<const Object*> visited_;
};

void PostOrderVisit(const Expr& e, std::function<void(const Expr&)> fvisit) {
  ExprApplyVisit(fvisit).VisitExpr(e);
}

TVM_REGISTER_GLOBAL("relay.analysis.post_order_visit").set_body_typed([](Expr expr, PackedFunc f) {
  PostOrderVisit(expr, [f](const Expr& n) { f(n); });
});

// Implement bind.
class ExprBinder : public MixedModeMutator, PatternMutator {
 public:
  explicit ExprBinder(const tvm::Map<Var, Expr>& args_map) : args_map_(args_map) {}

  using MixedModeMutator::VisitExpr_;

  Expr VisitExpr_(const LetNode* op) final {
    ICHECK(!args_map_.count(op->var)) << "Cannot bind an internel variable in let";
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    for (Var param : op->params) {
      ICHECK(!args_map_.count(param)) << "Cannnot bind an internal function parameter";
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

  Pattern VisitPattern(const Pattern& p) final { return PatternMutator::VisitPattern(p); }

  Clause VisitClause(const Clause& clause) final {
    Pattern lhs = VisitPattern(clause->lhs);
    return WithFields(clause, lhs, VisitExpr(clause->rhs));
  }

  Var VisitVar(const Var& v) final {
    ICHECK(!args_map_.count(v)) << "Cannnot bind an internal pattern variable";
    return v;
  }

 private:
  const tvm::Map<Var, Expr>& args_map_;
};

// This function should be called SubstAndBind, since it assumes any variables introduced
// in the substitution right hand side should be implicitly bound in the function.
Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& args_map) {
  if (const FunctionNode* func = expr.as<FunctionNode>()) {
    Expr new_body = ExprBinder(args_map).VisitExpr(func->body);
    Array<Var> new_params;
    for (size_t i = 0; i < func->params.size(); ++i) {
      if (!args_map.count(func->params[i])) {
        new_params.push_back(func->params[i]);
      }
    }
    if (new_body.same_as(func->body) && new_params.size() == func->params.size()) {
      return expr;
    }

    auto ret =
        Function(new_params, new_body, func->ret_type, func->type_params, func->attrs, func->span);
    ret->virtual_device_ = func->virtual_device();

    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> set;
    for (const auto& v : FreeVars(expr)) {
      set.insert(v);
    }
    for (const auto& v : FreeVars(ret)) {
      if (set.count(v) == 0) {
        new_params.push_back(v);
      }
    }

    ret =
        Function(new_params, new_body, func->ret_type, func->type_params, func->attrs, func->span);
    ret->virtual_device_ = func->virtual_device();

    VLOG(4) << "Expr:\n" << expr;
    VLOG(4) << "Ret:\n" << ret;

    ICHECK_EQ(FreeVars(expr).size(), FreeVars(ret).size());
    return std::move(ret);
  } else {
    return ExprBinder(args_map).VisitExpr(expr);
  }
}

TVM_REGISTER_GLOBAL("relay.ir.Bind").set_body([](TVMArgs args, TVMRetValue* ret) {
  ObjectRef input = args[0];
  if (input->IsInstance<ExprNode>()) {
    *ret = Bind(Downcast<Expr>(input), args[1]);
  } else {
    ICHECK(input->IsInstance<TypeNode>());
    *ret = Bind(Downcast<Type>(input), args[1]);
  }
});

Function SubstituteBoundVars(const Function& func, const tvm::Map<Var, Expr>& args_map) {
  Expr new_body = ExprBinder(args_map).VisitExpr(func->body);
  Array<Var> new_params;
  for (size_t i = 0; i < func->params.size(); i++) {
    if (!args_map.count(func->params[i])) {
      new_params.push_back(func->params[i]);
    } else {
      if (auto var = args_map[func->params[i]].as<Var>()) {
        new_params.push_back(var.value());
      } else {
        ICHECK(false) << "Expected all values in args_map to be vars, but found "
                      << args_map[func->params[i]]->GetTypeKey();
      }
    }
  }
  auto ret =
      Function(new_params, new_body, func->ret_type, func->type_params, func->attrs, func->span);
  ret->virtual_device_ = func->virtual_device();
  return ret;
}

void ExpandANormalForm(const LetNode* op, std::function<void(const LetNode*)> pre_visit,
                       std::function<void(const LetNode*)> post_visit) {
  std::stack<const LetNode*> stack;
  stack.push(op);
  bool is_anormal = true;
  while (is_anormal) {
    const LetNode* current_op = stack.top();
    pre_visit(current_op);
    if (const LetNode* new_op = current_op->body.as<LetNode>()) {
      stack.push(new_op);
    } else {
      is_anormal = false;
    }
  }
  while (stack.size()) {
    const LetNode* current_op = stack.top();
    stack.pop();
    post_visit(current_op);
  }
}

}  // namespace relay
}  // namespace tvm
