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
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>

#include <stack>

namespace tvm {
namespace relay {
/*!
 * \brief A function to iteratively traverse dataflow regions of a graph
 *
 * ExpandDataflow manually manages a stack and performs DFS to determine the processing
 * order of nodes in an input graph.
 *
 * If it finds a dataflow node (Call, Tuple, TupleGetItem), it checks if the arguments to that node
 * need to be processed via fcheck_visited. If so, the function pushes those arguments to the stack
 * and continues iteratively to process the top of the stack. When it finds a node that doesn't
 * match the dataflow types, or a node who's inputs have all been processed, it visits the current
 * leaf via fvisit_leaf.
 *
 * This function should be used internally to other classes to implement mixed-mode traversals. The
 * expectation is that fvisit_leaf will perform recursive analysis within mixed-mode traversal if it
 * hits a non-dataflow node.
 *
 * fcheck_visited and fvisit_leaf are templated to encourage compiler inlining.
 */
template <typename FCheckVisited, typename FVisitLeaf>
void ExpandDataflow(Expr expr, FCheckVisited fcheck_visited, FVisitLeaf fvisit_leaf) {
  std::stack<std::pair<Expr, bool>> stack;
  auto fpush_to_stack = [&fcheck_visited, &stack](const Expr& expr) {
    // The second state of the stack indicate whether the child has been
    // expanded in the pre-order.
    // NOTE: function will be inlined.
    if (!fcheck_visited(expr)) {
      stack.push({expr, false});
    }
  };
  fpush_to_stack(expr);
  while (stack.size() > 0) {
    auto node = stack.top().first;
    if (fcheck_visited(node)) {
      // if this node was visited through another path
      // after being added to the stack ignore it.
      stack.pop();
    } else if (stack.top().second) {
      // all the children have already been expanded.
      // we can just run post order visit on it.
      fvisit_leaf(node);
      stack.pop();
    } else if (const CallNode* op = node.as<CallNode>()) {
      // mark expanded = true
      stack.top().second = true;
      // push the children to the stack in reverse order
      // to match recursive processing order
      for (auto it = op->args.rbegin(); it != op->args.rend(); ++it) {
        fpush_to_stack(*it);
      }
      fpush_to_stack(op->op);
    } else if (const TupleNode* op = node.as<TupleNode>()) {
      stack.top().second = true;
      // push the children to the stack in reverse order
      // to match recursive processing order
      for (auto it = op->fields.rbegin(); it != op->fields.rend(); ++it) {
        fpush_to_stack(*it);
      }
    } else if (const TupleGetItemNode* op = node.as<TupleGetItemNode>()) {
      stack.top().second = true;
      fpush_to_stack(op->tuple);
    } else {
      // No need to expand the children directly run visit.
      fvisit_leaf(node);
      stack.pop();
    }
  }
}

MixedModeVisitor::MixedModeVisitor(int visit_limit) {
  CHECK(visit_limit > 0) << "Dataflow visit limit must be greater than 0";
  CHECK(visit_limit < 10) << "Dataflow visit limit must be less than 10";
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

Expr ExprMutator::VisitExpr_(const VarNode* op) {
  if (op->type_annotation.defined()) {
    auto type = this->VisitType(op->type_annotation);
    if (!op->type_annotation.same_as(type)) {
      return Var(op->vid, type);
    }
  }
  // default case return self.
  return GetRef<Expr>(op);
}

Expr ExprMutator::VisitExpr_(const ConstantNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const GlobalVarNode* op) { return GetRef<Expr>(op); }

Expr ExprMutator::VisitExpr_(const OpNode* op) { return GetRef<Expr>(op); }

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
    return Tuple(fields);
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

  if (all_ty_params_unchanged && all_params_unchanged && ret_type.same_as(op->ret_type) &&
      body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(params, body, ret_type, ty_params, op->attrs);
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
    return Call(new_op, call_args, call_node->attrs, ty_args);
  }
}

Expr ExprMutator::VisitExpr_(const LetNode* op) {
  Var var = Downcast<Var>(this->Mutate(op->var));
  auto value = this->Mutate(op->value);
  auto body = this->Mutate(op->body);

  if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Let(var, value, body);
  }
}

Expr ExprMutator::VisitExpr_(const IfNode* op) {
  auto guard = this->Mutate(op->cond);
  auto true_b = this->Mutate(op->true_branch);
  auto false_b = this->Mutate(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b);
  }
}

Expr ExprMutator::VisitExpr_(const TupleGetItemNode* g) {
  auto t = this->Mutate(g->tuple);
  if (g->tuple == t) {
    return GetRef<Expr>(g);
  } else {
    return TupleGetItem(t, g->index);
  }
}

Expr ExprMutator::VisitExpr_(const RefCreateNode* op) {
  Expr value = this->Mutate(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Expr>(op);
  } else {
    return RefCreate(value);
  }
}

Expr ExprMutator::VisitExpr_(const RefReadNode* op) {
  Expr ref = this->Mutate(op->ref);
  if (ref.same_as(op->ref)) {
    return GetRef<Expr>(op);
  } else {
    return RefRead(ref);
  }
}

Expr ExprMutator::VisitExpr_(const RefWriteNode* op) {
  Expr ref = this->Mutate(op->ref);
  Expr value = this->Mutate(op->value);
  if (ref.same_as(op->ref) && value.same_as(op->value)) {
    return GetRef<Expr>(op);
  } else {
    return RefWrite(ref, value);
  }
}

Expr ExprMutator::VisitExpr_(const ConstructorNode* c) { return GetRef<Expr>(c); }

Expr ExprMutator::VisitExpr_(const MatchNode* m) {
  bool unchanged = true;
  std::vector<Clause> clauses;
  for (const Clause& p : m->clauses) {
    Clause c = VisitClause(p);
    clauses.push_back(c);
    unchanged &= c.same_as(p);
  }
  Expr data = Mutate(m->data);
  unchanged &= data.same_as(m->data);
  if (unchanged) {
    return GetRef<Expr>(m);
  }
  return Match(data, clauses, m->complete);
}

Clause ExprMutator::VisitClause(const Clause& c) {
  Pattern p = VisitPattern(c->lhs);
  Expr rhs = Mutate(c->rhs);
  if (p.same_as(c->lhs) && rhs.same_as(c->rhs)) {
    return c;
  }
  return Clause(p, rhs);
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
  if (op->type_annotation.defined()) {
    this->VisitType(op->type_annotation);
  }
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) {}

void ExprVisitor::VisitExpr_(const ConstantNode* op) {}

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  for (auto field : op->fields) {
    this->VisitExpr(field);
  }
}

void ExprVisitor::VisitExpr_(const FunctionNode* op) {
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

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) { this->VisitExpr(op->tuple); }

void ExprVisitor::VisitExpr_(const RefCreateNode* op) { this->VisitExpr(op->value); }

void ExprVisitor::VisitExpr_(const RefReadNode* op) { this->VisitExpr(op->ref); }

void ExprVisitor::VisitExpr_(const RefWriteNode* op) {
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
  std::unordered_set<const Object*> visited_;
};

void PostOrderVisit(const Expr& e, std::function<void(const Expr&)> fvisit) {
  ExprApplyVisit(fvisit).VisitExpr(e);
}

TVM_REGISTER_GLOBAL("relay.analysis.post_order_visit").set_body_typed([](Expr expr, PackedFunc f) {
  PostOrderVisit(expr, [f](const Expr& n) { f(n); });
});

// Implement bind.
class ExprBinder : public ExprMutator, PatternMutator {
 public:
  explicit ExprBinder(const tvm::Map<Var, Expr>& args_map) : args_map_(args_map) {}

  Expr VisitExpr_(const LetNode* op) final {
    CHECK(!args_map_.count(op->var)) << "Cannot bind an internel variable in let";
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    for (Var param : op->params) {
      CHECK(!args_map_.count(param)) << "Cannnot bind an internal function parameter";
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

  Clause VisitClause(const Clause& c) final {
    Pattern pat = VisitPattern(c->lhs);
    return Clause(pat, VisitExpr(c->rhs));
  }

  Var VisitVar(const Var& v) final {
    CHECK(!args_map_.count(v)) << "Cannnot bind an internal pattern variable";
    return v;
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
    if (new_body.same_as(func->body) && new_params.size() == func->params.size()) {
      return expr;
    }
    auto ret = Function(new_params, new_body, func->ret_type, func->type_params, func->attrs);
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> set;
    for (const auto& v : FreeVars(expr)) {
      set.insert(v);
    }
    for (const auto& v : FreeVars(ret)) {
      if (set.count(v) == 0) {
        new_params.push_back(v);
      }
    }
    ret = Function(new_params, new_body, func->ret_type, func->type_params, func->attrs);
    CHECK_EQ(FreeVars(expr).size(), FreeVars(ret).size());
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
    CHECK(input->IsInstance<TypeNode>());
    *ret = Bind(Downcast<Type>(input), args[1]);
  }
});
}  // namespace relay
}  // namespace tvm
