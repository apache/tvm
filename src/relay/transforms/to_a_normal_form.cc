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
 *
 * \file to_a_normal_form.cc
 *
 * \brief Turn implicit sharing into observable sharing.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include "../../support/arena.h"
#include "../analysis/dependency_graph.h"
#include "let_list.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {

Scope LCA(Scope lhs, Scope rhs) {
  while (lhs != rhs) {
    if (lhs->level > rhs->level) {
      lhs = lhs->parent;
    } else if (lhs->level < rhs->level) {
      rhs = rhs->parent;
    } else {
      lhs = lhs->parent;
      rhs = rhs->parent;
    }
  }
  return lhs;
}

std::pair<NodeScopeMap, ExprSet> CalcScope(const DependencyGraph& dg) {
  NodeScopeMap expr_scope;
  ExprSet lifted_exprs;
  std::unordered_map<DependencyGraph::Node*, Expr> node_to_expr;
  for (auto expr_node : dg.expr_node) {
    node_to_expr[expr_node.second] = expr_node.first;
  }
  bool global_scope_used = false;
  Scope global_scope = std::make_shared<ScopeNode>();

  for (auto it = dg.post_dfs_order.rbegin(); it != dg.post_dfs_order.rend(); ++it) {
    DependencyGraph::Node* n = *it;
    auto iit = n->parents.head;
    Scope s;
    if (iit == nullptr) {
      ICHECK(!global_scope_used);
      s = global_scope;
      global_scope_used = true;
    } else {
      s = expr_scope.at(iit->value);
      const auto original_s = s;
      iit = iit->next;
      for (; iit != nullptr; iit = iit->next) {
        s = LCA(s, expr_scope.at(iit->value));
      }
      if (s != original_s && node_to_expr.find(n) != node_to_expr.end()) {
        // filter out exprs whose scope do not matter
        Expr expr = node_to_expr[n];
        if (!expr.as<OpNode>()) {
          lifted_exprs.insert(expr);
        }
      }
    }
    if (n->new_scope) {
      auto child_scope = std::make_shared<ScopeNode>(s);
      expr_scope.insert({n, child_scope});
    } else {
      expr_scope.insert({n, s});
    }
  }
  ICHECK(global_scope_used);
  return std::make_pair(expr_scope, lifted_exprs);
}

Expr Fill::ToANormalForm(const Expr& e, const DependencyGraph& dg, NodeScopeMap* node_scope) {
  Fill fi(dg, node_scope, nullptr);
  return fi.GetScope(e)->let_list->Get(fi.VisitExpr(e));
}

// For basic block normal form, bind expressions only if the original expression's scope
// should be lifted
Expr Fill::ToBasicBlockNormalForm(const Expr& e, const DependencyGraph& dg,
                                  NodeScopeMap* node_scope, ExprSet* lifted) {
  Fill fi(dg, node_scope, lifted);
  auto var = fi.VisitExpr(e);
  return fi.GetScope(e)->let_list->Get(var);
}

Scope Fill::GetScope(const Expr& e) { return node_scope_->at(dg_.expr_node.at(e)); }

Scope Fill::GetSubScope(const Expr& e, size_t i) {
  DependencyGraph::Node* n = dg_.expr_node.at(e);
  auto h = n->children.head;
  while (i != 0) {
    ICHECK(h);
    --i;
    h = h->next;
  }
  ICHECK(h);
  return node_scope_->at(h->value);
}

Expr Fill::VisitExpr(const Expr& e, const Var& v) {
  if (memo.count(e) == 0) {
    memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
  } else if (v.defined()) {
    GetScope(e)->let_list->Push(v, memo.at(e));
  }
  auto ret = memo.at(e);
  // if no include_set is specified, every expression should be atomic.
  if (include_set_ == nullptr) ICHECK(IsAtomic(ret));
  return ret;
}

Expr Fill::VisitExpr(const Expr& e) { return this->VisitExpr(e, Var()); }

Expr Fill::Atomic(const Expr& e, const Var& v) {
  return v.defined() ? GetScope(e)->let_list->Push(v, e) : e;
}

// Bind expression `now` to var `v` if the original expression is in the include set, or if
// v is already defined (e.g. coming from a Let expression). Otherwise return `now` directly
Expr Fill::Compound(const Expr& orig, const Expr& now, const Var& v) {
  Var var = v.defined() ? v : Var(String("x"), Type());
  bool not_included = include_set_ && include_set_->find(orig) == include_set_->end();
  if (!v.defined() && not_included) {
    return now;
  } else {
    return GetScope(orig)->let_list->Push(var, now);
  }
}

Expr Fill::VisitExpr_(const CallNode* c, const Var& v) {
  Expr e = GetRef<Expr>(c);
  std::vector<Expr> args;
  for (const auto& a : c->args) {
    args.push_back(VisitExpr(a));
  }
  return Compound(e, Call(VisitExpr(c->op), args, c->attrs, c->type_args), v);
}

Expr Fill::VisitExpr_(const TupleNode* t, const Var& v) {
  Expr e = GetRef<Expr>(t);
  std::vector<Expr> fields;
  for (const auto& a : t->fields) {
    fields.push_back(VisitExpr(a));
  }
  return Compound(e, Tuple(fields), v);
}

Expr Fill::VisitExpr_(const TupleGetItemNode* t, const Var& v) {
  Expr e = GetRef<Expr>(t);
  return Compound(e, TupleGetItem(VisitExpr(t->tuple), t->index), v);
}

Expr Fill::VisitExpr_(const RefCreateNode* r, const Var& v) {
  Expr e = GetRef<Expr>(r);
  return Compound(e, RefCreate(VisitExpr(r->value)), v);
}

Expr Fill::VisitExpr_(const RefReadNode* r, const Var& v) {
  Expr e = GetRef<Expr>(r);
  return Compound(e, RefRead(VisitExpr(r->ref)), v);
}

Expr Fill::VisitExpr_(const RefWriteNode* r, const Var& v) {
  Expr e = GetRef<Expr>(r);
  return Compound(e, RefWrite(VisitExpr(r->ref), VisitExpr(r->value)), v);
}

Expr Fill::VisitExpr_(const IfNode* i, const Var& v) {
  Expr e = GetRef<Expr>(i);
  Expr ret = If(VisitExpr(i->cond), GetSubScope(e, 1)->let_list->Get(VisitExpr(i->true_branch)),
                GetSubScope(e, 2)->let_list->Get(VisitExpr(i->false_branch)));
  return Compound(e, ret, v);
}

Expr Fill::VisitExpr_(const FunctionNode* f, const Var& v) {
  Expr e = GetRef<Expr>(f);
  Expr ret;
  if (f->HasNonzeroAttr(attr::kPrimitive)) {
    ret = e;
  } else {
    ret = Function(f->params, GetSubScope(e, 0)->let_list->Get(VisitExpr(f->body)), f->ret_type,
                   f->type_params, f->attrs);
  }
  return Compound(e, ret, v);
}

Expr Fill::VisitExpr_(const LetNode* l, const Var& v) {
  Expr e = GetRef<Expr>(l);
  VisitExpr(l->value, l->var);
  Expr ret = GetSubScope(e, 0)->let_list->Get(VisitExpr(l->body));
  return Compound(e, ret, v);
}

Expr Fill::VisitExpr_(const ConstantNode* c, const Var& v) {
  Expr e = GetRef<Expr>(c);
  return Compound(e, e, v);
}

Expr Fill::VisitExpr_(const VarNode* vn, const Var& v) {
  Expr e = GetRef<Expr>(vn);
  return Atomic(e, v);
}

Expr Fill::VisitExpr_(const GlobalVarNode* gvn, const Var& v) {
  GlobalVar gv = GetRef<GlobalVar>(gvn);
  return Atomic(gv, v);
}

Expr Fill::VisitExpr_(const OpNode* op, const Var& v) {
  Expr e = GetRef<Expr>(op);
  return Atomic(e, v);
}

Expr Fill::VisitExpr_(const ConstructorNode* c, const Var& v) {
  Expr e = GetRef<Expr>(c);
  return Atomic(e, v);
}

Expr Fill::VisitExpr_(const MatchNode* m, const Var& v) {
  Expr e = GetRef<Expr>(m);
  Expr data = VisitExpr(m->data);
  std::vector<Clause> clauses;
  for (const Clause& c : m->clauses) {
    clauses.push_back(
        Clause(c->lhs, GetSubScope(e, 1 + clauses.size())->let_list->Get(VisitExpr(c->rhs))));
  }
  return Compound(e, Match(data, clauses, m->complete), v);
}

IRModule ToANormalForm(const IRModule& m) {
  DLOG(INFO) << "ToANF:" << std::endl << m;

  tvm::Map<GlobalVar, Function> updates;
  auto funcs = m->functions;
  for (const auto& it : funcs) {
    ICHECK_EQ(FreeVars(it.second).size(), 0);
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
    }
    Expr ret = TransformF([&](const Expr& e) { return transform::ToANormalForm(e); }, it.second);
    ICHECK_EQ(FreeVars(ret).size(), 0)
        << AsText(ret) << "should not has free vars: " << FreeVars(ret);
    updates.Set(it.first, Downcast<Function>(ret));
  }

  for (auto pair : updates) {
    m->Add(pair.first, pair.second, true);
  }

  DLOG(INFO) << "ToANF: transformed" << std::endl << m;

  return m;
}

namespace transform {

Expr ToANormalForm(const Expr& e) {
  /* When you lift a lambda, what is inside is also being lift.
   *
   * So we must determine the scope of the lambda before determining the scope of it's body.
   *
   * To make this more principled,
   * we always determine the scope of parent before determining the scope of children.
   *
   * So we calculate all the dependency between nodes.
   */
  support::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, e);
  /* In order to model new subscopes created by lambda, if else and pattern matching,
   * we also assign scope to edge as well.
   * The scope of an edge is either the parent's scope, or a new subscope of the parent's scope.
   *
   * So, the scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   *
   * Every scope additionally contain a LetList which collect all value of that scope.
   * We do an additional pass to fill all the LetList and we are done.
   */
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return Fill::ToANormalForm(e, dg, &scopes.first);
}

Pass ToANormalForm() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::ToANormalForm(m); };
  return CreateModulePass(pass_func, 1, "ToANormalForm", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ToANormalForm").set_body_typed([]() {
  return ToANormalForm();
});

TVM_REGISTER_GLOBAL("relay._transform.ToANormalFormExpr").set_body_typed([](const Expr& e) {
  return ToANormalForm(e);
});

}  // namespace transform

}  // namespace relay
}  // namespace tvm
