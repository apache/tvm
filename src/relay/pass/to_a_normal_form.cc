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
 * Copyright (c) 2018 by Contributors
 *
 * \file to_anf.cc
 *
 * \brief Turn implicit sharing into observable sharing.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/logging.h>
#include "let_list.h"
#include "../../common/arena.h"
#include "pass_util.h"
#include "dependency_graph.h"

namespace tvm {
namespace relay {

Expr ToANormalForm(const Expr& e,
                   const Module& m,
                   std::unordered_set<GlobalVar, NodeHash, NodeEqual>* gv);

struct ScopeNode;
using Scope = std::shared_ptr<ScopeNode>;

/* Invariant: when parent is null level is 0
 *
 * Invariant: when parent is not null level is 1 + parent->level
 */
struct ScopeNode {
  size_t level;
  Scope parent;
  std::shared_ptr<LetList> ll = std::make_shared<LetList>();
  explicit ScopeNode(const Scope& parent) : level(1 + parent->level), parent(parent) { }
  ScopeNode() : level(0) { }
};

Scope ChildScope(const Scope& s) {
  return std::make_shared<ScopeNode>(s);
}

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

std::unordered_map<DependencyGraph::Node*, Scope> CalcScope(const DependencyGraph& dg) {
  std::unordered_map<DependencyGraph::Node*, Scope> expr_scope;
  Scope global_scope = std::make_shared<ScopeNode>();
  for (auto it = dg.post_dfs_order.rbegin(); it != dg.post_dfs_order.rend(); ++it) {
    DependencyGraph::Node* n = *it;
    auto iit = n->parents.head;
    Scope s;
    if (iit == nullptr) {
      s = global_scope;
    } else {
      s = expr_scope.at(iit->value);
      iit = iit->next;
      for (; iit != nullptr; iit = iit->next) {
        s = LCA(s, expr_scope.at(iit->value));
      }
    }
    expr_scope.insert({n, n->new_scope ? ChildScope(s) : s});
  }
  return expr_scope;
}

bool IsPrimitiveFunction(const Expr& e) {
  return e.as<FunctionNode>() && Downcast<Function>(e)->IsPrimitive();
}

/* Special care is needed to handle local recursion.
 * Fill additionally take a (possibly null) Var argument,
 * If it is not null, Fill is required to bind the transformed result to that var.
 */
class Fill : ExprFunctor<Expr(const Expr&, const Var&)> {
 public:
  static Expr ToANormalForm(const Expr& e,
                            const Module& m,
                            const DependencyGraph& dg,
                            std::unordered_map<DependencyGraph::Node*, Scope>* node_scope,
                            std::unordered_set<GlobalVar, NodeHash, NodeEqual>* gv) {
    Fill fi(m, dg, node_scope, gv);
    return fi.GetScope(e)->ll->Get(fi.VisitExpr(e));
  }

 private:
  Module mod_;
  const DependencyGraph& dg_;
  std::unordered_map<DependencyGraph::Node*, Scope>* node_scope_;
  std::unordered_set<GlobalVar, NodeHash, NodeEqual>* visited_;
  std::unordered_map<Expr, Expr, NodeHash, NodeEqual> memo;

  Fill(Module mod,
       const DependencyGraph& dg,
       std::unordered_map<DependencyGraph::Node*, Scope>* node_scope,
       std::unordered_set<GlobalVar, NodeHash, NodeEqual>* visited) :
    mod_(mod),
    dg_(dg),
    node_scope_(node_scope),
    visited_(visited) { }

  Scope GetScope(const Expr& e) {
    return node_scope_->at(dg_.expr_node.at(e));
  }

  Scope GetSubScope(const Expr& e, size_t i) {
    DependencyGraph::Node* n = dg_.expr_node.at(e);
    auto h = n->children.head;
    while (i != 0) {
      CHECK(h);
      --i;
      h = h->next;
    }
    CHECK(h);
    return node_scope_->at(h->value);
  }

  Expr VisitExpr(const Expr& e, const Var& v) final {
    if (memo.count(e) == 0) {
      memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
    }
    return memo.at(e);
  }

  Expr VisitExpr(const Expr& e) {
    return this->VisitExpr(e, Var());
  }

  Expr Atomic(const Expr& orig, const Expr& now, const Var& v) {
    return v.defined() ? GetScope(orig)->ll->Push(v, now) : now;
  }

  Expr Compound(const Expr& orig, const Expr& now, const Var& v) {
    Var var = v.defined() ?
      v :
      VarNode::make(std::string("x"), IncompleteTypeNode::make(Kind::kType));
    return GetScope(orig)->ll->Push(var, now);
  }

  Expr VisitExpr_(const CallNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    return Compound(e, CallNode::make(VisitExpr(c->op), args, c->attrs, c->type_args), v);
  }

  Expr VisitExpr_(const TupleNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    std::vector<Expr> fields;
    for (const auto& a : t->fields) {
      fields.push_back(VisitExpr(a));
    }
    return Compound(e, TupleNode::make(fields), v);
  }

  Expr VisitExpr_(const TupleGetItemNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    return Compound(e, TupleGetItemNode::make(VisitExpr(t->tuple), t->index), v);
  }

  Expr VisitExpr_(const RefCreateNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefCreateNode::make(VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const RefReadNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefReadNode::make(VisitExpr(r->ref)), v);
  }

  Expr VisitExpr_(const RefWriteNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefWriteNode::make(VisitExpr(r->ref), VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const IfNode* i, const Var& v) final {
    Expr e = GetRef<Expr>(i);
    Expr ret = IfNode::make(VisitExpr(i->cond),
                            GetSubScope(e, 1)->ll->Get(VisitExpr(i->true_branch)),
                            GetSubScope(e, 2)->ll->Get(VisitExpr(i->false_branch)));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const FunctionNode* f, const Var& v) final {
    Expr e = GetRef<Expr>(f);
    Expr ret;
    if (IsPrimitiveFunction(e)) {
      ret = e;
    } else {
      ret = FunctionNode::make(f->params,
                               GetSubScope(e, 0)->ll->Get(VisitExpr(f->body)),
                               f->ret_type,
                               f->type_params,
                               f->attrs);
    }
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const LetNode* l, const Var& v) final {
    Expr e = GetRef<Expr>(l);
    VisitExpr(l->value, l->var);
    Expr ret = GetSubScope(e, 0)->ll->Get(VisitExpr(l->body));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const ConstantNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    return Compound(e, e, v);
  }

  Expr VisitExpr_(const VarNode* vn, const Var& v) final {
    Expr e = GetRef<Expr>(vn);
    return Atomic(e, e, v);
  }

  Expr VisitExpr_(const GlobalVarNode* gvn, const Var& v) final {
    GlobalVar gv = GetRef<GlobalVar>(gvn);
    if (visited_->count(gv) == 0) {
      visited_->insert(gv);
      mod_->Update(gv, Downcast<Function>(relay::ToANormalForm(mod_->Lookup(gv), mod_, visited_)));
    }
    return Atomic(gv, gv, v);
  }

  Expr VisitExpr_(const OpNode* op, const Var& v) final {
    Expr e = GetRef<Expr>(op);
    return Atomic(e, e, v);
  }

  Expr VisitExpr_(const ConstructorNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    return Atomic(e, e, v);
  }

  Expr VisitExpr_(const MatchNode* m, const Var& v) final {
    Expr e = GetRef<Expr>(m);
    Expr data = VisitExpr(m->data);
    std::vector<Clause> clauses;
    for (const Clause& c : m->clauses) {
      clauses.push_back(ClauseNode::make(
        c->lhs,
        GetSubScope(e, 1 + clauses.size())->ll->Get(VisitExpr(c->rhs))));
    }
    return Compound(e, MatchNode::make(data, clauses), v);
  }
};

Expr ToANormalFormAux(const Expr& e,
                      const Module& m,
                      std::unordered_set<GlobalVar, NodeHash, NodeEqual>* gv) {
  /* When you lift a lambda, what is inside is also being lift.
   *
   * So we must determine the scope of the lambda before determining the scope of it's body.
   *
   * To make this more principled,
   * we always determine the scope of parent before determining the scope of children.
   *
   * So we calculate all the dependency between nodes.
   */
  common::Arena arena;
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
  std::unordered_map<DependencyGraph::Node*, Scope> node_scope = CalcScope(dg);
  return Fill::ToANormalForm(e, m, dg, &node_scope, gv);
}

Expr ToANormalForm(const Expr& e,
                   const Module& m,
                   std::unordered_set<GlobalVar, NodeHash, NodeEqual>* gv) {
  DLOG(INFO)
  << "ToANF:" << std::endl
  << AsText(e, false);

  Expr ret =
    TransformF([&](const Expr& e) {
      return ToANormalFormAux(e, m, gv);
    }, e);

  CHECK_EQ(FreeVars(ret).size(), 0);

  DLOG(INFO)
    << "ToANF: transformed" << std::endl
    << AsText(ret, false);

  return ret;
}

Expr ToANormalForm(const Expr& e, const Module& m) {
  std::unordered_set<GlobalVar, NodeHash, NodeEqual> gv;
  return ToANormalForm(e, m, &gv);
}

TVM_REGISTER_API("relay._ir_pass.to_a_normal_form")
.set_body_typed(static_cast<Expr (*)(const Expr&, const Module&)>(ToANormalForm));

}  // namespace relay
}  // namespace tvm
