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
 * \file to_basic_block_normal_form.cc
 *
 * \brief Turn an expression to the basic normal form.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/support/logging.h>

#include "../../support/arena.h"
#include "../analysis/dependency_graph.h"
#include "let_list.h"
#include "pass_util.h"

namespace tvm {
namespace relay {

/* Fill expressions based on each scope's let list. Different from FillANF,
 * only expressions with lifted scope will be pushed to the let list.
 */
class FillBasicBlock : ExprFunctor<Expr(const Expr&, const Var&)> {
 public:
  static Expr ToBasicBlockNormalForm(const Expr& e, const DependencyGraph& dg,
                                     NodeScopeMap* node_scope, ExprSet* lifted) {
    FillBasicBlock fi(dg, node_scope, lifted);
    auto var = fi.VisitExpr(e);
    return fi.GetScope(e)->ll->Get(var);
  }

 private:
  const DependencyGraph& dg_;
  NodeScopeMap* node_scope_;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo;
  ExprSet* lifted_;

  FillBasicBlock(const DependencyGraph& dg, NodeScopeMap* node_scope, ExprSet* lifted)
      : dg_(dg), node_scope_(node_scope), lifted_(lifted) {}

  Scope GetScope(const Expr& e) { return node_scope_->at(dg_.expr_node.at(e)); }

  Scope GetSubScope(const Expr& e, size_t i) {
    DependencyGraph::Node* n = dg_.expr_node.at(e);
    auto h = n->children.head;
    while (i != 0) {
      CHECK(h) << i << "-th dependency is not defined";
      --i;
      h = h->next;
    }
    CHECK(h);
    return node_scope_->at(h->value);
  }

  Expr VisitExpr(const Expr& e, const Var& v) final {
    if (memo.count(e) == 0) {
      memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
    } else if (v.defined()) {
      GetScope(e)->ll->Push(v, memo.at(e));
    }
    return memo.at(e);
  }

  Expr VisitExpr(const Expr& e) { return this->VisitExpr(e, Var()); }

  Expr Atomic(const Expr& e, const Var& v) { return v.defined() ? GetScope(e)->ll->Push(v, e) : e; }

  // Bind expression `now` to var `v` if the original expression's scope should be lifted, or
  // if v is defined (e.g. coming from a Let expression). Otherwise return `now` directly.
  Expr Compound(const Expr& orig, const Expr& now, const Var& v) {
    Var var = v.defined() ? v : Var(String("x"), Type());
    if (v.defined() || lifted_->find(orig) != lifted_->end()) {
      return GetScope(orig)->ll->Push(var, now);
    } else {
      return now;
    }
  }

  Expr VisitExpr_(const CallNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    return Compound(e, Call(VisitExpr(c->op), args, c->attrs, c->type_args), v);
  }

  Expr VisitExpr_(const TupleNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    std::vector<Expr> fields;
    for (const auto& a : t->fields) {
      fields.push_back(VisitExpr(a));
    }
    return Compound(e, Tuple(fields), v);
  }

  Expr VisitExpr_(const TupleGetItemNode* t, const Var& v) final {
    Expr e = GetRef<Expr>(t);
    return Compound(e, TupleGetItem(VisitExpr(t->tuple), t->index), v);
  }

  Expr VisitExpr_(const RefCreateNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefCreate(VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const RefReadNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefRead(VisitExpr(r->ref)), v);
  }

  Expr VisitExpr_(const RefWriteNode* r, const Var& v) final {
    Expr e = GetRef<Expr>(r);
    return Compound(e, RefWrite(VisitExpr(r->ref), VisitExpr(r->value)), v);
  }

  Expr VisitExpr_(const IfNode* i, const Var& v) final {
    Expr e = GetRef<Expr>(i);
    Expr ret = If(VisitExpr(i->cond), GetSubScope(e, 1)->ll->Get(VisitExpr(i->true_branch)),
                  GetSubScope(e, 2)->ll->Get(VisitExpr(i->false_branch)));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const FunctionNode* f, const Var& v) final {
    Expr e = GetRef<Expr>(f);
    Expr ret;
    if (f->HasNonzeroAttr(attr::kPrimitive)) {
      ret = e;
    } else {
      ret = Function(f->params, GetSubScope(e, 0)->ll->Get(VisitExpr(f->body)), f->ret_type,
                     f->type_params, f->attrs);
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
    return Atomic(e, v);
  }

  Expr VisitExpr_(const GlobalVarNode* gvn, const Var& v) final {
    GlobalVar gv = GetRef<GlobalVar>(gvn);
    return Atomic(gv, v);
  }

  Expr VisitExpr_(const OpNode* op, const Var& v) final {
    Expr e = GetRef<Expr>(op);
    return Atomic(e, v);
  }

  Expr VisitExpr_(const ConstructorNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    return Atomic(e, v);
  }

  Expr VisitExpr_(const MatchNode* m, const Var& v) final {
    Expr e = GetRef<Expr>(m);
    Expr data = VisitExpr(m->data);
    std::vector<Clause> clauses;
    for (const Clause& c : m->clauses) {
      clauses.push_back(
          Clause(c->lhs, GetSubScope(e, 1 + clauses.size())->ll->Get(VisitExpr(c->rhs))));
    }
    return Compound(e, Match(data, clauses, m->complete), v);
  }
};

Expr ToBasicBlockNormalFormAux(const Expr& e) {
  // calculate all the dependency between nodes.
  support::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, e);
  /* The scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   * We also record the set of expressions whose scope is lifted.
   */
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return FillBasicBlock::ToBasicBlockNormalForm(e, dg, &scopes.first, &scopes.second);
}

IRModule ToBasicBlockNormalForm(const IRModule& mod) {
  DLOG(INFO) << "ToBBlock:" << std::endl << mod;

  tvm::Map<GlobalVar, Function> updates;
  auto funcs = mod->functions;
  for (const auto& it : funcs) {
    CHECK_EQ(FreeVars(it.second).size(), 0) << "Expected no free variables";
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
    }
    Expr ret = TransformF([&](const Expr& e) { return ToBasicBlockNormalFormAux(e); }, it.second);
    updates.Set(it.first, Downcast<Function>(ret));
  }

  for (auto pair : updates) {
    mod->Add(pair.first, pair.second, true);
  }

  DLOG(INFO) << "ToBBlock: transformed" << std::endl << mod;

  return mod;
}

bool BasicBlockNormalFormCheck(const Expr& e) {
  // calculate all the dependency between nodes.
  support::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, e);
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return scopes.second.size() == 0;
}

TVM_REGISTER_GLOBAL("relay.analysis.check_basic_block_normal_form")
    .set_body_typed(BasicBlockNormalFormCheck);

namespace transform {

Pass ToBasicBlockNormalForm() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::ToBasicBlockNormalForm(m); };
  return CreateModulePass(pass_func, 1, "ToBasicBlockNormalForm", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ToBasicBlockNormalForm")
    .set_body_typed(ToBasicBlockNormalForm);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
