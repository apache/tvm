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
#include <tvm/support/logging.h>

#include "../../support/arena.h"
#include "../analysis/dependency_graph.h"
#include "let_list.h"
#include "pass_util.h"

namespace tvm {
namespace relay {

/* Special care is needed to handle local recursion.
 * FillBasicBlock additionally take a (possibly null) Var argument,
 * If it is not null, FillBasicBlock is required to bind the transformed result to that var.
 */
class FillBasicBlock : ExprFunctor<Expr(const Expr&, const Var&)> {
 public:
  static Expr ToBasicBlockNormalForm(const Expr& e, const DependencyGraph& dg,
                            std::unordered_map<DependencyGraph::Node*, Scope>* node_scope,
			    std::unordered_set<DependencyGraph::Node*>* lifted) {
    // if (lifted->size() == 0) {
    //   return e;
    // }
    LOG(INFO) << "======================== START FILLING ======================= \n";
    FillBasicBlock fi(dg, node_scope, lifted);
    auto var = fi.VisitExpr(e);
    auto scope = fi.GetScope(e);
    if (!scope->ll->size()) {
      LOG(INFO) << "nothing in scope: " << scope;
      return e;
    }
    auto ret = scope->ll->Get(var);
    LOG(INFO) << "======================== END FILLING ======================= \n ret = " << ret;
    return ret;
  }

 private:
  const DependencyGraph& dg_;
  std::unordered_map<DependencyGraph::Node*, Scope>* node_scope_;
  std::unordered_set<DependencyGraph::Node*>* lifted_;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo;

  FillBasicBlock(const DependencyGraph& dg, std::unordered_map<DependencyGraph::Node*, Scope>* node_scope,
		 std::unordered_set<DependencyGraph::Node*>* lifted)
      : dg_(dg), node_scope_(node_scope), lifted_(lifted) {}

  Scope GetScope(const Expr& e) { return node_scope_->at(dg_.expr_node.at(e)); }

  bool IsLifted(const Expr& e) { return lifted_->find(dg_.expr_node.at(e)) != lifted_->end(); }

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
    LOG(INFO) << "Begin VisitExpr " << e << " ( var = " << v << ")";
    if (memo.count(e) == 0) {
      memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
      LOG(INFO) << "inserted a new entry to memo: " << e << " (var = " << v << ") ---> " << memo.at(e);
    } else if (v.defined()) {
      GetScope(e)->ll->Push(v, memo.at(e));
      LOG(INFO) << "pushed a new entry to ll";
    }
    auto ret = memo.at(e);
    LOG(INFO) << "End VisitExpr " << e << " \n -> \n " << ret;
    return ret;
  }

  Expr VisitExpr(const Expr& e) {
    auto ff = this->VisitExpr(e, Var());
    return ff;
  }

  Expr Atomic(const Expr& e, const Var& v) { return v.defined() ? GetScope(e)->ll->Push(v, e) : e; }

  Expr Compound(const Expr& orig, const Expr& now, const Var& v, bool force = false) {
    // TODO: cannot assume var is allocated
    Var var = v.defined() ? v : Var(String("new_var"), Type());
    // return GetScope(orig)->ll->Push(var, now);
    if (IsLifted(orig) || force) {
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
    auto ff = Compound(e, Call(VisitExpr(c->op), args, c->attrs, c->type_args), v);
    //LOG(INFO) << "visit if " << e << "\n->\n" << ff << "\n";
    return ff;
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
    auto ff = Compound(e, ret, v);
    //LOG(INFO) << "visit if " << e << "\n->\n" << ff;
    return ff;
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
    auto ff = Compound(e, ret, v);
    //LOG(INFO) << "visit func " << e << "\n->\n" << ff;
    return ff;
  }

  Expr VisitExpr_(const LetNode* l, const Var& v) final {
    Expr e = GetRef<Expr>(l);
    DLOG(INFO) << "start to visit let " << e;
    // Expr val = VisitExpr(l->value);
    Expr var = VisitExpr(l->value, l->var);
    // Expr body = VisitExpr(l->body);
    Expr ret = GetSubScope(e, 0)->ll->Get(VisitExpr(l->body), true);
    LOG(INFO) << "visit let body = " << ret;
    // return Let(l->var, l->value, ret);
    // TODO: the LetNode itself was not added to the scope of the original expression
    // LOG(INFO) << "visit let " << e << "\n->\n" << ff;
    // if (lifted_->find(e) == lifted_->end()) {
    //   return Let(l->var, val, ret); // l->value could be in another scope
    // } else {
      auto ff = Compound(e, ret, v);
      return ff;
    //}
  }

  Expr VisitExpr_(const ConstantNode* c, const Var& v) final {
    Expr e = GetRef<Expr>(c);
    LOG(INFO) << "start to visit const " << e;
    auto ret = Compound(e, e, v);
    //LOG(INFO) << "visit const " << e << "\n->\n" << ff;
    return ret;
  }

  Expr VisitExpr_(const VarNode* vn, const Var& v) final {
    Expr e = GetRef<Expr>(vn);
    LOG(INFO) << "start to visit var " << e;
    auto ret = Atomic(e, v);
    //LOG(INFO) << "visit var " << e << "\n->\n" << ff;
    return ret;
  }

  Expr VisitExpr_(const GlobalVarNode* gvn, const Var& v) final {
    GlobalVar gv = GetRef<GlobalVar>(gvn);
    auto ret = Atomic(gv, v);
    //LOG(INFO) << "visit globalvar " << e << "\n->\n" << ff;
    return ret;
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
  std::unordered_set<DependencyGraph::Node*> lifted;
  std::unordered_map<DependencyGraph::Node*, Scope> node_scope = CalcScope(dg, &lifted);
  return FillBasicBlock::ToBasicBlockNormalForm(e, dg, &node_scope, &lifted);
}

IRModule ToBasicBlockNormalForm(const IRModule& m) {
  DLOG(INFO) << "ToBBlock:" << std::endl << m;

  tvm::Map<GlobalVar, Function> updates;
  auto funcs = m->functions;
  for (const auto& it : funcs) {
    CHECK_EQ(FreeVars(it.second).size(), 0);
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
    }
    Expr ret = TransformF([&](const Expr& e) { return ToBasicBlockNormalFormAux(e); }, it.second);
    // CHECK_EQ(FreeVars(ret).size(), 0)
    //     << AsText(ret) << "should not has free vars: " << FreeVars(ret);
    updates.Set(it.first, Downcast<Function>(ret));
  }

  for (auto pair : updates) {
    m->Add(pair.first, pair.second, true);
  }

  DLOG(INFO) << "ToBBlock: transformed" << std::endl << m;

  return m;
}

namespace transform {

Pass ToBasicBlockNormalForm() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::ToBasicBlockNormalForm(m); };
  return CreateModulePass(pass_func, 1, "ToBasicBlockNormalForm", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ToBasicBlockNormalForm").set_body_typed(ToBasicBlockNormalForm);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
