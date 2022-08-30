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
#include "../op/annotation/annotation.h"
#include "./device_aware_visitors.h"
#include "./let_list.h"
#include "./pass_utils.h"

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

namespace {

/* Special care is needed to handle local recursion.
 * Fill additionally take a (possibly null) Var argument,
 * If it is not null, Fill is required to bind the transformed result to that var.
 *
 * ToANormalForm and PlanDevices
 * -----------------------------
 * If PlanDevices has run this transform must respect the lexical scoping rules for the residual
 * "on_device" calls. Eg:
 * \code
 *   on_device(add(subtract(x, y), add(y, z)), device_type=2, is_fixed=true)
 *   ==>
 *   let %x0 = on_device(subtract(x, y), device_type=2, is_fixed=true)
 *   let %x1 = on_device(add(y, z), device_type=2, is_fixed=true)
 *   let %x2 = on_device(add(%x0, %x1), device_type=2, is_fixed=true)
 *   %x2
 * \endcode
 *
 * In addition to conversion to ANF this pass is also handling hoisting implicitly shared
 * sub-expressions to the inner-most scope common to all their uses:
 * \code
 *   on_device(
 *     if y {
 *       on_device(%0, device_type=2, is_fixed=true)
 *     } else {
 *       on_device(subtract(%0, b), device_type=2, is_fixed=true)
 *     },
 *     device_type=1, is_fixed=true)
 *   (where %0 = add(a, b))
 *   ==>
 *   let %x0 = on_device(add(a, b), device_type=2, is_fixed=true);
 *   on_device(
 *     if y {
 *       on_device(%x0, device_type=2, is_fixed=true)
 *     } else {
 *       let %x1 = on_device(subtract(%x0, b), device_type=2, is_fixed=true);
 *       %x1
 *     },
 *     device_type=1, is_fixed=true)
 * \endcode
 * Though the PlanDevices has already avoided inserting "on_device" calls where they are redundant
 * due to lexical scope, it's fiddly to do the same in this pass since the notion of 'scope' is
 * now determined by the scope map. So we'll just insert them mechanically on every let-binding.
 *
 * TODO(mbs): Rewrite to derive from DeviceAwareExprMutator and not track device types
 * explicitly. It's easy to get rid of the need for the extra var argument on VisitExpr by shifting
 * the recursion a '1/2 step' to return a possibly compound expression who's inner expressions are
 * all atomic. However the use of the scope map is currently subtle enough I want to  leave it
 * alone for now.
 */
class Fill : ExprFunctor<Expr(const Expr&, const Var&)>, private transform::LexicalOnDeviceMixin {
 public:
  static Expr ToANormalForm(const Expr& e, const DependencyGraph& dg, NodeScopeMap* node_scope) {
    Fill fi(dg, node_scope, nullptr);
    return fi.GetScope(e)->let_list->Get(fi.VisitExpr(e));
  }

  // For basic block normal form, bind expressions only if the original expression's scope
  // should be lifted
  static Expr ToBasicBlockNormalForm(const Expr& e, const DependencyGraph& dg,
                                     NodeScopeMap* node_scope, ExprSet* lifted) {
    Fill fi(dg, node_scope, lifted);
    return fi.GetScope(e)->let_list->Get(fi.VisitExpr(e));
  }

 private:
  // Note: Conversion to ANF needn't care about the devices for global vars since all that can
  // happen with them is to go from:
  //    ...@g...
  // to:
  //    let %x = @g;
  //    ...
  //    ...%x...
  // In that case the code will ask  for the device for @g, get kInvalidDeviceType, then
  // MaybeOnDevice @g, which is always a no-op.
  Fill(const DependencyGraph& dg, NodeScopeMap* node_scope, ExprSet* include_set)
      : transform::LexicalOnDeviceMixin(Optional<IRModule>()),
        dg_(dg),
        node_scope_(node_scope),
        include_set_(include_set) {}

  Scope GetScope(const Expr& e) { return node_scope_->at(dg_.expr_node.at(e)); }

  Scope GetSubScope(const Expr& e, size_t i) {
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

  Expr VisitExpr(const Expr& e) { return this->VisitExpr(e, Var()); }

  Expr VisitExpr(const Expr& e, const Var& v) final {
    if (memo.count(e) == 0) {
      memo.insert({e, ExprFunctor<Expr(const Expr&, const Var&)>::VisitExpr(e, v)});
    } else if (v.defined()) {
      GetScope(e)->let_list->Push(v, memo.at(e));
    }
    auto ret = memo.at(e);
    // if no include_set is specified, every expression should be atomic.
    // TODO(mbs): Note that Constants must be let-bound even though they are considered 'atomic'
    // by this test.
    if (include_set_ == nullptr && function_nesting() > 0) {
      ICHECK(IsAtomic(ret)) << "expression:" << std::endl << PrettyPrint(ret);
    }
    return ret;
  }

  Expr Atomic(const Expr& e, const Var& v) {
    Expr annotated_expr = MaybeOnDeviceFixed(e, GetVirtualDevice(e));
    return v.defined() ? GetScope(e)->let_list->Push(v, annotated_expr) : annotated_expr;
  }

  // Bind expression `now` to var `v` if the original expression is in the include set, or if
  // v is already defined (e.g. coming from a Let expression). Otherwise return `now` directly
  Expr Compound(const Expr& orig, const Expr& now, const Var& v) {
    Expr annotated_expr = MaybeOnDeviceFixed(now, GetVirtualDevice(orig));
    Var var = v.defined() ? v : Var::GenSym();
    bool not_included = include_set_ && include_set_->find(orig) == include_set_->end();
    if (!v.defined() && not_included) {
      return annotated_expr;
    } else if (const LetNode* let = AsIgnoringOnDevice<LetNode>(now)) {
      // Instead of making a nested binding "let var = (let x = ...; bindings...; body)", we push
      // the inner bindings into the outer scope and bind body to var, giving
      // "let x = ...; bindings...; let var = body;" as the resulting bindings.
      Expr e = GetRef<Expr>(let);
      while (const LetNode* inner_let = AsIgnoringOnDevice<LetNode>(e)) {
        GetScope(orig)->let_list->Push(inner_let->var, inner_let->value);
        e = inner_let->body;
      }
      Expr annotated_body = MaybeOnDeviceFixed(e, GetVirtualDevice(orig));
      return GetScope(orig)->let_list->Push(var, annotated_body);
    } else {
      return GetScope(orig)->let_list->Push(var, annotated_expr);
    }
  }

  Expr VisitExpr_(const CallNode* c, const Var& v) final {
    OnDeviceProps props = GetOnDeviceProps(c);
    if (props.body.defined() && props.is_fixed()) {
      // Keep track of expression device type for lexically enclosing sub-expressions.
      PushVirtualDevice(props.virtual_device);
      Expr body = VisitExpr(props.body, v);
      // We are done with this sub-expression.
      PopVirtualDevice();
      // Preserve the "on_device" annotations.
      return OnDeviceWithProps(body, props);
    }

    Expr e = GetRef<Expr>(c);
    std::vector<Expr> args;
    for (const auto& a : c->args) {
      args.push_back(VisitExpr(a));
    }
    return Compound(e, Call(VisitExpr(c->op), args, c->attrs, c->type_args), v);
  }

  Expr VisitExpr_(const TupleNode* tuple_node, const Var& v) final {
    Expr e = GetRef<Expr>(tuple_node);
    Array<Expr> fields;
    fields.reserve(tuple_node->fields.size());
    for (const auto& a : tuple_node->fields) {
      fields.push_back(VisitExpr(a));
    }
    return Compound(e, WithFields(GetRef<Tuple>(tuple_node), fields), v);
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
    Expr ret = If(VisitExpr(i->cond), GetSubScope(e, 1)->let_list->Get(VisitExpr(i->true_branch)),
                  GetSubScope(e, 2)->let_list->Get(VisitExpr(i->false_branch)));
    return Compound(e, ret, v);
  }

  Expr VisitExpr_(const FunctionNode* f, const Var& v) final {
    Expr e = GetRef<Expr>(f);
    Expr ret;
    if (f->HasNonzeroAttr(attr::kPrimitive)) {
      ret = e;
    } else {
      // Keep track of expression and bound variable device types for lexically enclosing
      // sub-expressions.
      PushVirtualDevice(f->virtual_device());
      for (auto param : f->params) {
        PushBoundVar(param, param->virtual_device());
      }
      EnterFunctionBody();
      ret = WithFields(GetRef<Function>(f), f->params,
                       GetSubScope(e, 0)->let_list->Get(VisitExpr(f->body)));
      // We are done with this function.
      ExitFunctionBody();
      for (size_t i = 0; i < f->params.size(); ++i) {
        PopBoundVar(f->params[i]);
      }
      PopVirtualDevice();
    }
    if (function_nesting() == 0) {
      ICHECK(!v.defined());
      // This is a global function which can be bound directly in the module.
      return ret;
    } else {
      // This is a local function which must be let-bound.
      return Compound(e, ret, v);
    }
  }

  Expr VisitExpr_(const LetNode* l, const Var& v) final {
    Expr e = GetRef<Expr>(l);
    // Keep track of bound variable device types for lexically enclosing sub-expressions.
    PushBoundVar(l->var, GetVirtualDevice(l->value));
    VisitExpr(l->value, l->var);
    Expr ret = GetSubScope(e, 0)->let_list->Get(VisitExpr(l->body));
    // We are done with these sub-expressions.
    PopBoundVar(l->var);
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
      clauses.emplace_back(c->lhs,
                           GetSubScope(e, 1 + clauses.size())->let_list->Get(VisitExpr(c->rhs)));
    }
    return Compound(e, Match(data, clauses, m->complete), v);
  }

  const DependencyGraph& dg_;
  NodeScopeMap* node_scope_ = nullptr;
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo;
  // a set of Expressions to include for let bindings. If set to nullptr
  // all Exprs will be pushed to the let list.
  ExprSet* include_set_ = nullptr;
};

IRModule ModuleToANormalForm(const IRModule& mod) {
  tvm::Map<GlobalVar, Function> updates;
  auto funcs = mod->functions;
  for (const auto& it : funcs) {
    ICHECK_EQ(FreeVars(it.second).size(), 0);
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);
      Function ret = Downcast<Function>(transform::ToANormalForm(func));
      ICHECK_EQ(FreeVars(ret).size(), 0) << "rewritten:" << std::endl
                                         << PrettyPrint(ret) << std::endl
                                         << "should not have free vars: " << FreeVars(ret);
      VLOG(1) << "rewritten:" << std::endl
              << PrettyPrint(func) << std::endl
              << "to ANF:" << std::endl
              << PrettyPrint(ret);
      updates.Set(it.first, ret);
    }
  }

  for (auto pair : updates) {
    mod->Add(pair.first, pair.second, true);
  }

  return mod;
}

}  // namespace

Expr ToBasicBlockNormalFormAux(const Expr& e) {
  // calculate all the dependency between nodes.
  support::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, e);
  /* The scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   * We also record the set of expressions whose scope is lifted.
   */
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return Fill::ToBasicBlockNormalForm(e, dg, &scopes.first, &scopes.second);
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
      [=](IRModule m, PassContext pc) { return ModuleToANormalForm(m); };
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
