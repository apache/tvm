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
 * \file src/relay/backend/vm/plan_memory.cc
 * \brief Tensor and storage liveness analysis and memory planning.
 */

#include <tvm/relay/transform.h>

#include "../../../support/arena.h"
#include "../../transforms/device_aware_visitors.h"
#include "../../transforms/let_list.h"

namespace tvm {
namespace relay {
namespace transform {

using support::LinkedList;
using support::LinkNode;

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

class ControlFlowGraph {
 public:
  struct Node {
    LinkedList<Node*> pred;
    LinkedList<Node*> succ;
    Expr expr;
  };

  std::unordered_map<Expr, Node*, ObjectPtrHash, ObjectPtrEqual> let_map;
  std::vector<Node*> reverse_post_order;
  // Node* entry;

  static ControlFlowGraph Create(support::Arena* arena, const Expr& body);

 private:
  class Creator;
};

using NodeList = std::vector<ControlFlowGraph::Node*>;

class ControlFlowGraph::Creator : private ExprFunctor<ControlFlowGraph::Node*(
                                      const Expr&, const NodeList&)> {
 public:
  Creator(support::Arena* arena) : arena_(arena) {}

  ControlFlowGraph Create(const Expr& body) {
    VisitExpr(body, {});
    return std::move(cfg_);
  }

 private:
  support::Arena* arena_;
  ControlFlowGraph cfg_;
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visited_;
  bool in_func_ = false;

  void Succ(Node* from, Node* to) {
    auto succ_link = arena_->make<LinkNode<Node*>>();
    succ_link->value = to;
    from->succ.Push(succ_link);

    auto pred_link = arena_->make<LinkNode<Node*>>();
    pred_link->value = from;
    to->pred.Push(pred_link);
  }

#define DEFAULT_CFG(OP)                                         \
  Node* VisitExpr_(const OP* op, const NodeList& preds) final { \
    Node* n = arena_->make<Node>();                             \
    n->expr = GetRef<Expr>(op);                                 \
    for (Node * pred : preds) {                                 \
      Succ(pred, n);                                            \
    }                                                           \
    cfg_.reverse_post_order.push_back(n);                       \
    return n;                                                   \
  }

  Node* VisitExpr_(const FunctionNode* f, const NodeList& preds) final {
    ICHECK(!in_func_) << "nested functions not supported by CFG analysis";
    in_func_ = true;

    if (f->HasNonzeroAttr(attr::kClosure)) {
      ICHECK(f->body.as<FunctionNode>());
      return VisitExpr(Downcast<Function>(f->body)->body, {});
    }

    // cfg_.entry = arena_->make<Node>();
    // Succ(cfg_.entry, VisitExpr(f->body));

    return VisitExpr(f->body, {});
  }

  Node* VisitExpr_(const LetNode* let_node, const NodeList& let_preds) final {
    Expr expr = GetRef<Expr>(let_node);
    NodeList preds = let_preds;

    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      Node* curr_node = arena_->make<Node>();
      curr_node->expr = expr;
      ICHECK(!cfg_.let_map.count(expr));
      cfg_.let_map[expr] = curr_node;

      // 2 predecessors if last let bound value was an If, else 1
      for (Node* pred : preds) {
        Succ(pred, curr_node);
      }

      cfg_.reverse_post_order.push_back(curr_node);
      if (const IfNode* ite = AsIgnoringOnDevice<IfNode>(inner_let_node->value)) {
        Node* t_node = VisitExpr(ite->true_branch, {curr_node});
        Node* f_node = VisitExpr(ite->false_branch, {curr_node});
        preds = {t_node, f_node};
      } else {
        preds = {curr_node};
      }
      expr = inner_let_node->body;
    }

    Node* body_node = VisitExpr(expr, preds);

    return body_node;
  }

  DEFAULT_CFG(VarNode);
  DEFAULT_CFG(GlobalVarNode);
  DEFAULT_CFG(ConstantNode);
  DEFAULT_CFG(IfNode);
  DEFAULT_CFG(CallNode);
  DEFAULT_CFG(OpNode);
  DEFAULT_CFG(TupleNode);
  DEFAULT_CFG(TupleGetItemNode);
};

ControlFlowGraph ControlFlowGraph::Create(support::Arena* arena, const Expr& body) {
  return Creator(arena).Create(body);
}

// 
class LivenessAnalyzer : private ExprFunctor<ControlFlowGraph::Node*(const Expr& e, ControlFlowGraph::Node*)> {
 private:
  using CFG = ControlFlowGraph;

 public:
  LivenessAnalyzer() {}

  // see https://lambda.uta.edu/cse5317/notes/node40.html for an overview of the algorithm
  void ComputeLiveness(const Expr& expr) {
    cfg_ = CFG::Create(&arena_, expr);
    VisitExpr(expr, nullptr);

    bool did_work = true;

    auto visitor = [&](CFG::Node* n) {
      VarSet old_in_n = this->live_in_[n];
      VarSet old_out_n = this->live_out_[n];

      this->live_in_[n] = this->use_[n];
      for (const Var& v : this->live_out_[n]) {
        if (!v.same_as(this->def_[n])) {
          this->live_in_[n].insert(v);
        }
      }

      this->live_out_[n] = VarSet();
      auto s = n->succ.head;
      while (s) {
        CFG::Node* s_node = s->value;
        this->live_out_[n].insert(this->live_in_[s_node].begin(), this->live_in_[s_node].end());
        s = s->next;
      }

      if (!SetEqual(old_in_n, this->live_in_[n])) {
        did_work = true;
      } else if (!SetEqual(old_out_n, this->live_out_[n])) {
        did_work = true;
      }
    };

    while (did_work) {
      did_work = false;
      for (auto it = cfg_.reverse_post_order.rbegin(); it != cfg_.reverse_post_order.rend(); ++it) {
        visitor(*it);
      }
    }
  }

 private:
  CFG::Node* VisitExpr_(const LetNode* let_node, CFG::Node* cfg_node) override {
    Expr expr = GetRef<Expr>(let_node);
    ICHECK(!cfg_node || cfg_node == cfg_.let_map[expr]) << cfg_node->expr << std::endl <<
    std::endl << cfg_.let_map[expr]->expr;

    while (const auto* inner_let_node = expr.as<LetNode>()) {
      const Var& var = inner_let_node->var;
      const Expr& value = inner_let_node->value;

      ICHECK(!cfg_node || cfg_node == cfg_.let_map[expr]) << cfg_node->expr << std::endl <<
      std::endl << cfg_.let_map[expr]->expr; cfg_node = cfg_.let_map[expr];

      // ICHECK(!alias_.count(var));
      // if (value.as<VarNode>()) {
      //   Var rhs = Downcast<Var>(value);
      //   ICHECK(alias_.count(rhs));
      //   alias_[var] = alias_[rhs];
      //   alias_[rhs]->insert(var);
      // } else {
      //   alias_[var] = std::make_shared<VarSet>();
      //   alias_[var]->insert(var);
      // }

      cfg_node = cfg_.let_map[expr];
      def_[cfg_node] = var;

      if (const IfNode* ite = AsIgnoringOnDevice<IfNode>(value)) {
        VisitExpr_(ite, cfg_node);

        // there should be exactly two successors: the true branch then the false branch
        ICHECK(cfg_node->succ.head);
        ICHECK(cfg_node->succ.head->next);
        ICHECK(!cfg_node->succ.head->next->next);
        CFG::Node* t_entry = cfg_node->succ.head->value;
        CFG::Node* f_entry = cfg_node->succ.head->next->value;

        CFG::Node* t_exit = VisitExpr(ite->true_branch, t_entry);
        CFG::Node* f_exit = VisitExpr(ite->false_branch, f_entry);

        // each branch should have exactly one succcessor, and it should be the same
        ICHECK(t_exit->succ.head && !t_exit->succ.head->next);
        ICHECK(f_exit->succ.head && !f_exit->succ.head->next);
        ICHECK(t_exit->succ.head->value == f_exit->succ.head->value);
        cfg_node = t_exit->succ.head->value;
        if (inner_let_node->body.as<LetNode>()) {
          ICHECK(cfg_node->expr.same_as(inner_let_node->body));
          ICHECK(cfg_.let_map[inner_let_node->body] == cfg_node);
        }
      } else {
        VisitExpr(value, cfg_node);

        // normal bindings should have just one successor
        ICHECK(cfg_node->succ.head && !cfg_node->succ.head->next);
        cfg_node = cfg_node->succ.head->value;
        if (inner_let_node->body.as<LetNode>()) {
          ICHECK(cfg_node->expr.same_as(inner_let_node->body));
          ICHECK(cfg_.let_map[inner_let_node->body] == cfg_node);
        }
      }

      expr = inner_let_node->body;
    }

    return VisitExpr(expr, cfg_node);
  }

  CFG::Node* VisitExpr_(const IfNode* if_node, CFG::Node* cfg_node) override {
    VisitExpr(if_node->cond, cfg_node);
    return cfg_node;
  }

  CFG::Node* VisitExpr_(const TupleNode* tuple_node, CFG::Node* cfg_node) override {
    for (const Expr& field : tuple_node->fields) {
      VisitExpr(field, cfg_node);
    }
    return cfg_node;
  }

  CFG::Node* VisitExpr_(const TupleGetItemNode* get_node, CFG::Node* cfg_node) override {
    VisitExpr(get_node->tuple, cfg_node);
    return cfg_node;
  }

  CFG::Node* VisitExpr_(const GlobalVarNode* global_var_node, CFG::Node* cfg_node) override {return cfg_node;}

  CFG::Node* VisitExpr_(const VarNode* var_node, CFG::Node* cfg_node) override {
    Var var = GetRef<Var>(var_node);
    // ICHECK(alias_.count(var));
    use_[cfg_node].insert(var);
    return cfg_node;
  }

  CFG::Node* VisitExpr_(const ConstantNode* const_node, CFG::Node* cfg_node) override {return cfg_node;}

  CFG::Node* VisitExpr_(const OpNode* op_node, CFG::Node* cfg_node) override {return cfg_node;}

  CFG::Node* VisitExpr_(const CallNode* call_node, CFG::Node* cfg_node) override {
    VisitExpr(call_node->op, cfg_node);
    for (const Expr& arg : call_node->args) {
      VisitExpr(arg, cfg_node);
    }
    return cfg_node;
  }

  CFG::Node* VisitExpr_(const FunctionNode* func_node, CFG::Node* cfg_node) override { 
    ICHECK(!used_);
    used_ = true;

    if (func_node->HasNonzeroAttr(attr::kPrimitive)) {
      return nullptr;
    }

    // TODO(@altanh): figure out the closure nesting thing
    // ICHECK(!function_node->HasNonzeroAttr(attr::kClosure)) << "closures not supported yet";
    ICHECK(func_depth_ == 0) << "nested functions should have been transformed away";

    Expr body = func_node->body;
    if (func_node->HasNonzeroAttr(attr::kClosure)) {
      ICHECK(body.as<FunctionNode>());
      body = Downcast<Function>(func_node->body)->body;
    }

    ++func_depth_;
    VisitExpr(body, nullptr);
    --func_depth_;

    return nullptr;
  }

  bool SetEqual(const VarSet& a, const VarSet& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (auto& xa : a) {
      if (!b.count(xa)) {
        return false;
      }
    }
    return true;
  }

 private:
  friend class MemoryPlanner;

  bool used_ = false;

  support::Arena arena_;
  CFG cfg_;

  CFG::Node* cfg_node_;

  // v in use_[n] means v is read in n.
  // NOTE: the use set of an If expression does not include the branches, just the condition.
  //       This lets us pretend the IR is composed of basic blocks (seq of bindings) + unstructured
  //       control flow, which is what most data-flow algorithms assume as input.
  // std::unordered_map<Expr, VarSet, ObjectPtrHash, ObjectPtrEqual> use_;
  std::unordered_map<CFG::Node*, VarSet> use_;

  // def_[n] = v means n is a node "let v = ...;"
  // TODO(@altanh): pretty sure this can be removed since we don't allow binding the same var twice
  //                (unless I'm misremembering).
  // std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> def_;
  std::unordered_map<CFG::Node*, Var> def_;

  // y in alias_[x] means y is an alias of x, created by let binding y to an alias of x.
  // NOTE: x in alias_[x] for all x.
  // std::unordered_map<Var, std::shared_ptr<VarSet>, ObjectPtrHash, ObjectPtrEqual> alias_;

  // Maps node -> {successor expr/basic block}.
  // NOTE: a pair of bindings without control flow, e.g. e = "b0; b1; body", results in a linear
  //       successor e -> b1. If expressions on the other hand, e.g.
  //       e = "let x = if (cond) { true_b } else { false_b }; body" have branching
  //       e -> {true_b, false_b} -> body.
  // std::unordered_map<Expr, std::vector<Expr>, ObjectPtrHash, ObjectPtrEqual> succ_;

  // Maps node -> {v: Var | v is live before node}
  std::unordered_map<CFG::Node*, VarSet> live_in_;
  // Maps node -> {v: Var | v is live after node}
  std::unordered_map<CFG::Node*, VarSet> live_out_;

  size_t func_depth_ = 0;

  const VarSet empty_set_;
};

// TODO(@altanh): figure out if letrec is a problem
// FIXME(@altanh): device_copy can be aliasing when src == dst

class AliasEliminator : public MixedModeMutator {
 public:
  Expr VisitExpr_(const LetNode* let_node) override {
    Expr expr = GetRef<Expr>(let_node);
    LetList ll;
    std::vector<Var> bound_vars;

    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      const Var& var = inner_let_node->var;
      const Expr& val = inner_let_node->value;
      ICHECK(!alias_.count(var));
      if (val.as<VarNode>()) {
        ICHECK(alias_.count(Downcast<Var>(val)));
        alias_[var] = alias_[Downcast<Var>(val)];
      } else {
        alias_[var] = var;
        ll.Push(var, VisitExpr(val));
      }

      bound_vars.push_back(var);

      expr = inner_let_node->body;
    }

    Expr body = ll.Get(VisitExpr(expr));

    // remove the bound vars so that alias_ only tracks things in scope
    for (const Var& v : bound_vars) {
      alias_.erase(v);
    }

    return body;
  }

  Expr VisitExpr_(const VarNode* var_node) override {
    Var var = GetRef<Var>(var_node);
    ICHECK(alias_.count(var));
    return alias_[var];
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    for (const Var& param : func_node->params) {
      alias_[param] = param;
    }

    Expr new_body = VisitExpr(func_node->body);
    Expr result = GetRef<Expr>(func_node);
    if (!new_body.same_as(func_node->body)) {
      result = Function(func_node->params, new_body, func_node->ret_type, func_node->type_params,
                        func_node->attrs, func_node->span);
    }

    for (const Var& param : func_node->params) {
      size_t erased = alias_.erase(param);
      ICHECK(erased);
    }

    return result;
  }

 private:
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> alias_;
};

class MemoryPlanner : public ExprMutator {
 public:
  MemoryPlanner() {}

  Expr PlanMemory(const Expr& e) {
    lva_.ComputeLiveness(e);
    return VisitExpr(e);
  }

  Expr VisitExpr_(const LetNode* let_node) override {
    Expr expr = GetRef<Expr>(let_node);
    LetList ll;

    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      ll.Push(inner_let_node->var, VisitExpr(inner_let_node->value));

      ICHECK(!inner_let_node->value.as<VarNode>());

      ICHECK(lva_.cfg_.let_map.count(expr));

      ControlFlowGraph::Node* n = lva_.cfg_.let_map[expr];

      auto& li = lva_.live_in_[n];
      auto& lo = lva_.live_out_[n];

      // std::cout << "let " << inner_let_node->var->name_hint() << " = ...;" << std::endl;
      // std::cout << "  live in:";
      // for (auto& v : li) {
      //   std::cout << " " << v->name_hint();
      // }
      // std::cout << std::endl << "  live out:";
      // for (auto& v : lo) {
      //   std::cout << " " << v->name_hint();
      // }
      // std::cout << std::endl << std::endl;

      // killed vars = live in - live out
      VarSet kills;
      for (auto& v : li) {
        if (!lo.count(v)) {
          kills.insert(v);
        }
      }

      for (auto& v : kills) {
        ll.Push(Call(Op::Get("memory.kill"), {v}));
      }

      expr = inner_let_node->body;
    }

    return ll.Get(VisitExpr(expr));
  }

 private:
  LivenessAnalyzer lva_;
  ControlFlowGraph::Node* curr_node_;
};

Pass VMPlanMemory() {
  auto pass_func = [](Function f, IRModule m, PassContext pc) -> Function {
    AliasEliminator el;
    MemoryPlanner mp;
    Expr nf = mp.PlanMemory(el.Mutate(f));
    return Downcast<Function>(nf);
  };
  return CreateFunctionPass(pass_func, 0, "VMPlanMemory", {});
}

TVM_REGISTER_GLOBAL("relay._transform.VMPlanMemory").set_body_typed(VMPlanMemory);

}  // namespace transform
}  // namespace relay
}  // namespace tvm

// class LivenessAnalyzer : public DeviceA
