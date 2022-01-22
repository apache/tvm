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

#include "../../transforms/device_aware_visitors.h"
#include "../../transforms/let_list.h"

// pipeline: ManifestAlloc -> CoalesceStorage -> PlanMemory

// PlanMemory: analyze liveness info and insert kill operations

// A LIVE storage:
//   - has live tensors or
//   - has pending tensor allocations
// So a DEAD storage has no live tensors and provably no pending tensor allocations

// A LIVE tensor:
//   - has pending dependent tensor computations
//   - a tuple is like a weird kind of aliasing
//   - var aliasing
//   - what if it escapes? e.g. func ret value -> always alive

namespace tvm {
namespace relay {
namespace transform {

class LivenessAnalyzer : public ExprVisitor {
 private:
  using TVarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
  using TSuper = ExprVisitor;

 public:
  LivenessAnalyzer() {}

  void PreVisitLetBinding_(const Var& var, const Expr& value) {
    // if value is a variable, alias[var] = alias[value]
    // else, alias[var] = var
    ICHECK(!alias_.count(var));
    if (value.as<VarNode>()) {
      Var rhs = Downcast<Var>(value);
      ICHECK(alias_.count(rhs)) << "aliasing info of rhs should be tracked";
      alias_[var] = alias_[rhs];
      alias_[rhs]->insert(var);
    } else {
      alias_[var] = std::make_shared<TVarSet>();
      alias_[var]->insert(var);
    }

    VisitExpr(var);
    VisitExpr(value);
  }

  void PostVisitLetBlock_(const LetNode* let_node) {
    Expr expr = GetRef<Expr>(let_node);
    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      ICHECK(use_.count(inner_let_node->value));
      def_[expr] = inner_let_node->var;
      use_[expr] = TVarSet(use_[inner_let_node->value].begin(), use_[inner_let_node->value].end());
      expr = inner_let_node->body;
    }
  }

  void VisitExpr_(const LetNode* let_node) {
    Expr expr = GetRef<Expr>(let_node);
    while (const auto* inner_let_node = expr.as<LetNode>()) {
      PreVisitLetBinding_(inner_let_node->var, inner_let_node->value);

      Expr let_body = inner_let_node->body;
      if (const auto* ite = AsIgnoringOnDevice<IfNode>(inner_let_node->value)) {
        succ_[expr] = {ite->true_branch, ite->false_branch};
        succ_[ite->true_branch] = {let_body};
        succ_[ite->false_branch] = {let_body};
      } else {
        succ_[expr] = {let_body};
      }

      expr = let_body;
    }

    VisitExpr(expr);

    PostVisitLetBlock_(let_node);
  }

  void VisitExpr_(const TupleNode* tuple_node) override {
    TSuper::VisitExpr_(tuple_node);
    TVarSet use;
    for (const Expr& field : tuple_node->fields) {
      ICHECK(use_.count(field));
      TVarSet field_use = use_[field];
      use.insert(field_use.begin(), field_use.end());
    }
    use_[GetRef<Expr>(tuple_node)] = use;
  }

  void VisitExpr_(const TupleGetItemNode* get_node) override {
    TSuper::VisitExpr_(get_node);
    use_[GetRef<Expr>(get_node)] = use_[get_node->tuple];
  }

  void VisitExpr_(const GlobalVarNode* global_var_node) override {
    use_[GetRef<Expr>(global_var_node)] = empty_set_;
  }

  void VisitExpr_(const VarNode* var_node) override {
    Var var = GetRef<Var>(var_node);
    ICHECK(alias_.count(var));
    use_[var] = {var};
  }

  void VisitExpr_(const ConstantNode* const_node) override {
    use_[GetRef<Expr>(const_node)] = empty_set_;
  }

  void VisitExpr_(const CallNode* call_node) override {
    TVarSet use;
    for (const Expr& arg : call_node->args) {
      VisitExpr(arg);
      ICHECK(use_.count(arg)) << arg;
      TVarSet arg_use = use_[arg];
      use.insert(arg_use.begin(), arg_use.end());
    }
    use_[GetRef<Expr>(call_node)] = use;
  }

  void VisitExpr_(const FunctionNode* function_node) override {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      TSuper::VisitExpr_(function_node);
      return;
    }

    // TODO(@altanh): figure out the closure nesting thing
    ICHECK(!function_node->HasNonzeroAttr(attr::kClosure)) << "closures not supported yet";
    ICHECK(func_depth_ == 0) << "nested functions should have been transformed away";

    use_.clear();
    def_.clear();
    alias_.clear();

    for (const Var& param : function_node->params) {
      alias_[param] = std::make_shared<TVarSet>();
      alias_[param]->insert(param);
    }

    ++func_depth_;

    VisitExpr(function_node->body);

    --func_depth_;
  }

  void VisitExpr_(const IfNode* if_node) override {
    TSuper::VisitExpr_(if_node);
    use_[GetRef<Expr>(if_node)] = use_[if_node->cond];
  }

  bool SetEqual(const TVarSet& a, const TVarSet& b) {
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

  void ComputeLiveness(const Expr& expr) {
    VisitExpr(expr);

    bool did_work = true;

    // see https://lambda.uta.edu/cse5317/notes/node40.html for an overview of the algorithm
    auto visitor = [&](const Expr& n) {
      TVarSet old_in_n = this->live_in_[n];
      TVarSet old_out_n = this->live_out_[n];

      // we only compute live in/out for bindings
      if (!n.as<LetNode>()) {
        return;
      }

      this->live_in_[n] = TVarSet(this->use_[n].begin(), this->use_[n].end());
      for (const Var& v : this->live_out_[n]) {
        if (!v.same_as(this->def_[n])) {
          this->live_in_[n].insert(v);
        }
      }

      ICHECK(succ_.count(n));
      this->live_out_[n] = TVarSet();
      for (const Expr& s : succ_[n]) {
        this->live_out_[n].insert(this->live_in_[s].begin(), this->live_in_[s].end());
      }

      if (!SetEqual(old_in_n, this->live_in_[n])) {
        did_work = true;
      } else if (!SetEqual(old_out_n, this->live_out_[n])) {
        did_work = true;
      }
    };

    while (did_work) {
      did_work = false;
      PostOrderVisit(expr, visitor);
    }
  }

 private:
  friend class MemoryPlanner;

  // v in use_[n] means v is read in n.
  // NOTE: the use set of an If expression does not include the branches, just the condition.
  //       This lets us pretend the IR is composed of basic blocks (seq of bindings) + unstructured
  //       control flow, which is what most data-flow algorithms assume as input.
  std::unordered_map<Expr, TVarSet, ObjectPtrHash, ObjectPtrEqual> use_;

  // def_[n] = v means n is an expr "let v = ...; ..."
  // TODO(@altanh): pretty sure this can be removed since we don't allow binding the same var twice
  //                (unless I'm misremembering).
  std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> def_;

  // y in alias_[x] means y is an alias of x, created by let binding y to an alias of x.
  // NOTE: x in alias_[x] for all x.
  std::unordered_map<Var, std::shared_ptr<TVarSet>, ObjectPtrHash, ObjectPtrEqual> alias_;

  // Maps expr -> {successor expr/basic block}.
  // NOTE: a pair of bindings without control flow, e.g. e = "b0; b1; body", results in a linear
  //       successor e -> b1. If expressions on the other hand, e.g.
  //       e = "let x = if (cond) { true_b } else { false_b }; body" have branching
  //       e -> {true_b, false_b} -> body.
  std::unordered_map<Expr, std::vector<Expr>, ObjectPtrHash, ObjectPtrEqual> succ_;

  // Maps expr -> {v: Var | v is live before expr}
  std::unordered_map<Expr, TVarSet, ObjectPtrHash, ObjectPtrEqual> live_in_;

  // Maps expr -> {v: Var | v is live after expr}
  std::unordered_map<Expr, TVarSet, ObjectPtrHash, ObjectPtrEqual> live_out_;

  size_t func_depth_ = 0;

  const TVarSet empty_set_;
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

      auto& li = lva_.live_in_[expr];
      auto& lo = lva_.live_out_[expr];

      // killed vars = live in - live out
      LivenessAnalyzer::TVarSet kills;
      for (auto& v : li) {
        if (!lo.count(v)) {
          kills.insert(v);
        }
      }

      // remove dead aliases from their alias sets
      for (auto& v : kills) {
        lva_.alias_[v]->erase(v);
        if (lva_.alias_[v]->empty()) {
          // all aliases are dead, so we can actually kill the register
          ll.Push(Call(Op::Get("memory.kill"), {v}));
        }
      }

      expr = inner_let_node->body;
    }

    return ll.Get(VisitExpr(expr));
  }

 private:
  LivenessAnalyzer lva_;
};

Pass VMPlanMemory() {
  auto pass_func = [](Function f, IRModule m, PassContext pc) -> Function {
    MemoryPlanner mp;
    Expr nf = mp.PlanMemory(f);
    return Downcast<Function>(nf);
  };
  return CreateFunctionPass(pass_func, 0, "VMPlanMemory", {});
}

TVM_REGISTER_GLOBAL("relay._transform.VMPlanMemory").set_body_typed(VMPlanMemory);

}  // namespace transform
}  // namespace relay
}  // namespace tvm

// class LivenessAnalyzer : public DeviceA
