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
#include "../../transforms/let_list.h"

#include "../../transforms/device_aware_visitors.h"

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

// x = 1
// y = x
// z = y
// return z

/*

let x = op(in);
let y = x;
let z = op(y);
let w = if (z) {
  let xx = op(x);
  op(xx)
} else {
  let zz = op(z);
  op(zz);
};
op(w)

---

block0:
  x <- op(in);
  y <- x;
  z <- = op(y);
  jz z block2

block1:
  xx <- op(x);
  w <- op(xx)
  jmp block3

block2:
  zz <- op(z)
  w <- op(zz)

block3:
  ret op(w)
*/


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
    } else {
      alias_[var] = var;
    }

    VisitExpr(var);
    VisitExpr(value);
  }

  void PostVisitLetBlock_(const LetNode* let_node) {
    Expr expr = GetRef<Expr>(let_node);
    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      if (!inner_let_node->value.as<VarNode>()) {
        ICHECK(use_.count(inner_let_node->value));
        def_[expr] = inner_let_node->var;
        use_[expr] = use_[inner_let_node->value];
      }
      expr = inner_let_node->body;
    }
  }

  void VisitExpr_(const LetNode* let_node) {
    std::vector<const LetNode*> bindings;
    Expr expr = GetRef<Expr>(let_node);
    while (const auto* inner_let_node = expr.as<LetNode>()) {
      PreVisitLetBinding_(inner_let_node->var, inner_let_node->value);
      bindings.emplace_back(inner_let_node);
      expr = inner_let_node->body;
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

  void VisitExpr_(const GlobalVarNode* global_var_node) override {
    use_[GetRef<Expr>(global_var_node)] = empty_set_;
  }

  void VisitExpr_(const VarNode* var_node) override {
    Var var = GetRef<Var>(var_node);
    ICHECK(alias_.count(var));
    use_[var] = {alias_[var]};
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

    // TODO: figure out the closure nesting thing
    ICHECK(!function_node->HasNonzeroAttr(attr::kClosure)) << "closures not supported yet";
    ICHECK(func_depth_ == 0) << "nested functions not supported";

    use_.clear();
    def_.clear();
    alias_.clear();

    for (const Var& param : function_node->params) {
      alias_[param] = param;
    }

    ++func_depth_;

    VisitExpr(function_node->body);

    --func_depth_;
  }

  void VisitExpr_(const IfNode* if_node) override {
    LOG(FATAL) << "if not supported yet";
  }

  void DebugDump() {
    // for (auto pr : alias_) {
    //   std::cout << pr.first << " -> " << pr.second << std::endl;
    // }
    // for (auto pr : use_) {
    //   std::cout << pr.first.get() << " uses";
    //   for (auto var : pr.second) {
    //     std::cout << " " << var;
    //   }
    //   std::cout << std::endl;
    // }
    // for (auto pr : def_) {
    //   std::cout << pr.first.get() << " defs " << pr.second << std::endl;
    // }
    for (auto pr : live_in_) {
      if (auto* l = pr.first.as<LetNode>()) {
        auto lo = live_out_[GetRef<Expr>(l)];
        std::cout << l->var->name_hint() << " kill";
        for (auto v : pr.second) {
          if (!lo.count(v)) {
            std::cout << " " << v->name_hint();
          }
        }
        std::cout << std::endl;
        // std::cout << l->var->name_hint() << " = " << (l->value) << " live in: ";
        // for (auto var : pr.second) {
        //   std::cout << " " << var->name_hint();
        // }
        // std::cout << std::endl;
      }
    }

    // for (auto pr : live_out_) {
    //   if (auto* l = pr.first.as<LetNode>()) {
    //     std::cout << l->var->name_hint() << " = " << (l->value) << " live out: ";
    //     for (auto var : pr.second) {
    //       std::cout << " " << var->name_hint();
    //     }
    //     std::cout << std::endl;
    //   }
    // }
  }

  void ComputeLiveness(const Expr& expr) {
    VisitExpr(expr);

    bool did_work = true;

    auto visitor = [&](const Expr& n) {
      TVarSet old_in_n = this->live_in_[n];
      TVarSet old_out_n = this->live_out_[n];

      this->live_in_[n] = TVarSet(this->use_[n].begin(), this->use_[n].end());
      for (auto v : this->live_out_[n]) {
        if (!v.same_as(this->def_[n])) {
          this->live_in_[n].insert(v);
        }
      }
      if (auto* l = n.as<LetNode>()) {
        this->live_out_[n] = this->live_in_[l->body];
      }

      if (old_in_n.size() != this->live_in_[n].size() || old_out_n.size() != this->live_out_[n].size()) {
        did_work = true;
      } else {
        for (auto o : old_in_n) {
          did_work |= !this->live_in_[n].count(o);
        }
        for (auto o : old_out_n) {
          did_work |= !this->live_out_[n].count(o);
        }
      }
    };

    while (did_work) {
      did_work = false;
      PostOrderVisit(expr, visitor);
    }
  }

 private:
  friend class MemoryPlanner;

  // v in use_[n] means v is read in n
  std::unordered_map<Expr, TVarSet, ObjectPtrHash, ObjectPtrEqual> use_;
  // def_[n] = v means n is an expr "let v = ...; ..."
  std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> def_;
  // e.g. alias_[x] = y means y is an alias of x, created by "let y = x; ..."
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> alias_;

  std::unordered_map<Expr, TVarSet, ObjectPtrHash, ObjectPtrEqual> live_in_;
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
    Expr value = VisitExpr(let_node->value);
    Let let = GetRef<Let>(let_node);
    auto li = lva_.live_in_[let];
    auto lo = lva_.live_out_[let];
    LivenessAnalyzer::TVarSet kills;
    for (auto v : li) {
      if (!lo.count(v)) {
        kills.insert(v);
      }
    }
    if (!kills.empty()) {
      LetList ll;
      ll.Push(let->var, value);
      for (auto v : kills) {
        ll.Push(Call(Op::Get("memory.kill"), {v}));
      }
      return ll.Get(VisitExpr(let_node->body));
    }
    return Let(let->var, value, VisitExpr(let_node->body));
  }

 private:
  LivenessAnalyzer lva_;
};

Pass VMPlanMemory() {
  auto pass_func = [](Function f, IRModule m, PassContext pc) -> Function {
    // LivenessAnalyzer lva;
    // lva.ComputeLiveness(f);
    // lva.DebugDump();
    MemoryPlanner mp;
    Expr nf = mp.PlanMemory(f);
    std::cout << PrettyPrint(nf);
    return Downcast<Function>(nf);
  };
  return CreateFunctionPass(pass_func, 0, "VMPlanMemory", {});
}

TVM_REGISTER_GLOBAL("relay._transform.VMPlanMemory").set_body_typed(VMPlanMemory);

}  // namespace transform
}  // namespace relay
}  // namespace tvm

// class LivenessAnalyzer : public DeviceA
