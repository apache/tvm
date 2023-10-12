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
 * \file src/relay/backend/vm/manifest_lifetimes.cc
 * \brief Analysis and explicit manifestation of variable lifetimes. NOTE: the input IR should be in
 * ANF and post-memory-lowering (explicit manifestation of allocations).
 */

#include <tvm/relay/transform.h>

#include "../../../support/arena.h"
#include "../../op/memory/device_copy.h"
#include "../../transforms/device_aware_visitors.h"
#include "../../transforms/let_list.h"
#include "../liveness_analysis.h"

namespace tvm {
namespace relay {
namespace transform {

/*!
 * \brief Helper class to insert kills using liveness information.
 */
class KillInserter : public ExprMutator {
 public:
  KillInserter(const ControlFlowGraph* cfg, const LivenessAnalysis* lva) : cfg_(cfg), lva_(lva) {}

  // Limitations
  // -----------
  // (1) For simplicity, we only insert kills when visiting Let bindings, and always emit the kill
  // as a single subsequent binding. This is slightly inaccurate; for example, if the condition of
  // an If is dead after the test, we can immediately kill the condition in each branch:
  //   let %x = if (%dead_cond) {
  //     let %_0 = memory.kill(%dead_cond);
  //     ...
  //   } else {
  //     let %_1 = memory.kill(%dead_cond);
  //     ...
  //   }
  // as opposed to:
  //   let %x = if (%dead_cond) ...
  //   let %_0 = memory.kill(%dead_cond);
  //
  // (2) Killed variables are calculated as live in - live out, which misses variables that are
  // actually dead but not in a live-in set. Example:
  //   @f(%x: int, %y: int, %c: bool) {
  //     let %w = if (%c) {
  //       let %z = %y + %y;
  //       %z
  //     } else {
  //       %y
  //     };
  //     %w
  //   }
  // After inserting kills:
  //   @f(%x: int, %y: int, %c: bool) {
  //     /* %x is always dead, so never in any live in or live out set */
  //     let %w = if (%c) {
  //       let %z = %y + %y;
  //       let %_0 = memory.kill(%y);
  //       %z
  //     } else {
  //       %y
  //       /* %y is dead at this point */
  //     };
  //     let %_1 = memory.kill(%c);
  //     /* no kill for %y since it's not in the live-in of %w AND %w isn't a let binding */
  //     %w
  //   }
  //
  // (3) When the result expr of an If branch is a variable, and this expr is the last use of the
  // var, we cannot "kill" the var since it is being returned. The VM compiler also emits a Move
  // instruction to merge the branch results, which creates another ObjectRef to the Object held
  // by the var. The var is also not in the subsequent live-in (since it is indeed dead by this
  // point), so it won't be killed. An example can be seen in the previous code block for (2), where
  // %y is not killed if the else-branch is taken (and indeed it can be killed, as %w is mapped to
  // a new register and holds a fresh reference to the object referenced by %y).
  //
  // However, these limitations are unlikely to cause large leaks in practice.

  Expr VisitExpr_(const LetNode* let_node) override {
    Expr expr = GetRef<Expr>(let_node);
    LetList ll;

    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      ll.Push(inner_let_node->var, VisitExpr(inner_let_node->value));

      ICHECK(!inner_let_node->value.as<VarNode>()) << "aliasing should have been eliminated.";
      ICHECK(cfg_->let_map.count(expr)) << "all Let exprs should be mapped in the CFG";

      const ControlFlowGraph::NodePtr n = cfg_->let_map.at(expr);

      const VarSet& li = lva_->live_in.at(n);
      const VarSet& lo = lva_->live_out.at(n);

      // Killed vars = live in - live out.
      VarSet kills;
      for (const Var& v : li) {
        if (!lo.count(v)) {
          kills.insert(v);
        }
      }

      for (const Var& v : kills) {
        ll.Push(Call(Op::Get("memory.kill"), {v}));
      }

      expr = inner_let_node->body;
    }

    return ll.Get(VisitExpr(expr));
  }

 private:
  const ControlFlowGraph* cfg_;
  const LivenessAnalysis* lva_;
};

/*!
 * \brief Helper class to eliminate variable aliasing. This pass anticipates the VM compiler's
 * register aliasing behavior so as to avoid killing vars that point to the same register. An
 * alternative approach would be to track aliasing within the VM compiler itself, so that kill
 * instructions are only emitted when all aliases are killed.
 */
class AliasEliminator : public MixedModeMutator {
 public:
  using MixedModeMutator::VisitExpr_;

  Expr VisitExpr_(const LetNode* let_node) override {
    Expr expr = GetRef<Expr>(let_node);
    LetList ll;
    std::vector<Var> aliased_vars;

    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      const Var& var = inner_let_node->var;
      const Expr& val = inner_let_node->value;
      bool aliased = false;
      ICHECK(!alias_.count(var));

      if (const VarNode* alias_of_n = AsIgnoringOnDevice<VarNode>(val)) {
        alias_[var] = Downcast<Var>(VisitExpr_(alias_of_n));
        aliased = true;
      } else if (AsIgnoringOnDevice<CallNode>(val)) {
        // Copying to the same device is aliasing.
        // WARNING: this must be kept in sync with the VM compiler logic in
        // src/relay/backend/vm/compiler.cc, line 541, in DeviceAwareVisitExpr_(const CallNode*).
        Expr unwrapped = IgnoreOnDevice(val);
        DeviceCopyProps copy_props = GetDeviceCopyProps(unwrapped);
        if (copy_props.body.defined()) {
          if (copy_props.src_virtual_device->device_type() ==
                  copy_props.dst_virtual_device->device_type() &&
              copy_props.src_virtual_device->virtual_device_id ==
                  copy_props.dst_virtual_device->virtual_device_id &&
              copy_props.src_virtual_device->memory_scope ==
                  copy_props.dst_virtual_device->memory_scope) {
            Expr to_copy = Downcast<Call>(unwrapped)->args[0];
            if (const VarNode* alias_of_n = to_copy.as<VarNode>()) {
              alias_[var] = Downcast<Var>(VisitExpr_(alias_of_n));
              aliased = true;
            }
          }
        }
      }

      if (!aliased) {
        ll.Push(var, VisitExpr(val));
      } else {
        aliased_vars.push_back(var);
      }

      expr = inner_let_node->body;
    }

    Expr body = ll.Get(VisitExpr(expr));

    // remove the aliased vars so that alias_ only tracks things in scope
    for (const Var& v : aliased_vars) {
      alias_.erase(v);
    }

    return body;
  }

  Expr VisitExpr_(const VarNode* var_node) override {
    Var var = GetRef<Var>(var_node);
    if (alias_.count(var)) {
      return alias_[var];
    }
    return std::move(var);
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    Expr new_body = VisitExpr(func_node->body);
    return WithFields(GetRef<Function>(func_node), /*opt_params=*/NullOpt, /*opt_body=*/new_body);
  }

  // The only register-level aliasing that occurs in Match expressions is when
  // the deconstructed expression is a Var, and the matched pattern is also a Var.
  Expr VisitExpr_(const MatchNode* match_node) override {
    if (const VarNode* data_var_node = AsIgnoringOnDevice<VarNode>(match_node->data)) {
      Var data_var = Downcast<Var>(VisitExpr_(data_var_node));
      std::vector<Clause> new_clauses;
      for (const Clause& clause : match_node->clauses) {
        const PatternVarNode* pv_node = nullptr;
        if ((pv_node = clause->lhs.as<PatternVarNode>())) {
          alias_[pv_node->var] = data_var;
        }
        new_clauses.push_back(Clause(clause->lhs, VisitExpr(clause->rhs)));
        if (pv_node) {
          alias_.erase(pv_node->var);
        }
      }
      return Match(data_var, new_clauses, match_node->complete, match_node->span);
    } else {
      return ExprMutator::VisitExpr_(match_node);
    }
  }

 private:
  /*!
   * \brief Mapping of var -> var it's an alias of. Note that transitive aliases
   * (e.g. x = 0; y = x; z = y) are mapped to the non-aliased variable (in this example "x").
   */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> alias_;
};

Pass ManifestLifetimes() {
  auto pass_func = [](Function f, IRModule m, PassContext pc) -> Function {
    f = Downcast<Function>(AliasEliminator().Mutate(f));
    Arena arena;
    ControlFlowGraph cfg = ControlFlowGraph::Create(&arena, f);
    UseDefAnalysis use_def = UseDefAnalysis::Analyze(cfg);
    LivenessAnalysis lva = LivenessAnalysis::Analyze(cfg, use_def);
    KillInserter ki(&cfg, &lva);
    Function nf = Downcast<Function>(ki.Mutate(f));
    return nf;
  };
  return CreateFunctionPass(pass_func, 0, "ManifestLifetimes", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ManifestLifetimes").set_body_typed(ManifestLifetimes);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
