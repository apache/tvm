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
 * \file src/relay/backend/manifest_lifetimes.cc
 * \brief Analysis and explicit manifestation of variable lifetimes. NOTE: the input IR should be in
 * ANF and post-memory-lowering (explicit manifestation of allocations).
 */

#include "manifest_lifetimes.h"

#include <list>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace transform {

using support::Arena;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

ControlFlowGraph ControlFlowGraph::Create(Arena* arena, const Expr& body) {
  return Creator().Create(arena, body);
}

ControlFlowGraph ControlFlowGraph::Creator::Create(Arena* arena, const Expr& body) {
  arena_ = arena;
  cfg_.entry = BasicBlock::Make(arena);
  VisitExpr(body, cfg_.entry);
  return std::move(cfg_);
}

void ControlFlowGraph::Creator::Succ(BasicBlockPtr from, BasicBlockPtr to) {
  from->succ.push_back(to);
  to->pred.push_back(from);
}

void ControlFlowGraph::Creator::VisitExpr_(const FunctionNode* f, BasicBlockPtr parent) {
  ICHECK(!in_func_) << "nested functions not supported by CFG analysis";
  in_func_ = true;

  // Unwrap the nested function and proceed normally.
  if (f->HasNonzeroAttr(attr::kClosure)) {
    ICHECK(f->body.as<FunctionNode>());
    return VisitExpr(Downcast<Function>(f->body)->body, parent);
  }

  return VisitExpr(f->body, parent);
}

void ControlFlowGraph::Creator::VisitExpr_(const LetNode* let_node, BasicBlockPtr parent) {
  Expr expr = GetRef<Expr>(let_node);

  while (const LetNode* inner_let_node = expr.as<LetNode>()) {
    NodePtr curr_node = Node::Make(arena_, parent, expr);

    ICHECK(!cfg_.let_map.count(expr));
    cfg_.let_map[expr] = curr_node;
    cfg_.reverse_post_order.push_back(curr_node);

    // The basic block ends upon reaching control flow, with successor blocks corresponding to the
    // control flow branch exprs (true/false in If, and one for each clause in Match).
    if (const IfNode* ite = AsIgnoringOnDevice<IfNode>(inner_let_node->value)) {
      // Create the basic blocks for each branch and mark them as successors to the current block.
      BasicBlockPtr t_block = BasicBlock::Make(arena_);
      BasicBlockPtr f_block = BasicBlock::Make(arena_);
      Succ(parent, t_block);
      Succ(parent, f_block);

      VisitExpr(ite->true_branch, t_block);
      VisitExpr(ite->false_branch, f_block);

      // All subsequent bindings (and/or the body expr) will be in a new basic block.
      BasicBlockPtr next = BasicBlock::Make(arena_);
      Succ(t_block, next);
      Succ(f_block, next);
      parent = next;
    } else if (const MatchNode* match = AsIgnoringOnDevice<MatchNode>(inner_let_node->value)) {
      // Same as above but one for each pattern.
      std::vector<BasicBlockPtr> clause_blocks;
      BasicBlockPtr next = BasicBlock::Make(arena_);
      for (const Clause& clause : match->clauses) {
        BasicBlockPtr clause_block = BasicBlock::Make(arena_);
        Succ(parent, clause_block);
        Succ(clause_block, next);
        VisitExpr(clause->rhs, clause_block);
      }
      parent = next;
    }

    expr = inner_let_node->body;
  }

  VisitExpr(expr, parent);
}

void ControlFlowGraph::Creator::VisitExpr_(const IfNode* if_node, BasicBlockPtr parent) {
  // TODO(@altanh): is there a way of making this work?
  LOG(FATAL) << "If expressions should be bound to variables.";
}

void ControlFlowGraph::Creator::VisitExpr_(const MatchNode* match_node, BasicBlockPtr parent) {
  // TODO(@altanh): same as If
  LOG(FATAL) << "Match expressions should be bound to variables.";
}

VarSet VarUseCollector::VisitExpr_(const VarNode* var_node) { return {GetRef<Var>(var_node)}; }

VarSet VarUseCollector::VisitExpr_(const CallNode* call_node) {
  VarSet use = VisitExpr(call_node->op);
  for (const Expr& arg : call_node->args) {
    VarSet arg_use = VisitExpr(arg);
    use.insert(arg_use.begin(), arg_use.end());
  }
  return use;
}

VarSet VarUseCollector::VisitExpr_(const TupleNode* tuple_node) {
  VarSet use;
  for (const Expr& field : tuple_node->fields) {
    VarSet field_use = VisitExpr(field);
    use.insert(field_use.begin(), field_use.end());
  }
  return use;
}

VarSet VarUseCollector::VisitExpr_(const TupleGetItemNode* get_node) {
  return VisitExpr(get_node->tuple);
}

VarSet VarUseCollector::VisitExpr_(const IfNode* if_node) { return VisitExpr(if_node->cond); }

VarSet VarUseCollector::VisitExpr_(const MatchNode* match_node) {
  return VisitExpr(match_node->data);
}

UseDefAnalysis UseDefAnalysis::Analyze(const CFG& cfg) {
  UseDefAnalysis a;

  // One pass is sufficient.
  for (auto it = cfg.reverse_post_order.begin(); it != cfg.reverse_post_order.end(); ++it) {
    const CFG::NodePtr& node = *it;
    if (const LetNode* let_node = AsIgnoringOnDevice<LetNode>(node->expr)) {
      a.use[node] = a.use_collector.VisitExpr(let_node->value);
      a.def[node] = let_node->var;
    } else {
      a.use[node] = a.use_collector.VisitExpr(node->expr);
      a.def[node] = Var();
    }
  }

  return a;
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

LivenessAnalysis LivenessAnalysis::Analyze(const ControlFlowGraph& cfg,
                                           const UseDefAnalysis& use_def) {
  LivenessAnalysis a;
  std::list<CFG::NodePtr> worklist;

  // Initialize worklist to post-order traversal for quick convergence.
  worklist.insert(worklist.end(), cfg.reverse_post_order.rbegin(), cfg.reverse_post_order.rend());

  // See https://lambda.uta.edu/cse5317/notes/node40.html for an overview of the algorithm.
  auto visitor = [&](const CFG::NodePtr n) {
    VarSet old_in_n = a.live_in[n];
    VarSet old_out_n = a.live_out[n];

    a.live_in[n] = use_def.use.at(n);
    for (const Var& v : a.live_out[n]) {
      if (!v.same_as(use_def.def.at(n))) {
        a.live_in[n].insert(v);
      }
    }

    a.live_out[n] = VarSet();
    for (const CFG::NodePtr& s : n->GetSucc()) {
      a.live_out[n].insert(a.live_in[s].begin(), a.live_in[s].end());
    }

    if (SetEqual(old_in_n, a.live_in[n]) && SetEqual(old_out_n, a.live_out[n])) {
      // No need to update the worklist.
    } else {
      // Add predecessor nodes back to worklist (no need to add successors, since each node's
      // in/out sets are not dependent on its predecessors).
      for (const CFG::NodePtr& p : n->GetPred()) {
        worklist.push_back(p);
      }
    }
  };

  while (!worklist.empty()) {
    const CFG::NodePtr n = worklist.front();
    worklist.pop_front();
    visitor(n);
  }

  return a;
}

Expr KillInserter::VisitExpr_(const LetNode* let_node) {
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

Expr AliasEliminator::VisitExpr_(const LetNode* let_node) {
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
                copy_props.dst_virtual_device->virtual_device_id) {
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

Expr AliasEliminator::VisitExpr_(const VarNode* var_node) {
  Var var = GetRef<Var>(var_node);
  if (alias_.count(var)) {
    return alias_[var];
  }
  return var;
}

Expr AliasEliminator::VisitExpr_(const FunctionNode* func_node) {
  Expr new_body = VisitExpr(func_node->body);
  return WithFields(GetRef<Function>(func_node), /*opt_params=*/NullOpt, /*opt_body=*/new_body);
}

Expr AliasEliminator::VisitExpr_(const MatchNode* match_node) {
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
