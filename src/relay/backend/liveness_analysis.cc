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
 * \file src/relay/backend/liveness_analysis.cc
 * \brief  Analysis that collects the live variables before and after each node.
 * NOTE: the input IR should be in ANF.
 */

#include "./liveness_analysis.h"

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

}  // namespace transform
}  // namespace relay
}  // namespace tvm
