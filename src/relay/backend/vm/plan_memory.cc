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
#include "../../op/memory/device_copy.h"
#include "../../transforms/device_aware_visitors.h"
#include "../../transforms/let_list.h"

namespace tvm {
namespace relay {
namespace transform {

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

class ControlFlowGraph {
 public:
  struct Node;
  struct BasicBlock;

  using NodePtr = std::shared_ptr<Node>;
  using BasicBlockPtr = std::shared_ptr<BasicBlock>;

  struct BasicBlock {
    // The nodes of the basic block.
    std::vector<NodePtr> nodes;
    // The predecessor basic blocks.
    std::vector<BasicBlockPtr> pred;
    // The successor basic blocks.
    std::vector<BasicBlockPtr> succ;

    static BasicBlockPtr Make() { return std::make_shared<BasicBlock>(); }
  };

  struct Node {
    // The basic block this node belongs to.
    BasicBlockPtr parent;
    // The index into the parent basic block where this node is.
    size_t index;
    // The expr corresponding to this node.
    Expr expr;

    // Returns whether or not this node is the last one in the parent basic block.
    bool IsLast() const { return index == parent->nodes.size() - 1; }

    // Returns the successor nodes of this node.
    std::vector<NodePtr> GetSucc() const {
      std::vector<NodePtr> succ;
      if (IsLast()) {
        for (const BasicBlockPtr& succ_block : parent->succ) {
          succ.push_back(succ_block->nodes[0]);
        }
      } else {
        succ.push_back(parent->nodes[index + 1]);
      }
      return succ;
    }

    // Creates a node with the given expr and pushes it to the end of the parent basic block.
    static NodePtr Make(BasicBlockPtr parent, Expr expr) {
      NodePtr n = std::make_shared<Node>();
      n->parent = parent;
      n->expr = expr;
      n->index = parent->nodes.size();
      parent->nodes.push_back(n);
      return n;
    }
  };

  BasicBlockPtr entry;

  // Let expressions are never shared in ANF (unlike vars), so this is an injection.
  std::unordered_map<Expr, NodePtr, ObjectPtrHash, ObjectPtrEqual> let_map;

  std::vector<NodePtr> reverse_post_order;

  static ControlFlowGraph Create(const Expr& body);

 private:
  class Creator;
};

using NodeList = std::vector<ControlFlowGraph::Node*>;

class ControlFlowGraph::Creator : private ExprFunctor<void(const Expr&, BasicBlockPtr)> {
 public:
  Creator() {}

  ControlFlowGraph Create(const Expr& body) {
    cfg_.entry = BasicBlock::Make();
    VisitExpr(body, cfg_.entry);
    return std::move(cfg_);
  }

 private:
  ControlFlowGraph cfg_;
  bool in_func_ = false;

  void Succ(BasicBlockPtr from, BasicBlockPtr to) {
    from->succ.push_back(to);
    to->pred.push_back(from);
  }

#define DEFAULT_CFG(OP)                                       \
  void VisitExpr_(const OP* op, BasicBlockPtr parent) final { \
    NodePtr n = Node::Make(parent, GetRef<Expr>(op));         \
    cfg_.reverse_post_order.push_back(n);                     \
  }

  void VisitExpr_(const FunctionNode* f, BasicBlockPtr parent) final {
    ICHECK(!in_func_) << "nested functions not supported by CFG analysis";
    in_func_ = true;

    if (f->HasNonzeroAttr(attr::kClosure)) {
      ICHECK(f->body.as<FunctionNode>());
      return VisitExpr(Downcast<Function>(f->body)->body, parent);
    }

    return VisitExpr(f->body, parent);
  }

  void VisitExpr_(const LetNode* let_node, BasicBlockPtr parent) final {
    Expr expr = GetRef<Expr>(let_node);

    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      NodePtr curr_node = Node::Make(parent, expr);

      ICHECK(!cfg_.let_map.count(expr));
      cfg_.let_map[expr] = curr_node;
      cfg_.reverse_post_order.push_back(curr_node);

      if (const IfNode* ite = AsIgnoringOnDevice<IfNode>(inner_let_node->value)) {
        // Create the basic blocks for each branch and mark them as successors to the current block.
        BasicBlockPtr t_block = BasicBlock::Make();
        BasicBlockPtr f_block = BasicBlock::Make();
        Succ(parent, t_block);
        Succ(parent, f_block);

        VisitExpr(ite->true_branch, t_block);
        VisitExpr(ite->false_branch, f_block);

        // All subsequent bindings (and/or the body expr) will be in a new basic block.
        BasicBlockPtr next = BasicBlock::Make();
        Succ(t_block, next);
        Succ(f_block, next);
        parent = next;
      } else if (const MatchNode* match = AsIgnoringOnDevice<MatchNode>(inner_let_node->value)) {
        // Same as above but one for each pattern.
        std::vector<BasicBlockPtr> clause_blocks;
        BasicBlockPtr next = BasicBlock::Make();
        for (const Clause& clause : match->clauses) {
          BasicBlockPtr clause_block = BasicBlock::Make();
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

  void VisitExpr_(const IfNode* if_node, BasicBlockPtr parent) {
    // TODO(@altanh): is there a way of making this work?
    LOG(FATAL) << "If expressions should be bound to variables.";
  }

  void VisitExpr_(const MatchNode* match_node, BasicBlockPtr parent) {
    // TODO(@altanh): same as If
    LOG(FATAL) << "Match expressions should be bound to variables.";
  }

  DEFAULT_CFG(VarNode);
  DEFAULT_CFG(GlobalVarNode);
  DEFAULT_CFG(ConstantNode);
  DEFAULT_CFG(CallNode);
  DEFAULT_CFG(OpNode);
  DEFAULT_CFG(TupleNode);
  DEFAULT_CFG(TupleGetItemNode);
};

ControlFlowGraph ControlFlowGraph::Create(const Expr& body) { return Creator().Create(body); }

// NOTE: for If exprs, only the condition is included (not the branches). Similarly, for Match
//       exprs only the value being deconstructed is included.
class VarUseCollector : public ExprFunctor<VarSet(const Expr& e)> {
 public:
  VarSet VisitExpr_(const VarNode* var_node) { return {GetRef<Var>(var_node)}; }

  VarSet VisitExpr_(const CallNode* call_node) {
    VarSet use = VisitExpr(call_node->op);
    for (const Expr& arg : call_node->args) {
      VarSet arg_use = VisitExpr(arg);
      use.insert(arg_use.begin(), arg_use.end());
    }
    return use;
  }

  VarSet VisitExpr_(const TupleNode* tuple_node) {
    VarSet use;
    for (const Expr& field : tuple_node->fields) {
      VarSet field_use = VisitExpr(field);
      use.insert(field_use.begin(), field_use.end());
    }
    return use;
  }

  VarSet VisitExpr_(const TupleGetItemNode* get_node) { return VisitExpr(get_node->tuple); }

  VarSet VisitExpr_(const IfNode* if_node) { return VisitExpr(if_node->cond); }

  VarSet VisitExpr_(const MatchNode* match_node) { return VisitExpr(match_node->data); }

  VarSet VisitExpr_(const ConstructorNode* cons_node) { return {}; }

  VarSet VisitExpr_(const GlobalVarNode* gvar_node) { return {}; }

  VarSet VisitExpr_(const ConstantNode* const_node) { return {}; }

  VarSet VisitExpr_(const OpNode* op_node) { return {}; }
};

struct UseDefAnalysis {
  using CFG = ControlFlowGraph;

  std::unordered_map<CFG::NodePtr, VarSet> use;
  std::unordered_map<CFG::NodePtr, Var> def;

  VarUseCollector use_collector;

  static UseDefAnalysis Analyze(const CFG& cfg) {
    UseDefAnalysis a;

    std::vector<CFG::BasicBlockPtr> worklist = {cfg.entry};
    while (!worklist.empty()) {
      CFG::BasicBlockPtr block = worklist.back();
      worklist.pop_back();

      for (const CFG::NodePtr& node : block->nodes) {
        if (const LetNode* let_node = AsIgnoringOnDevice<LetNode>(node->expr)) {
          a.use[node] = a.use_collector.VisitExpr(let_node->value);
          a.def[node] = let_node->var;
        } else {
          a.use[node] = a.use_collector.VisitExpr(node->expr);
          a.def[node] = Var();
        }
      }

      for (const CFG::BasicBlockPtr& s : block->succ) {
        worklist.push_back(s);
      }
    }

    return a;
  }
};

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

struct LivenessAnalysis {
  using CFG = ControlFlowGraph;

  std::unordered_map<CFG::NodePtr, VarSet> live_in;
  std::unordered_map<CFG::NodePtr, VarSet> live_out;

  static LivenessAnalysis Analyze(const ControlFlowGraph& cfg, const UseDefAnalysis& use_def) {
    LivenessAnalysis a;
    bool did_work = true;

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

      if (!SetEqual(old_in_n, a.live_in[n])) {
        did_work = true;
      } else if (!SetEqual(old_out_n, a.live_out[n])) {
        did_work = true;
      }
    };

    while (did_work) {
      did_work = false;
      for (auto it = cfg.reverse_post_order.rbegin(); it != cfg.reverse_post_order.rend(); ++it) {
        visitor(*it);
      }
    }

    return a;
  }
};

class KillInserter : public ExprMutator {
 public:
  KillInserter(const ControlFlowGraph* cfg, const LivenessAnalysis* lva) : cfg_(cfg), lva_(lva) {}

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

class AliasEliminator : public MixedModeMutator {
 public:
  Expr VisitExpr_(const LetNode* let_node) override {
    Expr expr = GetRef<Expr>(let_node);
    LetList ll;
    std::vector<Var> bound_vars;

    auto set_alias = [&](const Var& alias, const VarNode* alias_of_n) {
      Var alias_of = GetRef<Var>(alias_of_n);
      if (alias_.count(alias_of)) {
        alias_[alias] = alias_[alias_of];
      } else {
        alias_[alias] = alias_of;
      }
      bound_vars.push_back(alias);
    };

    while (const LetNode* inner_let_node = expr.as<LetNode>()) {
      const Var& var = inner_let_node->var;
      const Expr& val = inner_let_node->value;
      bool aliased = false;
      ICHECK(!alias_.count(var));

      if (const VarNode* alias_of_n = AsIgnoringOnDevice<VarNode>(val)) {
        set_alias(var, alias_of_n);
        aliased = true;
      } else if (AsIgnoringOnDevice<CallNode>(val)) {
        // Copying to the same device is aliasing.
        Expr unwrapped = IgnoreOnDevice(val);
        DeviceCopyProps copy_props = GetDeviceCopyProps(unwrapped);
        if (copy_props.body.defined()) {
          if (copy_props.src_virtual_device->device_type() ==
                  copy_props.dst_virtual_device->device_type() &&
              copy_props.src_virtual_device->virtual_device_id ==
                  copy_props.dst_virtual_device->virtual_device_id) {
            Expr to_copy = Downcast<Call>(unwrapped)->args[0];
            if (const VarNode* alias_of_n = to_copy.as<VarNode>()) {
              set_alias(var, alias_of_n);
              aliased = true;
            }
          }
        }
      }

      if (!aliased) {
        ll.Push(var, VisitExpr(val));
      }

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
    if (alias_.count(var)) {
      return alias_[var];
    }
    return var;
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    Expr new_body = VisitExpr(func_node->body);
    Function result = GetRef<Function>(func_node);
    if (!new_body.same_as(func_node->body)) {
      result = Function(func_node->params, new_body, func_node->ret_type, func_node->type_params,
                        func_node->attrs, func_node->span);
    }
    return result;
  }

  // The only register-level aliasing that occurs in Match expressions is when
  // the deconstructed expression is a Var, and the matched pattern is also a Var.
  Expr VisitExpr_(const MatchNode* match_node) override {
    if (const VarNode* data_var = AsIgnoringOnDevice<VarNode>(match_node->data)) {
      std::vector<Clause> new_clauses;
      for (const Clause& clause : match_node->clauses) {
        const PatternVarNode* pv_node = nullptr;
        if ((pv_node = clause->lhs.as<PatternVarNode>())) {
          alias_[pv_node->var] = GetRef<Var>(data_var);
        }
        new_clauses.push_back(Clause(clause->lhs, VisitExpr(clause->rhs)));
        if (pv_node) {
          alias_.erase(pv_node->var);
        }
      }
      return Match(match_node->data, new_clauses, match_node->complete, match_node->span);
    } else {
      return ExprMutator::VisitExpr_(match_node);
    }
  }

 private:
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> alias_;
};

Pass VMPlanMemory() {
  auto pass_func = [](Function f, IRModule m, PassContext pc) -> Function {
    f = Downcast<Function>(AliasEliminator().Mutate(f));
    ControlFlowGraph cfg = ControlFlowGraph::Create(f);
    UseDefAnalysis use_def = UseDefAnalysis::Analyze(cfg);
    LivenessAnalysis lva = LivenessAnalysis::Analyze(cfg, use_def);
    KillInserter ki(&cfg, &lva);
    Function nf = Downcast<Function>(ki.Mutate(f));
    return nf;
  };
  return CreateFunctionPass(pass_func, 0, "VMPlanMemory", {});
}

TVM_REGISTER_GLOBAL("relay._transform.VMPlanMemory").set_body_typed(VMPlanMemory);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
