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

namespace tvm {
namespace relay {
namespace transform {

using support::Arena;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

// TODO(@altanh, @mbs, @mbrookhart): we should do a survey of all "*-flow graphs" in the codebase
//                                   to see what can be deduplicated.

// TODO(@altanh): support Relay Refs once/if they are supported by the VM.

/*!
 * \brief A representation of an input expression (typically a Function) as a directed graph of
 * basic blocks, with edges between basic blocks corresponding to control flow branching.
 */
class ControlFlowGraph {
 public:
  struct Node;
  struct BasicBlock;

  using NodePtr = Node*;
  using BasicBlockPtr = BasicBlock*;

  /*!
   * \brief A chunk of IR that does not have any control flow branching. At this stage in the IR,
   * basic blocks correspond to:
   *   (1) a sequence of nested Let expressions, where each node in the block corresponds to a
   *       binding and the last node is either the (non-Let) body or a binding that branches
   *       (e.g. "let %x = if (%c) { true_block } else { false_block }").
   *   (2) an atomic expression representing the target expression of a control flow branch, e.g.
   *       %v and %u in "let %x = if (%c) { %v } else { %u }".
   */
  struct BasicBlock {
    // The nodes of the basic block.
    std::vector<NodePtr> nodes;
    // The predecessor basic blocks.
    std::vector<BasicBlockPtr> pred;
    // The successor basic blocks.
    std::vector<BasicBlockPtr> succ;

    static BasicBlockPtr Make(Arena* arena) { return arena->make<BasicBlock>(); }
  };

  /*!
   * \brief Roughly corresponds to a "statement" in the IR, such as an individual binding in a
   * basic block or the "return value" of a block. Each node maps to a single corresponding expr in
   * the IR, but the converse is not true (e.g. in the case of variables).
   */
  struct Node {
    /*! \brief The basic block this node belongs to. */
    BasicBlockPtr parent;
    /*! \brief The index into the parent basic block where this node is. */
    size_t index;
    /*! \brief The expr this node corresponds to. */
    Expr expr;

    /*! \brief Returns whether or not this node is the first one in the parent basic block. */
    bool IsFirst() const { return index == 0; }

    /*! \brief Returns whether or not this node is the last one in the parent basic block. */
    bool IsLast() const { return index == parent->nodes.size() - 1; }

    /*! \brief Returns the predecessor nodes of this node. */
    std::vector<NodePtr> GetPred() const {
      std::vector<NodePtr> pred;
      if (IsFirst()) {
        for (const BasicBlockPtr& pred_block : parent->pred) {
          pred.push_back(pred_block->nodes.back());
        }
      } else {
        pred.push_back(parent->nodes[index - 1]);
      }
      return pred;
    }

    /*! \brief Returns the successor nodes of this node. */
    std::vector<NodePtr> GetSucc() const {
      std::vector<NodePtr> succ;
      if (IsLast()) {
        for (const BasicBlockPtr& succ_block : parent->succ) {
          succ.push_back(succ_block->nodes.front());
        }
      } else {
        succ.push_back(parent->nodes[index + 1]);
      }
      return succ;
    }

    /*! \brief Creates a node with the given expr and appends it to the parent basic block. */
    static NodePtr Make(Arena* arena, BasicBlockPtr parent, Expr expr) {
      NodePtr n = arena->make<Node>();
      n->parent = parent;
      n->expr = expr;
      n->index = parent->nodes.size();
      parent->nodes.push_back(n);
      return n;
    }
  };

  /*! \brief The basic block where control flow begins. */
  BasicBlockPtr entry;

  /*!
   * \brief Mapping from Let expressions to their corresponding nodes. Note that Let expressions
   * are never shared in ANF (unlike vars), so this is an injection.
   */
  std::unordered_map<Expr, NodePtr, ObjectPtrHash, ObjectPtrEqual> let_map;

  /*! \brief The nodes of the CFG in reverse post order. */
  std::vector<NodePtr> reverse_post_order;

  /*! \brief Creates and returns the CFG of the given expression. */
  static ControlFlowGraph Create(Arena* arena, const Expr& body);

 private:
  class Creator;
};

/*! \brief Helper class for building CFGs. */
class ControlFlowGraph::Creator : private ExprFunctor<void(const Expr&, BasicBlockPtr)> {
 public:
  Creator() {}

  ControlFlowGraph Create(Arena* arena, const Expr& body) {
    arena_ = arena;
    cfg_.entry = BasicBlock::Make(arena);
    VisitExpr(body, cfg_.entry);
    return std::move(cfg_);
  }

 private:
  /*! \brief The arena allocator. */
  Arena* arena_;

  /*! \brief The CFG being built. */
  ControlFlowGraph cfg_;
  /*!
   * \brief Whether or not we are in a function. CFGs do not support nested functions so this is
   * used to error out in such a case.
   */
  bool in_func_ = false;

  /*!
   * \brief Link \p to as a successor block to \p from.
   */
  void Succ(BasicBlockPtr from, BasicBlockPtr to) {
    from->succ.push_back(to);
    to->pred.push_back(from);
  }

#define DEFAULT_CFG(OP)                                       \
  void VisitExpr_(const OP* op, BasicBlockPtr parent) final { \
    NodePtr n = Node::Make(arena_, parent, GetRef<Expr>(op)); \
    cfg_.reverse_post_order.push_back(n);                     \
  }

  void VisitExpr_(const FunctionNode* f, BasicBlockPtr parent) final {
    ICHECK(!in_func_) << "nested functions not supported by CFG analysis";
    in_func_ = true;

    // Unwrap the nested function and proceed normally.
    if (f->HasNonzeroAttr(attr::kClosure)) {
      ICHECK(f->body.as<FunctionNode>());
      return VisitExpr(Downcast<Function>(f->body)->body, parent);
    }

    return VisitExpr(f->body, parent);
  }

  void VisitExpr_(const LetNode* let_node, BasicBlockPtr parent) final {
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

ControlFlowGraph ControlFlowGraph::Create(Arena* arena, const Expr& body) {
  return Creator().Create(arena, body);
}

/*!
 * \brief Helper class for collecting the variables used/read by an expression. NOTE: for If exprs,
 * only the condition is included (not the branches). Similarly, for Match exprs only the value
 * being deconstructed is included.
 */
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

/*!
 * \brief Analysis that collects the variables used and defined at each node.
 */
struct UseDefAnalysis {
  using CFG = ControlFlowGraph;

  /*! \brief Mapping of node -> variables used/read by node. */
  std::unordered_map<CFG::NodePtr, VarSet> use;

  /*! \brief Mapping of node -> variable defined/written by node. */
  std::unordered_map<CFG::NodePtr, Var> def;

  VarUseCollector use_collector;

  static UseDefAnalysis Analyze(const CFG& cfg) {
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
};

/*! \brief Returns whether \p a and \p b are the same set of vars. */
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

/*!
 * \brief Analysis that collects the live variables before and after each node.
 */
struct LivenessAnalysis {
  using CFG = ControlFlowGraph;

  /*! \brief Mapping of node -> set of variables live before node. */
  std::unordered_map<CFG::NodePtr, VarSet> live_in;

  /*! \brief Mapping of node -> set of variables live after node. */
  std::unordered_map<CFG::NodePtr, VarSet> live_out;

  /*!
   * \brief Analyze the input \p cfg (using info from \p use_def).
   *
   * \param cfg The input control flow graph.
   * \param use_def Use-def analysis of \p cfg.
   * \return LivenessAnalysis
   */
  static LivenessAnalysis Analyze(const ControlFlowGraph& cfg, const UseDefAnalysis& use_def) {
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
};

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

  Expr VisitExpr_(const VarNode* var_node) override {
    Var var = GetRef<Var>(var_node);
    if (alias_.count(var)) {
      return alias_[var];
    }
    return var;
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
