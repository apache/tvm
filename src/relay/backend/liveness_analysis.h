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
 * \file src/relay/backend/liveness_analysis.h
 * \brief  Analysis that collects the live variables before and after each node.
 * NOTE: the input IR should be in ANF.
 */

#ifndef TVM_RELAY_BACKEND_LIVENESS_ANALYSIS_H_
#define TVM_RELAY_BACKEND_LIVENESS_ANALYSIS_H_

#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../support/arena.h"
#include "../op/memory/device_copy.h"
#include "../transforms/device_aware_visitors.h"
#include "../transforms/let_list.h"

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

    static BasicBlockPtr Make(support::Arena* arena) { return arena->make<BasicBlock>(); }
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

  ControlFlowGraph Create(Arena* arena, const Expr& body);

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
  void Succ(BasicBlockPtr from, BasicBlockPtr to);

#define DEFAULT_CFG(OP)                                       \
  void VisitExpr_(const OP* op, BasicBlockPtr parent) final { \
    NodePtr n = Node::Make(arena_, parent, GetRef<Expr>(op)); \
    cfg_.reverse_post_order.push_back(n);                     \
  }

  void VisitExpr_(const FunctionNode* f, BasicBlockPtr parent) final;
  void VisitExpr_(const LetNode* let_node, BasicBlockPtr parent) final;
  void VisitExpr_(const IfNode* if_node, BasicBlockPtr parent);
  void VisitExpr_(const MatchNode* match_node, BasicBlockPtr parent);

  DEFAULT_CFG(VarNode);
  DEFAULT_CFG(GlobalVarNode);
  DEFAULT_CFG(ConstantNode);
  DEFAULT_CFG(CallNode);
  DEFAULT_CFG(OpNode);
  DEFAULT_CFG(TupleNode);
  DEFAULT_CFG(TupleGetItemNode);
};

/*!
 * \brief Helper class for collecting the variables used/read by an expression. NOTE: for If exprs,
 * only the condition is included (not the branches). Similarly, for Match exprs only the value
 * being deconstructed is included.
 */
class VarUseCollector : public ExprFunctor<VarSet(const Expr& e)> {
 public:
  VarSet VisitExpr_(const VarNode* var_node);
  VarSet VisitExpr_(const CallNode* call_node);
  VarSet VisitExpr_(const TupleNode* tuple_node);
  VarSet VisitExpr_(const TupleGetItemNode* get_node);
  VarSet VisitExpr_(const IfNode* if_node);
  VarSet VisitExpr_(const MatchNode* match_node);

  VarSet VisitExpr_(const ConstructorNode* cons_node) { return {}; }
  VarSet VisitExpr_(const GlobalVarNode* gvar_node) { return {}; }
  VarSet VisitExpr_(const ConstantNode* const_node) { return {}; }
  VarSet VisitExpr_(const OpNode* op_node) { return {}; }
  VarSet VisitExpr_(const FunctionNode* func_node) { return {}; }
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

  static UseDefAnalysis Analyze(const CFG& cfg);
};

/*! \brief Returns whether \p a and \p b are the same set of vars. */
bool SetEqual(const VarSet& a, const VarSet& b);

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
  static LivenessAnalysis Analyze(const ControlFlowGraph& cfg, const UseDefAnalysis& use_def);
};

}  // namespace transform
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_LIVENESS_ANALYSIS_H_
