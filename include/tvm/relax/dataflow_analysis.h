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
 * \file tvm/relax/dataflow_analysis.h
 * \brief A reusable framework for dataflow analysis in Relax.
 *   Based on Adrian Sampson's course material:
 *   https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/4/
 *  Do not confuse with dataflow pattern matching (does not use this machinery)
 */

#ifndef TVM_RELAX_DATAFLOW_ANALYSIS_H_
#define TVM_RELAX_DATAFLOW_ANALYSIS_H_

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {

/*! \brief For dataflow analysis, we need to have a graph of basic blocks
 *  (i.e., a control flow graph).
 *  The trouble is that Relax's BindingBlocks are not necessarily basic blocks:
 *  A BindingBlock followed by a DataflowBlock followed by a BindingBlock
 *  is potentially a single basic blocks, whereas a single BindingBlock that
 *  contains an If expression may actually comprise multiple basic blocks.
 *  This representation is a lightweight way of representing basic blocks on top
 *  of Relax's AST
 */
class BasicBlockNode : public Object {
 public:
  /*! \brief The SeqExpr the basic block resides in.
   *  (In normal form, basic blocks cannot span multiple SeqExprs). */
  SeqExpr seq;

  /*! \brief The arguments to the basic block.
   *  If the basic block is the first in the function, args is the function arguments.
   *  The basic blocks corresponding to If branches have no arguments.
   *  The basic block corresponding to the merge point after the If
   *  will have one argument (corresponding to the merge of the value returned;
   *  this will be the variable that the If expression is bound to). */
  Array<Var> args;

  /*! \brief The final expression evaluated in the basic block.
   *  If the basic block ends with an If expression, the ret is the If *condition*.
   *  Otherwise, it will be the value returned by the SeqExpr
   *  (all other basic blocks will end where the SeqExpr ends).*/
  Expr ret;

  /*! \brief Index of the BindingBlock in the SeqExpr where the basic block starts
   *  (Convention: If the start_block_idx is past the final index of the SeqExpr,
   *  that means the basic block contains no bindings.) */
  size_t start_block_idx;

  /*! \brief Index of the binding in the BindingBlock where the basic block starts
   *  (convention: If the basic block is a merge point, use the index of the binding
   *  after the If node. Also, if the start_binding_idx is past the final index
   *  of the block, that means the basic block contains no bindings) */
  size_t start_binding_idx;

  /*! \brief Index of the BindingBlock in the SeqExpr where the basic block ends.
   *  (convention: If the basic block goes until the end of the SeqExpr,
   *   end_block_idx will be one _past_ the last index, i.e., seq->blocks.size()) */
  size_t end_block_idx;

  /*! \brief Index of the binding in the BindingBlock where the basic block ends
   *  (convention: If the end of the basic block is the end of the SeqExpr,
   *   end_binding_idx will be one _past_ the last idex, i.e., block->bindings.size()) */
  size_t end_binding_idx;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("seq", &seq);
    v->Visit("args", &args);
    v->Visit("ret", &ret);
    v->Visit("start_block_idx", &start_block_idx);
    v->Visit("start_binding_idx", &start_binding_idx);
    v->Visit("end_block_idx", &end_block_idx);
    v->Visit("end_binding_idx", &end_binding_idx);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.analysis.BasicBlock";
  TVM_DECLARE_BASE_OBJECT_INFO(BasicBlockNode, Object);
};

/* Representation of a basic block on top of Relax's AST.
 */
class BasicBlock : public ObjectRef {
 public:
  /*!
   * \brief Create a BasicBlock. See the docs on BasicBlockNode for further details.
   *
   * \param seq: The SeqExpr in which the basic block resides.
   * \param args: The arguments to the basic block.
   * \param ret: The final expression in the basic block.
   * \param start_block_idx: The index of the BindingBlock in the SeqExpr
   *   where the basic block starts.
   * \param start_binding_idx: The index of the binding in the BindingBlock where the
   *   basic block starts.
   * \param end_block_idx: The index of the BindingBlock in the SeqExpr
   *   where the basic block ends.
   * \param end_binding_idx: The index of the binding in the BindingBlock where the
   *   basic block ends.
   */
  TVM_DLL static BasicBlock Create(const SeqExpr& seq, const Array<Var>& args, const Expr& ret,
                                   size_t start_block_idx, size_t start_binding_idx,
                                   size_t end_block_idx, size_t end_binding_idx);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BasicBlock, ObjectRef, BasicBlockNode);
};

/* A control flow graph corresponding to a function.
 */
class ControlFlowGraphNode : public Object {
 public:
  /*! \brief The basic blocks in the graph. 0 is the entry point. */
  Array<BasicBlock> blocks;
  /*! \brief The ith member is the list of predecessors (indices) to block i in blocks. */
  Array<Array<Integer>> preds;
  /*! \brief The ith member is the list of successors (indices) to block i in blocks. */
  Array<Array<Integer>> succs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("blocks", &blocks);
    v->Visit("preds", &preds);
    v->Visit("succs", &succs);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.analysis.ControlFlowGraph";
  TVM_DECLARE_BASE_OBJECT_INFO(ControlFlowGraphNode, Object);
};

class ControlFlowGraph : public ObjectRef {
 public:
  /*!
   * \brief Create a ControlFlowGraph.
   *
   * \param blocks: The basic blocks corresponding to the graph nodes
   * \param preds: List of lists of predecessors to each basic block.
   * \param succs: List of lists of successors to each basic block.
   */
  TVM_DLL static ControlFlowGraph Create(const Array<BasicBlock>& blocks,
                                         const Array<Array<Integer>>& preds,
                                         const Array<Array<Integer>>& succs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ControlFlowGraph, ObjectRef, ControlFlowGraphNode);
};

/*!
 * \brief Extracts the control flow graph for a Relax function.
 * \param func The function. This conversion expects it to be normalized.
 * \return The control flow graph corresponding to the function.
 */
ControlFlowGraph ExtractCFG(const Function& func);

/*!
 * \brief Generic implementation of dataflow analysis, based on
 *   Adrian Sampson's course material:
 *   https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/4/
 *
 *  The analysis creates input and output maps (mapping basic block indices to a domain),
 *  sets the initial input and output for each basic block to the init value, and then
 *  performs a traversal of the CFG (BFS in this implementation, since unlike the general case,
 *  we do not have loops) and uses the transfer and merge function to update the inputs and
 *  outputs. The analysis can proceed forwards (from block 0 onwards) or backwards (from the last
 *  block back), flipping the roles of the input and output maps in the cases.
 *
 * \param forward Whether to perform a forward or backward analysis
 * \param cfg The input control flow graph
 * \param init The value corresponding to an initial domain
 * \param transfer_func Given an input domain and a basic block, determine the resulting domain
 * \param merge_func Given a set of domains, combine them to form a single new domain
 *   (note: in Relax, a basic block can never have more than two predecessors/successors)
 *
 * \return Two arrays, the first being the "input map" (domain being passed *into*
 *   each basic block in the CFG) and the second being the "output map" (the domain
 *   being passed *out of* the corresponding basic block)
 */
std::pair<Array<ObjectRef>, Array<ObjectRef>> DataflowAnalysis(
    const ControlFlowGraph& cfg, const ObjectRef& init,
    std::function<ObjectRef(const BasicBlock&, const ObjectRef&)> transfer_func,
    std::function<ObjectRef(const ObjectRef&, const ObjectRef&)> merge_func, bool forward = true);

}  // namespace relax
}  // namespace tvm
#endif