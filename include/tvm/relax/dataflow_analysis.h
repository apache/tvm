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

#include <utility>

namespace tvm {
namespace relax {

/*! \brief For dataflow analysis, we need to have a control flow graph.
 *  We will organize this graphs by bindings, which allows analyses to
 *  state their results for each binding in a SeqExpr.
 *
 *  There are a few cases that have to be handled:
 *  1. A normal binding (most common)ICHECK
 *  2. The condition expression in an If node (a "split" point)
 *  3. A merge point (the variable to which an If node is bound: it is a "merge" between
 *     the SeqExprs in the true and false branches)
 *  4. The body expression in a SeqExpr (not actually bound)
 */
enum BindingNodeKind : int { kBinding = 0, kIfCond = 1, kIfMerge = 2, kSeqBody = 3 };

class GraphBindingNode : public Object {
 public:
  /*! \brief The SeqExpr the binding resides in. */
  SeqExpr seq;

  /*! \brief The arguments to the binding. Only the first binding in the graph has arguments
   * (i.e., the function arguments). */
  Array<Var> args;

  /*! \brief Index of the binding block in the SeqExpr where the binding is found.
   *  Convention: We put the SeqExpr body at one block past the final block. */
  size_t block_idx;

  /*! \brief Index of the binding within the binding block corresponding to this binding.
   *  Convention: Both the If condition and merge are mapped to the same index.
   *  We use the kind to distinguish. */
  size_t binding_idx;

  /*! \brief The kind of binding this is. */
  BindingNodeKind kind;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("seq", &seq);
    v->Visit("args", &args);
    v->Visit("block_idx", &block_idx);
    v->Visit("binding_idx", &binding_idx);
    v->Visit("kind", &kind);
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.analysis.GraphBinding";
  TVM_DECLARE_BASE_OBJECT_INFO(GraphBindingNode, Object);
};

/*! \brief Representation of a binding in the control flow graph */
class GraphBinding : public ObjectRef {
 public:
  /*!
   * \brief Create a GraphBinding. See the docs on GraphBindingNode for further details.
   *
   * \param seq: The SeqExpr in which the binding resides.
   * \param args: The arguments to the binding (only nonempty for the first binding:
   *   these will be the function arguments)
   * \param block_idx: The index of the BindingBlock in the SeqExpr
   *   where the binding resides (for the return expression, use one past the final block).
   * \param binding_idx: The index of the binding in the BindingBlock corresponding to the binding.
   * \param kind: The kind of binding this is. (Used especially to distinguish If node conditions
   * from the merge after the If)
   */
  TVM_DLL static GraphBinding Create(const SeqExpr& seq, const Array<Var>& args, size_t block_idx,
                                     size_t binding_idx, BindingNodeKind kind);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(GraphBinding, ObjectRef, GraphBindingNode);
};

/* A control flow graph corresponding to a function.
 */
class ControlFlowGraphNode : public Object {
 public:
  /*! \brief The bindings in the graph. 0 is the entry point. */
  Array<GraphBinding> bindings;
  /*! \brief The ith member is the list of predecessors (indices) to binding i in bindings. */
  Array<Array<Integer>> preds;
  /*! \brief The ith member is the list of successors (indices) to binding i in bindings. */
  Array<Array<Integer>> succs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("bindings", &bindings);
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
   * \param bindings: The bindings in the graph
   * \param preds: List of lists of predecessors to each binding.
   * \param succs: List of lists of successors to each binding.
   */
  TVM_DLL static ControlFlowGraph Create(const Array<GraphBinding>& bindings,
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
 *   Adrian Sampson's course material, except binding by binding
 *   instead of basic block by basic block:
 *   https://www.cs.cornell.edu/courses/cs6120/2020fa/lesson/4/
 *
 *  The analysis creates input and output maps (mapping binding indices to a domain),
 *  sets the initial input and output for each binding to the init value, and then
 *  performs a traversal of the CFG (BFS in this implementation, since unlike the general case,
 *  we do not have loops) and uses the transfer and merge function to update the inputs and
 *  outputs. The analysis can proceed forwards (from binding 0 onwards) or backwards (from the
 *  last binding back), flipping the roles of the input and output maps in the cases.
 *
 * \param forward Whether to perform a forward or backward analysis
 * \param cfg The input control flow graph
 * \param init The value corresponding to an initial domain
 * \param transfer_func Given an input domain and a binding, determine the resulting domain
 * \param merge_func Given a set of domains, combine them to form a single new domain
 *   (note: in Relax, a binding can never have more than two predecessors/successors)
 *
 * \return Two arrays, the first being the "input map" (domain being passed *into*
 *   each binding in the CFG) and the second being the "output map" (the domain
 *   being passed *out of* the corresponding binding)
 */
std::pair<Array<ObjectRef>, Array<ObjectRef>> DataflowAnalysis(
    const ControlFlowGraph& cfg, const ObjectRef& init,
    std::function<ObjectRef(const GraphBinding&, const ObjectRef&)> transfer_func,
    std::function<ObjectRef(const ObjectRef&, const ObjectRef&)> merge_func, bool forward = true);

/*! \brief A helper function. Given an index into a SeqExpr, give the index of the GraphBinding
 *  in the CFG.
 *
 * \param cfg The control flow graph.
 * \param seq The target SeqExpr.
 * \param block_idx The target block in the SeqExpr.
 *   Convention: Use one past the last block to indicate the SeqExpr body.
 * \param binding_idx The target binding in the target block.
 * \param match_cond If the RHS of the target binding is an IfExpr, then if match_cond is true,
 *   the returned index will be for the condition node; otherwise it will be for the merge node.
 */
size_t GetBindingIndex(const ControlFlowGraph& cfg, const SeqExpr& seq, size_t block_idx,
                       size_t binding_idx, bool match_cond);

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_DATAFLOW_ANALYSIS_H_
