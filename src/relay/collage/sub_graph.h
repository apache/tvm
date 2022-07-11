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
 * \file src/relay/collage/sub_graph.h
 * \brief Represents a sub-graph of an overall Relay expression.
 */

#ifndef TVM_RELAY_COLLAGE_SUB_GRAPH_H_
#define TVM_RELAY_COLLAGE_SUB_GRAPH_H_

#include <tvm/ir/transform.h>
#include <tvm/relay/op_attr_types.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../ir/dataflow_matcher_impl.h"
#include "../ir/indexed_graph.h"
#include "./dataflow_graph.h"
#include "./index_set.h"

namespace tvm {
namespace relay {
namespace collage {

/*! \brief Returns operator pattern kind as single-letter string. */
std::string KindToString(OpPatternKind kind);

/*!
 * \brief Returns a kind and label for the single \p sub_expr, ignoring its nested sub expressions.
 */
std::pair<OpPatternKind, std::string> SubExprKindAndLabel(const Expr& sub_expr);

/*!
 * \brief Returns a kind and label for all the nodes in \p inside.
 */
std::pair<OpPatternKind, std::string> SubGraphKindAndLabel(const DataflowGraph& dataflow_graph,
                                                           const IndexSet& inside);

/*!
 * \brief Returns the index set representing all the sub-expression matched by \p matcher.
 */
IndexSet MatcherToIndexSet(const DFPatternMatcher& matcher);

/*!
 * \brief Configuration controlling which sub-graphs are considered valid.
 */
struct SubGraphConfig {
  /*! \brief Maximum number of exit nodes in the sub-graph, or zero if no limit. */
  size_t max_exits = 0;
  /*!
   * \brief Whether a node inside the sub-graph may flow to nodes both inside and outside
   * the sub-graph (which we call a 'tap'). Note that it is still possible to have multiple outputs
   * even with this flag false.
   */
  bool allow_taps = false;
  /*!
   * \brief Maximum allowed sub-graph depth, or zero if no-limit.
   */
  size_t max_depth = 0;

  std::string ToString() const;
};

class SubGraph;
using FunctionAttrsMap = Map<String, ObjectRef>;

/*!
 * \brief A nested sub-graph is a sub-graph which is to be nested inside a function as part of some
 * enclosing sub-graph.
 *
 * Extraction yields a function with input nodes replaced by parameters and exit nodes in the
 * function result. Rewriting replaces the sub-graph with a call to that function, and all
 * outputs with (projections from) the call result.
 *
 * (Note that it's tempting to move attrs_ into \p SubGraphNode and thus avoid this class.
 * However we found the implementation was easier to understand in this form since it makes
 * the result of \p Extract unambiguous.)
 */
class NestedSubGraphNode : public Object {
 public:
  /*! \brief The nested sub-graph. */
  ObjectRef /* actually SubGraph */ sub_graph_obj_;
  /*! \brief Attributes (possibly empty) to attach to the extracted function. */
  FunctionAttrsMap attrs_;

  void VisitAttrs(AttrVisitor* v);

  SubGraph sub_graph() const;

  bool operator==(const NestedSubGraphNode& that) const;
  bool operator!=(const NestedSubGraphNode& that) const { return !(*this == that); }
  bool operator<(const NestedSubGraphNode& that) const;
  size_t hash() const;

  std::string ToString() const;

  /*!
   * \brief Returns the function representing this nested sub-graph within the overall expression
   * represented by \p dataflow_graph:
   *  - All sub-graph inputs become parameters.
   *  - All sub-graph outputs become function results (either directly or as a field in a tuple).
   *  - The function has attrs_ for attributes (which may be empty).
   *  - The function body accounts for any rewrites implied by the nested sub-graph.
   */
  Function Extract(const DataflowGraph& dataflow_graph) const;

  /*!
   * \brief Returns \p expr rewritten to encode the partitioning implied by this nested sub-graph.
   *
   * It is valid for \p expr to not be the same as \p dataflow_graph.expr(), however all nodes
   * inside this nested sub-graph must correspond to nodes shared between \p dataflow_graph.expr()
   * and \p expr. See \p SubGraph::ParallelRewrite below.
   */
  Expr Rewrite(const DataflowGraph& dataflow_graph, const Expr& expr) const;

  static constexpr const char* _type_key = "relay.collage.NestedSubGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(NestedSubGraphNode, Object);
};

class NestedSubGraph : public ObjectRef {
 public:
  NestedSubGraph(SubGraph sub_graph, FunctionAttrsMap attrs);

  /*!
   * \brief Returns copy of this nested sub-graph with all indexes substituted according to
   * \p subst, whose range is w.r.t. \p new_dataflow_graph.
   */
  NestedSubGraph Subst(const DataflowGraph& new_dataflow_graph,
                       const std::unordered_map<PostDfsIndex, PostDfsIndex>& subst) const;

  /*!
   * \brief Returns true if this can be safely unioned.
   */
  bool TriviallyUnionable(const NestedSubGraph& that) const;

  /*!
   * \brief Returns the disjoint union of this and \p that nested sub-graphs, which must agree on
   * their attributes.
   */
  NestedSubGraph DisjointUnion(const DataflowGraph& dataflow_graph,
                               const NestedSubGraph& that) const;

  /*!
   * \brief Returns \p expr rewritten according to all the given nested sub-graphs. The
   * nested sub-graphs can be given in any order, but must be disjoint.
   *
   * It is valid for \p expr to not be the same as \p dataflow_graph.expr(), however all nodes
   * inside the nested sub-graphs must correspond to nodes shared between \p dataflow_graph.expr()
   * and \p expr. See \p SubGraph::ParallelRewrite below.
   */
  static Expr ParallelRewrite(const DataflowGraph& dataflow_graph, const Expr& expr,
                              std::vector<NestedSubGraph> nested_sub_graphs);

  TVM_DEFINE_OBJECT_REF_METHODS(NestedSubGraph, ObjectRef, NestedSubGraphNode);
};

using NestedSubGraphs = Array<NestedSubGraph>;

/*!
 * \brief A compact representation of a sub-graph within an (implied) overall Relay expression.
 *
 * Sub-graphs can be used to represent partitions/kernels/composite functions without having to
 * pay the cost of constructing or rewriting any expressions. We also allow 'extracting' a
 * function to use for measuring a partition/kernel's latency independently from 'rewriting'
 * the overall Relay expression since only a tiny subset of candidate partitions will end up being
 * needed after Collage has completed its search.
 *
 * We expect O(thousands) of sub-graphs to be in flight while processing a given model, so we are
 * mindful of space overhead.
 *
 * A sub-graph classifies every dataflow node of the overall expression as either 'inside' or
 * 'outside' the sub-graph. Obviously not all such divisions make sense, for example it is not
 * valid for an inside node to feed into another inside node via outside nodes. We provide the
 * \p IsValid method to check for validity, and \p SubGraphConfig to control which validity rules
 * apply (such as maximum depth).
 *
 * We generally work with the \p DataflowGraph representation of the overall Relay expression
 * rather than the expression itself. We use the post-dfs visit index to uniquely refer to
 * expression nodes.
 *
 * As well as 'inside' and 'outside' we have four other flavors of dataflow nodes, all uniquely
 * determined from the 'inside' nodes:
 *  - 'entry' nodes are those inside with at least one dataflow input outside.
 *  - 'exit' nodes are  those inside with at least one dataflow output outside, or which
 *    are considered 'external' in the underlying dataflow graph (eg because they represent
 *    the result of the overall function).
 *  - 'input' nodes are those outside with at least one dataflow output inside.
 *  - 'output' nodes are those outside with at least one dataflow input inside.
 * Index sets for these are cached with the sub-graph for performance.
 *
 * It is valid to have multiple entry nodes (we can bind a parameter for each). It may be valid to
 * have multiple exit nodes (we can build a tuple of all such). It may be valid to have exit nodes
 * which also contribute to other inside nodes (ie represent a 'tap' on an intermediate result).
 *
 * Sub-graphs are closed under:
 *  - Disjoint union.
 *  - Wrapping by a function with given attributes (see \p NestedSubGraph above). This can be used
 *    to encode "Composite" functions, or to represent a candidate kernel within a "Primitive"
 *    function. (By combining 'wrapping' with 'union' we can encode, eg, 'this sub-graph should
 *    be placed inside a primitive function which itself may have calls to composite functions).
 *  - Substitution, which allows a sub-graph w.r.t. one dataflow graph to be transformed to
 *    match some other (typically smaller) dataflow graph.
 *
 * See the subclasses of \p PartitionRule for how sub-graphs are built and combined during Collage
 * search.
 *
 * To support some of the \p OpPatternKind-based fusion rule processing we give sub-graphs
 * a kind, which is generally the maximum of the kinds of all the operator calls appearing
 * inside it. We also given sub-graphs a (not necessarily unique) label to help debugging
 * and guide the selection of global symbol names.
 */
class SubGraphNode : public Object {
 public:
  /*!
   * \brief Which sub-expressions are inside the sub-graph (using their post-dfs indexes w.r.t.
   * the implied DataflowGraph).
   */
  IndexSet inside_;

  /*!
   * \brief Index of first and last inside nodes.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  PostDfsIndex first_inside_index_ = 0;
  PostDfsIndex last_inside_index_ = 0;

  /*!
   * \brief Which sub-expressions are entry/exit/input/output for this sub-graph.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  IndexSet entry_;
  IndexSet exit_;
  IndexSet input_;
  IndexSet output_;

  /*!
   * \brief Maximum depth of any dataflow path from an entry to an output sub-expression.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  size_t depth_ = 0;

  /*!
   * \brief The \p OpPatternKind summarizing the input/output behavior of the sub-graph.
   *
   * A sub-graph consisting of a single Relay expression node is given kind:
   *  - For Call to a Relay operator, the "TOpPattern" attribute of that operator (provided the
   *    call does not involve data-dependent dynamic shapes).
   *  - For Call to Relay Function, the "TOpPattern" attribute of the function (provided it has
   *    that attribute)
   *  - For Constants, \p kElemWise.
   *  - For Tuple and tuple projections, \p kInjective (provided all tuple fields are of tensor
   *    type)
   *  - All other nodes \p kOpaque.
   * Sub-graphs with more than one node have the maximum of the kind of each node.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  OpPatternKind kind_ = kOpaque;

  /*!
   * \brief A label for the sub-graph. Not guaranteed to be unique, but is a human-readable summary
   * of the sub-graph which can help with debugging and guide the selection of global symbol names.
   */
  String label_;

  /*!
   * \brief Nested sub-graphs of this sub-graph which must be represented by functions. These must
   * be disjoint, but it's ok for this sub-graph to have nodes not inside any nested sub-graph.
   */
  NestedSubGraphs nested_sub_graphs_;

  void VisitAttrs(AttrVisitor* v);

  // TODO(mbs): 'Anchor nodes' and rules for unioning them.
  // In FuseOps it's just the unique kEWiseFusable node, if any.
  // I'd like to allow writing vertical fusion rules, eg if two candidates are directly
  // connected and have nn.conv2d anchors allow their join.
  // I'd also like to allow horizontal fusion rules, eg if two candidates are not directly
  // connected but could be joined without producing invalid (eg cyclic) and have nn.conv2d anchors
  // then do so. Come back to this.

  /*! \brief Number of nodes in overall dataflow graph. */
  size_t overall_size() const { return inside_.end_index(); }

  bool IsEmpty() const { return inside_.IsZero(); }

  /*! \brief Number of nodes in sub-graph. */
  size_t Size() const { return inside_.PopCount(); }

  /*!
   * \brief Returns the dataflow nodes downstream of all exit nodes.
   */
  IndexSet Downstream(const DataflowGraph& dataflow_graph) const;

  /*!
   * \brief Returns true if this sub-graph is valid. Ie:
   *  - no output of the sub-graph can flow to any input of the sub-graph (otherwise we'd end up
   *    with a dataflow cycle when we partition).
   *  - all inputs and outputs of the sub-graph are in the same scope, ie not separated by
   *    control flow (otherwise there'd be no consistent program point at which to eval the
   *    partitioned function).
   *  - no more than config.max_outputs outputs are required.
   *  - if config.allow_taps is false, no inside node has outputs to nodes both inside and
   *    outside the sub-graph.
   */
  bool IsValid(const DataflowGraph& dataflow_graph, const SubGraphConfig& config) const;

  /*!
   * \brief Returns this sub-graph extracted as a stand-alone function. The function will have
   * no attributes, and is suitable for building and profiling by the \p CostEstimator.
   */
  Function ExtractAsFunction(const DataflowGraph& dataflow_graph) const;

  /*!
   * \brief Returns \p expr rewritten to encode the partitioning implied by this sub-graph.
   *
   * It is valid for \p expr to not be the same as \p dataflow_graph.expr(), however all nodes
   * inside this sub-graph must correspond to nodes shared between \p dataflow_graph.expr() and
   * \p expr. See \p SubGraph::ParallelRewrite below.
   */
  Expr Rewrite(const DataflowGraph& dataflow_graph, const Expr& expr) const;

  std::string ToString() const;

  bool operator==(const SubGraphNode& that) const;
  bool operator!=(const SubGraphNode& that) const { return !(*this == that); }
  bool operator<(const SubGraphNode& that) const;
  size_t hash() const;

 private:
  /*! \brief Initialize the entry/exit/input/output sets given the inside and \p dataflow_graph. */
  void Init(const DataflowGraph& dataflow_graph);

  /*! \brief Calculates and returns the maximum path depth. */
  size_t Depth(const DataflowGraph& dataflow_graph) const;

  /*! \brief Returns true if any (input/output) of node is (outside/inside) the sub-graph. */
  bool AnyInputOutside(const DataflowGraph::Node* node) const;
  bool AnyInputInside(const DataflowGraph::Node* node) const;
  bool AnyOutputOutside(const DataflowGraph::Node* node) const;
  bool AnyOutputInside(const DataflowGraph::Node* node) const;

 public:
  static constexpr const char* _type_key = "relay.collage.SubGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubGraphNode, Object);

  friend class SubGraph;
};

class SubGraph : public ObjectRef {
 public:
  /*! \brief Primitive constructor. The following constructors are generally more convenient. */
  SubGraph(const DataflowGraph& dataflow_graph, IndexSet inside, OpPatternKind kind = kOpaque,
           String label = {}, std::vector<NestedSubGraph> nested_sub_graphs = {});

  /*! \brief Constructs the empty sub-graph for \p dataflow_graph. */
  explicit SubGraph(const DataflowGraph& dataflow_graph);

  /*! \brief Returns true if this and that are disjoint. */
  bool AreDisjoint(const SubGraph& that) const;

  /*!
   * \brief Returns true if:
   *  - \p this and \p that are disjoint, and
   *  - an output node of \p this coincides with an entry node of \p that, and
   *  - \p this and \p that are not obviously invalid after \p DisjointUnion
   *    (eg because such a sub-graph would produce a cycle).
   * Note however that the \p DisjointUnion may not necessarily be valid even with the above
   * checks.
   */
  bool AreTouching(const DataflowGraph& dataflow_graph, const SubGraph& that) const;

  /*!
   * \brief Returns true if:
   *  - all the outputs of \p this are entries for \p that, and
   *  - all the inputs of \p that are exits for \p this.
   */
  bool AreSelfContained(const SubGraph& that) const;

  /*!
   * \brief Returns disjoint union of this and \p that sub-graphs. The result may not be valid.
   */
  SubGraph DisjointUnion(const DataflowGraph& dataflow_graph, const SubGraph& that) const;

  /*!
   * \brief Returns copy of this sub-graph with all nodes placed inside a nested sub-graph with
   * given attributes.
   */
  SubGraph WithAttrs(const DataflowGraph& dataflow_graph, FunctionAttrsMap attrs) const;

  /*!
   * \brief Returns copy of this sub-graph with all indexes substituted according to \p subst,
   * whose range is w.r.t. \p new_dataflow_graph.
   */
  SubGraph Subst(const DataflowGraph& new_dataflow_graph,
                 const std::unordered_map<PostDfsIndex, PostDfsIndex>& subst) const;

  /*!
   * \brief Returns the root expression of \p dataflow_graph rewritten according to all the
   * given sub-graphs. The sub-graphs can be given in any order, but must be disjoint.
   */
  static Expr ParallelRewrite(const DataflowGraph& dataflow_graph,
                              std::vector<SubGraph> sub_graphs);

  TVM_DEFINE_OBJECT_REF_METHODS(SubGraph, ObjectRef, SubGraphNode);
};

struct SubGraphEqual {
  bool operator()(const SubGraph& left, const SubGraph& right) const {
    return *left.get() == *right.get();
  }
};

struct SubGraphHash {
  size_t operator()(const SubGraph& sub_graph) const { return sub_graph->hash(); }
};

/*!
 * \brief Pass to partition every global function according to the post-dfs indexes
 * given in an array. Visible for testing from Python only, would never make sense to use
 * as a generic pass!
 */
tvm::transform::Pass PartitionOnIndexesForTesting(Array<Integer> indexes);

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_SUB_GRAPH_H_
