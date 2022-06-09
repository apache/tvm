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
 * \file src/tvm/relay/dataflow_matcher_impl.h
 * \brief The auxiliary data structure for dataflow matcher.
 */
#ifndef TVM_RELAY_IR_DATAFLOW_MATCHER_IMPL_H_
#define TVM_RELAY_IR_DATAFLOW_MATCHER_IMPL_H_

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/dataflow_pattern_functor.h>
#include <tvm/relay/expr_functor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "indexed_graph.h"

namespace tvm {
namespace relay {

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  explicit DFPatternMatcher(const IndexedGraph<Expr>* expr_graph) : expr_graph_(expr_graph) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Array<Expr>> GetMemo() { return Map<DFPattern, Array<Expr>>(memo_); }

  const IndexedGraph<Expr>::Node* expr_to_node(const Expr& expr) const {
    return expr_graph_->item_to_node(expr);
  }
  const IndexedGraph<Expr>::Node* index_to_node(size_t index) const {
    return expr_graph_->index_to_node(index);
  }
  size_t size() const { return expr_graph_->size(); }
  const std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual>& memo() const {
    return memo_;
  }
  const IndexedGraph<Expr>& expr_graph() const { return *expr_graph_; }

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const AltPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const IfPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const LetPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  bool MatchesPath(const DominatorPatternNode* op, const Expr& expr);
  bool DominatesParent(const DominatorPatternNode* op, const Expr& expr);

  const IndexedGraph<Expr>* expr_graph_;
  std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
  bool memoize_ = true;
};

/*!
 * \brief PatternGrouper does pre-rewriting pattern matching and analysis
 *
 * This class creates a number of groups of matched expressions, ensures they don't overlap, and
 * returns them to the caller for post-analysis rewriting.
 *
 * This is primarily needed to support the post-dominator analysis required for dominator pattern
 * matching.
 */
class PatternGrouper {
 public:
  /*! \brief Internal Group class for storing analysis */
  struct Group {
    Expr root_node;
    int gid;
    Map<DFPattern, Array<Expr>> matched_nodes;
    std::string name;
    Function function;
    Array<Expr> args;
  };

  /*! \brief Return the group assignments of expressions */
  inline const std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual>& GetGIDAssignments() {
    return gid_assignments_;
  }
  /*! \brief Group expressions that match the pattern */
  const std::unordered_map<int, Group>& GroupMatches(const DFPattern& pattern, const Expr& pre);

 protected:
  /*! \brief Iteratively traverse the Expression in pre-order to find subgraphs
   *
   * If we traverse the graph in post-order, we can run into situtations where a small subgraph will
   * match the pattern. Due to options like AltPattern, a larger subgraph with more nodes later in
   * the graph may also match the pattern. With post-order traversal, we mark the smaller subgraph
   * as matched and fail to catch the larger subgraph. This problem is fixed by using pre-order
   * traversal.
   */
  void VisitExprs();

  /*! \brief Create a group based on a matched expression */
  void CreateGroup(const Expr& expr);

  /*! \brief EmbedConst implements rules for embedding constants into partitioned functions or
   * lifting them into the function arguments.
   *
   * The rules depend on what pattern the ConstantNode matched.
   *
   * The basic rules are:
   *  If the constant matches ExprPattern(relay.const(*)) or a ConstantPattern(), embed the constant
   * in the partitioned function. If the constant matched an AltPattern, recursively check the
   * matched side of the pattern. For any other matching pattern (i.e, wildcard, VarPattern, etc),
   * lift the constant into the arguments of the partitioned function.
   */
  bool EmbedConst(const Expr& expr, const DFPattern pattern);
  // Internal State
  DFPattern pattern_;
  std::unordered_map<int, Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
  DFPatternMatcher* matcher_ = nullptr;
  std::unique_ptr<IndexedGraph<DFPattern>> pattern_graph_;
  int gid_ = 0;
  int graph_number_ = 0;
};

/*!
 * \brief PatternRewriter rewrites the expression by finding matches and allowing user callback
 * function to rewrite those matches
 *
 * The class uses PatternGrouper to support the dominator pattern.
 */
class PatternRewriter : protected MixedModeMutator {
 public:
  explicit PatternRewriter(IRModule mod) : mod_(mod) {}
  /*! \brief Rewrite can take a number of callbacks and will repeatedly rewrite the graph with the
   * callbacks until it stops changing */
  virtual Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre);

 protected:
  virtual Expr DispatchVisitExpr(const Expr& pre);

  IRModule mod_;
  DFPatternCallback callback_;
  std::unordered_map<int, PatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_IR_DATAFLOW_MATCHER_IMPL_H_
