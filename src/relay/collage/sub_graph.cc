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
 * \file src/relay/collage/sub_graph.cc
 * \brief Represents a sub-graph of an overall Relay expression.
 */

#include "./sub_graph.h"

#include <tvm/relay/transform.h>

#include "../../support/scalars.h"
#include "../transforms/pass_utils.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

namespace {

class Extractor;

/*!
 * \brief Helper class for rewriting expressions to replace a sub-graph according to the
 * given extractor.
 */
class Rewriter : public ExprMutator {
 public:
  explicit Rewriter(const Extractor* extractor) : extractor_(extractor) {}

  Expr VisitExpr(const Expr& expr) final;

 private:
  /*! \brief Already prepared extractor which will guide the rewrite. */
  const Extractor* extractor_;
};

/*! \brief Helper class for extracting matched sub-graphs from the overall expression. */
class Extractor : public ExprMutator {
 public:
  Extractor(const DataflowGraph* dataflow_graph, const SubGraphNode* sub_graph,
            FunctionAttrsMap opt_attrs)
      : dataflow_graph_(dataflow_graph), sub_graph_(sub_graph), opt_attrs_(std::move(opt_attrs)) {
    ICHECK_EQ(dataflow_graph_->size(), sub_graph_->overall_size());
  }

  const DataflowGraph& dataflow_graph() const { return *dataflow_graph_; }

  /*!
   * \brief Collect the parameters and output expressions for the function representing
   * the sub-graph.
   */
  void Extract() {
    ICHECK(!sub_graph_->IsEmpty());
    VLOG(2) << "Extracting " << sub_graph_->ToString();
    const bool for_function = opt_attrs_.defined();

    //  In reverse dataflow order...
    for (PostDfsIndex i = dataflow_graph_->size(); i > 0; --i) {
      PostDfsIndex index = i - 1;
      if (!sub_graph_->inside_[index]) {
        // Node is outside sub-graph.
        continue;
      }
      VLOG(2) << "index " << index;
      auto node = dataflow_graph_->index_to_node(index);
      if (sub_graph_->exit_[node->index_] || node->is_external_ || memo_.count(node->ref()) == 0) {
        // This sub-expression is:
        //  - inside the sub-graph and needed outside the sub-graph. So it must contribute to an
        //    output (even if we've already visited it while constructing an output from a
        //    downstream sub-expression).
        //  - not yet visited, in which case it must still be considered an 'output' so it will
        //    be evaluated for any possible side effects.
        Expr output = VisitExpr(GetRef<Expr>(node->node_ref_));
        VLOG(2) << "index " << index << " added as output:\n"
                << PrettyPrint(output) << "\nat " << outputs_.size();
        expr_to_output_index_.emplace(node->node_ref_, outputs_.size());
        outputs_.emplace_back(std::move(output));
        output_types_.emplace_back(node->node_ref_->checked_type());
      }
    }
    ICHECK(!outputs_.empty());

    // Reverse the outputs so as to preserve the original evaluation order.
    std::reverse(outputs_.begin(), outputs_.end());
    std::reverse(output_types_.begin(), output_types_.end());
    for (auto& kv : expr_to_output_index_) {
      kv.second = static_cast<int>(outputs_.size()) - 1 - kv.second;
    }

    // Build a 'body' expression to represent the extracted sub-graph. If we have multiple
    // outputs we'll place them in a tuple.
    Type body_type;
    Expr body;
    if (outputs_.size() > 1) {
      body_type = TupleType(output_types_);
      body = Tuple(outputs_);
      body->checked_type_ = body_type;
    } else {
      body_type = output_types_.front();
      body = outputs_.front();
    }

    // Re-express all the nested sub-graphs in terms of the body.
    DataflowGraph body_dataflow_graph(body);
    std::vector<NestedSubGraph> nested_sub_graphs;
    IndexSubst subst = MakeIndexSubst(body_dataflow_graph);
    for (const auto& nested_sub_graph : sub_graph_->nested_sub_graphs_) {
      nested_sub_graphs.emplace_back(nested_sub_graph.Subst(body_dataflow_graph, subst));
    }

    // Sweep backwards through the body, rewriting to account for each nested sub-graph.
    body = NestedSubGraph::ParallelRewrite(body_dataflow_graph, body, std::move(nested_sub_graphs));

    if (for_function) {
      // Rewrite so all input nodes are now conveyed via call arguments to a new function.
      Array<Type> arg_types;
      arg_types.reserve(params_.size());
      for (const auto& param : params_) {
        arg_types.push_back(param->checked_type());
      }
      extracted_ = Function(std::move(params_), std::move(body), body_type,
                            /*ty_params=*/{}, DictAttrs(opt_attrs_));
      extracted_->checked_type_ =
          FuncType(std::move(arg_types), body_type, /*type_params=*/{}, /*type_constraints=*/{});
      body = Call(extracted_, std::move(args_));
      body->checked_type_ = body_type;
    } else {
      // Don't do anything with the inputs.
      extracted_ = body;
    }

    // Setup the output substitution.
    for (const auto& kv : expr_to_output_index_) {
      Expr expr;
      if (outputs_.size() == 1) {
        expr = body;
      } else if (for_function) {
        expr = TupleGetItem(body, kv.second);
        expr->checked_type_ = output_types_[kv.second];
      } else {
        const auto* tuple_node = body.as<TupleNode>();
        ICHECK(tuple_node);
        expr = tuple_node->fields[kv.second];
      }
      VLOG(2) << "output " << dataflow_graph_->item_to_node(kv.first)->index_ << " is at index "
              << kv.second << " (of " << outputs_.size() << " outputs)";
      output_substitution_.emplace(kv.first, std::move(expr));
    }
  }

  ////// Following members are valid only after Extract() has returned.

  /*!
   * \brief Returns the expression representing the extracted sub-graph. If opt_attrs_ is
   * defined then will be a function.
   */
  Expr extracted() const { return extracted_; }

  /*!
   * \brief Returns the substitution to apply to all expression nodes in the overall expression
   * so as to replace references to outputs of the sub-graph with their rewritten form.
   */
  const std::unordered_map<const ExprNode*, Expr>& output_substitution() const {
    return output_substitution_;
  }

 private:
  /*!
   * \brief Returns a map from original index to new index for each node inside the sub-graph. Only
   * valid after \p Extract has made its backwards dataflow sweep.
   */
  IndexSubst MakeIndexSubst(const DataflowGraph& new_dataflow_graph) const {
    VLOG(2) << "building extractor substitution";
    IndexSubst subst;
    for (PostDfsIndex index : sub_graph_->inside_) {
      auto orig_node = dataflow_graph_->index_to_node(index);
      ICHECK_EQ(orig_node->index_, index);
      auto itr = memo_.find(orig_node->ref());
      ICHECK(itr != memo_.end());
      auto new_node = new_dataflow_graph.item_to_node(itr->second);
      VLOG(2) << orig_node->index_ << " |-> " << new_node->index_;
      subst.emplace(orig_node->index_, new_node->index_);
    }
    return subst;
  }

  /*! \brief Returns true if \p expr is inside the sub-graph. */
  bool inside(const Expr& expr) {
    return sub_graph_->inside_[dataflow_graph_->item_to_node(expr)->index_];
  }

  /*!
   * \brief Returns the variable uniquely representing \p expr, which should be
   * an input node (ie outside the sub-graph but feeding into a node inside the sub-graph).
   *
   * It is valid for:
   *  - An expression outside the sub-graph to be used multiple times inside the sub-graph.
   *  - An expression outside the sub-graph to be used both inside and outside the sub-graph.
   */
  Var VarFor(const Expr& expr) {
    ICHECK(!inside(expr));
    ICHECK(opt_attrs_.defined());
    auto itr = expr_to_param_.find(expr.get());
    if (itr != expr_to_param_.end()) {
      return itr->second;
    }
    auto fresh_var = Var("FunctionVar_" + std::to_string(params_.size()), expr->checked_type());
    fresh_var->checked_type_ = expr->checked_type();
    params_.push_back(fresh_var);
    args_.push_back(expr);
    expr_to_param_.emplace(expr.get(), fresh_var);
    return fresh_var;
  }

  /*!
   * \brief If \p expr is inside the sub-graph then return it's rewritten form.
   * If \p expr is outside the sub-graph then it must correspond to an input node.
   *  - If opt_attrs_ is defined return the variable to represent it.
   *  - Otherwise just return the expression directly.
   *
   * Should be called only on inputs to nodes which are inside the sub-graph.
   */
  Expr VisitExpr(const Expr& expr) final {
    if (inside(expr)) {
      return ExprMutator::VisitExpr(expr);
    } else if (CanInline(expr)) {
      // Implicitly include inlinable input sub-expressions.
      return expr;
    } else if (opt_attrs_.defined()) {
      // Map to a function parameter.
      return VarFor(expr);
    } else {
      // Stop rewriting.
      return expr;
    }
  }

  Expr VisitExpr_(const FunctionNode* function_node) override {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Function>(function_node);
    }
    return ExprMutator::VisitExpr_(function_node);
  }

  //// Context fields, passed in constructor.

  /*! \brief The dataflow graph corresponding to the overall expression. */
  const DataflowGraph* dataflow_graph_;
  /*! \brief The sub-graph of the above we are extracting. */
  const SubGraphNode* sub_graph_;
  /*! \brief Optional attributes if the sub-graph should be extracted as a function. */
  FunctionAttrsMap opt_attrs_;

  //// Result fields, available after Extract() called.

  /*!
   * \brief The extracted expression. If opt_attrs_ is defined this will be a function.
   */
  Expr extracted_;
  /*!
   * \brief Map from output nodes to corresponding expressions. If the sub-graph has more than
   * one exit node then each entry will be a tuple projection.
   */
  std::unordered_map<const ExprNode*, Expr> output_substitution_;

  //// Accumulator fields, built as we visit expressions.

  /*! \brief (If opt_attrs_ is defined) Parameters representing input expression nodes. */
  Array<Var> params_;
  /*!
   * \brief (If opt_attrs_ is defined) The input expression nodes for each of the above params_.
   */
  Array<Expr> args_;
  /*!
   * \brief (If opt_attrs_ is defined) Map from existing input expression nodes to the parameters
   * in params_ which now representing them.
   */
  std::unordered_map<const ExprNode*, Var> expr_to_param_;
  /*!
   * \brief Accumulated new expressions which represent the exit nodes of the rewritten sub-graph.
   * It is possible to have multiple outputs. It is possible one output also contributes to other
   * outputs (ie the output is a 'tap').
   */
  std::vector<Expr> outputs_;
  /*! \brief (If opt_attrs_ is defined) Types of original expressions corresponding to outputs_. */
  std::vector<Type> output_types_;
  /*!
   * \brief Map from existing exit expression nodes to the index in outputs_ which should
   * represent them in the rewritten overall expression.
   */
  std::unordered_map<const ExprNode*, int> expr_to_output_index_;
};

Expr Rewriter::VisitExpr(const Expr& expr) {
  auto itr = extractor_->output_substitution().find(expr.get());
  if (itr == extractor_->output_substitution().end()) {
    return ExprMutator::VisitExpr(expr);
  } else {
    return itr->second;
  }
}

}  // namespace

std::pair<OpPatternKind, std::string> SubExprKindAndLabel(const Expr& sub_expr) {
  class Visitor : public ExprFunctor<std::pair<OpPatternKind, std::string>(const Expr&)> {
   private:
    std::pair<OpPatternKind, std::string> VisitExpr_(const CallNode* call_node) final {
      if (auto optional = call_node->op.as<Op>()) {
        auto op = optional.value();
        static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
        if (fpattern.count(op) == 0) {
          VLOG(1) << "no TOpPattern known for " << op->name << ", considering opaque";
          return {kOpaque, op->name};
        } else if (IsDynamic(call_node->checked_type()) && IsDataDependent(call_node)) {
          VLOG(1) << "call has dynamic shape which is data-dependent, considering opaque";
          return {kOpaque, op->name};
        } else {
          OpPatternKind kind = static_cast<OpPatternKind>(fpattern[op]);
          VLOG(2) << "TOpPattern for " << op->name << " is " << KindToString(kind);
          return {kind, op->name};
        }
      } else if (const auto* function_node = call_node->op.as<FunctionNode>()) {
        Optional<Integer> opt_i =
            function_node->GetAttr<Integer>("TOpPattern", Optional<Integer>());
        if (opt_i.defined()) {
          OpPatternKind kind = static_cast<OpPatternKind>(opt_i.value()->value);
          VLOG(1) << "TOpPattern for function is " << KindToString(kind);
          return {kind, "call_prim"};
        } else {
          VLOG(1) << "calling function without TOpPattern, considering opaque";
          return {kOpaque, "call_fun"};
        }
      } else {
        VLOG(1) << "unsupported call, considering opaque";
        return {kOpaque, "call_any"};
      }
    }

    std::pair<OpPatternKind, std::string> VisitExpr_(const ConstantNode* constant_node) final {
      VLOG(2) << "TOpPattern for constant is " << KindToString(kElemWise);
      if (support::IsSimpleScalar(constant_node)) {
        return {kElemWise, "scalar"};
      } else {
        return {kElemWise, "const"};
      }
    }

    std::pair<OpPatternKind, std::string> VisitExpr_(const TupleNode* tuple_node) final {
      const auto* tuple_type_node = tuple_node->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type_node != nullptr);
      if (std::all_of(tuple_type_node->fields.begin(), tuple_type_node->fields.end(),
                      [](const Type& type) { return type.as<TensorTypeNode>() != nullptr; })) {
        VLOG(2) << "TOpPattern for tuple is " << KindToString(kInjective);
        return {kInjective, "tuple"};
      } else {
        VLOG(1) << "tuple contains non-tensors, considering opaque";
        return {kOpaque, "tuple"};
      }
    }

    std::pair<OpPatternKind, std::string> VisitExpr_(
        const TupleGetItemNode* tuple_get_item_node) final {
      const auto* tuple_type_node = tuple_get_item_node->tuple->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type_node != nullptr);
      if (std::all_of(tuple_type_node->fields.begin(), tuple_type_node->fields.end(),
                      [](const Type& type) { return type.as<TensorTypeNode>() != nullptr; })) {
        VLOG(2) << "TOpPattern for tuple projection is " << KindToString(kInjective);
        return {kInjective, "proj"};
      } else {
        VLOG(1) << "tuple being projected contains non-tensors, considering opaque";
        return {kOpaque, "proj"};
      }
    }

    // TODO(mbs): We implement the following mostly so we have a lightweight way of describing
    // the current sub-expression. If partitioning is ever extended beyond the usual call/tuple/proj
    // sub-language we should revise the returned operator kinds to match.

    std::pair<OpPatternKind, std::string> VisitExpr_(const VarNode* var_node) final {
      return {kOpaque, "%" + var_node->name_hint()};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const GlobalVarNode* global_var_node) final {
      return {kOpaque, "@" + global_var_node->name_hint};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const OpNode* op_node) final {
      return {kOpaque, "`" + op_node->name};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const FunctionNode* function_node) final {
      return {kOpaque, "fn"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const LetNode* let_node) final {
      return {kOpaque, "let"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const IfNode* if_node) final {
      return {kOpaque, "if"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const RefCreateNode* ref_create_node) final {
      return {kOpaque, "ref"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const RefReadNode* op) final {
      return {kOpaque, "ref_read"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const RefWriteNode* op) final {
      return {kOpaque, "ref_write"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const ConstructorNode* op) final {
      return {kOpaque, "`" + op->name_hint};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const MatchNode* op) final {
      return {kOpaque, "match"};
    }
  };
  return Visitor().VisitExpr(sub_expr);
}

std::pair<OpPatternKind, std::string> SubGraphKindAndLabel(const DataflowGraph& dataflow_graph,
                                                           const IndexSet& inside) {
  std::ostringstream os;
  bool first = true;
  OpPatternKind max_kind = kElemWise;
  for (PostDfsIndex index : inside) {
    auto [sub_kind, sub_label] = SubExprKindAndLabel(dataflow_graph.index_to_node(index)->ref());
    if (!sub_label.empty()) {
      if (first) {
        first = false;
      } else {
        os << "+";
      }
      os << sub_label;
    }
    max_kind = CombineKinds(max_kind, sub_kind);
  }
  return {max_kind, os.str()};
}

IndexSet MatcherToIndexSet(const DFPatternMatcher& matcher) {
  IndexSet result(matcher.size());
  for (const auto& kv : matcher.memo()) {
    for (const auto& matched_sub_expr : kv.second) {
      if (CanInline(matched_sub_expr)) {
        // Trivial sub-expressions can just be included in the extracted function body
        // when we construct it and don't need to be considered part of the sub-graph.
        continue;
      }
      if (kv.first.as<WildcardPatternNode>()) {
        // Don't consider the expressions matched by a wildcard to be part of the sub-graph.
        continue;
      }
      result.Add(matcher.expr_to_node(matched_sub_expr)->index_);
    }
  }
  return result;
}

std::string SubGraphConfig::ToString() const {
  std::ostringstream os;
  os << "{max_exits=" << max_exits;
  os << ", allow_taps=" << allow_taps;
  os << ", max_depth=" << max_depth;
  os << "}";
  return os.str();
}

TVM_REGISTER_NODE_TYPE(NestedSubGraphNode);

void NestedSubGraphNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

SubGraph NestedSubGraphNode::sub_graph() const { return Downcast<SubGraph>(sub_graph_obj_); }

bool NestedSubGraphNode::operator==(const NestedSubGraphNode& that) const {
  return *sub_graph().get() == *that.sub_graph().get();
}

bool NestedSubGraphNode::operator<(const NestedSubGraphNode& that) const {
  return *sub_graph().get() < *that.sub_graph().get();
}

size_t NestedSubGraphNode::hash() const {
  size_t h = StructuralHash()(attrs_);
  h ^= sub_graph()->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
  return h;
}

std::string NestedSubGraphNode::ToString() const {
  std::ostringstream os;
  os << "{sub_graph=" << sub_graph()->ToString();
  os << ", attrs=" << PrettyPrint(attrs_);
  os << "}";
  return os.str();
}

Function NestedSubGraphNode::Extract(const DataflowGraph& dataflow_graph) const {
  Extractor extractor(&dataflow_graph, sub_graph().get(), attrs_);
  extractor.Extract();
  return Downcast<Function>(extractor.extracted());
}

Expr NestedSubGraphNode::Rewrite(const DataflowGraph& dataflow_graph, const Expr& expr) const {
  Extractor extractor(&dataflow_graph, sub_graph().get(), attrs_);
  extractor.Extract();
  Rewriter rewriter(&extractor);
  return rewriter.VisitExpr(expr);
}

NestedSubGraph::NestedSubGraph(SubGraph sub_graph, FunctionAttrsMap attrs) {
  auto data = runtime::make_object<NestedSubGraphNode>();
  data->sub_graph_obj_ = std::move(sub_graph);
  data->attrs_ = std::move(attrs);
  data_ = std::move(data);
}

NestedSubGraph NestedSubGraph::Subst(
    const DataflowGraph& new_dataflow_graph,
    const std::unordered_map<PostDfsIndex, PostDfsIndex>& subst) const {
  return NestedSubGraph(get()->sub_graph().Subst(new_dataflow_graph, subst), get()->attrs_);
}

bool NestedSubGraph::TriviallyUnionable(const NestedSubGraph& that) const {
  if (get()->attrs_.size() != that->attrs_.size()) {
    return false;
  }
  for (const auto& kv : get()->attrs_) {
    if (kv.first == "Composite") {
      // Even if all the attributes agree we don't consider "Composite" functions to
      // ever be unionable.
      // TODO(mbs): Find a cleaner way to do this.
      return false;
    }
    auto itr = that->attrs_.find(kv.first);
    if (itr == that->attrs_.end()) {
      return false;
    }
    if (!StructuralEqual()(kv.second, (*itr).second)) {
      return false;
    }
  }
  return true;
}

NestedSubGraph NestedSubGraph::DisjointUnion(const DataflowGraph& dataflow_graph,
                                             const NestedSubGraph& that) const {
  ICHECK(TriviallyUnionable(that));
  return NestedSubGraph(get()->sub_graph().DisjointUnion(dataflow_graph, that->sub_graph()),
                        get()->attrs_);
}

/*static*/
Expr NestedSubGraph::ParallelRewrite(const DataflowGraph& dataflow_graph, const Expr& expr,
                                     std::vector<NestedSubGraph> nested_sub_graphs) {
  // IMPORTANT: See the corresponding comment in SubGraph::ParallelRewrite.
  std::sort(nested_sub_graphs.begin(), nested_sub_graphs.end(),
            [](const NestedSubGraph& left, const NestedSubGraph& right) {
              return left->sub_graph()->last_inside_index_ > right->sub_graph()->last_inside_index_;
            });

  Expr result = expr;
  for (const auto& nested_sub_graph : nested_sub_graphs) {
    result = nested_sub_graph->Rewrite(dataflow_graph, result);
  }
  return result;
}

TVM_REGISTER_NODE_TYPE(SubGraphNode);

void SubGraphNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

IndexSet SubGraphNode::Downstream(const DataflowGraph& dataflow_graph) const {
  IndexSet downstream(dataflow_graph.size());
  for (PostDfsIndex exit_index : exit_) {
    downstream = downstream | dataflow_graph.downstream_of(exit_index);
  }
  return downstream;
}

bool SubGraphNode::IsValid(const DataflowGraph& dataflow_graph,
                           const SubGraphConfig& config) const {
  // Check we don't have too many exit nodes.
  if (config.max_exits > 0 && exit_.PopCount() > config.max_exits) {
    VLOG(1) << "Subgraph " << ToString() << " is invalid: " << exit_.PopCount()
            << " exits exceeds maximum " << config.max_exits;
    return false;
  }

  // Check the maximum path depth is in limit.
  if (config.max_depth > 0 && depth_ > config.max_depth) {
    VLOG(1) << "Subgraph " << ToString() << " is invalid: maximum depth " << depth_
            << " exceeds limit " << config.max_depth;
    return false;
  }

  // All inside nodes must be in the same basic block.
  const DataflowGraph::Node* basic_block = nullptr;
  for (PostDfsIndex index : inside_) {
    auto node = dataflow_graph.index_to_node(index);
    if (basic_block == nullptr) {
      basic_block = node->basic_block_;
    }
    if (node->basic_block_ != basic_block) {
      VLOG(1) << "Subgraph " << ToString() << " is invalid: nodes are from different basic blocks";
      return false;
    }
  }

  // The nested sub-graphs must be subsets and non-overlapping.
  IndexSet union_inside(dataflow_graph.size());
  for (const auto& nested_sub_graph : nested_sub_graphs_) {
    if (!nested_sub_graph->sub_graph()->inside_.AreDisjoint(union_inside)) {
      VLOG(1) << "Subgraph " << ToString() << " is invalid: nested sub-graphs overlap";
      return false;
    }
    if (!nested_sub_graph->sub_graph()->inside_.IsSubset(inside_)) {
      VLOG(1) << "Subgraph " << ToString()
              << " is invalid: nested sub-graph is not subset of overall sub-graph";
      return false;
    }
  }

  if (!config.allow_taps) {
    // Exit nodes cannot also contribute to inside nodes.
    for (PostDfsIndex index : exit_) {
      auto node = dataflow_graph.index_to_node(index);
      if (AnyOutputInside(node)) {
        VLOG(1) << "Subgraph " << ToString()
                << " is invalid: inner node is 'tapped' and also contributes to output, but taps "
                   "are disabled";
        return false;
      }
    }
  }

  // Check no output would end up feeding into any entry node.
  for (PostDfsIndex output_index : output_) {
    if (dataflow_graph.downstream_of(output_index).Intersects(entry_)) {
      VLOG(1) << "Subgraph " << ToString() << " is invalid: output node " << output_index
              << " feeds back into this sub-graph";
      return false;
    }
  }

  // Looks legit!
  return true;
}

Function SubGraphNode::ExtractAsFunction(const DataflowGraph& dataflow_graph) const {
  NestedSubGraph nested_sub_graph(GetRef<SubGraph>(this), FunctionAttrsMap());
  return nested_sub_graph->Extract(dataflow_graph);
}

Expr SubGraphNode::Rewrite(const DataflowGraph& dataflow_graph, const Expr& expr) const {
  if (nested_sub_graphs_.empty()) {
    // Nothing to rewrite.
    return expr;
  }
  Extractor extractor(&dataflow_graph, this, NullValue<FunctionAttrsMap>());
  extractor.Extract();
  Rewriter rewriter(&extractor);
  return rewriter.VisitExpr(expr);
}

std::string SubGraphNode::ToString() const {
  std::ostringstream os;
  os << "{inside=" << inside_.ToString();
  os << ", entry=" << entry_.ToString();
  os << ", exit=" << exit_.ToString();
  os << ", input=" << input_.ToString();
  os << ", output=" << output_.ToString();
  os << ", depth=" << depth_;
  os << ", kind=" << KindToString(kind_);
  if (!label_.empty()) {
    os << ", label=" << label_;
  }
  for (const auto& nested_sub_graph : nested_sub_graphs_) {
    os << ", nested_sub_graph=" << nested_sub_graph->ToString();
  }
  os << "}";
  return os.str();
}

bool SubGraphNode::operator==(const SubGraphNode& that) const {
  ICHECK_EQ(inside_.end_index(), that.inside_.end_index());
  if (inside_ != that.inside_) {
    return false;
  }
  if (nested_sub_graphs_.size() != that.nested_sub_graphs_.size()) {
    return false;
  }
  for (size_t i = 0; i < nested_sub_graphs_.size(); ++i) {
    if (*nested_sub_graphs_[i].get() != *that.nested_sub_graphs_[i].get()) {
      return false;
    }
  }
  return true;
}

bool SubGraphNode::operator<(const SubGraphNode& that) const {
  if (first_inside_index_ < that.first_inside_index_) {
    return true;
  }
  if (that.first_inside_index_ < first_inside_index_) {
    return false;
  }
  return inside_ < that.inside_;
}

size_t SubGraphNode::hash() const {
  size_t h = inside_.hash();
  for (const auto& nested_sub_graph : nested_sub_graphs_) {
    h ^= nested_sub_graph->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
  }
  return h;
}

void SubGraphNode::Init(const DataflowGraph& dataflow_graph) {
  for (PostDfsIndex index = 0; index < inside_.end_index(); ++index) {
    auto node = dataflow_graph.index_to_node(index);
    if (inside_[index]) {
      if (AnyInputOutside(node)) {
        entry_.Add(index);
      }
      if (AnyOutputOutside(node) || node->is_external_) {
        exit_.Add(index);
      }
    } else {
      if (AnyInputInside(node)) {
        output_.Add(index);
      }
      if (AnyOutputInside(node) && !CanInline(node->ref())) {
        input_.Add(index);
      }
    }
  }
  depth_ = Depth(dataflow_graph);
}

size_t SubGraphNode::Depth(const DataflowGraph& dataflow_graph) const {
  std::unordered_map<const DataflowGraph::Node*, size_t> max_depths;
  std::vector<const DataflowGraph::Node*> stack;
  size_t max_depth = 0;
  // All the entry nodes have max depth 0.
  for (PostDfsIndex index : entry_) {
    auto node = dataflow_graph.index_to_node(index);
    max_depths.emplace(node, 0);
    stack.push_back(node);
  }
  while (!stack.empty()) {
    const DataflowGraph::Node* node = stack.back();
    stack.pop_back();
    size_t next_depth = max_depths[node] + 1;
    if (exit_[node->index_]) {
      // If this node is external then it will have no outputs but we still wish to consider
      // the path to the implied output as requiring one more step.
      // Otherwise we're accounting for reaching one of the external outputs belowe.
      max_depth = std::max(max_depth, next_depth);
    }
    for (const DataflowGraph::Node* output_node : node->outputs_) {
      if (!inside_[output_node->index_]) {
        continue;
      }
      if (max_depths.count(output_node) == 0) {
        max_depths.emplace(output_node, next_depth);
        stack.push_back(output_node);
      } else if (next_depth > max_depths[output_node]) {
        // We found a deeper path to an already expanded node. We'll expand again.
        max_depths[output_node] = next_depth;
        stack.push_back(output_node);
      }
    }
  }
  return max_depth;
}

/*! \brief Returns true if any (input/output) of node is (outside/inside) the sub-graph.  */
bool SubGraphNode::AnyInputOutside(const DataflowGraph::Node* node) const {
  return std::any_of(node->inputs_.begin(), node->inputs_.end(),
                     [this](const DataflowGraph::Node* sub_node) {
                       return !inside_[sub_node->index_] && !CanInline(sub_node->ref());
                     });
}

bool SubGraphNode::AnyInputInside(const DataflowGraph::Node* node) const {
  return std::any_of(
      node->inputs_.begin(), node->inputs_.end(),
      [this](const DataflowGraph::Node* sub_node) { return inside_[sub_node->index_]; });
}

bool SubGraphNode::AnyOutputOutside(const DataflowGraph::Node* node) const {
  return std::any_of(
      node->outputs_.begin(), node->outputs_.end(),
      [this](const DataflowGraph::Node* sub_node) { return !inside_[sub_node->index_]; });
}

bool SubGraphNode::AnyOutputInside(const DataflowGraph::Node* node) const {
  return std::any_of(
      node->outputs_.begin(), node->outputs_.end(),
      [this](const DataflowGraph::Node* sub_node) { return inside_[sub_node->index_]; });
}

SubGraph::SubGraph(const DataflowGraph& dataflow_graph, IndexSet inside, OpPatternKind kind,
                   String label, std::vector<NestedSubGraph> nested_sub_graphs) {
  std::sort(nested_sub_graphs.begin(), nested_sub_graphs.end(),
            [](const NestedSubGraph& left, const NestedSubGraph& right) {
              return *left.get() < *right.get();
            });
  auto node = runtime::make_object<SubGraphNode>();
  node->inside_ = std::move(inside);
  node->first_inside_index_ = node->inside_.FirstInsideIndex();
  node->last_inside_index_ = node->inside_.LastInsideIndex();
  node->entry_ = IndexSet(node->inside_.end_index());
  node->exit_ = IndexSet(node->inside_.end_index());
  node->input_ = IndexSet(node->inside_.end_index());
  node->output_ = IndexSet(node->inside_.end_index());
  node->kind_ = kind;
  node->label_ = std::move(label);
  node->nested_sub_graphs_ = nested_sub_graphs;
  node->Init(dataflow_graph);
  data_ = std::move(node);
}

SubGraph::SubGraph(const DataflowGraph& dataflow_graph)
    : SubGraph(dataflow_graph, IndexSet(dataflow_graph.size())) {}

bool SubGraph::AreDisjoint(const SubGraph& that) const {
  return get()->inside_.AreDisjoint(that->inside_);
}

namespace {
/*! \brief Returns true if an output of \p left not in \p right ultimately flows into \p right. */
bool FlowsInto(const DataflowGraph& dataflow_graph, const SubGraph& left, const SubGraph& right) {
  for (PostDfsIndex output_index : left->output_) {
    if (!right->inside_[output_index] &&
        dataflow_graph.downstream_of(output_index).Intersects(right->entry_)) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool SubGraph::AreTouching(const DataflowGraph& dataflow_graph, const SubGraph& that) const {
  if (!get()->inside_.AreDisjoint(that->inside_)) {
    // Easy rejection.
    return false;
  }
  if (!get()->output_.Intersects(that->entry_)) {
    // Not touching.
    return false;
  }
  if (FlowsInto(dataflow_graph, *this, that) || FlowsInto(dataflow_graph, that, *this)) {
    // Unioning would create a cycle.
    return false;
  }
  return true;
}

bool SubGraph::AreSelfContained(const SubGraph& that) const {
  return get()->output_.IsSubset(that->entry_) && that->input_.IsSubset(get()->exit_);
}

SubGraph SubGraph::DisjointUnion(const DataflowGraph& dataflow_graph, const SubGraph& that) const {
  ICHECK(AreDisjoint(that));
  IndexSet inside = get()->inside_ | that->inside_;
  std::vector<NestedSubGraph> nested_sub_graphs;
  for (const auto& nested_sub_graph : get()->nested_sub_graphs_) {
    nested_sub_graphs.push_back(nested_sub_graph);
  }
  for (const auto& nested_sub_graph : that->nested_sub_graphs_) {
    auto existing_itr = std::find_if(nested_sub_graphs.begin(), nested_sub_graphs.end(),
                                     [&nested_sub_graph](const NestedSubGraph& existing) {
                                       return existing.TriviallyUnionable(nested_sub_graph);
                                     });
    if (existing_itr != nested_sub_graphs.end()) {
      *existing_itr = existing_itr->DisjointUnion(dataflow_graph, nested_sub_graph);
    } else {
      nested_sub_graphs.push_back(nested_sub_graph);
    }
  }
  return SubGraph(dataflow_graph, std::move(inside), CombineKinds(get()->kind_, that->kind_),
                  UnionLabels(get()->label_, that->label_), std::move(nested_sub_graphs));
}

SubGraph SubGraph::WithAttrs(const DataflowGraph& dataflow_graph, FunctionAttrsMap attrs) const {
  std::vector<NestedSubGraph> nested_sub_graphs;
  nested_sub_graphs.push_back(NestedSubGraph(*this, attrs));
  return SubGraph(dataflow_graph, get()->inside_, get()->kind_, get()->label_,
                  std::move(nested_sub_graphs));
}

SubGraph SubGraph::Subst(const DataflowGraph& new_dataflow_graph, const IndexSubst& subst) const {
  IndexSet new_inside = get()->inside_.Subst(new_dataflow_graph.size(), subst);
  std::vector<NestedSubGraph> new_nested_sub_graphs;
  for (const auto& nested_sub_graph : get()->nested_sub_graphs_) {
    new_nested_sub_graphs.push_back(nested_sub_graph.Subst(new_dataflow_graph, subst));
  }
  return SubGraph(new_dataflow_graph, std::move(new_inside), get()->kind_, get()->label_,
                  std::move(new_nested_sub_graphs));
}

/*static*/
Expr SubGraph::ParallelRewrite(const DataflowGraph& dataflow_graph,
                               std::vector<SubGraph> sub_graphs) {
  // IMPORTANT:
  //  - All the sub-graphs will be w.r.t. the dataflow graph for the original expression.
  //    Each time we call Rewrite on one of those graphs the result expression will be rewritten
  //    from the final output back to the inputs. The inputs will then be shared with the original
  //    expression. Thus it is safe to iteratively rewrite all the sub-graphs without redoing the
  //    dataflow_graph and substituting indexes provided we work in reverse dataflow order.
  //  - We rely on the dataflow_graph expression reference holding the original expression alive
  //    so that the dataflow_graph will never contain dangling pointers (even though as per above
  //    we'll never dereference them).
  std::sort(sub_graphs.begin(), sub_graphs.end(), [](const SubGraph& left, const SubGraph& right) {
    return left->last_inside_index_ > right->last_inside_index_;
  });
  Expr result = dataflow_graph.expr();
  for (const auto& sub_graph : sub_graphs) {
    result = sub_graph->Rewrite(dataflow_graph, result);
  }
  return result;
}

/*!
 * \brief A pass which partitions (the unique) global function in the module according to the
 * post-dfs indexes in \p indexes. The partitioning must respect the configuration with \p max_exits
 * and \p allow_taps.
 *
 * Each index is also paired with a label. A non-empty label denotes the index should also be
 * included in a nested sub-graph which will be extracted as a function with the label as its
 * "Composite" attribute. An empty label denotes the index should go into the overall partitioned
 * "Compiler" function. In this way we can simulate the usual partitioning needed by external
 * codegen integrations.
 *
 * This function is intended to support \p SubGraph unit tests and is not used by the regular
 * compilation flow.
 */
transform::Pass PartitionForTesting(Integer max_exits, Bool allow_taps, String compiler,
                                    Array<Integer> indexes, Array<String> labels) {
  auto pass_func = [=](Function function, IRModule mod, transform::PassContext ctxt) {
    ICHECK(max_exits.defined() && max_exits->value >= 0);
    ICHECK(allow_taps.defined());
    ICHECK(indexes.size() == labels.size());
    VLOG(1) << "Partitioning:" << std::endl << PrettyPrint(function);
    DataflowGraph dataflow_graph(function);
    VLOG(1) << "Dataflow graph is:" << std::endl << dataflow_graph.indexed_graph().ToString();

    // Collect the 'inside' indexes and any nested sub-graph indexes and labels.
    std::vector<PostDfsIndex> node_indexes;
    std::unordered_map<String, std::vector<PostDfsIndex>> nested_sub_graph_indexes;
    node_indexes.reserve(indexes.size());
    for (size_t i = 0; i < indexes.size(); ++i) {
      const Integer& index = indexes[i];
      ICHECK_GE(index->value, 0);
      ICHECK_LT(index->value, dataflow_graph.size());
      auto index_int = static_cast<PostDfsIndex>(index->value);
      node_indexes.push_back(index_int);
      const String& label = labels[i];
      if (!label.empty()) {
        nested_sub_graph_indexes[label].push_back(index_int);
      }
    }

    // Build the nested sub-graphs representing the "Composite" functions (if any).
    std::vector<NestedSubGraph> nested_sub_graphs;
    for (const auto& kv : nested_sub_graph_indexes) {
      FunctionAttrsMap composite_attrs;
      composite_attrs.Set("Composite", kv.first);
      nested_sub_graphs.emplace_back(
          SubGraph(dataflow_graph, IndexSet(dataflow_graph.size(), kv.second)), composite_attrs);
    }

    // Build the overall sub-graph, which will include any "Composite" functions as
    // well as any nodes without a label.
    IndexSet inside(dataflow_graph.size(), node_indexes);
    auto [kind, label] = SubGraphKindAndLabel(dataflow_graph, inside);
    SubGraph sub_graph(dataflow_graph, inside, kind, label, std::move(nested_sub_graphs));

    // Push the overall sub-graph into the final "Compiler" function.
    FunctionAttrsMap compiler_attrs;
    compiler_attrs.Set("Compiler", compiler);
    NestedSubGraph overall_nested_sub_graph(sub_graph, compiler_attrs);
    SubGraph overall_sub_graph(dataflow_graph, inside, kind, label, {overall_nested_sub_graph});

    // Check the sub-graph is valid.
    SubGraphConfig config;
    config.max_exits = static_cast<size_t>(max_exits->value);
    config.allow_taps = allow_taps;
    if (overall_sub_graph->IsValid(dataflow_graph, config)) {
      VLOG(1) << "Sub-graph " << overall_sub_graph->ToString() << " is considered valid";
    } else {
      VLOG(1) << "Sub-graph " << overall_sub_graph->ToString()
              << " is NOT considered valid, not partitioning";
      return function;
    }

    // Do the partitioning.
    Function result = Downcast<Function>(overall_sub_graph->Rewrite(dataflow_graph, function));
    VLOG(1) << "Extracted as:" << std::endl << PrettyPrint(result);

    return result;
  };
  return transform::CreateFunctionPass(pass_func, /*opt_level=*/0, "PartitionForTesting", {});
}

TVM_REGISTER_GLOBAL("relay.collage.PartitionForTesting").set_body_typed(PartitionForTesting);

}  // namespace collage
}  // namespace relay
}  // namespace tvm
