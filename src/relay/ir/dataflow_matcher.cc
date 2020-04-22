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
 * \file src/tvm/relay/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relay.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <stack>

namespace tvm {
namespace relay {

// Pattern Matcher


class DominatorMatcher;

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  explicit DFPatternMatcher(const Expr& root_expr) : expr_graph_(CreateIndexedGraph(root_expr)) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Array<Expr>> GetMemo() { return Map<DFPattern, Array<Expr>>(memo_); }

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const AltPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  bool MatchesPath(const DominatorPatternNode* op, const Expr& expr);
  bool DominatesParent(const DominatorPatternNode* op, const Expr& expr);

  std::unordered_map<DFPattern, Array<Expr>, ObjectHash, ObjectEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
  IndexedGraph<Expr> expr_graph_;
  IndexedGraph<DFPattern> pattern_graph_;
  bool memoize_ = true;
};

bool DFPatternMatcher::Match(const DFPattern& pattern, const Expr& expr) {
  memo_.clear();
  matched_nodes_.clear();
  return VisitDFPattern(pattern, expr);
}

void DFPatternMatcher::ClearMap(size_t watermark) {
  for (size_t i = watermark; i < matched_nodes_.size(); ++i) {
    memo_.erase(matched_nodes_[i]);
  }
  matched_nodes_.erase(matched_nodes_.begin() + watermark, matched_nodes_.end());
}

bool DFPatternMatcher::VisitDFPattern(const DFPattern& pattern, const Expr& expr) {
  if (memoize_ && memo_.count(pattern)) {
    CHECK_EQ(memo_[pattern].size(), 1);
    return expr.same_as(memo_[pattern][0]);
  } else {
    auto watermark = matched_nodes_.size();
    auto out = DFPatternFunctor::VisitDFPattern(pattern, expr);
    if (out) {
      memo_[pattern].push_back(expr);
      matched_nodes_.push_back(pattern);
    } else {
      ClearMap(watermark);
    }
    return out;
  }
}

bool DFPatternMatcher::VisitDFPattern_(const AltPatternNode* op, const Expr& expr) {
  return VisitDFPattern(op->left, expr) || VisitDFPattern(op->right, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const AttrPatternNode* attr_pattern, const Expr& expr) {
  bool matches = false;
  if (const auto* op_node = expr.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    auto attributes = attr_pattern->attrs.as<DictAttrsNode>()->dict;
    for (auto kv : attributes) {
      auto attr_name = kv.first;
      auto attr_value = kv.second;
      auto op_map = Op::GetAttr<TVMRetValue>(attr_name);
      if (op_map.count(op)) {
        switch (op_map[op].type_code()) {
          case kDLInt:
            if (auto* val = kv.second.as<IntImmNode>()) {
              matches = val->value == op_map[op].operator int64_t();
            }
            break;
          case kDLFloat:
            if (auto* val = kv.second.as<FloatImmNode>()) {
              matches = val->value == op_map[op].operator double();
            }
            break;
          case kTVMStr:
            if (auto* val = kv.second.as<tir::StringImmNode>()) {
              matches = val->value == op_map[op].operator std::string();
            }
            break;
          default:
            throw "Unsupported type";
        }
      }
    }
  }
  return matches;
}

Array<DFPattern> reverse(const Array<DFPattern>& args) {
  Array<DFPattern> new_args;
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    new_args.push_back(*it);
  }
  return new_args;
}

bool DFPatternMatcher::VisitDFPattern_(const CallPatternNode* op, const Expr& expr) {
  // utilities
  auto get_op_node = [](const CallPatternNode* op) -> const tvm::OpNode* {
    if (op) {
      if (auto* expr_pattern = op->op.as<ExprPatternNode>()) {
        return expr_pattern->expr.as<OpNode>();
      }
    }
    return nullptr;
  };
  auto is_pattern_op = [&get_op_node](const CallPatternNode* op, std::string op_type) {
    if (const auto* op_node = get_op_node(op)) {
      if (op_node->name == op_type) {
        return true;
      }
    }
    return false;
  };
  auto is_expr_op = [](const Expr& expr, std::string op_type) {
    if (const auto* call_node = expr.as<CallNode>()) {
      if (const auto* op_node = call_node->op.as<OpNode>()) {
        if (op_node->name == op_type) {
          return true;
        }
      }
    }
    return false;
  };
  // logic
  auto watermark = matched_nodes_.size();
  if (const auto* call_node = expr.as<CallNode>()) {
    auto matches_op = VisitDFPattern(op->op, call_node->op);
    if (matches_op) {
      auto watermark2 = matched_nodes_.size();

      auto match_args = [this, &watermark2](const Array<DFPattern> pattern_args,
                                            const Array<Expr> expr_args) {
        bool matches = true;
        size_t i = 0;
        if (pattern_args.size() == expr_args.size()) {
          while (matches && i < pattern_args.size()) {
            matches &= VisitDFPattern(pattern_args[i], expr_args[i]);
            ++i;
          }
        } else {
          matches = false;
        }
        if (!matches) {
          ClearMap(watermark2);
        }
        return matches;
      };

      // Standard case
      if (match_args(op->args, call_node->args)) {
        return true;
      }
      // Commutative Matching
      if (const OpNode* op_node = get_op_node(op)) {
        if ((op_node->name == "add") || (op_node->name == "multiply")) {
          if (match_args(reverse(op->args), call_node->args)) {
            return true;
          }
        }
      }
    } else {
      ClearMap(watermark);
      // associate divide/multiply
      if (is_pattern_op(op, "divide")) {
        if (const auto* arg_node = op->args[0].as<CallPatternNode>()) {
          if (is_pattern_op(arg_node, "multiply") && is_expr_op(expr, "multiply") &&
                 (is_expr_op(call_node->args[0], "divide") ||
                  is_expr_op(call_node->args[1], "divide"))) {
            bool out = false;
            for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
              auto div = CallPatternNode::make(op->op, {arg_node->args[arg_id], op->args[1]},
                                               op->attrs, op->type_args);
              auto mul =
                  CallPatternNode::make(arg_node->op, {arg_node->args[(arg_id + 1) % 2], div},
                                        arg_node->attrs, arg_node->type_args);
              out = VisitDFPattern(mul, expr);
              if (out) {
                return true;
              } else {
                ClearMap(watermark);
              }
            }
            return out;
          }
        }
      }
      if (is_pattern_op(op, "multiply")) {
        // associate multiply/divide
        for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
          if (auto* arg_node = op->args[arg_id].as<CallPatternNode>()) {
            if (is_pattern_op(arg_node, "divide") && is_expr_op(expr, "divide") &&
                   (is_expr_op(call_node->args[0], "multiply") ||
                    is_expr_op(call_node->args[1], "multiply"))) {
              auto mul =
                  CallPatternNode::make(op->op, {arg_node->args[0], op->args[(arg_id + 1) % 2]},
                                        op->attrs, op->type_args);
              auto div = CallPatternNode::make(arg_node->op, {mul, arg_node->args[1]},
                                               arg_node->attrs, arg_node->type_args);
              return VisitDFPattern(div, expr);
            }
          }
        }
      }
    }
  }
  return false;
}

// Recursively find the Dominator parent along all inputs paths.
bool DFPatternMatcher::MatchesPath(const DominatorPatternNode* op, const Expr& expr) {
  bool out = true;
  auto call_node = expr.as<CallNode>();
  for (auto node : expr_graph_.node_map_[expr]->inputs_) {
    if (!(call_node && node->ref_ == call_node->op)) {
      memoize_ = true;
      if (VisitDFPattern(op->parent, node->ref_)) {
        return true;
      } else {
        memoize_ = false;
        if (VisitDFPattern(op->path, node->ref_)) {
          out &= MatchesPath(op, node->ref_);
        } else {
          return false;
        }
      }
    }
  }
  return out;
}

// Iteratively ensure that the parent is dominated somewhere by the child or the path
bool DFPatternMatcher::DominatesParent(const DominatorPatternNode* op, const Expr& expr) {
  std::stack<Expr> stack;
  std::unordered_set<Expr, ObjectHash, ObjectEqual> visited;
  stack.push(expr);
  while (!stack.empty()) {
    Expr current = stack.top();
    stack.pop();
    for (auto node : expr_graph_.node_map_[current]->dominator_children_) {
      if (visited.count(node->ref_) == 0) {
        if (VisitDFPattern(op->parent, node->ref_)) {
          return true;
        } else {
          stack.push(node->ref_);
        }
        visited.insert(node->ref_);
      }
    }
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) {
  pattern_graph_ = CreateIndexedGraph(GetRef<DFPattern>(op));
  if (VisitDFPattern(op->child, expr)) {
    bool matches_path = MatchesPath(op, expr);
    memoize_ = true;
    if (matches_path) {
      return DominatesParent(op, expr);
    }
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) {
  return StructuralEqual()(op->expr, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_get_item_node = expr.as<TupleGetItemNode>()) {
    matches = (op->index == tuple_get_item_node->index) &&
              VisitDFPattern(op->tuple, tuple_get_item_node->tuple);
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_node = expr.as<TupleNode>()) {
    if (op->fields.size() == tuple_node->fields.size()) {
      matches = true;
      size_t i = 0;
      while (matches && i < op->fields.size()) {
        matches &= VisitDFPattern(op->fields[i], tuple_node->fields[i]);
        ++i;
      }
    }
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  return (StructuralEqual()(op->type, expr_type)) && VisitDFPattern(op->pattern, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const VarPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* var_node = expr.as<VarNode>()) {
    matches = true;
    if (op->name_hint() != "") {
      matches &= op->name_hint() == var_node->name_hint();
    }
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) {
  return true;
}

TVM_REGISTER_GLOBAL("relay.df_pattern.match").set_body_typed([](DFPattern pattern, Expr expr) {
  return DFPatternMatcher(expr).Match(pattern, expr);
});

/* \brief PatternGrouper does pre-rewriting pattern matching and analysis
 *
 * This class creates a number of groups of matched expressions, ensures they don't overlap, and
 * returns them to the caller for post-analysis rewriting.
 *
 * This is primarly needed to suppor the post-dominator analysis required for dominator pattern
 * matching.
 */
class PatternGrouper : protected MixedModeVisitor {
 public:
  /* \brief Internal Group class for storing analysis */
  struct Group {
    Expr root_node;
    int gid;
    Map<DFPattern, Array<Expr>> matched_nodes;
    Function function;
    Array<Expr> args;
  };

  /* \brief Return the discovered groups */
  const std::vector<Group>& GetGroups() { return this->groups_; }

  /* \brief Return the group assingnments of expressions */
  const std::unordered_map<Expr, int, ObjectHash, ObjectEqual>& GetGIDAssignments() {
    return gid_assignments_;
  }
  /* \brief Group expressions that match the pattern */
  void GroupMatches(const DFPattern& pattern, const Expr& pre) {
    groups_ = {Group()};
    gid_assignments_.clear();
    visit_counter_.clear();

    pattern_ = pattern;
    pattern_graph_ = CreateIndexedGraph(pattern_);
    auto matcher = DFPatternMatcher(pre);
    matcher_ = &matcher;
    this->VisitExpr(pre);
  }

 protected:
  void VisitLeaf(const Expr& pre) override {
    if (matcher_->Match(pattern_, pre)) {
      CreateGroup(pre);
    }
  }

  /* \brief Creates a new set of nodes based on Group inputs, used to create functions and perform
   * group overlap analysis */
  class MatchExtractor : public ExprMutator {
   public:
    explicit MatchExtractor(const std::unordered_map<Expr, Var, ObjectHash, ObjectEqual>& inputs)
        : inputs_(inputs) {}
    const std::unordered_map<Expr, Expr, ObjectHash, ObjectEqual>& GetMemo() { return this->memo_; }

   protected:
    Expr VisitExpr(const Expr& pre) override {
      if (inputs_.count(pre)) {
        return inputs_.at(pre);
      }
      return ExprMutator::VisitExpr(pre);
    }
    const std::unordered_map<Expr, Var, ObjectHash, ObjectEqual> inputs_;
  };

  /* \brief Create a group based on a matched expression */
  void CreateGroup(const Expr& expr) {
    var_number_ = 0;

    auto node_map = matcher_->GetMemo();

    // Get fuzzy patterns
    std::unordered_set<Expr, ObjectHash, ObjectEqual> fuzzy_matches;
    for (auto node : pattern_graph_.topological_order_) {
      if (auto op = node->ref_.as<DominatorPatternNode>()) {
        for (auto fuzzy_op : {op->parent, op->path}) {
          for (auto match : node_map[fuzzy_op]) {
            fuzzy_matches.insert(match);
          }
        }
      }
    }

    // Create input variables
    Group group;
    group.root_node = expr;
    group.matched_nodes = node_map;

    std::unordered_map<Expr, Var, ObjectHash, ObjectEqual> inputs;
    Array<Var> params;
    for (auto node : pattern_graph_.topological_order_) {
      if (node->inputs_.size() == 0) {
        if (node_map.count(node->ref_)) {
          auto matches = node_map[node->ref_];
          for (auto match : matches) {
            if (fuzzy_matches.count(match) == 0 && match.as<OpNode>() == nullptr &&
                match.as<FunctionNode>() == nullptr && match.as<ConstantNode>() == nullptr) {
              inputs[match] = Var("FunctionVar_" + std::to_string(graph_number_) + "_" +
                                      std::to_string(var_number_),
                                  NullValue<Type>());
              group.args.push_back(match);
              params.push_back(inputs[match]);
              var_number_++;
            }
          }
        }
      }
    }

    graph_number_++;

    // Extract a Function. Used in Parition directly,
    // used to determine Group overlap in other passes
    auto extractor = MatchExtractor(inputs);
    auto body = extractor.Mutate(expr);

    // Verify the pattern still holds
    CHECK(DFPatternMatcher(body).Match(pattern_, body));
    group.function = Function(params, body, NullValue<Type>(), Array<TypeVar>());

    // Check to make sure we aren't overlapping with another group
    for (auto kv : extractor.GetMemo()) {
      if (gid_assignments_.count(kv.first) != 0 && inputs.count(kv.first) == 0 &&
          kv.first.as<OpNode>() == nullptr && kv.first.as<FunctionNode>() == nullptr &&
          kv.first.as<ConstantNode>() == nullptr) {
        // Exit due to overlapping partitions
        return;
      }
    }
    // Assign Group Ids
    group.gid = ++gid_;
    for (auto kv : extractor.GetMemo()) {
      gid_assignments_[kv.first] = gid_;
    }

    // Save Group
    groups_.emplace_back(std::move(group));
    CHECK_EQ(groups_[gid_].gid, gid_);
  }

  // Internal State
  DFPattern pattern_;
  std::vector<Group> groups_;
  std::unordered_map<Expr, int, ObjectHash, ObjectEqual> gid_assignments_;
  DFPatternMatcher* matcher_ = nullptr;
  IndexedGraph<DFPattern> pattern_graph_;
  int gid_ = 0;
  int var_number_ = 0;
  int graph_number_ = 0;
};

// Rewrite

DFPatternCallback DFPatternCallbackNode::make(DFPattern pattern, PackedFunc function) {
  ObjectPtr<DFPatternCallbackNode> n = make_object<DFPatternCallbackNode>();
  n->pattern_ = std::move(pattern);
  n->function_ = std::move(function);
  return DFPatternCallback(n);
}

TVM_REGISTER_NODE_TYPE(DFPatternCallbackNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.DFPatternCallback")
    .set_body_typed(DFPatternCallbackNode::make);

/* \brief PatternRewriter rewrites the expression by finding matches and allowing user callback
 * function to rewrtie those matches
 *
 * The class uses PatternGrouper to support the dominator pattern.
 */
class PatternRewriter : protected MixedModeMutator {
 public:
  PatternRewriter() {}
  /*! \brief Rewrite can take a number of callbakcs and will repeatedly rewrite the graph with the
   * callbacks until it stops changing */
  Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) {
    auto post = pre;
    auto last = post;
    // rewrite the graph until it stops changing to make sure all rewrites are complete
    do {
      last = post;
      for (auto callback : callbacks) {
        callback_ = callback;
        auto grouper = PatternGrouper();
        grouper.GroupMatches(callback_->pattern_, post);
        groups_ = grouper.GetGroups();
        gid_assignments_ = grouper.GetGIDAssignments();
        memo_.clear();
        post = this->VisitExpr(post);
      }
    } while (last != post);
    return post;
  }

 protected:
  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    if (gid_assignments_.count(pre) && pre == groups_[gid_assignments_[pre]].root_node) {
      // Convert the pre-rewrite node map to a post-rewrite node map
      auto group = groups_[gid_assignments_[pre]];
      std::unordered_map<DFPattern, Array<Expr>, ObjectHash, ObjectEqual> node_map;
      for (auto kv : group.matched_nodes) {
        Array<Expr> tmp;
        for (size_t i = 0; i < kv.second.size(); ++i) {
          tmp.push_back(this->memo_[kv.second[i]]);
        }
        node_map.insert({kv.first, tmp});
      }
      // run the user callback function
      return callback_->function_(pre, post, Map<DFPattern, Array<Expr>>(node_map));
    }
    return post;
  }

  DFPatternCallback callback_;
  std::vector<PatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectHash, ObjectEqual> gid_assignments_;
};

Expr RewritePatterns(Array<DFPatternCallback> callbacks, Expr expr) {
  return PatternRewriter().Rewrite(callbacks, expr);
}

TVM_REGISTER_GLOBAL("relay.df_pattern.rewrite").set_body_typed(RewritePatterns);

/* \brief PatternParitioner replaces expressions that match a pattern with function call that
 * perform the same computation but allow for further analysis and lowering.
 *
 * The class uses PatternGrouper to support the dominator pattern.
 */
class PatternPartitioner : protected MixedModeMutator {
 public:
  Expr Partition(const DFPattern& pattern, const Expr& pre) {
    auto grouper = PatternGrouper();
    grouper.GroupMatches(pattern, pre);
    groups_ = grouper.GetGroups();
    gid_assignments_ = grouper.GetGIDAssignments();
    return this->VisitExpr(pre);
  }

 protected:
  Expr RewriteParition(const PatternGrouper::Group& group) {
    Array<Expr> args;
    for (size_t i = 0; i < group.args.size(); ++i) {
      args.push_back(memo_[group.args[i]]);
    }
    return Call(group.function, args);
  }

  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    if (gid_assignments_.count(pre) && pre == groups_[gid_assignments_[pre]].root_node) {
      post = RewriteParition(groups_[gid_assignments_[pre]]);
    }
    return post;
  }

  std::vector<PatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectHash, ObjectEqual> gid_assignments_;
};

Expr PartitionPattern(DFPattern pattern, Expr expr) {
  return PatternPartitioner().Partition(pattern, expr);
}

TVM_REGISTER_GLOBAL("relay.df_pattern.partition").set_body_typed(PartitionPattern);

}  // namespace relay
}  // namespace tvm
