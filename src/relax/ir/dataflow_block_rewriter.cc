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
 * \file src/relax/ir/dataflow_block_rewriter.cc
 * \brief A transform to match a Relax DataflowBlock and rewrite
 */

#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dataflow_matcher.h"

namespace tvm {
namespace relax {

class MatcherUseDefAnalysis : public relax::ExprVisitor {
 public:
  std::vector<const VarNode*> vars;
  std::map<const VarNode*, std::vector<const VarNode*>> def2use;
  // caller -> callee table.
  std::map<const VarNode*, std::vector<const VarNode*>> caller2callees;

  const VarNode* cur_user_;

  void VisitBinding_(const VarBindingNode* binding) override {
    // init
    cur_user_ = binding->var.get();
    this->VisitVarDef(binding->var);
    this->VisitExpr(binding->value);
    cur_user_ = nullptr;
  }

  void VisitExpr_(const VarNode* op) override {
    if (nullptr == cur_user_) return;

    auto check_and_push = [](std::vector<const VarNode*>& vec, const VarNode* var) {
      if (std::find(vec.begin(), vec.end(), var) == vec.end()) {
        vec.push_back(var);
      }
    };

    check_and_push(def2use[op], cur_user_);
    check_and_push(vars, op);

    caller2callees[cur_user_].push_back(op);
  }
};

struct PNode {
  const DFPatternNode* ptr;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> children;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> parents;
};

struct RNode {
  const VarNode* ptr;
  std::vector<RNode*> children;
  std::vector<RNode*> parents;
};

struct MatchState {
  void add(const PNode* p, const RNode* r) {
    match_p_r[p] = r;
    match_r_p[r] = p;
  }

  void add(const DFConstraintNode* constraint) { validated_constraints_.insert(constraint); }

  void add(MatchState&& other) {
    match_p_r.merge(std::move(other.match_p_r));
    match_r_p.merge(std::move(other.match_r_p));
    validated_constraints_.merge(other.validated_constraints_);
  }

  const VarNode* matched(const PNode* p) const {
    if (auto it = match_p_r.find(p); it != match_p_r.end()) {
      return it->second->ptr;
    }
    return nullptr;
  }

  const DFPatternNode* matched(const RNode* r) const {
    if (auto it = match_r_p.find(r); it != match_r_p.end()) {
      return it->second->ptr;
    }
    return nullptr;
  }

  const VarNode* matched(const PNode& p) const { return matched(&p); }
  const DFPatternNode* matched(const RNode& r) const { return matched(&r); }

  bool is_validated(const DFConstraintNode* constraint) const {
    return validated_constraints_.count(constraint);
  }

 private:
  std::unordered_map<const PNode*, const RNode*> match_p_r;
  std::unordered_map<const RNode*, const PNode*> match_r_p;
  std::unordered_set<const DFConstraintNode*> validated_constraints_;
};

/**
 * \brief This method try to match a real node and a pattern node along with its neighbors.
 */
static std::optional<MatchState> TryMatch(const PNode& p, const RNode& r,
                                          const MatchState& current_match, DFPatternMatcher* m,
                                          const MatcherUseDefAnalysis& ud_analysis) {
  if (!m->Match(GetRef<DFPattern>(p.ptr), GetRef<Var>(r.ptr))) return std::nullopt;

  MatchState new_match;

  new_match.add(&p, &r);

  // forward matching;
  for (const auto& [pchild, constraints] : p.children) {
    bool any_cons_sat = false;
    for (const auto& rchild : r.children) {
      if (new_match.matched(rchild)) {
        // The child variable is already matched to other child pattern in a previous iteration.
        continue;
      }
      if (auto v = current_match.matched(pchild); v && v != rchild->ptr) {
        // The child pattern is already matched to other variable in a earlier call to TryMatch.
        continue;
      }

      const auto& uses = ud_analysis.def2use.at(r.ptr);

      // check edge constraints.
      bool all_cons_pass = true;
      for (const auto& cons : constraints) {
        if (cons.type == PairCons::kOnlyUsedBy && uses.size() != 1) {
          all_cons_pass = false;
          break;
        }

        if (cons.index != -1) {
          const auto& callees = ud_analysis.caller2callees.at(rchild->ptr);
          if (callees.size() <= static_cast<size_t>(cons.index) || callees[cons.index] != r.ptr) {
            all_cons_pass = false;
            break;
          }
        }
      }
      if (!all_cons_pass || new_match.matched(pchild)) continue;
      any_cons_sat = true;

      if (auto match_rec = TryMatch(*pchild, *rchild, current_match, m, ud_analysis)) {
        new_match.add(pchild, rchild);
        new_match.add(std::move(*match_rec));
      }
    }
    if (!new_match.matched(pchild) || !any_cons_sat) return std::nullopt;
  }

  return new_match;
}

static std::optional<MatchState> TryValidate(
    const MatchState& current_match,
    const std::unordered_map<const DFPatternNode*, PNode>& pattern2node,
    const std::vector<DFConstraint>& validation_constraints, arith::Analyzer* analyzer) {
  MatchState new_match;

  std::function<Optional<Var>(const DFPatternNode*)> query_match_state =
      [&pattern2node, &current_match](const DFPatternNode* pattern) -> Optional<Var> {
    auto it = pattern2node.find(pattern);
    ICHECK(it != pattern2node.end())
        << "DFConstraint attempted to access DFPattern " << GetRef<DFPattern>(pattern)
        << ", which does not appear in the PatternContext";
    const auto& p_node = it->second;
    if (auto ptr = current_match.matched(p_node)) {
      return GetRef<Var>(ptr);
    } else {
      return NullOpt;
    }
  };

  for (const auto& constraint : validation_constraints) {
    if (!current_match.is_validated(constraint.get())) {
      auto [necessary_condition, is_sufficient] = constraint->AsPrimExpr(query_match_state);

      necessary_condition = analyzer->Simplify(necessary_condition);
      const auto* known = tir::as_const_int(necessary_condition);

      if (known && *known && is_sufficient) {
        // The condition passes, and the expression provided is both
        // necessary and sufficient for the constraint to pass.  Mark
        // the constraint as passing, to avoid re-checking it unless
        // we backtrack.
        new_match.add(constraint.get());
      } else if (known && !*known) {
        // The condition fails.  Even if additional information would
        // be required to pass a constraint, it may bail out early as
        // a failure (e.g. shape mismatch in the first two items out
        // of N shapes that must all match).
        return std::nullopt;
      } else if (is_sufficient) {
        // The condition depends on dynamic parameters.  In the
        // future, this may be exposed to the user as a condition for
        // optimization, or can be combined with the conditions
        // provided from other constraints.
        return std::nullopt;
      }
    }
  }

  return new_match;
}

static std::optional<MatchState> MatchTree(
    const MatchState& current_match, size_t current_root_idx,
    const std::unordered_map<const DFPatternNode*, PNode>& pattern2node,
    const std::unordered_map<const VarNode*, RNode>& var2node, DFPatternMatcher* matcher,
    const std::vector<DFPattern>& roots, const std::vector<DFConstraint>& validation_constraints,
    const MatcherUseDefAnalysis& ud_analysis, arith::Analyzer* analyzer) {
  auto get_next_root = [&](size_t root_idx) -> const PNode* {
    // Look for the next unmatched root node.
    for (; root_idx < roots.size(); ++root_idx) {
      const auto& root = pattern2node.at(roots[root_idx].get());
      if (!current_match.matched(root)) {
        return &root;
      }
    }
    return nullptr;
  };

  const auto root = get_next_root(current_root_idx);

  if (!root) {
    // All root nodes have been matched
    return current_match;
  }

  MatchState new_match = current_match;

  for (const auto& var : ud_analysis.vars) {
    const RNode& r_node = var2node.at(var);
    if (new_match.matched(r_node)) continue;
    if (auto match = TryMatch(*root, r_node, new_match, matcher, ud_analysis)) {
      // Recursively try to match the next subtree.
      new_match.add(std::move(*match));
      if (auto validation =
              TryValidate(new_match, pattern2node, validation_constraints, analyzer)) {
        new_match.add(std::move(*validation));
        if (auto match_rec =
                MatchTree(new_match, current_root_idx + 1, pattern2node, var2node, matcher, roots,
                          validation_constraints, ud_analysis, analyzer)) {
          new_match.add(std::move(*match_rec));
          return new_match;
        }
      }
      // Recursive matching has failed, backtrack.
      new_match = current_match;
      continue;
    }
  }

  return std::nullopt;
}

Optional<Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx, const DataflowBlock& dfb,
                                         const Map<Var, Expr>& bindings) {
  // TODO(@ganler): Handle non-may external use.
  ICHECK(ctx->allow_extern_use == PatternContextNode::kMay) << "Only kMay is supported yet.";
  DFPatternMatcher matcher(bindings);

  MatcherUseDefAnalysis ud_analysis;
  ud_analysis.VisitBindingBlock_(dfb.get());

  // First construct a graph of PNode and RNode.
  std::unordered_map<const VarNode*, RNode> var2node;
  var2node.reserve(dfb->bindings.size());

  for (const VarNode* cur_var : ud_analysis.vars) {
    const auto& uses = ud_analysis.def2use.at(cur_var);
    RNode& cur_node = var2node[cur_var];
    cur_node.ptr = cur_var;
    for (const VarNode* use : uses) {
      auto& use_node = var2node[use];
      use_node.ptr = use;
      cur_node.children.push_back(&use_node);
      use_node.parents.push_back(&cur_node);
    }
  }

  std::unordered_map<const DFPatternNode*, PNode> pattern2node;
  pattern2node.reserve(ctx->edge_constraints.size());

  for (const auto& def_pattern : ctx->src_ordered) {
    PNode& def_node = pattern2node[def_pattern.get()];
    const auto& uses = ctx->edge_constraints.at(def_pattern);
    def_node.ptr = def_pattern.get();
    def_node.children.reserve(uses.size());
    for (const auto& [use_pattern, cons] : uses) {
      PNode& use_node = pattern2node[use_pattern.get()];
      use_node.ptr = use_pattern.get();
      use_node.parents.emplace_back(&def_node, std::ref(cons));
      def_node.children.emplace_back(&use_node, std::ref(cons));
    }
  }

  std::vector<DFPattern> roots;
  for (const auto& pat : ctx->src_ordered) {
    if (pattern2node[pat.get()].parents.empty()) {
      roots.push_back(pat);
    }
  }

  if (roots.empty()) {
    return NullOpt;
  }

  arith::Analyzer analyzer;
  auto match = MatchTree({}, 0, pattern2node, var2node, &matcher, roots,
                         ctx->validation_constraints, ud_analysis, &analyzer);
  if (!match) {
    return NullOpt;
  }

  Map<DFPattern, Var> ret;
  for (const auto& [pat, p_node] : pattern2node) {
    ICHECK(match->matched(p_node));
    ret.Set(GetRef<DFPattern>(pat), GetRef<Var>(match->matched(p_node)));
  }
  return ret;
}

Optional<Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx, const DataflowBlock& dfb) {
  return MatchGraph(ctx, dfb, AnalyzeVar2Value(dfb));
}

TVM_REGISTER_GLOBAL("relax.dpl.match_dfb")
    .set_body_typed([](const PatternContext& ctx, const DataflowBlock& dfb) {
      return MatchGraph(ctx, dfb);
    });

/*!
 * \brief Apply pattern matching to each dataflow block, replacing matches
 * with the output of a user-provided rewriter function.
 */
class BlockPatternRewriter : ExprMutator {
 public:
  using ExprMutator::VisitBindingBlock_;
  using ExprMutator::VisitExpr_;

  BlockPatternRewriter(
      const PatternContext& ctx,
      TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>, Map<Var, Expr>)> rewriter_func)
      : ctx_(ctx), rewriter_func_(rewriter_func) {}

  template <typename PatternType>
  static Function Run(
      PatternType pat,
      TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>, Map<Var, Expr>)> rewriter_func,
      Function func) {
    BlockPatternRewriter rewriter(pat, rewriter_func);

    func = Downcast<Function>(rewriter(func));
    func = Downcast<Function>(RemoveAllUnused(func));
    return func;
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block_node) override {
    return RewriteDataflowBlockFixedPoint(GetRef<DataflowBlock>(block_node));
  }

 private:
  void EmitUsedVars(Expr val, const Array<Binding>& pending_bindings,
                    std::unordered_set<const VarNode*>* emitted_vars) {
    std::unordered_set<const VarNode*> unemitted_vars;
    PostOrderVisit(val, [=, &unemitted_vars](Expr e) {
      if (auto v = e.as<VarNode>(); v && !emitted_vars->count(v)) {
        unemitted_vars.insert(v);
      }
    });

    if (unemitted_vars.empty()) {
      return;
    }

    size_t num_unemitted = unemitted_vars.size();
    for (size_t i = 0; i < pending_bindings.size(); ++i) {
      const auto& binding = pending_bindings[i];
      if (auto var_bind = binding.as<VarBindingNode>();
          var_bind && unemitted_vars.count(var_bind->var.get())) {
        // var_bind->value may also depend on other unemitted vars in this range
        Array<Binding> prev_bindings(pending_bindings.begin(), pending_bindings.begin() + i);
        EmitUsedVars(var_bind->value, prev_bindings, emitted_vars);
        this->VisitBinding(binding);
        emitted_vars->insert(var_bind->var.get());
        if (--num_unemitted == 0) {
          return;
        }
      }
    }
  }

  // Repeat until all matchable subsets of bindings are rewritten.
  BindingBlock RewriteDataflowBlockFixedPoint(BindingBlock block) {
    auto df_block = Downcast<DataflowBlock>(block);
    Map<Var, Expr> bindings = AnalyzeVar2Value(df_block);
    if (auto matches = MatchGraph(ctx_, df_block, bindings)) {
      builder_->BeginDataflowBlock();
      Map<Var, Expr> replacements = rewriter_func_(matches.value(), bindings);

      std::unordered_set<const VarNode*> emitted_vars;

      bool changed = false;
      for (size_t i = 0; i < block->bindings.size(); ++i) {
        const auto& binding = block->bindings[i];
        if (auto var_bind = binding.as<VarBindingNode>()) {
          if (auto new_val = replacements.Get(var_bind->var).value_or(var_bind->value);
              !StructuralEqual()(var_bind->value, new_val)) {
            Array<Binding> pending_bindings(block->bindings.begin() + i + 1, block->bindings.end());
            // Make sure there is no unbound variable used in the new value before it is emitted
            EmitUsedVars(new_val, pending_bindings, &emitted_vars);
            this->ReEmitBinding(var_bind, builder_->Normalize(new_val));
            changed = true;
          } else if (!emitted_vars.count(var_bind->var.get())) {
            this->VisitBinding(binding);
            emitted_vars.insert(var_bind->var.get());
          }
        } else {
          this->VisitBinding(binding);
        }
      }

      auto new_block = builder_->EndBlock();

      if (!changed) return new_block;
      return RewriteDataflowBlockFixedPoint(new_block);
    }
    return block;
  }

  /*! \brief The pattern constraint contexts for rewriting dataflow blocks */
  PatternContext ctx_;
  /*!
   * \brief The user-provided rewriter function. Its signature and semantics are:
   *
   * - (Map<DFPattern, Var>, Map<Var, Expr>) -> Map<Var, Expr>
   *
   *    Given the map of patterns and corresponding variables (bound
   *    variables or parameters), it should return a map that
   *    specifies new values for matched bound variables. It can refer
   *    to the passed bindings to create the replacement expressions.
   */
  TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>, Map<Var, Expr>)> rewriter_func_;
};

Function RewriteBindings(
    const PatternContext& ctx,
    TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>, Map<Var, Expr>)> rewriter, Function func) {
  return BlockPatternRewriter::Run(ctx, rewriter, func);
}

TVM_REGISTER_GLOBAL("relax.dpl.rewrite_bindings").set_body_typed(RewriteBindings);

}  // namespace relax
}  // namespace tvm
