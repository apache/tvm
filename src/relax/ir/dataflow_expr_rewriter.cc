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
 * \file src/relax/ir/dataflow_expr_rewriter.cc
 * \brief A transform to match a Relax Expr and rewrite
 */

#include <tvm/node/structural_equal.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

#include "../transform/utils.h"
#include "dataflow_matcher.h"

namespace tvm {
namespace relax {

Optional<Map<DFPattern, Expr>> ExtractMatchedExpr(DFPattern pattern, Expr expr,
                                                  Optional<Map<Var, Expr>> bindings_opt) {
  auto bindings = bindings_opt.value_or({});
  DFPatternMatcher matcher(bindings);

  if (!matcher.Match(pattern, expr)) {
    return NullOpt;
  }

  Map<DFPattern, Expr> matching;
  for (const auto& [pat, matches] : matcher.GetMemo()) {
    ICHECK_EQ(matches.size(), 1) << "More than one match for the pattern " << pat;
    matching.Set(pat, matches[0]);
  }
  return matching;
}

TVM_REGISTER_GLOBAL("relax.dpl.extract_matched_expr").set_body_typed(ExtractMatchedExpr);

bool MatchExpr(DFPattern pattern, Expr expr, Optional<Map<Var, Expr>> bindings_opt) {
  return static_cast<bool>(ExtractMatchedExpr(pattern, expr, bindings_opt));
}

TVM_REGISTER_GLOBAL("relax.dpl.match_expr").set_body_typed(MatchExpr);

/*!
 * \brief Apply pattern matching to each expression, replacing
 * matches with the output of a user-provided rewriter function.
 */
class ExprPatternRewriter : ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  ExprPatternRewriter(DFPattern pat,
                      TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)> rewriter_func)
      : pattern_(pat), rewriter_func_(rewriter_func) {}

  template <typename PatternType>
  static Function Run(PatternType pat,
                      TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)> rewriter_func,
                      Function func) {
    ExprPatternRewriter rewriter(pat, rewriter_func);
    func = Downcast<Function>(rewriter(func));
    func = Downcast<Function>(RemoveAllUnused(func));
    return func;
  }

  Expr VisitExpr_(const SeqExprNode* seq) override {
    auto cache = bindings_;
    SeqExpr prev = GetRef<SeqExpr>(seq);

    StructuralEqual struct_equal;

    while (true) {
      SeqExpr next = Downcast<SeqExpr>(builder_->Normalize(ExprMutator::VisitExpr_(prev.get())));
      if (struct_equal(prev, next)) {
        return std::move(next);
      }

      // Canonicalization may result in two previously-different
      // expressions being recognized as identical.  Elimination of
      // common subexpressions may result in trival var-to-var
      // bindings that can be canonicalized.  Therefore, iterate the
      // simplification steps until converged.
      while (true) {
        auto start_of_loop = next;
        next = Downcast<SeqExpr>(CanonicalizeBindings(next));
        next = Downcast<SeqExpr>(EliminateCommonSubexpr(next));
        next = Downcast<SeqExpr>(RemoveAllUnused(next));
        if (struct_equal(start_of_loop, next)) {
          break;
        }
      }

      if (struct_equal(prev, next)) {
        return std::move(next);
      }

      // Reset all knowledge of bindings that were collected from
      // this SeqExpr.  The collected bindings are only after
      // the point where they were collected, and we are repeating
      // the mutation of this SeqExpr.
      bindings_ = cache;
      prev = next;
    }
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    auto expr = VisitExpr(binding->value);
    bindings_.Set(binding->var, expr);
    ReEmitBinding(binding, expr);
  }

  Expr VisitExpr(const Expr& expr) override {
    auto node = ExprMutator::VisitExpr(expr);

    std::vector<DFPattern> matches_top_level;
    if (auto rewritten = TryRewrite(node, pattern_, &matches_top_level)) {
      return builder_->Normalize(rewritten.value());
    }

    return node;
  }

 private:
  Optional<Expr> TryRewrite(const Expr& expr, const DFPattern& pattern,
                            std::vector<DFPattern>* matches_top_level) {
    ICHECK(matches_top_level);

    // Special handling if the user-supplied pattern is a `OrPattern`.
    // While the `ExtractMatchedExpr` can handle matching the
    // `OrPattern`, it will return on the first match, even if the
    // `rewriter_func_` doesn't apply a replacement.  Unpacking the
    // `OrPattern` here allows the match to be resumed if
    // `rewriter_func_` returns the original function unmodified.
    // This is only valid for a top-level match.
    if (auto or_pattern = pattern.as<OrPatternNode>()) {
      matches_top_level->push_back(pattern);
      Optional<Expr> output = TryRewrite(expr, or_pattern->left, matches_top_level);
      if (!output.defined()) {
        output = TryRewrite(expr, or_pattern->right, matches_top_level);
      }
      matches_top_level->pop_back();
      return output;
    }

    if (auto opt_matches = ExtractMatchedExpr(pattern, expr, bindings_)) {
      auto matches = opt_matches.value();

      // Append any additional matches that from the unwrapped
      // `OrPattern`.  When matching against `pat = pat_lhs |
      // pat_rhs`, we call `ExtractMatchedExpr` on `pat_lhs` and
      // `pat_rhs` separately.  The top-level `pat` is never seen by
      // `ExtractMatchedExpr`, and must be re-added afterward.
      if (matches_top_level->size()) {
        auto matched_expr = DFPatternMatcher::UnwrapBindings(expr, bindings_);
        for (const auto& pat : *matches_top_level) {
          matches.Set(pat, matched_expr);
        }
      }

      Expr rewritten_expr = rewriter_func_(expr, matches);
      if (!rewritten_expr.same_as(expr)) {
        return builder_->Normalize(rewritten_expr);
      }
    }

    return NullOpt;
  }

  /*! \brief The pattern for rewriting call nodes */
  DFPattern pattern_;
  /*!
   * \brief The user-provided rewriter function. Its signature and semantics are:
   *
   * - (Call, Map<DFPattern, Expr>) -> Call
   *
   *    Given the matched call node and the map of patterns and
   *    matched expressions, it should return a new call node to
   *    replace the original one or the original matched call node as
   *    is.
   */
  TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)> rewriter_func_;

  /*! \brief The known variable bindings
   *
   * The variable bindings whose value is known.  This must be tracked
   * separately from the block builder, so that it can be reset after
   * each iteration of the mutate-until-converged loop applied to
   * `SeqExpr`.
   */
  Map<Var, Expr> bindings_;
};

Function RewriteCall(const DFPattern& pat,
                     TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)> rewriter, Function func) {
  return ExprPatternRewriter::Run(pat, rewriter, func);
}

TVM_REGISTER_GLOBAL("relax.dpl.rewrite_call").set_body_typed(RewriteCall);

}  // namespace relax
}  // namespace tvm
