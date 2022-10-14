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
 * \file tvm/arith/conjunctive_normal_form.cc
 */

#include "conjunctive_normal_form.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pattern_match.h"
#include "rewrite_simplify.h"

namespace tvm {
namespace arith {

namespace {
/* \brief A utility for simplifying expressions using conjunctive/disjuctive normal forms */
class AndOfOrs {
 public:
  /*! \brief Construct the simplifier
   *
   * Convert a PrimExpr to the internal representation.
   *
   * \param expr The PrimExpr to be simplified.
   */
  explicit AndOfOrs(const PrimExpr& expr);

  /*! \brief Convert internal representation to PrimExpr */
  PrimExpr AsPrimExpr() const;

  /*! \brief Simplify the internal representation */
  void Simplify(Analyzer* analyzer);

 private:
  /*! \brief Internal utility, simplify within each group of expressions
   *
   * For each pair of values within a chunk, attempt to simplify them into
   * a single expression.
   *
   * For example,
   *    before = (a == 5) && ((b < 10) || (b > 10))
   *    after  = (a == 5) && ((b != 10) || false)
   */
  void SimplifyWithinChunks(Analyzer* analyzer);

  /*! \brief Internal utility, simplify across groups of expressions
   *
   * For each pair of chunks, if the two chunks differ by only a single
   * term, attempt to simplify those differing terms.
   *
   * For example,
   *    before = ((a == 5) || (b <= 10)) && ((a == 5) || (b >= 10))
   *    after  = ((a == 5) || (b == 10)) && ((a == 5) || true)
   */
  void SimplifyAcrossChunks(Analyzer* analyzer);

  /*! \brief Remove instances of true/false from internal representation
   *
   * To avoid invalidating iterators, `SimplifyWithinChunks` and
   * `SimplifyAcrossChunks` may replace keys, but may not remove keys
   * from the internal representation.  For example, `(a < 5) && (a <
   * 10)` would be simplified to `(a < 5) && true`.  The
   * `RemoveTrueFalse` function removes these leftover instances of
   * true/false.
   */
  void RemoveTrueFalse();

  /*! \brief Internal utility function used to convert to internal form */
  static void VisitAndExpressions(const PrimExpr& expr,
                                  std::function<void(const PrimExpr&)> callback);
  /*! \brief Internal utility function used to convert to internal form */
  static void VisitOrExpressions(const PrimExpr& expr,
                                 std::function<void(const PrimExpr&)> callback);

  /* \brief Type-safe wrapper class that represents an PrimExpr
   *
   * Because integer indices are used frequently through this class,
   * maintaining a separation between integer indices used to access
   * specific elements of the internal representation, and unique
   * identifiers used to represent expressions PrimExpr, is useful.
   */
  enum class Key : size_t {};

  /*! \brief Convert a PrimExpr to a Key */
  Key GetKey(const PrimExpr& expr);

  /*! \brief Convert a Key to a PrimExpr */
  PrimExpr GetExpr(Key key) const;

  /*! \brief Attempt to simplify (a && b)
   *
   * If successful, will overwrite the parameters `a` and `b` with the
   * simplified form.
   */
  void TrySimplifyOr(Key* a, Key* b, Analyzer* analyzer);

  /*! \brief Attempt to simplify (a || b)
   *
   * If successful, will overwrite the parameters `a` and `b` with the
   * simplified form.
   */
  void TrySimplifyAnd(Key* a, Key* b, Analyzer* analyzer);

  /*! \brief The internal representation
   *
   * `chunks[i][j]` is the j-th expression in the i-th OR-group.
   */
  std::vector<std::vector<Key>> chunks_;

  /*! \brief Mapping from internal Key to PrimExpr */
  std::unordered_map<Key, PrimExpr, StructuralHash, StructuralEqual> key_to_expr_;

  /*! \brief Mapping from PrimExpr to internal Key */
  std::unordered_map<PrimExpr, Key, StructuralHash, StructuralEqual> expr_to_key_;

  /*! \brief Cached key representing tir::Bool(true) */
  Key key_true_;

  /*! \brief Cached key representing tir::Bool(false) */
  Key key_false_;
};

AndOfOrs::AndOfOrs(const PrimExpr& expr)
    : key_true_(GetKey(Bool(true))), key_false_(GetKey(Bool(false))) {
  VisitAndExpressions(expr, [&](const PrimExpr& outer_expr) {
    std::vector<Key> or_components;
    VisitOrExpressions(outer_expr, [&](const PrimExpr& inner_expr) {
      Key key = GetKey(inner_expr);
      bool is_duplicate = std::any_of(or_components.begin(), or_components.end(),
                                      [&](Key prev) { return prev == key; });
      if (!is_duplicate) {
        or_components.push_back(key);
      }
    });

    bool is_permutation =
        std::any_of(chunks_.begin(), chunks_.end(), [&](const std::vector<Key>& prev_components) {
          return or_components.size() == prev_components.size() &&
                 std::is_permutation(prev_components.begin(), prev_components.end(),
                                     or_components.begin());
        });
    if (!is_permutation) {
      chunks_.push_back(std::move(or_components));
    }
  });
}

void AndOfOrs::VisitAndExpressions(const PrimExpr& expr,
                                   std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x && y).Match(expr)) {
    // These are separate AND conditions, recurse into them in case
    // they contain AND internally.
    VisitAndExpressions(x.Eval(), callback);
    VisitAndExpressions(y.Eval(), callback);
  } else if ((x || y).Match(expr)) {
    // This may be the bottom-most breakdown, but either x or y may
    // themselves contain AND.  (e.g. (A && B) || (C && D) should be
    // split into (A || C), (A || D), (B || C), and (B || D).)
    // Recurse into each, then reconstruct an OR condition.
    VisitAndExpressions(x.Eval(), [&](const PrimExpr& x_part) {
      VisitAndExpressions(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part || y_part); });
    });
  } else {
    // This is bottom-most breakdown.
    callback(expr);
  }
}

void AndOfOrs::VisitOrExpressions(const PrimExpr& expr,
                                  std::function<void(const PrimExpr&)> callback) {
  PVar<PrimExpr> x, y, z;
  if ((x || y).Match(expr)) {
    // These are separate OR conditions, recurse into them in case
    // they contain OR internally.
    VisitOrExpressions(x.Eval(), callback);
    VisitOrExpressions(y.Eval(), callback);
  } else if ((x && y).Match(expr)) {
    // This may be the bottom-most breakdown, but either x or y may
    // themselves contain OR.  (e.g. (A || B) && (C || D) should be
    // split into (A && C), (A && D), (B && C), and (B && D).)
    // Recurse into each, then reconstruct an AND condition.
    VisitOrExpressions(x.Eval(), [&](const PrimExpr& x_part) {
      VisitOrExpressions(y.Eval(), [&](const PrimExpr& y_part) { callback(x_part && y_part); });
    });
  } else {
    // This is bottom-most breakdown.
    callback(expr);
  }
}

AndOfOrs::Key AndOfOrs::GetKey(const PrimExpr& expr) {
  auto it = expr_to_key_.find(expr);
  if (it != expr_to_key_.end()) {
    return it->second;
  }

  Key key{expr_to_key_.size()};
  expr_to_key_[expr] = key;
  key_to_expr_[key] = expr;
  return key;
}

PrimExpr AndOfOrs::GetExpr(AndOfOrs::Key key) const {
  auto it = key_to_expr_.find(key);
  ICHECK(it != key_to_expr_.end());
  return it->second;
}

PrimExpr AndOfOrs::AsPrimExpr() const {
  PrimExpr expr = Bool(true);
  for (const auto& chunk : chunks_) {
    PrimExpr chunk_expr = Bool(false);
    for (Key j : chunk) {
      chunk_expr = chunk_expr || GetExpr(j);
    }
    expr = expr && chunk_expr;
  }
  return expr;
}

void AndOfOrs::TrySimplifyOr(Key* a_ptr, Key* b_ptr, Analyzer* analyzer) {
  Key& a = *a_ptr;
  Key& b = *b_ptr;
  PrimExpr joint = GetExpr(a) || GetExpr(b);
  PrimExpr simplified = analyzer->Simplify(joint);
  if (!ExprDeepEqual()(simplified, joint)) {
    if (auto* simplified_or = simplified.as<OrNode>()) {
      a = GetKey(simplified_or->a);
      b = GetKey(simplified_or->b);
    } else {
      a = GetKey(simplified);
      b = key_false_;
    }
  }
}

void AndOfOrs::TrySimplifyAnd(Key* a_ptr, Key* b_ptr, Analyzer* analyzer) {
  Key& a = *a_ptr;
  Key& b = *b_ptr;
  PrimExpr joint = GetExpr(a) && GetExpr(b);
  PrimExpr simplified = analyzer->Simplify(joint);
  if (!ExprDeepEqual()(simplified, joint)) {
    if (auto* simplified_and = simplified.as<AndNode>()) {
      a = GetKey(simplified_and->a);
      b = GetKey(simplified_and->b);
    } else {
      a = GetKey(simplified);
      b = key_true_;
    }
  }
}

void AndOfOrs::Simplify(Analyzer* analyzer) {
  SimplifyWithinChunks(analyzer);
  RemoveTrueFalse();
  SimplifyAcrossChunks(analyzer);
  RemoveTrueFalse();
}

void AndOfOrs::SimplifyWithinChunks(Analyzer* analyzer) {
  for (auto& chunk : chunks_) {
    for (size_t expr_i = 0; expr_i < chunk.size(); expr_i++) {
      for (size_t expr_j = expr_i + 1; expr_j < chunk.size(); expr_j++) {
        Key& key_i = chunk[expr_i];
        Key& key_j = chunk[expr_j];

        TrySimplifyOr(&key_i, &key_j, analyzer);
      }
    }
  }
}

void AndOfOrs::SimplifyAcrossChunks(Analyzer* analyzer) {
  for (size_t i_and = 0; i_and < chunks_.size(); i_and++) {
    for (size_t j_and = i_and + 1; j_and < chunks_.size(); j_and++) {
      auto& i_chunk = chunks_[i_and];
      auto& j_chunk = chunks_[j_and];

      if (i_chunk.size() == 1 && j_chunk.size() == 1) {
        auto& key_i = i_chunk[0];
        auto& key_j = j_chunk[0];
        TrySimplifyAnd(&key_i, &key_j, analyzer);
        continue;
      }
      std::unordered_set<Key> j_set(j_chunk.begin(), j_chunk.end());

      std::optional<size_t> i_distinct_index;
      for (size_t i = 0; i < i_chunk.size(); i++) {
        if (!j_set.count(i_chunk[i])) {
          i_distinct_index = i;
          break;
        }
      }

      if (!i_distinct_index.has_value()) {
        // I = (i_0 || i_1 || ... || i_N)
        // J = (i_0 || i_1 || ... || i_N || j_0 || ... || j_N)
        // I && J == I == I && true

        j_chunk = {key_true_};
        continue;
      }

      std::unordered_set<Key> i_set(i_chunk.begin(), i_chunk.end());

      std::optional<size_t> j_distinct_index;
      for (size_t j = 0; j < j_chunk.size(); j++) {
        if (!i_set.count(j_chunk[j])) {
          j_distinct_index = j;
          break;
        }
      }

      if (!j_distinct_index.has_value()) {
        // I = (i_0 || ... || i_N || j_0 || ... || j_N)
        // J = (j_0 || ... || j_N)
        // I && J == J == true && J

        i_chunk = {key_true_};
        continue;
      }

      if (i_chunk.size() == j_chunk.size()) {
        size_t num_shared_exprs = 0;
        for (const auto& j_key : j_chunk) {
          if (i_set.count(j_key)) {
            ++num_shared_exprs;
          }
        }

        if (num_shared_exprs + 1 == i_chunk.size()) {
          // All but one of the expressions are shared.  If the AND
          // of the distinct expressions can be simplified, we can
          // replace.
          //
          // (A or B) and (A or C) => A or (B and C)
          auto& key_i = i_chunk[i_distinct_index.value()];
          auto& key_j = j_chunk[j_distinct_index.value()];
          TrySimplifyAnd(&key_i, &key_j, analyzer);
        }
      }
    }
  }
}

void AndOfOrs::RemoveTrueFalse() {
  for (auto& chunk : chunks_) {
    // Any occurrence of True inside an OR makes the entire expression True.
    if (std::any_of(chunk.begin(), chunk.end(), [&](Key key) { return key == key_true_; })) {
      chunk = {key_true_};
    } else {
      // Any occurrence of False inside an OR can be removed
      chunk.erase(
          std::remove_if(chunk.begin(), chunk.end(), [&](Key key) { return key == key_false_; }),
          chunk.end());
    }
  }

  // Any occurence of False inside an AND makes the entire expression False.
  if (std::any_of(chunks_.begin(), chunks_.end(),
                  [&](const std::vector<Key>& chunk) { return chunk.size() == 0; })) {
    chunks_ = {{}};
  } else {
    // Any occurrence of True inside an AND can be removed.
    chunks_.erase(std::remove_if(chunks_.begin(), chunks_.end(),
                                 [&](const std::vector<Key>& chunk) {
                                   return chunk.size() == 1 && chunk[0] == key_true_;
                                 }),
                  chunks_.end());
  }
}

// Helper utility for temporarily disabling the
// kConvertBooleanToAndOfOrs flag on an analyzer, to prevent infinite
// recursion.
class DisableAndOfOrRecursion {
 public:
  explicit DisableAndOfOrRecursion(Analyzer* analyzer)
      : analyzer_(analyzer), cached_flags_(analyzer->rewrite_simplify.GetEnabledExtensions()) {
    auto new_flags = static_cast<RewriteSimplifier::Extension>(
        cached_flags_ & (~RewriteSimplifier::kConvertBooleanToAndOfOrs));
    analyzer->rewrite_simplify.SetEnabledExtensions(new_flags);
  }
  ~DisableAndOfOrRecursion() { analyzer_->rewrite_simplify.SetEnabledExtensions(cached_flags_); }

  DisableAndOfOrRecursion(const DisableAndOfOrRecursion&) = delete;
  DisableAndOfOrRecursion& operator=(const DisableAndOfOrRecursion&) = delete;

 private:
  Analyzer* analyzer_;
  RewriteSimplifier::Extension cached_flags_;
};

}  // namespace

PrimExpr SimplifyAsAndOfOrs(const PrimExpr& expr, Analyzer* analyzer) {
  DisableAndOfOrRecursion context(analyzer);
  AndOfOrs repr(analyzer->Simplify(expr));
  repr.Simplify(analyzer);
  return repr.AsPrimExpr();
}

}  // namespace arith
}  // namespace tvm
