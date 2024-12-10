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
 * \file src/arith/iter_affine_map.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

#include "../support/utils.h"
#include "const_fold.h"
#include "pattern_match.h"
#include "product_normal_form.h"

namespace tvm {
namespace arith {

using namespace tir;

IterMark::IterMark(PrimExpr source, PrimExpr extent) {
  auto n = make_object<IterMarkNode>();
  n->source = std::move(source);
  n->extent = std::move(extent);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("arith.IterMark").set_body_typed([](PrimExpr source, PrimExpr extent) {
  return IterMark(source, extent);
});

TVM_REGISTER_NODE_TYPE(IterMarkNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterMarkNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterMarkNode*>(node.get());
      p->stream << "IterMark(" << op->source << ", extent=" << op->extent << ")";
    });

IterSplitExpr::IterSplitExpr(IterMark source) {
  auto n = make_object<IterSplitExprNode>();
  auto one = make_const(source->source->dtype, 1);
  n->dtype = source->source->dtype;
  n->source = std::move(source);
  n->extent = n->source->extent;
  n->lower_factor = one;
  n->scale = one;
  data_ = std::move(n);
}

IterSplitExpr::IterSplitExpr(IterMark source, PrimExpr scale) {
  auto n = make_object<IterSplitExprNode>();
  auto one = make_const(source->source->dtype, 1);
  n->dtype = source->source->dtype;
  n->source = std::move(source);
  n->extent = n->source->extent;
  n->lower_factor = one;
  n->scale = std::move(scale);
  data_ = std::move(n);
}

IterSplitExpr::IterSplitExpr(IterMark source, PrimExpr lower_factor, PrimExpr extent,
                             PrimExpr scale) {
  auto n = make_object<IterSplitExprNode>();
  n->dtype = source->source->dtype;
  n->source = std::move(source);
  n->lower_factor = std::move(lower_factor);
  n->extent = std::move(extent);
  n->scale = std::move(scale);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("arith.IterSplitExpr")
    .set_body_typed([](IterMark source, PrimExpr lower_factor, PrimExpr extent, PrimExpr scale) {
      return IterSplitExpr(source, lower_factor, extent, scale);
    });

TVM_REGISTER_NODE_TYPE(IterSplitExprNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterSplitExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterSplitExprNode*>(node.get());
      p->stream << "IterSplit(" << op->source << ", lower_factor=" << op->lower_factor
                << ", extent=" << op->extent << ", scale=" << op->scale << ")";
    });

IterSumExpr::IterSumExpr(Array<IterSplitExpr> args, PrimExpr base) {
  auto n = make_object<IterSumExprNode>();
  n->dtype = base->dtype;
  n->args = std::move(args);
  n->base = std::move(base);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("arith.IterSumExpr")
    .set_body_typed([](Array<IterSplitExpr> args, PrimExpr base) {
      return IterSumExpr(args, base);
    });

TVM_REGISTER_NODE_TYPE(IterSumExprNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IterSumExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterSumExprNode*>(node.get());
      p->stream << "IterSum(" << op->args << ", " << op->base << ")";
    });

/*!
 * \brief Collector that collects the outgoing split reference of each IterMark.
 *
 *  These out-going splits can then be used to check if the iterators are independent.
 */
class IterMarkSplitCollector {
 public:
  // mark all IterMarks that are visited.
  std::unordered_set<IterMark, ObjectPtrHash, ObjectPtrEqual> visited_;
  // each iter mark to its outgoing splits that are referenced.
  std::unordered_map<IterMark, std::vector<IterSplitExpr>, ObjectPtrHash, ObjectPtrEqual>
      mark2splits_;
  /*!
   * \brief Collect all mark2splits recursively from indices.
   * \param indices The iterator of interest.
   */
  void Collect(const Array<IterSumExpr>& indices) {
    for (IterSumExpr sum_expr : indices) {
      for (IterSplitExpr split : sum_expr->args) {
        this->CollectInternal(split->source);
        mark2splits_[split->source].push_back(split);
      }
    }
  }

  void CollectInternal(const IterMark& mark) {
    if (visited_.count(mark)) return;
    visited_.insert(mark);
    if (auto* op = mark->source.as<IterSumExprNode>()) {
      for (IterSplitExpr split : op->args) {
        this->CollectInternal(split->source);
        mark2splits_[split->source].push_back(split);
      }
    }
  }
};

/*! \brief Record form of IterMark(x, extent) + offset */
struct IterMarkWithOffset {
  IterMark mark;
  PrimExpr offset{0};
  IterMarkWithOffset() {}
  IterMarkWithOffset(IterMark mark, PrimExpr offset) : mark(mark), offset(offset) {}
};

/*! \brief Rewriter to rewrite PrimExpr to IterMapExpr when possible */
class IterMapRewriter : public ExprMutator {
 public:
  using Parent = ExprMutator;

  explicit IterMapRewriter(Analyzer* analyzer, const Map<Var, Range>& input_iters,
                           IterMapLevel check_level, bool simplify_trivial_iterators,
                           Array<String>* errors)
      : analyzer_(analyzer),
        check_level_(check_level),
        errors_(*errors),
        padding_predicate_(const_false()) {
    for (auto kv : input_iters) {
      const Var& var = kv.first;
      const Range& vrng = kv.second;
      if (simplify_trivial_iterators && is_one(vrng->extent)) {
        var_map_[var] = IterSumExpr({}, vrng->min);
      } else if (is_zero(vrng->min)) {
        IterMark mark(var, vrng->extent);
        var_map_[var] = IterSplitExpr(mark);
        input_marks_.push_back(mark);
      } else {
        IterMark mark(var - vrng->min, vrng->extent);
        IterSumExpr sum_expr = ToIterSumExpr(IterSplitExpr(mark));
        sum_expr.CopyOnWrite()->base = vrng->min;
        var_map_[var] = sum_expr;
        input_marks_.push_back(mark);
      }
    }
  }

  PrimExpr padding_predicate() const { return padding_predicate_; }
  bool requires_padding() const { return requires_padding_; }

  IterSumExpr Rewrite(const PrimExpr& expr) {
    return NormalizeToIterWithOffset(ToIterSumExpr(DirectMutate(expr)));
  }

  IterSumExpr RewriteAndUpdatePadding(const PrimExpr& expr) {
    update_iterator_padding_ = true;
    auto res = Rewrite(expr);
    update_iterator_padding_ = false;
    return res;
  }

  IterSumExpr RewriteIterConstraint(const PrimExpr& expr,
                                    const Optional<PrimExpr>& predicate_induced_min,
                                    const Optional<PrimExpr>& predicate_induced_max) {
    return NormalizeToIterOnBoundExpr(ToIterSumExpr(DirectMutate(expr)), predicate_induced_min,
                                      predicate_induced_max);
  }

  /**
   * Rewrite expr to iter sum pattern
   * \param expr The input expression
   * \return The rewritten iter sum pattern
   * \note The result base may contain items that is not
   */
  IterSumExpr RewriteToNormalizedIterSum(const PrimExpr& expr) {
    return NormalizeToIterSum(ToIterSumExpr(DirectMutate(expr)));
  }

  /*!
   * \brief If require bijective mapping, this function checks two conditions:
   *   - C0: Each iter mark should be fully covered by non-overlapping splits.
   *   - C1: All of the input iterators are used.
   *   Example: given x in [0, 8) y in [0, 6)
   *   - bindings = [x, x + 1, y] won't pass because x and x+1 contribute
   *     two splits that overlaps with each other.
   *   - bindings = [x / 4, x % 4, y] will pass because x / 4 and x % 4
   *     contribute two non-overlapping splits that covers x.
   *   - bindings = [x / 4, x % 4] won't pass because y is not used.
   *
   *   If only require surjective mapping, this function checks one condition:
   *   - C0: Each iter mark has a chance to be fully covered by non-overlapping splits.
   *   Example: given x in [0, 8) y in [0, 6)
   *   - bindings = [x / 4] will pass because x / 4 can be one split of x
   *   - bindings = [x / 4, x % 4] will pass because x / 4 and x % 4
   *     contribute two non-overlapping splits that covers x.
   *   - bindings = [x / 3] will not pass because x / 3 can not be one split of x
   * \return whether the bindings are valid
   */
  bool CheckMapping(const Array<IterSumExpr>& bindings, IterMapLevel check_level) {
    IterMarkSplitCollector collector;
    // We can check that for each iter mark:
    // All the splits that refers to the iter_mark covers its extent.
    // The splits do not overlap with each other.
    collector.Collect(bindings);

    for (const IterMark& mark : collector.visited_) {
      if (TryNormalizeSplits(mark, collector.mark2splits_[mark], check_level).empty()) {
        return false;
      }
    }
    if (check_level == IterMapLevel::Bijective) {
      // all input marks must be visited
      for (const IterMark& mark : input_marks_) {
        if (collector.visited_.count(mark) == 0 && !is_one(mark->extent)) {
          return false;
        }
      }
    }
    return true;
  }

  /*!
   * \brief Check the validity of iterator constraints
   *    The flattened forms of two different iterator constraints
   *    either 1) follow inclusion relation or 2) have no intersection
   *
   *    For Example, x = i0*30 + i1*15 + i2*3 + i3,
   *    1) [i0*2 + i1 < 3, i2*3 + i3 < 5] is valid, since {i0, i1} \\intersect {i2, i3} = empty set.
   *    2) [i0*2 + i1 < 3, i1*5 + i2 < 5] is not valid,
   *       since {i0, i1} \\intersect {i1, i2} = {i1}, i0 \\in {i0, i1}, i0 \\notin {i1, i2}
   * \return whether the predicates are valid;
   */
  bool CheckConstraints() const {
    // the constrained_iters_flattened_ are in the order of shorter to longer
    // since we visit the predicates in the order of size
    for (size_t i = 0; i < constrained_iters_flattened_.size(); ++i) {
      for (size_t j = i + 1; j < constrained_iters_flattened_.size(); ++j) {
        // state: 0(start), -1(no intersection), 1(inclusion)
        int state = 0;
        for (const IterSplitExpr& arg1 : constrained_iters_flattened_[i]->args) {
          bool found = false;
          for (const IterSplitExpr& arg2 : constrained_iters_flattened_[j]->args) {
            if (IterSplitEqual(arg1, arg2)) {
              found = true;
              break;
            }
          }
          // Check either it is inclusion or intersection, but not both
          if (state == 0) {
            state = found ? 1 : -1;
          } else if ((state == -1 && found) || (state == 1 && !found)) {
            return false;
          }
        }
      }
    }
    return true;
  }

  // override the original mutate function.
  PrimExpr VisitExpr(const PrimExpr& input_expr) final {
    auto expr = ExprMutator::VisitExpr(input_expr);
    if (expr->IsInstance<IterMapExprNode>()) {
      ErrorLogger(this) << "IterMapExpr or subclasses should only result from calls in "
                        << "IterMapRewriter using DirectMutate.  "
                        << "Indirect return occurred in " << input_expr;
      return input_expr;
    }
    return expr;
  }

  // Normal mutation without normalization.
  PrimExpr DirectMutate(const PrimExpr& expr) { return ExprMutator::VisitExpr(expr); }

  PrimExpr VisitExpr_(const VarNode* op) final;
  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;

 private:
  /* \brief Preprocessing common to both FloorDiv and FloorMod
   *
   * \param dividend The dividend to be manipulated.
   */
  IterSumExpr PreprocessDividend(IterMapExpr dividend, PrimExpr original_dividend);

  // Create an iterator that represents the expression (split+base), with
  // padding such that the iterator's extents are evenly divisible by
  // `divisor`.
  //
  // If iterators can have padding added through UpdatePadding, pad a
  // dividend out to be evenly divisible.  Otherwise, validate that the
  // padding previously defined for the split using UpdatePadding can be
  // used.  If no such previous padding exists, return an empty
  // IterMark.
  //
  // Returns a pair of IterSplit that represents (split+base) in a
  // form that can be dividied by divisors, and PrimExpr that
  // represents the left padding applied to split.
  std::pair<IterSplitExpr, PrimExpr> PadDividendToDivisor(IterSplitExpr split, PrimExpr base,
                                                          PrimExpr divisor);

  friend struct ErrorLogger;

  /* \brief Utility class for logging errors.
   *
   * It is not an error for IterMapRewriter to receive an expression that
   * cannot be represented as an IterSumExpr.  In these cases,
   * IterMapRewriter returns the unrepresentable portions of the TIR graph
   * without modification.  As a result, the usual ICHECK or LOG(FATAL)
   * macros cannot be used.  Instead, ErrorLogger(this) can be used to
   * report an unrepresentable TIR graph, which may be used in error
   * messages at the calling scope.
   */
  class ErrorLogger {
   public:
    explicit ErrorLogger(IterMapRewriter* rewriter) : rewriter(rewriter) {}
    ~ErrorLogger() { rewriter->errors_.push_back(os.str()); }

    template <typename T>
    ErrorLogger& operator<<(T&& t) {
      os << std::forward<T>(t);
      return *this;
    }

   private:
    IterMapRewriter* rewriter;
    std::ostringstream os;
  };

  struct IterPaddingInfo {
    // GCD of padding factor collected during first pass
    PrimExpr padding_factor{1};

    PrimExpr left_pad{0};
    PrimExpr right_pad{0};

    // Padded form of original iter mark
    IterMark padded;
  };

  // temp hash for de-duplication purposes.
  struct IterSumHash {
    size_t operator()(const IterSumExpr& value) const {
      // for now only hash on source index.
      size_t hash = value->args.size();
      for (const IterSplitExpr& arg : value->args) {
        hash = support::HashCombine(hash, std::hash<const Object*>()(arg->source.get()));
      }
      return hash;
    }
  };

  static bool IterSplitEqual(const IterSplitExpr& lhs, const IterSplitExpr& rhs,
                             bool check_scale = true) {
    tir::ExprDeepEqual equal;
    if (!lhs->source.same_as(rhs->source)) return false;
    if (!equal(lhs->lower_factor, rhs->lower_factor)) return false;
    if (check_scale && !equal(lhs->scale, rhs->scale)) return false;
    if (!equal(lhs->extent, rhs->extent)) return false;
    return true;
  }

  struct IterSumEqual {
    bool operator()(const IterSumExpr& lhs, const IterSumExpr& rhs) const {
      tir::ExprDeepEqual equal;
      if (lhs->args.size() != rhs->args.size()) return false;
      if (!equal(lhs->base, rhs->base)) return false;
      for (size_t i = 0; i < lhs->args.size(); ++i) {
        if (!IterSplitEqual(lhs->args[i], rhs->args[i])) return false;
      }
      return true;
    }
  };

  // Internal analyzer
  Analyzer* analyzer_;
  // Iter map check level
  IterMapLevel check_level_;
  // Error messages for each unresolved expression.
  Array<String>& errors_;
  // The var map
  std::unordered_map<Var, PrimExpr> var_map_;
  // input iter marks
  std::vector<IterMark> input_marks_;

  // Map from an iter mark to the padded iterator information for
  // it.  This is necessary for introducing the same padding in all
  // usage of an input iterator.  (e.g. (i-1) occurring in the
  // expressions [(i-1)%8, ((i-1)//8)%4, (i-1)//32] should be
  // left-padded by 31 for each occurrence.)
  std::unordered_map<IterMark, IterPaddingInfo, StructuralHash, StructuralEqual> padded_iter_map_;

  // Map from padded iter mark to it's origin mark
  std::unordered_map<IterMark, IterMark, StructuralHash, StructuralEqual> padded_origin_map_;

  /* If update_iterator_padding_ is true, allow the extents of the IterMap to be
   * padded beyond the original iterators.
   *
   * For example, if update_iterator_padding_ is true, the expressions i//4 and
   * i%4, where i is on the range [0,18), would be represented as
   * IterSplit(i, lower_factor=4, extent=5) and IterSplit(i, extent=4).
   * This representation would be forbidden if update_iterator_padding_ is false,
   * because lower_factor=4 does not evenly divide the original extent of
   * 18.
   */
  bool update_iterator_padding_{false};

  /* A boolean expression that is true for any padding that has been
   * introduced, and false otherwise. If update_iterator_padding_ is false,
   * padding_predicate_ will always be false.
   *
   * Example: [i//4, i%4], i in range [0,16)
   *     padding_predicate_ will be false
   *
   * Example: [i//4, i%4], i in range [0,18)
   *     padding_predicate_ will be `(i//4 == 3) && (i%4 >= 2)`
   *
   * Example: [i//4, i%4], i in range [0,N)
   *     padding_predicate_ will be `(N%4!=0) && (i//4 == (N+3)//4-1) && (i%4 >= N%4)`
   */
  PrimExpr padding_predicate_;

  /* A boolean flag denotes there are padding iterations detected
   * in the first round of indices rewriting.
   */
  bool requires_padding_{false};

  // The map for sum that maps flattened form to IterMark with normal form and extent (and possibly
  // an extra offset). The normal form always has minimum value of zero.
  // Example(1): expr = i*9 + j*2 + k, i in [0, 4) j in [0, 5) k in [0, 2)
  //          predicate: j*2 + k < 9
  // Then,    flattened form = IterSum(IterSplit(i, scale=9),
  //                                   IterSplit(j, scale=2),
  //                                   IterSplit(k, scale=1))
  //          normal form    = IterSum(IterSplit(i, scale=9),
  //                                   IterSplit(IterMark(IterSum(IterSplit(j, scale=2),
  //                                                              IterSplit(k, scale=1)),
  //                                                      extent=9)
  //                                             scale=1))
  //          offset         = 0
  // Example(2): expr = i*8 + j*2 + k, i in [0, 4) j in [0, 5) k in [0, 2)
  //          predicate: 1 <= j*2 + k < 9
  // Then,    flattened form = IterSum(IterSplit(i, scale=8),
  //                                   IterSplit(j, scale=2),
  //                                   IterSplit(k, scale=1))
  //          normal form    = IterSum(IterSplit(i, scale=8),
  //                                   IterSplit(IterMark(IterSum(IterSplit(j, scale=2),
  //                                                              IterSplit(k, scale=1), base=-1),
  //                                                      extent=9-1)
  //                                             scale=1),
  //                                   base=0)
  //          offset         = 1
  std::unordered_map<IterSumExpr, IterMarkWithOffset, IterSumHash, IterSumEqual> sum_fuse_map_;
  // The map for sum that maps normal form to flattened form
  // For sum_fuse_map_ and flattened_map_ the following invariants hold:
  //     for any IterSumExpr e in the flattened_form, we have
  //         iter_mark, mark_offset = sum_fuse_map_[e]
  //         flattened_map_[normal_form] = e where normal_form = iter_mark->args[0] and
  //         iter_mark->args.size() = 1
  std::unordered_map<IterSumExpr, IterSumExpr, IterSumHash, IterSumEqual> flattened_map_;
  // The flattened forms of constrained iters
  std::vector<IterSumExpr> constrained_iters_flattened_;

  /*!
   * \brief Look for a split in splits that is not used such that its lower_factor is smallest.
   *        Note that here we use division to compare lower_factor.
   * \param splits the split array to search in.
   * \param used the input used array.
   * \param expected_lower_factor the skipped lower factor.
   * \return the index of the expected split, split.size() if not found.
   */
  size_t SearchSkipLowerFactor(const std::vector<IterSplitExpr>& splits,
                               const std::vector<bool>& used,
                               const PrimExpr& expected_lower_factor) {
    size_t res = splits.size();
    for (size_t i = 0; i < splits.size(); ++i) {
      if (used[i]) continue;
      if (!used[i] && !CanProveDivisible(splits[i]->lower_factor, expected_lower_factor)) {
        // all the remaining unused splits should have their lower factor divisible
        return splits.size();
      }
      if (res == splits.size() ||
          CanProveDivisible(splits[res]->lower_factor, splits[i]->lower_factor)) {
        // note down the split with smaller lower factor
        res = i;
      }
    }
    return res;
  }

  /*!
   * \brief If bijective is required, verify that splits fully covers mark in a non-overlapping
   *   fashion, If not, verify that splits are valid and compatible for the mark.
   *   If verification passes, return splits from outermost to innermost order.
   *   If not, return an empty array.
   * \param mark The iterator of interest.
   * \param splits The splits to be verified.
   * \param check_level Iteration mapping's check level.
   * \return The normalized splits.
   */
  Array<IterSplitExpr> TryNormalizeSplits(const IterMark& mark,
                                          const std::vector<IterSplitExpr>& splits,
                                          IterMapLevel check_level) {
    std::vector<bool> used(splits.size(), false);
    std::vector<IterSplitExpr> iters;
    PrimExpr expected_lower_factor = make_const(mark->source->dtype, 1);

    for (size_t i = 0; i < splits.size(); ++i) {
      size_t j = 0;
      for (; j < splits.size(); ++j) {
        if (used[j]) continue;
        if (!used[j] && analyzer_->CanProveEqual(splits[j]->lower_factor, expected_lower_factor)) {
          break;
        }
      }
      if (j == splits.size()) {
        // we do not allow incomplete split if the bindings should be bijective
        if (check_level == IterMapLevel::Bijective) {
          return Array<IterSplitExpr>();
        }
        // look for the next split skipping this lower factor
        // For example, y \in [0, 24) has 3 splits [y / 6, (y / 2) % 6, y % 2]
        // It is valid to only have [y / 6, y % 2] if bijective is not required
        // We can skip (y / 2) % 6
        j = SearchSkipLowerFactor(splits, used, expected_lower_factor);
        // split not found
        if (j == splits.size()) {
          return Array<IterSplitExpr>();
        }
      }

      used[j] = true;
      iters.push_back(splits[j]);
      expected_lower_factor = splits[j]->lower_factor * splits[j]->extent;
    }

    // Extract iteration mark info before padding
    auto pad_mark_it = padded_origin_map_.find(mark);
    bool has_padding = pad_mark_it != padded_origin_map_.end();

    bool match_full_iter = analyzer_->CanProveEqual(expected_lower_factor, mark->extent);
    bool match_iter_divisor =
        match_full_iter || CanProveDivisible(mark->extent, expected_lower_factor);

    // Case 1. bijective is required.
    //         We check the extent we calculate is consistent with the extent of the mark and
    //         iteration mark's padding is not allowed.
    //
    // Case 2. bijective is not required and there is no padding.
    //         We check the extent we calculate is a factor of the extent of the mark
    //         For example, y \in [0, 24) [(y / 2) % 6, y % 2] is valid, but y \in [0, 25) is not.
    //
    // Case 3. bijective is not required and there exists padding. We check either
    //   (3.1) The extent we calculate is consistent with the extent of the padded mark and it is
    //         the single split for the iter mark.
    //         For example, padded iter p in [0, 24), [(p / 12)] is valid because it is surjective
    //         according to how we pad the original iteration mark.
    //   (3.2) The extent we calculate is a factor of the extent of the padded mark, and the extent
    //         before padding is greater or equal than the extent we calculate.
    //         For example, the original extent is 14, [(p % 12)] is valid, with p padded to 24.
    //
    if (check_level == IterMapLevel::Bijective) {
      if (has_padding) {
        ErrorLogger(this) << "Bijectvie mapping should not take iter paddings";
        return {};
      } else if (!match_full_iter) {
        ErrorLogger(this) << "The iterations do not traverse full iter space";
        return {};
      }
    } else if (!has_padding) {
      if (!match_iter_divisor) {
        ErrorLogger(this) << "The lower factor is not divisible by the full iter space extent";
        return {};
      }
    } else if (check_level == IterMapLevel::Surjective) {
      PrimExpr extent_before_padding = pad_mark_it->second->extent;
      if (match_full_iter) {
        if (splits.size() != 1) {
          ErrorLogger(this) << "Dependent iterations on padding iter space";
          return Array<IterSplitExpr>();
        } else if (analyzer_->CanProveEqual(splits[0]->extent, expected_lower_factor) &&
                   !analyzer_->CanProve(extent_before_padding >= expected_lower_factor)) {
          ErrorLogger(this) << "Split on padding iteration is not surjective "
                            << "if the split extent equals to the full iter space extent";
          return Array<IterSplitExpr>();
        }
      } else if (match_iter_divisor) {
        if (!analyzer_->CanProve(extent_before_padding >= expected_lower_factor)) {
          ErrorLogger(this) << "The extent before padding is less than lower factor";
          return Array<IterSplitExpr>();
        }
      } else {
        ErrorLogger(this) << "The lower factor is not divisible by the full iter space extent";
        return {};
      }
    }
    return Array<IterSplitExpr>(iters.rbegin(), iters.rend());
  }

  /*!
   * \brief Normalize the iter expression with constraint (min <= expr < max)
   * \param expr The iter expression.
   * \param predicate_induced_min Closed lower bound from iter constraint, maybe undefined.
   * \param predicate_induced_max Open upper bound from iter constraint, maybe undefined.
   * \return The Normalized expression.
   */
  IterSumExpr NormalizeToIterOnBoundExpr(IterSumExpr expr, Optional<PrimExpr> predicate_induced_min,
                                         Optional<PrimExpr> predicate_induced_max) {
    // normalize to zero base
    PrimExpr base = expr->base;
    if (!is_zero(base)) {
      expr.CopyOnWrite()->base = 0;
      if (predicate_induced_min.defined())
        predicate_induced_min = predicate_induced_min.value() - base;
      if (predicate_induced_max.defined())
        predicate_induced_max = predicate_induced_max.value() - base;
    }
    Optional<IterSumExpr> opt = TryFuseIters(expr, check_level_, false);
    ICHECK(!opt.defined() || opt.value()->args.size() == 1);
    // scale should be 1
    if (opt.defined() && is_one(opt.value()->args[0]->scale)) {
      const IterSplitExpr split = opt.value()->args[0];
      IterSumExpr structured_form = Downcast<IterSumExpr>(split->source->source);
      // get the flattened form
      auto it = flattened_map_.find(structured_form);
      ICHECK(it != flattened_map_.end());
      IterSumExpr flattened_form = it->second;
      // get the mark and offset of the structured_form
      auto it_mark = sum_fuse_map_.find(flattened_form);
      ICHECK(it_mark != sum_fuse_map_.end());
      IterMark mark = it_mark->second.mark;
      PrimExpr mark_offset = it_mark->second.offset;
      PrimExpr iter_min = mark_offset;
      PrimExpr iter_max = iter_min + mark->extent;
      // the delta of iter_min when it is updated when the lower bound predicate is present
      PrimExpr iter_min_delta = make_const(iter_min.dtype(), 0);
      if (predicate_induced_min.defined()) {
        iter_min_delta = max(predicate_induced_min.value(), iter_min) - iter_min;
        iter_min = max(predicate_induced_min.value(), iter_min);
      }
      if (predicate_induced_max.defined()) {
        // NOTE: important to do explicit prove here
        // because we have a domain knowledge that most predicates
        // tries to constraint the expression and we favor predicate_induced_max
        // when available.
        //
        // This path can help enable predicate simplfication for
        // symbolic cases like:
        //
        // z = x * 32 + y < n * 16 where x in [0, (n+1)//2), y in [0, 32)
        if (analyzer_->CanProve(predicate_induced_max.value() <= iter_max)) {
          iter_max = predicate_induced_max.value();
        } else {
          iter_max = min(predicate_induced_max.value(), iter_max);
        }
      }
      // When iter_min_delta is present, we need to normalize the structured form to have minimum of
      // 0, and add the delta to the mark_offset
      if (!is_zero(iter_min_delta)) {
        // structured form's offset should be updated
        flattened_map_.erase(structured_form);
        structured_form.CopyOnWrite()->base -= iter_min_delta;
        mark.CopyOnWrite()->source = structured_form;
        flattened_map_[structured_form] = flattened_form;
      }
      mark.CopyOnWrite()->extent = iter_max - iter_min;
      sum_fuse_map_[flattened_form] = {mark, iter_min};
      // we need to note down the flattened form of constrained iterators
      // to check the validity of constraints, see also CheckConstraints()
      constrained_iters_flattened_.push_back(flattened_form);
      IterSumExprNode* normalized_expr = expr.CopyOnWrite();
      normalized_expr->args = Array<IterSplitExpr>({split});
      normalized_expr->base = base;
      return expr;
    }
    ErrorLogger(this) << "Could not normalize iterators using the constraints given.";
    return expr;
  }

  /*!
   * \brief Normalize expr to an iterator + offset.
   * \param expr The input expression.
   * \return The Normalized expression.
   */
  IterSumExpr NormalizeToIterWithOffset(IterSumExpr expr) {
    // We are normalizing a regular iter
    if (expr->args.size() < 1) return expr;
    Optional<IterSumExpr> opt = TryFuseIters(expr, check_level_, true);
    if (opt.defined()) {
      return opt.value();
    } else {
      ErrorLogger(this) << "Could not normalize iterators";
      return expr;
    }
  }

  /*!
   * \brief Normalize expr to iter sum.
   *
   * The normalized result ensures that
   * each scale is in the form of (symbol_prod) * cscale
   *
   * It will also sort entries in desc order by cscale then len(symbol_prod).
   *
   * This is a best effort sorting since some scale can be symbolic.
   * We first order them by the constant factors, then the number of symbols
   * involved in a multiply
   *
   * \param expr The input expression.
   * \return The Normalized expression.
   */
  IterSumExpr NormalizeToIterSum(IterSumExpr expr) {
    // We are normalizing a regular iter
    if (expr->args.size() < 1) return expr;
    if (auto opt = TryCombineSplitFromSameSource(expr)) {
      expr = opt.value();
      if (expr->args.size() < 1) return expr;
    }
    struct Item {
      int64_t cscale;
      int64_t symbol_prod_count;
      IterSplitExpr split;
    };

    std::vector<Item> items;

    for (IterSplitExpr split : expr->args) {
      int64_t symbol_prod_count = 0;
      int64_t cscale = 1;
      PrimExpr res = tir::make_const(split.dtype(), 1);
      auto fcollect = [&](PrimExpr val) {
        if (const auto* intimm = val.as<IntImmNode>()) {
          cscale *= intimm->value;
        } else {
          res = res * val;
          ++symbol_prod_count;
        }
      };
      UnpackReduction<tir::MulNode>(split->scale, fcollect);
      if (cscale != 1) {
        res = res * tir::make_const(res.dtype(), cscale);
      }
      split.CopyOnWrite()->scale = res;
      items.emplace_back(Item{cscale, symbol_prod_count, split});
    }

    std::stable_sort(items.begin(), items.end(), [](const Item& lhs, const Item& rhs) {
      if (lhs.cscale > rhs.cscale) return true;
      if (lhs.cscale < rhs.cscale) return false;
      return lhs.symbol_prod_count > rhs.symbol_prod_count;
    });

    Array<IterSplitExpr> args;
    for (const Item& item : items) {
      args.push_back(item.split);
    }

    expr.CopyOnWrite()->args = args;
    expr.CopyOnWrite()->base = NormalizeIterMapToExpr(expr->base);
    return expr;
  }

  /*!
   * \brief Create a IterSumExpr from expr.
   * \param expr The input expr.
   * \return The transformed IterSumExpr.
   */
  static IterSumExpr ToIterSumExpr(const PrimExpr& expr) {
    if (auto op = expr.as<IterSumExpr>()) {
      return op.value();
    } else if (auto op = expr.as<IterSplitExpr>()) {
      return IterSumExpr({op.value()}, make_zero(expr->dtype));
    } else {
      ICHECK(!expr->IsInstance<IterMapExprNode>());
      return IterSumExpr({}, expr);
    }
  }

  /**
   * \brief Helper method to find base iterator which is the
   * iterator with the smallest scale.
   *
   * \param expr The expression to search.
   * \param skip_flag Whether to skip the position
   * \param match_source Whether to only match the same source.
   * \param rbegin The last index to start reverse searching, -1 means everything.
   * \return Whether we can find one.
   */
  int FindBaseIter(const IterSumExpr& expr, const std::vector<bool>& skip_flag,
                   Optional<IterMark> match_source, int rbegin = -1) {
    if (rbegin == -1) {
      rbegin = static_cast<int>(expr->args.size()) - 1;
    }
    // First, find the scale with minimum size of constant scale.
    // use reverse search as usually smallest is ordered on the right
    int base_index = -1;
    int64_t min_const_scale = 0;

    for (int i = rbegin; i >= 0; --i) {
      if (skip_flag[i]) continue;
      if (match_source.defined() && !match_source.same_as(expr->args[i]->source)) continue;
      if (const auto* op = expr->args[i]->scale.as<IntImmNode>()) {
        if (base_index == -1 || op->value < min_const_scale) {
          min_const_scale = op->value;
          base_index = static_cast<int>(i);
        } else if (op->value == min_const_scale) {
          // for ties, we want to look into 1 extent trivial iters
          // prioritize trivial iterators
          if (is_one(expr->args[i]->extent) && !is_one(expr->args[base_index]->extent)) {
            base_index = static_cast<int>(i);
          }
        }
      }
    }
    // cannot find constant scale, try to find scale that comes with
    // smallest product size, which usually is smallest in symbolic land
    // x < x * y
    int min_reduce_size = 0;
    for (int i = rbegin; i >= 0; --i) {
      if (skip_flag[i]) continue;
      if (match_source.defined() && !match_source.same_as(expr->args[i]->source)) continue;
      int reduce_size = 0;
      auto fcollect = [&](const PrimExpr&) { ++reduce_size; };
      UnpackReduction<tir::MulNode>(expr->args[i]->scale, fcollect);
      if (base_index == -1 || reduce_size < min_reduce_size) {
        min_reduce_size = reduce_size;
        base_index = static_cast<int>(i);
      }
    }
    return base_index;
  }

  /*!
   * \brief Find the first possible location that have extent equals 1
   *
   * Unit extent can be rare in simplifications and not having them can
   * help us do early exit in scale matching.
   *
   * This parameter is being used in FindIterWithExactScale N times.
   * \param expr The input expression.
   */
  int FindFirstPossibleUnitExtentIndex(const IterSumExpr& expr) {
    for (size_t i = 0; i < expr->args.size(); ++i) {
      if (is_one(expr->args[i]->extent)) return static_cast<int>(i);
    }
    return static_cast<int>(expr->args.size());
  }

  /*!
   * \brief Helper method to find iterator with exact the expected scale.
   * \param expr The expression.
   * \param skip_flag skip_flag the position already visited to skip.
   * \param match_source Must match the same source.
   * \param expected_scale The expected_scale.
   * \param rbegin The last index to start reverse searching, -1 means everything.
   * \param first_possible_unit_extent_pos The last possible locatin with split->extent == 1
   * \return -1 if not no match found, otherwise return the index.
   */
  int FindIterWithExactScale(const IterSumExpr& expr, const std::vector<bool>& skip_flag,
                             const PrimExpr& expected_scale, Optional<IterMark> match_source,
                             int rbegin = -1, int first_possible_unit_extent_pos = 0) {
    if (rbegin == -1) {
      rbegin = static_cast<int>(expr->args.size()) - 1;
    }
    int matched_pos = -1;
    // use reverse search, as smallest scale usually are near the end.
    for (int j = rbegin; j >= 0; --j) {
      if (skip_flag[j]) continue;
      if (match_source.defined() && !match_source.same_as(expr->args[j]->source)) continue;
      const PrimExpr& cur_scale = expr->args[j]->scale;
      // for bijective mapping, the matched scale must equal to expected scale
      if (analyzer_->CanProveEqual(cur_scale, expected_scale)) {
        if (is_one(expr->args[j]->extent)) return j;
        // if extent is not one and there is a possible extent=1 split
        // further out, we need to extent the search
        // extent=1 gets higher priority since they don't change the scale
        // if there are multiple of them, we match the first encountered
        if (matched_pos == -1) {
          matched_pos = j;
        }
        if (j <= first_possible_unit_extent_pos) return matched_pos;
      }
    }
    return matched_pos;
  }

  /*!
   * \brief Helper method to find iterator whose scale is smaller
   *         than but closest to the expected scale.
   * \param expr The expression.
   * \param skip_flag skip_flag the position already visited to skip.
   * \param expected_scale The expected_scale.
   * \return -1 if not no match found, otherwise return the index.
   */
  int FindIterSmallerClosestToScale(const IterSumExpr& expr, const std::vector<bool>& skip_flag,
                                    const PrimExpr& expected_scale, PrimExpr* out_matched_scale) {
    // use reverse search, as smallest scale usually are near the end.
    int matched_pos = -1;
    PrimExpr matched_scale;
    for (int j = static_cast<int>(expr->args.size()) - 1; j >= 0; --j) {
      if (skip_flag[j]) continue;
      const PrimExpr& cur_scale = expr->args[j]->scale;
      // find the closest scale which is less or equal to expected scale
      if (analyzer_->CanProveGreaterEqual(expected_scale - cur_scale, 0) &&
          analyzer_->CanProveGreaterEqual(cur_scale, 0)) {
        if (matched_pos == -1 || analyzer_->CanProveLess(matched_scale - cur_scale, 0)) {
          matched_pos = j;
          matched_scale = cur_scale;
        }
      }
    }
    *out_matched_scale = matched_scale;
    return matched_pos;
  }
  /*!
   * \brief IterSum = x1*c1 + x2*c2 + ... + xn*cn + base
   *      = (x1*s1 + x2*s2 + ... + xn)*cn + base
   *
   * Try to combine consecutives IterSplit.
   * This is helpful to combine iterators from consecutive splits.
   *
   * \param expr The input sum.
   * \param check_level The check level if iter mapping.
   * \return The sum with the fused IterMark and extra offset if succeed.
   */
  Optional<IterSumExpr> TryCombineSplitFromSameSource(IterSumExpr expr) {
    if (expr->args.size() <= 1) return NullOpt;
    std::unordered_map<IterMark, int, ObjectPtrHash, ObjectPtrEqual> hit_count;
    // most iter map are small n < 5
    // so we can afford N^2 complexity
    bool has_overlap = false;

    for (size_t i = 0; i < expr->args.size(); ++i) {
      auto it = hit_count.find(expr->args[i]->source);
      if (it != hit_count.end()) {
        ++it->second;
        has_overlap = true;
      } else {
        hit_count[expr->args[i]->source] = 1;
      }
    }
    if (!has_overlap) return NullOpt;

    std::vector<bool> visited(expr->args.size(), false);
    std::vector<IterSplitExpr> reverse_flattened_iters;
    int first_possible_unit_extent_pos = FindFirstPossibleUnitExtentIndex(expr);

    // Start eliminating the iterators
    for (int rend = static_cast<int>(expr->args.size()) - 1; rend >= 0;) {
      if (visited[rend]) {
        --rend;
        continue;
      }
      if (hit_count.at(expr->args[rend]->source) == 1) {
        reverse_flattened_iters.push_back(expr->args[rend]);
        visited[rend] = true;
        --rend;
        continue;
      }
      // NOTE: split have the following pattern
      //
      // result = (source // lower_factor) % extent * scale
      //        = (source % (extent * lower_factor)) // lower_factor * scale (rule A)
      //
      // Try to simplify with the following rule:
      //
      //    ((x // (c * s)) % m) * s + ((x // c) % s)
      // => (x // c) % (m * s)
      //
      // Assume we have the following split relations:
      // - lhs = ((x // (c * s)) % m) * (s * t)
      // - rhs = ((x // c) % s) * t
      // - result = combine(lhs, rhs) = (x // c) % (m * s) * t
      //
      // Key things to match:
      // - lhs->lower_factor == rhs->lower_factor * rhs->extent
      // - lhs->scale == rhs->extent * rhs->scale
      //
      // The final result contains the following relation
      // - result->lower_factor = rhs->lower_factor
      // - result->scale = rhs->scale
      // - result->extent = lhs->extent * rhs->extent
      // Find base index, must have a candidate to make progress
      int matched_index = FindBaseIter(expr, visited, expr->args[rend]->source, rend);
      ICHECK_NE(matched_index, -1);
      visited[matched_index] = true;
      IterSplitExpr rhs_iter = expr->args[matched_index];

      while (true) {
        // NOTE: mul order [lower_factor, extent, scale]
        PrimExpr lhs_scale = MulAndNormalize(rhs_iter->extent, rhs_iter->scale);
        matched_index = FindIterWithExactScale(expr, visited, lhs_scale, rhs_iter->source, rend,
                                               first_possible_unit_extent_pos);
        if (matched_index == -1) break;
        IterSplitExpr lhs_iter = expr->args[matched_index];
        ICHECK(rhs_iter->source.same_as(lhs_iter->source));
        PrimExpr lhs_lower_factor = MulAndNormalize(rhs_iter->lower_factor, rhs_iter->extent);
        if (!analyzer_->CanProveEqual(lhs_iter->lower_factor, lhs_lower_factor)) break;
        // all patterns match
        visited[matched_index] = true;
        // Update rhs iter to result, only update of extent is necessary
        rhs_iter.CopyOnWrite()->extent = MulAndNormalize(lhs_iter->extent, rhs_iter->extent);
      }
      // push back the combined iter in rhs_iter
      reverse_flattened_iters.push_back(rhs_iter);
    }

    IterSumExpr simplified_sum = expr;
    // flip the order so we preserve the original order
    simplified_sum.CopyOnWrite()->args =
        Array<IterSplitExpr>(reverse_flattened_iters.rbegin(), reverse_flattened_iters.rend());
    return simplified_sum;
  }

  /*!
   * \brief IterSum = x1*c1 + x2*c2 + ... + xn*cn + base
   *      = (x1*s1 + x2*s2 + ... + xn)*cn + base
   *      = y*cn (IterMark y => x1*s1 + x2*s2 + ... + xn) + base
   *      = [IterSplit(IterMark(y), scale=cn)] + base
   *    return a corresponding IterSumExpr with extra offset if needed.
   *    Try to normalize IterSum into a fused IterMark
   * \param expr The input sum.
   * \param check_level The check level if iter mapping.
   * \param allow_early_skip Whether do we allow early skip if expr is simple
   *        (this may cause us to return parameters that are not canonically wrapped as
   * IterSum(IterMark)) \return The sum with the fused IterMark and extra offset if succeed.
   */
  Optional<IterSumExpr> TryFuseIters(IterSumExpr expr, IterMapLevel check_level,
                                     bool allow_early_skip) {
    if (auto opt = TryCombineSplitFromSameSource(expr)) {
      expr = opt.value();
      if (expr->args.size() <= 1 && allow_early_skip) {
        return expr;
      }
    }
    // select the iterators in order
    std::vector<bool> visited(expr->args.size(), false);
    int base_index = FindBaseIter(expr, visited, NullOpt);
    if (base_index == -1) return NullOpt;
    PrimExpr base_scale = expr->args[base_index]->scale;

    std::vector<IterSplitExpr> flattened_iters, grouped_iters;

    // check if it can be remapped into a fused pattern.
    PrimExpr expected_extra_base = make_const(expr.dtype(), 0);
    PrimExpr tail_extent = make_const(expr.dtype(), 0);
    PrimExpr expected_scale = base_scale;
    int first_possible_unit_extent_pos = FindFirstPossibleUnitExtentIndex(expr);

    for (size_t i = 0; i < expr->args.size();) {
      PrimExpr matched_scale{nullptr};
      bool is_exact_match{false};
      // find position such that expr->args[j] match expected scale
      // if it is first step, we can simply start with base index
      int matched_pos = i == 0 ? base_index
                               : FindIterWithExactScale(expr, visited, expected_scale, NullOpt, -1,
                                                        first_possible_unit_extent_pos);
      if (matched_pos != -1) {
        matched_scale = expected_scale;
        is_exact_match = true;
      }
      if (matched_pos == -1) {
        // if exact scale is not possible, try to find an iter with scale
        // that is smaller but closest to the scale.
        if (check_level != IterMapLevel::Bijective && is_const_int(base_scale, 1)) {
          matched_pos =
              FindIterSmallerClosestToScale(expr, visited, expected_scale, &matched_scale);
        }
      }
      if (matched_pos == -1) {
        return NullOpt;
      }
      ICHECK(matched_scale.defined());
      // look for the longest constrained iter started from expr->args[j]
      // Example: expr = i*9 + j*2 + k, i in [0, 4) j in [0, 5) k in [0, 2)
      //          predicate: j*2 + k < 9
      // We need to match the predicate in expr and adjust the expected scale,
      // otherwise we expect the scale of i to be 2*5=10
      Optional<IterSumExpr> constraint_to_match;
      for (const IterSumExpr& iter : constrained_iters_flattened_) {
        if (IterSplitEqual(expr->args[matched_pos], iter->args.back(), false)) {
          // find a predicate started from match position
          if (!constraint_to_match ||
              constraint_to_match.value()->args.size() < iter->args.size()) {
            constraint_to_match = iter;
          }
        }
      }
      if (constraint_to_match) {
        // match the predicate and mark the iterators in the constraint_to_match as visited
        // Example: expr = i*9 + j*2 + k, i in [0, 4) j in [0, 5) k in [0, 2)
        //          predicate = j*2 + k < 9
        //          then j*2 + k matches the lower two splits of expr
        for (auto it = constraint_to_match.value()->args.rbegin();
             it != constraint_to_match.value()->args.rend(); ++it) {
          size_t k = 0;
          for (; k < expr->args.size(); ++k) {
            if (!visited[k] && IterSplitEqual(expr->args[k], *it, false)) {
              if (analyzer_->CanProveEqual((*it)->scale * matched_scale, expr->args[k]->scale)) {
                break;
              }
            }
          }
          if (k == expr->args.size()) {
            return NullOpt;
          }
          visited[k] = true;
          flattened_iters.push_back(expr->args[k]);
        }
        auto iter = sum_fuse_map_.find(constraint_to_match.value());
        ICHECK(iter != sum_fuse_map_.end());
        const IterMarkWithOffset& iter_matched = iter->second;
        grouped_iters.emplace_back(iter_matched.mark, floordiv(matched_scale, base_scale));
        expected_extra_base += iter_matched.offset * matched_scale;
        if (!is_exact_match) {
          tail_extent += expected_scale - matched_scale;
        }
        // NOTE: order [lower_factor, extent, scale]
        expected_scale = MulAndNormalize(iter_matched.mark->extent, matched_scale);
        // move forward
        i += constraint_to_match.value()->args.size();
      } else {
        // constraint_to_match not found, skip this iterator
        visited[matched_pos] = true;
        IterSplitExpr arg = expr->args[matched_pos];
        arg.CopyOnWrite()->scale = analyzer_->Simplify(floordiv(arg->scale, base_scale));
        flattened_iters.push_back(arg);
        grouped_iters.push_back(arg);
        if (!is_exact_match) {
          tail_extent += expected_scale - matched_scale;
        }
        // NOTE: order [lower_factor, extent, scale]
        expected_scale = MulAndNormalize(expr->args[matched_pos]->extent, matched_scale);
        ++i;
      }
    }
    // Get the flattened form and structured form
    // both forms have splits from outermost to innermost
    IterSumExpr structured_form = expr, flattened_form = expr;
    flattened_form.CopyOnWrite()->args =
        Array<IterSplitExpr>(flattened_iters.rbegin(), flattened_iters.rend());
    flattened_form.CopyOnWrite()->base = make_const(expr.dtype(), 0);
    structured_form.CopyOnWrite()->args =
        Array<IterSplitExpr>(grouped_iters.rbegin(), grouped_iters.rend());
    structured_form.CopyOnWrite()->base = make_const(expr.dtype(), 0);
    auto it = sum_fuse_map_.find(flattened_form);
    if (it != sum_fuse_map_.end()) {
      // old iter
      if (!analyzer_->CanProveEqual(expected_extra_base, it->second.offset * base_scale)) {
        // the extra offset is not consistent with old
        return NullOpt;
      }
      return IterSumExpr({IterSplitExpr(it->second.mark, base_scale)},
                         expr->base + expected_extra_base);
    } else {
      // new iter, form a new mark
      IterMark mark = IterMark(structured_form, div(expected_scale, base_scale) + tail_extent);
      sum_fuse_map_[flattened_form] = IterMarkWithOffset(mark, expected_extra_base);
      flattened_map_[structured_form] = flattened_form;
      return IterSumExpr({IterSplitExpr(mark, base_scale)}, expr->base + expected_extra_base);
    }
  }

  bool CanProveDivisible(const PrimExpr& lhs, const PrimExpr& rhs);

  PrimExpr SplitFloorDivConst(IterSplitExpr lhs, PrimExpr base, PrimExpr rhs);
  PrimExpr SplitFloorModConst(IterSplitExpr lhs, PrimExpr base, PrimExpr rhs);

  static void AddToLhs(IterSumExprNode* lhs, IterSplitExpr rhs, int sign) {
    tir::ExprDeepEqual equal;
    for (size_t i = 0; i < lhs->args.size(); ++i) {
      IterSplitExpr lvalue = lhs->args[i];
      if (lvalue->source.same_as(rhs->source) && equal(lvalue->lower_factor, rhs->lower_factor) &&
          equal(lvalue->extent, rhs->extent)) {
        if (sign > 0) {
          rhs.CopyOnWrite()->scale = lvalue->scale + rhs->scale;
        } else {
          rhs.CopyOnWrite()->scale = lvalue->scale - rhs->scale;
        }
        lhs->args.Set(i, rhs);
        return;
      }
    }
    if (sign > 0) {
      lhs->args.push_back(rhs);
    } else {
      rhs.CopyOnWrite()->scale = make_zero(rhs->scale.dtype()) - rhs->scale;
      lhs->args.push_back(rhs);
    }
  }

  static void AddToLhs(IterSumExprNode* lhs, const IterSumExpr& rhs, int sign) {
    for (const auto& arg : rhs->args) {
      AddToLhs(lhs, arg, sign);
    }
    if (sign > 0) {
      lhs->base += rhs->base;
    } else {
      lhs->base -= rhs->base;
    }
  }

  static void MulToLhs(IterSumExprNode* lhs, const PrimExpr& rhs) {
    for (size_t i = 0; i < lhs->args.size(); ++i) {
      IterSplitExpr lvalue = lhs->args[i];
      lvalue.CopyOnWrite()->scale *= rhs;
      lhs->args.Set(i, lvalue);
    }
    lhs->base *= rhs;
  }
};

/*! \brief An internal struct to represent range extent on iterators(iter < upper_bound). */
struct IterConstraint {
  // The expr of the iter
  PrimExpr iter;
  // The expr of the lower_bound, maybe undefined
  Optional<PrimExpr> lower_bound;
  // The expr of the upper_bound, maybe undefined
  Optional<PrimExpr> upper_bound;
  // The size of the iter, which is the number of nodes
  size_t expr_size = 0;

  IterConstraint(PrimExpr iter, Optional<PrimExpr> lower_bound, Optional<PrimExpr> upper_bound,
                 size_t size)
      : iter(std::move(iter)),
        lower_bound(std::move(lower_bound)),
        upper_bound(std::move(upper_bound)),
        expr_size(size) {}
};

/*!
 * \brief Split the predicate into `(a < b) && (c < d) && ...`
 * \param pred The predicate to be split.
 * \param input_iters The input iterators.
 * \param result The result of predicate split.
 * \return A list of IterConstraint, empty if the split failed.
 */
bool MatchBoundConstraints(PrimExpr pred, Map<Var, Range>* input_iters,
                           std::vector<IterConstraint>* result) {
  arith::PVar<PrimExpr> lhs, rhs, rest;
  for (;;) {
    // try extract comparisions
    bool is_finish = false;
    bool is_greater = false;
    bool is_equal = false;
    if ((rest && (lhs < rhs)).Match(pred) || ((lhs < rhs) && rest).Match(pred)) {
      // pass
    } else if ((lhs < rhs).Match(pred)) {
      is_finish = true;
    } else if ((rest && (lhs <= rhs)).Match(pred) || ((lhs <= rhs) && rest).Match(pred)) {
      is_equal = true;
    } else if ((lhs <= rhs).Match(pred)) {
      is_equal = true;
      is_finish = true;
    } else if ((rest && (lhs > rhs)).Match(pred) || ((lhs > rhs) && rest).Match(pred)) {
      is_greater = true;
    } else if ((lhs > rhs).Match(pred)) {
      is_greater = true;
      is_finish = true;
    } else if ((rest && (lhs >= rhs)).Match(pred) || ((lhs >= rhs) && rest).Match(pred)) {
      is_greater = true;
      is_equal = true;
    } else if ((lhs >= rhs).Match(pred)) {
      is_greater = true;
      is_equal = true;
      is_finish = true;
    } else {
      return false;
    }
    PrimExpr lhs_expr = lhs.Eval();
    PrimExpr rhs_expr = rhs.Eval();
    // we only accept predicate of integers
    if (!((lhs_expr->dtype.is_int() || lhs_expr->dtype.is_uint()) &&
          (rhs_expr->dtype.is_int() || rhs_expr->dtype.is_uint()))) {
      return false;
    }
    // determine iter and bound, if we can not distinguish them simply,
    // try divide (lhs - rhs) into itervar aware and itervar free parts
    auto f_use_itervar = [&input_iters](const VarNode* v) {
      return input_iters->count(GetRef<Var>(v));
    };
    bool bound_at_left;
    if (UsesVar(lhs_expr, f_use_itervar) || UsesVar(rhs_expr, f_use_itervar)) {
      // At least it uses one input iter
      if (is_const_int(lhs_expr) || !UsesVar(lhs_expr, f_use_itervar)) {
        bound_at_left = true;
      } else if (is_const_int(rhs_expr) || !UsesVar(rhs_expr, f_use_itervar)) {
        bound_at_left = false;
      } else {
        bound_at_left = false;  // accumulate bound to rhs
        PrimExpr sum_parts = lhs_expr - rhs_expr;
        lhs_expr = 0;
        rhs_expr = 0;
        std::function<void(const PrimExpr&, bool)> f_extract =
            [&lhs_expr, &rhs_expr, f_use_itervar, &f_extract](const PrimExpr& part, bool sign) {
              if (const AddNode* add = part.as<AddNode>()) {
                f_extract(add->a, sign);
                f_extract(add->b, sign);
              } else if (const SubNode* sub = part.as<SubNode>()) {
                f_extract(sub->a, sign);
                f_extract(sub->b, !sign);
              } else if (UsesVar(part, f_use_itervar)) {
                lhs_expr = sign ? lhs_expr + part : lhs_expr - part;
              } else {
                rhs_expr = sign ? rhs_expr - part : rhs_expr + part;
              }
            };
        f_extract(sum_parts, true);
        arith::Analyzer analyzer;
        lhs_expr = analyzer.Simplify(lhs_expr);
        rhs_expr = analyzer.Simplify(rhs_expr);
      }
      Optional<PrimExpr> lower_bound = NullOpt, upper_bound = NullOpt;
      PrimExpr iter;
      if (is_greater) {
        if (bound_at_left) {
          // bound > iter / bound >= iter
          upper_bound = is_equal ? lhs_expr + 1 : lhs_expr;
          iter = rhs_expr;
        } else {
          // iter > bound / iter >= bound
          lower_bound = is_equal ? rhs_expr : rhs_expr + 1;
          iter = lhs_expr;
        }
      } else {
        if (bound_at_left) {
          // bound < iter / bound <= iter
          lower_bound = is_equal ? lhs_expr : lhs_expr + 1;
          iter = rhs_expr;
        } else {
          // iter < bound / iter <= bound
          upper_bound = is_equal ? rhs_expr + 1 : rhs_expr;
          iter = lhs_expr;
        }
      }
      // If it is a predicate for a single input iter
      if (auto opt = iter.as<Var>()) {
        auto var = opt.value();
        auto it = input_iters->find(var);
        if (it != input_iters->end()) {
          PrimExpr iter_min = (*it).second->min;
          PrimExpr iter_max = (*it).second->min + (*it).second->extent;
          if (lower_bound.defined()) iter_min = max(iter_min, lower_bound.value());
          if (upper_bound.defined()) iter_max = min(iter_max, upper_bound.value());
          input_iters->Set(var, Range(iter_min, iter_max));
        }
      } else {
        result->emplace_back(iter, lower_bound, upper_bound, 0);
      }
    }
    if (is_finish) {
      break;
    }
    pred = rest.Eval();
  }
  return true;
}

bool IterRangeSanityCheck(const Map<Var, Range>& iter_ranges) {
  std::unordered_set<Var> iters;
  for (const auto& it : iter_ranges) iters.insert(it.first);
  auto f = [&](const VarNode* var) { return iters.count(GetRef<Var>(var)); };
  for (const auto& it : iter_ranges) {
    if (UsesVar(it.second->min, f) || UsesVar(it.second->extent, f)) return false;
  }
  return true;
}

IterMapResult DetectIterMap(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                            const PrimExpr& predicate, IterMapLevel check_level,
                            arith::Analyzer* analyzer, bool simplify_trivial_iterators) {
  IterMapResult result;

  // Overall detection algorithm is divided into two steps:
  // - Step0: IterMapRewriter rewrites the expression to use IterMapExpr patterns.
  // - Step1: IterIndependenceChecker checks if the iterator are independent.
  if (!IterRangeSanityCheck(input_iters)) {
    result->errors.push_back("Invalid iterators.  Iterators may not be expressions of each other.");
    return result;
  }
  Map<Var, Range> constrained_input_iters = input_iters;
  std::vector<IterConstraint> constraints;
  if (!is_one(predicate) &&
      !MatchBoundConstraints(predicate, &constrained_input_iters, &constraints)) {
    result->errors.push_back("Could not parse predicate as constraints on the input iterators.");
    return result;
  }
  // We have to make sure when we visit an iterator, all the constraints related with its successors
  // in the iter var graph has been visited, where the expression of this iterator will contain the
  // expression of its successor, so we sort them by their sizes.
  for (IterConstraint& constraint : constraints) {
    constraint.expr_size = CalculateExprComplexity(constraint.iter);
  }

  std::sort(
      constraints.begin(), constraints.end(),
      [](const IterConstraint& a, const IterConstraint& b) { return a.expr_size < b.expr_size; });

  IterMapRewriter rewriter(analyzer, constrained_input_iters, check_level,
                           simplify_trivial_iterators, &result->errors);
  // Step0.0: rewrite constraints in the order from size-small ones to size-big ones
  for (const IterConstraint& constraint : constraints) {
    auto res = rewriter.RewriteIterConstraint(constraint.iter, constraint.lower_bound,
                                              constraint.upper_bound);
    if (result->errors.size() > 0) {
      return result;
    }
  }
  if (!rewriter.CheckConstraints()) {
    result->errors.push_back("Invalid constraints.");
    return result;
  }

  // Step0.1: Rewrite indicies and determine required padding,
  // if there is no padding, it should be the final result.
  Array<IterSumExpr> rewrite_indices;
  rewrite_indices.reserve(indices.size());
  bool allow_padding = check_level != IterMapLevel::Bijective;
  if (allow_padding) {
    for (PrimExpr value : indices) {
      rewrite_indices.push_back(rewriter.RewriteAndUpdatePadding(value));
      if (result->errors.size() > 0) {
        return result;
      }
    }
  }

  // Step0.2: Rewrite indices in the second round.
  if (!allow_padding || rewriter.requires_padding()) {
    rewrite_indices.clear();
    for (PrimExpr value : indices) {
      rewrite_indices.push_back(rewriter.Rewrite(value));
      if (result->errors.size() > 0) {
        return result;
      }
    }
  }
  result->padding_predicate = rewriter.padding_predicate();
  //

  // Step1: IterIndependenceChecker checks if the iterator are independent.
  if (!rewriter.CheckMapping(rewrite_indices, check_level)) {
    if (check_level == IterMapLevel::Bijective) {
      result->errors.push_back("Index mapping does not form a bijective transform.");
    } else {
      result->errors.push_back("Mapped indices are not independent.");
    }
    return result;
  }
  result->indices = rewrite_indices;
  return result;
}

TVM_REGISTER_GLOBAL("arith.DetectIterMap")
    .set_body_typed([](const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                       const PrimExpr& input_pred, int check_level,
                       bool simplify_trivial_iterators) {
      arith::Analyzer ana;
      return DetectIterMap(indices, input_iters, input_pred, IterMapLevel(check_level), &ana,
                           simplify_trivial_iterators);
    });

IterSumExpr NormalizeToIterSum(PrimExpr index, const Map<Var, Range>& input_iters,
                               arith::Analyzer* analyzer) {
  IterMapResult result;
  ICHECK(IterRangeSanityCheck(input_iters))
      << "Invalid iterators.  Iterators may not be expressions of each other.";

  // we skip constraint check as the most important thing here is only the pattern
  std::vector<IterConstraint> constraints;
  IterMapLevel check_level = IterMapLevel::NoCheck;
  bool simplify_trivial_iterators = true;
  IterMapRewriter rewriter(analyzer, input_iters, check_level, simplify_trivial_iterators,
                           &result->errors);

  return rewriter.RewriteToNormalizedIterSum(index);
}

TVM_REGISTER_GLOBAL("arith.NormalizeToIterSum")
    .set_body_typed([](PrimExpr index, const Map<Var, Range>& input_iters) {
      arith::Analyzer ana;
      return NormalizeToIterSum(index, input_iters, &ana);
    });

PrimExpr IterMapRewriter::VisitExpr_(const VarNode* op) {
  auto var = GetRef<Var>(op);
  auto it = var_map_.find(var);
  if (it != var_map_.end()) return it->second;
  return std::move(var);
}

PrimExpr IterMapRewriter::VisitExpr_(const AddNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }
  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Add>(a, b)) return const_res.value();
  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Add(a, b);
    }
  }

  // canonical form simplification.
  IterSumExpr ret = ToIterSumExpr(a);

  if (!b->IsInstance<IterMapExprNode>()) {
    ret.CopyOnWrite()->base += b;
  } else if (auto op = b.as<IterSumExpr>()) {
    AddToLhs(ret.CopyOnWrite(), op.value(), 1);
  } else if (auto op = b.as<IterSplitExpr>()) {
    AddToLhs(ret.CopyOnWrite(), op.value(), 1);
  } else {
    AddToLhs(ret.CopyOnWrite(), ToIterSumExpr(b), 1);
  }
  return std::move(ret);
}

PrimExpr IterMapRewriter::VisitExpr_(const SubNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Sub>(a, b)) return const_res.value();

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Sub(a, b);
    }
  }

  // canonical form simplification.
  IterSumExpr ret = ToIterSumExpr(a);

  if (!b->IsInstance<IterMapExprNode>()) {
    ret.CopyOnWrite()->base -= b;
  } else if (auto op = b.as<IterSumExpr>()) {
    AddToLhs(ret.CopyOnWrite(), op.value(), -1);
  } else if (auto op = b.as<IterSplitExpr>()) {
    AddToLhs(ret.CopyOnWrite(), op.value(), -1);
  } else {
    AddToLhs(ret.CopyOnWrite(), ToIterSumExpr(b), -1);
  }
  return std::move(ret);
}

PrimExpr IterMapRewriter::VisitExpr_(const MulNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Mul>(a, b)) return const_res.value();

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Mul(a, b);
    }
  }

  if (a->IsInstance<IterMapExprNode>() && b->IsInstance<IterMapExprNode>()) {
    // cannot multiply two iterators, mark as unresolved.
    ErrorLogger(this) << "Product of two iterators cannot be represented as an IterMap, "
                      << "occurs in " << GetRef<Mul>(op);
    return GetRef<PrimExpr>(op);
  }

  if (!a->IsInstance<IterMapExprNode>()) {
    std::swap(a, b);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(std::move(a));
    MulToLhs(ret.CopyOnWrite(), b);
    return std::move(ret);
  } else {
    ICHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    ret.CopyOnWrite()->scale *= b;
    return std::move(ret);
  }
}

IterSumExpr IterMapRewriter::PreprocessDividend(IterMapExpr dividend, PrimExpr original_dividend) {
  if (dividend->IsInstance<IterSplitExprNode>()) {
    auto split = Downcast<IterSplitExpr>(dividend);
    return IterSumExpr({split}, make_zero(split.dtype()));
  } else if (dividend->IsInstance<IterSumExprNode>()) {
    auto sum = Downcast<IterSumExpr>(dividend);
    if (sum->args.empty()) {
      return IterSumExpr();
    } else if (sum->args.size() == 1) {
      return sum;
    }
    auto opt_fused = TryFuseIters(sum, check_level_, true);
    if (!opt_fused) {
      ErrorLogger(this) << "Dividend  " << original_dividend
                        << ", can't be written as a single fused IterSum";
      return IterSumExpr();
    }
    IterSumExpr fused = opt_fused.value();
    ICHECK_EQ(fused->args.size(), 1U);
    return fused;
  } else {
    LOG(FATAL) << "Unsupported subclass of IterMarkExpr";
  }
}

/*! \brief Find approximate least common multiplier. */
PrimExpr ApproxLeastCommonMultiple(const PrimExpr& a, const PrimExpr& b, Analyzer* analyzer) {
  auto fsplit = [](const PrimExpr& e) -> std::pair<PrimExpr, int64_t> {
    if (const IntImmNode* imm = e.as<IntImmNode>()) {
      return {1, imm->value};
    }
    PVar<PrimExpr> pv;
    PVar<IntImm> pc;
    if ((pv * pc).Match(e) || (pc * pv).Match(e)) {
      return {pv.Eval(), pc.Eval()->value};
    } else {
      return {e, 1};
    }
  };
  auto p1 = fsplit(a);
  auto p2 = fsplit(b);
  auto const_lcm = Integer(LeastCommonMultiple(p1.second, p2.second));
  if (analyzer->CanProveEqual(p1.first, p2.first)) {
    return p1.first * const_lcm;
  } else if (analyzer->CanProveEqual(floormod(p1.first, p2.first), 0)) {
    return p1.first * const_lcm;
  } else if (analyzer->CanProveEqual(floormod(p2.first, p1.first), 0)) {
    return p2.first * const_lcm;
  } else {
    return (p1.first * p2.first) * const_lcm;
  }
}

std::pair<IterSplitExpr, PrimExpr> IterMapRewriter::PadDividendToDivisor(IterSplitExpr split,
                                                                         PrimExpr base,
                                                                         PrimExpr divisor) {
  // If FloorDiv: (((source//lower_factor) % extent) + base) // divisor
  // If FloorMod: (((source//lower_factor) % extent) + base) % divisor

  // First, adding any padding that is on the lower side of a
  // FloorDiv/FloorMod, such that floormod(split - left_pad, divisor) == 0
  // when iter == 0.
  PrimExpr left_pad = analyzer_->Simplify(floormod(base, divisor));

  // Next, adding any padding that is on the upper side of a
  // FloorDiv/FloorMod, such that floormod(left_pad + split + right_pad, divisor) == 0
  // when iter == extent.
  PrimExpr right_edge = left_pad + split->extent;
  PrimExpr right_pad;
  if (CanProveDivisible(right_edge, divisor)) {
    right_pad = 0;
  } else {
    right_pad = analyzer_->Simplify(floormod(-right_edge, divisor));
  }

  const IterMark& mark = split->source;
  if (update_iterator_padding_) {
    // In the first pass, the primary goal is to collect all the divisors
    // that may be used for padding. These will impact the divisor used
    // to determine padding in the second pass. We try add padding to
    // split's source iteraton mark thus all splits under the same mark will
    // share the same padded source iteration.
    auto& info = padded_iter_map_[mark];
    info.padding_factor =
        ApproxLeastCommonMultiple(info.padding_factor, divisor * split->lower_factor, analyzer_);

    // If the split itself require no padding, return directly.
    if (is_zero(left_pad) && is_zero(right_pad)) {
      return {split, 0};
    }

    // Update padding requirement on the lower side of the source iter mark.
    // In the second pass, all splits would check whether the maximum left pading
    // on the iter mark is compatible with it's own left padding.
    requires_padding_ = true;
    PrimExpr mark_left_pad = left_pad * split->lower_factor;
    info.left_pad = max(info.left_pad, mark_left_pad);

    // Since we only care the extent in the first pass's result
    // we just create result of compatible padded extent, ignoring
    // possible relations between different padded iters.
    PrimExpr padded_extent = analyzer_->Simplify(left_pad + split->extent + right_pad);
    split.CopyOnWrite()->extent = padded_extent;
    return {split, left_pad};
  }

  // In the second pass, update iteration mark's to padded form
  auto it = padded_iter_map_.find(mark);
  if (it == padded_iter_map_.end()) {
    return {split, left_pad};
  }
  auto& info = it->second;
  if (is_zero(info.left_pad) && CanProveDivisible(mark->extent, info.padding_factor)) {
    // the iter mark requires no padding
    return {split, left_pad};
  }

  // check that padding factor is compatible with current split and divisor
  ICHECK(CanProveDivisible(info.padding_factor, split->lower_factor))
      << "The padding factor " << info.padding_factor << " is not divisible by "
      << split->lower_factor << " for the split " << split;
  ICHECK(CanProveDivisible(info.padding_factor, divisor))
      << "The padding factor " << info.padding_factor << " is not divisible by " << divisor
      << " for the split " << split;

  if (!info.padded.defined()) {
    // the first time encounter the iter mark to pad, update the padded mark.
    PrimExpr mark_left_pad = info.left_pad;
    if (CanProveDivisible(mark_left_pad, split->lower_factor)) {
      // correct current split's left padding
      // (mark_left_pad + iter) // lower_factor % extent  =>
      // (left_pad * lower_factor + mark) // lower_factor % extent =>
      // (left_pad + mark // lower_factor) % extent =>
      // left_pad + (mark // lower_factor % extent) =>
      // left_pad + split
      // since the extent covers the full padding range.
      left_pad = floordiv(mark_left_pad, split->lower_factor);
    } else {
      ErrorLogger(this) << "Detect incompatible left padding on " << NormalizeIterMapToExpr(split)
                        << ", the iter mark is left padded with " << mark_left_pad;
      return {IterSplitExpr(), PrimExpr()};
    }

    PrimExpr right_edge = mark->extent + mark_left_pad;
    PrimExpr mark_right_pad;
    if (CanProveDivisible(right_edge, info.padding_factor)) {
      mark_right_pad = 0;
    } else {
      mark_right_pad = floormod(-right_edge, info.padding_factor);
    }
    PrimExpr padded_extent = analyzer_->Simplify(right_edge + mark_right_pad);
    info.right_pad = mark_right_pad;
    info.padded = IterMark(IterSumExpr({IterSplitExpr(mark)}, mark_left_pad), padded_extent);
    padded_origin_map_[info.padded] = mark;

    auto left_padding_introduced = (mark_left_pad != 0);

    // Equivalent to (0 <= split < left_pad), but easier to simplify in
    // terms of the transformed variables.
    auto left_padding_predicate =
        left_padding_introduced &&
        (floordiv(info.padded->source, info.padding_factor) == 0 &&
         floormod(info.padded->source, info.padding_factor) < mark_left_pad);
    auto right_padding_introduced = (mark_right_pad != 0);

    // Equivalent to (right_edge <= split < right_edge + right_pad), but
    // easier to simplify in terms of the transformed variables.
    auto right_padding_predicate =
        right_padding_introduced && (floordiv(info.padded->source, info.padding_factor) ==
                                         floordiv(right_edge, info.padding_factor) &&
                                     floormod(info.padded->source, info.padding_factor) >=
                                         floormod(right_edge, info.padding_factor));
    padding_predicate_ = padding_predicate_ || (left_padding_predicate || right_padding_predicate);
  }
  split.CopyOnWrite()->source = info.padded;
  split.CopyOnWrite()->extent = floordiv(info.padded->extent, split->lower_factor);
  return {split, left_pad};
}

PrimExpr IterMapRewriter::SplitFloorDivConst(IterSplitExpr lhs, PrimExpr base, PrimExpr rhs) {
  // (lhs + base) // rhs

  if (is_one(rhs)) {
    if (is_zero(base)) {
      // floordiv(x, 1) = x
      return std::move(lhs);
    } else {
      // floordiv(x+y, 1) = x+y
      return IterSumExpr({lhs}, base);
    }
  }

  if (!is_one(lhs->scale)) {
    if (CanProveDivisible(lhs->scale, rhs) && is_zero(base)) {
      // floordiv(x*c1*c2, c2) = x*c1, c1=scale/rhs
      lhs.CopyOnWrite()->scale = floordiv(lhs->scale, rhs);
      return std::move(lhs);
    } else if (CanProveDivisible(lhs->scale, rhs) && CanProveDivisible(base, rhs)) {
      // floordiv(x*c1*c2 + y*c2, c2) = x*c1 + y, c1=scale/rhs
      lhs.CopyOnWrite()->scale = floordiv(lhs->scale, rhs);
      return IterSumExpr({lhs}, floordiv(base, rhs));
    } else if (CanProveDivisible(rhs, lhs->scale) && is_zero(base)) {
      // floordiv(x*c1, c1*c2) = floordiv(x, c2), c2=rhs/scale
      rhs = floordiv(rhs, lhs->scale);
      lhs.CopyOnWrite()->scale = make_const(rhs->dtype, 1);
    } else if (CanProveDivisible(rhs, lhs->scale) && CanProveDivisible(base, lhs->scale)) {
      // floordiv(x*c1 + y*c1, c1*c2) = floordiv(x+y, c2), c2=rhs/scale
      base = floordiv(base, lhs->scale);
      rhs = floordiv(rhs, lhs->scale);
      lhs.CopyOnWrite()->scale = make_const(rhs->dtype, 1);
    } else {
      // mark as unresolved.
      ErrorLogger(this) << "Cannot represent as IterMap: the numerator's scaling factor, "
                        << lhs->scale << " and the divisor " << rhs
                        << " cannot be simplified to remove the scaling factor.";
      return PrimExpr();
    }
  }

  // We handle scale!=1 in above code, hence we only consider floordiv(x, rhs) below
  // where x=floormod(floordiv(iter, lower_factor), extent) + base

  auto pair = PadDividendToDivisor(lhs, base, rhs);
  IterSplitExpr padded = pair.first;
  PrimExpr left_pad = pair.second;
  if (!padded.defined()) {
    return PrimExpr();
  }

  // floordiv(floormod(floordiv(iter, lower_factor), c1c2), c1)
  // = floordiv(floormod(y, c1c2), c1), where y=floordiv(iter, lower_factor)
  // = floordiv(floormod(sc1c2+tc1+u, c1c2), c1), where y=sc1c2+tc1+u, t<c2, u<c1
  // = t
  // = floormod(sc2+t, c2)
  // = floormod(floordiv(y, c1), c2)
  // = floormod(floordiv(iter, lower_factor*c1), c2), where c1=rhs, c2=extent/rhs
  IterSplitExpr new_split;
  if (CanProveDivisible(padded->extent, rhs)) {
    new_split = IterSplitExpr(padded->source,
                              /* lower_factor = */ padded->lower_factor * rhs,
                              /* extent = */ analyzer_->Simplify(floordiv(padded->extent, rhs)),
                              /* scale = */ padded->scale);
  } else if (is_one(padded->lower_factor) &&
             analyzer_->CanProveEqual(padded->extent, padded->source->extent)) {
    // floordiv(floormod(floordiv(iter, lower_factor), ext), c)
    // = floordiv(iter, c)
    // when lower_factor = 1 and ext = iter.extent
    new_split = IterSplitExpr(padded->source,
                              /* lower_factor = */ rhs,
                              /* extent = */ analyzer_->Simplify(ceildiv(padded->extent, rhs)),
                              /* scale = */ padded->scale);
  } else {
    new_split = IterSplitExpr(IterMark(padded, padded->extent),
                              /* lower_factor = */ rhs,
                              /* extent = */ analyzer_->Simplify(ceildiv(padded->extent, rhs)),
                              /* scale = */ make_const(rhs->dtype, 1));
  }

  auto new_base = analyzer_->Simplify(floordiv(base - left_pad, rhs), 6);
  if (is_zero(new_base)) {
    return std::move(new_split);
  } else {
    return IterSumExpr({new_split}, new_base);
  }
}

PrimExpr IterMapRewriter::VisitExpr_(const FloorDivNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<FloorDiv>(a, b)) return const_res.value();

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return FloorDiv(a, b);
    }
  }

  if (b->IsInstance<IterMapExprNode>()) {
    // cannot divide an iterator, mark as unresolved.
    ErrorLogger(this) << "Cannot represent as an IterMap: the divisor in " << GetRef<PrimExpr>(op)
                      << " may not be an iterator";
    return GetRef<PrimExpr>(op);
  }

  IterSumExpr preprocessed = PreprocessDividend(Downcast<IterMapExpr>(a), op->a);
  if (!preprocessed.defined()) {
    return GetRef<PrimExpr>(op);
  }
  ICHECK_EQ(preprocessed->args.size(), 1U);
  PrimExpr remainder = SplitFloorDivConst(preprocessed->args[0], preprocessed->base, b);
  if (!remainder.defined()) {
    return GetRef<PrimExpr>(op);
  }
  return remainder;
}

PrimExpr IterMapRewriter::SplitFloorModConst(IterSplitExpr lhs, PrimExpr base, PrimExpr rhs) {
  // (lhs + base) % rhs

  if (is_one(rhs)) {
    // floormod(x, 1) = 0
    return make_zero(lhs->dtype);
  }

  if (!is_one(lhs->scale)) {
    if (CanProveDivisible(lhs->scale, rhs) && CanProveDivisible(base, rhs)) {
      // floormod(x*c1*c2, c1) = 0
      return make_zero(lhs->dtype);
    } else if (CanProveDivisible(rhs, lhs->scale) && is_zero(base)) {
      // floormod(x*c1, c1*c2) = (floormod(x, c2)) * c1, where c2 = rhs/scale
      rhs = floordiv(rhs, lhs->scale);
    } else if (CanProveDivisible(rhs, lhs->scale) && CanProveDivisible(base, lhs->scale)) {
      // floormod(x*c1 + y*c1, c1*c2) = (floormod(x+y, c2)) * c1, where c2 = rhs/scale
      rhs = floordiv(rhs, lhs->scale);
      base = floordiv(base, lhs->scale);
    } else {
      // mark as unresolved.
      ErrorLogger(this)
          << "Cannot represent as IterMap: the left-hand side of FloorMod has a scaling factor, "
          << lhs->scale << " and the right-hand " << rhs
          << " cannot be used to simplify out the scaling factor.";
      return PrimExpr();
    }
  }

  // We handle scale!=1 in above code, hence we only consider floormod(x, rhs) below
  // where x=floormod(floordiv(iter, lower_factor), extent) + base
  auto pair = PadDividendToDivisor(lhs, base, rhs);
  IterSplitExpr padded = pair.first;
  if (!padded.defined()) {
    return PrimExpr();
  }

  // floormod(floormod(floordiv(iter, lower_factor), c1c2), c1)
  // = floormod(floordiv(iter, lower_factor), c1), where c1=rhs
  return IterSplitExpr(padded->source,
                       /* lower_factor = */ padded->lower_factor,
                       /* extent = */ rhs,
                       /* scale = */ padded->scale);
}

PrimExpr IterMapRewriter::VisitExpr_(const FloorModNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<FloorMod>(a, b)) return const_res.value();

  // does not contain iter map.
  if (!a->IsInstance<IterMapExprNode>() && !b->IsInstance<IterMapExprNode>()) {
    if (op->a.same_as(a) && op->b.same_as(b)) {
      return GetRef<PrimExpr>(op);
    } else {
      return FloorMod(a, b);
    }
  }

  if (b->IsInstance<IterMapExprNode>()) {
    // cannot mod an iterator, mark as unresolved.
    ErrorLogger(this) << "Cannot represent as an IterMap: the right-hand side of FloorMod in "
                      << GetRef<PrimExpr>(op) << " may not be an iterator";
    return GetRef<PrimExpr>(op);
  }

  IterSumExpr preprocessed = PreprocessDividend(Downcast<IterMapExpr>(a), op->a);
  if (!preprocessed.defined()) {
    return GetRef<PrimExpr>(op);
  }

  ICHECK_EQ(preprocessed->args.size(), 1U);
  PrimExpr remainder = SplitFloorModConst(preprocessed->args[0], preprocessed->base, b);
  if (!remainder.defined()) {
    return GetRef<PrimExpr>(op);
  }
  return remainder;
}

/*! * \brief Given an expression that may contain IterVarMapExpr, transform it to normal PrimExpr.
 */
class IterMapToExprNormalizer : public ExprMutator {
 public:
  explicit IterMapToExprNormalizer(Analyzer* analyzer) : analyzer_(analyzer) {}

  PrimExpr Convert(const PrimExpr& expr) { return VisitExpr(expr); }

 private:
  /*! \brief Override VisitExpr for iter expr type processing */
  PrimExpr VisitExpr(const PrimExpr& expr) override {
    if (auto op = expr.as<IterSplitExpr>()) {
      return ConvertIterSplitExpr(op.value());
    } else if (auto op = expr.as<IterSumExpr>()) {
      return ConvertIterSumExpr(op.value());
    } else {
      return ExprMutator::VisitExpr(expr);
    }
  }

  PrimExpr ConvertIterSumExpr(const IterSumExpr& expr) {
    PrimExpr res = 0;
    for (const IterSplitExpr& arg : expr->args) {
      res += ConvertIterSplitExpr(arg);
    }
    res += expr->base;
    return res;
  }

  PrimExpr ConvertIterSplitExpr(const IterSplitExpr& expr) {
    PrimExpr source;
    if (auto opt = expr->source->source.as<Var>()) {
      source = opt.value();
    } else if (auto opt = expr->source->source.as<IterSumExpr>()) {
      source = ConvertIterSumExpr(opt.value());
    } else {
      source = VisitExpr(expr->source->source);
    }
    if (analyzer_->CanProve(expr->extent == expr->source->extent) && is_one(expr->lower_factor)) {
      return source * expr->scale;
    } else if (analyzer_->CanProve(expr->source->extent == expr->lower_factor * expr->extent)) {
      // Simplify if `expr` is always 0. The 2nd condition guarantess that we do not aggressively
      // simplify trivial iters like `vi \in [0, 1)`, which can be useful for subsequent analysis
      // like tensorization.
      if (is_one(expr->extent) && !is_one(expr->source->extent)) {
        return make_const(expr->extent->dtype, 0);
      }
      return floordiv(source, expr->lower_factor) * expr->scale;
    } else {
      return floordiv(floormod(source, expr->lower_factor * expr->extent), expr->lower_factor) *
             expr->scale;
    }
  }

 private:
  Analyzer* analyzer_;
};

bool IterMapRewriter::CanProveDivisible(const PrimExpr& lhs, const PrimExpr& rhs) {
  const auto* clhs = lhs.as<IntImmNode>();
  const auto* crhs = rhs.as<IntImmNode>();
  if (crhs && crhs->value == 0) {
    return false;
  } else if (clhs && crhs) {
    return clhs->value % crhs->value == 0;
  }

  IterMapToExprNormalizer normalizer(analyzer_);
  PrimExpr dividend = normalizer.Convert(lhs);
  PrimExpr divisor = normalizer.Convert(rhs);

  return analyzer_->CanProveEqual(dividend, divisor) ||
         analyzer_->CanProve(floormod(dividend, divisor) == 0);
}

PrimExpr NormalizeIterMapToExpr(const PrimExpr& expr) {
  arith::Analyzer analyzer;
  IterMapToExprNormalizer normalizer(&analyzer);
  return normalizer.Convert(expr);
}

TVM_REGISTER_GLOBAL("arith.NormalizeIterMapToExpr").set_body_typed(NormalizeIterMapToExpr);

Array<PrimExpr> IterMapSimplify(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                                const PrimExpr& input_pred, IterMapLevel check_level,
                                arith::Analyzer* ana, bool simplify_trivial_iterators) {
  if (!IterRangeSanityCheck(input_iters)) return indices;
  auto res = DetectIterMap(indices, input_iters, input_pred, check_level, ana,
                           /*simplify_trivial_iterators=*/simplify_trivial_iterators);
  Array<IterSumExpr> rewrite = res->indices;

  if (rewrite.empty() && !is_one(input_pred) && check_level != IterMapLevel::Bijective) {
    // The input predicate may cause detect iter map to fail
    // but we can still detect the iter map without the input predicate
    // in which case the resulting iter map is valid and can be used for simplification.
    rewrite = DetectIterMap(indices, input_iters, const_true(), check_level, ana,
                            /*simplify_trivial_iterators=*/simplify_trivial_iterators)
                  ->indices;
  }

  if (rewrite.empty()) {
    return indices;
  }
  Array<PrimExpr> simplified;
  simplified.reserve(rewrite.size());
  IterMapToExprNormalizer converter(ana);
  for (const auto& expr : rewrite) simplified.push_back(converter.Convert(expr));
  return simplified;
}

TVM_REGISTER_GLOBAL("arith.IterMapSimplify")
    .set_body_typed([](const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                       const PrimExpr& input_pred, int check_level,
                       bool simplify_trivial_iterators) {
      arith::Analyzer ana;
      return IterMapSimplify(indices, input_iters, input_pred, IterMapLevel(check_level), &ana,
                             simplify_trivial_iterators);
    });

/*!
 * \brief Divider to divide the bindings into two sets of bindings(outer and inner)
 *   such that binding_i = Y_i * E(Xi) + Xi, where E(X) is the extent of X.
 *   We do message passing among IterSplitExpr and IterSumExpr.
 *
 *   Example
 *   - If we encounter sum = i*10 + j*5 + k, and i, j, k are splits,
 *     and we know i = Yi*1 + 0, j = 0*E(Xj) + Xj, k = 0*E(Xk) + Xk through message passing,
 *     then sum = Yi*10 + (Xj*5 + Xk) = Y*E(X) + X, where Y = Yi, X = Xj*5 + Xk.
 *   - If we encounter split = (i / 2) % 4, and we know i = Y*E(X) + X through message passing.
 *     We inspect all the splits of i, which are i / 8, (i / 2) % 4, i % 2.
 *     Their extents are 2, 4, 2, if E(X) = 2, 8, 16, the splits can be divided.
 */
class SubspaceDivider {
 public:
  explicit SubspaceDivider(Analyzer* analyzer, const IterMarkSplitCollector& collector,
                           const std::unordered_set<Var>& sub_iters)
      : analyzer_(analyzer), collector_(collector), sub_iters_(sub_iters) {}

  size_t unresolved_count() const { return unresolved_count_; }

  // Denotes outer*inner_extent + inner, used as message passing carrier
  struct DivisionResult {
   public:
    // IterMapExpr of outer iters
    IterMapExpr outer;
    // IterMapExpr of inner iters
    IterMapExpr inner;
    // extent of outer
    PrimExpr outer_extent;
    // extent of inner
    PrimExpr inner_extent;

    // The kind of the division result.
    enum class Kind {
      kInner,  // Indicates the division result is totally in inner subspace.
      kOuter,  // Indicates the division result is totally in outer subspace.
      kMixed,  // Indicates the division result is mixed in both subspace.
    } kind;

    DivisionResult(IterMapExpr outer, PrimExpr outer_extent, IterMapExpr inner,
                   PrimExpr inner_extent, Kind kind = Kind::kMixed)
        : outer(std::move(outer)),
          inner(std::move(inner)),
          outer_extent(std::move(outer_extent)),
          inner_extent(std::move(inner_extent)),
          kind(kind) {}

    // whether the division result is totally in outer subspace
    bool IsOuter() const { return kind == Kind::kOuter; }

    // whether the division result is totally in inner subspace
    bool IsInner() const { return kind == Kind::kInner; }

    IterSplitExpr GetOuterAsSplit() const { return GetAsSplit(outer, outer_extent); }

    IterSplitExpr GetInnerAsSplit() const { return GetAsSplit(inner, inner_extent); }

    static DivisionResult Inner(const IterMapExpr& iter, const PrimExpr& extent) {
      auto dtype = iter.dtype();
      return DivisionResult(IterSumExpr({}, make_const(dtype, 0)), make_const(dtype, 1), iter,
                            extent, Kind::kInner);
    }

    static DivisionResult Outer(const IterMapExpr& iter, const PrimExpr& extent) {
      auto dtype = iter.dtype();
      return DivisionResult(iter, extent, IterSumExpr({}, make_const(dtype, 0)),
                            make_const(dtype, 1), Kind::kOuter);
    }

    // Special value to indicate the division is not possible
    static DivisionResult Failure() {
      return DivisionResult(IterSumExpr({}, 0), 0, IterSumExpr({}, 0), 0);
    }

   private:
    static IterSplitExpr GetAsSplit(const IterMapExpr& expr, const PrimExpr& extent) {
      if (auto op = expr.as<IterSplitExpr>()) {
        return op.value();
      } else if (auto op = expr.as<IterSumExpr>()) {
        return IterSplitExpr(IterMark(op.value(), extent));
      } else {
        LOG(FATAL) << "Unknown IterMapExpr type";
      }
    }
  };

  // Divide an IterSumExpr
  DivisionResult DivideIterSumExpr(const IterSumExpr& expr, const PrimExpr& mark_extent) {
    auto dtype = expr.dtype();
    if (expr->args.empty()) {
      // base
      return DivisionResult(IterSumExpr({}, make_const(dtype, 0)), make_const(dtype, 1),
                            IterSumExpr({}, expr->base), make_const(dtype, 1));
    } else if (expr->args.size() == 1) {
      // arg + base, if arg=Y*E(X)+X, then arg+base = Y*E(X)+(X+base)
      if (!is_one(expr->args[0]->scale)) {
        unresolved_count_++;
        return DivisionResult::Failure();
      }
      DivisionResult res = DivideIterSplitExpr(expr->args[0]);
      if (!is_zero(expr->base)) res = AddBase(res, expr->base);
      return res;
    }
    // arg1 + arg2 + ... + argn + base
    // then we can write it as Y*E(X)+X
    // if it starts with contiguous outer splits, followed by contiguous inner splits
    PrimExpr extent = make_const(dtype, 1);
    std::vector<IterSplitExpr> outer_args, inner_args;
    bool inner = true, scale_is_one = false;
    // we check in inverse order so we can visit from inner to outer
    for (auto it = expr->args.rbegin(); it != expr->args.rend(); ++it) {
      const IterSplitExpr& arg = *it;
      if (is_one(arg->scale)) scale_is_one = true;
      DivisionResult arg_division = DivideIterSplitExpr(arg);
      IterSplitExpr new_arg;
      if (arg_division.IsInner()) {
        if (!inner) {
          unresolved_count_++;
          return DivisionResult::Failure();
        }
        new_arg = arg_division.GetInnerAsSplit();
        inner_args.push_back(new_arg);
        inner = true;
      } else if (arg_division.IsOuter()) {
        new_arg = arg_division.GetOuterAsSplit();
        outer_args.push_back(new_arg);
        inner = false;
      } else {
        unresolved_count_++;
        return DivisionResult::Failure();
      }
      extent *= new_arg->extent;
    }
    if (!scale_is_one) {
      unresolved_count_++;
      return DivisionResult::Failure();
    }
    bool need_predicate = !analyzer_->CanProveEqual(extent, mark_extent);
    const IterMark& outer_mark = MarkFromArgsAndBase(outer_args, make_const(dtype, 0));
    const IterMark& inner_mark = MarkFromArgsAndBase(inner_args, expr->base);
    IterSumExpr outer_source = Downcast<IterSumExpr>(outer_mark->source);
    IterSumExpr inner_source = Downcast<IterSumExpr>(inner_mark->source);
    if (need_predicate) {
      // if we have a predicate on this sum expr, then we cannot divide it into Y*E+X
      // it should either be Y*1+0 or 0*E(X)+X
      IterMapToExprNormalizer converter(analyzer_);
      if (inner_args.empty()) {
        // Y*1+0
        outer_preds_ = outer_preds_ && (converter.Convert(outer_source) < mark_extent);
        return DivisionResult::Outer(outer_source, mark_extent);
      } else if (outer_args.empty()) {
        // 0*E(X)+X
        inner_preds_ = inner_preds_ && (converter.Convert(inner_source) < mark_extent);
        return DivisionResult::Inner(inner_source, mark_extent);
      } else {
        unresolved_count_++;
        return DivisionResult::Failure();
      }
    }
    return DivisionResult(outer_source, outer_mark->extent, inner_source, inner_mark->extent);
  }

  PrimExpr GetOuterPreds() const { return outer_preds_; }
  PrimExpr GetInnerPreds() const { return inner_preds_; }

 private:
  DivisionResult AddBase(DivisionResult division, PrimExpr base) {
    DivisionResult res = division;
    if (auto op = division.inner.as<IterSplitExpr>()) {
      res.inner = IterSumExpr({op.value()}, base);
    } else if (auto op = division.inner.as<IterSumExpr>()) {
      const auto& expr = op.value();
      res.inner = IterSumExpr(expr->args, expr->base + base);
    }
    return res;
  }

  // args are sorted from inner to outer
  static IterMark MarkFromArgsAndBase(const std::vector<IterSplitExpr>& args, PrimExpr base) {
    std::vector<IterSplitExpr> res;
    PrimExpr extent = make_const(base.dtype(), 1);
    for (const IterSplitExpr& it : args) {
      IterSplitExpr arg = it;
      arg.CopyOnWrite()->scale = extent;
      extent *= arg->extent;
      res.push_back(arg);
    }
    return IterMark(IterSumExpr(Array<IterSplitExpr>(res.rbegin(), res.rend()), base), extent);
  }

  DivisionResult DivideIterSplitExpr(const IterSplitExpr& expr) {
    auto it = split_map_.find(expr);
    if (it != split_map_.end()) {
      // We will calculate all the splits of an IterMark's division form when we first
      // encounter one of them. If we encounter another later, we directly return the record.
      return it->second;
    }
    const Array<IterSplitExpr>& splits = collector_.mark2splits_.at(expr->source);
    if (auto iter_ptr = expr->source->source.as<Var>()) {
      // source is input_iter
      bool inner = sub_iters_.count(iter_ptr.value());
      for (const IterSplitExpr& split : splits) {
        if (inner) {
          // 0*E(split)+split
          split_map_.emplace(split, DivisionResult::Inner(split, split->extent));
        } else {
          // split*1 + 0
          split_map_.emplace(split, DivisionResult::Outer(split, split->extent));
        }
      }
    } else if (auto iter_ptr = expr->source->source.as<IterSumExpr>()) {
      // source = Y*E+X
      // splits = [s1, s2, ..., sn]
      // we can divide if there exists i, such that extent(s1)extent(s2)...extent(si)=extent(Y)
      //                                            extent(si+1)...extent(sn)=extent(X)
      // For example, if source = Y*3+X \in [0, 12), Y \in [0, 4), X \in [0, 3)
      // Case 1. splits = [s1, s2, s3] = [source / 6, (source / 3) % 2, source % 3],
      //         where extent(s1) = 2, extent(s2) = 2, extent(s3) = 3.
      //         Since extent(s1)extent(s2) = extent(Y), extent(s3) = extent(X), we have
      //         s1 = (Y / 2)*1 + 0, s2 = (Y % 2)*1 + 0, s3 = 0*3 + X
      // Case 2. splits = [s1, s2, s3] = [source / 4, (source / 2) % 2, source % 2],
      //         where extent(s1) = 3, extent(s2) = 2, extent(s3) = 2.
      //         It's impossible to rewrite s1, s2, s3 in the form of Y*E(X) + X.
      DivisionResult mark_division = DivideIterSumExpr(iter_ptr.value(), expr->source->extent);
      if (splits.size() == 1) {
        return mark_division;
      }
      IterMark outer_mark(Downcast<IterSumExpr>(mark_division.outer), mark_division.outer_extent);
      IterMark inner_mark(Downcast<IterSumExpr>(mark_division.inner), mark_division.inner_extent);
      bool encountered_boundary = mark_division.IsOuter();
      std::vector<bool> used(splits.size(), false);
      std::vector<IterSplitExpr> inner_iters, outer_iters;
      PrimExpr expected_lower_factor = make_const(expr->source->source->dtype, 1);
      // find the boundary of outer and inner, like case 1 above
      for (size_t i = 0; i < splits.size(); ++i) {
        size_t j = 0;
        for (; j < splits.size(); ++j) {
          if (!used[j] && analyzer_->CanProveEqual(splits[j]->lower_factor, expected_lower_factor))
            break;
        }
        if (j == splits.size()) {
          unresolved_count_++;
          return DivisionResult::Failure();
        }
        used[j] = true;
        if (!encountered_boundary) {
          inner_iters.push_back(splits[j]);
        } else {
          outer_iters.push_back(splits[j]);
        }
        expected_lower_factor *= splits[j]->extent;
        if (analyzer_->CanProveEqual(expected_lower_factor, mark_division.inner_extent))
          encountered_boundary = true;
      }
      if (!encountered_boundary) {
        unresolved_count_++;
        return DivisionResult::Failure();
      }
      for (const IterSplitExpr& inner_iter : inner_iters) {
        IterSplitExpr new_iter = inner_iter;
        new_iter.CopyOnWrite()->source = inner_mark;
        split_map_.emplace(inner_iter, DivisionResult::Inner(new_iter, inner_iter->extent));
      }
      for (const IterSplitExpr& outer_iter : outer_iters) {
        IterSplitExpr new_iter = outer_iter;
        new_iter.CopyOnWrite()->source = outer_mark;
        new_iter.CopyOnWrite()->lower_factor =
            floordiv(outer_iter->lower_factor, outer_iters[0]->lower_factor);
        split_map_.emplace(outer_iter, DivisionResult::Outer(new_iter, outer_iter->extent));
      }
    } else {
      unresolved_count_++;
      return DivisionResult::Failure();
    }
    return split_map_.at(expr);
  }

  size_t unresolved_count_{0};
  // arithmetic analyzer used to call CanProve
  Analyzer* analyzer_;
  // collector that collects the outgoing split reference of each IterMark
  const IterMarkSplitCollector collector_;
  // the set of subspace iters
  const std::unordered_set<Var>& sub_iters_;
  // map from SplitExpr to its corresponding DivisionResult(Y*E(X)+X)
  std::unordered_map<IterSplitExpr, DivisionResult, ObjectPtrHash, ObjectPtrEqual> split_map_;
  // predicate of outer space and inner space;
  PrimExpr outer_preds_{Bool(true)}, inner_preds_{Bool(true)};
};

Array<Array<IterMark>> SubspaceDivide(const Array<PrimExpr>& bindings,
                                      const Map<Var, Range>& input_iters,
                                      const Array<Var>& sub_iters, const PrimExpr& predicate,
                                      IterMapLevel check_level, arith::Analyzer* analyzer,
                                      bool simplify_trivial_iterators) {
  if (!IterRangeSanityCheck(input_iters)) return Array<Array<IterMark>>();
  auto res = DetectIterMap(bindings, input_iters, predicate, check_level, analyzer,
                           simplify_trivial_iterators);
  const Array<IterSumExpr>& maps = res->indices;
  if (maps.empty()) return {};

  std::unordered_set<Var> inner_iter_set;
  for (const Var& inner_iter : sub_iters) {
    inner_iter_set.insert(inner_iter);
  }

  IterMarkSplitCollector collector;
  collector.Collect(maps);
  SubspaceDivider subspace_divider(analyzer, collector, inner_iter_set);

  std::vector<Array<IterMark>> results;
  for (const IterSumExpr& expr : maps) {
    SubspaceDivider::DivisionResult res = subspace_divider.DivideIterSumExpr(expr, 0);
    if (subspace_divider.unresolved_count()) return {};
    results.push_back(
        {IterMark(res.outer, res.outer_extent), IterMark(res.inner, res.inner_extent)});
  }

  results.push_back({IterMark(IterSumExpr({}, 0), subspace_divider.GetOuterPreds()),
                     IterMark(IterSumExpr({}, 0), subspace_divider.GetInnerPreds())});
  return results;
}

TVM_REGISTER_GLOBAL("arith.SubspaceDivide")
    .set_body_typed([](const Array<PrimExpr>& bindings, const Map<Var, Range>& root_iters,
                       const Array<Var>& sub_iters, const PrimExpr& predicate, int check_level,
                       bool simplify_trivial_iterators) {
      arith::Analyzer ana;
      return SubspaceDivide(bindings, root_iters, sub_iters, predicate, IterMapLevel(check_level),
                            &ana, simplify_trivial_iterators);
    });

class InverseAffineIterMapTransformer {
 public:
  explicit InverseAffineIterMapTransformer(Analyzer* analyzer) : analyzer_(analyzer) {}

  Map<Var, PrimExpr> operator()(const Array<IterSumExpr>& iter_map,
                                const Array<PrimExpr>& outputs) {
    ICHECK(iter_map.size() == outputs.size());
    std::vector<const IterMapExprNode*> post_dfs_order = ReverseTopologyOrder(iter_map);

    // initialize back propagation accumulator
    for (const IterMapExprNode* node : post_dfs_order) {
      backprop_.Set(GetRef<IterMapExpr>(node), Integer(0));
    }
    for (size_t i = 0; i < iter_map.size(); i++) {
      backprop_.Set(iter_map[i], outputs[i]);
    }

    // run back propagation
    for (const IterMapExprNode* node : post_dfs_order) {
      if (node->IsInstance<IterSumExprNode>()) {
        Visit_(Downcast<IterSumExpr>(GetRef<IterMapExpr>(node)));
      } else {
        ICHECK(node->IsInstance<IterSplitExprNode>());
        Visit_(Downcast<IterSplitExpr>(GetRef<IterMapExpr>(node)));
      }
    }
    return std::move(inverse_);
  }

 private:
  void Visit_(const IterSumExpr& iter_map_expr) {
    PrimExpr input = backprop_.at(iter_map_expr) - iter_map_expr->base;

    // Case 1: Propagate to the input node directly when the sum expression has only one components
    if (iter_map_expr->args.size() == 1) {
      const auto& source = iter_map_expr->args[0];
      ICHECK(analyzer_->CanProveEqual(abs(source->scale), 1));
      backprop_.Set(source, (backprop_.at(source) + input) * source->scale);
      return;
    }

    // Case 2: If the sum expression has multiple components, check the fuse pattern and then split
    // the sum expression for each components.
    // For example, consider the iterator i1[dom = (0, 16)], i2[dom = (0, 8)], fusing i1 and i2
    // we will have i1_i2_fused[dom = (0, 64)]. During back propagation, we need to split the
    // propagated value to get the corresponding components of i1 and i2, which are
    // floordiv(i1_i2_fused, 8) and floormod(i1_i2_fused, 8), respectively.
    CheckFusePattern(iter_map_expr);
    for (size_t i = iter_map_expr->args.size(); i > 0; i--) {
      const IterSplitExpr& split = iter_map_expr->args[i - 1];
      PrimExpr prop_value = floordiv(input, split->scale);
      // the first part has the same extent as the split expression, floormod is not needed
      if (i > 1) {
        prop_value = floormod(prop_value, split->extent);
      }
      backprop_.Set(split, backprop_.at(split) + prop_value);
    }
  }

  std::vector<const IterMapExprNode*> ReverseTopologyOrder(const Array<IterSumExpr>& iter_map) {
    std::vector<const IterMapExprNode*> post_dfs_order;
    std::unordered_map<IterMapExpr, bool, ObjectPtrHash, ObjectPtrEqual> visited;

    std::function<void(const IterMapExpr&)> fvisit = [&](const IterMapExpr& expr) {
      if (visited[expr]) {
        return;
      }
      visited[expr] = true;
      if (const auto* sum_expr = expr.as<IterSumExprNode>()) {
        for (const IterSplitExpr& child : sum_expr->args) {
          fvisit(child);
        }
      } else {
        const auto* split_expr = expr.as<IterSplitExprNode>();
        ICHECK(split_expr);
        if (auto source = split_expr->source->source.as<IterMapExpr>()) {
          fvisit(source.value());
        }
      }
      post_dfs_order.push_back(expr.get());
    };
    for (const IterSumExpr& expr : iter_map) {
      fvisit(expr);
    }
    std::reverse(post_dfs_order.begin(), post_dfs_order.end());
    return post_dfs_order;
  }

  void Visit_(const IterSplitExpr& iter_map_expr) {
    PrimExpr input = backprop_.at(iter_map_expr) * iter_map_expr->lower_factor;
    const IterMark& source = iter_map_expr->source;
    if (source->source.as<IterSumExprNode>()) {
      IterSumExpr source_expr = Downcast<IterSumExpr>(source->source);
      backprop_.Set(source_expr, backprop_.at(source_expr) + input);
    } else {
      Var source_var = Downcast<Var>(source->source);
      if (inverse_.count(source_var)) {
        inverse_.Set(source_var, inverse_.at(source_var) + input);
      } else {
        inverse_.Set(source_var, input);
      }
    }
  }

  /*
   * \brief Check the fuse pattern of sum_expr. We assume components of sum_expr is sorted in
   *        descending order of lower_factor.
   */
  void CheckFusePattern(const IterSumExpr sum_expr) {
    if (sum_expr->args.empty()) {
      return;
    }
    PrimExpr expected_scale = sum_expr->args.back()->scale;
    for (size_t i = sum_expr->args.size(); i > 0; i--) {
      ICHECK(analyzer_->CanProveEqual(sum_expr->args[i - 1]->scale, expected_scale));
      expected_scale *= sum_expr->args[i - 1]->extent;
    }
  }

  Analyzer* analyzer_;
  Map<IterMapExpr, PrimExpr> backprop_;  // the accumulator of backpropgation
  Map<Var, PrimExpr> inverse_;           // the result of inverse transformation
};

Map<Var, PrimExpr> InverseAffineIterMap(const Array<IterSumExpr>& iter_map,
                                        const Array<PrimExpr> outputs) {
  Analyzer analyzer;
  return InverseAffineIterMapTransformer(&analyzer)(iter_map, outputs);
}

TVM_REGISTER_GLOBAL("arith.InverseAffineIterMap").set_body_typed(InverseAffineIterMap);

TVM_REGISTER_NODE_TYPE(IterMapResultNode);

}  // namespace arith
}  // namespace tvm
