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

#include "../support/utils.h"
#include "const_fold.h"
#include "pattern_match.h"

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

/*! \brief Rewriter to rewrite PrimExpr to IterMapExpr when possible */
class IterMapRewriter : public ExprMutator {
 public:
  using Parent = ExprMutator;

  explicit IterMapRewriter(Analyzer* analyzer, const Map<Var, Range>& input_iters)
      : analyzer_(analyzer) {
    for (auto kv : input_iters) {
      const Var& var = kv.first;
      const Range& vrng = kv.second;
      if (is_one(vrng->extent)) {
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

  size_t unresolved_count() const { return unresolved_count_; }

  IterSumExpr Rewrite(const PrimExpr& expr) {
    return NormalizeToIterWithOffset(ToIterSumExpr(DirectMutate(expr)));
  }

  IterSumExpr RewriteIterConstraint(const PrimExpr& expr,
                                    const PrimExpr& predicate_induced_extent) {
    return NormalizeToIterOnBoundExpr(ToIterSumExpr(DirectMutate(expr)), predicate_induced_extent);
  }

  /*!
   * \brief If require_bijective is true, this function checks two conditions:
   *   - C0: Each iter mark should be fully covered by non-overlapping splits.
   *   - C1: All of the input iterators are used.
   *   Example: given x in [0, 8) y in [0, 6)
   *   - bindings = [x, x + 1, y] won't pass because x and x+1 contribute
   *     two splits that overlaps with each other.
   *   - bindings = [x / 4, x % 4, y] will pass because x / 4 and x % 4
   *     contribute two non-overlapping splits that covers x.
   *   - bindings = [x / 4, x % 4] won't pass because y is not used.
   *
   *   If require_bijective is false, this function checks one condition:
   *   - C0: Each iter mark has a chance to be fully covered by non-overlapping splits.
   *   Example: given x in [0, 8) y in [0, 6)
   *   - bindings = [x / 4] will pass because x / 4 can be one split of x
   *   - bindings = [x / 4, x % 4] will pass because x / 4 and x % 4
   *     contribute two non-overlapping splits that covers x.
   *   - bindings = [x / 3] will not pass because x / 3 can not be one split of x
   * \return whether the bindings are valid
   */
  bool CheckMapping(const Array<IterSumExpr>& bindings, bool require_bijective) {
    IterMarkSplitCollector collector;
    // We can check that for each iter mark:
    // All the splits that refers to the iter_mark covers its extent.
    // The splits do not overlap with each other.
    collector.Collect(bindings);
    for (const IterMark& mark : collector.visited_) {
      if (TryNormalizeSplits(mark, collector.mark2splits_[mark], require_bijective).empty())
        return false;
    }
    if (require_bijective) {
      // all input marks must be visited
      for (const IterMark& mark : input_marks_) {
        if (collector.visited_.count(mark) == 0) return false;
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
      ++unresolved_count_;
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
  // Counter to keep track of unresolved cases.
  int unresolved_count_{0};
  // The var map
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> var_map_;
  // input iter marks
  std::vector<IterMark> input_marks_;
  // The map for sum that maps flattened form to IterMark with normal form and extent
  // Example: expr = i*9 + j*2 + k, i in [0, 4) j in [0, 5) k in [0, 2)
  //          predicate: j*2 + k < 9
  // Then,    flattened form = IterSum(IterSplit(i, scale=9),
  //                                   IterSplit(j, scale=2),
  //                                   IterSplit(k, scale=1))
  //          normal form    = IterSum(IterSplit(i, scale=9),
  //                                   IterSplit(IterMark(IterSum(IterSplit(j, scale=2),
  //                                                              IterSplit(k, scale=1)),
  //                                                      extent=9)
  //                                             scale=1))
  std::unordered_map<IterSumExpr, IterMark, IterSumHash, IterSumEqual> sum_fuse_map_;
  // The map for sum that maps normal form to flattened form
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
   * \param require_bijective A boolean flag that indicates whether the bindings should be
   * bijective.
   * \return The normalized splits.
   */
  Array<IterSplitExpr> TryNormalizeSplits(const IterMark& mark,
                                          const std::vector<IterSplitExpr>& splits,
                                          bool require_bijective) {
    std::vector<bool> used(splits.size(), false);
    std::vector<IterSplitExpr> iters;
    PrimExpr expected_lower_factor = make_const(mark->source->dtype, 1);

    for (size_t i = 0; i < splits.size(); ++i) {
      size_t j = 0;
      for (; j < splits.size(); ++j) {
        if (used[j]) continue;
        if (!used[j] && analyzer_->CanProveEqual(splits[j]->lower_factor, expected_lower_factor))
          break;
      }
      if (j == splits.size()) {
        // we do not allow incomplete split if the bindings should be bijective
        if (require_bijective) return Array<IterSplitExpr>();
        // look for the next split skipping this lower factor
        // For example, y \in [0, 24) has 3 splits [y / 6, (y / 2) % 6, y % 2]
        // It is valid to only have [y / 6, y % 2] if bijective is not required
        // We can skip (y / 2) % 6
        j = SearchSkipLowerFactor(splits, used, expected_lower_factor);
        // split not found
        if (j == splits.size()) return Array<IterSplitExpr>();
      }
      used[j] = true;
      iters.push_back(splits[j]);
      expected_lower_factor = splits[j]->lower_factor * splits[j]->extent;
    }
    // Case 1. bijective is required.
    //         We check the extent we calculate is consistent with the extent of the mark
    // Case 2. bijective is not required.
    //         We check the extent we calculate is a factor of the extent of the mark
    //         For example, y \in [0, 24) [(y / 2) % 6, y % 2] is valid, but y \in [0, 25) is not.
    if ((require_bijective && !analyzer_->CanProveEqual(expected_lower_factor, mark->extent)) ||
        (!require_bijective && !CanProveDivisible(mark->extent, expected_lower_factor))) {
      return Array<IterSplitExpr>();
    }
    return Array<IterSplitExpr>(iters.rbegin(), iters.rend());
  }

  /*!
   * \brief Normalize the left hand side of iter constraint(expr < predicate_induced_extent)
   * \param expr The left hand side of iter constraint.
   * \param predicate_induced_extent Extent from iter constraint.
   * \return The Normalized expression.
   */
  IterSumExpr NormalizeToIterOnBoundExpr(IterSumExpr expr,
                                         const PrimExpr& predicate_induced_extent) {
    // We are normalizing the left hand side of iter constraint(iter < predicate_induced_extent)
    Optional<IterSplitExpr> opt = TryFuseIters(expr);
    // scale should be 1
    if (opt.defined() && is_one(opt.value()->scale)) {
      IterSumExpr sum = Downcast<IterSumExpr>(opt.value()->source->source);
      // get the flattened form
      auto it = flattened_map_.find(sum);
      ICHECK(it != flattened_map_.end());
      IterSumExpr flattened_form = it->second;
      // get the mark
      auto it_mark = sum_fuse_map_.find(flattened_form);
      ICHECK(it_mark != sum_fuse_map_.end());
      IterMark mark = it_mark->second;
      mark.CopyOnWrite()->extent = min(predicate_induced_extent, mark->extent);
      // update the bound of the lhs based on predicate_induced_extent
      sum_fuse_map_[flattened_form] = mark;
      // we need to note down the flattened form of constrained iterators
      // to check the validity of constraints, see also CheckConstraints()
      constrained_iters_flattened_.push_back(flattened_form);
      expr.CopyOnWrite()->args = Array<IterSplitExpr>({opt.value()});
      return expr;
    }
    ++unresolved_count_;
    return expr;
  }

  /*!
   * \brief Normalize expr to an iterator + offset.
   * \param expr The input expression.
   * \return The Normalized expression.
   */
  IterSumExpr NormalizeToIterWithOffset(IterSumExpr expr) {
    // We are normalizing a regular iter
    if (expr->args.size() <= 1) return expr;
    PrimExpr base = expr->base;
    expr.CopyOnWrite()->base = make_zero(expr->dtype);
    Optional<IterSplitExpr> opt = TryFuseIters(expr);
    expr.CopyOnWrite()->base = base;
    if (opt.defined()) {
      expr.CopyOnWrite()->args = Array<IterSplitExpr>({opt.value()});
      return expr;
    } else {
      ++unresolved_count_;
      return expr;
    }
  }

  /*!
   * \brief Create a IterSumExpr from expr.
   * \param expr The input expr.
   * \return The transformed IterSumExpr.
   */
  static IterSumExpr ToIterSumExpr(const PrimExpr& expr) {
    if (const auto* op = expr.as<IterSumExprNode>()) {
      return GetRef<IterSumExpr>(op);
    } else if (const auto* op = expr.as<IterSplitExprNode>()) {
      return IterSumExpr({GetRef<IterSplitExpr>(op)}, make_zero(expr->dtype));
    } else {
      ICHECK(!expr->IsInstance<IterMapExprNode>());
      return IterSumExpr({}, expr);
    }
  }

  /*!
   * \brief IterSum = x1*c1 + x2*c2 + ... + xn*cn
   *      = (x1*s1 + x2*s2 + ... + xn)*cn
   *      = y*cn (IterMark y => x1*s1 + x2*s2 + ... + xn)
   *      = [IterSplit(IterMark(y), scale=cn)]
   *    return a corresponding IterSplitExpr if needed.
   *    Try to normalize IterSum into a fused IterMark
   * \param expr The input sum.
   * \return The split with the fused IterMark if succeed.
   */
  Optional<IterSplitExpr> TryFuseIters(IterSumExpr expr) {
    if (!is_zero(expr->base)) return NullOpt;
    // select the iterators in order
    std::vector<bool> visited(expr->args.size(), false);
    std::vector<IterSplitExpr> flattened_iters, grouped_iters;
    // canonicalize the expression into two different forms: flattened form and structured form
    // step0. check if find the base scale first
    Optional<IntImm> base_scale = NullOpt;
    size_t base_index = 0;
    for (size_t i = 0; i < expr->args.size(); ++i) {
      if (const auto* op = expr->args[i]->scale.as<IntImmNode>()) {
        if (!base_scale || op->value < base_scale.value()->value) {
          base_scale = GetRef<IntImm>(op);
          base_index = i;
        }
      }
    }
    if (!base_scale) return NullOpt;
    // check if it can be remapped into a fused pattern.
    PrimExpr expected_scale = base_scale.value();
    for (size_t i = 0; i < expr->args.size();) {
      // find j such that expr->args[j] has expected scale
      size_t j = i == 0 ? base_index : 0;
      for (; j < expr->args.size(); ++j) {
        if (!visited[j] && analyzer_->CanProveEqual(expr->args[j]->scale, expected_scale)) break;
      }
      if (j == expr->args.size()) return NullOpt;
      // look for the longest constrained iter started from expr->args[j]
      // Example: expr = i*9 + j*2 + k, i in [0, 4) j in [0, 5) k in [0, 2)
      //          predicate: j*2 + k < 9
      // We need to match the predicate in expr and adjust the expected scale,
      // otherwise we expect the scale of i to be 2*5=10
      Optional<IterSumExpr> constraint_to_match;
      for (const IterSumExpr& iter : constrained_iters_flattened_) {
        if (IterSplitEqual(expr->args[j], iter->args.back(), false)) {
          // find a predicate started from expr->args[j]
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
              if (analyzer_->CanProveEqual((*it)->scale * expected_scale, expr->args[k]->scale))
                break;
            }
          }
          if (k == expr->args.size()) return NullOpt;
          visited[k] = true;
          flattened_iters.push_back(expr->args[k]);
        }
        auto iter = sum_fuse_map_.find(constraint_to_match.value());
        ICHECK(iter != sum_fuse_map_.end());
        IterMark iter_matched = iter->second;
        grouped_iters.emplace_back(iter_matched, expected_scale);
        expected_scale *= iter_matched->extent;
        // move forward
        i += constraint_to_match.value()->args.size();
      } else {
        // constraint_to_match not found, skip this iterator
        visited[j] = true;
        flattened_iters.push_back(expr->args[j]);
        grouped_iters.push_back(expr->args[j]);
        expected_scale *= expr->args[j]->extent;
        ++i;
      }
    }
    // Get the flattened form and structured form
    // both forms have splits from outermost to innermost
    IterSumExpr structured_form = expr, flattened_form = expr;
    flattened_form.CopyOnWrite()->args =
        Array<IterSplitExpr>(flattened_iters.rbegin(), flattened_iters.rend());
    structured_form.CopyOnWrite()->args =
        Array<IterSplitExpr>(grouped_iters.rbegin(), grouped_iters.rend());
    auto it = sum_fuse_map_.find(flattened_form);
    if (it != sum_fuse_map_.end()) {
      // old iter
      return IterSplitExpr(it->second, base_scale.value());
    } else {
      // new iter, form a new mark
      IterMark mark = IterMark(structured_form, div(expected_scale, base_scale.value()));
      sum_fuse_map_[flattened_form] = mark;
      flattened_map_[structured_form] = flattened_form;
      return IterSplitExpr(mark, base_scale.value());
    }
  }

  bool CanProveDivisible(const PrimExpr& lhs, const PrimExpr& rhs) {
    const auto* clhs = lhs.as<IntImmNode>();
    const auto* crhs = rhs.as<IntImmNode>();
    if (clhs && crhs) return clhs->value % crhs->value == 0;
    return analyzer_->CanProveEqual(lhs, rhs) || analyzer_->CanProve(floormod(lhs, rhs) == 0);
  }

  PrimExpr SplitFloorDivConst(IterSplitExpr lhs, PrimExpr rhs, const PrimExpr& orig);
  PrimExpr SplitFloorModConst(IterSplitExpr lhs, PrimExpr rhs, const PrimExpr& orig);

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
  // The expr of the upper_bound
  PrimExpr upper_bound;
  // The size of the iter, which is the number of nodes
  size_t expr_size = 0;

  IterConstraint(PrimExpr iter, PrimExpr upper_bound, size_t size)
      : iter(std::move(iter)), upper_bound(std::move(upper_bound)), expr_size(size) {}
};

/*!
 * \brief Split the predicate into `(a < b) && (c < d) && ...`
 * \param pred The predicate to be split.
 * \return A list of pairs, each element of which are lhs and rhs of the '<' sign,
 *         empty if the split failed.
 */
std::vector<IterConstraint> MatchUpperBoundConstraints(PrimExpr pred) {
  std::vector<IterConstraint> result;
  arith::PVar<PrimExpr> lhs, rhs, rest;
  for (;;) {
    if ((rest && (lhs < rhs)).Match(pred)) {
      result.emplace_back(lhs.Eval(), rhs.Eval(), 0);
      pred = rest.Eval();
    } else if ((lhs < rhs).Match(pred)) {
      result.emplace_back(lhs.Eval(), rhs.Eval(), 0);
      break;
    } else {
      return std::vector<IterConstraint>();
    }
  }
  return result;
}

bool IterRangeSanityCheck(const Map<Var, Range>& iter_ranges) {
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> iters;
  for (const auto& it : iter_ranges) iters.insert(it.first);
  auto f = [&](const VarNode* var) { return iters.count(GetRef<Var>(var)); };
  for (const auto& it : iter_ranges) {
    if (UsesVar(it.second->min, f) || UsesVar(it.second->extent, f)) return false;
  }
  return true;
}

Array<IterSumExpr> DetectIterMap(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                                 const PrimExpr& predicate, bool require_bijective,
                                 arith::Analyzer* analyzer) {
  // Overall detection algorithm is divided into two steps:
  // - Step0: IterMapRewriter rewrites the expression to use IterMapExpr patterns.
  // - Step1: IterIndependenceChecker checks if the iterator are independent.

  if (!IterRangeSanityCheck(input_iters)) return Array<IterSumExpr>();
  std::vector<IterConstraint> constraints = MatchUpperBoundConstraints(predicate);
  if (!is_one(predicate) && constraints.empty()) return Array<IterSumExpr>();

  // We have to make sure when we visit an iterator, all the constraints related with its successors
  // in the iter var graph has been visited, where the expression of this iterator will contain the
  // expression of its successor, so we sort them by their sizes.
  for (IterConstraint& constraint : constraints) {
    constraint.expr_size = CalculateExprComplexity(constraint.iter);
  }

  std::sort(
      constraints.begin(), constraints.end(),
      [](const IterConstraint& a, const IterConstraint& b) { return a.expr_size < b.expr_size; });

  IterMapRewriter rewriter(analyzer, input_iters);
  // Step0.0: rewrite constraints in the order from size-small ones to size-big ones
  for (const IterConstraint& constraint : constraints) {
    PrimExpr res = rewriter.RewriteIterConstraint(constraint.iter, constraint.upper_bound);
    if (rewriter.unresolved_count() != 0) return Array<IterSumExpr>();
  }
  if (!rewriter.CheckConstraints()) return Array<IterSumExpr>();
  // Step0.1: rewrite indices
  Array<IterSumExpr> results;
  for (PrimExpr value : indices) {
    results.push_back(rewriter.Rewrite(value));
    if (rewriter.unresolved_count() != 0) return Array<IterSumExpr>();
  }
  // Step1: IterIndependenceChecker checks if the iterator are independent.
  if (!rewriter.CheckMapping(results, require_bijective)) return Array<IterSumExpr>();

  return results;
}

TVM_REGISTER_GLOBAL("arith.DetectIterMap")
    .set_body_typed([](const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                       const PrimExpr& input_pred, bool is_bijective) {
      arith::Analyzer ana;
      return DetectIterMap(indices, input_iters, input_pred, is_bijective, &ana);
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
  PrimExpr const_res = TryConstFold<Add>(a, b);
  if (const_res.defined()) return const_res;
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
  } else if (const auto* op = b.as<IterSumExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSumExpr>(op), 1);
  } else if (const auto* op = b.as<IterSplitExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSplitExpr>(op), 1);
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
  PrimExpr const_res = TryConstFold<Sub>(a, b);
  if (const_res.defined()) return const_res;

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
  } else if (const auto* op = b.as<IterSumExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSumExpr>(op), -1);
  } else if (const auto* op = b.as<IterSplitExprNode>()) {
    AddToLhs(ret.CopyOnWrite(), GetRef<IterSplitExpr>(op), -1);
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
  PrimExpr const_res = TryConstFold<Mul>(a, b);
  if (const_res.defined()) return const_res;

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
    ++unresolved_count_;
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

PrimExpr IterMapRewriter::SplitFloorDivConst(IterSplitExpr lhs, PrimExpr rhs,
                                             const PrimExpr& orig) {
  // floordiv(x*scale, rhs)
  if (is_one(rhs)) return std::move(lhs);
  if (!is_one(lhs->scale)) {
    if (CanProveDivisible(lhs->scale, rhs)) {
      // floordiv(x*c1*c2, c2) = x*c1, c1=scale/rhs
      lhs.CopyOnWrite()->scale = floordiv(lhs->scale, rhs);
      return std::move(lhs);
    } else {
      if (CanProveDivisible(rhs, lhs->scale)) {
        // floordiv(x*c1, c1*c2) = floordiv(x, c2), c2=rhs/scale
        rhs = floordiv(rhs, lhs->scale);
        lhs.CopyOnWrite()->scale = make_const(rhs->dtype, 1);
      } else {
        // mark as unresolved.
        ++unresolved_count_;
        return orig;
      }
    }
  }

  // We handle scale!=1 in above code, hence we only consider floordiv(x, rhs) below
  // where x=floormod(floordiv(iter, lower_factor), extent)
  if (CanProveDivisible(lhs->extent, rhs)) {
    // floordiv(floormod(floordiv(iter, lower_factor), c1c2), c1)
    // = floordiv(floormod(y, c1c2), c1), where y=floordiv(iter, lower_factor)
    // = floordiv(floormod(sc1c2+tc1+u, c1c2), c1), where y=sc1c2+tc1+u, t<c2, u<c1
    // = t
    // = floormod(sc2+t, c2)
    // = floormod(floordiv(y, c1), c2)
    // = floormod(floordiv(iter, lower_factor*c1), c2), where c1=rhs, c2=extent/rhs
    auto* ptr_lhs = lhs.CopyOnWrite();
    ptr_lhs->lower_factor *= rhs;
    ptr_lhs->extent = analyzer_->Simplify(floordiv(ptr_lhs->extent, rhs));
    return std::move(lhs);
  } else {
    // mark as unresolved.
    ++unresolved_count_;
    return orig;
  }
}

PrimExpr IterMapRewriter::VisitExpr_(const FloorDivNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<FloorDiv>(a, b);
  if (const_res.defined()) return const_res;

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
    ++unresolved_count_;
    return GetRef<PrimExpr>(op);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(a);
    if (Optional<IterSplitExpr> opt = TryFuseIters(ret)) {
      return SplitFloorDivConst(opt.value(), b, GetRef<PrimExpr>(op));
    } else {
      ++unresolved_count_;
      return GetRef<PrimExpr>(op);
    }
  } else {
    ICHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    return SplitFloorDivConst(ret, b, GetRef<PrimExpr>(op));
  }
}

PrimExpr IterMapRewriter::SplitFloorModConst(IterSplitExpr lhs, PrimExpr rhs,
                                             const PrimExpr& orig) {
  // floormod(x*scale, rhs)
  if (is_one(rhs)) return make_zero(lhs->dtype);
  if (!is_one(lhs->scale)) {
    // floormod(x*c1*c2, c1) = 0
    if (CanProveDivisible(lhs->scale, rhs)) {
      return make_zero(lhs->dtype);
    } else {
      if (CanProveDivisible(rhs, lhs->scale)) {
        // floormod(x*c1, c1*c2) = (floormod(x, c2)) * c1, where c2 = rhs/scale
        rhs = floordiv(rhs, lhs->scale);
      } else {
        // mark as unresolved.
        ++unresolved_count_;
        return orig;
      }
    }
  }

  // floormod(x, rhs) where x=floormod(floordiv(iter, lower_factor), extent)
  if (CanProveDivisible(lhs->extent, rhs)) {
    // floormod(floormod(floordiv(iter, lower_factor), c1c2), c1)
    // = floormod(floordiv(iter, lower_factor), c1), where c1=rhs
    lhs.CopyOnWrite()->extent = rhs;
    return std::move(lhs);
  } else {
    // mark as unresolved.
    ++unresolved_count_;
    return orig;
  }
}

PrimExpr IterMapRewriter::VisitExpr_(const FloorModNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Parent::VisitExpr_(op);
  }

  PrimExpr a = this->DirectMutate(op->a);
  PrimExpr b = this->DirectMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<FloorMod>(a, b);
  if (const_res.defined()) return const_res;

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
    ++unresolved_count_;
    return GetRef<PrimExpr>(op);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(a);
    if (Optional<IterSplitExpr> opt = TryFuseIters(ret)) {
      return SplitFloorModConst(opt.value(), b, GetRef<PrimExpr>(op));
    } else {
      ++unresolved_count_;
      return GetRef<PrimExpr>(op);
    }
  } else {
    ICHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    return SplitFloorModConst(ret, b, GetRef<PrimExpr>(op));
  }
}

/*! * \brief Given an IterVarMapExpr, transform it to normal PrimExpr. */
class IterMapToExprNormalizer {
 public:
  explicit IterMapToExprNormalizer(Analyzer* analyzer) : analyzer_(analyzer) {}

  PrimExpr Convert(const IterMapExpr& expr) {
    if (const auto* op = expr.as<IterSplitExprNode>()) {
      return ConvertIterSplitExpr(GetRef<IterSplitExpr>(op));
    } else if (const auto* op = expr.as<IterSumExprNode>()) {
      return ConvertIterSumExpr(GetRef<IterSumExpr>(op));
    } else {
      ICHECK(expr.defined());
      LOG(FATAL) << "Unknown IterMapExpr type " << expr->GetTypeKey();
      return 0;
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
    if (const auto* op = expr->source->source.as<VarNode>()) {
      source = GetRef<Var>(op);
    } else if (const auto* op = expr->source->source.as<IterSumExprNode>()) {
      source = ConvertIterSumExpr(GetRef<IterSumExpr>(op));
    } else {
      LOG(FATAL) << "Unexpected source of IterSplitExpr";
    }
    if (analyzer_->CanProve(expr->extent == expr->source->extent) && is_one(expr->lower_factor)) {
      return source * expr->scale;
    } else if (analyzer_->CanProve(expr->source->extent == expr->lower_factor * expr->extent)) {
      return floordiv(source, expr->lower_factor) * expr->scale;
    } else {
      return floormod(floordiv(source, expr->lower_factor), expr->extent) * expr->scale;
    }
  }

 private:
  Analyzer* analyzer_;
};

PrimExpr NormalizeIterMapToExpr(const IterMapExpr& expr) {
  arith::Analyzer analyzer;
  IterMapToExprNormalizer normalizer(&analyzer);
  return normalizer.Convert(expr);
}

TVM_REGISTER_GLOBAL("arith.NormalizeIterMapToExpr").set_body_typed([](const IterMapExpr& expr) {
  return NormalizeIterMapToExpr(expr);
});

Array<PrimExpr> IterMapSimplify(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                                const PrimExpr& input_pred, bool require_bijective) {
  if (!IterRangeSanityCheck(input_iters)) return indices;
  Analyzer analyzer;
  Array<IterSumExpr> rewrite =
      DetectIterMap(indices, input_iters, input_pred, require_bijective, &analyzer);
  if (rewrite.empty()) {
    return indices;
  }
  Array<PrimExpr> res;
  res.reserve(rewrite.size());
  IterMapToExprNormalizer converter(&analyzer);
  for (const auto& expr : rewrite) res.push_back(converter.Convert(expr));
  return res;
}

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
                           const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& sub_iters)
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

    DivisionResult(IterMapExpr outer, PrimExpr outer_extent, IterMapExpr inner,
                   PrimExpr inner_extent)
        : outer(std::move(outer)),
          inner(std::move(inner)),
          outer_extent(std::move(outer_extent)),
          inner_extent(std::move(inner_extent)) {}

    // whether the division result is totally in outer subspace
    bool IsOuter() const { return is_one(inner_extent); }

    // whether the division result is totally in inner subspace
    bool IsInner() const { return is_one(outer_extent); }

    IterSplitExpr GetOuterAsSplit() const { return GetAsSplit(outer, outer_extent); }

    IterSplitExpr GetInnerAsSplit() const { return GetAsSplit(inner, inner_extent); }

    static DivisionResult Inner(const IterMapExpr& iter, const PrimExpr& extent) {
      return DivisionResult(IterSumExpr({}, 0), 1, iter, extent);
    }

    static DivisionResult Outer(const IterMapExpr& iter, const PrimExpr& extent) {
      return DivisionResult(iter, extent, IterSumExpr({}, 0), 1);
    }

   private:
    static IterSplitExpr GetAsSplit(const IterMapExpr& expr, const PrimExpr& extent) {
      if (const auto* op = expr.as<IterSplitExprNode>()) {
        return GetRef<IterSplitExpr>(op);
      } else if (const auto* op = expr.as<IterSumExprNode>()) {
        return IterSplitExpr(IterMark(GetRef<IterSumExpr>(op), extent));
      } else {
        LOG(FATAL) << "Unknown IterMapExpr type";
        return NullValue<IterSplitExpr>();
      }
    }
  };

  // Divide an IterSumExpr
  DivisionResult DivideIterSumExpr(const IterSumExpr& expr, const PrimExpr& mark_extent) {
    if (expr->args.empty()) {
      // base
      return DivisionResult(IterSumExpr({}, 0), 1, IterSumExpr({}, expr->base), 1);
    } else if (expr->args.size() == 1) {
      // arg + base, if arg=Y*E(X)+X, then arg+base = Y*E(X)+(X+base)
      if (!is_one(expr->args[0]->scale)) return Fail();
      DivisionResult res = DivideIterSplitExpr(expr->args[0]);
      if (!is_zero(expr->base)) res = AddBase(res, expr->base);
      return res;
    }
    // arg1 + arg2 + ... + argn + base
    // then we can write it as Y*E(X)+X
    // if it starts with contiguous outer splits, followed by contiguous inner splits
    PrimExpr extent = 1;
    std::vector<IterSplitExpr> outer_args, inner_args;
    bool inner = true, scale_is_one = false;
    // we check in inverse order so we can visit from inner to outer
    for (auto it = expr->args.rbegin(); it != expr->args.rend(); ++it) {
      const IterSplitExpr& arg = *it;
      if (is_one(arg->scale)) scale_is_one = true;
      DivisionResult arg_division = DivideIterSplitExpr(arg);
      IterSplitExpr new_arg;
      if (arg_division.IsInner()) {
        if (!inner) return Fail();
        new_arg = arg_division.GetInnerAsSplit();
        inner_args.push_back(new_arg);
        inner = true;
      } else if (arg_division.IsOuter()) {
        new_arg = arg_division.GetOuterAsSplit();
        outer_args.push_back(new_arg);
        inner = false;
      } else {
        return Fail();
      }
      extent *= new_arg->extent;
    }
    if (!scale_is_one) return Fail();
    bool need_predicate = !analyzer_->CanProveEqual(extent, mark_extent);
    const IterMark& outer_mark = MarkFromArgsAndBase(outer_args, 0);
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
        return Fail();
      }
    }
    return DivisionResult(outer_source, outer_mark->extent, inner_source, inner_mark->extent);
  }

  PrimExpr GetOuterPreds() const { return outer_preds_; }
  PrimExpr GetInnerPreds() const { return inner_preds_; }

 private:
  DivisionResult Fail() {
    unresolved_count_++;
    return DivisionResult(IterSumExpr({}, 0), 0, IterSumExpr({}, 0), 0);
  }

  DivisionResult AddBase(DivisionResult division, PrimExpr base) {
    DivisionResult res = division;
    if (const auto* op = division.inner.as<IterSplitExprNode>()) {
      res.inner = IterSumExpr({GetRef<IterSplitExpr>(op)}, base);
    } else if (const auto* op = division.inner.as<IterSumExprNode>()) {
      const auto& expr = GetRef<IterSumExpr>(op);
      res.inner = IterSumExpr(expr->args, expr->base + base);
    }
    return res;
  }

  // args are sorted from inner to outer
  static IterMark MarkFromArgsAndBase(const std::vector<IterSplitExpr>& args, PrimExpr base) {
    std::vector<IterSplitExpr> res;
    PrimExpr extent = 1;
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
    if (const auto* iter_ptr = expr->source->source.as<VarNode>()) {
      // source is input_iter
      bool inner = sub_iters_.count(GetRef<Var>(iter_ptr));
      for (const IterSplitExpr& split : splits) {
        if (inner) {
          // 0*E(split)+split
          split_map_.emplace(split, DivisionResult::Inner(split, split->extent));
        } else {
          // split*1 + 0
          split_map_.emplace(split, DivisionResult::Outer(split, split->extent));
        }
      }
    } else if (const auto* iter_ptr = expr->source->source.as<IterSumExprNode>()) {
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
      DivisionResult mark_division =
          DivideIterSumExpr(GetRef<IterSumExpr>(iter_ptr), expr->source->extent);
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
        if (j == splits.size()) return Fail();
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
      if (!encountered_boundary) return Fail();
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
      return Fail();
    }
    return split_map_.at(expr);
  }

  size_t unresolved_count_{0};
  // arithmetic analyzer used to call CanProve
  Analyzer* analyzer_;
  // collector that collects the outgoing split reference of each IterMark
  const IterMarkSplitCollector collector_;
  // the set of subspace iters
  const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& sub_iters_;
  // map from SplitExpr to its corresponding DivisionResult(Y*E(X)+X)
  std::unordered_map<IterSplitExpr, DivisionResult, ObjectPtrHash, ObjectPtrEqual> split_map_;
  // predicate of outer space and inner space;
  PrimExpr outer_preds_{Bool(true)}, inner_preds_{Bool(true)};
};

Array<Array<IterMark>> SubspaceDivide(const Array<PrimExpr>& bindings,
                                      const Map<Var, Range>& input_iters,
                                      const Array<Var>& sub_iters, const PrimExpr& predicate,
                                      bool require_bijective, arith::Analyzer* analyzer) {
  if (!IterRangeSanityCheck(input_iters)) return Array<Array<IterMark>>();
  const Array<IterSumExpr>& maps =
      DetectIterMap(bindings, input_iters, predicate, require_bijective, analyzer);
  if (maps.empty()) return {};

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> inner_iter_set;
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
                       const Array<Var>& sub_iters, const PrimExpr& predicate,
                       bool require_bijective) {
      arith::Analyzer ana;
      return SubspaceDivide(bindings, root_iters, sub_iters, predicate, require_bijective, &ana);
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
      backprop_.Set(source, backprop_.at(source) + input);
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
      backprop_.Set(split,
                    backprop_.at(split) + floormod(floordiv(input, split->scale), split->extent));
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
        if (const auto* source = split_expr->source->source.as<IterMapExprNode>()) {
          fvisit(GetRef<IterMapExpr>(source));
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
    ICHECK(sum_expr->args.size());
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

}  // namespace arith
}  // namespace tvm
