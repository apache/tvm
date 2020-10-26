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
 * \brief Collector that collects
 *  the outgoing split reference of each IterMark.
 *
 *  These out-going splits can then be used to
 *  check if the iterators are independent.
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

// Rewriter to rewrite PrimExpr to IterMapExpr
// when possible
class IterMapRewriter : public ExprMutator {
 public:
  using Parent = ExprMutator;

  explicit IterMapRewriter(Analyzer* analyzer, const Map<Var, Range>& input_iters)
      : analyzer_(analyzer) {
    for (auto kv : input_iters) {
      const auto& vrng = kv.second;
      if (is_zero(vrng->min)) {
        IterMark mark(kv.first, vrng->extent);
        var_map_[kv.first] = IterSplitExpr(mark);
        input_marks_.push_back(mark);
      } else {
        IterMark mark(kv.first - vrng->min, vrng->extent);
        auto sum_expr = ToIterSumExpr(IterSplitExpr(mark));
        sum_expr.CopyOnWrite()->base = vrng->min;
        var_map_[kv.first] = sum_expr;
        input_marks_.push_back(mark);
      }
    }
  }

  size_t unresolved_count() const { return unresolved_count_; }

  IterSumExpr Rewrite(PrimExpr expr) {
    return NormalizeToIterWithOffset(ToIterSumExpr(DirectMutate(expr)));
  }

  bool CheckBijective(const Array<IterSumExpr>& indices) {
    // This function checks two conditions:
    // - C0: Each iter mark should be fully covered by non-overlapping splits.
    // - C1: All of the input iterators are used.
    //
    // Example: given x in [0, 8) y in [0, 6)
    // - indices = [x, x+1, y] won't pass because x and x+1 contribute
    //   two splits that overlaps with each other.
    // - indices = [x / 4, x % 4, y] will pass because x / 4 and x % 4
    //   contribute two non-overlapping splits that covers x.
    // - indices = [x / 4, x % 4] won't pass because y is not used.
    //
    IterMarkSplitCollector collector;
    // We can check that for each iter mark:
    // All the splits that refers to the itermark covers its extent.
    // The splits do not overlap with each other.
    collector.Collect(indices);
    for (const IterMark& mark : collector.visited_) {
      if (TryNormalizeSplits(mark, collector.mark2splits_[mark]).empty()) return false;
    }
    // all input marks must be visited
    for (const auto& mark : input_marks_) {
      if (collector.visited_.count(mark) == 0) return false;
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
      for (const auto& arg : value->args) {
        hash = support::HashCombine(hash, std::hash<const Object*>()(arg->source.get()));
      }
      return hash;
    }
  };

  struct IterSumEqual {
    bool operator()(const IterSumExpr& lhs, const IterSumExpr& rhs) const {
      tir::ExprDeepEqual equal;
      if (lhs->args.size() != rhs->args.size()) return false;
      if (!equal(lhs->base, rhs->base)) return false;
      for (size_t i = 0; i < lhs->args.size(); ++i) {
        auto lvalue = lhs->args[i];
        auto rvalue = rhs->args[i];
        if (!lvalue->source.same_as(rvalue->source)) return false;
        if (!equal(lvalue->lower_factor, rvalue->lower_factor)) return false;
        if (!equal(lvalue->scale, rvalue->scale)) return false;
        if (!equal(lvalue->extent, rvalue->extent)) return false;
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
  // The canonical map for sum
  std::unordered_map<IterSumExpr, IterSplitExpr, IterSumHash, IterSumEqual> sum_fuse_map_;

  /*!
   * \brief Verify that splits fully covers mark in a non-overlapping fashion.
   *        If verification passes, return splits from outermost to inner most order.
   *        If not, return an empty array
   * \param mark The iterator of interest.
   * \param splits The splits to be verified.
   * \return The normalized splits.
   */
  Array<IterSplitExpr> TryNormalizeSplits(const IterMark& mark,
                                          const std::vector<IterSplitExpr>& splits) {
    std::vector<bool> used(splits.size(), false);
    std::vector<IterSplitExpr> iters;
    PrimExpr expected_lower_factor = make_const(mark->source->dtype, 1);

    for (size_t i = 0; i < splits.size(); ++i) {
      size_t j = 0;
      for (; j < splits.size(); ++j) {
        if (used[j]) continue;
        if (!used[j] && CanProveEqual(splits[j]->lower_factor, expected_lower_factor)) break;
      }
      if (j == splits.size()) {
        return Array<IterSplitExpr>();
      }
      used[j] = true;
      iters.push_back(splits[j]);
      expected_lower_factor *= splits[j]->extent;
    }
    if (!CanProveEqual(expected_lower_factor, mark->extent)) return Array<IterSplitExpr>();
    return Array<IterSplitExpr>(iters.rbegin(), iters.rend());
  }

  /*!
   * \brief Normalize expr to an iterator + offset.
   * \param expr The input expression.
   * \return The Normalized expression.
   */
  IterSumExpr NormalizeToIterWithOffset(IterSumExpr expr) {
    if (expr->args.size() <= 1) return expr;
    PrimExpr base = expr->base;
    expr.CopyOnWrite()->base = make_zero(expr->dtype);
    auto opt = TryFuseIters(expr);
    expr.CopyOnWrite()->base = base;
    if (opt) {
      expr.CopyOnWrite()->args = Array<IterSplitExpr>({opt.value()});
      return expr;
    } else {
      ++unresolved_count_;
      return expr;
    }
  }

  bool CanProveEqual(PrimExpr lhs, PrimExpr rhs) {
    const auto* clhs = lhs.as<IntImmNode>();
    const auto* crhs = rhs.as<IntImmNode>();
    if (clhs && crhs) return clhs->value == crhs->value;
    return analyzer_->CanProve(lhs - rhs == 0);
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

  // Try to normalize IterSum into a fused IterMark
  // return a corresponding splitexpr if needed.
  // IterSum = x1*c1 + x2*c2 + ... + xn*cn
  //         = (x1*s1 + x2*s2 + ... + xn)*cn
  //         = y*cn (IterMark y => x1*s1 + x2*s2 + ... + xn)
  //         = [IterSplit(IterMark(y), scale=cn)]
  // return a corresponding IterSplitExpr if needed.
  Optional<IterSplitExpr> TryFuseIters(IterSumExpr expr) {
    if (!is_zero(expr->base)) return NullOpt;
    if (expr->args.size() == 1) return expr->args[0];
    // select the iterators in order
    std::vector<bool> visited(expr->args.size(), false);
    std::vector<IterSplitExpr> iters;
    iters.reserve(expr->args.size());
    // canonicalize the expression
    // find the base scale first
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
    for (size_t i = 0; i < expr->args.size(); ++i) {
      size_t j = i == 0 ? base_index : 0;
      for (; j < expr->args.size(); ++j) {
        if (!visited[j] && CanProveEqual(expr->args[j]->scale, expected_scale)) break;
      }
      if (j == expr->args.size()) {
        return NullOpt;
      }
      visited[j] = true;
      auto arg = expr->args[j];
      arg.CopyOnWrite()->scale = div(expr->args[j]->scale, base_scale.value());
      iters.push_back(arg);
      expected_scale *= expr->args[j]->extent;
    }
    // update the iterator to use the canonicalized form
    expr.CopyOnWrite()->args = Array<IterSplitExpr>(iters.rbegin(), iters.rend());
    auto it = sum_fuse_map_.find(expr);
    if (it != sum_fuse_map_.end()) return it->second;
    auto mark = IterMark(expr, div(expected_scale, base_scale.value()));
    IterSplitExpr split(mark, base_scale.value());
    sum_fuse_map_[expr] = split;
    return split;
  }

  bool CanProveDivisible(const PrimExpr& lhs, const PrimExpr& rhs) {
    const auto* clhs = lhs.as<IntImmNode>();
    const auto* crhs = rhs.as<IntImmNode>();
    if (clhs && crhs) return clhs->value % crhs->value == 0;
    return analyzer_->CanProve(floormod(lhs, rhs) == 0);
  }

  PrimExpr SplitFloorDivConst(IterSplitExpr lhs, PrimExpr rhs);
  PrimExpr SplitFloorModConst(IterSplitExpr lhs, PrimExpr rhs);

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

Array<IterSumExpr> DetectIterMap(const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters,
                                 arith::Analyzer* analyzer) {
  // Overall detection algorithm is divided into two steps:
  // - Step0: IterMapRewriter rewrites the expression to use IterMapExpr patterns.
  // - Step1: IterIndependenceChecker checks if the iterator are independent.
  IterMapRewriter rewriter(analyzer, input_iters);
  Array<IterSumExpr> results;

  for (PrimExpr value : indices) {
    results.push_back(rewriter.Rewrite(value));
    if (rewriter.unresolved_count() != 0) return Array<IterSumExpr>();
  }
  if (!rewriter.CheckBijective(results)) return Array<IterSumExpr>();

  return results;
}

TVM_REGISTER_GLOBAL("arith.DetectIterMap")
    .set_body_typed([](const Array<PrimExpr>& indices, const Map<Var, Range>& input_iters) {
      arith::Analyzer ana;
      return DetectIterMap(indices, input_iters, &ana);
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
    return Mul(a, b);
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

PrimExpr IterMapRewriter::SplitFloorDivConst(IterSplitExpr lhs, PrimExpr rhs) {
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
        return floordiv(lhs, rhs);
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
    return floordiv(lhs, rhs);
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
    return FloorDiv(a, b);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(a);
    if (auto opt = TryFuseIters(ret)) {
      return SplitFloorDivConst(opt.value(), b);
    } else {
      ++unresolved_count_;
      return FloorDiv(a, b);
    }
  } else {
    ICHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    return SplitFloorDivConst(ret, b);
  }
}

PrimExpr IterMapRewriter::SplitFloorModConst(IterSplitExpr lhs, PrimExpr rhs) {
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
        return floormod(lhs, rhs);
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
    return floormod(lhs, rhs);
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
    return FloorMod(a, b);
  }

  if (a->IsInstance<IterSumExprNode>()) {
    IterSumExpr ret = Downcast<IterSumExpr>(a);
    if (auto opt = TryFuseIters(ret)) {
      return SplitFloorModConst(opt.value(), b);
    } else {
      ++unresolved_count_;
      return FloorMod(a, b);
    }
  } else {
    ICHECK(a->IsInstance<IterSplitExprNode>());
    IterSplitExpr ret = Downcast<IterSplitExpr>(std::move(a));
    return SplitFloorModConst(ret, b);
  }
}

}  // namespace arith
}  // namespace tvm
