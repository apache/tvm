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
 * \file canonical_simplify.cc
 * \brief Canonical form based simplification.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/analysis.h>

#include "const_fold.h"
#include "pattern_match.h"
#include "rewrite_simplify.h"

namespace tvm {
namespace arith {

using namespace tir;

class SumExpr;
class SplitExpr;


/*!
 * \brief Base class of all temporary expression introduced
 *        for canonicalization.
 */
class CanonicalExprNode : public PrimExprNode {
 public:
  virtual ~CanonicalExprNode() {}
  /*!
   * \brief Return the normal Expr that is equivalent to self.
   * \note Can mutate the internal data structure.
   * \return The normal expression.
   */
  virtual PrimExpr Normalize() const = 0;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
  }

  static constexpr const char* _type_key = "arith.CanonicalExpr";
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(CanonicalExprNode, PrimExprNode);
};

enum DivMode {
  /*! \brief Truncated division. */
  kTruncDiv,
  /*! \brief Floor division. */
  kFloorDiv
};

inline PrimExpr ModImpl(PrimExpr a, PrimExpr b, DivMode mode) {
  if (mode == kTruncDiv) {
    return truncmod(a, b);
  } else {
    CHECK_EQ(mode, kFloorDiv);
    return floormod(a, b);
  }
}

inline PrimExpr DivImpl(PrimExpr a, PrimExpr b, DivMode mode) {
  if (mode == kTruncDiv) {
    return truncdiv(a, b);
  } else {
    CHECK_EQ(mode, kFloorDiv);
    return floordiv(a, b);
  }
}

/*!
 * \brief Internal "Split normal form" of expression.
 *
 * This is a special expression that represents
 * a scaled value derived from a split of an index.
 *
 * result = ((index % upper_factor) / lower_factor) * scale
 */
class SplitExprNode : public CanonicalExprNode {
 public:
  /*! \brief The base index expression. */
  PrimExpr index;
  /*! \brief The division factor ratio. */
  int64_t lower_factor{1};
  /*!
   * \brief The upper factor.
   * invariance: (upper_factor == kPosInf || upper_factor % lower_factor == 0)
   */
  int64_t upper_factor{kPosInf};
  /*! \brief scale to the expression. */
  int64_t scale{1};
  /*! \brief Division mode. */
  DivMode div_mode{kTruncDiv};

  /*! \brief verify that this is a valid entry. */
  void Verify() const {
    CHECK(upper_factor == kPosInf || upper_factor % lower_factor == 0);
  }

  PrimExpr NormalizeWithScale(int64_t sscale) const {
    PrimExpr res = this->index;
    DataType dtype = this->dtype;
    if (this->scale == 0) {
      return make_const(dtype, 0);
    }
    if (this->upper_factor != SplitExprNode::kPosInf) {
      res = ModImpl(res, make_const(dtype, this->upper_factor), div_mode);
    }
    if (this->lower_factor != 1) {
      res = DivImpl(res, make_const(dtype, this->lower_factor), div_mode);
    }
    sscale *= this->scale;
    if (sscale != 1) {
      CHECK(!dtype.is_uint() || sscale > 0);
      res = res * make_const(dtype, sscale);
    }
    return res;
  }

  PrimExpr Normalize() const final {
    return NormalizeWithScale(1);
  }

  void MulToSelf(int64_t scale) {
    this->scale *= scale;
  }

  inline bool IndexEqual(const SplitExpr& other) const;
  inline bool DivModeCompatibleTo(DivMode mode) const;

  /*! \brief positive infty */
  static const constexpr int64_t kPosInf = ConstIntBoundNode::kPosInf;
  static constexpr const char* _type_key = "arith.SplitExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitExprNode, CanonicalExprNode);
};

class SplitExpr : public PrimExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SplitExpr, PrimExpr, SplitExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SplitExprNode);
};

inline bool SplitExprNode::IndexEqual(const SplitExpr& other) const {
  if (index.same_as(other->index)) return true;
  return tir::ExprDeepEqual()(index, other->index);
}

inline bool SplitExprNode::DivModeCompatibleTo(DivMode mode) const {
  if (this->div_mode == mode) return true;
  if (lower_factor == 1 && upper_factor == kPosInf) return true;
  return false;
}

/*!
 * \brief Normal form that represents sum of expressions.
 *
 *  result = sum(args) + base.
 */
class SumExprNode : public CanonicalExprNode {
 public:
  /*!
   * \brief arguments to be summed up.
   *
   * args are divided into segments with the same index.
   * within each segment, the SplitExpr is ordered in descending order of lower_factor.
   */
  std::vector<SplitExpr> args;
  /*! \brief Base value in the summation. */
  int64_t base{0};
  /*! \brief The expression equals zero. */
  bool IsZero() const {
    return base == 0 && args.size() == 0;
  }
  /*!
   * \brief Return the normal Expr that is equivalent to self.
   * \return The normal expression.
   */
  PrimExpr Normalize() const final {
    // quick path 1.
    if (this->args.size() == 0) {
      return make_const(this->dtype, this->base);
    }
    return Normalize_(this->dtype,
                      SimplifySplitExprs(args),
                      base);
  }
  /*!
   * \brief Whether self is divisible by scale.
   * \param scale The scale to be applied.
   */
  bool DivisibleBy(int64_t scale) {
    if (base % scale != 0) return false;
    for (size_t i = 0; i < this->args.size(); ++i) {
      if (args[i]->scale % scale != 0) return false;
    }
    return true;
  }
  /*!
   * \brief mul scale to self.
   * \param scale The scale to be applied.
   */
  void MulToSelf(int64_t scale) {
    this->base *= scale;
    for (size_t i = 0; i < this->args.size(); ++i) {
      args[i].CopyOnWrite()->scale *= scale;
    }
  }
  /*!
   * \brief divide by scale.
   * \param scale The scale to be applied.
   */
  void DivideBy(int64_t scale) {
    CHECK_EQ(this->base % scale, 0);
    this->base /= scale;
    for (size_t i = 0; i < this->args.size(); ++i) {
      CHECK_EQ(args[i]->scale % scale, 0);
      args[i].CopyOnWrite()->scale /= scale;
    }
  }
  /*!
   * \brief add constant value to self.
   * \param value to be added.
   */
  void AddToSelf(int64_t value) {
    this->base += value;
  }
  /*!
   * \brief self += other * scale;
   * \param other The expression to be added.
   * \param scale The additional scale on value.
   */
  void AddToSelf(SplitExpr other, int64_t scale) {
    if (other->scale == 0) return;
    // We need to maintain the segment invariance:
    // Same index are stored close to each other.
    // sorted from big lower_factor to small one.
    size_t start = 0;
    for (; start < args.size(); ++start) {
      if (args[start]->IndexEqual(other)) break;
    }
    for (size_t j = start; j < args.size(); ++j) {
      if (!args[j]->IndexEqual(other) ||
          other->lower_factor > args[j]->lower_factor) {
        other.CopyOnWrite()->scale *= scale;
        this->args.insert(this->args.begin() + j, other);
        return;
      }
      if (other->lower_factor == args[j]->lower_factor &&
          other->upper_factor == args[j]->upper_factor &&
          other->DivModeCompatibleTo(args[j]->div_mode)) {
        args[j].CopyOnWrite()->scale += other->scale * scale;
        return;
      }
    }
    // Insert other in the end.
    other.CopyOnWrite()->scale *= scale;
    this->args.emplace_back(std::move(other));
  }

  void AddToSelf(const SumExpr& other, int64_t scale);

  static constexpr const char* _type_key = "arith.SumExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(SumExprNode, CanonicalExprNode);

 private:
  /*!
   * \brief Simplify the args by merging SplitExprs
   * \param args The original list of arguments.
   * \return simplified version.
   */
  static std::vector<SplitExpr>
  SimplifySplitExprs(std::vector<SplitExpr> args) {
    // NOTE: This algorithm relies on the factor that args are divided into segments
    // and each segment is sorted in descending order of lower_factor.
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale == 0) continue;
      for (size_t j = i + 1; j < args.size(); ++j) {
        SplitExpr& lhs = args[i];
        SplitExpr& rhs = args[j];
        if (!lhs->IndexEqual(rhs)) break;
        if (lhs->upper_factor < rhs->lower_factor) break;
        if (lhs->upper_factor == rhs->upper_factor &&
            lhs->lower_factor == rhs->lower_factor &&
            lhs->DivModeCompatibleTo(rhs->div_mode)) {
          // folding same co-efficient.
          rhs.CopyOnWrite()->scale += lhs->scale;
          lhs.CopyOnWrite()->scale = 0;
        } else if (lhs->lower_factor == rhs->upper_factor &&
                   rhs->scale != 0 &&
                   lhs->scale % rhs->scale == 0 &&
                   lhs->lower_factor == (lhs->scale / rhs->scale) * rhs->lower_factor &&
                   lhs->DivModeCompatibleTo(rhs->div_mode)) {
          // Rules used in the proof:
          //
          // Rule 1:  (x % (c * s)) / c  =  (x / c) % s
          // Proof:
          //  x can always be decomposed into p * c * s + q * c + r
          //  where  0 <= q * c + r < c * s  and  0 <= r  <  c.
          //  Then, lhs = ((p * c * s + q * c + r) % (c * s)) / c = (q * c + r) / c = q
          //  rhs = ((p * c * s + q * c + r) / c) % s = (p * s + q) % s = q
          //  Thus, lhs = rhs
          //
          // The above proof is for the floordiv.
          // The same rule also holds for truncdiv(division rule in C).
          // Because both sides only involve mul, div and mod,
          // we can take abs of x, c and s, apply the floordiv proof,
          // and finally add the sign back.
          //
          // Rule 2:  (x / s) * s + x % s = x  (true for both trunc and floor div)
          //
          // General merge condition and proof:
          // - x = lhs->index % lhs->upper_factor
          // - s = lhs->scale / rhs->scale
          // - c = rhs->lower_factor
          //
          //    (x / (c * s)) * s + (x % (c * s)) / c
          // => ((x / c) / s) * s + ((x / c) % s)
          // => (x / c)
          //
          // Examples:
          //
          //    (z / 6) * 6 + ((z % 6) / 3) * 3
          // => ((z / 6) * 2 + (z % 6) / 3) * 3
          // => (z / 3) * 3
          // note: x = z, c = 3, s = 2
          //
          //    ((z % 12) / 6) * 6 + ((z % 6) / 3) * 3
          // => (((z % 12) / 6) * 2 + ((z % 12) % 6) / 3) * 3
          // => ((z % 12) / 3) * 3
          // note: x = z % 12, c = 3, s = 2
          // note also the invariance lhs->upper_factor % lhs->lower_factor == 0
          //
          SplitExprNode* merged = rhs.CopyOnWrite();
          merged->upper_factor = lhs->upper_factor;
          // reset args[i] to be zero.
          lhs.CopyOnWrite()->scale = 0;
          break;
        }
      }
    }
    // sort by the entry
    // Here we simply sort by descending order of scales.
    // For now, we do not compare by index because that comparison
    // can be runtime dependent and create inderminism.
    // we do not sort by index for now because it can be costly
    // to deep compare Exprs, and address of Vars can be runtime dependent.
    //
    auto fcompare = [](const SplitExpr& lhs, const SplitExpr& rhs) {
      // order by scale first
      if (lhs->scale > rhs->scale) return true;
      if (lhs->scale < rhs->scale) return false;
      // then order by factor
      if (lhs->lower_factor > rhs->lower_factor) return true;
      if (lhs->lower_factor < rhs->lower_factor) return false;
      // then order by upper factor
      if (lhs->upper_factor > rhs->upper_factor) return true;
      if (lhs->upper_factor < rhs->upper_factor) return false;
      // then order by div mode
      if (lhs->div_mode > rhs->div_mode) return true;
      if (lhs->div_mode < rhs->div_mode) return false;
      // tie.
      // TODO(tvm-team) We might consider index as the last comparison point,
      // after we make deep comparator more derministic.
      // Specifically, we can consider comparing names of vars and break ties with address.
      return false;
    };
    std::stable_sort(args.begin(), args.end(), fcompare);
    return args;
  }
  static PrimExpr Normalize_(DataType dtype,
                         const std::vector<SplitExpr>& args,
                         int64_t base) {
    // Positive scales first
    PrimExpr res = make_const(dtype, 0);
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale > 0) {
        res = res + args[i]->Normalize();
      }
    }
    if (base > 0) {
      res = res + make_const(dtype, base);
    }
    // negative scales follows using sub.
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale < 0) {
        res = res - args[i]->NormalizeWithScale(-1);
      }
    }
    if (base < 0) {
      res = res - make_const(dtype, -base);
    }
    return res;
  }
};

class SumExpr : public PrimExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SumExpr, PrimExpr, SumExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SumExprNode);
};

void SumExprNode::AddToSelf(const SumExpr& other, int64_t scale) {
  // NOTE: it is rare to have a balanced long expression,
  // linear scan is fine for our case.
  for (size_t i = 0; i < other->args.size(); ++i) {
    this->AddToSelf(other->args[i], scale);
  }
  this->AddToSelf(other->base * scale);
}

// Sub-class RewriteSimplifier::Impl to take benefit of
// rewriter for condition simplification etc.
class CanonicalSimplifier::Impl : public RewriteSimplifier::Impl {
 public:
  using Rewriter = RewriteSimplifier::Impl;

  explicit Impl(Analyzer* parent)
      : Rewriter(parent) {}


  PrimExpr CanonicalSimplify(PrimExpr expr) {
    expr = operator()(expr);
    return expr;
  }

  // override the original mutate function.
  PrimExpr VisitExpr(const PrimExpr& input_expr) final {
    auto expr = Rewriter::VisitExpr(input_expr);
    return Normalize(expr);
  }

  // Normal mutation without normalization.
  PrimExpr CanonicalMutate(PrimExpr expr) {
    return Rewriter::VisitExpr(expr);
  }

  using Rewriter::VisitExpr_;
  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const DivNode* op) final;
  PrimExpr VisitExpr_(const ModNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;
  PrimExpr VisitExpr_(const ReduceNode* op) final;

 private:
  /*!
   * \brief compute lhs / cval
   * \param lhs The left operand.
   * \param cval The constant value.
   * \param div_mode The division mode.
   * \return The result expression;
   */
  SplitExpr SplitDivConst(SplitExpr lhs, int64_t cval, DivMode div_mode);
  /*!
   * \brief compute lhs % cval
   * \param lhs The left operand.
   * \param cval The constant value.
   * \param div_mode The division mode.
   * \return The result expression;
   */
  SplitExpr SplitModConst(SplitExpr lhs, int64_t cval, DivMode div_mode);
  /*!
   * \brief Separate psum into divisible and non-divisible parts.
   * \param psum The sum expression.
   * \param coeff The co-efficient.
   * \param out_divisible The result divisible component.
   * \param out_non_divisible The non-divisible component.
   */
  void SeparateDivisibleParts(const SumExprNode* psum,
                              int64_t coeff,
                              SumExpr* out_divisible,
                              SumExpr* out_non_divisible);
  /*!
   * \brief Normalize expr to normal expr.
   * \param expr The input expression.
   * \return Normalized expr.
   */
  PrimExpr Normalize(PrimExpr expr) {
    if (const auto* op = expr.as<CanonicalExprNode>()) {
      return op->Normalize();
    } else {
      return expr;
    }
  }
  /*!
   * \brief Create a SplitExpr from expr.
   * \param expr The input expr.
   * \return The transformed SplitExpr.
   */
  SplitExpr ToSplitExpr(PrimExpr expr) {
    if (const auto* op = expr.as<SplitExprNode>()) {
      return GetRef<SplitExpr>(op);
    }
    if (const auto* op = expr.as<SumExprNode>()) {
      if (op->base == 0 && op->args.size() == 1) return op->args[0];
    }
    if (const auto* op = expr.as<CanonicalExprNode>()) {
      expr = op->Normalize();
    }
    ObjectPtr<SplitExprNode> n = make_object<SplitExprNode>();
    n->dtype = expr.dtype();
    n->index = std::move(expr);
    n->div_mode = kTruncDiv;
    return SplitExpr(n);
  }
  /*!
   * \brief Convert expr to an equivalent SplitExpr
   *        that has the specified div_mode.
   *
   * This function will return the same expr if its
   * div_mode already satisfies the need.
   *
   * \param expr The input expr.
   * \param div_mode The new div_mode.
   * \return The transformed SplitExpr.
   */
  SplitExpr ConvertDivMode(SplitExpr expr, DivMode div_mode) {
    if (expr->div_mode == div_mode) return expr;
    if (expr->DivModeCompatibleTo(div_mode)) {
      expr.CopyOnWrite()->div_mode = div_mode;
      return expr;
    }
    expr = ToSplitExpr(Normalize(expr));
    CHECK(expr->DivModeCompatibleTo(div_mode));
    expr.CopyOnWrite()->div_mode = div_mode;
    return expr;
  }
  /*!
   * \brief Create a SumExpr from expr.
   * \param expr The input expr.
   * \return The transformed SumExpr.
   */
  SumExpr ToSumExpr(PrimExpr expr) {
    if (const auto* op = expr.as<SumExprNode>()) {
      return GetRef<SumExpr>(op);
    }
    ObjectPtr<SumExprNode> n = make_object<SumExprNode>();
    n->dtype = expr.dtype();
    if (const auto* op = expr.as<IntImmNode>()) {
      n->base = op->value;
      return SumExpr(n);
    } else {
      n->args.emplace_back(ToSplitExpr(expr));
      return SumExpr(n);
    }
  }
  // Simplify the combiner used in reduce.
  PrimExpr SimplifyReduceCombiner(const ReduceNode* op);
};

PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const AddNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<AddNode>(a, b);
  if (const_res.defined()) return const_res;

  // canonical form simplification.
  SumExpr ret = ToSumExpr(std::move(a));

  if (const auto* op = b.as<IntImmNode>()) {
    ret.CopyOnWrite()->AddToSelf(op->value);
  } else if (const auto* op = b.as<SumExprNode>()) {
    ret.CopyOnWrite()->AddToSelf(GetRef<SumExpr>(op), 1);
  } else {
    ret.CopyOnWrite()->AddToSelf(ToSplitExpr(b), 1);
  }
  return std::move(ret);
}

PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const SubNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<SubNode>(a, b);
  if (const_res.defined()) return const_res;

  // canonical form simplification.
  SumExpr ret = ToSumExpr(std::move(a));

  if (const auto* op = b.as<IntImmNode>()) {
    ret.CopyOnWrite()->AddToSelf(-op->value);
  } else if (const auto* op = b.as<SumExprNode>()) {
    ret.CopyOnWrite()->AddToSelf(GetRef<SumExpr>(op), -1);
  } else {
    ret.CopyOnWrite()->AddToSelf(ToSplitExpr(b), -1);
  }
  return std::move(ret);
}


PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const MulNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<MulNode>(a, b);
  if (const_res.defined()) return const_res;

  // x * c
  if (a.as<IntImmNode>()) {
    std::swap(a, b);
  }
  if (const auto* bconst = b.as<IntImmNode>()) {
    if (a.as<SumExprNode>()) {
      SumExpr ret = Downcast<SumExpr>(std::move(a));
      ret.CopyOnWrite()->MulToSelf(bconst->value);
      return std::move(ret);
    } else {
      SplitExpr ret = ToSplitExpr(std::move(a));
      ret.CopyOnWrite()->MulToSelf(bconst->value);
      return std::move(ret);
    }
  }

  // normal path.
  a = Normalize(a);
  b = Normalize(b);
  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return MulNode::make(a, b);
  }
}

void CanonicalSimplifier::Impl::
SeparateDivisibleParts(const SumExprNode* psum,
                       int64_t coeff,
                       SumExpr* out_divisible,
                       SumExpr* out_non_divisible) {
  auto divisible = make_object<SumExprNode>();
  auto non_divisible = make_object<SumExprNode>();
  divisible->dtype = psum->dtype;
  non_divisible->dtype = psum->dtype;

  if (psum->base % coeff == 0) {
    divisible->base = psum->base;
  } else {
    non_divisible->base = psum->base;
  }
  for (const auto& e : psum->args) {
    if (e->scale % coeff == 0) {
      divisible->args.push_back(e);
    } else {
      non_divisible->args.push_back(e);
    }
  }
  *out_divisible = SumExpr(divisible);
  *out_non_divisible = SumExpr(non_divisible);
}

SplitExpr CanonicalSimplifier::Impl::
SplitDivConst(SplitExpr lhs, int64_t cval, DivMode div_mode) {
  CHECK_GT(cval, 0);
  lhs = ConvertDivMode(lhs, div_mode);

  // the following rule works for both floordiv and truncdiv
  if (lhs->scale % cval == 0) {
    lhs.CopyOnWrite()->scale /= cval;
    return lhs;
  }

  if (cval % lhs->scale == 0) {
    int64_t scaled_cval = cval / lhs->scale;
    if (lhs->upper_factor == SplitExprNode::kPosInf ||
        lhs->upper_factor % (lhs->lower_factor * scaled_cval) == 0) {
      // directly fold division.
      lhs.CopyOnWrite()->scale = 1;
      lhs.CopyOnWrite()->lower_factor *= scaled_cval;
      lhs->Verify();
      return lhs;
    } else if (lhs->upper_factor <= (lhs->lower_factor * scaled_cval)) {
      // (x % c1) / c2  => 0 when c2 >= c1
      return ToSplitExpr(make_zero(lhs.dtype()));
    } else {
      // move the upper_factor modular into index.
      lhs.CopyOnWrite()->index =
          ModImpl(lhs->index, make_const(lhs.dtype(), lhs->upper_factor), div_mode);
      lhs.CopyOnWrite()->upper_factor = SplitExprNode::kPosInf;
      lhs.CopyOnWrite()->scale = 1;
      lhs.CopyOnWrite()->lower_factor *= scaled_cval;
      lhs->Verify();
      return lhs;
    }
  }
  // directly return the split with cval == 1
  lhs = ToSplitExpr(Normalize(lhs));
  CHECK(lhs->DivModeCompatibleTo(div_mode));
  CHECK_EQ(lhs->scale, 1);
  lhs.CopyOnWrite()->lower_factor *= cval;
  return lhs;
}

PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const DivNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }

  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<DivNode>(a, b);
  if (const_res.defined()) return const_res;
  PVar<IntImm> c1;
  // x / c1
  if (c1.Match(b) && c1.Eval()->value > 0) {
    int64_t cval = c1.Eval()->value;
    if (cval == 1) return a;

    if (const auto* psum = a.as<SumExprNode>()) {
      SumExpr lhs, extra;
      SeparateDivisibleParts(psum, cval, &lhs, &extra);
      // can be divided by cval
      if (extra->IsZero()) {
        lhs.CopyOnWrite()->DivideBy(cval);
        return std::move(lhs);
      }
      // both lhs and extra are non-negative
      if (analyzer_->CanProveGreaterEqual(lhs->Normalize(), 0) &&
          analyzer_->CanProveGreaterEqual(extra->Normalize(), 0)) {
        lhs.CopyOnWrite()->DivideBy(cval);
        PrimExpr temp = Normalize(extra);
        if (const auto* pconst = temp.as<IntImmNode>()) {
          lhs.CopyOnWrite()->AddToSelf(pconst->value / cval);
        } else {
          // if 0 <= extra < cval, it means the extra can be eliminated.
          if (TryCompare(temp, cval) != kLT) {
            lhs.CopyOnWrite()->AddToSelf(
                SplitDivConst(ToSplitExpr(temp), cval, kTruncDiv), 1);
          }
        }
        return std::move(lhs);
      }
    } else {
      // if a >= 0 && a < cval, then result == 0
      auto cbound = analyzer_->const_int_bound(Normalize(a));
      if (cbound->min_value >= 0 && cbound->max_value < cval) {
        return make_zero(a.dtype());
      }
    }
    return SplitDivConst(ToSplitExpr(std::move(a)), cval, kTruncDiv);
  }
  // normal path
  a = Normalize(a);
  b = Normalize(b);
  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return DivNode::make(a, b);
  }
}

PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const FloorDivNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<FloorDivNode>(a, b);
  if (const_res.defined()) return const_res;
  PVar<IntImm> c1;
  // x / c1
  if (c1.Match(b) && c1.Eval()->value > 0) {
    int64_t cval = c1.Eval()->value;
    if (cval == 1) return a;

    if (const auto* psum = a.as<SumExprNode>()) {
      SumExpr lhs, extra;
      SeparateDivisibleParts(psum, cval, &lhs, &extra);
      if (extra->IsZero()) {
        lhs.CopyOnWrite()->DivideBy(cval);
        return std::move(lhs);
      }
      // continue simplification.
      lhs.CopyOnWrite()->DivideBy(cval);
      PrimExpr temp = Normalize(extra);
      if (const auto* pconst = temp.as<IntImmNode>()) {
        lhs.CopyOnWrite()->AddToSelf(floordiv(pconst->value, cval));
      } else {
        // if 0 <= extra < cval, it means the extra can be eliminated.
        if (!(TryCompare(temp, cval) == kLT && analyzer_->CanProveGreaterEqual(temp, 0))) {
          lhs.CopyOnWrite()->AddToSelf(
              SplitDivConst(ToSplitExpr(temp), cval, kFloorDiv), 1);
        }
      }
      return std::move(lhs);
    } else {
      // if a >= 0 && a < cval, then result == 0
      auto cbound = analyzer_->const_int_bound(Normalize(a));
      if (cbound->min_value >= 0 && cbound->max_value < cval) {
        return make_zero(a.dtype());
      }
    }
    return SplitDivConst(ToSplitExpr(std::move(a)), cval, kFloorDiv);
  }
  // normal path
  a = Normalize(a);
  b = Normalize(b);
  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return FloorDivNode::make(a, b);
  }
}

SplitExpr CanonicalSimplifier::Impl::
SplitModConst(SplitExpr lhs, int64_t cval, DivMode div_mode) {
  CHECK_GT(cval, 0);
  lhs = ConvertDivMode(lhs, div_mode);

  if (lhs->scale % cval == 0) {
    lhs.CopyOnWrite()->scale = 0;
    return lhs;
  }
  if (cval % lhs->scale == 0) {
    // (x * c1) % (c2 * c1) => (x % c2) * c1
    int64_t scaled_cval = cval / lhs->scale;
    //  (x / c1) % c2  =>  (x % (c1 * c2)) / c2
    int64_t new_upper_factor = lhs->lower_factor * scaled_cval;
    // try to see if we can reduce the existing upper modular.
    if (lhs->upper_factor == SplitExprNode::kPosInf ||
        lhs->upper_factor % new_upper_factor == 0) {
      // we gained a new upper factor that is smaller
      // than the original one
      // Perhaps there are more chances in simplifying the index
      // Do a recursive call to simplify the mod with the new factor.
      if (new_upper_factor < lhs->upper_factor &&
          lhs->upper_factor != SplitExprNode::kPosInf) {
        auto updated = ToSplitExpr(this->VisitExpr(ModImpl(
            lhs->index, make_const(lhs.dtype(), new_upper_factor), div_mode)));
        updated.CopyOnWrite()->scale = lhs->scale;
        // re-apply the lower_factor
        if (lhs->lower_factor != 1) {
          return SplitDivConst(updated, lhs->lower_factor, div_mode);
        } else {
          return updated;
        }
      } else {
        lhs.CopyOnWrite()->upper_factor = new_upper_factor;
        return lhs;
      }
    } else if (new_upper_factor % lhs->upper_factor == 0) {
      // (x % 2) % 4 => x % 2
      return lhs;
    }
  }
  // Normalize the value.
  lhs = ToSplitExpr(Normalize(lhs));
  CHECK(lhs->DivModeCompatibleTo(div_mode));
  CHECK_EQ(lhs->scale, 1);
  CHECK_EQ(lhs->lower_factor, 1);
  lhs.CopyOnWrite()->div_mode = div_mode;
  lhs.CopyOnWrite()->upper_factor = cval;
  return lhs;
}

PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const ModNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<ModNode>(a, b);
  if (const_res.defined()) return const_res;

  PVar<IntImm> c1;
  // x % c1
  if (c1.Match(b) && c1.Eval()->value > 0) {
    int64_t cval = c1.Eval()->value;
    if (const auto* psum = a.as<SumExprNode>()) {
      SumExpr lhs, extra;
      SeparateDivisibleParts(psum, cval, &lhs, &extra);
      if (extra->IsZero()) {
        return make_zero(a.dtype());
      }
      // both lhs and extra are non-negative
      if (analyzer_->CanProveGreaterEqual(lhs->Normalize(), 0) &&
          analyzer_->CanProveGreaterEqual(extra->Normalize(), 0)) {
        PrimExpr temp = Normalize(extra);
        if (temp.as<IntImmNode>()) {
          return truncmod(temp, c1.Eval());
        } else {
          // If temp < cval && temp >=0 then can remove the mod.
          if (TryCompare(temp, cval) == kLT) {
            return temp;
          } else {
            // contonue to use logic below.
            a = extra;
            psum = a.as<SumExprNode>();
            CHECK(psum != nullptr);
          }
        }
      }
      // Simplify the offset constant if necessary.
      // (x - 5) % 3 => (x - 2) % 3 if x - 5 >= 0
      auto cbound = analyzer_->const_int_bound(Normalize(a));
      int64_t new_base = psum->base % cval;
      if (cbound->min_value >= 0 &&
          cbound->min_value - psum->base + new_base >= 0) {
        SumExpr sum_expr = Downcast<SumExpr>(a);
        sum_expr.CopyOnWrite()->base = new_base;
        return SplitModConst(ToSplitExpr(std::move(sum_expr)), cval, kTruncDiv);
      }
    } else {
      // if a >= 0 && a < cval, then result == 0
      auto cbound = analyzer_->const_int_bound(Normalize(a));
      if (cbound->min_value >= 0 && cbound->max_value < cval) {
        return a;
      }
    }
    return SplitModConst(ToSplitExpr(std::move(a)), cval, kTruncDiv);
  }
  // normal path
  a = Normalize(a);
  b = Normalize(b);
  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return ModNode::make(a, b);
  }
}

PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const FloorModNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  PrimExpr const_res = TryConstFold<FloorModNode>(a, b);
  if (const_res.defined()) return const_res;

  PVar<IntImm> c1;
  // x % c1
  if (c1.Match(b) && c1.Eval()->value > 0) {
    int64_t cval = c1.Eval()->value;
    if (const auto* psum = a.as<SumExprNode>()) {
      SumExpr lhs, extra;
      SeparateDivisibleParts(psum, cval, &lhs, &extra);
      PrimExpr temp = Normalize(extra);
      if (temp.as<IntImmNode>()) {
        return floormod(temp, c1.Eval());
      } else {
        // If temp < cval && temp >=0 then can remove the mod.
        if (TryCompare(temp, cval) == kLT &&
            analyzer_->CanProveGreaterEqual(temp, 0)) {
          return temp;
        } else {
          // contonue to use logic below.
          a = extra;
          psum = a.as<SumExprNode>();
          CHECK(psum != nullptr);
        }
      }
      // Simplify the offset constant if necessary.
      // floormod(x - 5, 3) => floormod(x + 1, 3)
      int64_t new_base = floormod(psum->base, cval);
      SumExpr sum_expr = Downcast<SumExpr>(std::move(a));
      sum_expr.CopyOnWrite()->base = new_base;
      return SplitModConst(ToSplitExpr(std::move(sum_expr)), cval, kFloorDiv);
    } else {
      // if a >= 0 && a < cval, then result == a
      auto cbound = analyzer_->const_int_bound(Normalize(a));
      if (cbound->min_value >= 0 && cbound->max_value < cval) {
        return a;
      }
    }
    return SplitModConst(ToSplitExpr(std::move(a)), cval, kFloorDiv);
  }
  // normal path
  a = Normalize(a);
  b = Normalize(b);
  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return FloorModNode::make(a, b);
  }
}

// Simplify reduce expression.
PrimExpr CanonicalSimplifier::Impl::
SimplifyReduceCombiner(const ReduceNode* op) {
  // First simplify the results
  Array<PrimExpr> simplified_result;
  for (const auto& res : op->combiner->result) {
    PrimExpr new_res = this->VisitExpr(res);
    simplified_result.push_back(new_res);
  }

  // Which components to keep
  std::vector<int> used(op->combiner->result.size(), false);

  // This function recursively marks the used components starting from
  // the index idx
  std::function<void(int)> mark_used;
  mark_used = [&used, &simplified_result, op, &mark_used](size_t idx) {
    // if the idx-th component was marked as used before, do nothing
    if (used[idx]) return;
    used[idx] = true;

    // check if the idx-th result expr uses some lhs or rhs variables
    // and recursively mark the corresponding components
    for (size_t i = 0; i < simplified_result.size(); ++i)
      if (!used[i]) {
        if (ExprUseVar(simplified_result[idx], op->combiner->lhs[i]) ||
            ExprUseVar(simplified_result[idx], op->combiner->rhs[i]))
          mark_used(i);
      }
  };

  // mark all used components starting from the value_index
  mark_used(op->value_index);

  // components which have side effects should also be preserved
  for (size_t i = 0; i < used.size(); ++i) {
    if (HasSideEffect(op->source[i]) ||
        HasSideEffect(op->combiner->identity_element[i]) ||
        HasSideEffect(op->combiner->result[i])) {
      mark_used(i);
    }
  }

  int new_value_index = op->value_index;
  Array<PrimExpr> new_result;
  Array<PrimExpr> new_identity;
  Array<Var> new_lhs;
  Array<Var> new_rhs;
  Array<PrimExpr> new_source;

  // new stuff is old stuff which is used
  for (size_t i = 0; i < used.size(); ++i) {
    if (used[i]) {
      // We simplify the result and identity, but not the source
      new_result.push_back(simplified_result[i]);
      new_identity.push_back(this->VisitExpr(op->combiner->identity_element[i]));
      new_lhs.push_back(op->combiner->lhs[i]);
      new_rhs.push_back(op->combiner->rhs[i]);
      new_source.push_back(op->source[i]);
    } else if (static_cast<int>(i) < op->value_index) {
      // value_index should also be adjusted
      new_value_index--;
    }
  }

  CommReducer new_combiner =
      CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
  return ReduceNode::make(
      new_combiner, new_source, op->axis, op->condition, new_value_index);
}

PrimExpr CanonicalSimplifier::Impl::
VisitExpr_(const ReduceNode* op) {
  // Recursively call simplification when necessary.
  PrimExpr ret = RewriteSimplifier::Impl::VisitExpr_(op);
  op = ret.as<ReduceNode>();
  // already been simplified by const reduction axis removal
  if (op == nullptr) return ret;
  if (op->axis.empty()) {
    // Note that here we assume that the identity element is indeed identity. Without this
    // assumption we would have to perform a single iteration of the loop, i.e. use
    // `(*op->combiner.get())(op->combineop->identity_element, op->source)[op->value_index]`
    // instead of `op->source[op->value_index]`. The former may be more difficult to simplify.
    return this->VisitExpr(
        SelectNode::make(op->condition,
                     op->source[op->value_index],
                     op->combiner->identity_element[op->value_index]));
  }
  // combiner simplification.
  ret = SimplifyReduceCombiner(op);
  return ret;
}

PrimExpr CanonicalSimplifier::operator()(const PrimExpr& expr) {
  return impl_->CanonicalSimplify(expr);
}

void CanonicalSimplifier::Update(const Var& var,
                                 const PrimExpr& info,
                                 bool override) {
  impl_->Update(var, info, override);
}

CanonicalSimplifier::CanonicalSimplifier(Analyzer* parent)
    : impl_(new Impl(parent)) {
}

CanonicalSimplifier::~CanonicalSimplifier() {
  delete impl_;
}

}  // namespace arith
}  // namespace tvm
