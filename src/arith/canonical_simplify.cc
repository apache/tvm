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
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>

#include "const_fold.h"
#include "pattern_match.h"
#include "product_normal_form.h"
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
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "arith.CanonicalExpr";
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(CanonicalExprNode, PrimExprNode);
};

inline PrimExpr ModImpl(PrimExpr a, PrimExpr b, DivMode mode) {
  if (mode == kTruncDiv) {
    return truncmod(a, b);
  } else {
    ICHECK_EQ(mode, kFloorDiv);
    return floormod(a, b);
  }
}

inline PrimExpr DivImpl(PrimExpr a, PrimExpr b, DivMode mode) {
  if (mode == kTruncDiv) {
    return truncdiv(a, b);
  } else {
    ICHECK_EQ(mode, kFloorDiv);
    return floordiv(a, b);
  }
}

/*!
 * \brief check if value fits in dtype
 * \param value The value to be analyzed
 * \param dtype The target dtype
 * \param analyzer The analyzer
 * \return whether value fits in dtype
 */
bool CastIsSafe(DataType dtype, PrimExpr value, Analyzer* analyzer) {
  if (!IsIndexType(dtype)) {
    return false;
  }
  ConstIntBound bound = analyzer->const_int_bound(value);
  int64_t ubound = Downcast<IntImm>(max_value(dtype))->value;
  int64_t lbound = Downcast<IntImm>(min_value(dtype))->value;
  if (value.dtype().bits() <= dtype.bits() ||  // upcast is safe
      (bound->max_value <= ubound && bound->min_value >= lbound)) {
    return true;
  }
  return false;
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
  void Verify() const { ICHECK(upper_factor == kPosInf || upper_factor % lower_factor == 0); }

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
      ICHECK(!dtype.is_uint() || sscale > 0);
      res = res * make_const(dtype, sscale);
    }
    return res;
  }

  PrimExpr Normalize() const final { return NormalizeWithScale(1); }

  void MulToSelf(int64_t scale) { this->scale *= scale; }

  /*!
   * \brief check if cast can be pushed to sub-expressions
   * \param dtype The target datatype
   * \param analyzer The analyzer
   * \return whether the cast can be safely pushed to children
   */
  bool CanPushCastToChildren(DataType dtype, Analyzer* analyzer) const {
    // cast(dtype, index % upper_factor / lower_factor * scale) ==
    // cast(dtype, index) % upper_factor / lower_factor * scale
    // iff it is an upcast (dtype.bits >= self.dtype.bits) or all of
    // its intermediate results fit in the range of dtype
    if (dtype.bits() >= this->dtype.bits()) {
      return true;  // upcast is safe
    }
    PrimExpr res = this->index;
    if (this->scale == 0) {
      return true;
    }
    if (!CastIsSafe(dtype, res, analyzer)) {
      return false;
    }
    if (this->upper_factor != SplitExprNode::kPosInf) {
      res = ModImpl(res, make_const(this->dtype, this->upper_factor), div_mode);
      if (!CastIsSafe(dtype, res, analyzer)) {
        return false;
      }
    }
    if (this->lower_factor != 1) {
      res = DivImpl(res, make_const(this->dtype, this->lower_factor), div_mode);
      if (!CastIsSafe(dtype, res, analyzer)) {
        return false;
      }
    }
    if (this->scale != 1) {
      ICHECK(!this->dtype.is_uint() || this->scale > 0);
      res = res * make_const(this->dtype, this->scale);
      if (!CastIsSafe(dtype, res, analyzer)) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief self = cast(dtype, self)
   * \param dtype The target datatype
   */
  void PushCastToChildren(DataType dtype) {
    this->index = cast(dtype, this->index);
    this->dtype = dtype;
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
  bool IsZero() const { return base == 0 && args.size() == 0; }
  /*!
   * \brief Return the normal Expr that is equivalent to self.
   * \return The normal expression.
   */
  PrimExpr Normalize() const final {
    // quick path 1.
    if (this->args.size() == 0) {
      return make_const(this->dtype, this->base);
    }
    return Normalize_(this->dtype, SimplifySplitExprs(args), base);
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
    ICHECK_EQ(this->base % scale, 0);
    this->base /= scale;
    for (size_t i = 0; i < this->args.size(); ++i) {
      ICHECK_EQ(args[i]->scale % scale, 0);
      args[i].CopyOnWrite()->scale /= scale;
    }
  }
  /*!
   * \brief add constant value to self.
   * \param value to be added.
   */
  void AddToSelf(int64_t value) { this->base += value; }
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
      if (!args[j]->IndexEqual(other) || other->lower_factor > args[j]->lower_factor) {
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

  /*!
   * \brief check if cast can be pushed to sub-expressions
   * \param dtype The target datatype
   * \param analyzer The analyzer
   * \return whether the cast can be safely pushed to children
   */
  bool CanPushCastToChildren(DataType dtype, Analyzer* analyzer) const {
    bool is_min_value = dtype.bits() == 64 ? base == std::numeric_limits<int64_t>::lowest()
                                           : base == -(1LL << (dtype.bits() - 1));
    // cast(dtype, arg_1 + arg_2 + ... arg_n) ==
    // cast(dtype, arg_1) + ... + cast(dtype, arg_n)
    // iff it is an upcast (dtype.bits >= self.dtype.bits) or all of
    // its intermediate results fit in the range of dtype
    if (dtype.bits() >= this->dtype.bits()) {
      return true;  // upcast is safe
    }
    PrimExpr res = make_const(dtype, 0);
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale > 0) {
        res = res + args[i]->Normalize();
        if (!CastIsSafe(dtype, res, analyzer)) {
          return false;
        }
      }
    }
    if (base > 0 || is_min_value) {
      res = res + make_const(dtype, base);
      if (!CastIsSafe(dtype, res, analyzer)) {
        return false;
      }
    }
    // negative scales follows using sub.
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale < 0) {
        res = res - args[i]->NormalizeWithScale(-1);
        if (!CastIsSafe(dtype, res, analyzer)) {
          return false;
        }
      }
    }
    if (base < 0 && !is_min_value) {
      res = res - make_const(dtype, -base);
      if (!CastIsSafe(dtype, res, analyzer)) {
        return false;
      }
    }
    for (const auto& arg : args) {
      if (!arg->CanPushCastToChildren(dtype, analyzer)) {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief self = cast(dtype, self)
   * \param dtype The target datatype
   */
  void PushCastToChildren(DataType dtype) {
    for (auto& arg : args) {
      arg.CopyOnWrite()->PushCastToChildren(dtype);
    }
    this->dtype = dtype;
  }

  static constexpr const char* _type_key = "arith.SumExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(SumExprNode, CanonicalExprNode);

 private:
  /*!
   * \brief Simplify the args by merging SplitExprs
   * \param args The original list of arguments.
   * \return simplified version.
   */
  static std::vector<SplitExpr> SimplifySplitExprs(std::vector<SplitExpr> args) {
    // NOTE: This algorithm relies on the factor that args are divided into segments
    // and each segment is sorted in descending order of lower_factor.
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale == 0) continue;
      for (size_t j = i + 1; j < args.size(); ++j) {
        SplitExpr& lhs = args[i];
        SplitExpr& rhs = args[j];
        if (!lhs->IndexEqual(rhs)) break;
        if (lhs->upper_factor < rhs->lower_factor) break;
        if (lhs->upper_factor == rhs->upper_factor && lhs->lower_factor == rhs->lower_factor &&
            lhs->DivModeCompatibleTo(rhs->div_mode)) {
          // folding same co-efficient.
          rhs.CopyOnWrite()->scale += lhs->scale;
          lhs.CopyOnWrite()->scale = 0;
        } else if (lhs->lower_factor == rhs->upper_factor && rhs->scale != 0 &&
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
      // after we make deep comparator more deterministic.
      // Specifically, we can consider comparing names of vars and break ties with address.
      return false;
    };
    std::stable_sort(args.begin(), args.end(), fcompare);
    return args;
  }
  static PrimExpr Normalize_(DataType dtype, const std::vector<SplitExpr>& args, int64_t base) {
    bool is_min_value = dtype.bits() == 64 ? base == std::numeric_limits<int64_t>::lowest()
                                           : base == -(1LL << (dtype.bits() - 1));
    // Positive scales first
    PrimExpr res = make_const(dtype, 0);
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale > 0) {
        res = res + args[i]->Normalize();
      }
    }
    if (base > 0 || is_min_value) {
      res = res + make_const(dtype, base);
    }
    // negative scales follows using sub.
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->scale < 0) {
        res = res - args[i]->NormalizeWithScale(-1);
      }
    }
    if (base < 0 && !is_min_value) {
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SplitExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SplitExprNode*>(node.get());
      auto factor_str = [](int64_t f) {
        return f == SplitExprNode::kPosInf ? std::string("+inf") : std::to_string(f);
      };
      p->stream << "split(";
      p->Print(op->index);
      p->stream << ", lower=" << factor_str(op->lower_factor)
                << ", upper=" << factor_str(op->upper_factor) << ", scale=" << op->scale
                << ", div_mode=";
      switch (op->div_mode) {
        // No "default", so that the compiler will emit a warning if more div modes are
        // added that are not covered by the switch.
        case kTruncDiv:
          p->stream << "truncdiv";
          break;
        case kFloorDiv:
          p->stream << "floordiv";
          break;
      }
      p->stream << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SumExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SumExprNode*>(node.get());
      p->stream << "sum(base=" << op->base;
      for (const SplitExpr& s : op->args) {
        p->stream << ", ";
        p->Print(s);
      }
      p->stream << ')';
    });

// Sub-class RewriteSimplifier::Impl to take benefit of
// rewriter for condition simplification etc.
class CanonicalSimplifier::Impl : public RewriteSimplifier::Impl {
 public:
  using Rewriter = RewriteSimplifier::Impl;

  explicit Impl(Analyzer* parent) : Rewriter(parent) {}

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
  PrimExpr CanonicalMutate(PrimExpr expr) { return Rewriter::VisitExpr(expr); }

  using Rewriter::VisitExpr_;
  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const DivNode* op) final;
  PrimExpr VisitExpr_(const ModNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;
  PrimExpr VisitExpr_(const ReduceNode* op) final;
  PrimExpr VisitExpr_(const CastNode* op) final;
  PrimExpr VisitExpr_(const LTNode* op) final;

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
  void SeparateDivisibleParts(const SumExprNode* psum, int64_t coeff, SumExpr* out_divisible,
                              SumExpr* out_non_divisible);
  /*!
   * \brief Pattern match and check whether lhs is fully divisible by
   *        rhs using prod pattern simplification expressions.
   *
   * The following two relations holds for floordiv/mod and truncdiv/mod
   * Note that the relation do not hold for euclidean divide and mod.
   *
   * This is because the floordiv/mod and truncdiv/mod result can be
   * uniquely determined by the value of the realdiv result and the
   * relation holds for realdiv.
   *
   * - div((a0 * a1 * c), (b0 * b1 * c)) = div((a0 * a1), (b0 * b1))
   * - mod((a0 * a1 * c), (b0 * b1 * c)) = mod((a0 * a1), (b0 * b1)) * c
   *
   * \param lhs The left operand to be updated.
   * \param rhs The right operand to be updated.
   * \param common_scale The common scale between lhs and rhs.
   * \returns The simplified result if it is successful.
   * \note This simplification mainly target when rhs is symbolic.
   */
  bool ProdDivSimplify(PrimExpr* lhs, PrimExpr* rhs, PrimExpr* common_scale);
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
    if (auto op = expr.as<SplitExpr>()) {
      return op.value();
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
    ICHECK(expr->DivModeCompatibleTo(div_mode));
    expr.CopyOnWrite()->div_mode = div_mode;
    return expr;
  }
  /*!
   * \brief Create a SumExpr from expr.
   * \param expr The input expr.
   * \return The transformed SumExpr.
   */
  SumExpr ToSumExpr(PrimExpr expr) {
    if (auto op = expr.as<SumExpr>()) {
      return op.value();
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

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const AddNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Add>(a, b)) return const_res.value();

  // canonical form simplification.
  SumExpr ret = ToSumExpr(std::move(a));

  if (const auto* op = b.as<IntImmNode>()) {
    ret.CopyOnWrite()->AddToSelf(op->value);
  } else if (auto op = b.as<SumExpr>()) {
    ret.CopyOnWrite()->AddToSelf(op.value(), 1);
  } else {
    ret.CopyOnWrite()->AddToSelf(ToSplitExpr(b), 1);
  }
  return std::move(ret);
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const SubNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Sub>(a, b)) return const_res.value();

  // canonical form simplification.
  SumExpr ret = ToSumExpr(std::move(a));

  if (const auto* op = b.as<IntImmNode>()) {
    ret.CopyOnWrite()->AddToSelf(-op->value);
  } else if (auto op = b.as<SumExpr>()) {
    ret.CopyOnWrite()->AddToSelf(op.value(), -1);
  } else {
    ret.CopyOnWrite()->AddToSelf(ToSplitExpr(b), -1);
  }
  return std::move(ret);
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const MulNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Mul>(a, b)) return const_res.value();

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
  // this only happens when b is symbolic
  a = Normalize(a);
  b = Normalize(b);

  PrimExpr ret = MulAndNormalize(a, b);
  const MulNode* mul = ret.as<MulNode>();

  if (mul && mul->a.same_as(op->a) && mul->b.same_as(op->b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return ret;
  }
}

void CanonicalSimplifier::Impl::SeparateDivisibleParts(const SumExprNode* psum, int64_t coeff,
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

SplitExpr CanonicalSimplifier::Impl::SplitDivConst(SplitExpr lhs, int64_t cval, DivMode div_mode) {
  ICHECK_GT(cval, 0);
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
  ICHECK(lhs->DivModeCompatibleTo(div_mode));
  ICHECK_EQ(lhs->scale, 1);
  lhs.CopyOnWrite()->lower_factor *= cval;
  lhs.CopyOnWrite()->div_mode = div_mode;
  return lhs;
}

bool CanonicalSimplifier::Impl::ProdDivSimplify(PrimExpr* plhs, PrimExpr* prhs,
                                                PrimExpr* common_scale) {
  // the constant rhs case is covered by other simplifier so
  // we just skip to save the time
  if (prhs->as<IntImmNode>()) return false;
  // collect lhs products and try to eliminate by matching them to prod in rhs
  Array<Optional<PrimExpr>> lhs_prods;
  PrimExpr new_rhs = make_const(prhs->dtype(), 1);
  PrimExpr new_common_scale = make_const(prhs->dtype(), 1);
  int64_t lhs_cscale = 1, rhs_cscale = 1;
  int num_elimination = 0;

  // collect lhs product and constant scale.
  auto fcollect_lhs = [&](PrimExpr value) {
    if (auto* intimm = value.as<tir::IntImmNode>()) {
      lhs_cscale *= intimm->value;
    } else {
      lhs_prods.push_back(value);
    }
  };
  UnpackReduction<tir::MulNode>(*plhs, fcollect_lhs);

  // collect rhs product and try to eliminate when possible
  PEqualChecker<PrimExpr> deep_equal;
  auto fcollect_rhs = [&](PrimExpr value) {
    if (auto* intimm = value.as<tir::IntImmNode>()) {
      rhs_cscale *= intimm->value;
    } else {
      // try eliminate from lhs
      for (size_t i = 0; i < lhs_prods.size(); ++i) {
        if (lhs_prods[i].defined() && deep_equal(value, lhs_prods[i].value())) {
          lhs_prods.Set(i, NullOpt);
          ++num_elimination;
          new_common_scale = new_common_scale * value;
          return;
        }
      }
      // if elimination is not possible then construct the expression.
      new_rhs = new_rhs * value;
    }
  };
  UnpackReduction<tir::MulNode>(*prhs, fcollect_rhs);
  // find gcd of const scales.
  int64_t cscale_gcd = ZeroAwareGCD(lhs_cscale, rhs_cscale);
  lhs_cscale /= cscale_gcd;
  rhs_cscale /= cscale_gcd;
  // if no elimination is possible
  if (num_elimination == 0 && cscale_gcd == 1) return false;

  // construct prod via canonical form
  PrimExpr new_lhs = make_const(plhs->dtype(), 1);
  for (Optional<PrimExpr> val : lhs_prods) {
    if (val.defined()) new_lhs = new_lhs * val.value();
  }
  *plhs = new_lhs * make_const(plhs->dtype(), lhs_cscale);
  *prhs = new_rhs * make_const(prhs->dtype(), rhs_cscale);
  *common_scale = new_common_scale * make_const(prhs->dtype(), cscale_gcd);
  return true;
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const DivNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }

  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Div>(a, b)) return const_res.value();
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
          if (TryCompare(temp, cval) != CompareResult::kLT) {
            lhs.CopyOnWrite()->AddToSelf(SplitDivConst(ToSplitExpr(temp), cval, kTruncDiv), 1);
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
  PrimExpr scale;
  // note this is the case where b is not constant
  if (ProdDivSimplify(&a, &b, &scale)) {
    // use operator ver so it can constant fold if b == 1
    return truncdiv(a, b);
  }
  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Div(a, b);
  }
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const FloorDivNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<FloorDiv>(a, b)) return const_res.value();
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
        if (!(TryCompare(temp, cval) == CompareResult::kLT &&
              analyzer_->CanProveGreaterEqual(temp, 0))) {
          lhs.CopyOnWrite()->AddToSelf(SplitDivConst(ToSplitExpr(temp), cval, kFloorDiv), 1);
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
  PrimExpr scale;
  if (ProdDivSimplify(&a, &b, &scale)) {
    // use operator ver so it can const fold.
    return floordiv(a, b);
  }
  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return FloorDiv(a, b);
  }
}

SplitExpr CanonicalSimplifier::Impl::SplitModConst(SplitExpr lhs, int64_t cval, DivMode div_mode) {
  ICHECK_GT(cval, 0);
  lhs = ConvertDivMode(lhs, div_mode);

  if (lhs->scale % cval == 0) {
    lhs.CopyOnWrite()->scale = 0;
    return lhs;
  }
  if (cval % lhs->scale == 0) {
    // The rationale:
    //   (index % upper) / lower * scale % cval, given cval = scaled_cval * scale
    //   by the rule (x * c1) % (c2 * c1) => (x % c2) * c1,
    // = (index % upper) / lower % scaled_cval * scale
    //   by the rule (x / c1) % c2  =>  (x % (c1 * c2)) / c1,
    // = (index % upper) % (new_upper_factor) / lower * scale
    int64_t scaled_cval = cval / lhs->scale;
    int64_t new_upper_factor = lhs->lower_factor * scaled_cval;
    // try to see if we can reduce the existing upper modular.
    if (lhs->upper_factor == SplitExprNode::kPosInf || lhs->upper_factor % new_upper_factor == 0) {
      // we gained a new upper factor that is smaller
      // than the original one
      // Perhaps there are more chances in simplifying the index
      // Do a recursive call to simplify the mod with the new factor.
      if (new_upper_factor < lhs->upper_factor && lhs->upper_factor != SplitExprNode::kPosInf) {
        auto updated = ToSplitExpr(this->VisitExpr(
            ModImpl(lhs->index, make_const(lhs.dtype(), new_upper_factor), div_mode)));
        // re-apply the lower_factor
        if (lhs->lower_factor != 1) {
          auto ret = SplitDivConst(updated, lhs->lower_factor, div_mode);
          ret.CopyOnWrite()->MulToSelf(lhs->scale);
          return ret;
        } else {
          updated.CopyOnWrite()->MulToSelf(lhs->scale);
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
  ICHECK(lhs->DivModeCompatibleTo(div_mode));
  ICHECK_EQ(lhs->scale, 1);
  ICHECK_EQ(lhs->lower_factor, 1);
  lhs.CopyOnWrite()->div_mode = div_mode;
  lhs.CopyOnWrite()->upper_factor = cval;
  return lhs;
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const ModNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<Mod>(a, b)) return const_res.value();

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
          if (TryCompare(temp, cval) == CompareResult::kLT) {
            return temp;
          } else {
            // continue to use logic below.
            a = extra;
            psum = a.as<SumExprNode>();
            ICHECK(psum != nullptr);
          }
        }
      }
      // Simplify the offset constant if necessary.
      // (x - 5) % 3 => (x - 2) % 3 if x - 5 >= 0
      auto cbound = analyzer_->const_int_bound(Normalize(a));
      int64_t new_base = psum->base % cval;
      if (cbound->min_value >= 0 && cbound->min_value - psum->base + new_base >= 0) {
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

  PrimExpr scale;
  if (ProdDivSimplify(&a, &b, &scale)) {
    // use operator version here so it can const fold b == 1
    return truncmod(a, b) * scale;
  }

  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Mod(a, b);
  }
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const FloorModNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr a = this->CanonicalMutate(op->a);
  PrimExpr b = this->CanonicalMutate(op->b);

  // const folding
  if (auto const_res = TryConstFold<FloorMod>(a, b)) return const_res.value();

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
        if (TryCompare(temp, cval) == CompareResult::kLT &&
            analyzer_->CanProveGreaterEqual(temp, 0)) {
          return temp;
        } else {
          // continue to use logic below.
          a = extra;
          psum = a.as<SumExprNode>();
          ICHECK(psum != nullptr);
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

  PrimExpr scale;
  if (ProdDivSimplify(&a, &b, &scale)) {
    // use operator version here so it can const fold b == 1
    return floormod(a, b) * scale;
  }

  if (op->a.same_as(a) && op->b.same_as(b)) {
    return GetRef<PrimExpr>(op);
  } else {
    return FloorMod(a, b);
  }
}

// Simplify reduce expression.
PrimExpr CanonicalSimplifier::Impl::SimplifyReduceCombiner(const ReduceNode* op) {
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
        if (UsesVar(simplified_result[idx],
                    [v = op->combiner->lhs[i].get()](const VarNode* var) { return var == v; }) ||
            UsesVar(simplified_result[idx],
                    [v = op->combiner->rhs[i].get()](const VarNode* var) { return var == v; }))
          mark_used(i);
      }
  };

  // mark all used components starting from the value_index
  mark_used(op->value_index);

  // components which have side effects should also be preserved
  for (size_t i = 0; i < used.size(); ++i) {
    if (SideEffect(op->source[i]) > CallEffectKind::kReadState ||
        SideEffect(op->combiner->identity_element[i]) > CallEffectKind::kReadState ||
        SideEffect(op->combiner->result[i]) > CallEffectKind::kReadState ||
        (!op->init.empty() && SideEffect(op->init[i]) > CallEffectKind::kReadState)) {
      mark_used(i);
    }
  }

  int new_value_index = op->value_index;
  Array<PrimExpr> new_result;
  Array<PrimExpr> new_identity;
  Array<Var> new_lhs;
  Array<Var> new_rhs;
  Array<PrimExpr> new_source;
  Array<PrimExpr> new_init;

  // new stuff is old stuff which is used
  for (size_t i = 0; i < used.size(); ++i) {
    if (used[i]) {
      // We simplify the result and identity, but not the source
      new_result.push_back(simplified_result[i]);
      new_identity.push_back(this->VisitExpr(op->combiner->identity_element[i]));
      new_lhs.push_back(op->combiner->lhs[i]);
      new_rhs.push_back(op->combiner->rhs[i]);
      new_source.push_back(op->source[i]);
      if (!op->init.empty()) new_init.push_back(op->init[i]);
    } else if (static_cast<int>(i) < op->value_index) {
      // value_index should also be adjusted
      new_value_index--;
    }
  }

  CommReducer new_combiner = CommReducer(new_lhs, new_rhs, new_result, new_identity);
  return Reduce(new_combiner, new_source, op->axis, op->condition, new_value_index, new_init);
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const ReduceNode* op) {
  // Recursively call simplification when necessary.
  PrimExpr ret = RewriteSimplifier::Impl::VisitExpr_(op);
  op = ret.as<ReduceNode>();
  // already been simplified by const reduction axis removal
  if (op == nullptr) return ret;
  if (op->axis.empty()) {
    if (!op->init.empty()) {
      return this->VisitExpr(Select(op->condition,
                                    (*op->combiner.get())(op->init, op->source)[op->value_index],
                                    op->init[op->value_index]));
    }
    // Note that here we assume that the identity element is indeed identity. Without this
    // assumption we would have to perform a single iteration of the loop, i.e. use
    // `(*op->combiner.get())(op->combineop->identity_element, op->source)[op->value_index]`
    // instead of `op->source[op->value_index]`. The former may be more difficult to simplify.
    return this->VisitExpr(Select(op->condition, op->source[op->value_index],
                                  op->combiner->identity_element[op->value_index]));
  }
  // combiner simplification.
  ret = SimplifyReduceCombiner(op);
  return ret;
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const CastNode* op) {
  if (!IsIndexType(op->dtype)) {
    return Rewriter::VisitExpr_(op);
  }
  // normalize
  PrimExpr value = this->CanonicalMutate(op->value);
  // PushCastToChildren
  if (value.as<SumExprNode>()) {
    SumExpr se = Downcast<SumExpr>(value);
    if (se->CanPushCastToChildren(op->dtype, analyzer_)) {
      se.CopyOnWrite()->PushCastToChildren(op->dtype);
      return std::move(se);
    }
  }
  if (value.as<SplitExprNode>()) {
    SplitExpr se = Downcast<SplitExpr>(value);
    if (se->CanPushCastToChildren(op->dtype, analyzer_)) {
      se.CopyOnWrite()->PushCastToChildren(op->dtype);
      return std::move(se);
    }
  }
  return Rewriter::VisitExpr_(op);
}

PrimExpr CanonicalSimplifier::Impl::VisitExpr_(const LTNode* op) {
  // First convert a < b into a - b < 0
  PrimExpr expr = this->CanonicalMutate(op->a - op->b);
  // Case: x0 * s0 + x1 * s1 + ... + xn + c < 0, let d = gcd(s0, s1, ..., s{n-1}, c)
  // 1. if can prove -d < xn < d, then we can simplify
  //    the expression to x0 * (s0/d) + x1 * (s1/d) + ... + x{n-1} * (s{n-1}/d) < c/d,
  //    e.g. `x * 8 + y < 16` where `y` \in [0, 8), we can simplify it to `x < 2`
  // 2. if xn is in pattern of yn % m, where m % d == 0, convert it to yn // d % (m/d)
  //    e.g. `x1 * 64 + (x2 * 8 + x3) % 64 < 120`, `x3` \in [0, 8), we can simplify it to
  //    `x1 * 8 + (x2 * 8 + x3) // 8 % 8 < 15` ==> `x1 * 8 + x2 % 8 < 15`

  if (const auto* lhs = expr.as<SumExprNode>()) {
    int64_t gcd = lhs->base;
    bool has_non_one_scale = false;
    for (const SplitExpr& split_expr : lhs->args) {
      if (split_expr->scale > 1 || split_expr->scale < -1) {
        has_non_one_scale = true;
        gcd = ZeroAwareGCD(gcd, std::abs(split_expr->scale));
      }
    }
    // Skip if gcd == 1 or all s_n are 1
    if (!has_non_one_scale || gcd <= 1) {
      return Rewriter::VisitExpr_(op);
    }
    SumExpr divisible, extra;
    SeparateDivisibleParts(lhs, gcd, &divisible, &extra);
    DataType dtype = divisible->dtype;
    ICHECK(extra->dtype == dtype);
    PrimExpr normal_extra = extra->Normalize();
    if (this->analyzer_->CanProve(normal_extra < make_const(dtype, gcd)) &&
        this->analyzer_->CanProve(normal_extra > make_const(dtype, -gcd))) {
      // Case 1. -d < xn < d
      divisible.CopyOnWrite()->DivideBy(gcd);
      return Rewriter::VisitExpr(divisible->Normalize() < make_zero(dtype));
    } else if (extra->args.size() == 1 &&
               extra->args[0]->upper_factor != ConstIntBoundNode::kPosInf &&
               extra->args[0]->upper_factor % (gcd * extra->args[0]->lower_factor) == 0) {
      // Case 2. xn == yn % m, where m % d == 0
      divisible.CopyOnWrite()->DivideBy(gcd);
      const auto split_expr = extra->args[0];
      int64_t lower_factor = gcd * extra->args[0]->lower_factor;
      PrimExpr extra_expr = floormod(floordiv(split_expr->index, lower_factor),
                                     floordiv(split_expr->upper_factor, lower_factor));
      return Rewriter::VisitExpr(divisible->Normalize() + extra_expr < make_zero(dtype));
    }
  }

  return Rewriter::VisitExpr_(op);
}

PrimExpr CanonicalSimplifier::operator()(const PrimExpr& expr) {
  return impl_->CanonicalSimplify(expr);
}

void CanonicalSimplifier::Update(const Var& var, const PrimExpr& info, bool override) {
  impl_->Update(var, info, override);
}

CanonicalSimplifier::CanonicalSimplifier(Analyzer* parent) : impl_(new Impl(parent)) {}

CanonicalSimplifier::~CanonicalSimplifier() { delete impl_; }

}  // namespace arith
}  // namespace tvm
