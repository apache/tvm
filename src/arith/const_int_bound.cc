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
 * \file tvm/arith/const_int_bound.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>

#include <algorithm>

#include "constraint_extract.h"
#include "int_operator.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

TVM_REGISTER_NODE_TYPE(ConstIntBoundNode);

ConstIntBound::ConstIntBound(int64_t min_value, int64_t max_value) {
  auto node = make_object<ConstIntBoundNode>();
  node->min_value = min_value;
  node->max_value = max_value;
  data_ = std::move(node);
}

ConstIntBound MakeConstIntBound(int64_t min_value, int64_t max_value) {
  return ConstIntBound(min_value, max_value);
}

TVM_REGISTER_GLOBAL("arith.ConstIntBound").set_body_typed(MakeConstIntBound);

inline void PrintBoundValue(std::ostream& os, int64_t val) {
  if (val == ConstIntBound::kPosInf) {
    os << "pos_inf";
  } else if (val == ConstIntBound::kNegInf) {
    os << "neg_inf";
  } else {
    os << val;
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstIntBoundNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ConstIntBoundNode*>(node.get());
      p->stream << "ConstIntBound[";
      PrintBoundValue(p->stream, op->min_value);
      p->stream << ',';
      PrintBoundValue(p->stream, op->max_value);
      p->stream << ']';
    });

// internal entry for const int bound
struct ConstIntBoundAnalyzer::Entry {
  int64_t min_value;
  int64_t max_value;

  bool is_const(int64_t value) const { return min_value == max_value && min_value == value; }

  bool operator==(const Entry& other) const {
    return min_value == other.min_value && max_value == other.max_value;
  }
};

class ConstIntBoundAnalyzer::Impl
    : public ExprFunctor<ConstIntBoundAnalyzer::Entry(const PrimExpr&)> {
 public:
  /*! \brief additional bound info about expr in bound */
  struct BoundInfo {
    /*! \brief The expr */
    PrimExpr expr;
    /*! \brief The additional bound */
    Entry bound;

    BoundInfo() {}
    BoundInfo(PrimExpr expr, Entry bound) : expr(expr), bound(bound) {}
  };

  void Bind(const Var& var, const Range& range, bool allow_override) {
    Entry a = VisitExpr(range->min);
    Entry b = VisitExpr(range->extent);
    Entry ret;
    ret.min_value = a.min_value;
    ret.max_value = InfAwareAdd(a.max_value, InfAwareAdd(b.max_value, -1));
    Update(var, ret, allow_override);
  }

  void Update(const Var& var, const Entry& info, bool allow_override) {
    if (!allow_override) {
      auto it = var_map_.find(var);
      if (it != var_map_.end()) {
        ICHECK(it->second == info)
            << "Trying to update var \'" << var << "\'"
            << " with a different const bound: "
            << "original=" << ConstIntBound(it->second.min_value, it->second.max_value)
            << ", new=" << ConstIntBound(info.min_value, info.max_value);
      }
    }
    var_map_[var] = info;
  }

  Entry VisitExpr_(const LetNode* op) final {
    auto it = var_map_.find(op->var);
    // if the var has not been binded, update the info.
    if (it == var_map_.end()) {
      var_map_[op->var] = this->VisitExpr(op->value);
      Entry ret = VisitExpr(op->body);
      var_map_.erase(op->var);
      return ret;
    } else {
      return VisitExpr(op->body);
    }
  }

  void Update(const Var& var, const ConstIntBound& info, bool allow_override) {
    Update(var, MakeBound(info->min_value, info->max_value), allow_override);
  }

  // Override visitor behaviors
  Entry VisitExprDefault_(const Object* op) final {
    return Everything(static_cast<const PrimExprNode*>(op)->dtype);
  }

  Entry VisitExpr(const PrimExpr& expr) final {
    Entry res = ExprFunctor::VisitExpr(expr);
    tir::ExprDeepEqual equal;
    // a linear search over additional info
    // assume we won't have a lot of conditions
    for (const BoundInfo& info : additional_info_) {
      if (equal(expr, info.expr)) {
        res = Intersect(res, info.bound);
      }
    }
    if (bound_) {
      auto val = bound_->find(expr);
      if (val != bound_->end()) {
        auto everything = Everything(expr->dtype);
        ICHECK(
            (val->second->min_value == res.min_value && val->second->max_value == res.max_value) ||
            (val->second->min_value == everything.min_value &&
             val->second->max_value == everything.max_value))
            << "Detected bound for " << expr << "conflicts with memorization";
      }
      (*bound_)[expr] = ConstIntBound(res.min_value, res.max_value);
    }
    return res;
  }

  Entry VisitExpr_(const RampNode* op) final {
    // op = {base + i * stride | 0 <= i < lanes}
    // Entry(op) = Union(Entry(base + i * stride) | 0 <= i < lanes)
    // Note that `base + i * stride` is linear w.r.t. `i`
    // Entry(op) = Union(Entry(base + i * stride) | i = 0, i = lanes-1)
    Entry a = VisitExpr(op->base);
    Entry b = VisitExpr(op->base + (op->lanes - 1) * op->stride);
    return Union(a, b);
  }

  Entry VisitExpr_(const BroadcastNode* op) final { return VisitExpr(op->value); }

  Entry VisitExpr_(const CastNode* op) final {
    Entry a;

    // int(ceil(log2(cast(n,"float64")))) is used as the
    // implementation of topi.math.ceil_log2, and appears in iteration
    // bounds.
    if (auto opt = FindCeilLog2Arg(op)) {
      a = CeilLog2Bounds(opt.value());
    } else {
      a = VisitExpr(op->value);
    }

    Entry b = Everything(op->dtype);
    return Intersect(a, b);
  }

  /*!
   * \brief Process the divisor by making assumption that divide by zero
   * won't happen in a valid program.
   *
   * This is important for us to get a lot of symbolic shape bound right
   * now that the shape n >= 0, but in cases
   * when mod or divide of n occur, the intention is actually n > 0
   *
   * \param divisor The input divsor entry
   * \return The processed entry
   */
  Entry AssumeNoZeroDivisor(Entry divisor) {
    ICHECK(!divisor.is_const(0)) << "Find divide by zero";
    // NOTE: here we make the assumption that
    // divide by zero won't happen in a valid program
    // this is important for us to get a lot of symbolic shape bound right
    // where most conditions know that the shape n >= 0, but in cases
    // when mod or divide of n occur, the intention is actually n > 0
    if (divisor.min_value == 0) {
      divisor.min_value = 1;
      ICHECK_GE(divisor.max_value, 1);
    }
    return divisor;
  }

  Entry VisitExpr_(const IntImmNode* op) final { return MakeBound(op->value, op->value); }

  Entry VisitExpr_(const AddNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = InfAwareAdd(a.min_value, b.min_value);
    ret.max_value = InfAwareAdd(a.max_value, b.max_value);
    return ret;
  }

  Entry VisitExpr_(const SubNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = InfAwareAdd(a.min_value, -b.max_value);
    ret.max_value = InfAwareAdd(a.max_value, -b.min_value);
    return ret;
  }

  Entry VisitExpr_(const MulNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    return BinaryOpBoundary(a, b, InfAwareMul);
  }

  Entry VisitExpr_(const DivNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = AssumeNoZeroDivisor(VisitExpr(op->b));
    return HandleDivision(a, b, op->dtype, InfAwareDiv);
  }

  Entry VisitExpr_(const ModNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = AssumeNoZeroDivisor(VisitExpr(op->b));

    if (b.min_value > 0) {
      int64_t b_max_cap = InfAwareAdd(b.max_value, -1);
      if (a.min_value >= 0) {
        // 0 <= [a_min, a_max] < b_min
        if (a.max_value < b.min_value) return a;
        // other case, we can get close to 0
        return MakeBound(0, std::min(a.max_value, b_max_cap));
      } else {
        return MakeBound(std::max(a.min_value, -b_max_cap),
                         std::min(std::max(a.max_value, (int64_t)0), b_max_cap));
      }
    } else {
      ICHECK(!b.is_const(0)) << "mod by zero";
      // mod by negative value is rare,
      // and we just use the simpliest rule.
      return Everything(op->dtype);
    }
  }

  Entry VisitExpr_(const FloorDivNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = AssumeNoZeroDivisor(VisitExpr(op->b));
    return HandleDivision(a, b, op->dtype, InfAwareFloorDiv);
  }

  Entry VisitExpr_(const FloorModNode* op) final {
    /* let a / b = x + y, where x is integer, y \in [0, 1)
     * floormod(a, b) = a - floordiv(a, b) * b
     * floordiv(a, b) = x
     * floormod(a, b) = a - floordiv(a, b) * b
     *                = a - x * b
     *                = a - (a / b - y) * b
     *                = a - a + y * b
     *                = y * b
     * note that 0 <= y < 1
     * when b > 0, 0 <= b * y < b
     *             0 <= b * y <= b - 1
     * when b < 0, b < b * y <= 0
     *             b + 1 <= b * y <= 0
     * In all cases, min(0, b + 1) <= b * y <= max(0, b - 1)
     *               min(0, b_min + 1) <= b * y <= max(0, b_max - 1)
     * That is, min(0, b_min + 1) <= floormod(a, b) <= max(0, b_max - 1)
     */
    Entry a = VisitExpr(op->a);
    Entry b = AssumeNoZeroDivisor(VisitExpr(op->b));

    if (b.min_value > 0) {
      int64_t b_max_cap = InfAwareAdd(b.max_value, -1);
      if (a.min_value >= 0) {
        // 0 <= [a_min, a_max] < b_min
        if (a.max_value < b.min_value) return a;
        // other case, we can get close to 0
        return MakeBound(0, std::min(a.max_value, b_max_cap));
      } else {
        return MakeBound(0, b_max_cap);
      }
    } else {
      ICHECK(!b.is_const(0)) << "floormod by zero";
      int64_t b_min_cap = InfAwareAdd(b.min_value, 1);
      int64_t b_max_cap = InfAwareAdd(b.max_value, -1);
      return Intersect(MakeBound(std::min(static_cast<int64_t>(0), b_min_cap),
                                 std::max(static_cast<int64_t>(0), b_max_cap)),
                       Everything(op->dtype));
    }
  }

  Entry VisitExpr_(const MinNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = std::min(a.min_value, b.min_value);
    ret.max_value = std::min(a.max_value, b.max_value);
    return ret;
  }

  Entry VisitExpr_(const MaxNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.min_value = std::max(a.min_value, b.min_value);
    ret.max_value = std::max(a.max_value, b.max_value);
    return ret;
  }

  Entry VisitExpr_(const SelectNode* op) final {
    Entry a = VisitExpr(op->true_value);
    Entry b = VisitExpr(op->false_value);
    return Union(a, b);
  }

  Entry VisitExpr_(const CallNode* op) final {
    // only special handle >> and & which can be
    // used for index calculation.

    if (op->op.same_as(tir::builtin::shift_right())) {
      return VisitRightShift(op);
    } else if (op->op.same_as(tir::builtin::shift_left())) {
      return VisitLeftShift(op);
    } else if (op->op.same_as(tir::builtin::bitwise_and())) {
      return VisitBitwiseAnd(op);
    } else {
      return Everything(op->dtype);
    }
  }

  Entry VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return Everything(op->dtype);
    }
  }

  Entry VisitExpr_(const SizeVarNode* op) final {
    SizeVar v = GetRef<SizeVar>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return MakeBound(0, kPosInf);
    }
  }

  Entry VisitLeftShift(const CallNode* op) {
    Entry a = VisitExpr(op->args[0]);
    Entry b = VisitExpr(op->args[1]);

    if (a.min_value < 0 || b.min_value < 0) {
      // If either operand can negative, we may run into undefined
      // behavior for some targets.  In these cases, avoid making any
      // assumptions about the result.
      return Everything(op->dtype);
    }

    return BinaryOpBoundary(a, b, InfAwareLeftShift);
  }

  Entry VisitRightShift(const CallNode* op) {
    Entry a = VisitExpr(op->args[0]);
    Entry b = VisitExpr(op->args[1]);
    return BinaryOpBoundary(a, b, InfAwareRightShift);
  }

  Entry VisitBitwiseAnd(const CallNode* op) {
    Entry a = VisitExpr(op->args[0]);
    Entry b = VisitExpr(op->args[1]);
    // handle positive index case.
    if (a.min_value >= 0 && b.min_value >= 0) {
      return MakeBound(0, std::min(a.max_value, b.max_value));
    } else {
      if (b.min_value >= 0) {
        return MakeBound(0, b.max_value);
      }
      if (a.min_value >= 0) {
        return MakeBound(0, a.max_value);
      }
      return Everything(op->dtype);
    }
  }

  std::function<void()> EnterConstraint(const PrimExpr& constraint) {
    std::vector<BoundInfo> info = DetectBoundInfo(constraint);
    if (info.size() == 0) return nullptr;
    size_t old_size = additional_info_.size();
    additional_info_.insert(additional_info_.end(), info.begin(), info.end());
    size_t new_size = old_size + info.size();
    auto frecover = [old_size, new_size, this]() {
      ICHECK_EQ(additional_info_.size(), new_size);
      additional_info_.resize(old_size);
    };
    return frecover;
  }

 private:
  friend class ConstIntBoundAnalyzer;
  // internal variable map
  std::unordered_map<Var, Entry, ObjectPtrHash, ObjectPtrEqual> var_map_;
  // additional bound info
  std::vector<BoundInfo> additional_info_;
  // look up table for memorization
  BoundMapType* bound_{nullptr};
  // constants: the limit value means umlimited
  // NOTE: kNegInf/kPosInf are used to represent infinity.
  static const constexpr int64_t kNegInf = ConstIntBound::kNegInf;
  static const constexpr int64_t kPosInf = ConstIntBound::kPosInf;
  static_assert(-kNegInf == kPosInf, "invariant of inf");
  // internal helper functions
  /*!
   * \brief Get boundary of binary op who are monotonic wrt to one argument.
   * \param a The entry of the left operand.
   * \param b The entry of the right operand.
   * \param op The operator.
   * \tparam F the operator function type.
   * \return The result.
   */
  template <typename F>
  static Entry BinaryOpBoundary(Entry a, Entry b, const F& op) {
    Entry ret;
    // The boundary point must be shihft of the original boundary.
    int64_t v1 = op(a.min_value, b.min_value);
    int64_t v2 = op(a.max_value, b.max_value);
    int64_t v3 = op(a.min_value, b.max_value);
    int64_t v4 = op(a.max_value, b.min_value);
    ret.min_value = std::min(std::min(std::min(v1, v2), v3), v4);
    ret.max_value = std::max(std::max(std::max(v1, v2), v3), v4);
    return ret;
  }
  /*!
   * \brief Get value boundaries of division (e.g. Div or FloorDiv).
   * \param a The entry of the left operand.
   * \param b The entry of the right operand.
   * \param dt The data type of the division operator.
   * \param op The division operator.
   * \tparam F the operator function type.
   * \return The result.
   */
  template <typename F>
  static Entry HandleDivision(Entry a, Entry b, DataType dt, const F& op) {
    // Here we have a / b.
    // The largest value of the division will be for the smallest (with
    // respect to the absolute value) value of b. If the range of b starts
    // at a negative value and ends at a positive one, narrow it down to
    // be closer to 0, because BinaryOpBoundary only checks end-points of
    // the domain ranges.
    // If the range of b contains 0, then some infinity will be involved
    if (b.min_value <= 0 && 0 <= b.max_value && dt.is_int()) {
      Entry b_neg = b.min_value < 0 ? MakeBound(b.min_value, -1) : Everything(dt);
      Entry b_pos = b.max_value > 0 ? MakeBound(1, b.max_value) : Everything(dt);

      Entry e_neg = BinaryOpBoundary(a, b_neg, op);
      Entry e_pos = BinaryOpBoundary(a, b_pos, op);

      return MakeBound(std::min(e_neg.min_value, e_pos.min_value),
                       std::max(e_neg.max_value, e_pos.max_value));
    } else if (b.min_value == 0 && dt.is_uint()) {
      // uints only have one sided bounds
      Entry assumed_b = MakeBound(1, b.max_value);
      return BinaryOpBoundary(a, assumed_b, op);
    }
    // If the range of b does not have 0, use BinaryOpBoundary.
    return BinaryOpBoundary(a, b, op);
  }
  /*!
   * \brief Compute x + y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareAdd(int64_t x, int64_t y) {
    if (x == kPosInf) {
      ICHECK(y != kNegInf);
      return kPosInf;
    }
    if (x == kNegInf) {
      ICHECK(y != kPosInf);
      return kNegInf;
    }
    if (y == kPosInf || y == kNegInf) return y;
    if (WillOverflow<AddNode>(x, y, kNegInf, kPosInf)) {
      if (x > 0) return kPosInf;
      return kNegInf;
    }
    return x + y;
  }
  /*!
   * \brief Compute x * y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareMul(int64_t x, int64_t y) {
    if (!WillOverflow<MulNode>(x, y, kNegInf, kPosInf)) return x * y;
    if ((x > 0 && y > 0) || (x < 0 && y < 0)) return kPosInf;
    return kNegInf;
  }
  /*!
   * \brief Compute x / y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareDiv(int64_t x, int64_t y) {
    ICHECK_NE(y, 0);
    if (x == kPosInf || x == kNegInf) {
      if (y > 0) return x;
      return -x;
    }
    return x / y;
  }
  /*!
   * \brief Compute floodiv(x, y), aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareFloorDiv(int64_t x, int64_t y) {
    ICHECK_NE(y, 0);
    if (x == kPosInf || x == kNegInf) {
      if (y > 0) return x;
      return -x;
    }
    return floordiv(x, y);
  }
  /*!
   * \brief Compute x << y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareLeftShift(int64_t x, int64_t y) {
    if (x == kPosInf || x == kNegInf) return x;

    // Can be replaced with std::bit_width in C++20
    auto bit_width = [](int64_t as_signed) {
      uint64_t val = std::abs(as_signed);
      int num_bits = 0;
      while (val) {
        ++num_bits;
        val >>= 1;
      }
      return num_bits;
    };
    int x_bits = bit_width(x);
    if (x_bits + y < 64) {
      return x << y;
    } else {
      return kPosInf;
    }
  }
  /*!
   * \brief Compute x >> y, aware of inf.
   * \param x The left operand.
   * \param y The right operand.
   * \return the result.
   */
  static int64_t InfAwareRightShift(int64_t x, int64_t y) {
    if (x == kPosInf || x == kNegInf) return x;
    return x >> y;
  }
  /*!
   * \brief Make a new bound entry.
   */
  static Entry MakeBound(int64_t min_value, int64_t max_value) {
    Entry e;
    e.min_value = (min_value == kPosInf) ? min_value - 1 : min_value;
    e.max_value = (max_value == kNegInf) ? max_value + 1 : max_value;
    return e;
  }
  /*!
   * \brief Create union of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Union(Entry a, Entry b) {
    Entry ret;
    ret.min_value = std::min(a.min_value, b.min_value);
    ret.max_value = std::max(a.max_value, b.max_value);
    return ret;
  }
  /*!
   * \brief Create intersect of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Intersect(Entry a, Entry b) {
    Entry ret;
    ret.min_value = std::max(a.min_value, b.min_value);
    ret.max_value = std::min(a.max_value, b.max_value);
    return ret;
  }
  /*!
   * \brief return everything dtype can represent.
   * \param dtype The data type.
   * \return Bound that represent everything dtype can represent.
   */
  static Entry Everything(DataType dtype) {
    if (!dtype.is_int() && !dtype.is_uint()) {
      return MakeBound(kNegInf, kPosInf);
    }
    Entry ret;
    int64_t vbits = dtype.bits() - static_cast<int>(dtype.is_int());
    if (dtype.is_uint()) {
      ret.min_value = 0;
    } else {
      if (vbits >= 63) {
        ret.min_value = kNegInf;
      } else {
        ret.min_value = -(static_cast<int64_t>(1) << vbits);
      }
    }
    if (vbits >= 63) {
      ret.max_value = kPosInf;
    } else {
      ret.max_value = (static_cast<int64_t>(1) << vbits) - 1;
    }
    return ret;
  }

  /*!
   * \brief Detect additional constant bound from cond, if any
   * \param cond The constraint condition.
   * \return List of detected bounds.
   */
  static std::vector<BoundInfo> DetectBoundInfo(const PrimExpr& cond) {
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;

    std::vector<BoundInfo> info;
    auto add_info = [&](const PrimExpr& expr, int64_t min_value, int64_t max_value) {
      // If the conditional is comparing two integers, do not assign a
      // value to them.
      if (!expr->IsInstance<IntImmNode>()) {
        info.push_back(BoundInfo(expr, MakeBound(min_value, max_value)));
      }
    };

    for (const auto& subexpr : ExtractConstraints(cond)) {
      // NOTE: The canonical form always uses <= or <, but a
      // user-supplied constraint from the python API might not be
      // canonicalized.
      if ((c <= x).Match(subexpr) || (x >= c).Match(subexpr)) {
        add_info(x.Eval(), c.Eval()->value, kPosInf);
      } else if ((c < x).Match(subexpr) || (x > c).Match(subexpr)) {
        add_info(x.Eval(), c.Eval()->value + 1, kPosInf);
      } else if ((x <= c).Match(subexpr) || (x >= c).Match(subexpr)) {
        add_info(x.Eval(), kNegInf, c.Eval()->value);
      } else if ((x < c).Match(subexpr) || (c > x).Match(subexpr)) {
        add_info(x.Eval(), kNegInf, c.Eval()->value - 1);
      } else if ((x == c).Match(subexpr) || (c == x).Match(subexpr)) {
        add_info(x.Eval(), c.Eval()->value, c.Eval()->value);
      }
    }

    return info;
  }

  /*!
   * \brief Extract the argument from int(ceil(log2(arg)))
   *
   * This expression is used as the implementation of
   * topi.math.ceil_log2, and can appear in iteration bounds.
   */
  static Optional<PrimExpr> FindCeilLog2Arg(const CastNode* op) {
    if (op->dtype.is_int()) {
      if (auto as_call = op->value.as<CallNode>()) {
        if (as_call->op.same_as(Op::Get("tir.ceil"))) {
          PrimExpr ceil_arg = as_call->args[0];
          if (auto arg_call = ceil_arg.as<CallNode>()) {
            if (arg_call->op.same_as(Op::Get("tir.log2"))) {
              PrimExpr log_arg = arg_call->args[0];
              return log_arg;
            }
          }
        }
      }
    }
    return NullOpt;
  }

  /*! \brief Propagate constraints through ceil(log2(arg))
   *
   * Helper function for CastNode visitor
   */
  Entry CeilLog2Bounds(PrimExpr arg) {
    if (auto as_float = arg.as<FloatImmNode>()) {
      // A cast from int to float may have already been simplified
      // out.  Normally we don't inspect floating-point arguments, but here we can
      int64_t val = std::ceil(std::log2(as_float->value));
      return MakeBound(val, val);
    } else {
      Entry arg_bounds = VisitExpr(arg);
      return MakeBound(std::ceil(std::log2(arg_bounds.min_value)),
                       std::ceil(std::log2(arg_bounds.max_value)));
    }
  }
};

ConstIntBound ConstIntBoundAnalyzer::operator()(const PrimExpr& expr) const {
  Entry ret = impl_->VisitExpr(expr);
  return ConstIntBound(ret.min_value, ret.max_value);
}

ConstIntBound ConstIntBoundAnalyzer::operator()(const PrimExpr& expr, BoundMapType* bound) {
  impl_->bound_ = bound;
  Entry ret = impl_->VisitExpr(expr);
  impl_->bound_ = nullptr;
  return ConstIntBound(ret.min_value, ret.max_value);
}

void ConstIntBoundAnalyzer::Update(const Var& var, const ConstIntBound& info, bool allow_override) {
  impl_->Update(var, info, allow_override);
}

void ConstIntBoundAnalyzer::Bind(const Var& var, const Range& range, bool allow_override) {
  impl_->Bind(var, range, allow_override);
}

std::function<void()> ConstIntBoundAnalyzer::EnterConstraint(const PrimExpr& constraint) {
  return impl_->EnterConstraint(constraint);
}

ConstIntBoundAnalyzer::ConstIntBoundAnalyzer(Analyzer* parent) : impl_(new Impl()) {}

ConstIntBoundAnalyzer::~ConstIntBoundAnalyzer() { delete impl_; }

}  // namespace arith
}  // namespace tvm
