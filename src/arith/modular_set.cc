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
 * \file modular_set.cc
 * \brief Modular set analysis
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>

#include <limits>
#include <unordered_map>
#include <utility>

#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

TVM_REGISTER_NODE_TYPE(ModularSetNode);

ModularSet::ModularSet(int64_t coeff, int64_t base) {
  auto node = make_object<ModularSetNode>();
  node->coeff = coeff;
  node->base = base;
  // finish construction.
  data_ = std::move(node);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ModularSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ModularSetNode*>(node.get());
      p->stream << "ModularSet("
                << "coeff=" << op->coeff << ", base=" << op->base << ')';
    });

ModularSet MakeModularSet(int64_t coeff, int64_t base) { return ModularSet(coeff, base); }

TVM_REGISTER_GLOBAL("arith.ModularSet").set_body_typed(MakeModularSet);

// internal entry for const int bound
struct ModularSetAnalyzer::Entry {
  int64_t coeff{1};
  int64_t base{0};

  Entry() = default;

  Entry(int64_t coeff, int64_t base) {
    if (coeff < 0) {
      // `analyzer->canonical_simplify()` can generate expressions with
      // negative coefficients (e.g. simplifying `floormod(-i, 2)`
      // into `floormod(i, -2) * -1`).  When this happens, the
      // ModularSet may enter a constraint based on this expression.
      //
      // Handling a negative coeff uses the same sign convention as
      // canonical_simplify, requiring that
      // `floormod(var, coeff) == -floormod(var, -coeff)`.
      coeff *= -1;
      base *= -1;
    }
    this->coeff = coeff;
    if (coeff != 0) {
      base = base % coeff;
      if (base < 0) base += coeff;
    }
    this->base = base;
  }

  bool is_const() const { return coeff == 0; }

  bool operator==(const Entry& other) const { return coeff == other.coeff && base == other.base; }

  bool operator==(const ModularSet& other) const {
    return other.defined() && coeff == other->coeff && base == other->base;
  }
};

class ModularSetAnalyzer::Impl : public ExprFunctor<ModularSetAnalyzer::Entry(const PrimExpr&)> {
 public:
  explicit Impl(Analyzer* parent) : parent_(parent) {}

  void Update(const Var& var, const ModularSet& info, bool allow_override) {
    if (!allow_override) {
      auto it = var_map_.find(var);
      if (it != var_map_.end()) {
        ICHECK(it->second == info)
            << "Trying to update var \'" << var << "\'"
            << " with a different const bound: "
            << "original=" << ModularSet(it->second.coeff, it->second.base) << ", new=" << info;
      }
    }
    var_map_[var] = Entry(info->coeff, info->base);
  }

  // Detect useful constraints and use them in the analysis scope.
  std::function<void()> EnterConstraint(const PrimExpr& constraint) {
    PVar<Var> var;
    PVar<IntImm> coeff, base;
    // pattern match interesting constraints
    if ((truncmod(var, coeff) == base).Match(constraint) ||
        (floormod(var, coeff) == base).Match(constraint)) {
      Entry entry(coeff.Eval()->value, base.Eval()->value);
      return UpdateByIntersect(var.Eval(), entry);
    }
    if ((var == base).Match(constraint) || (base == var).Match(constraint)) {
      Entry entry(1, base.Eval()->value);
      return UpdateByIntersect(var.Eval(), entry);
    }
    return nullptr;
  }

  // Override visitor behaviors
  Entry VisitExprDefault_(const Object* op) final { return Everything(); }

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

  Entry VisitExpr_(const CastNode* op) final { return VisitExpr(op->value); }

  Entry VisitExpr_(const IntImmNode* op) final { return Entry(0, op->value); }

  Entry VisitExpr_(const AddNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    int64_t coeff = ZeroAwareGCD(a.coeff, b.coeff);
    return Entry(coeff, a.base + b.base);
  }

  Entry VisitExpr_(const SubNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    int64_t coeff = ZeroAwareGCD(a.coeff, b.coeff);
    return Entry(coeff, a.base - b.base);
  }

  Entry VisitExpr_(const MulNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    // Simplification rule, x, y, z are in Z
    // (p x + n) (q y + m)
    // -> pq xy + pm x + qn y + mn
    // -> pq z + pm x + qn y + mn
    int64_t pq = a.coeff * b.coeff;
    int64_t pm = a.coeff * b.base;
    int64_t qn = a.base * b.coeff;
    int64_t coeff = ZeroAwareGCD(pq, ZeroAwareGCD(pm, qn));
    return Entry(coeff, a.base * b.base);
  }

  Entry DivByConst(const PrimExpr& lhs, int64_t val, bool round_down) {
    Entry a = VisitExpr(lhs);
    ICHECK_NE(val, 0);
    if (a.coeff % val == 0) {
      if (a.base == 0) {
        // a c x  / c -> a x
        return Entry(std::abs(a.coeff / val), 0);
      }
      // positive division have a clear rounding mode.
      // Only handle case where we clearly know we need to round down.
      if (a.base > 0 && val > 0 && (round_down || parent_->CanProveGreaterEqual(lhs, 0))) {
        return Entry(a.coeff / val, a.base / val);
      }
    }
    return Everything();
  }

  Entry VisitExpr_(const DivNode* op) final {
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      return DivByConst(op->a, b.base, false);
    }
    return Everything();
  }

  Entry VisitExpr_(const FloorDivNode* op) final {
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      return DivByConst(op->a, b.base, true);
    }
    return Everything();
  }

  Entry VisitExpr_(const MinNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    return Union(a, b);
  }

  Entry VisitExpr_(const MaxNode* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    return Union(a, b);
  }

  Entry VisitExpr_(const SelectNode* op) final {
    Entry a = VisitExpr(op->true_value);
    Entry b = VisitExpr(op->false_value);
    return Union(a, b);
  }

  Entry ModByConst(const PrimExpr& lhs, int64_t val, bool round_down) {
    Entry a = VisitExpr(lhs);
    ICHECK_NE(val, 0);
    int64_t coeff = ZeroAwareGCD(a.coeff, val);
    if (a.base % coeff == 0 ||
        (a.base > 0 && (round_down || parent_->CanProveGreaterEqual(lhs, 0)))) {
      return Entry(coeff, a.base % coeff);
    }
    return Everything();
  }

  Entry VisitExpr_(const FloorModNode* op) final {
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      return ModByConst(op->a, b.base, true);
    }
    return Everything();
  }

  Entry VisitExpr_(const ModNode* op) final {
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      return ModByConst(op->a, b.base, false);
    }
    return Everything();
  }

  Entry VisitExpr_(const CallNode* op) final {
    // only special handle >> which can be
    // used for index calculation.
    if (op->op.same_as(tir::builtin::shift_right())) {
      return VisitRightShift(op);
    } else if (op->op.same_as(tir::builtin::bitwise_and())) {
      return VisitBitwiseAnd(op);
    } else {
      return Everything();
    }
  }

  Entry VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return Everything();
    }
  }

  Entry VisitRightShift(const CallNode* op) {
    Entry b = VisitExpr(op->args[1]);
    // a c x  / c -> a x
    if (b.is_const()) {
      return DivByConst(op->args[0], static_cast<int64_t>(1) << b.base, true);
    }
    return Everything();
  }

  Entry VisitBitwiseAnd(const CallNode* op) {
    Entry b = VisitExpr(op->args[1]);
    if (b.is_const()) {
      int shift;
      if (is_const_power_of_two_integer(Integer(b.base + 1), &shift)) {
        return ModByConst(op->args[0], static_cast<int64_t>(1) << shift, true);
      }
    }
    return Everything();
  }

 private:
  /*! \brief pointer to parent. */
  Analyzer* parent_{nullptr};
  // internal variable map
  std::unordered_map<Var, Entry> var_map_;
  /*!
   * \brief Update var by intersecting entry with var's current set.
   * \param var The variable.
   * \param entry The entry to be updated.
   * \return The recovery function of the scope.
   */
  std::function<void()> UpdateByIntersect(const Var& var, Entry entry) {
    Entry old = Everything();
    auto it = var_map_.find(var);
    if (it != var_map_.end()) {
      old = it->second;
    }
    var_map_[var] = Intersect(old, entry);
    // reover function.
    return [this, old, var]() { var_map_[var] = old; };
  }
  /*!
   * \brief Create union of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Union(Entry a, Entry b) {
    // {ax + y} \cup {bz + h} => {gcd(a, b) x + {y or h}}
    int64_t coeff = ZeroAwareGCD(a.coeff, b.coeff);
    if (coeff == 0) {
      if (a.base == b.base) return a;
      return Everything();
    }
    int64_t base0 = a.base % coeff;
    int64_t base1 = b.base % coeff;
    if (base0 == base1) {
      return Entry(coeff, base0);
    } else {
      return Entry(ZeroAwareGCD(ZeroAwareGCD(base0, base1), coeff), base0);
    }
  }

  /*!
   * \brief Create interect of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Intersect(Entry a, Entry b) {
    int64_t x, y;
    int64_t c1 = a.coeff, b1 = a.base, c2 = b.coeff, b2 = b.base;
    // z = c1 * p + b1
    // z = c2 * q + b2
    // c1 * x + c2 * y = gcd(c1, c2)
    // -> c1 * p - c2 * q = b2 - b1
    // -> p = (b2 - b1) / gcd * x
    // -> q = (b2 - b1) / gcd * (-y)
    // -> z = LCM(x, y) * k + (c1 * p + b1)
    int64_t gcd = ExtendedEuclidean(c1, c2, &x, &y);
    int64_t v = b2 - b1;
    if (v % gcd == 0) {
      x = v / gcd * x;
      y = v / gcd * (-y);
      int64_t coeff = c1 / gcd * c2;
      return Entry(coeff, x * c1 + b1);
    } else {
      return Nothing();
    }
  }
  /*!
   * \brief return everything dtype can represent.
   * \return Bound that represent everything dtype can represent.
   */
  static Entry Everything() { return Entry(1, 0); }
  /*!
   * \brief return an empty set
   * \return Bound that represent everything dtype can represent.
   */
  static Entry Nothing() { return Entry(0, 1); }
};

ModularSet ModularSetAnalyzer::operator()(const PrimExpr& expr) {
  Entry ret = impl_->VisitExpr(expr);
  return ModularSet(ret.coeff, ret.base);
}

void ModularSetAnalyzer::Update(const Var& var, const ModularSet& info, bool allow_override) {
  impl_->Update(var, info, allow_override);
}

std::function<void()> ModularSetAnalyzer::EnterConstraint(const PrimExpr& constraint) {
  return impl_->EnterConstraint(constraint);
}

ModularSetAnalyzer::ModularSetAnalyzer(Analyzer* parent) : impl_(new Impl(parent)) {}

ModularSetAnalyzer::~ModularSetAnalyzer() { delete impl_; }

}  // namespace arith
}  // namespace tvm
