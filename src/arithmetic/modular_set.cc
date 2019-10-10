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
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_functor_ext.h>
#include <limits>
#include <utility>
#include <unordered_map>
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace ir;

TVM_REGISTER_NODE_TYPE(ModularSetNode);

ModularSet::ModularSet(int64_t coeff, int64_t base) {
  auto node = make_node<ModularSetNode>();
  node->coeff = coeff;
  node->base = base;
  // finish construction.
  node_ = std::move(node);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ModularSetNode>([](const ModularSetNode *op, IRPrinter *p) {
    p->stream << "ModularSet("
              << "coeff=" << op->coeff << ", base="
              << op->base << ')';
  });


// internal entry for const int bound
struct ModularSetAnalyzer::Entry {
  int64_t coeff{1};
  int64_t base{0};

  Entry() = default;

  Entry(int64_t coeff, int64_t base) {
    CHECK_GE(coeff, 0);
    this->coeff = coeff;
    if (coeff != 0) {
      base = base % coeff;
      if (base < 0) base += coeff;
    }
    this->base = base;
  }

  bool is_const() const {
    return coeff == 0;
  }

  bool operator==(const Entry& other) const {
    return coeff == other.coeff && base == other.base;
  }

  bool operator==(const ModularSet& other) const {
    return other.defined() &&
        coeff == other->coeff && base == other->base;
  }
};

class ModularSetAnalyzer::Impl :
      public ExprFunctor<ModularSetAnalyzer::Entry(const Expr&)> {
 public:
  explicit Impl(Analyzer* parent)
      : parent_(parent) {}

  void Update(const Var& var,
              const ModularSet& info,
              bool override) {
    if (!override) {
      auto it = var_map_.find(var);
      if (it != var_map_.end()) {
        CHECK(it->second == info)
            << "Trying to update var \'" << var << "\'"
            << " with a different const bound: "
            << "original=" << ModularSet(it->second.coeff, it->second.base)
            << ", new=" << info;
      }
    }
    var_map_[var] = Entry(info->coeff, info->base);
  }

  // Detect useful constraints and use them in the analysis scope.
  std::function<void()> EnterConstraint(const Expr& constraint) {
    PVar<Var> var;
    PVar<Integer> coeff, base;
    // pattern match interesting constraints
    if ((truncmod(var, coeff) == base).Match(constraint) ||
        (floormod(var, coeff) == base).Match(constraint)) {
      Entry entry(coeff.Eval()->value, base.Eval()->value);
      return UpdateByIntersect(var.Eval(), entry);
    }
    return nullptr;
  }

  // Override visitor behaviors
  Entry VisitExprDefault_(const Node* op) final {
    return Everything();
  }

  Entry VisitExpr_(const Cast* op) final {
    return VisitExpr(op->value);
  }

  Entry VisitExpr_(const IntImm* op) final {
    return Entry(0, op->value);
  }

  Entry VisitExpr_(const UIntImm* op) final {
    if (op->value < std::numeric_limits<int64_t>::max()) {
      return Entry(0, static_cast<int>(op->value));
    } else {
      return Everything();
    }
  }

  Entry VisitExpr_(const Add* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    int64_t coeff = ZeroAwareGCD(a.coeff, b.coeff);
    return Entry(coeff, a.base + b.base);
  }

  Entry VisitExpr_(const Sub* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    int64_t coeff = ZeroAwareGCD(a.coeff, b.coeff);
    return Entry(coeff, a.base - b.base);
  }

  Entry VisitExpr_(const Mul* op) final {
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

  Entry DivByConst(const Expr& lhs,
                   int64_t val,
                   bool round_down) {
    Entry a = VisitExpr(lhs);
    CHECK_NE(val, 0);
    if (a.coeff % val == 0) {
      if (a.base == 0) {
        // a c x  / c -> a x
        return Entry(std::abs(a.coeff / val), 0);
      }
      // positive division have a clear rounding mode.
      // Only handle case where we clearly know we need to round down.
      if (a.base > 0 && val > 0 &&
          (round_down || parent_->CanProveGreaterEqual(lhs, 0))) {
        return Entry(a.coeff / val, a.base / val);
      }
    }
    return Everything();
  }

  Entry VisitExpr_(const Div* op) final {
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      return DivByConst(op->a, b.base, false);
    }
    return Everything();
  }

  Entry VisitExpr_(const FloorDiv* op) final {
    Entry b = VisitExpr(op->b);
    if (b.is_const()) {
      return DivByConst(op->a, b.base, true);
    }
    return Everything();
  }

  Entry VisitExpr_(const Min* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    return Union(a, b);
  }

  Entry VisitExpr_(const Max* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    return Union(a, b);
  }

  Entry VisitExpr_(const Select* op) final {
    Entry a = VisitExpr(op->true_value);
    Entry b = VisitExpr(op->false_value);
    return Union(a, b);
  }

  Entry VisitExpr_(const Call* op) final {
    // only special handle >> which can be
    // used for index calculation.
    if (op->is_intrinsic(Call::shift_right)) {
      return VisitRightShift(op);
    } else {
      return Everything();
    }
  }

  Entry VisitExpr_(const Variable* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return it->second;
    } else {
      return Everything();
    }
  }

  Entry VisitRightShift(const Call* op) {
    Entry b = VisitExpr(op->args[1]);
    // a c x  / c -> a x
    if (b.is_const()) {
      return DivByConst(op->args[0], 1 << b.base, true);
    }
    return Everything();
  }

 private:
  /*! \brief pointer to parent. */
  Analyzer* parent_{nullptr};
  // internal variable map
  std::unordered_map<Var, Entry, ExprHash, ExprEqual> var_map_;
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
    return [this, old, var]() {
      var_map_[var] = old;
    };
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
   * \brief Use Extended Euclidean algorithm to solve ax + by = gcd(a, b)
   * \param a The first coefficient.
   * \param b The second coefficient.
   * \param x The solution of x.
   * \param y The solution of y.
   * \return The GCD of a and b.
   */
  static int64_t ExtendedEuclidean(int64_t a, int64_t b, int64_t* x, int64_t* y) {
    // Extended Euclidean algorithm
    // if a < 0, the problem can be convert into
    // |a|* (-x) + b * y = gcd(|a|, b)
    //
    // initial condition:
    // a * 0 + b * 1 = b
    // a * 1 + b * 0 = a
    int64_t s = 0, old_s = 1;
    int64_t r = b, old_r = a >= 0 ? a : -a;
    // Iteration (r2 < r1):
    // a * x1 + b * y1 = r1
    // a * x2 + b * y2 = r2
    // The above two eqs can derive the following eq (q = r1 / r2)
    // a * (x1 - x2 * q) + b * (y1 - y2 * q) = r1 - r2 * q = r3
    // Because r3 < r2, the iteration can eventually terminate
    while (r != 0) {
      int64_t q = old_r / r;
      int64_t tmp = old_r;
      old_r = r;
      r = tmp - q * r;
      tmp = old_s;
      old_s = s;
      s = tmp - q * s;
    }

    *x = a >= 0 ? old_s : -old_s;
    if (b != 0) {
      *y = (old_r - (*x) * a) / b;
    } else {
      *y = 1;
    }

    return old_r;
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
   * \brief Take GCD of a and b.
   * \param a The first operand.
   * \param b The second operand.
   * \return The result.
   */
  static int64_t ZeroAwareGCD(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    if (a < b) std::swap(a, b);
    if (b == 0) return a;
    // perform GCD (greatest common divisor)
    // ax + by = gcd(a, b) z if a != 0, b != 0
    while (a % b != 0) {
      a = a % b;
      std::swap(a, b);
    }
    return b;
  }
  /*!
   * \brief return everything dtype can represent.
   * \return Bound that represent everything dtype can represent.
   */
  static Entry Everything() {
    return Entry(1, 0);
  }
  /*!
   * \brief return an empty set
   * \return Bound that represent everything dtype can represent.
   */
  static Entry Nothing() {
    return Entry(0, 1);
  }
};

ModularSet ModularSetAnalyzer::operator()(const Expr& expr) {
  Entry ret = impl_->VisitExpr(expr);
  return ModularSet(ret.coeff, ret.base);
}

void ModularSetAnalyzer::Update(const Var& var,
                                const ModularSet& info,
                                bool override) {
  impl_->Update(var, info, override);
}

std::function<void()> ModularSetAnalyzer::EnterConstraint(const Expr& constraint) {
  return impl_->EnterConstraint(constraint);
}

ModularSetAnalyzer::ModularSetAnalyzer(Analyzer* parent)
    : impl_(new Impl(parent)) {
}

ModularSetAnalyzer::~ModularSetAnalyzer() {
  delete impl_;
}

}  // namespace arith
}  // namespace tvm
