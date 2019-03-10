/*!
 *  Copyright (c) 2019 by Contributors
 * \file modular_set.cc
 * \brief Modular set analysis
 */
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_functor_ext.h>
#include <limits>
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace ir;

TVM_REGISTER_NODE_TYPE(ModularSetNode);

ModularSet ModularSetNode::make(int64_t coeff, int64_t base) {
  auto node = make_node<ModularSetNode>();
  node->coeff = coeff;
  node->base = base;
  return ModularSet(node);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ModularSetNode>([](const ModularSetNode *op, IRPrinter *p) {
    p->stream << "ModularSet("
              << "coeff=" << op->coeff << ", base="
              << op->base << ')';
  });


// internal entry for const int bound
// This condition holds for all instances: coeff >= 0, base in [0, coeff]
struct ModularSetAnalyzer::Entry {
  int64_t coeff{1};
  int64_t base{0};

  Entry() = default;

  Entry(int64_t coeff, int64_t base) {
    this->coeff = coeff;

    if (coeff < 0) {
      coeff = -coeff;
    }

    if (coeff != 0) {
      base = base % coeff;
      if (base < 0) base += coeff;
    }
    this->base = base;
  }

  bool is_const() const {
    return coeff == 0;
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
      CHECK(!var_map_.count(var));
    }
    var_map_[var] = Entry(info->coeff, info->base);
  }

  // Detect useful constraints and use them in the analysis scope.
  std::function<void()> EnterConstraint(const Expr& constraint) {
    PVar<Var> var;
    PVar<Integer> coeff, base;
    // pattern match interesting constraints
    if (((var % coeff) == base).Match(constraint)) {
      return UpdateByIntersect(var.Eval(), Entry(coeff.Eval()->value, base.Eval()->value));
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
    int64_t coeff = GCD(a.coeff, b.coeff);
    return Entry(coeff, a.base + b.base);
  }

  Entry VisitExpr_(const Sub* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    int64_t coeff = GCD(a.coeff, b.coeff);
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

    int64_t coeff = GCD(pq, GCD(pm, qn));
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
    int64_t coeff = GCD(a.coeff, b.coeff);
    if (coeff == 0) {
      if (a.base == b.base) return a;
      return Everything();
    }
    int64_t base0 = a.base % coeff;
    int64_t base1 = b.base % coeff;
    if (base0 == base1) {
      return Entry(coeff, base0);
    } else {
      return Entry(GCD(GCD(base0, base1), coeff), 0);
    }
  }
  /*!
   * \brief Create intersection of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Intersect(Entry x, Entry y) {
    int64_t n, m;
    int64_t a = x.coeff, b = x.base, c = y.coeff, d = y.base;
    int64_t gcd = ExtendedEuclidean(a, c, &n, &m);
    int64_t v = d - b;
    if (v % gcd == 0) {
      n = v / gcd * n;
      m = v / gcd * (-m);

      int64_t coeff = a / gcd * c;
      return Entry(coeff, n*a + b);
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
  static int64_t GCD(int64_t a, int64_t b) {
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
   * \brief Use Extended Euclidean algorithm to solve ax + by = gcd(a, b)
   * \param a The first coefficient.  (a >= 0)
   * \param b The second coefficient. (b >= 0)
   * \param x The solution of x.
   * \param y The solution of y.
   * \return The GCD of a and b.
   */
  static int64_t ExtendedEuclidean(int64_t a, int64_t b, int64_t *x, int64_t *y) {
    int64_t s = 0, old_s = 1;
    int64_t r = b, old_r = a;

    while (r != 0) {
      int64_t q = old_r / r;
      int64_t tmp = old_r;
      old_r = r;
      r = tmp - q * r;
      tmp = old_s;
      old_s = s;
      s = tmp - q * s;
    }

    *x = old_s;
    if (b != 0) {
      *y = (old_r - old_s * a) / b;
    } else {
      *y = 1;
    }

    return old_r;
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
   * \return An empty modular set.
   */
  static Entry Nothing() {
    return Entry(0, 1);
  }
};

ModularSet ModularSetAnalyzer::operator()(const Expr& expr) {
  Entry ret = impl_->VisitExpr(expr);
  return ModularSetNode::make(ret.coeff, ret.base);
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
