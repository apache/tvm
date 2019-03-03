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
struct ModularSetAnalyzer::Entry {
  int64_t coeff{1};
  int64_t base{0};

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
    Entry e;
    e.coeff = info->coeff;
    e.base = info->base;
    var_map_[var] = e;
  }

  // Detect useful constraints and use them in the analysis scope.
  std::function<void()> EnterConstraint(const Expr& constraint) {
    PVar<Var> var;
    PVar<Integer> coeff, base;
    // pattern match interesting constraints
    if (((var % coeff) == base).Match(constraint)) {
      Entry entry;
      entry.coeff = coeff.Eval()->value;
      entry.base = base.Eval()->value;
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
    Entry ret;
    ret.base = op->value;
    ret.coeff = 0;
    return ret;
  }

  Entry VisitExpr_(const UIntImm* op) final {
    if (op->value < std::numeric_limits<int64_t>::max()) {
      Entry ret;
      ret.base = static_cast<int>(op->value);
      ret.coeff = 0;
      return ret;
    } else {
      return Everything();
    }
  }

  Entry VisitExpr_(const Add* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.coeff = ZeroAwareGCD(a.coeff, b.coeff);
    ret.base = BaseSimplify(a.base + b.base, ret.coeff);
    return ret;
  }

  Entry VisitExpr_(const Sub* op) final {
    Entry a = VisitExpr(op->a);
    Entry b = VisitExpr(op->b);
    Entry ret;
    ret.coeff = ZeroAwareGCD(a.coeff, b.coeff);
    ret.base = BaseSimplify(a.base - b.base, ret.coeff);
    return ret;
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
    Entry ret;
    ret.coeff = ZeroAwareGCD(pq, ZeroAwareGCD(pm, qn));
    ret.base = BaseSimplify(a.base * b.base, ret.coeff);
    return ret;
  }

  Entry DivByConst(const Expr& lhs,
                   int64_t val,
                   bool round_down) {
    Entry a = VisitExpr(lhs);
    CHECK_NE(val, 0);
    if (a.coeff % val == 0) {
      Entry ret;
      if (a.base == 0) {
        // a c x  / c -> a x
        ret.coeff = std::abs(a.coeff / val);
        ret.base = 0;
        return ret;
      }
      // positive division have a clear rounding mode.
      // Only handle case where we clearly know we need to round down.
      if (a.base > 0 && val > 0 &&
          (round_down || parent_->CanProveGreaterEqual(lhs, 0))) {
        ret.coeff = a.coeff / val;
        ret.base = a.base / val;
        return ret;
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
    int64_t coeff = ZeroAwareGCD(a.coeff, b.coeff);
    if (coeff == 0) {
      if (a.base == b.base) return a;
      return Everything();
    }
    int64_t base0 = a.base % coeff;
    int64_t base1 = b.base % coeff;
    Entry ret;
    if (base0 == base1) {
      ret.coeff = coeff;
      ret.base = base0;
      return ret;
    } else {
      ret.coeff = ZeroAwareGCD(ZeroAwareGCD(base0, base1), coeff);
      ret.base = 0;
      return ret;
    }
  }
  /*!
   * \brief Create interect of two sets.
   * \param a The left operand.
   * \param b the right operand.
   */
  static Entry Intersect(Entry a, Entry b) {
    // simple rule for now: pick higher constraints.
    // TODO(team-team): Use extended euclidean algorithm.
    if (a.coeff == 0) return a;
    if (b.coeff == 0) return b;
    if (a.coeff >= b.coeff) return a;
    return b;
  }
  /*!
   * \brief Simplify base so that it is in [0, coeff) when coeff != 0.
   * \param base The base value.
   * \param coeff The coeff value.
   * \return The simplified base.
   */
  static int64_t BaseSimplify(int64_t base, int64_t coeff) {
    if (coeff == 0) return base;
    base = base % coeff;
    if (base < 0) base += coeff;
    return base;
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
    Entry ret;
    ret.coeff = 1; ret.base = 0;
    return ret;
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
