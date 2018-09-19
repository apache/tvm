/*!
 *  Copyright (c) 2017 by Contributors
 * \file modular.cc
 * \brief Modular analysis
 */
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_visitor.h>
#include <tvm/arithmetic.h>
#include <limits>
#include "int_set_internal.h"

namespace tvm {
namespace arith {

using namespace ir;

class ModularEvaluator
    : public ExprFunctor<ModularEntry(const Expr&)> {
 public:
  explicit ModularEvaluator(
      const std::unordered_map<
      const Variable*, ModularEntry>& mod_map)
      : mod_map_(mod_map) {
  }
  ModularEntry Eval(const Expr& e) {
    return VisitExpr(e);
  }
  // default
  ModularEntry VisitExprDefault_(const Node*) final {
    return ModularEntry::everything();
  }
  // override combination rules.
  ModularEntry VisitExpr_(const IntImm* op) final {
    if (op->value < std::numeric_limits<int>::max()) {
      ModularEntry ret;
      ret.base = static_cast<int>(op->value);
      ret.coeff = 0;
      return ret;
    } else {
      return ModularEntry::everything();
    }
  }
  ModularEntry VisitExpr_(const UIntImm* op) final {
    if (op->value < static_cast<uint64_t>(
            std::numeric_limits<int>::max())) {
      ModularEntry ret;
      ret.base = static_cast<int>(op->value);
      ret.coeff = 0;
      return ret;
    } else {
      return ModularEntry::everything();
    }
  }
  ModularEntry VisitExpr_(const Variable* op) final {
    auto it = mod_map_.find(op);
    if (it != mod_map_.end()) {
      return it->second;
    } else {
      return ModularEntry::everything();
    }
  }
  ModularEntry VisitExpr_(const Add* op) final {
    ModularEntry a = Eval(op->a);
    ModularEntry b = Eval(op->b);
    ModularEntry ret;
    ret.coeff = ZeroAwareGCD(a.coeff, b.coeff);
    ret.base = BaseSimplify(a.base + b.base, ret.coeff);
    return ret;
  }
  ModularEntry VisitExpr_(const Sub* op) final {
    ModularEntry a = Eval(op->a);
    ModularEntry b = Eval(op->b);
    ModularEntry ret;
    ret.coeff = ZeroAwareGCD(a.coeff, b.coeff);
    ret.base = BaseSimplify(a.base - b.base, ret.coeff);
    return ret;
  }
  ModularEntry VisitExpr_(const Mul* op) final {
    ModularEntry a = Eval(op->a);
    ModularEntry b = Eval(op->b);
    // Simplification rule, x, y, z are in Z
    // (p x + n) (q y + m)
    // -> pq xy + pm x + qn y + mn
    // -> pq z + pm x + qn y + mn
    int pq = a.coeff * b.coeff;
    int pm = a.coeff * b.base;
    int qn = a.base * b.coeff;
    ModularEntry ret;
    ret.coeff = ZeroAwareGCD(pq, ZeroAwareGCD(pm, qn));
    ret.base = BaseSimplify(a.base * b.base, ret.coeff);
    return ret;
  }
  ModularEntry VisitExpr_(const Div* op) final {
    // a c x  / c -> a x
    // We cannot do cases where offset is non-zero
    // because of different integer rounding in pos/neg
    ModularEntry a = Eval(op->a);
    ModularEntry b = Eval(op->b);
    if (b.coeff == 0 &&
        a.base == 0) {
      CHECK_NE(b.base, 0);
      if (a.coeff % b.base == 0) {
        ModularEntry ret;
        ret.coeff = a.coeff / b.base;
        ret.base = 0;
        return ret;
      }
    }
    return ModularEntry::everything();
  }

 private:
  const std::unordered_map<
    const Variable*, ModularEntry>& mod_map_;
  friend struct ModularEntry;
  // simplify the base by putting it in range.
  static int BaseSimplify(int base, int coeff) {
    if (coeff == 0) return base;
    base = base % coeff;
    if (base < 0) base += coeff;
    return base;
  }
  static int ZeroAwareGCD(int a, int b) {
    CHECK_GE(a, 0);
    CHECK_GE(b, 0);
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
};

ModularEntry ModularEntry::Add(const ModularEntry& a,
                               const ModularEntry& b) {
  ModularEntry ret;
  ret.coeff = ModularEvaluator::ZeroAwareGCD(a.coeff, b.coeff);
  ret.base = ModularEvaluator::BaseSimplify(a.base + b.base, ret.coeff);
  return ret;
}


ModularEntry EvalModular(
    const Expr& e,
    const std::unordered_map<const Variable*, ModularEntry>& mod_map) {
  return ModularEvaluator(mod_map)(e);
}

IntSet EvalModular(const Expr& e,
                   const Map<Var, IntSet>& mod_map) {
  std::unordered_map<const Variable*, ModularEntry> mmap;
  for (auto& kv : mod_map) {
    const ModularSet* m = kv.second.as<ModularSet>();
    CHECK(m) << "Need to pass ModularSet for Modular Analysis";
    mmap[kv.first.get()] = m->e;
  }
  NodePtr<ModularSet> n = make_node<ModularSet>();
  n->e = ModularEvaluator(mmap)(e);
  return IntSet(n);
}

}  // namespace arith
}  // namespace tvm
