/*!
 *  Copyright (c) 2017 by Contributors
 *  Lower intrinsic calls to device specific ir when possible.
 * \file lower_intrin.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/api_registry.h>
#include <unordered_set>
#include "./ir_util.h"

namespace tvm {
namespace ir {

class IntrinInjecter : public IRMutator {
 public:
  explicit IntrinInjecter(std::string target) {
    std::istringstream is(target);
    std::string starget;
    is >> starget;
    patterns_.push_back("tvm.intrin.rule." + starget + ".");
    patterns_.push_back("tvm.intrin.rule.default.");
    fma_ = runtime::Registry::Get(patterns_[0] + "fma");
  }

  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->call_type == Call::Intrinsic ||
        op->call_type == Call::PureIntrinsic) {
      Expr r = ApplyPattern(op->name, e);
      if (r.defined()) return r;
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Add* op, const Expr& e) final {
    if (fma_ == nullptr || !op->type.is_float()) {
      return IRMutator::Mutate_(op, e);
    }
    if (const Mul* mb = op->b.as<Mul>()) {
      Expr r = (*fma_)(Call::make(
          op->type, "fma", {mb->a, mb->b, op->a}, Call::PureIntrinsic));
      if (r.defined()) return this->Mutate(r);
    } else if (const Mul* ma = op->a.as<Mul>()) {
      Expr r = (*fma_)(Call::make(
          op->type, "fma", {ma->a, ma->b, op->b}, Call::PureIntrinsic));
      if (r.defined()) return this->Mutate(r);
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  Expr ApplyPattern(const std::string& name, const Expr& e) {
    for (size_t i = 0; i < patterns_.size(); ++i) {
      std::string& p = patterns_[i];
      size_t psize = p.length();
      p.resize(psize + name.length());
      name.copy(&p[0] + psize, name.length());
      const runtime::PackedFunc* f = runtime::Registry::Get(p);
      p.resize(psize);
      // if pattern exists.
      if (f != nullptr) {
        Expr r = (*f)(e);
        CHECK(r.defined()) << "intrinsic rule must always return valid Expr";
        if (!r.same_as(e)) {
          return this->Mutate(r);
        }
      }
    }
    return Expr();
  }
  // patterns
  std::vector<std::string> patterns_;
  const PackedFunc* fma_{nullptr};
};

LoweredFunc
LowerIntrin(LoweredFunc f, const std::string& target) {
  auto n = std::make_shared<LoweredFuncNode>(*f.operator->());
  n->body = IntrinInjecter(target).Mutate(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
