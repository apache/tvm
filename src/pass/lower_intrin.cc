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
 *  Copyright (c) 2017 by Contributors
 *  Lower intrinsic calls to device specific ir when possible.
 * \file lower_intrin.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/api_registry.h>
#include <unordered_set>
#include "ir_util.h"

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
    if (const Mul* mb = op->b.as<Mul>()) {
      return MakeFMA(mb->a, mb->b, op->a, op, e);
    } else if (const Mul* ma = op->a.as<Mul>()) {
      return MakeFMA(ma->a, ma->b, op->b, op, e);
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  Expr SwapBroadcastCast(const Expr& e) {
    // Try to change broadcast(cast(x)) to cast(broadcast(x))
    // For some targets, LLVM will generate more efficient FMA
    // instruction with the latter. For example, vmla vs. vmlal
    // on ARM.
    if (const Broadcast* bcast = e.as<Broadcast>()) {
      if (const Cast* cast = bcast->value.as<Cast>()) {
        auto should_swap = [&]() {
          // Maintain behaviour (int8 -> int16, fp16 -> fp32).
          if (cast->type.bits() == cast->value.type().bits() * 2) {
            return true;
          }
          // Check both operands are integer-like.
          if (!cast->type.is_uint() && !cast->type.is_int()) {
            return false;
          }
          if (!cast->value.type().is_uint() && !cast->value.type().is_int()) {
            return false;
          }
          // If both are integer-like, swap if we have a widening cast.
          return cast->type.bits() > cast->value.type().bits();
        };

        if (should_swap()) {
          Expr new_bcast = Broadcast::make(cast->value, bcast->lanes);
          return Cast::make(bcast->type, new_bcast);
        }
      }
    }
    return e;
  }

  Expr MakeFMA(const Expr& a, const Expr& b, const Expr& c,
               const Add* op, const Expr& e) {
    // emit fma instruction: a * b + c
    Expr lhs = SwapBroadcastCast(a);
    Expr rhs = SwapBroadcastCast(b);

    if (fma_ != nullptr && op->type.is_float()) {
      Expr r = (*fma_)(Call::make(
          op->type, "fma", {lhs, rhs, c}, Call::PureIntrinsic));
      if (r.defined()) return this->Mutate(r);
    } else {
      if (!lhs.same_as(a) || !rhs.same_as(b)) {
        Expr mul = this->Mutate(Mul::make(lhs, rhs));
        return Add::make(mul, this->Mutate(c));
      }
    }
    return IRMutator::Mutate_(op, e);
  }

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
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = IntrinInjecter(target).Mutate(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
