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
 *  Lower intrinsic calls and ops to device specific ir when possible.
 * \file lower_intrin.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/api_registry.h>
#include <tvm/expr_operator.h>
#include <unordered_set>
#include "ir_util.h"
#include "../arithmetic/pattern_match.h"
#include "../arithmetic/ir_mutator_with_analyzer.h"

namespace tvm {
namespace ir {

class IntrinInjecter : public arith::IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::Mutate_;

  IntrinInjecter(arith::Analyzer* analyzer, std::string target)
      : IRMutatorWithAnalyzer(analyzer) {
    std::istringstream is(target);
    std::string starget;
    is >> starget;
    patterns_.push_back("tvm.intrin.rule." + starget + ".");
    patterns_.push_back("tvm.intrin.rule.default.");
    fma_ = runtime::Registry::Get(patterns_[0] + "fma");
    if (target == "stackvm") {
      support_bitwise_op_ = false;
    }
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

  // We use floordiv for integer analysis,
  // but will need to lower them to native truncdiv instructions
  Expr Mutate_(const FloorDiv* op, const Expr& e) final {
    Expr ret = IRMutatorWithAnalyzer::Mutate_(op, e);
    op = ret.as<FloorDiv>();
    if (op == nullptr) return ret;
    int shift;
    const DataType& dtype = op->type;
    CHECK(dtype.is_int() || !dtype.is_uint());

    if (support_bitwise_op_ &&
        is_const_power_of_two_integer(op->b, &shift)) {
      // lower to right shift if possible.
      return op->a >> make_const(dtype, shift);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common path, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0) ||
          analyzer_->CanProveGreaterEqual(e, 0)) {
        return truncdiv(op->a, op->b);
      } else {
        DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divident";
        Expr rdiv = truncdiv(op->a, op->b);
        Expr rmod = truncmod(op->a, op->b);
        // condition on b >= 0.
        // truncmod(a, b) < 0 will implies ceildiv,
        // So we need to correct these cases.
        if ((dtype == Int(32) || dtype == Int(64)) && support_bitwise_op_) {
          // equivalent to rdiv + (rmod >= 0 ? 0: -1);
          return rdiv + (rmod >> make_const(dtype, dtype.bits() - 1));
        } else {
          return ir::Select::make(rmod >= 0 , rdiv, rdiv - make_const(dtype, 1));
        }
      }
    } else {
      // uncommon case
      DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divisor";
      // b >= 0 => (rmod >=0 ? rdiv : rdiv - 1)
      // b < 0  => (rmod <= 0 ? rdiv : rdiv - 1)
      Expr rdiv = truncdiv(op->a, op->b);
      Expr rmod = truncmod(op->a, op->b);
      return ir::Select::make(
          (op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0),
          rdiv, rdiv - make_const(dtype, 1));
    }
  }

  Expr Mutate_(const FloorMod* op, const Expr& e) final {
    Expr ret = IRMutatorWithAnalyzer::Mutate_(op, e);
    op = ret.as<FloorMod>();
    if (op == nullptr) return ret;
    // Lower floordiv to native truncdiv.
    int shift;
    const DataType& dtype = op->type;
    CHECK(dtype.is_int() || !dtype.is_uint());

    if (support_bitwise_op_ &&
        is_const_power_of_two_integer(op->b, &shift)) {
      // lower to masking if possible.
      int64_t mask = (
          static_cast<int64_t>(1) << static_cast<int64_t>(shift)) - 1;
      return op->a & make_const(dtype, mask);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common pass, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0) ||
          analyzer_->CanProveGreaterEqual(e, 0)) {
        return truncmod(op->a, op->b);
      } else {
        DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divident";
        // NOTE:condition on b >= 0.
        // mod(a, b) < 0 will imply we are doing ceildiv,
        // So we need to correct these cases.
        Expr rmod = truncmod(op->a, op->b);
        if ((dtype == Int(32) || dtype == Int(64)) && support_bitwise_op_) {
          // (rmod >> shift) & b
          // -> (rmod >= 0 ? 0: -1) & b
          // -> rmod >= 0 ? 0 : b
          return rmod + (op->b & (rmod >> make_const(dtype, dtype.bits() - 1)));
        } else {
          return ir::Select::make(rmod >= 0, rmod, rmod + op->b);
        }
      }
    } else {
      // uncommon case
      DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divsor and divident";
      Expr rmod = truncmod(op->a, op->b);
      // b > 0 && rmod >= 0 -> rmod
      // b > 0 && rmod < 0  -> rmod + b
      // b < 0 && rmod < 0 -> rmod
      // b < 0 && rmod > 0 -> rmod + b
      return ir::Select::make(
          (op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0),
          rmod, rmod + op->b);
    }
  }

  Expr Mutate_(const Max* op, const Expr& e) final {
    using namespace arith;
    PVar<Expr> x, y;
    PVar<Integer> c;
    if (max(floordiv(x, y), c).Match(e) &&
        c.Eval()->value >= 0 &&
        analyzer_->CanProveGreaterEqual(y.Eval(), 0)) {
      return max(Mutate(truncdiv(x, y).Eval()), c.Eval());
    }
    return IRMutatorWithAnalyzer::Mutate_(op, e);
  }

  Expr Mutate_(const EQ* op, const Expr& e) final {
    using namespace arith;
    PVar<Expr> x, y;
    if ((floormod(x, y) == 0).Match(e)) {
      return Mutate((truncmod(x, y) == 0).Eval());
    }
    return IRMutatorWithAnalyzer::Mutate_(op, e);
  }

  Expr Mutate_(const NE* op, const Expr& e) final {
    using namespace arith;
    PVar<Expr> x, y;
    if ((floormod(x, y) != 0).Match(e)) {
      return Mutate((truncmod(x, y) == 0).Eval());
    }
    return IRMutatorWithAnalyzer::Mutate_(op, e);
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
  bool support_bitwise_op_{true};
};

Stmt LowerIntrinStmt(Stmt stmt, const std::string& target) {
  arith::Analyzer analyzer;
  return IntrinInjecter(&analyzer, target).Mutate(stmt);
}

LoweredFunc
LowerIntrin(LoweredFunc f, const std::string& target) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = LowerIntrinStmt(n->body, target);
  return LoweredFunc(n);
}

// Register the api only for test purposes
TVM_REGISTER_API("ir_pass._LowerIntrinStmt")
.set_body_typed(LowerIntrinStmt);

}  // namespace ir
}  // namespace tvm
