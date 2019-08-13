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
 *  Copyright (c) 2019 by Contributors
 * \file rewrite_simplify.cc
 * \brief Rewrite-rule based simplification.
 */
// Acknowledgement: Most rewrite-rules are from Halide.
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_mutator.h>
#include <algorithm>
#include "const_fold.h"
#include "pattern_match.h"
#include "rewrite_simplify.h"

namespace tvm {
namespace arith {

using namespace ir;

// macro for doing simple rewrite
#define TVM_TRY_REWRITE(SrcExpr, ResExpr)       \
  if ((SrcExpr).Match(ret)) {                   \
    return (ResExpr).Eval();                    \
  }

// macro for rewrite + recursively rewrite ResExpr
#define TVM_TRY_RECURSIVE_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {                       \
    return RecursiveRewrite((ResExpr).Eval());      \
  }

// macro rewrite only if CondExor is true after match.
#define TVM_TRY_REWRITE_IF(SrcExpr, ResExpr, CondExpr)  \
  if ((SrcExpr).Match(ret) && (CondExpr)) {             \
    return (ResExpr).Eval();                            \
  }

// macro rewrite + recursive_rewrite only if CondExor is true after match.
#define TVM_TRY_RECURSIVE_REWRITE_IF(SrcExpr, ResExpr, CondExpr)  \
  if ((SrcExpr).Match(ret) && (CondExpr)) {                       \
    return RecursiveRewrite((ResExpr).Eval());                    \
  }

// NOTE for developers:
//
// We mainly focus on index expression simplification.
// Besides the RewriteSimplifier, some cases can be better
// handled by CanonicalSimplifier.
//

// try to prove x equals val
RewriteSimplifier::Impl::CompareResult RewriteSimplifier::Impl::
TryCompare(const Expr& x, int64_t val) {
  Expr diff = Mutate(x);
  if (const auto* ptr = diff.as<IntImm>()) {
    if (ptr->value == val) {
      return kEQ;
    } else if (ptr->value > val) {
      return kGT;
    } else if (ptr->value < val) {
      return kLT;
    }
  }
  ConstIntBound dbound = parent_->const_int_bound(diff);
  if (dbound->min_value > val) {
    return kGT;
  }
  if (dbound->max_value < val) {
    return kLT;
  }
  if (dbound->min_value >= val) {
    return kGE;
  }
  if (dbound->max_value <= val) {
    return kLE;
  }
  if (val == 0) {
    ModularSet dmod = parent_->modular_set(diff);
    if (dmod->base != 0) {
      return kNE;
    }
  }
  return kUnknown;
}

void RewriteSimplifier::Impl::
Update(const Var& var, const Expr& info, bool override) {
  if (!override) {
    auto it = var_map_.find(var);
    if (it != var_map_.end()) {
      CHECK(Equal(it->second, info))
          << "Trying to update var \'" << var << "\'"
          << " with a different value: "
          << "original=" << it->second
          << ", new=" << info;
    }
  }
  var_map_[var] = info;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Add* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Add>();
  Expr const_res = TryConstFold<Add>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) + ramp(b2, s2, lanes),
                    ramp(b1 + b2, s1 + s2, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) + broadcast(x, lanes),
                    ramp(b1 + x, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) + ramp(b1, s1, lanes),
                    ramp(x + b1, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) + broadcast(y, lanes),
                    broadcast(x + y, lanes));
  }

  if (IsIndexType(op->type)) {
    // Index rules
    // cancelation rules
    TVM_TRY_REWRITE((x - y) + y, x);
    TVM_TRY_REWRITE(x + (y - x), y);

    TVM_TRY_REWRITE((x - y) + (y - z), x - z);
    TVM_TRY_REWRITE((x - y) + (z - x), z - y);

    TVM_TRY_REWRITE(min(x, y - z) + z, min(x + z, y));
    TVM_TRY_REWRITE(min(x - z, y) + z, min(x, y + z));
    TVM_TRY_REWRITE(max(x, y - z) + z, max(x + z, y));
    TVM_TRY_REWRITE(max(x - z, y) + z, max(x, y + z));

    TVM_TRY_REWRITE_IF(min(x, y + z * c1) + z * c2, min(x + z * c2, y),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, y + z * c1) + z * c2, max(x + z * c2, y),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(y + z * c1, x) + z * c2, min(x + z * c2, y),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(y + z * c1, x) + z * c2, max(x + z * c2, y),
                       c1.Eval()->value == -c2.Eval()->value);

    TVM_TRY_REWRITE(max(x, y) + min(x, y), x + y);
    TVM_TRY_REWRITE(min(x, y) + max(x, y), x + y);
    TVM_TRY_REWRITE(max(x, y) + min(y, x), x + y);
    TVM_TRY_REWRITE(min(x, y) + max(y, x), x + y);

    TVM_TRY_REWRITE_IF(min(x, y + c1) + c2, min(x + c2, y),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(x + c1, y) + c2, min(x, y + c2),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, y + c1) + c2, max(x + c2, y),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x + c1, y) + c2, max(x, y + c2),
                       c1.Eval()->value == -c2.Eval()->value);

    // constant folding
    // NOTE: canonicalization might better at this.
    TVM_TRY_REWRITE((x + c1) + c2, x + (c1 + c2));

    // mul co-efficient folding
    TVM_TRY_REWRITE(x + x, x * 2);
    TVM_TRY_REWRITE(x * y + x, x * (y + 1));
    TVM_TRY_REWRITE(y * x + x, x * (y + 1));
    TVM_TRY_REWRITE(x + y * x, x * (1 + y));
    TVM_TRY_REWRITE(x + x * y, x * (1 + y));
    TVM_TRY_REWRITE(x * y + x * z, x * (y + z));
    TVM_TRY_REWRITE(y * x + x * z, x * (y + z));
    TVM_TRY_REWRITE(x * y + z * x, x * (y + z));
    TVM_TRY_REWRITE(y * x + z * x, x * (y + z));

    // DivMod rules
    // truc div
    TVM_TRY_REWRITE((x / c1) * c1 + x % c1, x);
    // floor div
    TVM_TRY_REWRITE(floordiv(x, c1) * c1 + floormod(x, c1), x);

    // canonicalization rule
    // will try rewrite again after canonicalization.
    TVM_TRY_RECURSIVE_REWRITE(x + (c1 - y), (x - y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + c1 + y, (x + y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + (c1 + y), (x + y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + max(y, z), max(y, z) + x);
    TVM_TRY_RECURSIVE_REWRITE(x + min(y, z), min(y, z) + x);

    // DivMod rules
    // truc div
    TVM_TRY_RECURSIVE_REWRITE((y % c1) + x * c1, x * c1 + (y % c1));
    // floor div
    TVM_TRY_RECURSIVE_REWRITE(floormod(y, c1) + x * c1, x * c1 + floormod(y, c1));
  }

  // condition rules.
  TVM_TRY_REWRITE(select(x, b1, b2) + select(x, s1, s2),
                  select(x, b1 + s1, b2 + s2));
  // default value
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Sub* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Sub>();
  Expr const_res = TryConstFold<Sub>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) - ramp(b2, s2, lanes),
                    ramp(b1 - b2, s1 - s2, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) - broadcast(x, lanes),
                    ramp(b1 - x, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) - ramp(b1, s1, lanes),
                    ramp(x - b1, 0 - s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) - broadcast(y, lanes),
                    broadcast(x - y, lanes));
  }

  if (IsIndexType(op->type)) {
    // Index rules
    // cancelation rules
    TVM_TRY_REWRITE((x + y) - y, x);
    TVM_TRY_REWRITE((x + y) - x, y);
    TVM_TRY_REWRITE(x - (y + x), 0 - y);
    TVM_TRY_REWRITE(x - (x + y), 0 - y);

    TVM_TRY_REWRITE(min(x, y) - x, min(0, y - x));
    TVM_TRY_REWRITE(min(x, y) - y, min(x - y, 0));
    TVM_TRY_REWRITE(max(x, y) - x, max(0, y - x));
    TVM_TRY_REWRITE(max(x, y) - y, max(x - y, 0));

    TVM_TRY_REWRITE(x - max(x, y), min(0, x - y));
    TVM_TRY_REWRITE(y - max(x, y), min(y - x, 0));
    TVM_TRY_REWRITE(x - min(x, y), max(0, x - y));
    TVM_TRY_REWRITE(y - min(x, y), max(y - x, 0));

    // mul co-efficient folding
    TVM_TRY_REWRITE(x - x, ZeroWithTypeLike(x));
    TVM_TRY_REWRITE(x * y - x, x * (y - 1));
    TVM_TRY_REWRITE(y * x - x, x * (y - 1));
    TVM_TRY_REWRITE(x - y * x, x * (1 - y));
    TVM_TRY_REWRITE(x - x * y, x * (1 - y));
    TVM_TRY_REWRITE(x * y - x * z, x * (y - z));
    TVM_TRY_REWRITE(y * x - x * z, x * (y - z));
    TVM_TRY_REWRITE(x * y - z * x, x * (y - z));
    TVM_TRY_REWRITE(y * x - z * x, x * (y - z));

    // constant cancelation
    TVM_TRY_REWRITE((x + c1) - c2, x + (c1 - c2));
    TVM_TRY_REWRITE((c1 - x) - (c2 - y), (y - x) + (c1 - c2));

    // cancelization rule involving 4 operands
    TVM_TRY_REWRITE((x + y) - (x + z), y - z);
    TVM_TRY_REWRITE((x + y) - (z + x), y - z);
    TVM_TRY_REWRITE((y + x) - (z + x), y - z);
    TVM_TRY_REWRITE((y + x) - (x + z), y - z);

    TVM_TRY_REWRITE(min(x + y, z) - x,  min(y, z - x));
    TVM_TRY_REWRITE(min(y + x, z) - x,  min(y, z - x));
    TVM_TRY_REWRITE(min(z, x + y) - x,  min(z - x, y));
    TVM_TRY_REWRITE(min(z, y + x) - x,  min(z - x, y));

    TVM_TRY_REWRITE(max(x + y, z) - x,  max(y, z - x));
    TVM_TRY_REWRITE(max(y + x, z) - x,  max(y, z - x));
    TVM_TRY_REWRITE(max(z, x + y) - x,  max(z - x, y));
    TVM_TRY_REWRITE(max(z, y + x) - x,  max(z - x, y));

    TVM_TRY_REWRITE(x - min(x + y, z),  max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(y + x, z),  max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(z, x + y),  max(x - z, 0 - y));
    TVM_TRY_REWRITE(x - min(z, y + x),  max(x - z, 0 - y));

    TVM_TRY_REWRITE(min(x, y) - min(y, x), ZeroWithTypeLike(x));
    TVM_TRY_REWRITE(max(x, y) - max(y, x), ZeroWithTypeLike(x));

    TVM_TRY_REWRITE_IF(min(b1, b2) - min(s1, s2), b1 - s1,
                       CanProveEqual(((b1 - s1) - (b2 - s2)).Eval(), 0));

    TVM_TRY_REWRITE_IF(min(b1, b2) - min(s1, s2), b1 - s2,
                       CanProveEqual(((b1 - s2) - (b2 - s1)).Eval(), 0));
    TVM_TRY_REWRITE_IF(max(b1, b2) - max(s1, s2), b1 - s1,
                       CanProveEqual(((b1 - s1) - (b2 - s2)).Eval(), 0));
    TVM_TRY_REWRITE_IF(max(b1, b2) - max(s1, s2), b1 - s2,
                       CanProveEqual(((b1 - s2) - (b2 - s1)).Eval(), 0));

    // DivMod rules
    // trucdiv
    // NOTE: c*(x/c) + x % c == x is true all division mode.
    TVM_TRY_REWRITE_IF(x - (x / c1) * c1, x % c1,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF((x / c1) * c1 - x, 0 - (x % c1),
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - ((x + y) / c1) * c1, (x + y) % c1 - y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(((x + y) / c1) * c1 - x, y - ((x + y) % c1),
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - ((x - y) / c1) * c1, (x - y) % c1 + y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(((x - y) / c1) * c1 - x, 0 - (x - y) % c1 - y,
                       c1.Eval()->value != 0);

    TVM_TRY_REWRITE_IF(x * c2 - (x / c1) * c3, (x % c1) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF((x / c1) * c3 - x * c2, 0 - (x % c1) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(x * c2 - ((x + y) / c1) * c3, ((x + y) % c1 - y) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(((x + y) / c1) * c3 - x * c2, (y - ((x + y) % c1)) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(x * c2 - ((x - y) / c1) * c3, ((x - y) % c1 + y) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(((x - y) / c1) * c3 - x * c2, (0 - (x - y) % c1 - y) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);

    // Proof in the case of floordiv, need positive condition.
    // let x = a * c3 + r
    // (x + c1) / c3 - x / c3 => (r + c1) / c3
    TVM_TRY_REWRITE_IF((x + c1) / c3  - (x + c2) / c3,
                       ((x + ((c2 % c3) + c3) % c3) % c3 + (c1 - c2)) / c3,
                       CanProveGreaterEqual(x.Eval(), -c2.Eval()->value) &&
                       c1.Eval()->value >= c2.Eval()->value &&
                       c3.Eval()->value > 0);
    TVM_TRY_REWRITE_IF((x + c1) / c3  - x / c3,
                       (x % c3 + c1) / c3,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       c1.Eval()->value >= 0 &&
                       c3.Eval()->value > 0);

    // floordiv
    TVM_TRY_REWRITE_IF(x - floordiv(x, c1) * c1, floormod(x, c1),
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 - x, 0 - floormod(x, c1),
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - floordiv(x + y, c1) * c1, floormod(x + y, c1) - y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(floordiv(x + y, c1) * c1 - x, y - floormod(x + y, c1),
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - floordiv(x - y, c1) * c1, floormod(x - y, c1) + y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(floordiv(x - y, c1) * c1 - x, 0 - floormod(x - y, c1) - y,
                       c1.Eval()->value != 0);

    TVM_TRY_REWRITE_IF(x * c2 - floordiv(x, c1) * c3, floormod(x, c1) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c3 - x * c2, 0 - floormod(x, c1) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(x * c2 - floordiv(x + y, c1) * c3, (floormod(x + y, c1) - y) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(floordiv(x + y, c1) * c3 - x * c2, (y - floormod(x + y, c1)) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(x * c2 - floordiv(x - y, c1) * c3, (floormod(x - y, c1) + y) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(floordiv(x - y, c1) * c3 - x * c2, (0 - floormod(x - y, c1) - y) * c2,
                       c1.Eval()->value != 0 &&
                       c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);

    TVM_TRY_REWRITE_IF(floordiv(x + c1, c3) - floordiv(x + c2, c3),
                       floordiv(floormod(x + floormod(c2, c3), c3) + (c1 - c2), c3),
                       c3.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c1, c3)  - floordiv(x, c3),
                       floordiv(floormod(x, c3) + c1, c3),
                       c3.Eval()->value > 0);

    // canonicalization rule
    // will try rewrite again after canonicalization.
    TVM_TRY_REWRITE(x - c1, x + (0 - c1));
    TVM_TRY_RECURSIVE_REWRITE((x + c1) - y, (x - y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x - (y - z), (x + z) - y);
    TVM_TRY_RECURSIVE_REWRITE(x - y * c1, x + y * (0 - c1));
  }

  // condition rules.
  TVM_TRY_REWRITE(select(x, b1, b2) - select(x, s1, s2),
                  select(x, b1 - s1, b2 - s2));
  TVM_TRY_REWRITE(select(x, y, z) - z,
                  select(x, y - z, ZeroWithTypeLike(z)));
  TVM_TRY_REWRITE(select(x, y, z) - y,
                  select(x, ZeroWithTypeLike(y), z - y));
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Mul* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Mul>();
  Expr const_res = TryConstFold<Mul>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) * broadcast(y, lanes),
                    broadcast(x * y, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) * broadcast(x, lanes),
                    ramp(b1 * x, s1 * x, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) * ramp(b1, s1, lanes),
                    ramp(b1 * x, s1 * x, lanes));
  }

  if (IsIndexType(op->type)) {
    // constant simplification rule
    TVM_TRY_REWRITE((x + c1) * c2, x * c2 + c1 * c2);
    TVM_TRY_REWRITE((x * c1) * c2, x * (c1 * c2));
    TVM_TRY_REWRITE(min(x, y) * max(x, y), x * y);
    TVM_TRY_REWRITE(max(x, y) * min(x, y), x * y);

    // canonicalization
    TVM_TRY_RECURSIVE_REWRITE(x * (c1 * y), (x * y) * c1);
    TVM_TRY_RECURSIVE_REWRITE(c1 * x, x * c1);
    TVM_TRY_RECURSIVE_REWRITE_IF(
        (x - y) * c1, (y - x) * (0 - c1),
        c1.Eval()->value < 0);
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Div* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Div>();
  Expr const_res = TryConstFold<Div>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // x / 2.0 = x * 0.5
  if (const FloatImm* ptr = op->b.as<FloatImm>()) {
    CHECK(op->type.is_float());
    return op->a * make_const(op->b.type(), 1.0 / ptr->value);
  }

  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) / broadcast(y, lanes),
                    broadcast(x / y, lanes));
    // ramp / bcast
    if ((ramp(b1, c1, lanes) / broadcast(c2, lanes)).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return ramp(b1 / c2, c1 / c2, lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      if (CanProveGreaterEqual(b1.Eval(), 0)) {
        ModularSet bmod = parent_->modular_set(b1.Eval());
        int64_t ramp_min = bmod->base / c2val;
        int64_t ramp_max = (bmod->base + (lanes.Eval() - 1) * c1val) / c2val;
        if (bmod->coeff % c2val == 0 && ramp_min == ramp_max) {
          return broadcast(b1 / c2, lanes).Eval();
        }
      }
    }
  }

  if (IsIndexType(op->type)) {
    // Be-aware of the division rules:
    // We adopt the default C division uses truncation instead of floordiv.
    // This means most rules need to check non-negativeness of the operands.

    // TryConstFold doesn't work for negative cases because it is also used by legacy
    // parts of tvm which still assume euclidean div. In this simplifier we assume that the division
    // is truncated, so perform const folding again.
    // NOTE: trunc div required
    if ((c1 / c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      return make_const(op->type, c1val / c2val);
    }

    // while it is always true for trunc div
    // restrict to common case(positive div)
    TVM_TRY_REWRITE_IF((x / c1) / c2, x / (c1 * c2),
                       c1.Eval()->value > 0 && c2.Eval()->value > 0);

    TVM_TRY_REWRITE_IF((x / c1 + c2) / c3, (x + c1 * c2) / (c1 * c3),
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value >= 0 &&
                       c3.Eval()->value > 0 &&
                       CanProveGreaterEqual(x.Eval(), 0));

    if (((x * c1) / c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val > 0 && c2val > 0) {
        if (c1val % c2val == 0) return (x * (c1 / c2)).Eval();
        if (c2val % c1val == 0) return (x / (c2 / c1)).Eval();
      }
    }

    TVM_TRY_REWRITE(x / x, OneWithTypeLike(x));
    TVM_TRY_REWRITE(x * c1 / x, c1);
    TVM_TRY_REWRITE(c1 * x / x, c1);

    // Rules involving 2-operands.
    TVM_TRY_REWRITE_IF((x * c1 + y) / c2, x * (c1 / c2) + y / c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(min(x * c1, y) / c2, min(x * (c1 / c2), y / c2),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(max(x * c1, y) / c2, max(x * (c1 / c2), y / c2),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF((y + x * c1) / c2, y / c2 + x * (c1 / c2),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(min(y, x * c1) / c2, min(y / c2, x * (c1 / c2)),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(max(y, x * c1) / c2, max(y / c2, x * (c1 / c2)),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    // Rules involving 3-operands.
    TVM_TRY_REWRITE_IF((x * c1 + y + z) / c2, x * (c1 / c2) + (y + z)/ c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF((x * c1 - y + z) / c2, x * (c1 / c2) + (z - y)/ c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((z - y).Eval(), 0));

    TVM_TRY_REWRITE_IF((x * c1 + y - z) / c2, x * (c1 / c2) + (y - z)/ c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y - z).Eval(), 0));

    TVM_TRY_REWRITE_IF((y + x * c1 + z) / c2, x * (c1 / c2) + (y + z) / c2,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF((x + c1) / c2, x / c2 + c1 / c2,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF((x + y) / x, y / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF((y + x) / x, y / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(((x + y) + z) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF(((y + x) + z) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF((y + (z + x)) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF((y + (x + z)) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF((x * y) / y, x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF((y * x) / y, x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF((x * z + y) / z, x + y / z,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF((z * x + y) / z, x + y / z,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF((y + x * z) / z, y / z + x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF((y + z * x) / z, y / z + x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Mod* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Mod>();
  Expr const_res = TryConstFold<Mod>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) % broadcast(y, lanes),
                    broadcast(x % y, lanes));

    // ramp % bcast
    if ((ramp(b1, c1, lanes) % broadcast(c2, lanes)).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return broadcast(b1 % c2, lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      if (CanProveGreaterEqual(b1.Eval(), 0)) {
        ModularSet bmod = parent_->modular_set(b1.Eval());
        int64_t ramp_min = bmod->base / c2val;
        int64_t ramp_max = (bmod->base + (lanes.Eval() - 1) * c1val) / c2val;
        if (bmod->coeff % c2val == 0) {
          if (ramp_min == ramp_max) {
            return ramp(bmod->base % c2, c1, lanes).Eval();
          } else {
            return (ramp(bmod->base % c2, c1, lanes) % broadcast(c2, lanes)).Eval();
          }
        }
      }
    }
  }

  if (IsIndexType(op->type)) {
    // Be-aware of the division rules:
    // We adopt the default C division uses truncation instead of floordiv.
    // This means most rules need to check non-negativeness of the operands.
    TVM_TRY_REWRITE_IF((x * c1) % c2, ZeroWithTypeLike(x),
                       c2.Eval()->value != 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF((x * c1 + y) % c2, y % c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual((x * c1).Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF((x + c1) % c2, x % c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value >= 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF((x + y * c1) % c2, x % c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y * c1).Eval(), 0));

    // canonicalization: x % c == x % (-c) for truncated division
    // NOTE: trunc div required
    TVM_TRY_RECURSIVE_REWRITE_IF(x % c1,
                                 x % PConst<Expr>(make_const(op->type, -c1.Eval()->value)),
                                 c1.Eval()->value < 0);

    // try modular analysis
    if ((x % c1).Match(ret)) {
      ModularSet mod = parent_->modular_set(x.Eval());
      int64_t c1val = c1.Eval()->value;
      if (mod->coeff % c1val == 0 &&
          c1val > 0 &&
          CanProveGreaterEqual(x.Eval(), 0)) {
        return (mod->base % c1).Eval();
      }
    }
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const FloorDiv* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<FloorDiv>();
  Expr const_res = TryConstFold<FloorDiv>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(floordiv(broadcast(x, lanes), broadcast(y, lanes)),
                    broadcast(floordiv(x, y), lanes));
    // ramp // bcast
    if (floordiv(ramp(b1, c1, lanes), broadcast(c2, lanes)).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return ramp(floordiv(b1, c2), floordiv(c1, c2), lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      ModularSet bmod = parent_->modular_set(b1.Eval());
      int64_t ramp_min = floordiv(bmod->base, c2val);
      int64_t ramp_max = floordiv(bmod->base + (lanes.Eval() - 1) * c1val, c2val);
      if (bmod->coeff % c2val == 0 && ramp_min == ramp_max) {
        return broadcast(floordiv(b1, c2), lanes).Eval();
      }
    }
  }

  if (IsIndexType(op->type)) {
    // Be-aware of the division rules: this is floor division.
    TVM_TRY_REWRITE_IF(floordiv(floordiv(x, c1), c2), floordiv(x, c1 * c2),
                       c1.Eval()->value > 0 && c2.Eval()->value > 0);

    TVM_TRY_REWRITE_IF(floordiv(floordiv(x, c1) + c2, c3), floordiv(x + c1 * c2, c1 * c3),
                       c1.Eval()->value > 0 && c3.Eval()->value > 0);

    if (floordiv(x * c1, c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val > 0 && c2val > 0) {
        if (c1val % c2val == 0) return (x * floordiv(c1, c2)).Eval();
        if (c2val % c1val == 0) return (floordiv(x, floordiv(c2, c1))).Eval();
      }
    }

    TVM_TRY_REWRITE(floordiv(x, x), OneWithTypeLike(x));
    TVM_TRY_REWRITE(floordiv(x * c1, x), c1);
    TVM_TRY_REWRITE(floordiv(c1 * x, x), c1);

    // Rules involving 2-operands.
    TVM_TRY_REWRITE_IF(floordiv(x * c1 + y, c2),
                       x * floordiv(c1, c2) + floordiv(y, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(min(x * c1, y), c2),
                       min(x * floordiv(c1, c2), floordiv(y, c2)),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(max(x * c1, y), c2),
                       max(x * floordiv(c1, c2), floordiv(y, c2)),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(y + x * c1, c2),
                       floordiv(y, c2) + x * floordiv(c1, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(min(y, x * c1), c2),
                       min(floordiv(y, c2), x * floordiv(c1, c2)),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(max(y, x * c1), c2),
                       max(floordiv(y, c2), x * floordiv(c1, c2)),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    // Rules involving 3-operands.
    TVM_TRY_REWRITE_IF(floordiv(x * c1 + y + z, c2),
                       x * floordiv(c1, c2) + floordiv(y + z, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x * c1 - y + z, c2),
                       x * floordiv(c1, c2) + floordiv(z - y, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x * c1 + y - z, c2),
                       x * floordiv(c1, c2) + floordiv(y - z, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(y + x * c1 + z, c2),
                       x * floordiv(c1, c2) + floordiv(y + z, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x + c1, c2),
                       floordiv(x, c2) + floordiv(c1, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x + y, x), floordiv(y, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(floordiv(y + x, x), floordiv(y, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(floordiv((x + y) + z, x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv((y + x) + z, x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y + (z + x), x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y + (x + z), x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(floordiv(x * y, y), x,
                       CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y * x, y), x,
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(floordiv(x * z + y, z), x + floordiv(y, z),
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(z * x + y, z), x + floordiv(y, z),
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y + x * z, z), floordiv(y, z) + x,
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y + z * x, z), floordiv(y, z) + x,
                       CanProveGreaterEqual(z.Eval(), 0));
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const FloorMod* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<FloorMod>();
  Expr const_res = TryConstFold<FloorMod>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(floormod(broadcast(x, lanes), broadcast(y, lanes)),
                    broadcast(floormod(x, y), lanes));

    // floormod(ramp, bcast)
    if (floormod(ramp(b1, c1, lanes), broadcast(c2, lanes)).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return broadcast(floormod(b1, c2), lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      ModularSet bmod = parent_->modular_set(b1.Eval());
      int64_t ramp_min = floordiv(bmod->base, c2val);
      int64_t ramp_max = floordiv(bmod->base + (lanes.Eval() - 1) * c1val, c2val);
      if (bmod->coeff % c2val == 0) {
        if (ramp_min == ramp_max) {
          return ramp(floormod(bmod->base, c2), c1, lanes).Eval();
        } else {
          return floormod(ramp(floormod(bmod->base, c2), c1, lanes), broadcast(c2, lanes)).Eval();
        }
      }
    }
  }

  if (IsIndexType(op->type)) {
    // Be-aware of the division rules: we use floordiv/floormod here
    TVM_TRY_REWRITE_IF(floormod(x * c1, c2), ZeroWithTypeLike(x),
                       c2.Eval()->value != 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floormod(x * c1 + y, c2), floormod(y, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floormod(x + c1, c2), floormod(x, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floormod(x + y * c1, c2), floormod(x, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    // try modular analysis
    if (floormod(x, c1).Match(ret)) {
      ModularSet mod = parent_->modular_set(x.Eval());
      int64_t c1val = c1.Eval()->value;
      if (mod->coeff % c1val == 0 && c1val > 0) {
        return floormod(mod->base, c1).Eval();
      }
    }
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Min* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Min>();
  Expr const_res = TryConstFold<Min>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y, z, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  PVar<int> lanes;

  // vector rule
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(min(broadcast(x, lanes), broadcast(y, lanes)),
                    broadcast(min(x, y), lanes));
    TVM_TRY_REWRITE(min(min(x, broadcast(y, lanes)), broadcast(z, lanes)),
                    min(x, broadcast(min(y, z), lanes)));
  }
  if (IsIndexType(op->type)) {
    TVM_TRY_REWRITE(min(x, x), x);

    // constant int bound
    ConstIntBound a_bound = parent_->const_int_bound(op->a);
    ConstIntBound b_bound = parent_->const_int_bound(op->b);
    if (a_bound->max_value <= b_bound->min_value) {
      return op->a;
    }
    if (b_bound->max_value <= a_bound->min_value) {
      return op->b;
    }

    // constant comparison
    if (min(x + c1, x + c2).Match(ret)) {
      if (c1.Eval()->value < c2.Eval()->value) {
        return (x + c1).Eval();
      } else {
        return (x + c2).Eval();
      }
    }
    if (min(x + c1, x).Match(ret) ||
        min(x, x + c1).Match(ret)) {
      if (c1.Eval()->value < 0) {
        return (x + c1).Eval();
      } else {
        return x.Eval();
      }
    }
    if (min(c1 - x, c2 - x).Match(ret)) {
      if (c1.Eval()->value < c2.Eval()->value) {
        return (c1 - x).Eval();
      } else {
        return (c2 - x).Eval();
      }
    }

    // DivMod rules
    // Divide up rounding: truc div
    // NOTE: trucdiv(x, y) >= floordiv(x, y)
    TVM_TRY_REWRITE_IF(min(((x + c1) / c2) * c2, x), x,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(((x + c1) / c2) * c2, max(x, c2)), max(x, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value &&
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(min(x, ((x + c1) / c2) * c2), x,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(max(x, c2), ((x + c1) / c2) * c2), max(x, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value &&
                       CanProveGreaterEqual(x.Eval(), 0));

    // Divide up rounding: floor div
    TVM_TRY_REWRITE_IF(min(floordiv(x + c1, c2) * c2, x), x,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(floordiv(x + c1, c2) * c2, max(x, c2)), max(x, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);

    TVM_TRY_REWRITE_IF(min(x, floordiv(x + c1, c2) * c2), x,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(max(x, c2), floordiv(x + c1, c2) * c2), max(x, c2),
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);

    TVM_TRY_REWRITE_IF(min(x, floordiv(x, c2) * c2), floordiv(x, c2) * c2,
                       c2.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(min(floordiv(x, c2) * c2, x), floordiv(x, c2) * c2,
                       c2.Eval()->value > 0);

    TVM_TRY_REWRITE(min(max(x, y), min(x, y)), min(x, y));
    TVM_TRY_REWRITE(min(max(x, y), min(y, x)), min(x, y));
    TVM_TRY_REWRITE(min(min(x, y), max(x, y)), min(x, y));
    TVM_TRY_REWRITE(min(min(x, y), max(y, x)), min(x, y));

    TVM_TRY_REWRITE(min(max(x, y), x), x);
    TVM_TRY_REWRITE(min(max(x, y), y), y);
    TVM_TRY_REWRITE(min(min(x, y), x), min(x, y));
    TVM_TRY_REWRITE(min(min(x, y), y), min(x, y));

    TVM_TRY_REWRITE(min(x, max(x, y)), x);
    TVM_TRY_REWRITE(min(y, max(x, y)), y);
    TVM_TRY_REWRITE(min(x, min(x, y)), min(x, y));
    TVM_TRY_REWRITE(min(y, min(x, y)), min(x, y));

    TVM_TRY_REWRITE(min(min(min(x, y), z), y), min(min(x, y), z));
    TVM_TRY_REWRITE(min(min(min(min(x, y), z), s1), y), min(min(min(x, y), z), s1));
    TVM_TRY_REWRITE(min(min(min(min(min(x, y), z), s1), s2), y),
                    min(min(min(min(x, y), z), s1), s2));

    TVM_TRY_REWRITE(min(max(x, y), max(x, z)), max(min(y, z), x));
    TVM_TRY_REWRITE(min(max(x, y), max(z, x)), max(min(y, z), x));
    TVM_TRY_REWRITE(min(max(y, x), max(x, z)), max(min(y, z), x));
    TVM_TRY_REWRITE(min(max(y, x), max(z, x)), max(min(y, z), x));

    TVM_TRY_REWRITE(min(min(x, y), min(x, z)), min(min(y, z), x));
    TVM_TRY_REWRITE(min(min(x, y), min(z, x)), min(min(y, z), x));
    TVM_TRY_REWRITE(min(min(y, x), min(x, z)), min(min(y, z), x));
    TVM_TRY_REWRITE(min(min(y, x), min(z, x)), min(min(y, z), x));

    TVM_TRY_REWRITE(min(y + x, z + x), min(y, z) + x);
    TVM_TRY_REWRITE(min(y + x, x + z), min(y, z) + x);
    TVM_TRY_REWRITE(min(x + y, x + z), min(y, z) + x);
    TVM_TRY_REWRITE(min(x + y, z + x), min(y, z) + x);

    // sub distribution
    TVM_TRY_REWRITE(min(y - x, z - x), min(y, z) - x);
    TVM_TRY_REWRITE(min(x - y, x - z), x - max(y, z));

    // constant folding rule.
    TVM_TRY_REWRITE(min(min(x, c1), c2), min(x, min(c1, c2)));

    // scaling rule
    if (min(x / c1, y / c1).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return (min(x, y) / c1).Eval();
      } else {
        return (max(x, y) / c1).Eval();
      }
    }
    if (min(floordiv(x, c1), floordiv(y, c1)).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return floordiv(min(x, y), c1).Eval();
      } else {
        return floordiv(max(x, y), c1).Eval();
      }
    }
    if (min(x * c1, y * c1).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return (min(x, y) * c1).Eval();
      } else {
        return (max(x, y) * c1).Eval();
      }
    }
    if (min(x * c1, c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c2val % c1val == 0) {
        if (c2val / c1val >= 0) {
          return (min(x, c2val / c1val) * c1val).Eval();
        } else {
          return (max(x, c2val / c1val) * c1val).Eval();
        }
      }
    }

    // canonicalization
    TVM_TRY_RECURSIVE_REWRITE(min(min(x, c1), y), min(min(x, y), c1));
    TVM_TRY_RECURSIVE_REWRITE_IF(
        min(c1 - x, c2), c1 - max(x, c1 - c2),
        c2.Eval()->value != 0);
  }

  // condition rules.
  TVM_TRY_REWRITE(min(select(x, y, z), select(x, s1, s2)),
                  select(x, min(y, s1), min(z, s2)));
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Max* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Max>();
  Expr const_res = TryConstFold<Max>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y, z, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  PVar<int> lanes;

  // vector rule
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(max(broadcast(x, lanes), broadcast(y, lanes)),
                    broadcast(max(x, y), lanes));
    TVM_TRY_REWRITE(max(max(x, broadcast(y, lanes)), broadcast(z, lanes)),
                    max(x, broadcast(max(y, z), lanes)));
  }
  if (IsIndexType(op->type)) {
    TVM_TRY_REWRITE(max(x, x), x);

    // constant int bound
    ConstIntBound a_bound = parent_->const_int_bound(op->a);
    ConstIntBound b_bound = parent_->const_int_bound(op->b);
    if (a_bound->min_value >= b_bound->max_value) {
      return op->a;
    }
    if (b_bound->min_value >= a_bound->max_value) {
      return op->b;
    }

    // constant comparison
    if (max(x + c1, x + c2).Match(ret)) {
      if (c1.Eval()->value > c2.Eval()->value) {
        return (x + c1).Eval();
      } else {
        return (x + c2).Eval();
      }
    }
    if (max(x + c1, x).Match(ret) ||
        max(x, x + c1).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return (x + c1).Eval();
      } else {
        return x.Eval();
      }
    }
    if (max(c1 - x, c2 - x).Match(ret)) {
      if (c1.Eval()->value > c2.Eval()->value) {
        return (c1 - x).Eval();
      } else {
        return (c2 - x).Eval();
      }
    }

    // DivMod rules
    // Divide up rounding: truc div
    // NOTE: trucdiv(x, y) >= floordiv(x, y)
    TVM_TRY_REWRITE_IF(max(((x + c1) / c2) * c2, x), ((x + c1) / c2) * c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, ((x + c1) / c2) * c2), ((x + c1) / c2) * c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);

    // Divide up rounding: floor div
    TVM_TRY_REWRITE_IF(max(floordiv(x + c1, c2) * c2, x), floordiv(x + c1, c2) * c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, floordiv(x + c1, c2) * c2), floordiv(x + c1, c2) * c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value + 1 == c2.Eval()->value);

    TVM_TRY_REWRITE_IF(max(floordiv(x, c2) * c2, x), x,
                       c2.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(max(x, floordiv(x, c2) * c2), x,
                       c2.Eval()->value > 0);

    TVM_TRY_REWRITE(max(min(x, y), max(x, y)), max(x, y));
    TVM_TRY_REWRITE(max(min(x, y), max(y, x)), max(x, y));
    TVM_TRY_REWRITE(max(max(x, y), min(x, y)), max(x, y));
    TVM_TRY_REWRITE(max(max(x, y), min(y, x)), max(x, y));

    TVM_TRY_REWRITE(max(min(x, y), x), x);
    TVM_TRY_REWRITE(max(min(x, y), y), y);
    TVM_TRY_REWRITE(max(max(x, y), x), max(x, y));
    TVM_TRY_REWRITE(max(max(x, y), y), max(x, y));

    TVM_TRY_REWRITE(max(x, min(x, y)), x);
    TVM_TRY_REWRITE(max(y, min(x, y)), y);
    TVM_TRY_REWRITE(max(x, max(x, y)), max(x, y));
    TVM_TRY_REWRITE(max(y, max(x, y)), max(x, y));

    TVM_TRY_REWRITE(max(max(max(x, y), z), y), max(max(x, y), z));
    TVM_TRY_REWRITE(max(max(max(max(x, y), z), s1), y), max(max(max(x, y), z), s1));
    TVM_TRY_REWRITE(max(max(max(max(max(x, y), z), s1), s2), y),
                    max(max(max(max(x, y), z), s1), s2));

    // max/max cancelation
    TVM_TRY_REWRITE(max(max(x, y), max(x, z)), max(max(y, z), x));
    TVM_TRY_REWRITE(max(max(x, y), max(z, x)), max(max(y, z), x));
    TVM_TRY_REWRITE(max(max(y, x), max(x, z)), max(max(y, z), x));
    TVM_TRY_REWRITE(max(max(y, x), max(z, x)), max(max(y, z), x));

    // max/min distribution
    TVM_TRY_REWRITE(max(min(x, y), min(x, z)), min(max(y, z), x));
    TVM_TRY_REWRITE(max(min(x, y), min(z, x)), min(max(y, z), x));
    TVM_TRY_REWRITE(max(min(y, x), min(x, z)), min(max(y, z), x));
    TVM_TRY_REWRITE(max(min(y, x), min(z, x)), min(max(y, z), x));

    // add distribution
    TVM_TRY_REWRITE(max(y + x, z + x), max(y, z) + x);
    TVM_TRY_REWRITE(max(y + x, x + z), max(y, z) + x);
    TVM_TRY_REWRITE(max(x + y, x + z), max(y, z) + x);
    TVM_TRY_REWRITE(max(x + y, z + x), max(y, z) + x);

    // sub distribution
    TVM_TRY_REWRITE(max(y - x, z - x), max(y, z) - x);
    TVM_TRY_REWRITE(max(x - y, x - z), x - min(y, z));

    // constant folding rule.
    TVM_TRY_REWRITE(max(max(x, c1), c2), max(x, max(c1, c2)));

    // scaling rule
    if (max(x / c1, y / c1).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return (max(x, y) / c1).Eval();
      } else {
        return (min(x, y) / c1).Eval();
      }
    }
    if (max(floordiv(x, c1), floordiv(y, c1)).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return floordiv(max(x, y), c1).Eval();
      } else {
        return floordiv(min(x, y), c1).Eval();
      }
    }
    if (max(x * c1, y * c1).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return (max(x, y) * c1).Eval();
      } else {
        return (min(x, y) * c1).Eval();
      }
    }
    if (max(x * c1, c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c2val % c1val == 0) {
        if (c2val / c1val >= 0) {
          return (max(x, c2val / c1val) * c1val).Eval();
        } else {
          return (min(x, c2val / c1val) * c1val).Eval();
        }
      }
    }

    // canonicalization
    TVM_TRY_RECURSIVE_REWRITE(max(max(x, c1), y), max(max(x, y), c1));
    TVM_TRY_RECURSIVE_REWRITE_IF(
        max(c1 - x, c2), c1 - min(x, c1 - c2), c2.Eval()->value != 0);
  }

  // condition rules.
  TVM_TRY_REWRITE(max(select(x, y, z), select(x, s1, s2)),
                  select(x, max(y, s1), max(z, s2)));
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const EQ* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<EQ>();
  Expr const_res = TryConstFold<EQ>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y;
  // Pattern var match IntImm
  PVar<Integer> c1;
  PVar<int> lanes;

  // vector rule
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) == broadcast(y, lanes),
                    broadcast(x == y, lanes));
  }

  if (IsIndexType(op->a.type())) {
    CompareResult result = TryCompare(op->a - op->b, 0);
    if (result == kEQ) {
      return make_const(op->type, true);
    } else if (result == kNE || result == kGT || result == kLT) {
      return make_const(op->type, false);
    }
    TVM_TRY_REWRITE(x - c1 == 0, x == c1);
    TVM_TRY_REWRITE(c1 - x == 0, x == c1);
    TVM_TRY_REWRITE(x + c1 == 0, x == 0 - c1);
    TVM_TRY_REWRITE(x * y == 0, x == 0 || y == 0);
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const NE* op, const Expr& self) {
  return Mutate(Not::make(op->a == op->b));
}

Expr RewriteSimplifier::Impl::
Mutate_(const LE* op, const Expr& self) {
  return Mutate(Not::make(op->b < op->a));
}

Expr RewriteSimplifier::Impl::
Mutate_(const GT* op, const Expr& self) {
  return Mutate(op->b < op->a);
}

Expr RewriteSimplifier::Impl::
Mutate_(const GE* op, const Expr& self) {
  return Mutate(Not::make(op->a < op->b));
}

Expr RewriteSimplifier::Impl::
Mutate_(const LT* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<LT>();
  Expr const_res = TryConstFold<LT>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y, z, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  PVar<int> lanes;

  // vector rule
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) < broadcast(y, lanes),
                    broadcast(x < y, lanes));
    TVM_TRY_REWRITE(ramp(x, s1, lanes) < ramp(y, s1, lanes),
                    broadcast(x < y, lanes));
  }

  if (IsIndexType(op->a.type())) {
    CompareResult result = TryCompare(op->a - op->b, 0);
    if (result == kLT) {
      return make_const(op->type, true);
    }
    if (result == kEQ || result == kGT || result == kGE) {
      return make_const(op->type, false);
    }

    TVM_TRY_REWRITE(x + y < x + z, y < z);
    TVM_TRY_REWRITE(x + y < z + x, y < z);
    TVM_TRY_REWRITE(y + x < x + z, y < z);
    TVM_TRY_REWRITE(y + x < z + x, y < z);
    TVM_TRY_REWRITE(y - x < z - x, y < z);
    TVM_TRY_REWRITE(x - y < x - z, z < y);

    TVM_TRY_REWRITE(x < x + z, 0 < z);
    TVM_TRY_REWRITE(x < z + x, 0 < z);
    TVM_TRY_REWRITE(x < x - z, z < 0);
    TVM_TRY_REWRITE(c1 < x + c2, c1 - c2 < x);
    TVM_TRY_REWRITE(c1 < c2 - x, x < c2 - c1);

    TVM_TRY_REWRITE_IF(x * c1 < y * c1, x < y,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(x * c1 < y * c1, y < x,
                       c1.Eval()->value < 0);

    // constant cancelation: only need to make use of one mod
    // truc div
    TVM_TRY_REWRITE_IF(x * c2 < c1, x < (c1 - 1) / c2 + 1,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value > 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(x * c2 < c1, x < c1 / c2,
                       c1.Eval()->value <= 0 &&
                       c2.Eval()->value > 0);
    // NOTE: trunc div required (euclidean is ok too, floored is not)
    TVM_TRY_REWRITE_IF(x * c2 < c1, (c1 - 1) / c2 - 1 < x,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value < 0);
    // NOTE: trunc div required (floored is ok too, euclidean is not)
    TVM_TRY_REWRITE_IF(x * c2 < c1, c1 / c2 < x,
                       c1.Eval()->value <= 0 &&
                       c2.Eval()->value < 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(c1 < x * c2, (c1 + 1) / c2 - 1 < x,
                       c1.Eval()->value < 0 &&
                       c2.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(c1 < x * c2, c1 / c2 < x,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0);
    // NOTE: trunc div required (floored is ok too, euclidean is not)
    TVM_TRY_REWRITE_IF(c1 < x * c2, x < (c1 + 1) / c2 + 1,
                       c1.Eval()->value < 0 &&
                       c2.Eval()->value < 0);
    // NOTE: trunc div required (euclidean is ok too, floored is not)
    TVM_TRY_REWRITE_IF(c1 < x * c2, x < c1 / c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value < 0);
    // DivMod rules
    // trucdiv
    TVM_TRY_REWRITE_IF(x / c1 < c2, x < c1 * c2,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value > 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(x / c1 < c2, x < c1 * (c2 - 1) + 1,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value <= 0);

    TVM_TRY_REWRITE_IF(c1 < x / c2, (c1 + 1) * c2 - 1 < x,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(c1 < x / c2, c1 * c2 < x,
                       c1.Eval()->value < 0 &&
                       c2.Eval()->value > 0);

    // invariance for any div mod: x - (x / c1) * c1 == x % c1
    TVM_TRY_REWRITE_IF((x / c1) * c1 < x, 0 < x % c1,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF((x / c1) * c1 < x + y, 0 < x % c1 + y,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF((x / c1) * c1 < x - y, y < x % c1,
                       c1.Eval()->value > 0);

    TVM_TRY_REWRITE_IF(((x + c2) / c1) * c1 < x,
                       c2 < (x + c2) % c1,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(((x + c2) / c1) * c1 < x + y,
                       c2 < (x + c2) % c1 + y,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(((x + c2) / c1) * c1 < x - y,
                       y < (x + c2) % c1 + (0 - c2),
                       c1.Eval()->value > 0);

    // floordiv
    TVM_TRY_REWRITE_IF(floordiv(x, c1) < c2, x < c1 * c2,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(c1 < floordiv(x, c2), (c1 + 1) * c2 - 1 < x,
                       c2.Eval()->value > 0);

    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x, 0 < floormod(x, c1),
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x + y, 0 < floormod(x, c1) + y,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x - y, y < floormod(x, c1),
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c2, c1) * c1 < x,
                       c2 < floormod(x + c2, c1),
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c2, c1) * c1 < x + y,
                       c2 < floormod(x + c2, c1) + y,
                       c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c2, c1) * c1 < x - y,
                       y < floormod(x + c2, c1) + (0 - c2),
                       c1.Eval()->value > 0);

    // canonicalization rule
    TVM_TRY_RECURSIVE_REWRITE(min(x, y) < z, x < z || y < z);
    TVM_TRY_RECURSIVE_REWRITE(max(x, y) < z, x < z && y < z);
    TVM_TRY_RECURSIVE_REWRITE(z < min(x, y), z < x && z < y);
    TVM_TRY_RECURSIVE_REWRITE(z < max(x, y), z < x || z < y);

    TVM_TRY_RECURSIVE_REWRITE(x < c1 - y, x + y < c1);
    TVM_TRY_RECURSIVE_REWRITE(x < c1 + y, x - y < c1);
    TVM_TRY_RECURSIVE_REWRITE(c1 - y < x, c1 < x + y);
    TVM_TRY_RECURSIVE_REWRITE(c1 + y < x, c1 < x - y);

    TVM_TRY_RECURSIVE_REWRITE(x + c1 < c2, x < c2 - c1);
    TVM_TRY_RECURSIVE_REWRITE(x - c1 < c2, x < c2 + c1);
    TVM_TRY_REWRITE(x - c1 < 0, x < c1);
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Not* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Not>();
  Expr const_res = TryConstFold<Not>(op->a);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y;
  PVar<int> lanes;
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(!broadcast(x, lanes), broadcast(!x, lanes));
  }

  TVM_TRY_REWRITE(!(!x), x);
  TVM_TRY_REWRITE(!(x <= y), y < x);
  TVM_TRY_REWRITE(!(x >= y), x < y);
  TVM_TRY_REWRITE(!(x < y), y <= x);
  TVM_TRY_REWRITE(!(x > y), x <= y);
  TVM_TRY_REWRITE(!(x == y), x != y);
  TVM_TRY_REWRITE(!(x != y), x == y);
  TVM_TRY_RECURSIVE_REWRITE(!(x || y), (!x) && (!y));
  TVM_TRY_RECURSIVE_REWRITE(!(x && y), (!x) || (!y));
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const And* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<And>();
  Expr const_res = TryConstFold<And>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  PVar<int> lanes;

  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) && broadcast(y, lanes),
                    broadcast(x && y, lanes));
  }

  auto cfalse = PConst<Expr>(make_const(op->type, false));
  TVM_TRY_REWRITE(x == y && x != y, cfalse);
  TVM_TRY_REWRITE(x != y && x == y, cfalse);
  TVM_TRY_REWRITE(x && !x, cfalse);
  TVM_TRY_REWRITE(x <= y && y < x, cfalse);
  TVM_TRY_REWRITE(y < x && y <= x, cfalse);

  TVM_TRY_REWRITE_IF(x < c1 && c2 < x, cfalse,
                     c2.Eval()->value + 1 >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x && x < c1, cfalse,
                     c2.Eval()->value + 1 >= c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x < c1 && c2 <= x, cfalse,
                     c2.Eval()->value >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 <= x && x < c1, cfalse,
                     c2.Eval()->value >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(x <= c1 && c2 < x, cfalse,
                     c2.Eval()->value >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x && x <= c1, cfalse,
                     c2.Eval()->value >= c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x <= c1 && c2 <= x, cfalse,
                     c2.Eval()->value > c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 <= x && x <= c1, cfalse,
                     c2.Eval()->value > c1.Eval()->value);

  TVM_TRY_REWRITE(x == c1 && x != c2, x == c1 && c1 != c2);
  TVM_TRY_REWRITE(x != c2 && x == c1, x == c1 && c1 != c2);
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Or* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Or>();
  Expr const_res = TryConstFold<Or>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  PVar<int> lanes;

  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) || broadcast(y, lanes),
                    broadcast(x || y, lanes));
  }

  auto ctrue = PConst<Expr>(make_const(op->type, true));

  TVM_TRY_REWRITE(x == y || x != y, ctrue);
  TVM_TRY_REWRITE(x != y || x == y, ctrue);
  TVM_TRY_REWRITE(x || !x, ctrue);
  TVM_TRY_REWRITE(x <= y || y < x, ctrue);
  TVM_TRY_REWRITE(y < x || x <= y, ctrue);

  TVM_TRY_REWRITE_IF(x < c1 || c2 < x, ctrue,
                     c2.Eval()->value < c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x || x < c1, ctrue,
                     c2.Eval()->value < c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x <= c1 || c2 < x, ctrue,
                     c2.Eval()->value <= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x || x <= c1, ctrue,
                     c2.Eval()->value <= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(x < c1 || c2 <= x, ctrue,
                     c2.Eval()->value <= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 <= x || x < c1, ctrue,
                     c2.Eval()->value <= c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x <= c1 || c2 <= x, ctrue,
                     c2.Eval()->value <= c1.Eval()->value + 1);
  TVM_TRY_REWRITE_IF(c2 <= x || x <= c1, ctrue,
                     c2.Eval()->value <= c1.Eval()->value + 1);

  TVM_TRY_REWRITE(x != c1 || x == c2, x != c1 || c1 == c2);
  TVM_TRY_REWRITE(x == c2 || x != c1, x != c1 || c1 == c2);
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Select* op, const Expr& self) {
  Expr cond = Mutate(op->condition);
  Expr true_value, false_value;
  {
    With<ConstraintContext> constraint(parent_, cond);
    true_value = Mutate(op->true_value);
  }
  {
    With<ConstraintContext> constraint(parent_, Mutate(Not::make(cond)));
    false_value = Mutate(op->false_value);
  }
  if (is_zero(cond)) {
    return false_value;
  }
  if (is_one(cond)) {
    return true_value;
  }
  // normal path
  Expr ret;
  if (cond.same_as(op->condition) &&
      true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    ret = self;
  } else {
    ret = Select::make(cond, true_value, false_value);
  }
  op = ret.as<Select>();
  // Pattern var to match any expression
  PVar<Expr> x, y;
  TVM_TRY_REWRITE(select(x, y, y), y);
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Call* op, const Expr& self) {
  // add condition context to if_then_else
  Expr ret;
  if (op->is_intrinsic(ir::intrinsic::tvm_if_then_else)) {
    Expr cond = Mutate(op->args[0]);
    Expr true_value, false_value;
    {
      With<ConstraintContext> constraint(parent_, cond);
      true_value = Mutate(op->args[1]);
    }
    {
      With<ConstraintContext> constraint(parent_, Mutate(Not::make(cond)));
      false_value = Mutate(op->args[2]);
    }
    if (is_zero(cond)) {
      return false_value;
    }
    if (is_one(cond)) {
      return true_value;
    }
    if (cond.same_as(op->args[0]) &&
        true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      ret = self;
    } else {
      ret = Call::make(op->type, op->name,
                        {cond, true_value, false_value},
                        op->call_type);
    }
  } else {
    ret = IRMutator::Mutate_(op, self);
  }
  op = ret.as<Call>();
  if (op->is_intrinsic(Call::likely) && is_const(op->args[0])) {
    return op->args[0];
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Let* op, const Expr& self) {
  // For now assume value does not has side-effect.
  Expr value = this->Mutate(op->value);
  if (!ir::HasSideEffect(value)) {
    parent_->Bind(op->var, value);
    return this->Mutate(op->body);
  }
  Expr body = this->Mutate(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return self;
  } else {
    return Let::make(op->var, value, body);
  }
}

Expr RewriteSimplifier::Impl::
Mutate_(const Variable* op, const Expr& self) {
  Var var = GetRef<Var>(op);
  auto it = var_map_.find(var);
  if (it != var_map_.end()) {
    return it->second;
  }
  return self;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Cast* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Cast>();
  return cast(op->type, op->value);
}

Expr RewriteSimplifier::operator()(const Expr& expr) {
  // Run simplification in post order
  Expr res = expr;
  int max_iter = 2;
  for (int i = 0; i < max_iter; ++i) {
    Expr new_expr = impl_->Mutate(res);
    if (new_expr.same_as(res)) return res;
    res = new_expr;
  }
  return res;
}

void RewriteSimplifier::Update(const Var& var,
                               const Expr& info,
                               bool override) {
  impl_->Update(var, info, override);
}

RewriteSimplifier::RewriteSimplifier(Analyzer* parent)
    : impl_(new Impl(parent)) {
}

RewriteSimplifier::~RewriteSimplifier() {
  delete impl_;
}

}  // namespace arith
}  // namespace tvm
