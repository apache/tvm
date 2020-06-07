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
 * \file rewrite_simplify.cc
 * \brief Rewrite-rule based simplification.
 */
// Acknowledgement: Most rewrite-rules are from Halide.
#include "rewrite_simplify.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

#include <algorithm>

#include "const_fold.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

// macro for doing simple rewrite
#define TVM_TRY_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {             \
    return (ResExpr).Eval();              \
  }

// macro for rewrite + recursively rewrite ResExpr
#define TVM_TRY_RECURSIVE_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {                       \
    return RecursiveRewrite((ResExpr).Eval());      \
  }

// macro rewrite only if CondExor is true after match.
#define TVM_TRY_REWRITE_IF(SrcExpr, ResExpr, CondExpr) \
  if ((SrcExpr).Match(ret) && (CondExpr)) {            \
    return (ResExpr).Eval();                           \
  }

// macro rewrite + recursive_rewrite only if CondExor is true after match.
#define TVM_TRY_RECURSIVE_REWRITE_IF(SrcExpr, ResExpr, CondExpr) \
  if ((SrcExpr).Match(ret) && (CondExpr)) {                      \
    return RecursiveRewrite((ResExpr).Eval());                   \
  }

// NOTE for developers:
//
// We mainly focus on index expression simplification.
// Besides the RewriteSimplifier, some cases can be better
// handled by CanonicalSimplifier.
//

// try to prove x equals val
RewriteSimplifier::Impl::CompareResult RewriteSimplifier::Impl::TryCompare(const PrimExpr& x,
                                                                           int64_t val) {
  PrimExpr diff = this->VisitExpr(x);
  if (const auto* ptr = diff.as<IntImmNode>()) {
    if (ptr->value == val) {
      return kEQ;
    } else if (ptr->value > val) {
      return kGT;
    } else if (ptr->value < val) {
      return kLT;
    }
  }
  ConstIntBound dbound = analyzer_->const_int_bound(diff);
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
    ModularSet dmod = analyzer_->modular_set(diff);
    if (dmod->base != 0) {
      return kNE;
    }
  }
  return kUnknown;
}

void RewriteSimplifier::Impl::Update(const Var& var, const PrimExpr& info, bool can_override) {
  if (!can_override) {
    auto it = var_map_.find(var);
    if (it != var_map_.end()) {
      CHECK(ExprDeepEqual()(it->second, info)) << "Trying to update var \'" << var << "\'"
                                               << " with a different value: "
                                               << "original=" << it->second << ", new=" << info;
    }
  }
  var_map_[var] = info;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const AddNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<AddNode>();
  PrimExpr const_res = TryConstFold<AddNode>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) + ramp(b2, s2, lanes), ramp(b1 + b2, s1 + s2, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) + broadcast(x, lanes), ramp(b1 + x, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) + ramp(b1, s1, lanes), ramp(x + b1, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) + broadcast(y, lanes), broadcast(x + y, lanes));
  }

  if (IsIndexType(op->dtype)) {
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

    TVM_TRY_REWRITE_IF(min(x, y + c1) + c2, min(x + c2, y), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(x + c1, y) + c2, min(x, y + c2), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, y + c1) + c2, max(x + c2, y), c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x + c1, y) + c2, max(x, y + c2), c1.Eval()->value == -c2.Eval()->value);

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
    TVM_TRY_REWRITE(truncdiv(x, c1) * c1 + truncmod(x, c1), x);
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
    TVM_TRY_RECURSIVE_REWRITE(truncmod(y, c1) + x * c1, x * c1 + truncmod(y, c1));
    // floor div
    TVM_TRY_RECURSIVE_REWRITE(floormod(y, c1) + x * c1, x * c1 + floormod(y, c1));
  }

  // condition rules.
  TVM_TRY_REWRITE(select(x, b1, b2) + select(x, s1, s2), select(x, b1 + s1, b2 + s2));
  // default value
  return ret;
}

std::function<void()> RewriteSimplifier::Impl::EnterConstraint(const PrimExpr& constraint) {
  size_t old_literal_size = literal_constraints_.size();
  // we will compare the already simplified result with the constraint,
  // so simplify the constarint as well
  literal_constraints_.push_back(operator()(constraint));
  size_t new_literal_size = literal_constraints_.size();
  auto frecover = [old_literal_size, new_literal_size, this]() {
    CHECK_EQ(literal_constraints_.size(), new_literal_size);
    literal_constraints_.resize(old_literal_size);
  };
  return frecover;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const SubNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<SubNode>();
  PrimExpr const_res = TryConstFold<SubNode>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) - ramp(b2, s2, lanes), ramp(b1 - b2, s1 - s2, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) - broadcast(x, lanes), ramp(b1 - x, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) - ramp(b1, s1, lanes), ramp(x - b1, 0 - s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) - broadcast(y, lanes), broadcast(x - y, lanes));
  }

  if (IsIndexType(op->dtype)) {
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

    TVM_TRY_REWRITE(min(x + y, z) - x, min(y, z - x));
    TVM_TRY_REWRITE(min(y + x, z) - x, min(y, z - x));
    TVM_TRY_REWRITE(min(z, x + y) - x, min(z - x, y));
    TVM_TRY_REWRITE(min(z, y + x) - x, min(z - x, y));

    TVM_TRY_REWRITE(max(x + y, z) - x, max(y, z - x));
    TVM_TRY_REWRITE(max(y + x, z) - x, max(y, z - x));
    TVM_TRY_REWRITE(max(z, x + y) - x, max(z - x, y));
    TVM_TRY_REWRITE(max(z, y + x) - x, max(z - x, y));

    TVM_TRY_REWRITE(x - min(x + y, z), max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(y + x, z), max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(z, x + y), max(x - z, 0 - y));
    TVM_TRY_REWRITE(x - min(z, y + x), max(x - z, 0 - y));

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
    TVM_TRY_REWRITE_IF(x - truncdiv(x, c1) * c1, truncmod(x, c1), c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(truncdiv(x, c1) * c1 - x, 0 - truncmod(x, c1), c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - (truncdiv(x + y, c1)) * c1, truncmod(x + y, c1) - y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF((truncdiv(x + y, c1)) * c1 - x, y - truncmod(x + y, c1),
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - truncdiv(x - y, c1) * c1, truncmod(x - y, c1) + y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(truncdiv(x - y, c1) * c1 - x, 0 - truncmod(x - y, c1) - y,
                       c1.Eval()->value != 0);

    TVM_TRY_REWRITE_IF(
        x * c2 - truncdiv(x, c1) * c3, truncmod(x, c1) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        truncdiv(x, c1) * c3 - x * c2, 0 - truncmod(x, c1) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        x * c2 - truncdiv(x + y, c1) * c3, (truncmod(x + y, c1) - y) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        truncdiv(x + y, c1) * c3 - x * c2, (y - truncmod(x + y, c1)) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        x * c2 - truncdiv(x - y, c1) * c3, (truncmod(x - y, c1) + y) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        truncdiv(x - y, c1) * c3 - x * c2, (0 - truncmod(x - y, c1) - y) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);

    // Proof in the case of floordiv, need positive condition.
    // let x = a * c3 + r
    // (x + c1) / c3 - x / c3 => (r + c1) / c3
    // NOTE: the use of floormod(c2, c3) was intentional to simplify the const.
    TVM_TRY_REWRITE_IF(truncdiv(x + c1, c3) - truncdiv(x + c2, c3),
                       truncdiv(truncmod(x + floormod(c2, c3), c3) + (c1 - c2), c3),
                       CanProveGreaterEqual(x.Eval(), -c2.Eval()->value) &&
                           c1.Eval()->value >= c2.Eval()->value && c3.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(
        truncdiv(x + c1, c3) - truncdiv(x, c3), truncdiv(truncmod(x, c3) + c1, c3),
        CanProveGreaterEqual(x.Eval(), 0) && c1.Eval()->value >= 0 && c3.Eval()->value > 0);

    // floordiv
    TVM_TRY_REWRITE_IF(x - floordiv(x, c1) * c1, floormod(x, c1), c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 - x, 0 - floormod(x, c1), c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - floordiv(x + y, c1) * c1, floormod(x + y, c1) - y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(floordiv(x + y, c1) * c1 - x, y - floormod(x + y, c1),
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(x - floordiv(x - y, c1) * c1, floormod(x - y, c1) + y,
                       c1.Eval()->value != 0);
    TVM_TRY_REWRITE_IF(floordiv(x - y, c1) * c1 - x, 0 - floormod(x - y, c1) - y,
                       c1.Eval()->value != 0);

    TVM_TRY_REWRITE_IF(
        x * c2 - floordiv(x, c1) * c3, floormod(x, c1) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        floordiv(x, c1) * c3 - x * c2, 0 - floormod(x, c1) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        x * c2 - floordiv(x + y, c1) * c3, (floormod(x + y, c1) - y) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        floordiv(x + y, c1) * c3 - x * c2, (y - floormod(x + y, c1)) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        x * c2 - floordiv(x - y, c1) * c3, (floormod(x - y, c1) + y) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);
    TVM_TRY_REWRITE_IF(
        floordiv(x - y, c1) * c3 - x * c2, (0 - floormod(x - y, c1) - y) * c2,
        c1.Eval()->value != 0 && c3.Eval()->value == c1.Eval()->value * c2.Eval()->value);

    TVM_TRY_REWRITE_IF(floordiv(x + c1, c3) - floordiv(x + c2, c3),
                       floordiv(floormod(x + floormod(c2, c3), c3) + (c1 - c2), c3),
                       c3.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c1, c3) - floordiv(x, c3), floordiv(floormod(x, c3) + c1, c3),
                       c3.Eval()->value > 0);

    // canonicalization rule
    // will try rewrite again after canonicalization.
    TVM_TRY_REWRITE(x - c1, x + (0 - c1));
    TVM_TRY_RECURSIVE_REWRITE((x + c1) - y, (x - y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x - (y - z), (x + z) - y);
    TVM_TRY_RECURSIVE_REWRITE(x - y * c1, x + y * (0 - c1));
  }

  // condition rules.
  TVM_TRY_REWRITE(select(x, b1, b2) - select(x, s1, s2), select(x, b1 - s1, b2 - s2));
  TVM_TRY_REWRITE(select(x, y, z) - z, select(x, y - z, ZeroWithTypeLike(z)));
  TVM_TRY_REWRITE(select(x, y, z) - y, select(x, ZeroWithTypeLike(y), z - y));
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const MulNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<MulNode>();
  PrimExpr const_res = TryConstFold<MulNode>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) * broadcast(y, lanes), broadcast(x * y, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) * broadcast(x, lanes), ramp(b1 * x, s1 * x, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) * ramp(b1, s1, lanes), ramp(b1 * x, s1 * x, lanes));
  }

  if (IsIndexType(op->dtype)) {
    // constant simplification rule
    TVM_TRY_REWRITE((x + c1) * c2, x * c2 + c1 * c2);
    TVM_TRY_REWRITE((x * c1) * c2, x * (c1 * c2));
    TVM_TRY_REWRITE(min(x, y) * max(x, y), x * y);
    TVM_TRY_REWRITE(max(x, y) * min(x, y), x * y);

    // canonicalization
    TVM_TRY_RECURSIVE_REWRITE(x * (c1 * y), (x * y) * c1);
    TVM_TRY_RECURSIVE_REWRITE(c1 * x, x * c1);
    TVM_TRY_RECURSIVE_REWRITE_IF((x - y) * c1, (y - x) * (0 - c1), c1.Eval()->value < 0);
  }
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const DivNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<DivNode>();
  PrimExpr const_res = TryConstFold<DivNode>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // x / 2.0 = x * 0.5
  if (const FloatImmNode* ptr = op->b.as<FloatImmNode>()) {
    CHECK(op->dtype.is_float());
    return op->a * make_const(op->b.dtype(), 1.0 / ptr->value);
  }

  // Vector rules
  if (op->dtype.lanes() != 1) {
    // NOTE: use div as the pattern also works for float.
    TVM_TRY_REWRITE(div(broadcast(x, lanes), broadcast(y, lanes)), broadcast(div(x, y), lanes));
    // ramp / bcast
    if ((div(ramp(b1, c1, lanes), broadcast(c2, lanes))).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return ramp(div(b1, c2), div(c1, c2), lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      if (CanProveGreaterEqual(b1.Eval(), 0)) {
        ModularSet bmod = analyzer_->modular_set(b1.Eval());
        int64_t ramp_min = bmod->base / c2val;
        int64_t ramp_max = (bmod->base + (lanes.Eval() - 1) * c1val) / c2val;
        if (bmod->coeff % c2val == 0 && ramp_min == ramp_max) {
          return broadcast(div(b1, c2), lanes).Eval();
        }
      }
    }
  }

  if (IsIndexType(op->dtype)) {
    // Be-aware of the division rules:
    // We adopt the default C division uses truncation instead of floordiv.
    // This means most rules need to check non-negativeness of the operands.

    // TryConstFold doesn't work for negative cases because it is also used by legacy
    // parts of tvm which still assume euclidean div. In this simplifier we assume that the division
    // is truncated, so perform const folding again.
    // NOTE: trunc div required
    if (truncdiv(c1, c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      return make_const(op->dtype, truncdiv(c1val, c2val));
    }

    // while it is always true for trunc div
    // restrict to common case(positive div)
    TVM_TRY_REWRITE_IF(truncdiv(truncdiv(x, c1), c2), truncdiv(x, c1 * c2),
                       c1.Eval()->value > 0 && c2.Eval()->value > 0);

    TVM_TRY_REWRITE_IF(truncdiv(truncdiv(x, c1) + c2, c3), truncdiv(x + c1 * c2, c1 * c3),
                       c1.Eval()->value > 0 && c2.Eval()->value >= 0 && c3.Eval()->value > 0 &&
                           CanProveGreaterEqual(x.Eval(), 0));

    if (truncdiv(x * c1, c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val > 0 && c2val > 0) {
        if (c1val % c2val == 0) return (x * truncdiv(c1, c2)).Eval();
        if (c2val % c1val == 0) return truncdiv(x, truncdiv(c2, c1)).Eval();
      }
    }

    TVM_TRY_REWRITE(truncdiv(x, x), OneWithTypeLike(x));
    TVM_TRY_REWRITE(truncdiv(x * c1, x), c1);
    TVM_TRY_REWRITE(truncdiv(c1 * x, x), c1);

    // Rules involving 2-operands.
    TVM_TRY_REWRITE_IF(truncdiv(x * c1 + y, c2), x * truncdiv(c1, c2) + truncdiv(y, c2),
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(min(x * c1, y), c2), min(x * truncdiv(c1, c2), truncdiv(y, c2)),
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(max(x * c1, y), c2), max(x * truncdiv(c1, c2), truncdiv(y, c2)),
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(y + x * c1, c2), truncdiv(y, c2) + x * truncdiv(c1, c2),
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(min(y, x * c1), c2), min(truncdiv(y, c2), x * truncdiv(c1, c2)),
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(max(y, x * c1), c2), max(truncdiv(y, c2), x * truncdiv(c1, c2)),
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    // Rules involving 3-operands.
    TVM_TRY_REWRITE_IF(
        truncdiv(x * c1 + y + z, c2), x * truncdiv(c1, c2) + truncdiv(y + z, c2),
        c1.Eval()->value >= 0 && c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0 &&
            CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF(
        truncdiv(x * c1 - y + z, c2), x * truncdiv(c1, c2) + truncdiv(z - y, c2),
        c1.Eval()->value >= 0 && c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0 &&
            CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((z - y).Eval(), 0));

    TVM_TRY_REWRITE_IF(
        truncdiv(x * c1 + y - z, c2), x * truncdiv(c1, c2) + truncdiv(y - z, c2),
        c1.Eval()->value >= 0 && c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0 &&
            CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((y - z).Eval(), 0));

    TVM_TRY_REWRITE_IF(
        truncdiv(y + x * c1 + z, c2), x * truncdiv(c1, c2) + truncdiv(y + z, c2),
        c1.Eval()->value > 0 && c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0 &&
            CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(x + c1, c2), truncdiv(x, c2) + truncdiv(c1, c2),
                       c1.Eval()->value > 0 && c2.Eval()->value > 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(x + y, x), truncdiv(y, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF(truncdiv(y + x, x), truncdiv(y, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(
        truncdiv((x + y) + z, x), truncdiv(y + z, x) + 1,
        CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF(
        truncdiv((y + x) + z, x), truncdiv(y + z, x) + 1,
        CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF(
        truncdiv(y + (z + x), x), truncdiv(y + z, x) + 1,
        CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF(
        truncdiv(y + (x + z), x), truncdiv(y + z, x) + 1,
        CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(x * y, y), x,
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF(truncdiv(y * x, y), x,
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncdiv(x * z + y, z), x + truncdiv(y, z),
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0) &&
                           CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF(truncdiv(z * x + y, z), x + truncdiv(y, z),
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0) &&
                           CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF(truncdiv(y + x * z, z), truncdiv(y, z) + x,
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0) &&
                           CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF(truncdiv(y + z * x, z), truncdiv(y, z) + x,
                       CanProveGreaterEqual(x.Eval(), 0) && CanProveGreaterEqual(y.Eval(), 0) &&
                           CanProveGreaterEqual(z.Eval(), 0));
  }
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const ModNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<ModNode>();
  PrimExpr const_res = TryConstFold<ModNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(truncmod(broadcast(x, lanes), broadcast(y, lanes)),
                    broadcast(truncmod(x, y), lanes));

    // ramp % bcast
    if (truncmod(ramp(b1, c1, lanes), broadcast(c2, lanes)).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return broadcast(truncmod(b1, c2), lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      if (CanProveGreaterEqual(b1.Eval(), 0)) {
        ModularSet bmod = analyzer_->modular_set(b1.Eval());
        int64_t ramp_min = bmod->base / c2val;
        int64_t ramp_max = (bmod->base + (lanes.Eval() - 1) * c1val) / c2val;
        if (bmod->coeff % c2val == 0) {
          if (ramp_min == ramp_max) {
            return ramp(truncmod(bmod->base, c2), c1, lanes).Eval();
          } else {
            return truncmod(ramp(truncmod(bmod->base, c2), c1, lanes), broadcast(c2, lanes)).Eval();
          }
        }
      }
    }
  }

  if (IsIndexType(op->dtype)) {
    // Be-aware of the division rules:
    // We adopt the default C division uses truncation instead of floordiv.
    // This means most rules need to check non-negativeness of the operands.
    TVM_TRY_REWRITE_IF(truncmod(x * c1, c2), ZeroWithTypeLike(x),
                       c2.Eval()->value != 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(truncmod(x * c1 + y, c2), truncmod(y, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual((x * c1).Eval(), 0) &&
                           CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncmod(x + c1, c2), truncmod(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value >= 0 &&
                           c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(truncmod(x + y * c1, c2), truncmod(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0 &&
                           CanProveGreaterEqual(x.Eval(), 0) &&
                           CanProveGreaterEqual((y * c1).Eval(), 0));

    // canonicalization: x % c == x % (-c) for truncated division
    // NOTE: trunc div required
    TVM_TRY_RECURSIVE_REWRITE_IF(
        truncmod(x, c1), truncmod(x, PConst<PrimExpr>(make_const(op->dtype, -c1.Eval()->value))),
        c1.Eval()->value < 0);

    // try modular analysis
    if (truncmod(x, c1).Match(ret)) {
      ModularSet mod = analyzer_->modular_set(x.Eval());
      int64_t c1val = c1.Eval()->value;
      if (mod->coeff % c1val == 0 && c1val > 0 && CanProveGreaterEqual(x.Eval(), 0)) {
        return truncmod(mod->base, c1).Eval();
      }
    }
  }
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const FloorDivNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<FloorDivNode>();
  PrimExpr const_res = TryConstFold<FloorDivNode>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->dtype.lanes() != 1) {
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
      ModularSet bmod = analyzer_->modular_set(b1.Eval());
      int64_t ramp_min = floordiv(bmod->base, c2val);
      int64_t ramp_max = floordiv(bmod->base + (lanes.Eval() - 1) * c1val, c2val);
      if (bmod->coeff % c2val == 0 && ramp_min == ramp_max) {
        return broadcast(floordiv(b1, c2), lanes).Eval();
      }
    }
  }

  if (IsIndexType(op->dtype)) {
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
        if (c2val % c1val == 0) return floordiv(x, floordiv(c2, c1)).Eval();
      }
    }

    TVM_TRY_REWRITE(floordiv(x, x), OneWithTypeLike(x));
    TVM_TRY_REWRITE(floordiv(x * c1, x), c1);
    TVM_TRY_REWRITE(floordiv(c1 * x, x), c1);

    // Rules involving 2-operands.
    TVM_TRY_REWRITE_IF(floordiv(x * c1 + y, c2), x * floordiv(c1, c2) + floordiv(y, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(min(x * c1, y), c2), min(x * floordiv(c1, c2), floordiv(y, c2)),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(max(x * c1, y), c2), max(x * floordiv(c1, c2), floordiv(y, c2)),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(y + x * c1, c2), floordiv(y, c2) + x * floordiv(c1, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(min(y, x * c1), c2), min(floordiv(y, c2), x * floordiv(c1, c2)),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(max(y, x * c1), c2), max(floordiv(y, c2), x * floordiv(c1, c2)),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    // Rules involving 3-operands.
    TVM_TRY_REWRITE_IF(floordiv(x * c1 + y + z, c2), x * floordiv(c1, c2) + floordiv(y + z, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x * c1 - y + z, c2), x * floordiv(c1, c2) + floordiv(z - y, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x * c1 + y - z, c2), x * floordiv(c1, c2) + floordiv(y - z, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(y + x * c1 + z, c2), x * floordiv(c1, c2) + floordiv(y + z, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x + c1, c2), floordiv(x, c2) + floordiv(c1, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floordiv(x + y, x), floordiv(y, x) + 1, CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(floordiv(y + x, x), floordiv(y, x) + 1, CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(floordiv((x + y) + z, x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv((y + x) + z, x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y + (z + x), x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y + (x + z), x), floordiv(y + z, x) + 1,
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(floordiv(x * y, y), x, CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF(floordiv(y * x, y), x, CanProveGreaterEqual(y.Eval(), 0));

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

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const FloorModNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<FloorModNode>();
  PrimExpr const_res = TryConstFold<FloorModNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->dtype.lanes() != 1) {
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
      ModularSet bmod = analyzer_->modular_set(b1.Eval());
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

  if (IsIndexType(op->dtype)) {
    // Be-aware of the division rules: we use floordiv/floormod here
    TVM_TRY_REWRITE_IF(floormod(x * c1, c2), ZeroWithTypeLike(x),
                       c2.Eval()->value != 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floormod(x * c1 + y, c2), floormod(y, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floormod(x + c1, c2), floormod(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF(floormod(x + y * c1, c2), floormod(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value % c2.Eval()->value == 0);

    // try modular analysis
    if (floormod(x, c1).Match(ret)) {
      ModularSet mod = analyzer_->modular_set(x.Eval());
      int64_t c1val = c1.Eval()->value;
      if (mod->coeff % c1val == 0 && c1val > 0) {
        return floormod(mod->base, c1).Eval();
      }
    }
  }
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const MinNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<MinNode>();
  PrimExpr const_res = TryConstFold<MinNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, s1, s2;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  PVar<int> lanes;

  // vector rule
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(min(broadcast(x, lanes), broadcast(y, lanes)), broadcast(min(x, y), lanes));
    TVM_TRY_REWRITE(min(min(x, broadcast(y, lanes)), broadcast(z, lanes)),
                    min(x, broadcast(min(y, z), lanes)));
  }
  if (IsIndexType(op->dtype)) {
    TVM_TRY_REWRITE(min(x, x), x);

    // constant int bound
    ConstIntBound a_bound = analyzer_->const_int_bound(op->a);
    ConstIntBound b_bound = analyzer_->const_int_bound(op->b);
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
    if (min(x + c1, x).Match(ret) || min(x, x + c1).Match(ret)) {
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
    TVM_TRY_REWRITE_IF(min(truncdiv(x + c1, c2) * c2, x), x,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(truncdiv(x + c1, c2) * c2, max(x, c2)), max(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value &&
                           CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF(min(x, truncdiv(x + c1, c2) * c2), x,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(max(x, c2), truncdiv(x + c1, c2) * c2), max(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value &&
                           CanProveGreaterEqual(x.Eval(), 0));

    // Divide up rounding: floor div
    TVM_TRY_REWRITE_IF(min(floordiv(x + c1, c2) * c2, x), x,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(floordiv(x + c1, c2) * c2, max(x, c2)), max(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);

    TVM_TRY_REWRITE_IF(min(x, floordiv(x + c1, c2) * c2), x,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(max(x, c2), floordiv(x + c1, c2) * c2), max(x, c2),
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);

    TVM_TRY_REWRITE_IF(min(x, floordiv(x, c2) * c2), floordiv(x, c2) * c2, c2.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(min(floordiv(x, c2) * c2, x), floordiv(x, c2) * c2, c2.Eval()->value > 0);

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
    if (min(truncdiv(x, c1), truncdiv(y, c1)).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return truncdiv(min(x, y), c1).Eval();
      } else {
        return truncdiv(max(x, y), c1).Eval();
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
    TVM_TRY_RECURSIVE_REWRITE_IF(min(c1 - x, c2), c1 - max(x, c1 - c2), c2.Eval()->value != 0);
  }

  // condition rules.
  TVM_TRY_REWRITE(min(select(x, y, z), select(x, s1, s2)), select(x, min(y, s1), min(z, s2)));
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const MaxNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<MaxNode>();
  PrimExpr const_res = TryConstFold<MaxNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, s1, s2;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  PVar<int> lanes;

  // vector rule
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(max(broadcast(x, lanes), broadcast(y, lanes)), broadcast(max(x, y), lanes));
    TVM_TRY_REWRITE(max(max(x, broadcast(y, lanes)), broadcast(z, lanes)),
                    max(x, broadcast(max(y, z), lanes)));
  }
  if (IsIndexType(op->dtype)) {
    TVM_TRY_REWRITE(max(x, x), x);

    // constant int bound
    ConstIntBound a_bound = analyzer_->const_int_bound(op->a);
    ConstIntBound b_bound = analyzer_->const_int_bound(op->b);
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
    if (max(x + c1, x).Match(ret) || max(x, x + c1).Match(ret)) {
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
    TVM_TRY_REWRITE_IF(max(truncdiv(x + c1, c2) * c2, x), truncdiv(x + c1, c2) * c2,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, truncdiv(x + c1, c2) * c2), truncdiv(x + c1, c2) * c2,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);

    // Divide up rounding: floor div
    TVM_TRY_REWRITE_IF(max(floordiv(x + c1, c2) * c2, x), floordiv(x + c1, c2) * c2,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, floordiv(x + c1, c2) * c2), floordiv(x + c1, c2) * c2,
                       c2.Eval()->value > 0 && c1.Eval()->value + 1 == c2.Eval()->value);

    TVM_TRY_REWRITE_IF(max(floordiv(x, c2) * c2, x), x, c2.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(max(x, floordiv(x, c2) * c2), x, c2.Eval()->value > 0);

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
    if (max(truncdiv(x, c1), truncdiv(y, c1)).Match(ret)) {
      if (c1.Eval()->value > 0) {
        return truncdiv(max(x, y), c1).Eval();
      } else {
        return truncdiv(min(x, y), c1).Eval();
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
    TVM_TRY_RECURSIVE_REWRITE_IF(max(c1 - x, c2), c1 - min(x, c1 - c2), c2.Eval()->value != 0);
  }

  // condition rules.
  TVM_TRY_REWRITE(max(select(x, y, z), select(x, s1, s2)), select(x, max(y, s1), max(z, s2)));
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const EQNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<EQNode>();
  PrimExpr const_res = TryConstFold<EQNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y;
  // Pattern var match IntImm
  PVar<IntImm> c1;
  PVar<int> lanes;

  // vector rule
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) == broadcast(y, lanes), broadcast(x == y, lanes));
  }

  if (IsIndexType(op->a.dtype())) {
    CompareResult result = TryCompare(op->a - op->b, 0);
    if (result == kEQ) {
      return make_const(op->dtype, true);
    } else if (result == kNE || result == kGT || result == kLT) {
      return make_const(op->dtype, false);
    }
    TVM_TRY_REWRITE(x - c1 == 0, x == c1);
    TVM_TRY_REWRITE(c1 - x == 0, x == c1);
    TVM_TRY_REWRITE(x + c1 == 0, x == 0 - c1);
    TVM_TRY_REWRITE(x * y == 0, x == 0 || y == 0);
  }
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const NENode* op) {
  return this->VisitExpr(NotNode::make(op->a == op->b));
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const LENode* op) {
  return this->VisitExpr(NotNode::make(op->b < op->a));
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const GTNode* op) {
  return this->VisitExpr(op->b < op->a);
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const GENode* op) {
  return this->VisitExpr(NotNode::make(op->a < op->b));
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const LTNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<LTNode>();
  PrimExpr const_res = TryConstFold<LTNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y, z, s1, s2;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  PVar<int> lanes;

  // vector rule
  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) < broadcast(y, lanes), broadcast(x < y, lanes));
    TVM_TRY_REWRITE(ramp(x, s1, lanes) < ramp(y, s1, lanes), broadcast(x < y, lanes));
  }

  if (IsIndexType(op->a.dtype())) {
    CompareResult result = TryCompare(op->a - op->b, 0);
    if (result == kLT) {
      return make_const(op->dtype, true);
    }
    if (result == kEQ || result == kGT || result == kGE) {
      return make_const(op->dtype, false);
    }

    // clang-format off
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

    TVM_TRY_REWRITE_IF(x * c1 < y * c1, x < y, c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(x * c1 < y * c1, y < x, c1.Eval()->value < 0);

    // constant cancelation: only need to make use of one mod
    // truc div
    TVM_TRY_REWRITE_IF(x * c2 < c1,
                       x < truncdiv(c1 - 1, c2) + 1, c1.Eval()->value > 0 && c2.Eval()->value > 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(x * c2 < c1, x < truncdiv(c1, c2),
                       c1.Eval()->value <= 0 && c2.Eval()->value > 0);
    // NOTE: trunc div required (euclidean is ok too, floored is not)
    TVM_TRY_REWRITE_IF(x * c2 < c1, truncdiv(c1 - 1, c2) - 1 < x, c1.Eval()->value > 0 &&
                       c2.Eval()->value < 0);
    // NOTE: trunc div required (floored is ok too, euclidean is not)
    TVM_TRY_REWRITE_IF(x * c2 < c1, truncdiv(c1, c2) < x,
                       c1.Eval()->value <= 0 && c2.Eval()->value < 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(c1 < x * c2, truncdiv(c1 + 1, c2) - 1 < x,
                       c1.Eval()->value < 0 && c2.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(c1 < x * c2, truncdiv(c1, c2) < x,
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0);
    // NOTE: trunc div required (floored is ok too, euclidean is not)
    TVM_TRY_REWRITE_IF(c1 < x * c2, x < truncdiv(c1 + 1, c2) + 1,
                       c1.Eval()->value < 0 && c2.Eval()->value < 0);
    // NOTE: trunc div required (euclidean is ok too, floored is not)
    TVM_TRY_REWRITE_IF(c1 < x * c2, x < truncdiv(c1, c2),
                       c1.Eval()->value >= 0 && c2.Eval()->value < 0);
    // DivMod rules
    // trucdiv
    TVM_TRY_REWRITE_IF(truncdiv(x, c1) < c2,
                       x<c1 * c2, c1.Eval()->value> 0 && c2.Eval()->value > 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(truncdiv(x, c1) < c2,
                       x<c1*(c2 - 1) + 1, c1.Eval()->value> 0 && c2.Eval()->value <= 0);

    TVM_TRY_REWRITE_IF(c1 < truncdiv(x, c2), (c1 + 1) * c2 - 1 < x,
                       c1.Eval()->value >= 0 && c2.Eval()->value > 0);
    // NOTE: trunc div required
    TVM_TRY_REWRITE_IF(c1 < truncdiv(x, c2), c1 * c2 < x,
                       c1.Eval()->value < 0 && c2.Eval()->value > 0);

    // invariance for any div mod: x - (x / c1) * c1 == x % c1
    TVM_TRY_REWRITE_IF(truncdiv(x, c1) * c1 < x, 0 < truncmod(x, c1), c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(truncdiv(x, c1) * c1 < x + y,
                       0 < truncmod(x, c1) + y, c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(truncdiv(x, c1) * c1 < x - y,
                       y < truncmod(x, c1), c1.Eval()->value > 0);

    TVM_TRY_REWRITE_IF(truncdiv(x + c2, c1) * c1 < x,
                       c2 < truncmod(x + c2, c1), c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(truncdiv(x + c2, c1) * c1 < x + y,
                       c2 < truncmod(x + c2, c1) + y, c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(truncdiv(x + c2, c1) * c1 < x - y,
                       y < truncmod(x + c2, c1) + (0 - c2), c1.Eval()->value > 0);

    // floordiv
    TVM_TRY_REWRITE_IF(floordiv(x, c1) < c2, x < c1 * c2, c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(c1 < floordiv(x, c2), (c1 + 1) * c2 - 1 < x, c2.Eval()->value > 0);

    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x, 0 < floormod(x, c1), c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x + y,
                       0 < floormod(x, c1) + y, c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x, c1) * c1 < x - y,
                       y < floormod(x, c1), c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c2, c1) * c1 < x,
                       c2 < floormod(x + c2, c1), c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c2, c1) * c1 < x + y,
                       c2 < floormod(x + c2, c1) + y, c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF(floordiv(x + c2, c1) * c1 < x - y,
                       y < floormod(x + c2, c1) + (0 - c2), c1.Eval()->value > 0);

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
    // clang-format on
  }
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const NotNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<NotNode>();
  PrimExpr const_res = TryConstFold<NotNode>(op->a);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<PrimExpr> x, y;
  PVar<int> lanes;
  if (op->dtype.lanes() != 1) {
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

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const AndNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<AndNode>();
  PrimExpr const_res = TryConstFold<AndNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  PVar<int> lanes;

  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) && broadcast(y, lanes), broadcast(x && y, lanes));
  }

  auto cfalse = PConst<PrimExpr>(make_const(op->dtype, false));
  TVM_TRY_REWRITE(x == y && x != y, cfalse);
  TVM_TRY_REWRITE(x != y && x == y, cfalse);
  TVM_TRY_REWRITE(x && !x, cfalse);
  TVM_TRY_REWRITE(x <= y && y < x, cfalse);
  TVM_TRY_REWRITE(y < x && x <= y, cfalse);

  TVM_TRY_REWRITE_IF(x < c1 && c2 < x, cfalse, c2.Eval()->value + 1 >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x && x < c1, cfalse, c2.Eval()->value + 1 >= c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x < c1 && c2 <= x, cfalse, c2.Eval()->value >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 <= x && x < c1, cfalse, c2.Eval()->value >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(x <= c1 && c2 < x, cfalse, c2.Eval()->value >= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x && x <= c1, cfalse, c2.Eval()->value >= c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x <= c1 && c2 <= x, cfalse, c2.Eval()->value > c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 <= x && x <= c1, cfalse, c2.Eval()->value > c1.Eval()->value);

  TVM_TRY_REWRITE(x == c1 && x != c2, x == c1 && c1 != c2);
  TVM_TRY_REWRITE(x != c2 && x == c1, x == c1 && c1 != c2);
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const OrNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<OrNode>();
  PrimExpr const_res = TryConstFold<OrNode>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<PrimExpr> x, y;
  // Pattern var match IntImm
  PVar<IntImm> c1, c2;
  PVar<int> lanes;

  if (op->dtype.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) || broadcast(y, lanes), broadcast(x || y, lanes));
  }

  auto ctrue = PConst<PrimExpr>(make_const(op->dtype, true));

  TVM_TRY_REWRITE(x == y || x != y, ctrue);
  TVM_TRY_REWRITE(x != y || x == y, ctrue);
  TVM_TRY_REWRITE(x || !x, ctrue);
  TVM_TRY_REWRITE(x <= y || y < x, ctrue);
  TVM_TRY_REWRITE(y < x || x <= y, ctrue);

  TVM_TRY_REWRITE_IF(x < c1 || c2 < x, ctrue, c2.Eval()->value < c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x || x < c1, ctrue, c2.Eval()->value < c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x <= c1 || c2 < x, ctrue, c2.Eval()->value <= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 < x || x <= c1, ctrue, c2.Eval()->value <= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(x < c1 || c2 <= x, ctrue, c2.Eval()->value <= c1.Eval()->value);
  TVM_TRY_REWRITE_IF(c2 <= x || x < c1, ctrue, c2.Eval()->value <= c1.Eval()->value);

  TVM_TRY_REWRITE_IF(x <= c1 || c2 <= x, ctrue, c2.Eval()->value <= c1.Eval()->value + 1);
  TVM_TRY_REWRITE_IF(c2 <= x || x <= c1, ctrue, c2.Eval()->value <= c1.Eval()->value + 1);

  TVM_TRY_REWRITE(x != c1 || x == c2, x != c1 || c1 == c2);
  TVM_TRY_REWRITE(x == c2 || x != c1, x != c1 || c1 == c2);
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const SelectNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<SelectNode>();
  if (op == nullptr) return ret;
  // Pattern var to match any expression
  PVar<PrimExpr> x, y;
  TVM_TRY_REWRITE(select(x, y, y), y);
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const CallNode* op) {
  // add condition context to if_then_else
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<CallNode>();
  if (op == nullptr) return ret;
  if (op->is_intrinsic(CallNode::likely) && is_const(op->args[0])) {
    return op->args[0];
  } else if (op->is_intrinsic(CallNode::shift_right)) {
    if (op->args[0].as<IntImmNode>() && op->args[1].as<IntImmNode>()) {
      // the operator overload will eagerly constant fold.
      return op->args[0] >> op->args[1];
    }
  } else if (op->is_intrinsic(CallNode::bitwise_and)) {
    if (op->args[0].as<IntImmNode>() && op->args[1].as<IntImmNode>()) {
      // the operator overload will eagerly constant fold.
      return op->args[0] & op->args[1];
    }
  }
  ExprDeepEqual expr_equal;
  if (op->is_intrinsic(CallNode::likely)) {
    for (const auto& constraint : literal_constraints_) {
      // Cases such as for (i, 0, bound) {if (likely(iter_var < bound)) { .. } }
      if (expr_equal(constraint, op->args[0])) {
        return make_const(op->dtype, true);
      }
    }
  }
  return ret;
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const VarNode* op) {
  Var var = GetRef<Var>(op);
  auto it = var_map_.find(var);
  if (it != var_map_.end()) {
    return it->second;
  }
  return GetRef<PrimExpr>(op);
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const CastNode* op) {
  PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
  op = ret.as<CastNode>();
  return cast(op->dtype, op->value);
}

PrimExpr RewriteSimplifier::Impl::VisitExpr_(const LetNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (!tir::HasSideEffect(value)) {
    // it is fine to discard the let binding
    // because the value will always be inlined in the simplifier.
    analyzer_->Bind(op->var, value);
    return this->VisitExpr(op->body);
  }
  PrimExpr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<PrimExpr>(op);
  } else {
    return LetNode::make(op->var, value, body);
  }
}

PrimExpr RewriteSimplifier::operator()(const PrimExpr& expr) {
  // Run simplification in post order
  PrimExpr res = expr;
  int max_iter = 2;
  for (int i = 0; i < max_iter; ++i) {
    PrimExpr new_expr = impl_->operator()(res);
    if (new_expr.same_as(res)) return res;
    res = new_expr;
  }
  return res;
}

void RewriteSimplifier::Update(const Var& var, const PrimExpr& info, bool override) {
  impl_->Update(var, info, override);
}

std::function<void()> RewriteSimplifier::EnterConstraint(const PrimExpr& constraint) {
  return impl_->EnterConstraint(constraint);
}

RewriteSimplifier::RewriteSimplifier(Analyzer* parent) : impl_(new Impl(parent)) {}

RewriteSimplifier::~RewriteSimplifier() { delete impl_; }

}  // namespace arith
}  // namespace tvm
