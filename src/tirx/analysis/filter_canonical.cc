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
 * \file filter_canonical.cc
 * \brief Implementation of the canonical-form classifier for thread-filter
 * predicates. See filter_canonical.h for the grammar and semantics.
 */

#include "filter_canonical.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tirx {

namespace {

// Recognized conjunction shapes: logical-And and bitwise-And calls.
// Mirrors FlattenConjuncts in tile_primitive_dispatch.cc so the classifier
// accepts the same set of "fully conjunctive" predicates that the existing
// pass-internal helpers do.
bool IsBitwiseAndCall(const CallNode* call) {
  return call->op.same_as(tirx::builtin::bitwise_and()) && call->args.size() == 2;
}

bool IsPtxElectSyncCall(const CallNode* call) {
  static const Op& ptx_elect_sync_op = Op::Get("tirx.ptx_elect_sync");
  if (call->op.same_as(ptx_elect_sync_op)) return true;
  if (auto op = call->op.as<Op>()) {
    return op.value()->name == "tirx.ptx.elect_sync";
  }
  return false;
}

// Strip implicit Cast wrappers from a predicate. Bool-vs-int mixing in the
// Python frontend can insert ``Cast<uint32>(bool_expr)`` (e.g. when an
// ``elect_sync()`` uint32 result is combined with a bool comparison via
// bitwise-AND). These casts are semantic no-ops for thread-filter purposes:
// the inner expression is what we need to classify.
PrimExpr StripCast(const PrimExpr& expr) {
  PrimExpr cur = expr;
  while (const auto* cast = cur.as<CastNode>()) {
    cur = cast->value;
  }
  return cur;
}

void FlattenConjuncts(const PrimExpr& pred, std::vector<PrimExpr>* out) {
  PrimExpr stripped = StripCast(pred);
  if (const auto* and_node = stripped.as<AndNode>()) {
    FlattenConjuncts(and_node->a, out);
    FlattenConjuncts(and_node->b, out);
    return;
  }
  if (const auto* call = stripped.as<CallNode>()) {
    if (IsBitwiseAndCall(call)) {
      FlattenConjuncts(call->args[0], out);
      FlattenConjuncts(call->args[1], out);
      return;
    }
  }
  out->push_back(stripped);
}

// Encoding of a comparison operator after normalization to `var <op> const`.
enum class CmpOp { kEq, kLT, kLE, kGT, kGE };

// Reflect an operator when the operands are swapped (i.e. user wrote
// `const <op> var` and we want to emit it as `var <op_reflected> const`).
CmpOp Reflect(CmpOp op) {
  switch (op) {
    case CmpOp::kEq:
      return CmpOp::kEq;
    case CmpOp::kLT:
      return CmpOp::kGT;
    case CmpOp::kLE:
      return CmpOp::kGE;
    case CmpOp::kGT:
      return CmpOp::kLT;
    case CmpOp::kGE:
      return CmpOp::kLE;
  }
  return CmpOp::kEq;  // unreachable; silences -Wreturn-type
}

// Compute the half-open range [lo, hi) for `var <op> c`.
// Uses arith::ConstIntBound sentinels for unbounded sides.
void OpToRange(CmpOp op, int64_t c, int64_t* lo, int64_t* hi) {
  switch (op) {
    case CmpOp::kEq:
      *lo = c;
      *hi = c + 1;
      return;
    case CmpOp::kLT:
      *lo = arith::ConstIntBound::kNegInf;
      *hi = c;
      return;
    case CmpOp::kLE:
      *lo = arith::ConstIntBound::kNegInf;
      *hi = c + 1;
      return;
    case CmpOp::kGT:
      *lo = c + 1;
      *hi = arith::ConstIntBound::kPosInf;
      return;
    case CmpOp::kGE:
      *lo = c;
      *hi = arith::ConstIntBound::kPosInf;
      return;
  }
}

// Try to read `expr` as a single comparison atom of the form
// `scopeid_var <op> const` (or its mirrored `const <op> scopeid_var`).
// On success populates `*out` with `kRange` semantics.
//
// Returns false if `expr` is not a comparison, has shape `var op var`,
// `const op const`, or the var fails the `is_scope_id` predicate.
bool TryParseCompareAtom(const PrimExpr& expr, const ScopeIdPredicate& is_scope_id,
                         FilterAtom* out) {
  // Decode op + (lhs, rhs). The five comparison node types map to CmpOp.
  CmpOp op;
  PrimExpr lhs, rhs;
  if (const auto* eq = expr.as<EQNode>()) {
    op = CmpOp::kEq;
    lhs = eq->a;
    rhs = eq->b;
  } else if (const auto* lt = expr.as<LTNode>()) {
    op = CmpOp::kLT;
    lhs = lt->a;
    rhs = lt->b;
  } else if (const auto* le = expr.as<LENode>()) {
    op = CmpOp::kLE;
    lhs = le->a;
    rhs = le->b;
  } else if (const auto* gt = expr.as<GTNode>()) {
    op = CmpOp::kGT;
    lhs = gt->a;
    rhs = gt->b;
  } else if (const auto* ge = expr.as<GENode>()) {
    op = CmpOp::kGE;
    lhs = ge->a;
    rhs = ge->b;
  } else {
    return false;
  }

  // Identify the var side and the const side. Reject if both sides are vars
  // or both are constants -- neither shape is in the canonical grammar.
  const VarNode* var_node = lhs.as<VarNode>();
  const IntImmNode* imm_node = rhs.as<IntImmNode>();
  bool mirrored = false;
  if (var_node == nullptr || imm_node == nullptr) {
    var_node = rhs.as<VarNode>();
    imm_node = lhs.as<IntImmNode>();
    mirrored = true;
  }
  if (var_node == nullptr || imm_node == nullptr) return false;

  Var var = ffi::GetRef<Var>(var_node);
  if (!is_scope_id(var)) return false;

  CmpOp normalized = mirrored ? Reflect(op) : op;
  int64_t lo = 0;
  int64_t hi = 0;
  OpToRange(normalized, imm_node->value, &lo, &hi);

  out->kind = FilterAtomKind::kRange;
  out->scopeid_var = var;
  out->lo = lo;
  out->hi = hi;
  out->elect_sync_call = PrimExpr();
  return true;
}

// Try to read `expr` as a direct `Call("tirx.ptx_elect_sync")` atom.
// Composed forms like `elect_sync() != 0` or `not elect_sync()` are NOT
// accepted -- the canonical grammar requires a bare elect_sync call.
bool TryParseElectSyncAtom(const PrimExpr& expr, FilterAtom* out) {
  const auto* call = expr.as<CallNode>();
  if (call == nullptr) return false;
  if (!IsPtxElectSyncCall(call)) return false;
  out->kind = FilterAtomKind::kElectSync;
  out->scopeid_var = Var();
  out->lo = 0;
  out->hi = 0;
  out->elect_sync_call = expr;
  return true;
}

}  // namespace

std::optional<CanonicalForm> TryClassifyCanonical(const PrimExpr& cond,
                                                  const ScopeIdPredicate& is_scope_id) {
  std::vector<PrimExpr> terms;
  FlattenConjuncts(cond, &terms);

  CanonicalForm result;
  result.atoms.reserve(terms.size());
  for (const PrimExpr& term : terms) {
    FilterAtom atom;
    if (TryParseElectSyncAtom(term, &atom) || TryParseCompareAtom(term, is_scope_id, &atom)) {
      result.atoms.push_back(std::move(atom));
      continue;
    }
    // Term does not match any atom shape. The whole predicate is rejected.
    return std::nullopt;
  }
  if (result.atoms.empty()) return std::nullopt;
  return result;
}

}  // namespace tirx
}  // namespace tvm
