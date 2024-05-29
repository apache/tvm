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
 * \file tvm/arith/scalable_expression.cc
 * \brief Analyze scalable expressions.
 */

#include "scalable_expression.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <vector>

#include "../tir/analysis/check_contains.h"
#include "../tir/transforms/replace_selected_expr.h"
#include "./pattern_match.h"

namespace tvm {
namespace arith {

bool IsVScaleCall(const PrimExpr& expr) {
  if (auto call = expr.as<tir::CallNode>()) {
    return call->op.same_as(tir::builtin::vscale());
  }
  return false;
}

bool ContainsVscaleCall(const PrimExpr& expr) {
  return tir::CheckContains::ExprContains(expr, IsVScaleCall);
}

PrimExpr SubstituteVScaleWithKnownValue(const PrimExpr& expr, unsigned int vscale_value) {
  std::function<bool(const PrimExpr&)> predicate_selector = [](const PrimExpr& current_expr) {
    return IsVScaleCall(current_expr);
  };
  std::function<bool(const PrimExpr&)> can_replace_inside = [](const PrimExpr& current_expr) {
    return true;
  };

  return tir::ReplaceSelectedExpr::ReplaceSelectedExprInExpr(
      expr, predicate_selector, tir::MakeConstScalar(DataType::Int(32), vscale_value),
      can_replace_inside);
}

std::optional<int> ExtractVscaleFactor(const PrimExpr& lanes) {
  PVar<IntImm> multiplier;
  PCallExpr<PVscaleOp> vscale;

  if (PMatchesOneOf(multiplier * vscale, vscale * multiplier).Match(lanes)) {
    return multiplier.Eval()->value;
  } else {
    return std::nullopt;
  }
}

bool CanProveVscaleExpressionFromKnownValues(arith::Analyzer* analyzer, const PrimExpr& expr,
                                             const std::vector<unsigned int>& vscale_values) {
  bool can_prove_expr = true;
  for (const unsigned int vscale_value : vscale_values) {
    PrimExpr result = SubstituteVScaleWithKnownValue(expr, vscale_value);
    result = analyzer->Simplify(result);
    const int64_t* as_int = tir::as_const_int(result);
    if (!as_int || *as_int == 0) {
      can_prove_expr = false;
      break;
    }
  }
  return can_prove_expr;
}

bool TargetHasSVE(Target current_target) {
  bool has_sve{false};
  if (current_target.defined()) {
    has_sve = current_target->GetFeature<Bool>("has_sve").value_or(Bool(false));
  }
  return has_sve;
}

}  // namespace arith
}  // namespace tvm
