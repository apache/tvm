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
 * \file detect_common_subexpr.cc
 * \brief Utility to detect common sub expressions.
 */
#include <tvm/tir/expr.h>

#include <limits>

#include "../tir/transforms/common_subexpr_elim_tools.h"

namespace tvm {
namespace arith {

using namespace tir;

Map<PrimExpr, Integer> DetectCommonSubExpr(const PrimExpr& e, int thresh) {
  // Check the threshold in the range of size_t
  CHECK_GE(thresh, std::numeric_limits<size_t>::min());
  CHECK_LE(thresh, std::numeric_limits<size_t>::max());
  size_t repeat_thr = static_cast<size_t>(thresh);
  auto IsEligibleComputation = [](const PrimExpr& expr) {
    return (SideEffect(expr) <= CallEffectKind::kPure && CalculateExprComplexity(expr) > 1 &&
            (expr.as<RampNode>() == nullptr) && (expr.as<BroadcastNode>() == nullptr));
  };

  // Analyze the sub expressions
  ComputationTable table_syntactic_comp_done_by_expr = ComputationsDoneBy::GetComputationsDoneBy(
      e, IsEligibleComputation, [](const PrimExpr& expr) { return true; });

  std::vector<std::pair<PrimExpr, size_t>> semantic_comp_done_by_expr =
      SyntacticToSemanticComputations(table_syntactic_comp_done_by_expr, true);

  // Find eligible sub expr if occurrence is under thresh
  for (size_t i = 0; i < semantic_comp_done_by_expr.size(); i++) {
    std::pair<PrimExpr, size_t>& computation_and_nb = semantic_comp_done_by_expr[i];
    if (computation_and_nb.second < repeat_thr) {
      std::vector<PrimExpr> direct_subexprs =
          DirectSubexpr::GetDirectSubexpressions(computation_and_nb.first, IsEligibleComputation,
                                                 [](const PrimExpr& expr) { return true; });
      InsertVectorToSortedSemanticComputations(&semantic_comp_done_by_expr, direct_subexprs, true,
                                               computation_and_nb.second);
    }
  }

  // Return the common sub expr that occur more than thresh times
  Map<PrimExpr, Integer> results;
  for (auto& it : semantic_comp_done_by_expr) {
    if (it.second >= repeat_thr) results.Set(it.first, it.second);
  }
  return results;
}

TVM_REGISTER_GLOBAL("arith.DetectCommonSubExpr").set_body_typed(DetectCommonSubExpr);
}  // namespace arith
}  // namespace tvm
