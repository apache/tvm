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

PrimExpr CanonicalizeScalableLanes(const PrimExpr& lanes) {
  PVar<IntImm> multiplier;
  PCallExpr<PVscaleOp> vscale;

  PrimExpr new_lanes;

  if ((multiplier * vscale).Match(lanes)) {
    new_lanes = lanes;
  } else if ((vscale * multiplier).Match(lanes)) {
    new_lanes =
        tir::Mul(multiplier.Eval(), tir::Call(DataType::Int(32), tir::builtin::vscale(), {}));
  } else {
    LOG(FATAL) << "Illegal form for scalable vector lanes: " << lanes;
  }

  return new_lanes;
}

}  // namespace arith
}  // namespace tvm