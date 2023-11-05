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
 * \file tvm/arith/constraint_extract.cc
 */

#include "constraint_extract.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include "pattern_match.h"

namespace tvm {
namespace arith {

// The BufferLoad node cannot be considered as a valid constraint to be provided to analyzer.
// Because the result of BufferLoad may be modified in the later program.
// Here we will find if the expr contains BufferLoad node. if so, it is a unclear constraint
bool IsUnclearConstraint(const PrimExpr& expr) {
  class UnclearExprFinder : private tir::ExprVisitor {
   public:
    void VisitExpr(const PrimExpr& expr) final {
      if (unclear) return;
      ExprVisitor::VisitExpr(expr);
    }
    void VisitExpr_(const tir::BufferLoadNode* op) final { unclear = true; }
    bool unclear{false};
  };
  UnclearExprFinder finder;
  finder.VisitExpr(expr);
  return finder.unclear;
}

template <typename F>
void CollectConstraints(PrimExpr expr, F callback, bool keep_composite_constraints) {
  if (IsUnclearConstraint(expr)) {
    return;
  }
  if (keep_composite_constraints) {
    callback(expr);
  }

  PVar<PrimExpr> x, y;
  if ((x && y).Match(expr)) {
    CollectConstraints(x.Eval(), callback, keep_composite_constraints);
    CollectConstraints(y.Eval(), callback, keep_composite_constraints);
  } else if (!keep_composite_constraints) {
    callback(expr);
  }
}

std::vector<PrimExpr> ExtractConstraints(const PrimExpr& expr, bool keep_composite_constraints) {
  std::vector<PrimExpr> out;
  CollectConstraints(
      expr, [&](const PrimExpr& part) { out.push_back(part); }, keep_composite_constraints);
  return out;
}

template <typename F>
void CollectComponents(PrimExpr expr, F callback) {
  PVar<PrimExpr> x, y;
  if ((x || y).Match(expr)) {
    CollectComponents(x.Eval(), callback);
    CollectComponents(y.Eval(), callback);
  } else {
    callback(expr);
  }
}

std::vector<PrimExpr> ExtractComponents(const PrimExpr& expr) {
  std::vector<PrimExpr> out;
  CollectComponents(expr, [&](const PrimExpr& part) { out.push_back(part); });
  return out;
}

}  // namespace arith
}  // namespace tvm
