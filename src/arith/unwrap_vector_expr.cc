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
 * \file unwrap_vector_expr.cc
 * \brief Utility for tracking currently active constraints
 */

#include "unwrap_vector_expr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>

#include <unordered_map>

namespace tvm {
namespace arith {

using namespace tir;

class Scalarizer : public ExprMutator {
 public:
  explicit Scalarizer(PrimExpr lane) : lane_(lane) {}

  PrimExpr VisitExpr_(const RampNode* op) final { return op->base + lane_ * op->stride; }

  PrimExpr VisitExpr_(const BroadcastNode* op) final { return op->value; }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    auto it = let_var_remap_.find(op);
    if (it != let_var_remap_.end()) {
      return it->second;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }
  PrimExpr VisitExpr_(const LetNode* op) final {
    if (op->value.dtype().lanes() == 1) {
      return ExprMutator::VisitExpr_(op);
    }

    auto it = let_var_remap_.find(op->var.get());
    ICHECK(it == let_var_remap_.end()) << "Duplicate binding of variable " << op->var;

    Var new_var(op->var->name_hint + "_scalar", op->var.dtype().element_of());
    let_var_remap_[op->var.get()] = new_var;

    PrimExpr value = this->VisitExpr(op->value);
    PrimExpr body = this->VisitExpr(op->body);

    let_var_remap_.erase(op->var.get());
    return Let(op->var, value, body);
  }

 private:
  // The lane to extract
  PrimExpr lane_;

  // Let binding
  std::unordered_map<const VarNode*, Var> let_var_remap_;
};

PrimExpr UnwrapVectorExpr(const PrimExpr& vector_expr, const PrimExpr& lane) {
  return Scalarizer(lane)(vector_expr);
}

}  // namespace arith
}  // namespace tvm
