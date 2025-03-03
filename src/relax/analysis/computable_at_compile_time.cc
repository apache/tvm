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
 * \file computable_at_compile_time.cc
 *
 * \brief Utilities for identifying potentially compile-time variables
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

#include "../../support/ordered_set.h"

namespace tvm {
namespace relax {

namespace {
class CompileTimeCollector : ExprVisitor {
 public:
  static Array<Var> Collect(const Function& func) {
    CompileTimeCollector visitor;
    visitor(func);
    return Array<Var>(visitor.known_relax_vars_.begin(), visitor.known_relax_vars_.end());
  }

 private:
  void VisitExpr_(const FunctionNode* func) override {
    if (auto opt_num_input = func->attrs.GetAttr<Integer>(attr::kNumInput)) {
      size_t num_input = opt_num_input.value()->value;
      for (size_t i = num_input; i < func->params.size(); i++) {
        MarkAsKnown(func->params[i]);
      }
    }

    ExprVisitor::VisitExpr_(func);
  }

  void VisitBinding(const Binding& binding) override {
    Expr value = GetBoundValue(binding);
    bool can_compute_at_compile_time = [&]() {
      for (const auto& relax_var : FreeVars(value)) {
        if (!known_relax_vars_.count(relax_var)) {
          return false;
        }
      }
      for (const auto& tir_var : FreeSymbolicVars(value)) {
        if (!known_tir_vars_.count(tir_var)) {
          return false;
        }
      }

      return true;
    }();

    if (can_compute_at_compile_time) {
      MarkAsKnown(binding->var);
    }

    ExprVisitor::VisitBinding(binding);
  }

  void MarkAsKnown(const Var& var) {
    known_relax_vars_.insert(var);
    for (const auto& tir_var : DefinableTIRVarsInStructInfo(GetStructInfo(var))) {
      known_tir_vars_.insert(tir_var);
    }
  }

  support::OrderedSet<Var> known_relax_vars_;
  std::unordered_set<tir::Var> known_tir_vars_;
};
}  // namespace

Array<Var> ComputableAtCompileTime(const Function& func) {
  return CompileTimeCollector::Collect(func);
}

TVM_REGISTER_GLOBAL("relax.analysis.computable_at_compile_time")
    .set_body_typed(ComputableAtCompileTime);

}  // namespace relax
}  // namespace tvm
