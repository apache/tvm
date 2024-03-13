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
 * \file src/relax/analysis/udchain.cc
 * \brief Implementation of use-def analysis.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../support/ordered_set.h"

namespace tvm {
namespace relax {

class UDChain : relax::ExprVisitor {
 public:
  static VarUsageInfo Collect(const Expr& expr) {
    UDChain visitor;
    visitor.VisitExpr(expr);

    Array<Var> output(visitor.outputs.begin(), visitor.outputs.end());

    Map<Var, Array<Var>> use_def;
    for (const auto& [var, usage] : visitor.usage_map) {
      use_def.Set(var, Array<Var>(usage.begin(), usage.end()));
    }

    return VarUsageInfo{visitor.bound_values, use_def, output};
  }

 private:
  Map<Var, Expr> bound_values;
  std::unordered_map<Var, support::OrderedSet<Var>, ObjectPtrHash, ObjectPtrEqual> usage_map;
  support::OrderedSet<Var> outputs;

  Optional<Var> cur_user_{nullptr};

  void VisitBinding_(const VarBindingNode* binding) override {
    CHECK(!bound_values.count(binding->var))
        << "Variable " << binding->var << " was defined multiple times";
    bound_values.Set(binding->var, binding->value);

    auto cache = cur_user_;
    cur_user_ = binding->var;
    ExprVisitor::VisitBinding_(binding);
    cur_user_ = cache;
  }

  void VisitVarDef(const Var& var) override {
    CHECK(!usage_map.count(var)) << "Variable " << var << " was used before its definition";
    usage_map[var] = {};
  }
  void VisitExpr_(const VarNode* op) override {
    auto var = GetRef<Var>(op);

    if (cur_user_) {
      usage_map[var].insert(cur_user_.value());
    } else {
      outputs.insert(var);
    }
  }

  void VisitExpr_(const FunctionNode* op) override {
    cur_user_ = nullptr;
    ExprVisitor::VisitExpr_(op);
  }
};

std::pair<runtime::Map<Var, runtime::Array<Var>>, runtime::Array<Var>> FunctionUseDef(
    const Expr& fn) {
  auto usage = UDChain::Collect(fn);
  return {usage.downstream_usage, usage.outputs};
}

runtime::Map<Var, Array<Var>> DataflowBlockUseDef(const DataflowBlock& dfb) {
  auto usage = UDChain::Collect(SeqExpr({dfb}, Tuple(Array<Expr>())));
  return usage.downstream_usage;
}

TVM_REGISTER_GLOBAL("relax.analysis.udchain").set_body_typed(DataflowBlockUseDef);

VarUsageInfo CollectVarUsage(const Expr& expr) { return UDChain::Collect(expr); }

}  // namespace relax
}  // namespace tvm
