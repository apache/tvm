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

namespace tvm {
namespace relax {

class UDChain : public relax::ExprVisitor {
 public:
  // nullptr users means it is the output of the function.
  std::map<const VarNode*, std::set<const VarNode*>> to_users;

  const VarNode* cur_user_;

  void VisitBinding_(const VarBindingNode* binding) override {
    // init
    cur_user_ = binding->var.get();
    this->VisitVarDef(binding->var);
    this->VisitExpr(binding->value);
    cur_user_ = nullptr;
  }

  void VisitExpr_(const VarNode* op) override { to_users[op].insert(cur_user_); }
  void VisitVarDef(const Var& var) override { to_users[var.get()] = {}; }
  void VisitExpr_(const FunctionNode* op) override { ExprVisitor::VisitExpr_(op); }

  void VisitExpr_(const DataflowVarNode* op) override {
    VisitExpr_(static_cast<const VarNode*>(op));
  }
};

std::pair<runtime::Map<Var, runtime::Array<Var>>, runtime::Array<Var>> FunctionUseDef(
    const Function& fn) {
  UDChain udchain;
  udchain.VisitExpr_(fn.get());

  Map<Var, Array<Var>> user_map;
  Array<Var> fn_outs;

  for (const auto& kv : udchain.to_users) {
    Array<Var> uses{};
    uses.reserve(kv.second.size());
    for (const auto& v : kv.second) {
      if (nullptr == v &&
          fn_outs.end() == std::find(fn_outs.begin(), fn_outs.end(), GetRef<Var>(kv.first))) {
        fn_outs.push_back(GetRef<Var>(kv.first));
      } else {
        uses.push_back(GetRef<Var>(v));
      }
    }
    user_map.Set(GetRef<Var>(kv.first), std::move(uses));
  }
  return std::make_pair(std::move(user_map), std::move(fn_outs));
}

runtime::Map<Var, Array<Var>> DataflowBlockUseDef(const DataflowBlock& dfb) {
  UDChain udchain;
  udchain.VisitBindingBlock_(dfb.get());
  runtime::Map<Var, Array<Var>> ret;
  for (const auto& kv : udchain.to_users) {
    Array<Var> uses{};
    uses.reserve(kv.second.size());
    for (const auto& v : kv.second) uses.push_back(GetRef<Var>(v));
    ret.Set(GetRef<Var>(kv.first), std::move(uses));
  }
  return ret;
}

TVM_REGISTER_GLOBAL("relax.analysis.udchain").set_body_typed(DataflowBlockUseDef);

}  // namespace relax
}  // namespace tvm
