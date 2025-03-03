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
 * \file global_var_supply.cc
 * \brief GlobalVarSupply that can be used to generate unique GlobalVars.
 */
#include "tvm/ir/global_var_supply.h"

#include <tvm/runtime/registry.h>

#include <utility>

#include "tvm/ir/expr.h"

namespace tvm {
GlobalVarSupply::GlobalVarSupply(const NameSupply& name_supply,
                                 std::unordered_map<std::string, GlobalVar> name_to_var_map) {
  auto n = make_object<GlobalVarSupplyNode>(name_supply, name_to_var_map);
  data_ = std::move(n);
}

std::string GetModuleName(const IRModule& module) {
  return module->GetAttr<String>(tvm::attr::kModuleName).value_or("tvmgen_default");
}

GlobalVarSupply::GlobalVarSupply(const Array<IRModule>& modules) : GlobalVarSupply() {
  if (!modules.empty()) {
    IRModule first_mod = modules.front();
    this->operator->()->name_supply_->prefix_ = GetModuleName(first_mod);
  }
  for (auto& mod : modules) {
    for (auto kv : mod->functions) {
      this->operator->()->ReserveGlobalVar(kv.first);
    }
  }
}

GlobalVarSupply::GlobalVarSupply(const IRModule module)
    : GlobalVarSupply(Array<IRModule>{module}) {}

void GlobalVarSupplyNode::ReserveGlobalVar(const GlobalVar& var, bool allow_conflict) {
  name_supply_->ReserveName(var->name_hint, false);
  if (!allow_conflict) {
    ICHECK(name_to_var_map_.count(var->name_hint) == 0)
        << "GlobalVar " << var << " conflicts by name in this supply.";
  }
  name_to_var_map_[var->name_hint] = var;
}

GlobalVarSupplyNode::GlobalVarSupplyNode(NameSupply name_supply,
                                         std::unordered_map<std::string, GlobalVar> name_to_var_map)
    : name_supply_(std::move(name_supply)), name_to_var_map_(std::move(name_to_var_map)) {}

GlobalVar GlobalVarSupplyNode::UniqueGlobalFor(const String& name, bool add_prefix) {
  String final_name = name_supply_->ReserveName(name, add_prefix);

  auto it = name_to_var_map_.find(final_name);
  if (it != name_to_var_map_.end()) {
    return it->second;
  } else {
    GlobalVar var = GlobalVar(final_name);
    name_to_var_map_.emplace(final_name, var);
    return var;
  }
}

GlobalVar GlobalVarSupplyNode::FreshGlobal(String name, bool add_prefix) {
  String final_name = name_supply_->FreshName(name, add_prefix);
  ICHECK(name_to_var_map_.find(final_name) == name_to_var_map_.end())
      << "GlobalVar already exists for name " << final_name;
  GlobalVar var = GlobalVar(final_name);
  name_to_var_map_.emplace(final_name, var);
  return var;
}

TVM_REGISTER_NODE_TYPE(GlobalVarSupplyNode);

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_NameSupply")
    .set_body_typed([](const NameSupply& name_supply) { return GlobalVarSupply(name_supply); });

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_IRModule").set_body_typed([](IRModule mod) {
  return GlobalVarSupply(std::move(mod));
});

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_IRModules").set_body_typed([](const Array<IRModule>& mods) {
  return GlobalVarSupply(mods);
});

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_FreshGlobal")
    .set_body_method<GlobalVarSupply>(&GlobalVarSupplyNode::FreshGlobal);

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_UniqueGlobalFor")
    .set_body_method<GlobalVarSupply>(&GlobalVarSupplyNode::UniqueGlobalFor);

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_ReserveGlobalVar")
    .set_body_method<GlobalVarSupply>(&GlobalVarSupplyNode::ReserveGlobalVar);

}  // namespace tvm
