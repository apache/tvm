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

#include "tvm/ir/global_var_supply.h"

#include <tvm/runtime/registry.h>

#include <utility>

#include "tvm/ir/expr.h"

namespace tvm {

GlobalVarSupply::GlobalVarSupply(const NameSupply& name_supply,
                                 std::unordered_map<std::string, GlobalVar> name_to_var_map) {
  auto n = make_object<GlobalVarSupplyNode>(name_supply);
  n->name_to_var_map_ = std::move(name_to_var_map);
  data_ = std::move(n);
}

GlobalVarSupply GlobalVarSupply::GlobalVarSupplyFromNameSupply(const NameSupply& name_supply) {
  auto global_var_supply = GlobalVarSupply(name_supply);
  return global_var_supply;
}

GlobalVarSupply GlobalVarSupply::EmptySupply() {
  return GlobalVarSupplyFromNameSupply(NameSupply::NameSupplyWithPrefix(""));
}

GlobalVarSupplyNode::GlobalVarSupplyNode(NameSupply name_supply)
    : name_supply_(std::move(name_supply)) {}

GlobalVar GlobalVarSupplyNode::UniqueGlobalFor(const String& name, bool add_prefix) {
  String final_name = name_supply_->ReserveName(name, add_prefix);

  auto it = name_to_var_map_.find(final_name);
  if (it != name_to_var_map_.end()) {
    return it->second;
  } else {
    GlobalVar var = GlobalVar(final_name);
    name_to_var_map_[final_name] = var;
    return var;
  }
}

GlobalVar GlobalVarSupplyNode::FreshGlobal(String name, bool add_prefix) {
  String final_name = name_supply_->FreshName(name, add_prefix);
  ICHECK(name_to_var_map_.find(final_name) == name_to_var_map_.end())
      << "GlobalVar already exists for name " << final_name;
  GlobalVar var = GlobalVar(final_name);
  name_to_var_map_[final_name] = var;
  return var;
}

TVM_REGISTER_NODE_TYPE(GlobalVarSupplyNode);

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply").set_body_typed([](NameSupply name_supply) {
  return GlobalVarSupply(name_supply);
});

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_FreshGlobal")
    .set_body_method<GlobalVarSupply>(&GlobalVarSupplyNode::FreshGlobal);

TVM_REGISTER_GLOBAL("ir.GlobalVarSupply_UniqueGlobalFor")
    .set_body_method<GlobalVarSupply>(&GlobalVarSupplyNode::UniqueGlobalFor);

}  // namespace tvm
