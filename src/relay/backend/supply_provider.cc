
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

#include "supply_provider.h"

#include <string>

namespace tvm {

// TODO(gigiblender): move this method
std::string GetModuleName(const IRModule& module) {
  return module->GetAttr<String>(tvm::attr::kModuleName).value_or("tvmgen_default");
}

GlobalVarSupply BuildGlobalVarSupply(const IRModule module) {
  return BuildGlobalVarSupply(Array<IRModule>({module}));
}

GlobalVarSupply BuildGlobalVarSupply(const Array<IRModule>& modules) {
  GlobalVarSupply global_var_supply = GlobalVarSupply::EmptySupply();
  // TODO(gigiblender): For now use as prefix the name of the first module.
  if (!modules.empty()) {
    IRModule first_mod = modules.front();
    global_var_supply->name_supply_->prefix_ = GetModuleName(first_mod);
  }
  for (auto& mod : modules) {
    for (auto kv : mod->functions) {
      global_var_supply->ReserveGlobalVar(kv.first);
    }
  }

  return global_var_supply;
}

}  // namespace tvm
