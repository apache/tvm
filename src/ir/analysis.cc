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
 * \file src/ir/analysis.cc
 * \brief Analysis functions that must span multiple IR types
 */
#include <tvm/ir/analysis.h>

#include "../support/ordered_set.h"

namespace tvm {
namespace ir {

Map<GlobalVar, Array<GlobalVar>> CollectCallMap(const IRModule& mod) {
  struct CalleeCollectorImpl : CalleeCollector {
    void Mark(GlobalVar gvar) override { gvars.push_back(gvar); }
    support::OrderedSet<GlobalVar> gvars;
  };

  Map<GlobalVar, Array<GlobalVar>> call_map;
  for (const auto& [gvar, base_func] : mod->functions) {
    CalleeCollectorImpl collector;
    CalleeCollector::vtable()(base_func, &collector);
    call_map.Set(gvar, Array<GlobalVar>{collector.gvars.begin(), collector.gvars.end()});
  }
  return call_map;
}

TVM_REGISTER_GLOBAL("ir.analysis.CollectCallMap").set_body_typed(CollectCallMap);

}  // namespace ir
}  // namespace tvm
