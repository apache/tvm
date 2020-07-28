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
 * \file auto_scheduler/search_policy/search_policy.cc
 * \brief The base class of search policies.
 */

#include <tvm/auto_scheduler/search_policy.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(SearchCallbackNode);
TVM_REGISTER_OBJECT_TYPE(SearchPolicyNode);

void SearchPolicyNode::RunCallbacks(const Optional<Array<SearchCallback>>& callbacks) {
  if (callbacks) {
    for (const auto& callback : callbacks.value()) {
      callback->Callback(this);
    }
  }
}

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyRunCallbacks")
    .set_body_typed([](SearchPolicy policy, Optional<Array<SearchCallback>> callbacks) {
      policy->RunCallbacks(callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicySetTask")
    .set_body_typed([](SearchPolicy policy, SearchTask task) { policy->cur_task = task; });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicySetVerbose")
    .set_body_typed([](SearchPolicy policy, int verbose) { policy->verbose = verbose; });

}  // namespace auto_scheduler
}  // namespace tvm
