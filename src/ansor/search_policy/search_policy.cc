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
 * \file ansor/search_policy/search_policy.cc
 * \brief The base class for search policy
 */

#include "search_policy.h"
#include <tvm/runtime/registry.h>

namespace tvm {
namespace ansor {

TVM_REGISTER_OBJECT_TYPE(SearchPolicyNode);

// Search Policy
TVM_REGISTER_GLOBAL("ansor.SearchPolicyContinueSearchOneRound")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  SearchPolicy policy = args[0];
  SearchTask task = args[1];
  int num_measure = args[2];
  int verbose = args[3];
  ProgramMeasurer measurer = args[4];

  Array<MeasureInput> inputs;
  Array<MeasureResult> results;
  std::tie(inputs, results) = policy->ContinueSearchOneRound(task, num_measure, verbose, measurer);

  *ret = Array<ObjectRef>{inputs, results};
});

}  // namespace ansor
}  // namespace tvm
