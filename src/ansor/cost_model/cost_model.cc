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
 * \file ansor/cost_model.h
 * \brief Cost model that estimates the performance of programs
 */

#include "cost_model.h"

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <utility>

namespace tvm {
namespace ansor {

using ::tvm::runtime::NDArray;

TVM_REGISTER_OBJECT_TYPE(CostModelNode);
TVM_REGISTER_OBJECT_TYPE(RandomModelNode);

void RandomNumber(TVMArgs args, TVMRetValue* rv) {
  int n = args[0];
  void* data = args[1];
  float* fdata = reinterpret_cast<float*>(data);
  for (int i = 0; i < n; i++) {
    fdata[i] = static_cast<float>(rand_r(nullptr)) / (static_cast<float>(RAND_MAX));
  }
}

RandomModel::RandomModel() {
  ObjectPtr<RandomModelNode> node = make_object<RandomModelNode>();
  node->random_number_func =
      runtime::Registry::Get("ansor.cost_model.random_number");
  if (node->random_number_func == nullptr) {
    LOG(WARNING) << "ansor.cost_model.random_number is not registered, "
                 << "use C++ default random_number func instead.";
    static PackedFunc cost_model_random_number(RandomNumber);
    node->random_number_func = &cost_model_random_number;
  }
  data_ = std::move(node);
}

void RandomModelNode::Update(const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {}

void RandomModelNode::Predict(const SearchTask& task,
                              const std::vector<State>& states,
                              std::vector<float>* scores) {
  scores->resize(states.size());
  (*random_number_func)(states.size(), static_cast<void*>(scores->data()));
}

TVM_REGISTER_GLOBAL("ansor.RandomModel").set_body_typed([]() {
  return RandomModel();
});

}  // namespace ansor
}  // namespace tvm
