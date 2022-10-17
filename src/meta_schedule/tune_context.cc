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
#include <utility>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

TuneContext::TuneContext(Optional<IRModule> mod, Optional<Target> target,
                         Optional<SpaceGenerator> space_generator,
                         Optional<SearchStrategy> search_strategy, Optional<String> task_name,
                         int num_threads, TRandState rand_state, PackedFunc logger) {
  CHECK(rand_state == -1 || rand_state >= 0) << "ValueError: Invalid random state: " << rand_state;
  ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>();
  n->mod = mod;
  n->target = target;
  n->space_generator = space_generator;
  n->search_strategy = search_strategy;
  n->task_name = task_name;
  n->num_threads = num_threads;
  n->rand_state = support::LinearCongruentialEngine::NormalizeSeed(rand_state);
  n->logger = logger;
  data_ = std::move(n);
}

TuneContext TuneContextNode::Clone() const {
  ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>(*this);
  if (this->space_generator.defined()) {
    n->space_generator = this->space_generator.value()->Clone();
  }
  if (this->search_strategy.defined()) {
    n->search_strategy = this->search_strategy.value()->Clone();
  }
  n->rand_state = ForkSeed(&n->rand_state);
  n->Initialize();
  return TuneContext(n);
}

void TuneContextNode::Initialize() {
  if (this->space_generator.defined()) {
    this->space_generator.value()->InitializeWithTuneContext(GetRef<TuneContext>(this));
  }
  if (this->search_strategy.defined()) {
    this->search_strategy.value()->InitializeWithTuneContext(GetRef<TuneContext>(this));
  }
}

TVM_REGISTER_NODE_TYPE(TuneContextNode);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContext")
    .set_body_typed([](Optional<IRModule> mod, Optional<Target> target,
                       Optional<SpaceGenerator> space_generator,
                       Optional<SearchStrategy> search_strategy, Optional<String> task_name,
                       int num_threads, TRandState rand_state, PackedFunc logger) -> TuneContext {
      return TuneContext(mod, target, space_generator, search_strategy, task_name, num_threads,
                         rand_state, logger);
    });
TVM_REGISTER_GLOBAL("meta_schedule._SHash2Hex").set_body_typed(SHash2Hex);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextInitialize")
    .set_body_method<TuneContext>(&TuneContextNode::Initialize);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextClone")
    .set_body_method<TuneContext>(&TuneContextNode::Clone);

}  // namespace meta_schedule
}  // namespace tvm
