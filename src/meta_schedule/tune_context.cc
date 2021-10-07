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
#include <random>
#include <utility>

#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Constructor function of TuneContext class.
 * \param mod The mod to be optimized.
 * \param target The target to be optimized for.
 * \param space_generator The design space generator.
 * \param task_name The name of the tuning task.
 * \param rand_state The random state.
 * \param num_threads The number of threads to be used.
 * \param verbose The verbosity level.
 */
TuneContext::TuneContext(Optional<IRModule> mod,                                    //
                         Optional<Target> target,                                   //
                         Optional<SpaceGenerator> space_generator,                  //
                         Optional<SearchStrategy> search_strategy,                  //
                         Optional<String> task_name,                                //
                         support::LinearCongruentialEngine::TRandState rand_state,  //
                         int num_threads) {
  ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>();
  n->mod = mod;
  n->target = target;
  n->space_generator = space_generator;
  n->search_strategy = search_strategy;
  n->task_name = task_name;
  if (rand_state == -1) {
    rand_state = std::random_device()();
  }
  support::LinearCongruentialEngine(&n->rand_state).Seed(rand_state);
  n->num_threads = num_threads;
  n->is_stopped = false;
  n->runner_futures = NullOpt;
  n->measure_candidates = NullOpt;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TuneContextNode);

TVM_REGISTER_GLOBAL("meta_schedule.TuneContext")
    .set_body_typed([](Optional<IRModule> mod,                                    //
                       Optional<Target> target,                                   //
                       Optional<SpaceGenerator> space_generator,                  //
                       Optional<SearchStrategy> search_strategy,                  //
                       Optional<String> task_name,                                //
                       support::LinearCongruentialEngine::TRandState rand_state,  //
                       int num_threads) -> TuneContext {
      return TuneContext(mod, target, space_generator, search_strategy, task_name, rand_state,
                         num_threads);
    });
}  // namespace meta_schedule
}  // namespace tvm
