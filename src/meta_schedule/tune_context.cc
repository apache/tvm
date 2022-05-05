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

TuneContext::TuneContext(Optional<IRModule> mod,                                    //
                         Optional<Target> target,                                   //
                         Optional<SpaceGenerator> space_generator,                  //
                         Optional<SearchStrategy> search_strategy,                  //
                         Optional<Array<ScheduleRule>> sch_rules,                   //
                         Optional<Array<Postproc>> postprocs,                       //
                         Optional<Map<Mutator, FloatImm>> mutator_probs,            //
                         Optional<String> task_name,                                //
                         PackedFunc logging_func,                                   //
                         support::LinearCongruentialEngine::TRandState rand_state,  //
                         int num_threads) {
  ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>();
  n->mod = mod;
  n->target = target;
  n->space_generator = space_generator;
  n->search_strategy = search_strategy;
  n->sch_rules = sch_rules.value_or({});
  n->postprocs = postprocs.value_or({});
  n->mutator_probs = mutator_probs.value_or({});
  n->task_name = task_name;
  n->logging_func = logging_func;
  support::LinearCongruentialEngine(&n->rand_state).Seed(rand_state);
  n->num_threads = num_threads;
  n->is_terminated = false;
  n->runner_futures = NullOpt;
  n->measure_candidates = NullOpt;
  data_ = std::move(n);
}

void TuneContextNode::Initialize() {
  if (this->space_generator.defined()) {
    this->space_generator.value()->InitializeWithTuneContext(GetRef<TuneContext>(this));
  }
  if (this->search_strategy.defined()) {
    this->search_strategy.value()->InitializeWithTuneContext(GetRef<TuneContext>(this));
  }
  for (const ScheduleRule& sch_rule : sch_rules) {
    sch_rule->InitializeWithTuneContext(GetRef<TuneContext>(this));
  }
  for (const Postproc& postproc : postprocs) {
    postproc->InitializeWithTuneContext(GetRef<TuneContext>(this));
  }
  for (const auto& kv : mutator_probs) {
    kv.first->InitializeWithTuneContext(GetRef<TuneContext>(this));
  }
}

TVM_REGISTER_NODE_TYPE(TuneContextNode);

TVM_REGISTER_GLOBAL("meta_schedule.TuneContext")
    .set_body_typed([](Optional<IRModule> mod,                                    //
                       Optional<Target> target,                                   //
                       Optional<SpaceGenerator> space_generator,                  //
                       Optional<SearchStrategy> search_strategy,                  //
                       Optional<Array<ScheduleRule>> sch_rules,                   //
                       Optional<Array<Postproc>> postprocs,                       //
                       Optional<Map<Mutator, FloatImm>> mutator_probs,            //
                       Optional<String> task_name,                                //
                       PackedFunc logging_func,                                   //
                       support::LinearCongruentialEngine::TRandState rand_state,  //
                       int num_threads) -> TuneContext {
      return TuneContext(mod, target, space_generator, search_strategy, sch_rules, postprocs,
                         mutator_probs, task_name, logging_func, rand_state, num_threads);
    });

TVM_REGISTER_GLOBAL("meta_schedule._SHash2Hex").set_body_typed(SHash2Hex);

}  // namespace meta_schedule
}  // namespace tvm
