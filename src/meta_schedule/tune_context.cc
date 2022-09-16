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

TuneContext TuneContextNode::Clone() const {
  ObjectPtr<TuneContextNode> n = make_object<TuneContextNode>(*this);
  if (this->sch_rules.defined()) {
    n->sch_rules = Array<ScheduleRule>();
    for (const ScheduleRule& sch_rule : this->sch_rules) {
      n->sch_rules.push_back(sch_rule->Clone());
    }
  }
  if (this->postprocs.defined()) {
    n->postprocs = Array<Postproc>();
    for (const Postproc& postproc : this->postprocs) {
      n->postprocs.push_back(postproc->Clone());
    }
  }
  if (this->mutator_probs.defined()) {
    n->mutator_probs = Map<Mutator, FloatImm>();
    for (const auto& kv : this->mutator_probs) {
      n->mutator_probs.Set(kv.first->Clone(), kv.second);
    }
  }
  if (this->space_generator.defined()) n->space_generator = this->space_generator.value()->Clone();
  if (this->search_strategy.defined()) n->search_strategy = this->search_strategy.value()->Clone();
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

void TuneContextNode::_SetMeasureCandidates(const Array<MeasureCandidate>& candidates) {
  this->measure_candidates = candidates;
}

void TuneContextNode::_SendToBuilder(const Builder& builder) {
  auto _ = Profiler::TimedScope("SendToBuilder");
  Array<MeasureCandidate> candidates = this->measure_candidates.value();
  Target target = this->target.value();
  Array<BuilderInput> inputs;
  inputs.reserve(candidates.size());
  for (const MeasureCandidate& candidate : candidates) {
    inputs.push_back(BuilderInput(candidate->sch->mod(), target));
  }
  this->builder_results = builder->Build(inputs);
}

void TuneContextNode::_SendToRunner(const Runner& runner) {
  auto _ = Profiler::TimedScope("SendToRunner");
  Array<MeasureCandidate> candidates = this->measure_candidates.value();
  Array<BuilderResult> builder_results = this->builder_results.value();
  Target target = this->target.value();
  ICHECK_EQ(candidates.size(), builder_results.size());
  int n = candidates.size();
  int n_build_errors = 0;
  Array<RunnerInput> inputs;
  inputs.reserve(n);
  for (int i = 0; i < n; ++i) {
    const MeasureCandidate& candidate = candidates[i];
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.defined()) {
      ++n_build_errors;
      continue;
    }
    inputs.push_back(RunnerInput(/*artifact_path=*/builder_result->artifact_path.value(),
                                 /*device_type=*/target->kind->name,
                                 /*args_info=*/candidate->args_info));
  }
  Array<RunnerFuture> futures = runner->Run(inputs);
  if (n_build_errors == 0) {
    this->runner_futures = futures;
    return;
  }
  Array<RunnerFuture> results;
  results.reserve(n);
  for (int i = 0, j = 0; i < n; ++i) {
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.defined()) {
      results.push_back(RunnerFuture(
          /*f_done=*/[]() -> bool { return true; },
          /*f_result=*/
          [msg = builder_result->error_msg]() -> RunnerResult {
            return RunnerResult(NullOpt, msg);
          }));
    } else {
      results.push_back(futures[j++]);
    }
  }
  this->runner_futures = results;
}

Array<RunnerResult> TuneContextNode::_Join() {
  ICHECK(this->runner_futures.defined());
  Array<RunnerFuture> futures = this->runner_futures.value();
  int n = futures.size();
  Array<RunnerResult> results;
  {
    auto _ = Profiler::TimedScope("JoinRunnerFutures");
    results.reserve(n);
    for (RunnerFuture future : futures) {
      results.push_back(future->Result());
    }
  }
  if (this->search_strategy.defined()) {
    this->search_strategy.value()->NotifyRunnerResults(this->measure_candidates.value(), results);
  }
  ICHECK(this->measure_candidates.defined());
  ICHECK(this->builder_results.defined());
  ICHECK_EQ(results.size(), this->measure_candidates.value().size());
  ICHECK_EQ(results.size(), this->builder_results.value().size());
  return results;
}

void TuneContextNode::_ClearMeasureState() {
  this->measure_candidates = NullOpt;
  this->builder_results = NullOpt;
  this->runner_futures = NullOpt;
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
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextInitialize")
    .set_body_method<TuneContext>(&TuneContextNode::Initialize);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextSetMeasureCandidates")
    .set_body_method<TuneContext>(&TuneContextNode::_SetMeasureCandidates);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextSendToBuilder")
    .set_body_method<TuneContext>(&TuneContextNode::_SendToBuilder);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextSendToRunner")
    .set_body_method<TuneContext>(&TuneContextNode::_SendToRunner);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextJoin")
    .set_body_method<TuneContext>(&TuneContextNode::_Join);
TVM_REGISTER_GLOBAL("meta_schedule.TuneContextClearMeasureState")
    .set_body_method<TuneContext>(&TuneContextNode::_ClearMeasureState);

}  // namespace meta_schedule
}  // namespace tvm
