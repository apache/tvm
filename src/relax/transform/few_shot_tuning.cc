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

#include <tvm/relax/transform.h>

#include "../../meta_schedule/utils.h"

namespace tvm {
namespace relax {
namespace transform {

tir::PrimFunc FewShotTunePrimFunc(const tir::PrimFunc& prim_func, const Target& target,
                                  int64_t valid_count, Optional<meta_schedule::Runner> runner) {
  // fetch a local builder
  static const auto* f_get_local_builder =
      runtime::Registry::Get("meta_schedule.builder.get_local_builder");
  ICHECK(f_get_local_builder)
      << "ValueError: Cannot find the packed function \"meta_schedule.builder.get_local_builder\"";
  meta_schedule::Builder builder = (*f_get_local_builder)();
  // create an IRModule
  IRModule mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar("main"), prim_func}}));
  // fetch the number of physical cores
  static const auto* f_cpu_count = runtime::Registry::Get("meta_schedule.cpu_count");
  ICHECK(f_cpu_count) << "ValueError: Cannot find the packed function \"meta_schedule._cpu_count\"";
  int num_threads = (*f_cpu_count)(false);
  // store the results
  Array<IRModule> results;
  std::vector<double> costs;
  // create a TuneContext
  meta_schedule::TuneContext task = meta_schedule::TuneContext(
      /*mod=*/mod,
      /*target=*/target,
      /*space_generator=*/
      meta_schedule::SpaceGenerator::PostOrderApply(/*f_block_filter=*/nullptr,
                                                    /*sch_rules=*/NullOpt,
                                                    /*postprocs=*/NullOpt,
                                                    /*mutator_probs=*/NullOpt),
      /*search_strategy=*/meta_schedule::SearchStrategy::ReplayTrace(/*max_fail_count=*/100),
      /*task_name=*/NullOpt,
      /*num_threads=*/num_threads,  // use all available local threads
      /*rand_state=*/-1,            // -1 means use random seed
      /*logger=*/nullptr);
  task->search_strategy.value()->PreTuning(
      /*max_trials=*/valid_count, /*num_trials_per_iter=*/valid_count,
      /*design_spaces=*/task->space_generator.value()->GenerateDesignSpace(mod),
      /*database=*/NullOpt,
      /*cost_model=*/NullOpt);
  while (valid_count > 0) {
    Optional<Array<meta_schedule::MeasureCandidate>> candidates =
        task->search_strategy.value()->GenerateMeasureCandidates();
    if (!candidates.defined()) break;
    Array<meta_schedule::BuilderInput> builder_inputs;
    for (const meta_schedule::MeasureCandidate& candidate : candidates.value()) {
      builder_inputs.push_back(meta_schedule::BuilderInput(
          /*mod=*/candidate->sch->mod(),
          /*target=*/target));
    }
    Array<meta_schedule::BuilderResult> builder_results = builder->Build(builder_inputs);
    ICHECK_EQ(builder_results.size(), candidates.value().size());
    int idx = 0;
    for (const meta_schedule::BuilderResult& builder_result : builder_results) {
      if (!builder_result->error_msg.defined()) {
        results.push_back(candidates.value()[idx]->sch->mod());
        valid_count--;
      }
      idx++;
    }
    if (runner.defined()) {
      Array<meta_schedule::RunnerInput> runner_inputs;
      int idx = 0;
      for (const meta_schedule::BuilderResult& builder_result : builder_results) {
        if (!builder_result->error_msg.defined()) {
          runner_inputs.push_back(meta_schedule::RunnerInput(
              /*artifact_path=*/builder_result->artifact_path.value(),
              /*device_type=*/target->kind->name,
              /*args_info=*/candidates.value()[idx]->args_info));
        }
        idx++;
      }
      Array<meta_schedule::RunnerFuture> runner_futures = runner.value()->Run(runner_inputs);
      for (const meta_schedule::RunnerFuture& runner_future : runner_futures) {
        meta_schedule::RunnerResult runner_result = runner_future->Result();
        if (runner_result->error_msg.defined()) {
          costs.push_back(1e10);
        } else {
          double sum = 0;
          for (const FloatImm& cost : runner_result->run_secs.value()) {
            sum += cost->value;
          }
          costs.push_back(sum / runner_result->run_secs.value().size());
        }
      }
      ICHECK_EQ(costs.size(), results.size());
    }
  }
  if (results.size() == 0) {
    LOG(WARNING) << "No valid schedule found";
    return prim_func;
  }
  if (runner.defined()) {
    int best_idx = 0;
    for (int i = 1; i < costs.size(); ++i) {
      if (costs[i] < costs[best_idx]) {
        best_idx = i;
      }
    }
    return WithAttr(Downcast<tir::PrimFunc>(results[best_idx]->Lookup("main")),
                    tvm::tir::attr::kIsScheduled, Bool(true));
  } else {
    int best_idx = rand_r(0) % results.size();
    return Downcast<tir::PrimFunc>(results[best_idx]->Lookup("main"));
  }
}

Pass FewShotTuning(Integer valid_count, ObjectRef runner) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) {
        CHECK(valid_count.defined() && valid_count->value > 0) << "Valid_count must be positive.";
        // get the target from context.
        tvm::Target target = tvm::Target::Current();
        ICHECK(target.defined()) << "Target is not set in current context";
        // generate the few shot tuned prim funcs.
        Map<GlobalVar, BaseFunc> result;
        for (const auto& [gv, func] : m->functions) {
          Optional<meta_schedule::Runner> runner_ = NullOpt;
          if (runner.defined()) {
            runner_ = Downcast<meta_schedule::Runner>(runner);
          }
          if (func->IsInstance<tir::PrimFuncNode>() &&
              !func->HasNonzeroAttr(tir::attr::kIsScheduled)) {
            result.Set(gv, FewShotTunePrimFunc(GetRef<tir::PrimFunc>(func.as<tir::PrimFuncNode>()),
                                               target, valid_count->value, runner_));
          } else {
            result.Set(gv, func);
          }
        }
        return IRModule(result,               // functions
                        m->type_definitions,  // type_definitions
                        m->import_set_,       // import_set
                        m->source_map,        // map
                        m->attrs);            // attrs);
      };
  return CreateModulePass(/*pass_function=*/pass_func,    //
                          /*opt_level=*/0,                //
                          /*pass_name=*/"FewShotTuning",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.FewShotTuning").set_body_typed(FewShotTuning);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
