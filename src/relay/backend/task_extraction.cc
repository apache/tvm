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

#include <tvm/meta_schedule/extracted_task.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/target/target.h>

#include "../../te/operation/create_primfunc.h"
#include "./te_compiler_cache.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace backend {

namespace metaschedule {

using meta_schedule::ExtractedTask;

Array<ExtractedTask> ExtractTask(IRModule mod, Target target,
                                 Map<String, runtime::NDArray> params) {
  backend::BindParamsInModule(mod, params);

  // is_vm=true for backward compatibility
  Array<Pass> pass_seqs = relay::backend::GetPassPrefix(/*is_homogenous=*/true, /*is_vm=*/true);
  pass_seqs.push_back(transform::FuseOps());

  transform::Sequential seq(pass_seqs);
  auto opt_mod = seq(std::move(mod));

  std::vector<ExtractedTask> tasks;
  std::unordered_map<tec::CCacheKey, ExtractedTask> cache;

  PostOrderVisit(opt_mod->Lookup("main"), [target, &tasks, &cache](const Expr& exp) {
    if (exp->IsInstance<FunctionNode>()) {
      Function relay_func = Downcast<Function>(exp);
      if (!relay_func->HasNonzeroAttr(attr::kPrimitive)) {
        return;
      }
      tec::CCacheKey cache_key(relay_func, target);
      auto it = cache.find(cache_key);
      if (it != cache.end()) {
        it->second->weight += 1;
        return;
      }
      Array<te::Tensor> inputs_outputs;
      std::string fused_name;
      std::tie(inputs_outputs, fused_name) =
          tec::LowerTECompute(relay_func, target, /*return_inputs=*/true);
      auto prim_func = tir::CreatePrimFunc(inputs_outputs);
      GlobalVar prim_fn_var(fused_name);
      IRModule relay_mod({{prim_fn_var, relay_func}});
      IRModule tir_mod({{prim_fn_var, prim_func}});
      ExtractedTask extracted_task(fused_name, relay_mod, target, {tir_mod}, 1);
      tasks.push_back(extracted_task);
      cache.emplace(cache_key, extracted_task);
    }
  });
  // Tasks are extracted via post order visit, return the reversed list.
  std::reverse(tasks.begin(), tasks.end());
  std::unordered_map<std::string, int> name_map;
  for (ExtractedTask task : tasks) {
    task->task_name = tec::GetUniqueName(task->task_name, &name_map);
  }
  return tasks;
}

}  // namespace metaschedule

TVM_REGISTER_GLOBAL("relay.backend.MetaScheduleExtractTask")
    .set_body_typed([](IRModule mod, Target target, Map<String, runtime::NDArray> params) {
      return metaschedule::ExtractTask(mod, target, params);
    });

}  // namespace backend
}  // namespace relay
}  // namespace tvm
