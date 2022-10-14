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

#include "../../meta_schedule/module_equality.h"
#include "../../te/operation/create_primfunc.h"
#include "./te_compiler_cache.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace backend {

Array<meta_schedule::ExtractedTask> ExtractTask(IRModule mod, Target target,
                                                Map<String, runtime::NDArray> params,
                                                String mod_eq_name) {
  using meta_schedule::ExtractedTask;
  using meta_schedule::ModuleEqual;
  using meta_schedule::ModuleHash;
  backend::FTECompilerTIRConverter tir_converter = backend::GetTIRConverter();
  backend::BindParamsInModule(mod, params);
  // is_vm=true for backward compatibility
  Array<Pass> pass_seqs = relay::backend::GetPassPrefix(/*is_homogenous=*/true, /*is_vm=*/true);
  pass_seqs.push_back(transform::FuseOps());

  mod = transform::Sequential(pass_seqs)(std::move(mod));

  std::vector<ExtractedTask> tasks;

  auto mod_eq = meta_schedule::ModuleEquality::Create(mod_eq_name);

  std::unordered_map<IRModule, ExtractedTask, ModuleHash, ModuleEqual> cache(
      /*bucket_count*/ 0, ModuleHash(*mod_eq), ModuleEqual(*mod_eq));

  PostOrderVisit(mod->Lookup("main"), [&target, &tasks, &cache, &tir_converter](const Expr& exp) {
    if (exp->IsInstance<FunctionNode>()) {
      Function relay_func = Downcast<Function>(exp);
      if (!relay_func->HasNonzeroAttr(attr::kPrimitive)) {
        return;
      }

      auto [inputs_outputs, constants, fused_name] =
          tec::LowerTECompute(relay_func, target, /*return_inputs=*/true);

      if (Optional<tir::PrimFunc> f = tir_converter(inputs_outputs, constants)) {
        IRModule tir_mod = PrimFuncToIRModule(f.value());

        auto it = cache.find(tir_mod);
        if (it != cache.end()) {
          it->second->weight += 1;
          return;
        }

        // Note that the cache is key-ed on the tir mod, rather than the relay mod
        IRModule relay_mod({{GlobalVar(fused_name), relay_func}});
        ExtractedTask task(fused_name, relay_mod, target, {tir_mod}, 1);
        tasks.push_back(task);
        cache.emplace(tir_mod, task);
      }
    }
  });
  // Tasks are extracted via post order visit, return the reversed list.
  std::reverse(tasks.begin(), tasks.end());
  NameSupply name_supply = NameSupply("");
  for (ExtractedTask task : tasks) {
    task->task_name = name_supply->FreshName(task->task_name);
  }
  return tasks;
}

TVM_REGISTER_GLOBAL("relay.backend.MetaScheduleExtractTask").set_body_typed(ExtractTask);

}  // namespace backend
}  // namespace relay
}  // namespace tvm
