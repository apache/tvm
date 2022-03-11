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

#include <tvm/meta_schedule/integration.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/target/target.h>

#include "../../te/operation/create_primfunc.h"
#include "te_compiler_cache.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {

namespace metaschedule {

using meta_schedule::ExtractedTask;

Array<ExtractedTask> ExtractTask(IRModule mod, Target target, Map<String, Constant> params) {
  backend::BindParamsInModule(mod, params);

  // is_vm=true for backward compatibility
  Array<Pass> pass_seqs = relay::backend::GetPassPrefix(/*is_homogenous=*/true, /*is_vm=*/true);
  pass_seqs.push_back(transform::FuseOps());

  transform::Sequential seq(pass_seqs);
  auto opt_mod = seq(std::move(mod));

  Array<ExtractedTask> tasks;
  std::unordered_set<tec::CCacheKey> cache_;
  std::unordered_map<std::string, int> name_map;

  PostOrderVisit(opt_mod->Lookup("main"), [target, &tasks, &cache_, &name_map](const Expr& exp) {
    if (exp->IsInstance<FunctionNode>()) {
      Function relay_func = Downcast<Function>(exp);
      tec::CCacheKey cache_key(relay_func, target);
      if (relay_func->HasNonzeroAttr(attr::kPrimitive) && cache_.find(cache_key) == cache_.end()) {
        Array<te::Tensor> outputs;
        std::string fused_name;
        std::tie(outputs, fused_name) =
            tec::LowerTECompute(relay_func, target, /*return_inputs*/ true);
        auto prim_func = tir::CreatePrimFunc(outputs);
        auto prim_fn_var = GlobalVar(fused_name);
        auto relay_mod = IRModule({{prim_fn_var, relay_func}});
        auto tir_mod = IRModule({{prim_fn_var, prim_func}});
        auto task_name = tec::GetUniqueName(fused_name, &name_map);
        tasks.push_back(ExtractedTask(task_name, relay_mod, target, {tir_mod}));
        cache_.insert(cache_key);
      }
    }
  });

  return tasks;
}

}  // namespace metaschedule

TVM_REGISTER_GLOBAL("relay.backend.MetaScheduleExtractTask")
    .set_body_typed([](IRModule mod, Target target, Map<String, Constant> params) {
      return metaschedule::ExtractTask(mod, target, params);
    });

}  // namespace backend
}  // namespace relay
}  // namespace tvm
