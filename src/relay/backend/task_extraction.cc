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

bool DefaultTaskFilter(const Array<te::Tensor>& args) {
  using namespace ::tvm::te;
  std::vector<Tensor> stack;
  std::unordered_set<const TensorNode*> visited;
  for (const Tensor& v : args) {
    for (const PrimExpr& e : v->shape) {
      // Dynamic shape is not supported for now
      if (!e->IsInstance<IntImmNode>()) {
        return false;
      }
    }
    if (!visited.count(v.get())) {
      visited.insert(v.get());
      stack.push_back(v);
    }
  }
  while (!stack.empty()) {
    Tensor tensor = stack.back();
    stack.pop_back();
    if (tensor->op->IsInstance<PlaceholderOpNode>()) {
      // do nothing
    } else if (tensor->op->IsInstance<ComputeOpNode>() || tensor->op->IsInstance<ExternOpNode>()) {
      Array<Tensor> inputs = tensor->op->InputTensors();
      for (const Tensor& v : inputs) {
        if (!visited.count(v.get())) {
          visited.insert(v.get());
          stack.push_back(v);
        }
      }
    } else {
      return false;
    }
  }
  return true;
}

Array<meta_schedule::ExtractedTask> ExtractTask(
    IRModule mod, Target target, Map<String, runtime::NDArray> params,
    runtime::TypedPackedFunc<bool(const Array<te::Tensor>&)> filter_func) {
  using meta_schedule::ExtractedTask;
  if (filter_func == nullptr) {
    filter_func = DefaultTaskFilter;
  }
  backend::BindParamsInModule(mod, params);
  // is_vm=true for backward compatibility
  Array<Pass> pass_seqs = relay::backend::GetPassPrefix(/*is_homogenous=*/true, /*is_vm=*/true);
  pass_seqs.push_back(transform::FuseOps());
  mod = transform::Sequential(pass_seqs)(std::move(mod));

  std::vector<ExtractedTask> tasks;
  std::unordered_map<tec::CCacheKey, ExtractedTask> cache;
  PostOrderVisit(mod->Lookup("main"), [&target, &tasks, &cache, &filter_func](const Expr& exp) {
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
      Array<te::Tensor> inputs_outputs{nullptr};
      std::string fused_name;
      std::tie(inputs_outputs, fused_name) =
          tec::LowerTECompute(relay_func, target, /*return_inputs=*/true);
      if (filter_func(inputs_outputs)) {
        tir::PrimFunc prim_func = tir::CreatePrimFunc(inputs_outputs);
        GlobalVar prim_fn_var(fused_name);
        IRModule relay_mod({{prim_fn_var, relay_func}});
        IRModule tir_mod({{prim_fn_var, prim_func}});
        ExtractedTask extracted_task(fused_name, relay_mod, target, {tir_mod}, 1);
        tasks.push_back(extracted_task);
        cache.emplace(cache_key, extracted_task);
      }
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

TVM_REGISTER_GLOBAL("relay.backend.MetaScheduleExtractTask").set_body_typed(ExtractTask);

}  // namespace backend
}  // namespace relay
}  // namespace tvm
