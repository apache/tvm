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
#include <tvm/ir/name_supply.h>
#include <tvm/meta_schedule/extracted_task.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/target/target.h>

#include <numeric>

#include "../../meta_schedule/module_equality.h"
#include "../../te/operation/create_primfunc.h"
#include "./te_compiler_cache.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace backend {

class OpCounter : public ExprVisitor {
 public:
  static size_t GetOpCount(relay::Function func) {
    OpCounter counter;
    counter(func->body);
    return counter.count;
  }

 private:
  void VisitExpr_(const CallNode* call) final {
    if (call->op->IsInstance<OpNode>()) {
      ++count;
    }
    ExprVisitor::VisitExpr_(call);
  }

  size_t count{0};
};

Array<meta_schedule::ExtractedTask> ExtractTask(IRModule mod, Target target,
                                                Map<String, runtime::NDArray> params,
                                                String mod_eq_name) {
  using meta_schedule::ExtractedTask;
  using meta_schedule::ModuleEqual;
  using meta_schedule::ModuleHash;
  backend::BindParamsInModule(mod, params);
  // is_vm=true for backward compatibility
  Array<Pass> pass_seqs = relay::backend::GetPassPrefix(/*is_homogenous=*/true, /*is_vm=*/true);
  pass_seqs.push_back(transform::FuseOps());

  mod = transform::Sequential(pass_seqs)(std::move(mod));

  std::vector<ExtractedTask> tasks;

  auto mod_eq = meta_schedule::ModuleEquality::Create(mod_eq_name);

  std::unordered_map<IRModule, ExtractedTask, ModuleHash, ModuleEqual> cache(
      /*bucket_count*/ 0, ModuleHash(*mod_eq), ModuleEqual(*mod_eq));

  std::vector<std::tuple<std::string, Function, IRModule>> lower_results;

  NameSupply constant_name_supply("");

  PostOrderVisit(mod->Lookup("main"), [&](const Expr& exp) {
    if (exp->IsInstance<FunctionNode>()) {
      Function relay_func = Downcast<Function>(exp);
      if (!relay_func->HasNonzeroAttr(attr::kPrimitive)) {
        return;
      }

      auto [f, fused_name] = tec::LowerToPrimFunc(relay_func, target, constant_name_supply);
      if (f) {
        IRModule tir_mod = PrimFuncToIRModule(f.value());
        lower_results.push_back(std::make_tuple(fused_name, relay_func, tir_mod));
      }
    }
  });

  std::vector<int> indices(lower_results.size());
  std::iota(indices.begin(), indices.end(), 0);

  if (mod_eq_name == "anchor-block") {
    std::vector<size_t> op_counts(lower_results.size());
    for (size_t i = 0; i < op_counts.size(); ++i) {
      op_counts[i] = OpCounter::GetOpCount(std::get<1>(lower_results[i]));
    }

    // When anchor-block based equality is used, tuning tasks "nn_conv2d_add_nn_relu" and
    // "nn_conv2d_add_add_nn_relu", for example, can be identified as equal. Thus, one of
    // them will be filtered by the cache below.
    //
    // To make sure that we tune "nn_conv2d_add_nn_relu" and not "nn_conv2d_add_add_nn_relu",
    // we sort the TE lowering results based on the number of relay ops. This way,
    // "nn_conv2d_add_nn_relu" will be added to the cache first, and "nn_conv2d_add_add_nn_relu"
    // will be filtered.
    std::sort(indices.begin(), indices.end(),
              [&op_counts](int i1, int i2) { return op_counts[i1] < op_counts[i2]; });
  }

  for (auto i : indices) {
    const auto& [fused_name, relay_func, tir_mod] = lower_results[i];
    auto it = cache.find(tir_mod);
    if (it != cache.end()) {
      it->second->weight += 1;
      continue;
    }
    // Note that the cache is key-ed on the tir mod, rather than the relay mod
    IRModule relay_mod({{GlobalVar(fused_name), relay_func}});
    ExtractedTask task(fused_name, relay_mod, target, {tir_mod}, 1);
    tasks.push_back(task);
    cache.emplace(tir_mod, task);
  }

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
