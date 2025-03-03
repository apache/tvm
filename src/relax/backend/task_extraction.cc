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
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include "../../meta_schedule/module_equality.h"

namespace tvm {
namespace relax {
namespace backend {

using meta_schedule::ExtractedTask;
using meta_schedule::ModuleEqual;
using meta_schedule::ModuleEquality;
using meta_schedule::ModuleHash;

/*!
 * \brief Extract the Meta-Schedule tuning task from a given IRModule.
 * \note
 *   1. The task extractor is responsible for task deduplication. The
 *   deduplication is achieved by comparing structural hashes of PrimFuncs.
 *   2. For a PrimFunc, the weight of its corresponding task is the number
 *   of times it called by op Call-TIR. Say in an IRModule there are three
 *   PrimFuncs `fn1`, `fn2` and `fn3` sharing the same structural hash.
 *   Suppose `fn1` is called by 5 Call-TIR ops among all Relax function,
 *   `fn2` is called by 3 Call-TIR and `fn3` is called by 5 Call-TIR.
 *   Then we will have a ExtractedTask for all three functions, whose weight
 *   is 5 + 3 + 2 = 10.
 */
class BlockCounter : public tir::StmtVisitor {
 public:
  static size_t GetBlockCount(const tir::PrimFunc& func) {
    BlockCounter counter;
    counter(func->body);
    return counter.count;
  }

 private:
  void VisitStmt_(const tir::BlockNode* op) final {
    ++count;
    StmtVisitor::VisitStmt_(op);
  }
  size_t count{0};
};

class TaskExtractor : public ExprVisitor {
 public:
  static Array<ExtractedTask> ExtractTask(IRModule mod, Target target, String mod_eq_name) {
    TaskExtractor extractor(mod, target, mod_eq_name);
    // We go through each Relax function in the module.
    for (const auto& kv : mod->functions) {
      if (const auto* func = kv.second.as<FunctionNode>()) {
        extractor(GetRef<Function>(func));
      }
    }
    Array<ExtractedTask> tasks;
    for (const auto& it : extractor.func2task_) {
      tasks.push_back(it.second);
    }
    return tasks;
  }

 private:
  explicit TaskExtractor(IRModule mod, Target target, String mod_eq_name)
      : mod_(std::move(mod)),
        target_(std::move(target)),
        mod_eq_(ModuleEquality::Create(mod_eq_name)),
        func2task_(/*bucket_count*/ 0, ModuleHash(*mod_eq_), ModuleEqual(*mod_eq_)) {
    normalize_mod_func_ = runtime::Registry::Get("tvm.meta_schedule.normalize_mod");
    ICHECK(normalize_mod_func_) << "Normalization function is not found.";
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    // TODO(@tvm-team): When we differentiate the call for tir function and packed function,
    // this logic should be changed accordingly.
    if (!call->op.same_as(call_tir_op)) {
      // Since the Relax function is of A-normal form, the arguments of this call cannot be another
      // Calls. And hence we do not need to recurse into this Call.
      return;
    }

    const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
    const tir::PrimFunc& func = Downcast<tir::PrimFunc>(mod_->Lookup(global_var));
    IRModule mod = (*normalize_mod_func_)(func);
    size_t weight = 1;
    auto it = func2task_.find(mod);
    if (it != func2task_.end()) {
      it->second->weight += 1;
      const tir::PrimFunc& alt_func = Downcast<tir::PrimFunc>(it->first->Lookup("main"));
      // When anchor-block based equality is used, tuning tasks "nn_conv2d_add_nn_relu" and
      // "nn_conv2d_add_add_nn_relu", for example, can be identified as equal. Thus, one of them
      // will be selected to tune by the code below.
      //
      // To make sure that we tune "nn_conv2d_add_nn_relu" and not "nn_conv2d_add_add_nn_relu", we
      // count the PrinFunc number of blocks and leave only the function with the smallest number of
      // blocks. This way, "nn_conv2d_add_nn_relu" will have a smaller number of blocks than
      // "nn_conv2d_add_add_nn_relu" and will be selected to tune.
      if (BlockCounter::GetBlockCount(func) < BlockCounter::GetBlockCount(alt_func)) {
        weight += it->second->weight;
        func2task_.erase(it->first);
      }
    }

    ExtractedTask task(/*task_name=*/global_var->name_hint,  //
                       /*mod=*/mod,                          //
                       /*target=*/target_,                   //
                       /*dispatched=*/{mod},                 //
                       /*weight=*/weight);
    func2task_.emplace(mod, task);
  }

  IRModule mod_;
  Target target_;
  std::unique_ptr<ModuleEquality> mod_eq_;
  std::unordered_map<IRModule, ExtractedTask, ModuleHash, ModuleEqual> func2task_;
  const runtime::PackedFunc* normalize_mod_func_;
};

TVM_REGISTER_GLOBAL("relax.backend.MetaScheduleExtractTask")
    .set_body_typed([](IRModule mod, Target target, String mod_eq_name) {
      return TaskExtractor::ExtractTask(std::move(mod), std::move(target), std::move(mod_eq_name));
    });

}  // namespace backend
}  // namespace relax
}  // namespace tvm
