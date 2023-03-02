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
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include "../../meta_schedule/utils.h"
#include "../../relay/analysis/graph_partitioner.h"
#include "../../support/arena.h"
#include "../../tir/ir/functor_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief The helper class to schedule functions and build a new module which calls the new TIR
 * function.
 */
class ThreadBindMutator : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod, int max_thread_per_block) {
    ThreadBindMutator mutator(mod);

    for (const auto& kv : mod->functions) {
      const GlobalVar& gv = kv.first;
      const BaseFunc& func = kv.second;

      if (func->IsInstance<tir::PrimFuncNode>()) {
        IRModule mod = IRModule(Map<GlobalVar, BaseFunc>({kv}));
        tir::Schedule sch = tir::Schedule::Traced(mod, /*seed=*/-1, /*debug_mask=*/0,
                                                  tir::ScheduleErrorRenderLevel::kDetail);
        Array<tir::BlockRV> blocks = meta_schedule::BlockCollector::Collect(sch);
        for (const tir::BlockRV& block : blocks) {
          Array<tir::LoopRV> loops = sch->GetLoops(block);
          tir::LoopRV fused = sch->Fuse(loops, /*preserve_unit_iters=*/false);
          Array<tir::LoopRV> splits =
              sch->Split(fused, /*factors=*/{NullOpt, Integer(256), Integer(max_thread_per_block)});
          sch->Reorder(/*ordered_loop_rvs=*/{splits[1], splits[2], splits[0]});
          sch->Bind(splits[1], "blockIdx.x");
          sch->Bind(splits[2], "threadIdx.x");
        }
        mutator.builder_->AddFunction(sch->mod()->Lookup(gv->name_hint), gv->name_hint);
      } else {
        mutator.builder_->AddFunction(func, gv->name_hint);
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit ThreadBindMutator(const IRModule& mod) : mod_(mod) {}

 private:
  /*! \brief The IRModule */
  const IRModule& mod_;
};

IRModule DefaultSchedule(IRModule mod, int max_thread_per_block) {
  mod = ThreadBindMutator::Transform(mod, max_thread_per_block);
  return mod;
}

namespace transform {

Pass DefaultSchedule(tvm::Target target) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) {
        Integer max_thread_per_block = target->GetAttr<Integer>("max_num_threads").value_or(-1);
        if (target->kind->name == "llvm" || max_thread_per_block == -1) {
          return m;
        }
        return relax::DefaultSchedule(m, max_thread_per_block.IntValue());
      };
  return CreateModulePass(/*pass_function=*/pass_func,      //
                          /*opt_level=*/0,                  //
                          /*pass_name=*/"DefaultSchedule",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.DefaultSchedule").set_body_typed(DefaultSchedule);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
