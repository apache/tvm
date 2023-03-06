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

#include "../../meta_schedule/utils.h"

namespace tvm {
namespace tir {
namespace transform {
/*!
 * \brief A helper function to do default thread binding for a block.
 * \param sch The schedule to work on.
 * \param block The block to be scheduled.
 * \param max_thread_per_block The maximum number of threads per block.
 * \param max_threadblocks The maximum number of threadblocks.
 */
void ThreadBind(tir::Schedule sch, const tir::BlockRV& block, int64_t max_thread_per_block,
                int64_t max_threadblocks = 256) {
  // fetch the loops
  Array<tir::LoopRV> loops = sch->GetLoops(block);
  for (const tir::LoopRV& loop : loops) {
    // skip block if already scheduled
    if (sch->Get(loop)->thread_binding.defined()) {
      return;
    }
  }
  Array<tir::IterVar> iters = sch->Get(block)->iter_vars;
  ICHECK_EQ(loops.size(), iters.size());
  Array<tir::LoopRV> data_parallel_loops;
  // only fuse data parallel loops
  for (size_t i = 0; i < loops.size(); ++i) {
    if (iters[i]->iter_type == tir::IterVarType::kDataPar) {
      data_parallel_loops.push_back(loops[i]);
    }
  }
  // skip if no data parallel loops
  if (data_parallel_loops.size() == 0) {
    return;
  }
  // fuse all data parallel loops
  tir::LoopRV fused = sch->Fuse(data_parallel_loops, /*preserve_unit_iters=*/false);
  int64_t product = std::numeric_limits<int64_t>::max();
  if (sch->Get(fused)->extent->IsInstance<tir::IntImmNode>()) {
    product = sch->Get(fused)->extent.as<tir::IntImmNode>()->value;
  }
  // schedule the fused loop
  if (product > max_thread_per_block * max_threadblocks) {
    Array<tir::LoopRV> splits =
        sch->Split(fused,
                   /*factors=*/{NullOpt, Integer(max_threadblocks), Integer(max_thread_per_block)});
    sch->Reorder(/*ordered_loop_rvs=*/{splits[1], splits[2], splits[0]});
    sch->Bind(splits[1], "blockIdx.x");
    sch->Bind(splits[2], "threadIdx.x");
  } else {
    Array<tir::LoopRV> splits =
        sch->Split(fused, /*factors=*/{NullOpt, Integer(std::min(product, max_thread_per_block))});
    sch->Bind(splits[0], "blockIdx.x");
    sch->Bind(splits[1], "threadIdx.x");
  }
}

Pass DefaultGPUSchedule() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) {
        // get the target from context.
        tvm::Target target = tvm::Target::Current();
        ICHECK(target.defined()) << "Target is not set in current context";
        // skip non-cuda targets.
        if (target->kind->name != "cuda") {
          return m;
        }
        // get the max thread per block from target.
        Optional<Integer> opt_max_thread_per_block = target->GetAttr<Integer>("max_num_threads");
        ICHECK(opt_max_thread_per_block.defined())
            << "max_num_threads is not set for target " << target;
        int64_t max_thread_per_block = opt_max_thread_per_block.value().IntValue();
        tir::Schedule sch = tir::Schedule::Traced(m, /*seed=*/-1, /*debug_mask=*/0,
                                                  tir::ScheduleErrorRenderLevel::kDetail);
        for (const auto& [gv, func] : m->functions) {
          if (func->IsInstance<tir::PrimFuncNode>()) {
            sch->WorkOn(gv->name_hint);
            Array<tir::BlockRV> blocks = meta_schedule::BlockCollector::Collect(sch);
            for (const tir::BlockRV& block : blocks) {
              ThreadBind(sch, block, max_thread_per_block);
            }
          }
        }
        return sch->mod();
      };
  return CreateModulePass(/*pass_function=*/pass_func,         //
                          /*opt_level=*/0,                     //
                          /*pass_name=*/"DefaultGPUSchedule",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("tir.transform.DefaultGPUSchedule").set_body_typed(DefaultGPUSchedule);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
