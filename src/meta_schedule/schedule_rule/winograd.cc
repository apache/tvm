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
#include "../utils.h"
#include "./auto_bind.h"

namespace tvm {
namespace meta_schedule {

using namespace tvm::tir;

TVM_REGISTER_GLOBAL("meta_schedule.compute_inline")
    .set_body_typed([](Schedule sch, BlockRV block) -> Array<Schedule> {
      sch->ComputeInline(block);
      return {sch};
    });

inline BlockRV GetOnlyProducer(Schedule sch, BlockRV block) {
  Array<BlockRV> producers = sch->GetProducers(block);
  ICHECK_EQ(producers.size(), 1);
  return producers[0];
}

inline LoopRV ScheduleDataPack(Schedule sch, BlockRV block) {
  Array<ExprRV> factors{nullptr};
  Array<LoopRV> loops = sch->GetLoops(block);
  ICHECK_EQ(loops.size(), 6);

  factors = sch->SamplePerfectTile(loops[2], /*n=*/2, /*max_innermost_factor=*/64);
  Array<LoopRV> t0 = sch->Split(loops[2], {factors.begin(), factors.end()});
  ICHECK_EQ(t0.size(), 2);

  factors = sch->SamplePerfectTile(loops[3], /*n=*/2, /*max_innermost_factor=*/64);
  Array<LoopRV> t1 = sch->Split(loops[3], {factors.begin(), factors.end()});
  ICHECK_EQ(t1.size(), 2);

  if (const int64_t* i = tir::GetLoopIntExtent(sch->GetSRef(loops[0]))) {
    if (*i <= 16) {
      sch->Unroll(loops[0]);
    }
  }
  if (const int64_t* i = tir::GetLoopIntExtent(sch->GetSRef(loops[1]))) {
    if (*i <= 16) {
      sch->Unroll(loops[1]);
    }
  }
  sch->Unroll(loops[4]);
  sch->Unroll(loops[5]);
  sch->Reorder({
      t0[0],
      t1[0],
      t0[1],
      t1[1],
      loops[0],
      loops[1],
      loops[4],
      loops[5],
  });
  return t1[1];
}

TVM_REGISTER_GLOBAL("meta_schedule.winograd_inverse.llvm")
    .set_body_typed([](Schedule sch, BlockRV block) -> Array<Schedule> {
      ScheduleDataPack(sch, block);
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.winograd_data_pack.llvm")
    .set_body_typed([](Schedule sch, BlockRV data_pack) -> Array<Schedule> {
      BlockRV input_tile = GetOnlyProducer(sch, data_pack);
      BlockRV data_pad = GetOnlyProducer(sch, input_tile);
      ScheduleDataPack(sch, data_pack);
      sch->ComputeAt(input_tile, /*loop_rv=*/sch->SampleComputeLocation(input_tile),
                     /*preserve_unit_loops=*/true);
      sch->ComputeAt(data_pad, /*loop_rv=*/sch->SampleComputeLocation(data_pad),
                     /*preserve_unit_loops=*/true);
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.winograd_inverse.cuda")
    .set_body_typed([](Schedule sch, BlockRV block) -> Array<Schedule> {
      ScheduleDataPack(sch, block);
      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, block, max_threadblocks, max_threads_per_block, get_factor);
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.winograd_data_pack.cuda")
    .set_body_typed([](Schedule sch, BlockRV data_pack) -> Array<Schedule> {
      BlockRV input_tile = GetOnlyProducer(sch, data_pack);
      BlockRV data_pad = GetOnlyProducer(sch, input_tile);
      LoopRV loop = ScheduleDataPack(sch, data_pack);
      sch->ComputeAt(input_tile, /*loop_rv=*/loop, /*preserve_unit_loops=*/true);
      sch->SetScope(input_tile, /*buffer_index=*/0, /*storage_scope=*/"local");
      sch->ComputeInline(data_pad);
      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, data_pack, max_threadblocks, max_threads_per_block, get_factor);
      return {sch};
    });

}  // namespace meta_schedule
}  // namespace tvm
