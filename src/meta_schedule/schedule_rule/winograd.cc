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

inline BlockRV GetOnlyConsumer(Schedule sch, BlockRV block) {
  Array<BlockRV> consumers = sch->GetConsumers(block);
  ICHECK_EQ(consumers.size(), 1);
  return consumers[0];
}

inline LoopRV ScheduleDataPack(Schedule sch, BlockRV block) {
  Array<ExprRV> factors{nullptr};
  Array<LoopRV> loops = sch->GetLoops(block);
  ICHECK_EQ(loops.size(), 6);

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

  factors = sch->SamplePerfectTile(loops[3], /*n=*/2, /*max_innermost_factor=*/64);
  Array<LoopRV> t0 = sch->Split(loops[3], {factors[0], factors[1]});
  ICHECK_EQ(t0.size(), 2);

  LoopRV fused = sch->Fuse({loops[2], t0[0]});
  sch->Reorder({fused, t0[1], loops[0], loops[1]});
  return t0[1];
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

TVM_REGISTER_GLOBAL("meta_schedule.winograd_output.cuda")
    .set_body_typed([](Schedule sch, BlockRV output) -> Array<Schedule> {
      // get loops
      Array<LoopRV> loops = sch->GetLoops(output);
      ICHECK_EQ(loops.size(), 4);
      Array<ExprRV> factors{nullptr};
      // compute at
      LoopRV loop = loops[3];
      BlockRV inverse = GetOnlyProducer(sch, output);
      sch->ComputeAt(inverse, /*loop_rv=*/loop,
                     /*preserve_unit_loops=*/true);
      if (sch->GetConsumers(output).size() > 0) {
        BlockRV OL = GetOnlyConsumer(sch, output);
        sch->SetScope(OL, /*buffer_index=*/0, /*storage_scope=*/"local");
        sch->ReverseComputeAt(OL, /*loop_rv=*/loop,
                              /*preserve_unit_loops=*/true);
      }
      // tile
      factors = sch->SamplePerfectTile(loops[2], /*n=*/2, /*max_innermost_factor=*/64);
      Array<LoopRV> t0 = sch->Split(loops[2], {factors[0], factors[1]});
      factors = sch->SamplePerfectTile(loops[3], /*n=*/2, /*max_innermost_factor=*/64);
      Array<LoopRV> t1 = sch->Split(loops[3], {factors[0], factors[1]});
      sch->Reorder({t0[0], t1[0], t0[1], t1[1]});
      // fuse
      LoopRV fused = sch->Fuse({loops[0], loops[1], t0[0], t1[0]});

      // bind
      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, output, max_threadblocks, max_threads_per_block, get_factor);

      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.winograd_inverse.cuda")
    .set_body_typed([](Schedule sch, BlockRV inverse) -> Array<Schedule> {
      sch->SetScope(inverse, /*buffer_index=*/0, /*storage_scope=*/"local");
      Array<LoopRV> loops = sch->GetLoops(inverse);
      ICHECK_EQ(loops.size(), 6);
      if (const int64_t* i = tir::GetLoopIntExtent(sch->GetSRef(loops[2]))) {
        if (*i <= 16) {
          sch->Unroll(loops[2]);
        }
      }
      if (const int64_t* i = tir::GetLoopIntExtent(sch->GetSRef(loops[3]))) {
        if (*i <= 16) {
          sch->Unroll(loops[3]);
        }
      }
      sch->Unroll(loops[4]);
      sch->Unroll(loops[5]);
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.winograd_bgemm.cuda")
    .set_body_typed([](Schedule sch, BlockRV bgemm) -> Array<Schedule> {
      BlockRV data_pack = GetOnlyProducer(sch, bgemm);

      BlockRV OL = sch->CacheWrite(bgemm, /*buffer_index=*/0, /*storage_scope=*/"local");
      BlockRV AA = sch->CacheRead(bgemm, /*buffer_index=*/0, /*storage_scope=*/"shared");
      BlockRV BB = sch->CacheRead(bgemm, /*buffer_index=*/1, /*storage_scope=*/"shared");

      Array<LoopRV> loops = sch->GetLoops(bgemm);
      ICHECK_EQ(loops.size(), 5);  // SSSSR
      LoopRV fused = sch->Fuse({loops[0], loops[1]});
      Array<LoopRV> t0 = sch->Split(
          fused, /*factors=*/{sch->Get(fused)->extent, Integer(1), Integer(1), Integer(1)},
          /*preserve_unit_loops=*/true);
      Array<ExprRV> factors =
          sch->SamplePerfectTile(loops[2], /*n=*/4, /*max_innermost_factor=*/64);
      Array<LoopRV> t1 =
          sch->Split(loops[2], /*factors=*/{factors[0], factors[1], factors[2], factors[3]},
                     /*preserve_unit_loops=*/true);
      factors = sch->SamplePerfectTile(loops[3], /*n=*/4, /*max_innermost_factor=*/64);
      Array<LoopRV> t2 =
          sch->Split(loops[3], /*factors=*/{factors[0], factors[1], factors[2], factors[3]},
                     /*preserve_unit_loops=*/true);
      factors = sch->SamplePerfectTile(loops[4], /*n=*/2, /*max_innermost_factor=*/64);
      Array<LoopRV> t3 =
          sch->Split(loops[4], /*factors=*/{factors[0], factors[1]}, /*preserve_unit_loops=*/true);
      sch->Bind(t0[0], "blockIdx.z");
      sch->Bind(t1[0], "blockIdx.y");
      sch->Bind(t2[0], "blockIdx.x");
      sch->Bind(t0[2], "threadIdx.z");
      sch->Bind(t1[2], "threadIdx.y");
      sch->Bind(t2[2], "threadIdx.x");
      sch->Reorder(
          {t0[0], t1[0], t2[0], t0[1], t1[1], t2[1], t0[2], t1[2], t2[2], t0[3], t1[3], t2[3]});

      // tile reduction axes
      loops = sch->GetLoops(OL);
      // tile reduction axes
      loops = sch->GetLoops(OL);
      // tile reduction axes
      loops = sch->GetLoops(OL);
      ICHECK_EQ(loops.size(), 4);  // SSSS
      fused = sch->Fuse({loops[0], loops[1]});
      sch->ReverseComputeAt(OL, t2[2], /*preserve_unit_loops=*/true);
      sch->ComputeAt(AA, t3[1], /*preserve_unit_loops=*/true);
      sch->ComputeAt(BB, t3[1], /*preserve_unit_loops=*/true);

      // cooperative fetching
      ExprRV ann_val = sch->SampleCategorical(
          {1, 2, 3, 4}, {FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25),
                         FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25)});
      sch->Annotate(AA, tir::attr::meta_schedule_cooperative_fetch, ann_val);
      sch->Annotate(BB, tir::attr::meta_schedule_cooperative_fetch, ann_val);

      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, bgemm, max_threadblocks, max_threads_per_block, get_factor);
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.winograd_data_pack.cuda")
    .set_body_typed([](Schedule sch, BlockRV data_pack) -> Array<Schedule> {
      BlockRV data_l = sch->CacheWrite(data_pack, /*buffer_index=*/0, /*storage_scope=*/"local");
      BlockRV input_tile = GetOnlyProducer(sch, data_pack);
      BlockRV data_pad = GetOnlyProducer(sch, input_tile);
      LoopRV loop = ScheduleDataPack(sch, data_pack);
      sch->ReverseComputeAt(data_l, loop, /*preserve_unit_loops=*/true);
      sch->ComputeAt(input_tile, /*loop_rv=*/loop, /*preserve_unit_loops=*/true);
      sch->ComputeInline(data_pad);
      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, data_pack, max_threadblocks, max_threads_per_block, get_factor);
      return {sch};
    });

}  // namespace meta_schedule
}  // namespace tvm
