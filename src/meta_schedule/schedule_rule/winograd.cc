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

inline LoopRV ScheduleDataPackNCHW(Schedule sch, BlockRV block) {
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

  Array<ExprRV> factors = sch->SamplePerfectTile(loops[3], /*n=*/2, /*max_innermost_factor=*/64);
  Array<LoopRV> split =
      sch->Split(loops[3], /*factors=*/{factors[0], factors[1]}, /*preserve_unit_loops=*/true);

  LoopRV fused = sch->Fuse({loops[2], split[0]});
  sch->Reorder({fused, split[1], loops[0], loops[1]});
  return split[1];
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

TVM_REGISTER_GLOBAL("meta_schedule.winograd_output.nchw.cuda")
    .set_body_typed([](Schedule sch, BlockRV output) -> Array<Schedule> {
      // get loops
      Array<LoopRV> loops = sch->GetLoops(output);
      ICHECK_EQ(loops.size(), 4);

      BlockRV OL{nullptr};

      // tile
      Optional<PrimExpr> tile_size =
          tir::GetAnn<PrimExpr>(sch->GetSRef(output), "winograd_tile_size");
      ICHECK(tile_size.defined()) << "Winograd tile size is not defined in block annotation!";
      Array<LoopRV> split0 = sch->Split(loops[2], {NullOpt, tile_size.value()});
      Array<LoopRV> split1 = sch->Split(loops[3], {NullOpt, tile_size.value()});
      sch->Reorder({split0[0], split1[0], split0[1], split1[1]});

      // compute_at
      BlockRV inverse = GetOnlyProducer(sch, output);
      sch->ComputeAt(inverse, /*loop_rv=*/split1[0],
                     /*preserve_unit_loops=*/true);

      // fuse
      LoopRV fused = sch->Fuse({loops[0], loops[1], split0[0], split1[0]});

      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, output, max_threadblocks, max_threads_per_block, get_factor);
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

TVM_REGISTER_GLOBAL("meta_schedule.winograd_inverse.nchw.cuda")
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

TVM_REGISTER_GLOBAL("meta_schedule.winograd_kernel_pack.nchw.cuda")
    .set_body_typed([](Schedule sch, BlockRV kernel_pack) -> Array<Schedule> {
      Array<LoopRV> loops = sch->GetLoops(kernel_pack);
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

      LoopRV fused = sch->Fuse({loops[2], loops[3]});

      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, kernel_pack, max_threadblocks, max_threads_per_block, get_factor);
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

TVM_REGISTER_GLOBAL("meta_schedule.winograd_data_pack.nchw.cuda")
    .set_body_typed([](Schedule sch, BlockRV data_pack) -> Array<Schedule> {
      BlockRV input_tile = GetOnlyProducer(sch, data_pack);
      BlockRV data_pad = GetOnlyProducer(sch, input_tile);

      BlockRV data_l = sch->CacheWrite(data_pack, /*buffer_index=*/0, /*storage_scope=*/"local");
      BlockRV d = sch->CacheRead(data_pack, /*buffer_index=*/0, /*storage_scope=*/"local");
      LoopRV loop = ScheduleDataPackNCHW(sch, data_pack);
      sch->ReverseComputeAt(data_l, loop, /*preserve_unit_loops=*/true);
      sch->ComputeAt(d, /*loop_rv=*/loop, /*preserve_unit_loops=*/true);
      sch->ComputeInline(data_pad);

      int64_t max_threadblocks = 256;
      int64_t max_threads_per_block = 1024;
      auto get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
      BindBlockThreadIdx(sch, data_pack, max_threadblocks, max_threads_per_block, get_factor);
      return {sch};
    });

}  // namespace meta_schedule
}  // namespace tvm
