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

TVM_REGISTER_GLOBAL("meta_schedule.winograd_output.cuda")
    .set_body_typed([](Schedule sch, BlockRV output) -> Array<Schedule> {
      // get loops
      Array<LoopRV> loops = sch->GetLoops(output);
      ICHECK_EQ(loops.size(), 4);

      BlockRV OL{nullptr};
      if (sch->GetConsumers(output).size() > 0) {
        OL = GetOnlyConsumer(sch, output);
        sch->SetScope(OL, /*buffer_index=*/0, /*storage_scope=*/"local");
      }

      // tile
      BlockRV inverse = GetOnlyProducer(sch, output);
      Optional<PrimExpr> tile_size =
          tir::GetAnn<PrimExpr>(sch->GetSRef(output), "winograd_tile_size");
      ICHECK(tile_size.defined()) << "Winograd tile size is not defined in block annotation!";
      Array<LoopRV> split0 = sch->Split(loops[2], {NullOpt, tile_size.value()});
      Array<LoopRV> split1 = sch->Split(loops[3], {NullOpt, tile_size.value()});
      sch->Reorder({split0[0], split1[0], split0[1], split1[1]});

      // compute_at
      sch->ComputeAt(inverse, /*loop_rv=*/split1[0],
                     /*preserve_unit_loops=*/true);
      if (OL.defined()) {
        while (sch->GetConsumers(OL).size() > 0) {
          BlockRV next_OL = GetOnlyConsumer(sch, OL);
          sch->ComputeInline(OL);
          OL = next_OL;
        }
        sch->ReverseComputeAt(OL, /*loop_rv=*/split1[0],
                              /*preserve_unit_loops=*/true);
      }

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

TVM_REGISTER_GLOBAL("meta_schedule.winograd_bgemm.nchw.cuda")
    .set_body_typed([](Schedule sch, BlockRV bgemm) -> Array<Schedule> {
      BlockRV OL = sch->CacheWrite(bgemm, /*buffer_index=*/0, /*storage_scope=*/"local");
      BlockRV AA = sch->CacheRead(bgemm, /*buffer_index=*/0, /*storage_scope=*/"shared");
      BlockRV BB = sch->CacheRead(bgemm, /*buffer_index=*/1, /*storage_scope=*/"shared");

      Array<LoopRV> loops = sch->GetLoops(bgemm);
      ICHECK_EQ(loops.size(), 5);  // SSSSR
      LoopRV fused = sch->Fuse({loops[0], loops[1]});
      Array<ExprRV> factors0 = sch->SamplePerfectTile(fused, /*n=*/4, /*max_innermost_factor=*/64);
      Array<LoopRV> split0 =
          sch->Split(fused, /*factors=*/{factors0[0], factors0[1], factors0[2], factors0[3]},
                     /*preserve_unit_loops=*/true);
      Array<ExprRV> factors1 =
          sch->SamplePerfectTile(loops[2], /*n=*/4, /*max_innermost_factor=*/64);
      Array<LoopRV> split1 =
          sch->Split(loops[2], /*factors=*/{factors1[0], factors1[1], factors1[2], factors1[3]},
                     /*preserve_unit_loops=*/true);
      Array<ExprRV> factors2 =
          sch->SamplePerfectTile(loops[3], /*n=*/4, /*max_innermost_factor=*/64);
      Array<LoopRV> split2 =
          sch->Split(loops[3], /*factors=*/{factors2[0], factors2[1], factors2[2], factors2[3]},
                     /*preserve_unit_loops=*/true);
      Array<ExprRV> factors3 =
          sch->SamplePerfectTile(loops[4], /*n=*/2, /*max_innermost_factor=*/64);
      Array<LoopRV> split3 = sch->Split(loops[4], /*factors=*/{factors3[0], factors3[1]},
                                        /*preserve_unit_loops=*/true);
      sch->Bind(split0[0], "blockIdx.z");
      sch->Bind(split1[0], "blockIdx.y");
      sch->Bind(split2[0], "blockIdx.x");
      sch->Bind(split0[1], "vthread.z");
      sch->Bind(split1[1], "vthread.y");
      sch->Bind(split2[1], "vthread.x");
      sch->Bind(split0[2], "threadIdx.z");
      sch->Bind(split1[2], "threadIdx.y");
      sch->Bind(split2[2], "threadIdx.x");
      sch->Reorder({split0[0], split1[0], split2[0], split0[1], split1[1], split2[1], split0[2],
                    split1[2], split2[2], split3[0], split3[1], split0[3], split1[3], split2[3]});

      // tile reduction axes
      sch->ReverseComputeAt(OL, split2[2], /*preserve_unit_loops=*/false);

      sch->ComputeAt(AA, split3[0], /*preserve_unit_loops=*/true);
      sch->ComputeAt(BB, split3[0], /*preserve_unit_loops=*/true);

      // cooperative fetching
      ExprRV ann_val = sch->SampleCategorical(
          {Integer(1), Integer(2), Integer(3), Integer(4)},
          {FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25),
           FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25)});
      sch->Annotate(AA, tir::attr::meta_schedule_cooperative_fetch, ann_val);
      sch->Annotate(BB, tir::attr::meta_schedule_cooperative_fetch, ann_val);

      ann_val = sch->SampleCategorical(
          {Integer(0), Integer(16), Integer(64), Integer(512)},
          {FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25),
           FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25)});
      sch->Annotate(bgemm, tir::attr::meta_schedule_unroll_explicit, ann_val);

      ann_val = sch->SampleCategorical(
          {Integer(128), Integer(256), Integer(512), Integer(1024)},
          {FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25),
           FloatImm(DataType::Float(32), 0.25), FloatImm(DataType::Float(32), 0.25)});
      sch->Annotate(bgemm, tir::attr::pragma_auto_unroll_max_step, ann_val);

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
      LoopRV loop = ScheduleDataPackNCHW(sch, data_pack);
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
