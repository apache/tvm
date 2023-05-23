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
#include <tvm/meta_schedule/schedule/generic/winograd.h>

#include "../../utils.h"

namespace tvm {
namespace meta_schedule {

using namespace tvm::tir;

static Array<tir::LoopRV> ScheduleDataPack(tir::Schedule sch, tir::BlockRV block,
                                           std::vector<int> tiled, std::vector<int> unrolled) {
  using namespace tvm::tir;
  ICHECK_EQ(tiled.size(), 2);
  ICHECK_EQ(unrolled.size(), 4);
  Array<ExprRV> factors{nullptr};
  Array<LoopRV> loops = sch->GetLoops(block);
  ICHECK_EQ(loops.size(), 6);

  factors = sch->SamplePerfectTile(loops[tiled[0]], /*n=*/2, /*max_innermost_factor=*/64);
  Array<LoopRV> t0 = sch->Split(loops[tiled[0]], {factors.begin(), factors.end()});
  ICHECK_EQ(t0.size(), 2);

  factors = sch->SamplePerfectTile(loops[tiled[1]], /*n=*/2, /*max_innermost_factor=*/64);
  Array<LoopRV> t1 = sch->Split(loops[tiled[1]], {factors.begin(), factors.end()});
  ICHECK_EQ(t1.size(), 2);

  sch->Unroll(loops[unrolled[0]]);
  sch->Unroll(loops[unrolled[1]]);
  sch->Unroll(loops[unrolled[2]]);
  sch->Unroll(loops[unrolled[3]]);
  sch->Reorder({
      t0[0],
      t1[0],
      t0[1],
      t1[1],
      loops[unrolled[0]],
      loops[unrolled[1]],
      loops[unrolled[2]],
      loops[unrolled[3]],
  });
  return {t0[0], t1[0], t0[1], t1[1]};
}

TVM_REGISTER_GLOBAL("meta_schedule.cpu.conv2d_nhwc_winograd_data_pack")
    .set_body_typed([](Schedule sch, BlockRV data_pack) -> Array<Schedule> {
      BlockRV input_tile = GetWinogradProducerAndInlineConst(sch, data_pack);
      BlockRV data_pad = GetWinogradProducerAndInlineConst(sch, input_tile);
      ScheduleDataPack(sch, data_pack, {2, 3}, {0, 1, 4, 5});
      sch->ComputeAt(input_tile, /*loop_rv=*/sch->SampleComputeLocation(input_tile),
                     /*preserve_unit_loops=*/true);
      sch->ComputeAt(data_pad, /*loop_rv=*/sch->SampleComputeLocation(data_pad),
                     /*preserve_unit_loops=*/true);
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.cpu.conv2d_nhwc_winograd_inverse")
    .set_body_typed([](Schedule sch, BlockRV block) -> Array<Schedule> {
      GetWinogradProducerAndInlineConst(sch, block);
      ScheduleDataPack(sch, block, {2, 3}, {0, 1, 4, 5});
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.cpu.conv2d_nchw_winograd_data_pack")
    .set_body_typed([](Schedule sch, BlockRV data_pack) -> Array<Schedule> {
      BlockRV input_tile = GetWinogradProducerAndInlineConst(sch, data_pack);
      BlockRV data_pad = GetWinogradProducerAndInlineConst(sch, input_tile);
      ScheduleDataPack(sch, data_pack, {2, 3}, {0, 1, 4, 5});
      sch->ComputeAt(input_tile, /*loop_rv=*/sch->SampleComputeLocation(input_tile),
                     /*preserve_unit_loops=*/true);
      sch->ComputeAt(data_pad, /*loop_rv=*/sch->SampleComputeLocation(data_pad),
                     /*preserve_unit_loops=*/true);
      return {sch};
    });

TVM_REGISTER_GLOBAL("meta_schedule.cpu.conv2d_nchw_winograd_inverse")
    .set_body_typed([](Schedule sch, BlockRV block) -> Array<Schedule> {
      GetWinogradProducerAndInlineConst(sch, block);
      ScheduleDataPack(sch, block, {0, 1}, {2, 3, 4, 5});
      return {sch};
    });

}  // namespace meta_schedule
}  // namespace tvm
