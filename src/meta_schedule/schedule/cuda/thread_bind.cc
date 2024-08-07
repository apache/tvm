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
#include <tvm/meta_schedule/schedule/cuda/thread_bind.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "../../utils.h"

namespace tvm {
namespace meta_schedule {

using namespace tvm::tir;

std::function<ExprRV(int64_t)> MakeFactorSampler(Schedule sch, Array<Integer> thread_extents) {
  return [sch = std::move(sch),
          thread_extents = std::move(thread_extents)](int64_t max_extent) -> ExprRV {
    Array<Integer> extents;
    extents.reserve(thread_extents.size());
    for (const Integer extent : thread_extents) {
      if (extent->value <= max_extent) {
        extents.push_back(extent);
      }
    }
    int n = extents.size();
    if (n == 0) {
      return Integer(max_extent);
    }
    if (n == 1) {
      return Integer(extents[0]);
    }
    Array<FloatImm> probs(n, FloatImm(DataType::Float(64), 1.0 / n));
    return sch->SampleCategorical(extents, probs);
  };
}

Array<LoopRV> BindSpatialLoop(Schedule sch, LoopRV loop, int64_t max_threadblocks,
                              int64_t max_threads_per_block,
                              std::function<ExprRV(int64_t)> get_factor) {
  int64_t extent = -1;
  if (const int64_t* e = as_const_int(sch->Get(loop)->extent)) {
    extent = *e;
  } else {
    extent = std::numeric_limits<int64_t>::max();
  }
  if (extent <= max_threadblocks * max_threads_per_block) {
    if (!get_factor) {
      get_factor = MakeFactorSampler(sch, {32, 64, 128, 256, 512, 1024});
    }
    ExprRV factor = get_factor(std::min(extent, max_threads_per_block));
    Array<LoopRV> splits = sch->Split(loop, {NullOpt, factor});
    ICHECK_EQ(splits.size(), 2);
    sch->Bind(splits[0], "blockIdx.x");
    sch->Bind(splits[1], "threadIdx.x");
    return {splits[0], splits[1]};
  } else {
    Array<LoopRV> splits = sch->Split(loop, {NullOpt,
                                             Integer(max_threadblocks),  //
                                             Integer(max_threads_per_block)});
    ICHECK_EQ(splits.size(), 3);
    sch->Reorder({splits[1], splits[2], splits[0]});
    sch->Bind(splits[1], "blockIdx.x");
    sch->Bind(splits[2], "threadIdx.x");
    return {splits[1], splits[2], splits[0]};
  }
}

void BindBlockThreadIdx(tir::Schedule sch, tir::BlockRV block_rv,  //
                        int64_t max_threadblocks, int64_t max_threads_per_block,
                        std::function<tir::ExprRV(int64_t)> get_factor) {
  using namespace tvm::tir;
  StmtSRef block_sref = sch->GetSRef(block_rv);
  if (block_sref->parent == nullptr) {
    return;
  }
  if (tir::HasBeenMultiLevelTiled(block_sref)) {
    return;
  }
  Array<StmtSRef> loops = tir::GetLoops(block_sref);
  int n = loops.size();
  int i_block_idx = -1;
  int i_thread_idx = -1;
  int i_multi_child = -1;
  int i_spatial_loop = -1;
  for (int i = 0; i < n; ++i) {
    const StmtSRef& loop_sref = loops[i];
    const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
    runtime::ThreadScope thread_scope = GetThreadScope(loop);
    if (IsBlockIdx(thread_scope)) {
      if (i_block_idx == -1) {
        i_block_idx = i;
      }
    }
    if (IsThreadIdx(thread_scope)) {
      if (i_thread_idx == -1) {
        i_thread_idx = i;
      }
    }
    if (loop->kind != ForKind::kSerial) {
      if (i_multi_child == -1) {
        i_multi_child = i;
      }
    }
    if (!IsSingleStmt(loop->body)) {
      if (i_multi_child == -1) {
        i_multi_child = i + 1;
      }
    }
    if (GetLoopIterType(loop_sref) == IterVarType::kDataPar) {
      if (i_spatial_loop == i - 1) {
        ++i_spatial_loop;
      }
    }
  }
  if (i_multi_child == -1) {
    i_multi_child = n;
  }
  if (i_block_idx != -1 && i_thread_idx != -1) {
    return;
  }
  if (i_block_idx != -1 && i_thread_idx == -1) {
    ICHECK(false) << "Unsupported case, where blockIdx is bound but threadIdx is not";
    throw;
  }
  LoopRV loop_rv{nullptr};
  {
    Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    if (i_spatial_loop == -1) {
      LoopRV spatial_loop_rv{nullptr};
      if (loop_rvs.empty()) {
        spatial_loop_rv = sch->AddUnitLoop(block_rv);
      } else {
        spatial_loop_rv = sch->AddUnitLoop(loop_rvs[0]);
      }
      loop_rvs.insert(loop_rvs.begin(), spatial_loop_rv);
      i_spatial_loop = 0;
      if (i_block_idx != -1) {
        i_block_idx += 1;
      }
      if (i_thread_idx != -1) {
        i_thread_idx += 1;
      }
      if (i_multi_child != -1) {
        i_multi_child += 1;
      }
    }
    if (i_block_idx == -1 && i_thread_idx != -1) {
      int num_fuse = std::min(std::min(i_multi_child, i_thread_idx), i_spatial_loop + 1);
      Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
      loop_rv = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + num_fuse});
      sch->Bind(loop_rv, "blockIdx.x");
      return;
    } else {  // i_block_idx == -1 && i_thread_idx == -1
      int num_fuse = std::min(i_multi_child, i_spatial_loop + 1);
      loop_rv = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + num_fuse});
    }
  }
  BindSpatialLoop(sch, loop_rv, max_threadblocks, max_threads_per_block, get_factor);
}

}  // namespace meta_schedule
}  // namespace tvm
