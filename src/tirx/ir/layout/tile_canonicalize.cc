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

/*
 * Canonicalization routines for TileLayout.
 */
#include "utils.h"

namespace tvm {
namespace tirx {

// Forward declarations for helpers used before their definitions
TileLayout SortReplicaIters(TileLayout layout);

TileLayout RemoveUnitIters(TileLayout layout) {
  auto new_layout = layout.CopyOnWrite();
  std::vector<Iter> new_shard;
  std::copy_if(layout->shard.begin(), layout->shard.end(), std::back_inserter(new_shard),
               [](const Iter& iter) { return !is_one(iter->extent); });
  // if new_shard is empty, add a unit iter (using axis from original shard)
  if (new_shard.empty() && !layout->shard.empty()) {
    new_shard.push_back(Iter(1, 1, layout->shard[0]->axis));
  }
  new_layout->shard = new_shard;
  return ffi::GetRef<TileLayout>(new_layout);
}

TileLayout RemoveZeroOffsets(TileLayout layout) {
  auto new_layout = layout.CopyOnWrite();
  ffi::Map<Axis, PrimExpr> new_offset;
  for (const auto& [axis, off] : layout->offset) {
    if (!is_zero(off)) {
      new_offset.Set(axis, off);
    }
  }
  new_layout->offset = new_offset;
  return ffi::GetRef<TileLayout>(new_layout);
}

TileLayout FuseContiguousShardIters(TileLayout layout) {
  std::vector<Iter> fused_shard;
  arith::Analyzer ana;
  const auto& shard = layout->shard;
  for (size_t cur = 0; cur < shard.size();) {
    // Find consecutive fusable axes
    PrimExpr extent = shard[cur]->extent;
    size_t next = cur + 1;
    while (next < shard.size() && shard[next]->axis.same_as(shard[cur]->axis) &&
           ana.CanProveEqual(shard[next]->extent * shard[next]->stride, shard[next - 1]->stride)) {
      extent *= shard[next]->extent;
      ++next;
    }
    if (next == cur + 1) {
      fused_shard.push_back(shard[cur]);
    } else {
      fused_shard.push_back(Iter(extent, shard[next - 1]->stride, shard[cur]->axis));
    }
    cur = next;
  }
  auto new_layout = layout.CopyOnWrite();
  new_layout->shard = fused_shard;
  return ffi::GetRef<TileLayout>(new_layout);
}

TileLayout FuseAxesByScope(TileLayout layout) {
  // Step 1: Get the target and scope information
  auto scope_pair_opt = layout->GetScope();
  Target target = Target::Current();
  if (!scope_pair_opt.has_value() || !target.defined()) {
    return layout;
  }
  auto subscope = scope_pair_opt.value().get<0>()->name();
  auto scope = scope_pair_opt.value().get<1>()->name();

  // Step 2: Create vectors for the new layout components
  std::vector<Iter> shard;
  std::vector<Iter> replica;
  ffi::Map<Axis, PrimExpr> offset;

  // Step 3: Define the axis fusion function
  auto try_fuse_axis = [&](const Iter& iter) -> Iter {
    const auto& fuser = iter->axis->GetFuser();
    return fuser.has_value() ? fuser.value()(target, subscope, scope, iter).value_or(iter) : iter;
  };

  // Step 4: Process shard iterators
  for (auto iter : layout->shard) {
    shard.push_back(try_fuse_axis(iter));
  }
  // Step 5: Process replicate iterators
  for (auto iter : layout->replica) {
    replica.push_back(try_fuse_axis(iter));
  }
  // Step 6: Process offset iterators
  for (auto [axis, off] : layout->offset) {
    Iter iter = try_fuse_axis(Iter(1, off, axis));
    offset.Set(iter->axis, iter->stride);
  }
  // Step 7: Create and return the new layout
  auto result = TileLayout(shard, replica, offset);
  return result;
}

Layout TileLayoutNode::Canonicalize() const {
  // 0. Remove unit iters in shard
  TileLayout res = RemoveUnitIters(ffi::GetRef<TileLayout>(this));
  // 1. Remove zero offset
  res = RemoveZeroOffsets(res);
  // 2. Try fuse axes
  res = FuseAxesByScope(res);
  // 3. Fuse shard iters
  res = FuseContiguousShardIters(res);
  // 3. Sort replicate iters
  res = SortReplicaIters(res);
  return res;
}

TileLayout SortReplicaIters(TileLayout layout) {
  auto n = layout.CopyOnWrite();
  std::vector<Iter> replicate(n->replica.begin(), n->replica.end());
  auto hash_compare = [](const auto& a, const auto& b) {
    return StructuralHash()(a) < StructuralHash()(b);
  };
  std::sort(replicate.begin(), replicate.end(), hash_compare);
  n->replica = std::move(replicate);
  return ffi::GetRef<TileLayout>(n);
}

}  // namespace tirx
}  // namespace tvm
