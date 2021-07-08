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

/*!
 * \file tir/analysis/usmp/algo/greedy_by_size.cc
 * \brief Implement greedy by size memory planning algorithm
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {
namespace usmp {
namespace algo {

size_t round_up_to_byte_alignment(const size_t& non_aligned_byte_offset,
                                  const int& byte_alignment) {
  return ((non_aligned_byte_offset + byte_alignment - 1) / byte_alignment) * byte_alignment;
}

bool IsValidPlacement(const PoolInfo& candidate_pool, const size_t& next_offset,
                      const size_t& size_bytes) {
  if (candidate_pool->size_hint_bytes == -1) {
    // this means pool is not bounded
    return true;
  }
  auto pool_size = static_cast<size_t>(candidate_pool->size_hint_bytes->value);
  auto max_address = next_offset + size_bytes;
  if (max_address <= pool_size) {
    return true;
  }
  return false;
}

PoolInfo SelectPlacementPool(
    const Array<PoolInfo>& pool_candidates,
    const std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual>& pool_offsets) {
  for (const auto& pool_info : pool_candidates) {
    if (pool_offsets.count(pool_info)) {
      return pool_info;
    }
  }
  ICHECK(false) << "TVM USMP Internal Error: no candidate have been selected!";
  return PoolInfo();
}

Map<BufferInfo, PoolAllocation> GreedyBySize(const Array<BufferInfo>& buffer_info_arr) {
  std::vector<BufferInfo> buffer_info_vec;
  Map<BufferInfo, PoolAllocation> pool_allocations;
  for (const auto& buffer_info : buffer_info_arr) {
    buffer_info_vec.push_back(std::move(buffer_info));
  }
  std::sort(buffer_info_vec.begin(), buffer_info_vec.end(),
            [](const BufferInfo& a, const BufferInfo& b) {
              if (a->size_bytes->value == b->size_bytes->value) {
                if (a->conflicts.size() == b->conflicts.size()) {
                  auto a_name_hash = std::hash<std::string>{}(a->name_hint->data);
                  auto b_name_hash = std::hash<std::string>{}(b->name_hint->data);
                  return a_name_hash > b_name_hash;
                } else {
                  return a->conflicts.size() > b->conflicts.size();
                }
              }
              return a->size_bytes > b->size_bytes;
            });

  for (const auto& buf_info : buffer_info_vec) {
    std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual> pool_offset_candidates;
    for (const auto& pool_info : buf_info->pool_candidates) {
      if (IsValidPlacement(pool_info, 0, buf_info->size_bytes->value)) {
        pool_offset_candidates[pool_info] = 0;
      }
    }

    for (const auto& conflict_buf_info_obj : buf_info->conflicts) {
      auto conflict_buf_info = Downcast<BufferInfo>(conflict_buf_info_obj);
      size_t next_offset = 0;
      if (pool_allocations.count(conflict_buf_info)) {
        auto pool_allocation = pool_allocations[conflict_buf_info];
        next_offset = pool_allocation->byte_offset + conflict_buf_info->size_bytes;
        next_offset = round_up_to_byte_alignment(next_offset, conflict_buf_info->alignment->value);
        if (IsValidPlacement(pool_allocation->pool_info, next_offset,
                             buf_info->size_bytes->value)) {
          if (next_offset > pool_offset_candidates[pool_allocation->pool_info]) {
            pool_offset_candidates[pool_allocation->pool_info] = next_offset;
          }
        } else {
          pool_offset_candidates.erase(pool_allocation->pool_info);
        }
      }
    }
    auto selected_pool = SelectPlacementPool(buf_info->pool_candidates, pool_offset_candidates);
    pool_allocations.Set(
        buf_info, PoolAllocation(selected_pool, Integer(pool_offset_candidates[selected_pool])));
  }
  return pool_allocations;
}

TVM_REGISTER_GLOBAL("tir.usmp.algo.greedy_by_size")
    .set_body_typed([](Array<BufferInfo> buffer_info_arr) {
      return GreedyBySize(buffer_info_arr);
    });

}  // namespace algo
}  // namespace usmp
}  // namespace tir
}  // namespace tvm
