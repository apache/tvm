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
 * \file tir/analysis/usmp/algo/greedy.cc
 * \brief This source contains greedy algorithms for planning
 * memory for USMP. There are two algorithms present here :
 * 1) greedy_by_size and 2) greedy_by_conflicts.
 *
 * greedy_by_size : this algorithm prioritizes placing the
 * largest size buffer to the given pools. The BufferInfo objects
 * are sorted based on the size and placed on each pool adhering
 * to size_hint constraint.
 *
 * greedy_by_conflicts : this algorithm prioritizes placing the
 * the most liveness conflicted buffer to the given pools. The
 * BufferInfo objects are sorted based on the number of conflicts
 * and placed on each pool adhering to size_hint constraint.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/algo/greedy.h>
#include <tvm/tir/usmp/algorithms.h>
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {
namespace usmp {
namespace algo {

/*!
 * \brief Rounds up the offset to satisfy the alignement requirement
 */
size_t GreedyBase::round_up_to_byte_alignment(const size_t& non_aligned_byte_offset,
                                              const int& byte_alignment) {
  return ((non_aligned_byte_offset + byte_alignment - 1) / byte_alignment) * byte_alignment;
}

/*!
 * \brief A helper function check whether a offset is valid given the constraints
 */
bool GreedyBase::IsValidPlacement(const PoolInfo& candidate_pool, const size_t& next_offset,
                                  const size_t& size_bytes) {
  Integer size_hint_bytes = -1;
  if (const auto* p = candidate_pool.as<WorkspacePoolInfoNode>()) {
    size_hint_bytes = p->size_hint_bytes;
  } else if (const auto* p = candidate_pool.as<ConstantPoolInfoNode>()) {
    size_hint_bytes = p->size_hint_bytes;
  } else {
    LOG(FATAL) << "Pool '" << candidate_pool->GetTypeKey() << "' is not supported";
  }

  if (size_hint_bytes == kUnrestrictedPoolSizeHint) {
    // this means pool is not bounded
    return true;
  }
  auto pool_size = static_cast<size_t>(size_hint_bytes.IntValue());
  auto max_address = next_offset + size_bytes;
  if (max_address <= pool_size) {
    return true;
  }
  return false;
}

/*!
 * \brief Selects a pool for placement in the given set of ordered pool candidates
 */
PoolInfo GreedyBase::SelectPlacementPool(
    const BufferInfo& buf_info,
    const std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual>& pool_offsets) {
  // Here the pool candidates are ordered when it is consumed by the algorithm.
  // This could be from order the user has specified. However, schedulers are
  // welcome to change the order for performance reasons.
  for (const auto& pool_info : buf_info->pool_candidates) {
    if (pool_offsets.count(pool_info)) {
      return pool_info;
    }
  }
  CHECK(false) << "TVM USMP Error: the space available in the provided pools exceeded when "
                  "trying to allocate the buffer : "
               << buf_info << "\n. Please increase the size_hints for memory pools.";
  return PoolInfo();
}

/*!
 * \brief This is the base allocation function that works on sorted BufferInfo objects based
 * on the greedy heuristic. The sorting algorithm has to be called before calling this.
 */
Map<BufferInfo, PoolAllocation> GreedyBase::PostSortAllocation(
    const std::vector<BufferInfo>& buffer_info_vec) {
  Map<BufferInfo, PoolAllocation> pool_allocations;
  for (const auto& buf_info : buffer_info_vec) {
    std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual> pool_offset_candidates;
    for (const auto& pool_info : buf_info->pool_candidates) {
      // Mark pool candidates that satisfy the size constraints.
      if (IsValidPlacement(pool_info, 0, buf_info->size_bytes->value)) {
        pool_offset_candidates[pool_info] = 0;
      }
    }

    for (const auto& conflict_buf_info_obj : buf_info->conflicts) {
      auto conflict_buf_info = Downcast<BufferInfo>(conflict_buf_info_obj);
      size_t next_offset = 0;
      // We only look at already allocated BufferInfo in-terms of conflicts.
      if (pool_allocations.count(conflict_buf_info)) {
        auto pool_allocation = pool_allocations[conflict_buf_info];
        next_offset =
            pool_allocation->byte_offset.IntValue() + conflict_buf_info->size_bytes.IntValue();
        next_offset = round_up_to_byte_alignment(next_offset, conflict_buf_info->alignment->value);
        // Checks whether the next offset in the same pool as the conflicting BufferInfo is valid.
        if (IsValidPlacement(pool_allocation->pool_info, next_offset,
                             buf_info->size_bytes->value)) {
          // There could be multiple conflicting BufferInfo in the same pool.
          // Thus, we need to make sure we pick the largest offset of them all.
          if (next_offset > pool_offset_candidates[pool_allocation->pool_info]) {
            pool_offset_candidates[pool_allocation->pool_info] = next_offset;
          }
        } else {
          pool_offset_candidates.erase(pool_allocation->pool_info);
        }
      }
    }
    auto selected_pool = SelectPlacementPool(buf_info, pool_offset_candidates);
    pool_allocations.Set(
        buf_info, PoolAllocation(selected_pool, Integer(pool_offset_candidates[selected_pool])));
  }
  return pool_allocations;
}

/*!
 * \brief This class implements Greedy by the size of BufferInfo
 * greedy algorithm. Please refer to main documentation of the file
 * for more details.
 */
class GreedySize : public GreedyBase {
 public:
  GreedySize() {}
  Map<BufferInfo, PoolAllocation> PlanMemory(const Array<BufferInfo>& buffer_info_arr) {
    std::vector<BufferInfo> buffer_info_vec;
    Map<BufferInfo, PoolAllocation> pool_allocations;
    for (const auto& buffer_info : buffer_info_arr) {
      buffer_info_vec.push_back(std::move(buffer_info));
    }
    std::sort(buffer_info_vec.begin(), buffer_info_vec.end(),
              [](const BufferInfo& a, const BufferInfo& b) {
                if (a->size_bytes->value == b->size_bytes->value) {
                  if (a->conflicts.size() == b->conflicts.size()) {
                    return std::string(a->name_hint->data) > std::string(b->name_hint->data);
                  } else {
                    return a->conflicts.size() > b->conflicts.size();
                  }
                }
                return a->size_bytes.IntValue() > b->size_bytes.IntValue();
              });
    return PostSortAllocation(buffer_info_vec);
  }
};

/*!
 * \brief This class implements Greedy by the number of conflicts of
 * BufferInfo greedy algorithm. Please refer to main documentation
 * of the file for more details.
 */
class GreedyConflicts : public GreedyBase {
 public:
  GreedyConflicts() {}
  Map<BufferInfo, PoolAllocation> PlanMemory(const Array<BufferInfo>& buffer_info_arr) {
    std::vector<BufferInfo> buffer_info_vec;
    Map<BufferInfo, PoolAllocation> pool_allocations;
    for (const auto& buffer_info : buffer_info_arr) {
      buffer_info_vec.push_back(std::move(buffer_info));
    }
    std::sort(buffer_info_vec.begin(), buffer_info_vec.end(),
              [](const BufferInfo& a, const BufferInfo& b) {
                if (a->conflicts.size() == b->conflicts.size()) {
                  if (a->size_bytes->value == b->size_bytes->value) {
                    return std::string(a->name_hint->data) > std::string(b->name_hint->data);
                  } else {
                    return a->size_bytes->value > b->size_bytes->value;
                  }
                }
                return a->conflicts.size() > b->conflicts.size();
              });
    return PostSortAllocation(buffer_info_vec);
  }
};

Map<BufferInfo, PoolAllocation> GreedyBySize(const Array<BufferInfo>& buffer_info_arr,
                                             const Integer& memory_pressure) {
  return GreedySize().PlanMemory(buffer_info_arr);
}

Map<BufferInfo, PoolAllocation> GreedyByConflicts(const Array<BufferInfo>& buffer_info_arr,
                                                  const Integer& memory_pressure) {
  return GreedyConflicts().PlanMemory(buffer_info_arr);
}

TVM_REGISTER_GLOBAL("tir.usmp.algo.greedy_by_size")
    .set_body_typed([](Array<BufferInfo> buffer_info_arr, Integer memory_pressure) {
      return GreedyBySize(buffer_info_arr, memory_pressure);
    });

TVM_REGISTER_GLOBAL("tir.usmp.algo.greedy_by_conflicts")
    .set_body_typed([](Array<BufferInfo> buffer_info_arr, Integer memory_pressure) {
      return GreedyByConflicts(buffer_info_arr, memory_pressure);
    });

}  // namespace algo
}  // namespace usmp
}  // namespace tir
}  // namespace tvm
