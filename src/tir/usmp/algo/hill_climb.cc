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

#include <algorithm>
#include <numeric>
#include <sstream>

namespace tvm {
namespace tir {
namespace usmp {
namespace algo {
/*!
 * \brief Rounds up the offset to satisfy the alignement requirement
 */
static size_t round_up_to_byte_alignment(const size_t& non_aligned_byte_offset,
                                         const int& byte_alignment) {
  return ((non_aligned_byte_offset + byte_alignment - 1) / byte_alignment) * byte_alignment;
}

/*!
 * \brief A helper function check whether a offset is valid given the constraints
 */
static bool IsValidPlacement(const PoolInfo& candidate_pool, const size_t& next_offset,
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

/*!
 * \brief Selects a pool for placement in the given set of ordered pool candidates
 */
static PoolInfo SelectPlacementPool(
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

struct _ptr_hash {
  template <typename T>
  size_t operator()(const T& a) const {
    return std::hash<T>()(a);
  }
};

using alloc_map_t = std::unordered_map<const BufferInfoNode*, PoolAllocation, _ptr_hash>;

static void sort_vector(std::vector<BufferInfo>* buffer_info_vec) {
  std::sort(buffer_info_vec->begin(), buffer_info_vec->end(),
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
              return a->size_bytes->value > b->size_bytes->value;
            });
}

/*
 * Modified version of greedy allocation from greedy_by_size.cc
 */
static void greedy(std::vector<BufferInfo>* buffer_info_vec, alloc_map_t* pool_allocations) {
  for (const auto& buf_info : *buffer_info_vec) {
    std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual> pool_offset_candidates;
    for (const auto& pool_info : buf_info->pool_candidates) {
      if (algo::IsValidPlacement(pool_info, 0, buf_info->size_bytes->value)) {
        pool_offset_candidates[pool_info] = 0;
      }
    }

    std::vector<const BufferInfoNode*> buf_conf;
    for (const auto& conflict_buf_info_obj : buf_info->conflicts) {
      const BufferInfoNode* conflict_buf_info = conflict_buf_info_obj.as<BufferInfoNode>();
      if (pool_allocations->end() != pool_allocations->find(conflict_buf_info)) {
        buf_conf.push_back(conflict_buf_info);
      }
    }

    // extra sorting for pool offsets
    std::sort(buf_conf.begin(), buf_conf.end(), [&pool_allocations](const auto* a, const auto* b) {
      return pool_allocations->operator[](a)->byte_offset->value <
             pool_allocations->operator[](b)->byte_offset->value;
    });

    for (const auto* conflict_buf_info : buf_conf) {
      size_t next_offset = 0;
      auto pool_allocation = pool_allocations->operator[](conflict_buf_info);
      next_offset = pool_allocation->byte_offset + conflict_buf_info->size_bytes;
      next_offset = round_up_to_byte_alignment(next_offset, conflict_buf_info->alignment->value);
      if (!pool_offset_candidates.count(pool_allocation->pool_info)) {
        continue;
      }
      if (IsValidPlacement(pool_allocation->pool_info, next_offset, buf_info->size_bytes->value)) {
        if (next_offset > pool_offset_candidates[pool_allocation->pool_info] &&
            pool_offset_candidates[pool_allocation->pool_info] +
                    static_cast<size_t>(buf_info->size_bytes) >
                static_cast<size_t>(pool_allocation->byte_offset)) {
          pool_offset_candidates[pool_allocation->pool_info] = next_offset;
        }
      } else {
        pool_offset_candidates.erase(pool_allocation->pool_info);
      }
    }
    auto selected_pool = algo::SelectPlacementPool(buf_info, pool_offset_candidates);
    pool_allocations->operator[](buf_info.as<BufferInfoNode>()) =
        PoolAllocation(selected_pool, Integer(pool_offset_candidates[selected_pool]));
  }
}

/*
 * Finds highes allocated memory address for each pool
 */
static std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual> find_highest(
    alloc_map_t* pool_allocations) {
  std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual> max_pool_size;
  for (const auto it : *pool_allocations) {
    const BufferInfoNode* buf = it.first;
    const PoolAllocation& pa = it.second;
    size_t high_sz = pa->byte_offset + buf->size_bytes;
    if (max_pool_size[pa->pool_info] <= high_sz) {
      max_pool_size[pa->pool_info] = high_sz;
    }
  }
  return max_pool_size;
}

/*
 * Simulated annealing / Hill climb
 *
 * Works by continiously invoking modified 'greedy-by-size' allocation
 * assessing the result and introduce permutations which hopefully
 * will led to more 'compact' memory allocation.
 */
Map<BufferInfo, PoolAllocation> HillClimb(const Array<BufferInfo>& buffer_info_arr,
                                          const Integer& desired_bytes) {
// rand_r does not exist on Windows platform
#if defined(__linux__) || defined(__ANDROID__)
  unsigned int _seedp = 0;
#define rnd_func() rand_r(&_seedp)
#else
#define rnd_func() rand()
#endif

  std::vector<BufferInfo> buffer_info_vec;
  for (const auto& buffer_info : buffer_info_arr) {
    ICHECK(buffer_info->pool_candidates.size())
        << "Cannot process buffer \"" << buffer_info->name_hint << "\" with no pool candidates";
    buffer_info_vec.push_back(std::move(buffer_info));
  }

  sort_vector(&buffer_info_vec);

  // populate positional index map
  std::unordered_map<const BufferInfoNode*, int, _ptr_hash> _pos_map;
  for (size_t index = 0; index < buffer_info_vec.size(); ++index) {
    _pos_map[buffer_info_vec[index].as<BufferInfoNode>()] = index;
  }

  // size_t first_attempt_size = 0;
  size_t total_size = 0;
  int attempts = 0;
  // int successful_iteration = 0;

  int swap_i1 = -1;
  int swap_i2 = -1;
  size_t desired_bytes_ = desired_bytes;
  constexpr auto _max_attempts = 500;
  alloc_map_t rollback_pool_allocations;
  alloc_map_t result_pool_allocations;
  alloc_map_t pool_allocations;

  auto swap_buffers = [&buffer_info_vec, &_pos_map](int i1, int i2) {
    if (i1 == i2) return;
    auto b1 = buffer_info_vec[i1];
    auto b2 = buffer_info_vec[i2];
    buffer_info_vec[i1] = b2;
    buffer_info_vec[i2] = b1;

    _pos_map[b1.as<BufferInfoNode>()] = i2;
    _pos_map[b2.as<BufferInfoNode>()] = i1;
  };

  auto _pos = [&_pos_map](const auto* e) {
    auto it = _pos_map.find(e);
    if (it != _pos_map.end()) {
      return it->second;
    }
    LOG(FATAL) << "not indexed";
    return -1;
  };

  for (; attempts < _max_attempts; ++attempts) {
    rollback_pool_allocations = std::move(pool_allocations);
    greedy(&buffer_info_vec, &pool_allocations);

    // estimate result buffers
    auto max_pool_size = find_highest(&pool_allocations);

    // calculate summary
    size_t total = 0;
    for (const auto& el : max_pool_size) {
      total += el.second;
    }
    // accept/reject result heuristic
    if (!total_size ||
        (total_size > total ||
         rnd_func() % 100 < static_cast<int>(300 * (total - total_size) / total / attempts))) {
      // remember winning combination
      result_pool_allocations = pool_allocations;
      total_size = total;

      // reached desired size
      if (total_size <= desired_bytes_) {
        break;
      }

    } else {
      // rollback
      swap_buffers(swap_i2, swap_i1);
      pool_allocations = std::move(rollback_pool_allocations);
      max_pool_size = find_highest(&pool_allocations);
    }

    std::vector<const BufferInfoNode*> max_pool_buf;

    for (const auto& it : pool_allocations) {
      const auto* buf = it.first;
      const auto pa = it.second;
      size_t high_sz = pa->byte_offset + buf->size_bytes;
      if (max_pool_size[pa->pool_info] == high_sz) {
        max_pool_buf.push_back(buf);
      }
    }

    // pick highest
    const BufferInfoNode* suspect = max_pool_buf[rand() % max_pool_buf.size()];
    PoolAllocation suspect_pa = pool_allocations[suspect];

    std::unordered_map<int, const BufferInfoNode*, _ptr_hash> first_level_set;
    std::unordered_map<int, const BufferInfoNode*, _ptr_hash> second_level_set;

    auto suspect_pos = _pos(suspect);
    for (const auto& c1 : suspect->conflicts) {
      const auto* c1_buf = c1.as<BufferInfoNode>();
      int c1_pos = _pos(c1_buf);
      if (suspect_pos > c1_pos) {
        first_level_set[c1_pos] = c1_buf;
      }
      int c2_pos = -1;
      for (const auto& c2 : c1_buf->conflicts) {
        const auto c2_buf = c2.as<BufferInfoNode>();
        if (c1_pos > (c2_pos = _pos(c2_buf))) {
          second_level_set[c2_pos] = c2_buf;
        }
      }
    }

    std::vector<const BufferInfoNode*> first_level;
    for (const auto& i : first_level_set) {
      first_level.push_back(i.second);
    }
    std::vector<const BufferInfoNode*> second_level;
    for (const auto& i : second_level_set) {
      second_level.push_back(i.second);
    }

    if (!(first_level.size() + second_level.size())) {
      continue;
    }

    // pick the buffers
    const BufferInfoNode* swap_buf2 =
        second_level.size() && (!first_level.size() || (rnd_func() % 100 > 30))
            ? second_level[rand() % second_level.size()]
            : first_level[rand() % first_level.size()];
    const BufferInfoNode* swap_buf1 =
        second_level.size() && (!first_level.size() || (rnd_func() % 100 > 30))
            ? second_level[rand() % second_level.size()]
            : first_level[rand() % first_level.size()];

    if (swap_buf1 == swap_buf2) {
      continue;
    }

    swap_i1 = _pos(swap_buf1);
    swap_i2 = _pos(swap_buf2);
    // do swap
    swap_buffers(swap_i1, swap_i2);
  }

  Map<BufferInfo, PoolAllocation> result;
  for (auto it : pool_allocations) {
    result.Set(GetRef<BufferInfo>(it.first), it.second);
  }
  return result;
}

TVM_REGISTER_GLOBAL("tir.usmp.algo.hill_climb")
    .set_body_typed([](Array<BufferInfo> buffer_info_arr, Integer memory_pressure) {
      return HillClimb(buffer_info_arr, memory_pressure);
    });

}  // namespace algo
}  // namespace usmp
}  // namespace tir
}  // namespace tvm
