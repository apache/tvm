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
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {
namespace usmp {
namespace algo {

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
                return a->size_bytes > b->size_bytes;
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
