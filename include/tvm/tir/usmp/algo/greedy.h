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
 * \file include/tvm/tir/usmp/algo/greedy.h
 * \brief This header file contains helper methods used in greedy algorithms
 * for planning  memory for USMP
 */
#pragma once
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace tir {
namespace usmp {
namespace algo {

/*!
 * \brief This is the base class for Greedy Algorithms where the sorting
 * is specialized in the extended classes based on the greedy criteria.
 */
class GreedyBase {
 public:
  GreedyBase() {}
  /*!
   * \brief This function should be implemented by the extended classes to sort the BufferInfo
   * objects based on a criteria and then calling PostSortAllocation.
   */
  virtual Map<BufferInfo, PoolAllocation> PlanMemory(const Array<BufferInfo>& buffer_info_arr) = 0;

 protected:
  /*!
   * \brief Rounds up the offset to satisfy the alignement requirement
   */
  size_t round_up_to_byte_alignment(const size_t& non_aligned_byte_offset,
                                    const int& byte_alignment);

  /*!
   * \brief A helper function check whether a offset is valid given the constraints
   */
  bool IsValidPlacement(const PoolInfo& candidate_pool, const size_t& next_offset,
                        const size_t& size_bytes);

  /*!
   * \brief Selects a pool for placement in the given set of ordered pool candidates
   */
  PoolInfo SelectPlacementPool(
      const BufferInfo& buf_info,
      const std::unordered_map<PoolInfo, size_t, ObjectPtrHash, ObjectPtrEqual>& pool_offsets);

  /*!
   * \brief This is the base allocation function that works on sorted BufferInfo objects based
   * on the greedy heuristic. The sorting algorithm has to be called before calling this.
   */
  Map<BufferInfo, PoolAllocation> PostSortAllocation(
      const std::vector<BufferInfo>& buffer_info_vec);
};

}  // namespace algo
}  // namespace usmp
}  // namespace tir
}  // namespace tvm
