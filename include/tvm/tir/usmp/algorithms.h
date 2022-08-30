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
 * \file tir/usmp/algorithms.h
 * \brief The memory planning algorithm for USMP
 */

#ifndef TVM_TIR_USMP_ALGORITHMS_H_
#define TVM_TIR_USMP_ALGORITHMS_H_

#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {
namespace usmp {
namespace algo {

/*!
 * \brief The Greedy-by-Size algorithm to plan memory
 *
 * This will perform a greedy algorithm in deciding the offsets
 * within provided Pools, using the size of the buffer.
 *
 * \return A Map of BufferInfo objects and their associated PoolAllocation
 */
Map<BufferInfo, PoolAllocation> GreedyBySize(const Array<BufferInfo>& buffer_info_arr,
                                             const Integer& memory_pressure);

/*!
 * \brief The Greedy-by-Conflicts algorithm to plan memory
 *
 * This will perform a greedy algorithm in deciding the offsets
 * within provided Pools, using the number of liveness conflicts of the buffer.
 *
 * \return A Map of BufferInfo objects and their associated PoolAllocation
 */
Map<BufferInfo, PoolAllocation> GreedyByConflicts(const Array<BufferInfo>& buffer_info_arr,
                                                  const Integer& memory_pressure);
/*!
 *\brief The Hill-Climb algoritm to plan memory
 *
 * This will perform an attempt to utilize probabalistic approach to memory
 * allocation. Typically better than greedy family, but quite slow due to large
 * number of iterations.
 *
 * \return A Map of BufferInfo objects and their associated PoolAllocation
 */
Map<BufferInfo, PoolAllocation> HillClimb(const Array<BufferInfo>& buffer_info_arr,
                                          const Integer& memory_pressure);

/*!
 * \brief The Hill-Climb algorithm to plan memory
 *
 * This will perform a hill climbing algorithm in deciding the offsets
 * within provided Pools.
 *
 * \return A Map of BufferInfo objects and their associated PoolAllocation
 */
Map<BufferInfo, PoolAllocation> HillClimb(const Array<BufferInfo>& buffer_info_arr,
                                          const Integer& memory_pressure);

}  // namespace algo
}  // namespace usmp
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_USMP_ALGORITHMS_H_
