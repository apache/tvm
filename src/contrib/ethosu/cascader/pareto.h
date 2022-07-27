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
 * \file src/contrib/ethosu/cascader/pareto.h
 * \brief Pareto optimisation functions for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PARETO_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PARETO_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>

#include <algorithm>
#include <array>
#include <vector>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class Plan;
class MemoryRegion;
class Proposal;

/*!
 * \brief Determine the Pareto optimal points.
 * \param costs The points as a vector of N-dimensional costs.
 * \return A vector that is true where a point is Pareto optimal and false otherwise.
 */
template <int N>
std::vector<bool> GetParetoFrontier(const std::vector<std::array<float, N>>& costs);

/*!
 * \brief Evenly sample items from a vector to reduce its size.
 * \param vec The vector to thin.
 * \param max_size The maximum size of the thinned vector.
 * \return The thinned vector.
 */
template <class T>
std::vector<T> ThinVector(const std::vector<T>& vec, size_t max_size);

/*!
 * \brief Cull plans which are not Pareto optimal then thin them down.
 * \param plans The plans to apply the Pareto culling to.
 * \param max_plans The maximum number of plans after the culling.
 * \param disable_pareto_metric Whether to only select from Pareto frontier or not.
 * \return The culled plans.
 * \note Plan Pareto-optimality is determined based upon a Plan's memory_usage
 * and cycles.
 */
std::vector<Plan> ParetoCullPlans(std::vector<Plan> plans, size_t max_plans,
                                  bool disable_pareto_metric);

std::vector<Proposal> ParetoCullProposals(std::vector<Proposal> proposals, size_t max_proposals,
                                          bool disable_pareto_metric);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PARETO_H_
