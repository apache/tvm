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
 * \file src/contrib/ethosu/cascader/proposal_generator.h
 * \brief Algorithm to generate possible Proposals in the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PROPOSAL_GENERATOR_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PROPOSAL_GENERATOR_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class CascaderGraph;
class MemoryRegion;
class Tensor;
class Proposal;
class CascaderOptions;

using HomeMap =
    std::unordered_map<Tensor, std::vector<MemoryRegion>, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Generate Pareto optimal Proposals for a CascaderGraph.
 * \param graph The CascaderGraph to generate Proposals for.
 * \param home_map The Tensor homing map defining valid memory homes for Tensors.
 * \param options The configuration options with which to run the generator.
 * \return A vector of Pareto optimal Proposals.
 * \note This algorithm takes a top-down dynamic programming approach to determining how
 * to optimally combine Plans into Proposals. It does the following:
 *
 * First, run GenerateGraphPlans to generate the Pareto optimal Plans that cover all the
 * Part groups in the CascaderGraph.
 *
 * Solve the problem recursively, generating optimal Proposals for increasingly small
 * portions of the overall graph.
 *
 * Take the first Part in the graph:
 *   1. Find all the Plans for which the Part is both in the Plan's Part group and has the
 *      highest Part ID of any Part in the Part group (i.e. it's the 'first' Part in the
 *      group).
 *   For each Plan:
 *     2. Get the Part group covered by the Plan and subtract it from the 'total Part group'
 *        covering all the Parts. This forms a 'residual Part group'.
 *     3. Recursively, determine the optimal Proposals for the 'residual Part group' (the graph
 *        minus the Parts included in the Plan). Memoize the results.
 *     For each residual Proposal:
 *       4. Create a new Proposal by adding the current Plan to the residual Proposal.
 *   5. Pareto cull all the newly created Proposals (which all share the same Part group).
 * 6. Return the Proposals which cover all the Parts in the CascaderGraph.
 *
 */
std::vector<Proposal> GenerateProposals(const CascaderGraph& graph, const HomeMap& home_map,
                                        const CascaderOptions& options);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PROPOSAL_GENERATOR_H_
