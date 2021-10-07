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
 * \file src/contrib/ethosu/cascader/proposal.h
 * \brief Proposal object for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PROPOSAL_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PROPOSAL_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph.h"
#include "plan.h"
#include "tensor_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

using MemoryUsageMap = std::unordered_map<MemoryRegion, int, ObjectPtrHash, ObjectPtrEqual>;
using TensorConfigMap = std::unordered_map<Tensor, TensorConfig, ObjectPtrHash, ObjectPtrEqual>;

/*! \brief Node to represent a Proposal */
class ProposalNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \return The CascaderGraph to which the Proposal applies */
  const CascaderGraph GetGraph() const { return graph_; }
  /*! \return The Parts which are covered by the Proposal */
  const std::vector<Part> GetPartGroup() const { return part_group_; }
  /*! \return The Plans used in the Proposal */
  const std::vector<Plan> GetPlans() const { return plans_; }
  /*! \return The TensorConfigs indexed by Tensor in the Proposal which aren't produced by a Plan */
  const TensorConfigMap GetInputTensorConfigs() const { return input_tensor_configs_; }
  /*! \return The MemoryRegion where cascading buffers should be homed */
  const MemoryRegion GetCascadeRegion() const { return cascade_region_; }
  /*! \return The memory required to execute the Proposal in the cascading MemoryRegion */
  const int GetMemoryUsage() const { return memory_usage_; }
  /*! \return The estimated cycles taken to execute the Proposal */
  int GetCycles() const { return cycles_; }

  static constexpr const char* _type_key = "contrib.ethosu.cascader.Proposal";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProposalNode, Object);

 protected:
  friend class Proposal;

  /*! \brief The CascaderGraph to which the Proposal applies */
  CascaderGraph graph_;
  /*! \brief The Parts which are covered by the Proposal */
  std::vector<Part> part_group_;
  /*! \brief The Plans used in the Proposal */
  std::vector<Plan> plans_;
  /*! \brief The TensorConfigs indexed by Tensor in the Proposal which aren't produced by a Plan */
  TensorConfigMap input_tensor_configs_;
  /*! \brief The MemoryRegion where cascading buffers should be homed */
  MemoryRegion cascade_region_;
  /*! \brief The memory required to execute the Proposal in the cascading MemoryRegion */
  int memory_usage_;
  /*! \brief The estimated cycles taken to execute the Proposal */
  int cycles_;
};

/*!
 * \brief A class which describes how to schedule a CascaderGraph as a series of disjoint Plans.
 */
class Proposal : public ObjectRef {
 public:
  Proposal(const CascaderGraph& graph, const std::vector<Part>& part_group,
           const std::vector<Plan>& plans, const TensorConfigMap& input_tensor_configs,
           const MemoryRegion& cascade_region, int memory_usage, int cycles);

  TVM_DEFINE_OBJECT_REF_METHODS(Proposal, ObjectRef, ProposalNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PROPOSAL_H_
