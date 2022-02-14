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
 * \file src/contrib/ethosu/cascader/plan.h
 * \brief Plan object for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PLAN_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PLAN_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph.h"
#include "tensor_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

/*! \brief Node to represent a Plan */
class PlanNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \return The TensorConfigs specified by the Plan */
  const std::vector<TensorConfig>& GetTensorConfigs() const { return tensor_configs_; }
  /*! \return The TensorConfigs which are 'open' meaning they are a Plan input/output but have
   * INTERIOR state */
  const std::vector<TensorConfig>& GetOpenConfigs() const { return open_configs_; }
  /*! \return The TensorConfig of the Plan's output tensor */
  const TensorConfig GetOutputConfig() const { return output_config_; }
  /*! \return The Parts which are covered by the Plan */
  const std::vector<Part>& GetPartGroup() const { return part_group_; }
  /*! \return The memory region in which to store interior Plan buffers */
  MemoryRegion const GetInteriorRegion() const { return interior_region_; }
  /*!
   * \return The interior memory used by the Plan in bytes.
   * \note The interior memory usage is defined as being the memory required in the interior region
   * to execute the Plan excluding input and output buffers.
   */
  int GetMemoryUsage() const { return memory_usage_; }
  /*! \return The cycles taken to execute the Plan */
  int GetCycles() const { return cycles_; }
  /*! \return Whether the Plan is 'closed' meaning it has no 'open' TensorConfigs */
  bool IsClosed() const { return open_configs_.size() == 0; }

  static constexpr const char* _type_key = "contrib.ethosu.cascader.Plan";
  TVM_DECLARE_FINAL_OBJECT_INFO(PlanNode, Object);

 protected:
  friend class Plan;

  /*! \brief The TensorConfigs specified by the Plan */
  std::vector<TensorConfig> tensor_configs_;
  /*! \brief The TensorConfigs which are 'open' meaning they are a Plan input/output but have
   * INTERIOR state */
  std::vector<TensorConfig> open_configs_;
  /*! \brief The TensorConfig of the Plan's output tensor */
  TensorConfig output_config_;
  /*! \brief The Parts which are covered by the Plan */
  std::vector<Part> part_group_;
  /*! \brief The memory region in which to store interior Plan buffers */
  MemoryRegion interior_region_;
  /*! \brief The interior memory used by the Plan in bytes */
  int memory_usage_;
  /*! \brief The cycles taken to execute the Plan */
  int cycles_;
};

/*!
 * \brief A class which describes how to schedule a subgraph of Parts together.
 * \note A Plan takes the form of a subgraph of connected Parts (recorded in part_group) with
 * TensorConfigs for all of the required Tensors (recorded in tensor_configs). This information can
 * be used to produce a Tensor Expression schedule with inter-operator scheduling. A Plan is
 * necessarily single-output such that all non-output Parts are 'computed_at'ed the scope of the
 * output Part. This is what achieves the technique referred to as 'cascading'. A Plan also has an
 * interior memory region which specifies the region of memory into which all the Plans intermediate
 * buffers should be allocated.
 *
 * Additionally, a Plan contains some other information used during the Plan generation and
 * selection algorithms. Both the memory and cycles required to run the Plan are accounted for so
 * that Plans can be ranked and Pareto-culled on these metrics. Furthermore, the TensorConfigs which
 * are 'open' is recorded indicating that these are valid points to merge with another Plan. A Plan
 * can only be turned into a schedule if it has no 'open' TensorConfigs - at which point the Plan is
 * said to be 'closed'.
 */
class Plan : public ObjectRef {
 public:
  Plan(const std::vector<TensorConfig>& tensor_configs,
       const std::vector<TensorConfig>& open_configs, const TensorConfig& output_config,
       const std::vector<Part>& part_group, const MemoryRegion& interior_region, int memory_usage,
       int cycles);
  /*!
   * \brief Merge two Plans which share an 'open' TensorConfig.
   * \param other The Plan to merge with.
   * \return The merged Plan.
   * \note The current Plan is referred to as the 'upper Plan' and the other Plan as the 'lower
   * Plan'. The 'open' output config of the upper Plan must be an 'open' input config of the lower
   * Plan. The Tensor referenced by these configs is the Tensor on which the two Plans will be
   * merged. The merge process does the following:
   *
   * The tensor config maps will be merged with TensorConfigs from the upper Plan taking priority.
   * The open configs will be merged with the TensorConfigs that are being merged having been
   * removed. The output config will be that of the lower Plan. The part groups will be merged. The
   * interior region is necessarily the same for both the upper and lower Plan. The cycles and
   * memory usage will be summed.
   */
  Plan Merge(const Plan& other) const;

  TVM_DEFINE_OBJECT_REF_METHODS(Plan, ObjectRef, PlanNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

// Hash functions TensorConfig and Part sets
namespace std {

using TensorConfigSet = std::vector<::tvm::contrib::ethosu::cascader::TensorConfig>;
using PartSet = std::vector<::tvm::contrib::ethosu::cascader::Part>;

template <>
struct hash<TensorConfigSet> {
  std::size_t operator()(const TensorConfigSet& tensor_config_set) const {
    size_t seed = 0;
    for (const auto& tensor_config : tensor_config_set) {
      seed ^= hash<::tvm::contrib::ethosu::cascader::TensorConfig>()(tensor_config);
    }
    return seed;
  }
};

template <>
struct equal_to<TensorConfigSet> {
  bool operator()(const TensorConfigSet& lhs, const TensorConfigSet& rhs) const {
    std::unordered_set<::tvm::contrib::ethosu::cascader::TensorConfig> lh_set(lhs.begin(),
                                                                              lhs.end());
    std::unordered_set<::tvm::contrib::ethosu::cascader::TensorConfig> rh_set(rhs.begin(),
                                                                              rhs.end());
    return lh_set == rh_set;
  }
};

template <>
struct hash<PartSet> {
  std::size_t operator()(const PartSet& part_set) const {
    size_t seed = 0;
    for (const auto& part : part_set) {
      seed ^= tvm::runtime::ObjectHash()(part);
    }
    return seed;
  }
};

template <>
struct equal_to<PartSet> {
  bool operator()(const PartSet& lhs, const PartSet& rhs) const { return lhs == rhs; }
};

}  // namespace std

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PLAN_H_
