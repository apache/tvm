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
 * \file src/relay/collage/cost_estimator.cc
 * \brief Interface for measuring candidate partition cost.
 */

#ifndef TVM_RELAY_COLLAGE_COST_ESTIMATOR_H_
#define TVM_RELAY_COLLAGE_COST_ESTIMATOR_H_

#include <tvm/relay/function.h>

#include "./cost.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief An (abstract) estimator for the cost of executing "main" in an \p IRModule representing
 * a candidate partition, using the given target for lowering and codegen.
 *
 * Generally the implementation will compile to a \p runtime::Module (possibly on a target-specific
 * worker if cross-compilation is not available), repeatedly invoke "main" with random data until
 * measure variance is acceptable (on a target-specific worker), and return the summarized costs.
 *
 * If using a TVM native \p Target, it is possible compilation will itself invoke TVM tuning.
 *
 * TODO(mbs): Actually, currently not abstract so can get some local measurements.
 */
class CostEstimatorNode : public Object {
 public:
  /*!
   * \brief Returns the estimated cost (possibly after many many minutes of training time) of
   * running "main" in \p mod using \p target, which represents a possible partitioning of
   * some overall Relay expression.
   */
  virtual Cost Estimate(const IRModule& mod, const Target& target, bool needs_tvm_tuning) const;

  static constexpr const char* _type_key = "relay.collage.CostEstimator";
  TVM_DECLARE_BASE_OBJECT_INFO(CostEstimatorNode, Object);
};

class CostEstimator : public ObjectRef {
 public:
  CostEstimator();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CostEstimator, ObjectRef, CostEstimatorNode);
};

/*!
 * \brief A mock cost estimator which can determine the cost of a candidate based on both
 * the candidate's target and the number of operator calls inside it.
 *
 * The estimator also ICHECKs the given module has all "Compiler" functions outlined and @main
 * takes only tensor arguments (ie no tuple types).
 *
 * To support testing only.
 */
class MockEstimatorNode : public CostEstimatorNode {
 public:
  Cost Estimate(const IRModule& mod, const Target& target, bool needs_tvm_tuning) const override;

  static constexpr const char* _type_key = "relay.collage.MockEstimator";
  TVM_DECLARE_FINAL_OBJECT_INFO(MockEstimatorNode, CostEstimatorNode);

 protected:
  friend class MockEstimator;

  /*!
   * \brief Map from target kind name to assumed baseline cost (in integer seconds) for all
   * operator calls.
   */
  Map<String, Integer> target_costs_;
};

class MockEstimator : public CostEstimator {
 public:
  explicit MockEstimator(Map<String, Integer> target_costs);

  TVM_DEFINE_OBJECT_REF_METHODS(MockEstimator, CostEstimator, MockEstimatorNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_COST_ESTIMATOR_H_
