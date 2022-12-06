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
 * \file src/relay/collage/mock_cost_estimator.cc
 * \brief A mock CostEstimator to support unit tests.
 */

#ifndef TVM_RELAY_COLLAGE_MOCK_COST_ESTIMATOR_H_
#define TVM_RELAY_COLLAGE_MOCK_COST_ESTIMATOR_H_

#include <tvm/relay/function.h>

#include "./cost.h"
#include "./cost_estimator.h"

namespace tvm {
namespace relay {
namespace collage {

// Clang (15.0.3, at least) validly complains about `@main`, but it invalidly
// complains even about `\c @main`.
#if __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif

/*!
 * \brief A mock cost estimator which can determine the cost of a candidate based on both
 * the candidate's target and the number of operator calls inside it.
 *
 * The help unit tests the estimator also ICHECK fails if:
 *  - the module has inlined "Compiler" functions
 *  - @main has non-tensor arguments (eg a tuple)
 *  - more than the given number of candidate modules are measured
 *
 * To support unit testing only.
 */
class MockCostEstimatorNode : public CostEstimatorNode {
 public:
  Cost Estimate(const IRModule& mod, const Target& target) const override;

  static constexpr const char* _type_key = "relay.collage.MockCostEstimator";
  TVM_DECLARE_FINAL_OBJECT_INFO(MockCostEstimatorNode, CostEstimatorNode);

 protected:
  /*!
   * \brief Map from target kind name to assumed baseline cost (in integer seconds) for all
   * operator calls.
   */
  Map<String, Integer> target_costs_;

  /*!
   * \brief If non-zero, the maximum number of distinct modules which may be estimated.
   */
  Integer max_estimates_;

  /*! \brief Number of calls to Estimate. */
  mutable size_t num_estimates_ = 0;

  friend class MockCostEstimator;
};
#if __clang__
#pragma clang diagnostic pop
#endif

class MockCostEstimator : public CostEstimator {
 public:
  explicit MockCostEstimator(Map<String, Integer> target_costs, Integer max_estimates = 0);

  TVM_DEFINE_OBJECT_REF_METHODS(MockCostEstimator, CostEstimator, MockCostEstimatorNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_MOCK_COST_ESTIMATOR_H_
