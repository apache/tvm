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
 * \file src/relay/collage/custom_cost_estimator.cc
 * \brief A custom CostEstimator to support target-specific cost functions.
 */

#ifndef TVM_RELAY_COLLAGE_CUSTOM_COST_ESTIMATOR_H_
#define TVM_RELAY_COLLAGE_CUSTOM_COST_ESTIMATOR_H_

#include <tvm/relay/function.h>

#include "./cost.h"
#include "./cost_estimator.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief A cost estimator that uses a target-specific cost function.
 */
class CustomCostEstimatorNode : public CostEstimatorNode {
 public:
  Cost Estimate(const IRModule& mod, const Target& target) const override;

  static constexpr const char* _type_key = "relay.collage.CustomCostEstimator";
  TVM_DECLARE_FINAL_OBJECT_INFO(CustomCostEstimatorNode, CostEstimatorNode);

 protected:
  /*!
   * \brief Python implemented cost function name.
   */
  String py_fn_estimator_;

  friend class CustomCostEstimator;
};

class CustomCostEstimator : public CostEstimator {
 public:
  explicit CustomCostEstimator(String py_fn_estimator);

  TVM_DEFINE_OBJECT_REF_METHODS(CustomCostEstimator, CostEstimator, CustomCostEstimatorNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_CUSTOM_COST_ESTIMATOR_H_
