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
  virtual Cost Estimate(const IRModule& mod, const Target& target) const;

  static constexpr const char* _type_key = "relay.collage.CostEstimator";
  TVM_DECLARE_BASE_OBJECT_INFO(CostEstimatorNode, Object);
};

class CostEstimator : public ObjectRef {
 public:
  CostEstimator();
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CostEstimator, ObjectRef, CostEstimatorNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_COST_ESTIMATOR_H_
