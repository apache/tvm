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

#include "./cost_estimator.h"

#include <cmath>

namespace tvm {
namespace relay {
namespace collage {

TVM_REGISTER_OBJECT_TYPE(CostEstimatorNode);

CostEstimator::CostEstimator() {
  auto node = make_object<CostEstimatorNode>();
  data_ = std::move(node);
}

Cost CostEstimatorNode::Estimate(const IRModule& mod, const Target& target) const {
  // TODO(mbs): Eventually should be abstract. For now bounce to the Python local impl.
  static const runtime::PackedFunc* estimate_seconds =
      runtime::Registry::Get("tvm.relay.collage.estimate_seconds");
  ICHECK(estimate_seconds);
  const double value = (*estimate_seconds)(mod, target);
  if (std::isinf(value)) {
    return Cost::Invalid();
  } else if (std::isnan(value)) {
    return Cost::Unknown();
  } else {
    return Cost::Value(value);
  }
}

TVM_REGISTER_GLOBAL("relay.collage.CostEstimator").set_body_typed([]() { return CostEstimator(); });

}  // namespace collage
}  // namespace relay
}  // namespace tvm
