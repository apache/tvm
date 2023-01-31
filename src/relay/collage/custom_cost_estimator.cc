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
 * \brief A custom CostEstimator to support alternative cost functions.
 */

#include "./custom_cost_estimator.h"

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {
namespace collage {

TVM_REGISTER_OBJECT_TYPE(CustomCostEstimatorNode);

Cost CustomCostEstimatorNode::Estimate(const IRModule& mod, const Target& target) const {
  static const runtime::PackedFunc* estimate_seconds = runtime::Registry::Get(py_fn_estimator_);
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

CustomCostEstimator::CustomCostEstimator(String py_fn_estimator) {
  auto node = make_object<CustomCostEstimatorNode>();
  node->py_fn_estimator_ = std::move(py_fn_estimator);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("relay.collage.CustomCostEstimator").set_body_typed([](String py_fn_estimator) {
  return CustomCostEstimator(std::move(py_fn_estimator));
});

}  // namespace collage
}  // namespace relay
}  // namespace tvm
