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

#include <math.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {
namespace collage {

TVM_REGISTER_OBJECT_TYPE(CostEstimatorNode);
TVM_REGISTER_OBJECT_TYPE(MockEstimatorNode);

CostEstimator::CostEstimator() {
  auto node = make_object<CostEstimatorNode>();
  data_ = std::move(node);
}

Cost CostEstimatorNode::Estimate(const IRModule& mod, const Target& target,
                                 bool needs_tvm_turning) const {
  static const runtime::PackedFunc* estimate_seconds =
      runtime::Registry::Get("tvm.relay.collage.estimate_seconds");
  ICHECK(estimate_seconds);
  const double value = (*estimate_seconds)(mod, target, needs_tvm_turning);
  if (std::isinf(value)) {
    return Cost::Invalid();
  } else if (std::isnan(value)) {
    return Cost::Unknown();
  } else {
    return Cost::Value(value);
  }
}

/*!
 * \brief Visitor to accumulate the costs of all calls to operators in an expression.
 */
class MockEstimationVisitor : private ExprVisitor {
 public:
  MockEstimationVisitor(double op_cost, double fusion_benefit)
      : op_cost_(op_cost), fusion_benefit_(fusion_benefit) {}

  double EstimateCost(const Expr& body) {
    this->VisitExpr(body);
    return cost_;
  }

 private:
  /*! \brief The assumed baseline cost of each operator call. */
  double op_cost_;
  /*!
   * \brief The factor by which each operator call cost is to be changed for every other
   * operator call in the same group.
   */
  double fusion_benefit_;
  /*! \brief The number of operator calls seen so far. */
  size_t num_ops_ = 0;
  /*! \brief Accumulate overall cost. */
  double cost_ = 0.0;

  void VisitExpr_(const CallNode* call_node) final {
    if (call_node->op->IsInstance<OpNode>()) {
      cost_ += op_cost_ * pow(fusion_benefit_, num_ops_);
      num_ops_++;
    }
    ExprVisitor::VisitExpr_(call_node);
  }

  void VisitExpr_(const FunctionNode* function_node) final {
    // No "Compiler" functions can be inlined.
    ICHECK(!function_node->GetAttr<String>(attr::kCompiler).defined());
    ExprVisitor::VisitExpr_(function_node);
  }
};

Cost MockEstimatorNode::Estimate(const IRModule& mod, const Target& target,
                                 bool needs_tvm_tuning) const {
  double op_cost = static_cast<double>(target_costs_.at(target->kind->name)->value);
  double cost = 0.0;
  for (const auto& kv : mod->functions) {
    if (const auto* function_node = kv.second.as<FunctionNode>()) {
      auto function = GetRef<Function>(function_node);
      if (kv.first->name_hint == "main") {
        // Only tensor args are allowed to main.
        for (const auto& param : function->params) {
          ICHECK(param->type_annotation->IsInstance<TensorTypeNode>());
        }
      }
      cost += MockEstimationVisitor(op_cost, /*fusion_benefit=*/0.9).EstimateCost(function->body);
    }
  }
  return Cost::Value(cost);
}

MockEstimator::MockEstimator(Map<String, Integer> target_costs) {
  auto node = make_object<MockEstimatorNode>();
  node->target_costs_ = std::move(target_costs);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("relay.collage.CostEstimator").set_body_typed([]() { return CostEstimator(); });

TVM_REGISTER_GLOBAL("relay.collage.MockEstimator")
    .set_body_typed([](Map<String, Integer> target_costs) {
      return MockEstimator(std::move(target_costs));
    });

}  // namespace collage
}  // namespace relay
}  // namespace tvm
