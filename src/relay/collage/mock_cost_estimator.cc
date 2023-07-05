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

#include "./mock_cost_estimator.h"

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {
namespace collage {

TVM_REGISTER_OBJECT_TYPE(MockCostEstimatorNode);

namespace {

/*!
 * \brief Visitor to accumulate the costs of all calls to operators in an expression.
 */
class MockEstimationVisitor : private ExprVisitor {
 public:
  MockEstimationVisitor(double op_cost, double fusion_benefit)
      : op_cost_(op_cost), fusion_benefit_(fusion_benefit) {}

  double EstimateCost(const Expr& body) {
    VisitExpr(body);
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
      // Account for number of ops seens os far.
      cost_ += op_cost_ * pow(fusion_benefit_, static_cast<double>(num_ops_));
      num_ops_++;
    }
    ExprVisitor::VisitExpr_(call_node);
  }

  void VisitExpr_(const FunctionNode* function_node) final {
    // No "Compiler" functions can be inlined.
    ICHECK(!function_node->GetAttr<String>(attr::kCompiler).defined())
        << "All Compiler functions should have been outlined when preparing to estimate costs";
    ExprVisitor::VisitExpr_(function_node);
  }
};

}  // namespace

Cost MockCostEstimatorNode::Estimate(const IRModule& mod, const Target& target) const {
  // Limit the number of estimations.
  ICHECK(max_estimates_->value == 0 || num_estimates_ < static_cast<size_t>(max_estimates_->value))
      << "At most " << max_estimates_->value
      << " non-trivial distinct candidates should have been generated.";
  ++num_estimates_;
  double op_cost = static_cast<double>(target_costs_.at(target->kind->name)->value);
  double cost = 0.0;
  for (const auto& kv : mod->functions) {
    if (const auto* function = kv.second.as<FunctionNode>()) {
      if (kv.first->name_hint == "main") {
        // Only tensor args are allowed to main.
        for (const auto& param : function->params) {
          ICHECK(param->type_annotation->IsInstance<TensorTypeNode>())
              << "Any tuple-of-tensor arguments should have been eta-exanded when preparing to "
                 "estimate costs";
        }
      }
      cost += MockEstimationVisitor(op_cost, /*fusion_benefit=*/0.9).EstimateCost(function->body);
    }
  }
  return Cost::Value(cost);
}

MockCostEstimator::MockCostEstimator(Map<String, Integer> target_costs, Integer max_estimates) {
  auto node = make_object<MockCostEstimatorNode>();
  node->target_costs_ = std::move(target_costs);
  node->max_estimates_ = std::move(max_estimates);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("relay.collage.MockCostEstimator")
    .set_body_typed([](Map<String, Integer> target_costs, Integer max_estimates) {
      return MockCostEstimator(std::move(target_costs), std::move(max_estimates));
    });

}  // namespace collage
}  // namespace relay
}  // namespace tvm
