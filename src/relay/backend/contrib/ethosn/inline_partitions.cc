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
 * \file src/relay/backend/contrib/ethosn/inline_partitions.cc
 * \brief A pass to inline NPU partitions that are not considered compute
 * intensive.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

#include "../../../transforms/compiler_function_utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

class IsComputeIntensivePartition : MixedModeVisitor {
 public:
  /*!
   * \brief Check if the partitioned function is compute
   * intensive. If it has not multiply-accumulate operations
   * it is not considered compute intensive.
   *
   * \param expr The partitioned function to check.
   */
  bool CheckSubgraph(const Expr& expr) {
    is_compute_intensive = false;
    VisitExpr(expr);
    return is_compute_intensive;
  }

  /*!
   * \brief Visit the call nodes of a partitioned function
   * and check if operators or composite functions make the
   * partitioned function compute intensive.
   *
   * \param op The call node to check.
   */
  void VisitExpr_(const CallNode* op) override {
    Call call = GetRef<Call>(op);
    std::string op_name = "";
    if (const auto* op = call->op.as<OpNode>()) {
      op_name = op->name;
    } else if (const auto* func = call->op.as<FunctionNode>()) {
      op_name = func->GetAttr<String>(attr::kComposite, "").value();
    }

    if (op_name != "") {
      if (compute_intensive_operators.find(op_name) != compute_intensive_operators.end()) {
        is_compute_intensive = true;
      }
    }
  }

 private:
  /*! \brief Whether or not the partitioned function is consdiered compute intensive. */
  bool is_compute_intensive;
  /*! \brief A set of operators considered compute intensive. */
  const std::unordered_set<std::string> compute_intensive_operators{
      "ethos-n.qnn_conv2d",     "ethos-n.qnn_conv2d_transpose",
      "ethos-n.qnn_avg_pool2d", "ethos-n.qnn_sigmoid",
      "ethos-n.qnn_fc",         "ethos-n.qnn_mean",
      "ethos-n.qnn_resize",     "nn.max_pool2d",
  };
};

/*!
 * \brief This pass checks whether functions partitioned for the NPU are considered
 * non-compute intensive. If they are not, they will be unpartitioned and passed onto
 * other backends to consider.
 *
 * A partitioned function is currently considered non-compute intensive if it contains
 * no multiply accumulate operations. Note that this is not an optimal heuristic.
 *
 * Some suggestions for future exploration:
 * - Making a better choice about large non-compute-intensive subgraphs
 *   as currently these are inlined.
 * - Allowing the user to input ops that are considered compute-intensive.
 * - Inline "small" compute intensive operations.
 */
tvm::transform::Pass InlineNonComputeIntensivePartitions() {
  runtime::TypedPackedFunc<IRModule(IRModule, tvm::transform::PassContext)> pass_func =
      [=](IRModule mod, tvm::transform::PassContext ctx) {
        auto analyzer = IsComputeIntensivePartition();
        Array<GlobalVar> gvs_to_inline;
        for (auto gv : mod->GetGlobalVars()) {
          Function func = Downcast<Function>(mod->Lookup(gv));
          auto compiler_name = func->GetAttr<String>(attr::kCompiler);
          if (compiler_name.defined() && compiler_name == "ethos-n") {
            if (!analyzer.CheckSubgraph(func->body)) {
              gvs_to_inline.push_back(gv);
            }
          }
        }
        return relay::transform::InlineCompilerFunctionsBoundTo(gvs_to_inline)(mod);
      };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "relay.backend.contrib.ethos-n.InlineNonComputeIntensivePartitions", {});
}

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethos-n.InlineNonComputeIntensivePartitions")
    .set_body_typed(InlineNonComputeIntensivePartitions);

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
