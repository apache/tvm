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

/*
 * \file src/relay/transforms/calibrate_partition_graph.cc
 *
 * \brief Partition an input function into multiple functions according based
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <unordered_set>

namespace tvm {
namespace relay {

namespace calibrate_partition {

/*
*/

IRModule CalibratePartition(IRModule module) {
  class OutputCollector : public ExprRewriter {
   public:
    OutputCollector() = default;

    Expr Rewrite_(const CallNode* call, const Expr& post) final {
      if (call->op->IsInstance<GlobalVarNode>()) {
        LOG(INFO) << call->op;
        auto var = Downcast<GlobalVar>(call->op);
        output_map.Set(var, post);
      }
      return post;
    }

    Map<GlobalVar, Expr> GetOutputMap() {
      return output_map;
    }

   private:
    Map<GlobalVar, Expr> output_map; 

  };

  auto glob_funcs = module->functions;
  // module is mutable, hence, we make a copy of it.
  module.CopyOnWrite();
  for (const auto& pair : glob_funcs) {
    if (auto* fn = pair.second.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      // Collect the output
      OutputCollector output_collector;
      auto body = PostOrderRewrite(func->body, &output_collector);
      auto output_map = output_collector.GetOutputMap();
      for (const auto& kv : output_map) {
        LOG(INFO) << kv.first << " " << kv.second;
      }
      func = Function(func->params, body, func->ret_type, func->type_params, func->attrs);
      // Reset the compiler attribute to null
      func = WithAttr(std::move(func), attr::kCompiler, NullValue<ObjectRef>());
      module->Update(pair.first, func);
    }
  }
  return module;
}

}  // namespace calibrate_partition

namespace transform {

Pass CalibratePartitionGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> calib_func = [=](IRModule m,
                                                                             PassContext pc) {
    return calibrate_partition::CalibratePartition(m);
  };

  auto partition_pass = CreateModulePass(calib_func, 0, "CalibratePartitionGraph", {});
  return Sequential({partition_pass, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.CalibratePartitionGraph").set_body_typed(transform::CalibratePartitionGraph);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
