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
 * \file src/relay/transforms/fold_explicit_padding.cc
 * \brief A pass for folding explicit pads into other ops.
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <utility>
#include <vector>

#include "../op/tensor/transform.h"
#include "./pattern_fuse.h"

namespace tvm {
namespace relay {

namespace transform {

Pass AnnotatePostFuseFuncs() {
  auto pass_info = PassInfo(0, "AnnotatePostFuseFuncs", {});
  return tvm::transform::CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        // Execute the pass function and return a new module.
        IRModule updated_mod = mod->ShallowCopy();

        pass_ctx->diag_ctx = DiagnosticContext::Default(updated_mod);

        std::vector<std::pair<GlobalVar, Function> > updates;
        for (const auto& it : updated_mod->functions) {
          if (auto* func_node = it.second.as<FunctionNode>()) {
            auto func = GetRef<Function>(func_node);

            // add check from where it originate
            func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(1));

            updates.push_back({it.first, Downcast<Function>(func)});
          }
        }

        for (const auto& pair : updates) {
          updated_mod->Add(pair.first, pair.second, true);
        }

        return updated_mod;
      },
      0, "AnnotatePostFuseFuncs", {});
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotatePostFuseFuncs").set_body_typed([]() {
  return AnnotatePostFuseFuncs();
});
}  // namespace transform

}  // namespace relay
}  // namespace tvm
