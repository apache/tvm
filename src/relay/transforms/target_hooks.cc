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
 * \file target_hooks.cc
 * \brief Relay passes for processing Target Hooks which have been registered on functions within
 * the IRModule
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace transform {

class TargetHookVisitor : public tvm::relay::MixedModeVisitor {
  /*! \brief Collected pass list for all nodes */
  std::vector<Pass> pass_list_;
  /*! \brief Attribute map for all registered targets */
  TargetKindAttrMap<Pass> target_attr_map_;

 public:
  TargetHookVisitor() : target_attr_map_(tvm::TargetKind::GetAttrMap<Pass>("RelayToTIR")) {}

  std::vector<Pass> Visit(const IRModule& ir_mod) {
    for (const auto& it : ir_mod->functions) {
      const BaseFunc& base_func = it.second;
      VisitExpr(base_func);
    }
    return pass_list_;
  }

  void VisitExpr_(const CallNode* call) override {
    // Descend the call tree
    for (auto arg : call->args) {
      VisitExpr(arg);
    }

    if (const FunctionNode* func = call->op.as<FunctionNode>()) {
      if (!func->GetAttr<String>(attr::kCompiler).defined()) {
        return;
      }
      String code_gen_name = func->GetAttr<String>(attr::kCompiler).value();
      Optional<TargetKind> target_kind = tvm::TargetKind::Get(code_gen_name);
      if (!target_kind || !target_attr_map_.count(target_kind.value())) {
        return;
      }
      Pass custom_target_pass = target_attr_map_[target_kind.value()];
      if (std::find(pass_list_.begin(), pass_list_.end(), custom_target_pass) == pass_list_.end()) {
        pass_list_.push_back(custom_target_pass);
      }
    }
  }
};

Pass RelayToTIRTargetHook() {
  auto pass_func = [=](IRModule mod, const PassContext& pass_ctx) {
    auto target_hook_visitor = TargetHookVisitor();
    std::vector<Pass> pass_list = target_hook_visitor.Visit(mod);
    Sequential run_hooks(pass_list);

    return run_hooks(mod);
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "RelayToTIRTargetHook", {});
}

}  // namespace transform
}  // namespace relay
}  // namespace tvm
