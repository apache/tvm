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

namespace {

/*!
 * \brief A pass extracted from a target kind's "RelayToTIR" attribute, along with any
 * 'external codegen' Target instance with matching kind name which should be current when
 * the pass is applied.
 */
struct CustomPass {
  std::string target_kind_name;
  Pass pass;
  Optional<Target> opt_target;

  CustomPass(std::string target_kind_name, Pass pass, Optional<Target> opt_target)
      : target_kind_name(std::move(target_kind_name)),
        pass(std::move(pass)),
        opt_target(std::move(opt_target)) {}
};

/*!
 * \brief Collect all the \p CustomPasses needed according to the "Compiler" attributes on
 * inlined or global functions.
 */
class TargetHookVisitor : public MixedModeVisitor {
 public:
  TargetHookVisitor(IRModule mod, CompilationConfig config)
      : mod_(std::move(mod)),
        config_(std::move(config)),
        target_attr_map_(tvm::TargetKind::GetAttrMap<Pass>(tvm::attr::kRelayToTIR)) {}

  std::vector<CustomPass> Visit() {
    ICHECK(custom_passes_.empty());
    // To ensure the passes are run in a deterministic order we'll search for functions in
    // lexicographic order.
    std::vector<std::pair<std::string, BaseFunc>> functions;
    for (const auto& kv : mod_->functions) {
      functions.emplace_back(kv.first->name_hint, kv.second);
    }
    std::sort(functions.begin(), functions.end());
    for (const auto& kv : functions) {
      if (const auto* function_node = kv.second.as<FunctionNode>()) {
        // May be a top-level function with a "Compiler" attribute.
        MaybeAddPassForFunction(function_node);
      }
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        // May have calls to inlined "Compiler" functions in body.
        VisitExpr(GetRef<Function>(function_node));
      }
    }
    return std::move(custom_passes_);
  }

 private:
  using tvm::relay::MixedModeVisitor::VisitExpr_;

  void VisitExpr_(const LetNode* let_node) final {
    auto pre_visit = [this](const LetNode* inner_let_node) {
      this->VisitExpr(inner_let_node->var);
      this->VisitExpr(inner_let_node->value);
    };
    auto post_visit = [this](const LetNode* inner_let_node) {
      this->VisitExpr(inner_let_node->body);
      this->visit_counter_[inner_let_node] += 1;
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
  }

  void VisitExpr_(const FunctionNode* function_node) override {
    ExprVisitor::VisitExpr_(function_node);
    MaybeAddPassForFunction(function_node);
  }

  /*!
   * \brief If \p function_node has a "Compiler" attribute, checks if we should include a
   * matching custom pass. Otherwise no-op.
   */
  void MaybeAddPassForFunction(const FunctionNode* function_node) {
    Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
    if (!opt_compiler) {
      // No external codegen required.
      return;
    }
    // First cross-over: use "Compiler" attribute name as target kind.
    std::string kind_name = opt_compiler.value();
    Optional<TargetKind> opt_target_kind = tvm::TargetKind::Get(kind_name);
    if (!opt_target_kind || !target_attr_map_.count(opt_target_kind.value())) {
      // Target kind does not exist or have the "RelayToTIR" attribute, no custom pass to consider.
      return;
    }
    if (!seen_kinds_.emplace(kind_name).second) {
      // Already accounted for custom pass.
      return;
    }
    // Second (optional) cross-over: find unique Target instance in overall available targets with
    // the same kind so that it can be made available when custom pass is invoked.
    Optional<Target> opt_target = config_->FindPrimitiveTargetForKind(opt_compiler.value());
    Pass custom_target_pass = target_attr_map_[opt_target_kind.value()];
    custom_passes_.emplace_back(std::move(kind_name), std::move(custom_target_pass),
                                std::move(opt_target));
  }

  /*! \brief IRModule we are visiting. */
  IRModule mod_;
  /*! \brief All available targets. */
  CompilationConfig config_;
  /*! \brief Cached attribute map for all registered targets */
  TargetKindAttrMap<Pass> target_attr_map_;
  /*! \brief Which target kind names have already contributed to the custom passes list. */
  std::unordered_set<std::string> seen_kinds_;
  /*!
   * \brief All the custom passes to run, paired with their corresponding target instances, if any.
   */
  std::vector<CustomPass> custom_passes_;
};

}  // namespace

Pass RelayToTIRTargetHook(CompilationConfig config) {
  auto pass_func = [config = std::move(config)](IRModule mod, const PassContext& pass_ctx) {
    VLOG(1) << "RelayToTIRTargetHook before:" << std::endl << PrettyPrint(mod);
    TargetHookVisitor target_hook_visitor(mod, config);
    std::vector<CustomPass> custom_passes = target_hook_visitor.Visit();
    for (const auto& custom_pass : custom_passes) {
      if (custom_pass.opt_target.defined()) {
        VLOG(0) << "Invoking custom pass for target "
                << custom_pass.opt_target.value()->ToDebugString();
        // Push the target on the stack.
        With<Target> with_target(custom_pass.opt_target.value());
        // Invoke the pass with target in scope.
        mod = custom_pass.pass(mod);
      } else {
        // Invoke the pass.
        // Note that there may be a non-external codegen target in scope. Each custom pass
        // must be prepared to handle this, eg by creating a default target instance if the
        // current target is either null or of a generic kind such as 'cuda' or 'llvm'.
        VLOG(0) << "Invoking custom pass for target kind '" << custom_pass.target_kind_name << "'";
        mod = custom_pass.pass(mod);
      }
    }
    VLOG(1) << "RelayToTIRTargetHook after:" << std::endl << PrettyPrint(mod);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "RelayToTIRTargetHook", {});
}

}  // namespace transform
}  // namespace relay
}  // namespace tvm
