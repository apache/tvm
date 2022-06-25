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
 * \file src/relay/transforms/compiler_function_utils.cc
 * \brief Helper passes for working with functions with the "Compiler" attribute.
 */

#include "./compiler_function_utils.h"

#include "../op/call/call.h"
#include "tvm/relay/analysis.h"
#include "tvm/relay/expr_functor.h"

namespace tvm {
namespace relay {
namespace transforms {
namespace {

/*!
 * \brief Rewrite calls to inlined "Compiler" functions to global functions. The given
 * module will be extended with the newly outlined functions.
 */
class Outliner : public MixedModeMutator {
 public:
  Outliner(GlobalSymbolCache* cache, std::string compiler_filter, IRModule mod)
      : cache_(cache), compiler_filter_(std::move(compiler_filter)), mod_(std::move(mod)) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    Call new_call = Downcast<Call>(post);
    if (const auto* function_node = new_call->op.as<FunctionNode>()) {
      Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
      if (opt_compiler.defined() &&
          (compiler_filter_.empty() || opt_compiler.value() == compiler_filter_)) {
        auto function = GetRef<Function>(function_node);
        DCHECK(FreeVars(function).empty()) << "Function marked with '" << attr::kCompiler
                                           << "' attribute should not have free variables";
        // Ask the cache to supply a unique  global var for this function.
        GlobalVar global_symbol = cache_->GetGlobalSymbol(function);
        // Depending on the cache's implementation, two structurally equal (but not object equal)
        // functions may be assigned the same global symbol. If so we'll lift it just once, but
        // rewrite all the calls.
        if (!mod_->ContainGlobalVar(global_symbol->name_hint)) {
          function =
              WithAttr(std::move(function), tvm::attr::kGlobalSymbol, global_symbol->name_hint);
          mod_->Add(global_symbol, function);
        }
        // Update the call.
        return WithFields(new_call, global_symbol);
      }
    }
    return post;
  }

 private:
  /*!
   * \brief A cached mapping from functions to global variables. Depending on the implementation
   * the cache may generate fresh symbols or require the function to already have a "global_symbol"
   * attribute, and may share symbols between structurally equal functions.
   */
  GlobalSymbolCache* cache_;
  /*! \brief If non-empty, the "Compiler" attribute value to require on functions to outline. */
  std::string compiler_filter_;
  /*! \brief Module being rewritten. */
  IRModule mod_;
};

}  // namespace

GlobalSymbolCache::~GlobalSymbolCache() = default;

GlobalVar ExistingGlobalSymbolCache::GetGlobalSymbol(const Function& function) {
  Optional<String> opt_global_symbol = function->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(opt_global_symbol.defined())
      << "ExistingGlobalSymbolCache requires all functions to already have a '"
      << tvm::attr::kGlobalSymbol << "' attribute";
  std::string global_symbol = opt_global_symbol.value();
  auto itr = global_vars_.find(global_symbol);
  if (itr != global_vars_.end()) {
    return itr->second;
  }
  // Ok if function does not have a checked_type, but if it does capture it in the global var.
  GlobalVar global_var(global_symbol, function->checked_type_, function->span);
  global_vars_.emplace(global_symbol, global_var);
  return global_var;
}

transform::Pass OutlineCompilerFunctions(std::shared_ptr<GlobalSymbolCache> cache,
                                         std::string compiler_filter) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [cache = std::move(cache), compiler_filter = std::move(compiler_filter)](
          IRModule mod, transform::PassContext ctx) {
        IRModule output_mod = GetRef<IRModule>(mod.CopyOnWrite());
        for (const auto& kv : mod->functions) {
          const FunctionNode* function_node = AsOptimizableFunctionNode(kv.second);
          if (function_node) {
            Expr new_body =
                Outliner(cache.get(), compiler_filter, output_mod).VisitExpr(function_node->body);
            Function new_function =
                WithFields(GetRef<Function>(function_node), /*opt_params=*/{}, new_body);
            output_mod->Add(kv.first, new_function);
          }
        }
        return output_mod;
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "OutlineCompilerFunctions", {});
}

// Any Java programmers in the house?
transform::Pass OutlineCompilerFunctionsWithExistingGlobalSymbols(std::string compiler_filter) {
  return OutlineCompilerFunctions(std::make_shared<ExistingGlobalSymbolCache>(),
                                  std::move(compiler_filter));
}

transform::Pass MarkCompilerFunctionsAsExtern(std::string compiler_filter) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [compiler_filter = std::move(compiler_filter)](IRModule mod, transform::PassContext ctx) {
        IRModule output_mod = mod->ShallowCopy();
        for (const auto& kv : mod->functions) {
          if (const auto* function_node = kv.second.as<FunctionNode>()) {
            Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
            if (opt_compiler.defined() &&
                (compiler_filter.empty() || opt_compiler.value() == compiler_filter)) {
              auto new_function = WithFields(
                  GetRef<Function>(function_node), function_node->params, function_node->body,
                  function_node->ret_type, function_node->type_params,
                  /* erase attributes */ DictAttrs(Map<String, ObjectRef>()));
              new_function = WithAttr(std::move(new_function), attr::kExtern, Integer(1));
              output_mod->Update(kv.first, new_function);
            }
          }
        }
        return output_mod;
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "MarkCompilerFunctionsAsExtern", {});
}

TVM_REGISTER_GLOBAL("relay._transform.OutlineCompilerFunctionsWithExistingGlobalSymbols")
    .set_body_typed(OutlineCompilerFunctionsWithExistingGlobalSymbols);
TVM_REGISTER_GLOBAL("relay._transform.MarkCompilerFunctionsAsExtern")
    .set_body_typed(MarkCompilerFunctionsAsExtern);

}  // namespace transforms
}  // namespace relay
}  // namespace tvm
