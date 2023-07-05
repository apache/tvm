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

#include "tvm/relay/analysis.h"
#include "tvm/relay/expr_functor.h"
#include "tvm/relay/transform.h"

namespace tvm {
namespace relay {
namespace transform {
namespace {

/*!
 * \brief Returns the \p FunctionNode of if \p expr if it is a "Compiler" function which should
 * be processed by a pass using \p compiler_filter. Otherwise returns null.
 */
const FunctionNode* AsFunctionNode(const Expr& expr, const std::string& compiler_filter) {
  if (const auto* function_node = expr.as<FunctionNode>()) {
    Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
    if (opt_compiler.defined() &&
        (compiler_filter.empty() || opt_compiler.value() == compiler_filter)) {
      return function_node;
    }
  }
  return nullptr;
}

/*!
 * \brief Rewrite calls to inlined and let-bound "Compiler" functions to global functions. The given
 * module will be extended with the newly outlined functions.
 */
class Outliner : public MixedModeMutator {
 public:
  using MixedModeMutator::VisitExpr_;

  Outliner(GlobalSymbolCache* cache, std::string compiler_filter, IRModule mod)
      : cache_(cache), compiler_filter_(std::move(compiler_filter)), mod_(std::move(mod)) {}

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      Expr var = this->VisitExpr(op->var);
      Expr value = this->VisitExpr(op->value);

      if (AsFunctionNode(value, compiler_filter_)) {
        // Inline on-the-fly if the let-bound value is a function of interest.
        this->memo_[var] = value;
      }
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);

      if (AsFunctionNode(value, compiler_filter_)) {
        // The let binding is no longer needed since inlined on-the-fly above.
        this->memo_[expr] = this->VisitExpr(op->body);
      } else {
        Var var = Downcast<Var>(this->VisitExpr(op->var));
        if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    Call new_call = Downcast<Call>(post);
    if (const auto* function_node = AsFunctionNode(new_call->op, compiler_filter_)) {
      auto function = GetRef<Function>(function_node);
      DCHECK(FreeVars(function).empty()) << "Function marked with '" << attr::kCompiler
                                         << "' attribute should not have free variables";
      // Ask the cache to supply a unique  global var for this function.
      GlobalVar global_symbol = cache_->GetGlobalSymbol(function);
      // Depending on the cache's implementation, two structurally equal (but not object
      // equal) functions may be assigned the same global symbol. If so we'll lift it just
      // once, but rewrite all the calls.
      if (!mod_->ContainGlobalVar(global_symbol->name_hint)) {
        function =
            WithAttr(std::move(function), tvm::attr::kGlobalSymbol, global_symbol->name_hint);
        mod_->Add(global_symbol, function);
      }
      // Update the call.
      return WithFields(new_call, global_symbol);
    }
    return post;
  }

 private:
  /*!
   * \brief A cached mapping from functions to global variables. Depending on the implementation
   * the cache may generate fresh symbols or require the function to already have a
   * "global_symbol" attribute, and may share symbols between structurally equal functions.
   */
  GlobalSymbolCache* cache_;
  /*! \brief If non-empty, the "Compiler" attribute value to require on functions to outline. */
  std::string compiler_filter_;
  /*! \brief Module being rewritten. */
  IRModule mod_;
};

/*!
 * \brief Inline immediate calls to "Composite" functions.
 */
class InnerInliner : public MixedModeMutator {
 public:
  InnerInliner() = default;

 private:
  using MixedModeMutator::Rewrite_;

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    Call new_call = Downcast<Call>(post);
    if (const auto* function_node = new_call->op.as<FunctionNode>()) {
      ICHECK(function_node->GetAttr<String>(attr::kComposite).defined());
      ICHECK_EQ(function_node->params.size(), new_call->args.size());
      Map<Var, Expr> subst;
      for (size_t i = 0; i < new_call->args.size(); ++i) {
        subst.Set(function_node->params[i], new_call->args[i]);
      }
      return Bind(function_node->body, subst);
    }
    return post;
  }
};

/*!
 * \brief Inline calls to global "Compiler" functions with global var in \p global_vars.
 * Both the 'outer' "Compiler" function and any 'inner' "Composite" functions in its body
 * are inlined.
 */
class OuterInliner : public MixedModeMutator {
 public:
  OuterInliner(IRModule mod, Array<GlobalVar> global_vars_)
      : mod_(std::move(mod)), global_vars_(std::move(global_vars_)) {}

 private:
  using MixedModeMutator::Rewrite_;

  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    Call new_call = Downcast<Call>(post);
    if (auto global_var_node = new_call->op.as<GlobalVar>()) {
      auto global_var = global_var_node.value();
      if (std::find(global_vars_.begin(), global_vars_.end(), global_var) != global_vars_.end()) {
        BaseFunc base_func = mod_->Lookup(global_var);
        const auto* function_node = base_func.as<FunctionNode>();
        ICHECK(function_node);
        ICHECK(function_node->GetAttr<String>(attr::kCompiler).defined());
        ICHECK_EQ(function_node->params.size(), new_call->args.size());
        Map<Var, Expr> subst;
        for (size_t i = 0; i < new_call->args.size(); ++i) {
          subst.Set(function_node->params[i], new_call->args[i]);
        }
        Expr new_body = InnerInliner().VisitExpr(function_node->body);
        return Bind(new_body, subst);
      }
    }
    return post;
  }

 private:
  /*! \brief Original module we are processing. */
  IRModule mod_;
  /*! \brief Global vars of functions to inline. */
  Array<GlobalVar> global_vars_;
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

tvm::transform::Pass OutlineCompilerFunctions(std::shared_ptr<GlobalSymbolCache> cache,
                                              std::string compiler_filter) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [cache = std::move(cache), compiler_filter = std::move(compiler_filter)](
          IRModule mod, transform::PassContext ctx) {
        VLOG(1) << "OutlineCompilerFunctions input:" << std::endl << PrettyPrint(mod);
        IRModule output_mod = mod->ShallowCopy();
        for (const auto& kv : mod->functions) {
          if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
            Expr new_body =
                Outliner(cache.get(), compiler_filter, output_mod).VisitExpr(function_node->body);
            Function new_function =
                WithFields(GetRef<Function>(function_node), /*opt_params=*/{}, new_body);
            output_mod->Add(kv.first, new_function);
          }
        }
        VLOG(1) << "OutlineCompilerFunctions result:" << std::endl << PrettyPrint(output_mod);
        return output_mod;
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "OutlineCompilerFunctions", {});
}

// Any Java programmers in the house?
tvm::transform::Pass OutlineCompilerFunctionsWithExistingGlobalSymbols(
    std::string compiler_filter) {
  return OutlineCompilerFunctions(std::make_shared<ExistingGlobalSymbolCache>(),
                                  std::move(compiler_filter));
}

tvm::transform::Pass MarkCompilerFunctionsAsExtern(std::string compiler_filter) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [compiler_filter = std::move(compiler_filter)](IRModule mod, transform::PassContext ctx) {
        VLOG(1) << "MarkCompilerFunctionsAsExtern input:" << std::endl << PrettyPrint(mod);
        IRModule output_mod = mod->ShallowCopy();
        for (const auto& kv : mod->functions) {
          if (const auto* function_node = AsFunctionNode(kv.second, compiler_filter)) {
            auto new_function =
                WithFields(GetRef<Function>(function_node), function_node->params,
                           function_node->body, function_node->ret_type, function_node->type_params,
                           /* erase attributes */ DictAttrs(Map<String, ObjectRef>()));
            new_function = WithAttr(std::move(new_function), attr::kExtern, Integer(1));
            output_mod->Update(kv.first, new_function);
          }
        }
        VLOG(1) << "MarkCompilerFunctionsAsExtern result:" << std::endl << PrettyPrint(output_mod);
        return output_mod;
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "MarkCompilerFunctionsAsExtern", {});
}

tvm::transform::Pass InlineCompilerFunctionsBoundTo(Array<GlobalVar> global_vars) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [global_vars = std::move(global_vars)](IRModule mod, transform::PassContext ctx) {
        VLOG(1) << "InlineCompilerFunctionsBoundTo with global_vars: " << PrettyPrint(global_vars);
        if (global_vars.empty()) {
          return mod;
        }
        VLOG(1) << "InlineCompilerFunctions input:" << std::endl << PrettyPrint(mod);
        IRModule output_mod = mod->ShallowCopy();
        for (const auto& kv : mod->functions) {
          if (std::find(global_vars.begin(), global_vars.end(), kv.first) != global_vars.end()) {
            output_mod->Remove(kv.first);
          } else if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
            Expr new_body = OuterInliner(mod, global_vars).VisitExpr(function_node->body);
            Function new_function =
                WithFields(GetRef<Function>(function_node), /*opt_params=*/{}, new_body);
            output_mod->Add(kv.first, new_function);
          }
        }
        VLOG(1) << "InlineCompilerFunctionsBoundTo result:" << std::endl << PrettyPrint(output_mod);
        return output_mod;
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "InlineCompilerFunctionsBoundTo", {});
}

TVM_REGISTER_GLOBAL("relay._transform.OutlineCompilerFunctionsWithExistingGlobalSymbols")
    .set_body_typed(OutlineCompilerFunctionsWithExistingGlobalSymbols);
TVM_REGISTER_GLOBAL("relay._transform.MarkCompilerFunctionsAsExtern")
    .set_body_typed(MarkCompilerFunctionsAsExtern);
TVM_REGISTER_GLOBAL("relay._transform.InlineCompilerFunctionsBoundTo")
    .set_body_typed(InlineCompilerFunctionsBoundTo);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
