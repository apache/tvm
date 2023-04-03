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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/run_codegen.cc
 * \brief Run codegen for annotated relax functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

#include <iostream>

#include "utils.h"

namespace tvm {
namespace relax {

class CodeGenRunner : ExprMutator {
 public:
  using OptionMap = Map<String, ObjectRef>;

  explicit CodeGenRunner(IRModule mod) : ExprMutator(mod) {}

  IRModule Run(Optional<Map<String, OptionMap>> target_options, Array<String> entry_functions) {
    IRModule mod = builder_->GetContextIRModule();
    for (const String& entry_func_name : entry_functions) {
      auto entry_func = mod->Lookup(entry_func_name);
      auto gvar = mod->GetGlobalVar(entry_func_name);
      builder_->UpdateFunction(gvar, Downcast<BaseFunc>(VisitExpr(entry_func)));
    }

    auto ext_mods = InvokeCodegen(mod, target_options.value_or({}));
    auto out_mod = builder_->GetContextIRModule();

    if (ext_mods.size()) {
      out_mod = WithAttr(out_mod, tvm::attr::kExternalMods, std::move(ext_mods));
    }

    if (constant_names.size()) {
      // Some backends (e.g. TensorRT) expect constants to be passed when they are instantiated
      Map<String, runtime::NDArray> constants;
      for (const auto& [constant, name] : constant_names) {
        ICHECK(!constants.count(name)) << "More than one constant with the name " << name;
        constants.Set(name, constant->data);
      }
      out_mod = WithAttr(out_mod, tvm::attr::kConstNameToConstant, std::move(constants));
    }

    // TODO(@tvm-team): Implicit pass dependency. Revisit when we have a better way to handle this.
    return DeadCodeElimination(out_mod, entry_functions);
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) override {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto const* gvar_node = call_node->op.as<GlobalVarNode>()) {
      const GlobalVar gvar = GetRef<GlobalVar>(gvar_node);

      auto create_call_dps_packed = [call_node, this](Expr extern_func,
                                                      StructInfo ret_struct_info) {
        Array<Expr> new_args({extern_func});
        new_args.push_back(Tuple(call_node->args.Map([this](Expr arg) { return VisitExpr(arg); })));

        static const Op& call_op = Op::Get("relax.call_dps_packed");

        return Call(call_op, new_args, tvm::Attrs(), {ret_struct_info});
      };

      if (auto it = extern_funcs_.find(gvar_node); it != extern_funcs_.end()) {
        return create_call_dps_packed(it->second.first, it->second.second);
      } else {
        // TODO(@sunggg): Is there any better way to get this func?
        Function func = Downcast<Function>(builder_->GetContextIRModule()->Lookup(gvar));
        Expr new_func = VisitExpr(func);

        if (new_func->IsInstance<ExternFuncNode>()) {
          auto ret_sinfo = GetStructInfo(call);
          extern_funcs_[gvar_node] = {new_func, ret_sinfo};
          // Remove the global symbol and codegen attributes from the function so that it can be
          // removed the module.
          static const runtime::PackedFunc* RemoveFuncAttrFunc =
              runtime::Registry::Get("ir.BaseFuncWithoutAttr");
          ICHECK(RemoveFuncAttrFunc);
          func = (*RemoveFuncAttrFunc)(func, tvm::attr::kGlobalSymbol);
          func = (*RemoveFuncAttrFunc)(func, attr::kCodegen);
          builder_->UpdateFunction(gvar, func);
          return create_call_dps_packed(new_func, ret_sinfo);
        }
      }
    }
    Array<Expr> new_args;
    for (const auto& arg : call_node->args) {
      new_args.push_back(VisitExpr(arg));
    }

    return Call(call_node->op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    Function func = GetRef<Function>(func_node);
    auto opt_codegen = func->GetAttr<String>(attr::kCodegen);
    if (opt_codegen) {
      auto ext_symbol = GetExtSymbol(func);
      size_t count = 0;
      PostOrderVisit(func->body, [=, &count](Expr e) {
        if (e->IsInstance<ConstantNode>()) {
          // Make sure to pick a unique name
          auto name = ext_symbol + "_" + opt_codegen.value() + "_const_" + std::to_string(count++);
          auto constant = Downcast<Constant>(e);
          constant_names.Set(constant, name);
        }
      });
      return ExternFunc(GetExtSymbol(func));
    } else {
      return ExprMutator::VisitExpr_(func_node);
    }
  }

 private:
  Array<runtime::Module> InvokeCodegen(IRModule mod, Map<String, OptionMap> target_options) {
    std::unordered_map<std::string, Array<Function>> target_functions;

    for (const auto& entry : mod->functions) {
      if (entry.second->IsInstance<tir::PrimFuncNode>()) {
        continue;
      }
      PostOrderVisit(entry.second, [&target_functions](Expr e) {
        if (e->IsInstance<FunctionNode>()) {
          auto f = Downcast<Function>(e);
          if (auto target_opt = f->GetAttr<String>(attr::kCodegen)) {
            String target = target_opt.value();
            target_functions[target].push_back(f);
          }
        }
      });
    }

    Array<runtime::Module> ext_mods;

    for (const auto& [target, functions] : target_functions) {
      OptionMap options = target_options.Get(target).value_or({});
      // Start the codegen process.
      // Get the codegen with its ffi key.
      String codegen_name = "relax.ext." + target;
      auto codegen = runtime::Registry::Get(codegen_name);
      ICHECK(codegen) << "Codegen is not found: " << codegen_name << "\n";

      Array<runtime::Module> compiled_functions = (*codegen)(functions, options, constant_names);
      ext_mods.insert(ext_mods.end(), compiled_functions.begin(), compiled_functions.end());
    }

    return ext_mods;
  }

  /*! \brief The names of all constants in the original module. */
  Map<Constant, String> constant_names;
  /*! \brief Extern funcs and their return struct infos for each global variable.  */
  std::unordered_map<const GlobalVarNode*, std::pair<Expr, StructInfo>> extern_funcs_;
};

}  // namespace relax

namespace transform {
Pass RunCodegen(Optional<Map<String, Map<String, ObjectRef>>> target_options,
                Array<String> entry_functions) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    return relax::CodeGenRunner(m).Run(target_options, entry_functions);
  };
  return CreateModulePass(pass_func, 0, "RunCodegen", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RunCodegen").set_body_typed(RunCodegen);

}  // namespace transform
}  // namespace tvm
