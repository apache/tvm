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
 * \file relay/backend/contrib/generic/codegen.cc
 *
 * \brief this file contains the target hooks for generic scale4edge codegen.
 */

#include <tvm/ir/error.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {
namespace generic {

/*!
 * \brief This mutator lowers each external
 * relay function to a TIR PrimFunc
 *
 * TODO: Just a slightly modified copy of Ethos-U. Needs refactoring for generic use-case.
 */
class RelayToTIRMutator : public MixedModeMutator {
 public:
  explicit RelayToTIRMutator(IRModule ir_module, String target_name)
      : ir_module_(ir_module),
        target_name_(target_name) {}

  IRModule operator()() {
    GlobalVar main_global_var = ir_module_->GetGlobalVar("main");
    Function main_func = Downcast<Function>(ir_module_->Lookup(main_global_var));

    // Copy everything across and mutate the body
    Function mutated_main =
        Function(main_func->params, VisitExpr(main_func->body), main_func->ret_type,
                 main_func->type_params, main_func->attrs, main_func->span);

    ir_module_->Update(main_global_var, mutated_main);

    return ir_module_;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    Call call = Downcast<Call>(post);
    if (call->op->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(call->op);
      auto codegen_name = func->GetAttr<String>(attr::kCompiler);
      if (codegen_name.defined() && codegen_name == target_name_) {
        auto relay_to_tir_func_pf =
            tvm::runtime::Registry::Get("relay.ext.generic.relay_to_tir_func_" + target_name_);
        ICHECK(relay_to_tir_func_pf);
        tir::PrimFunc prim_func = (*relay_to_tir_func_pf)(func);
        prim_func = WithAttr(prim_func, tvm::attr::kTarget, Target(target_name_));
        String symbol_name = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
        GlobalVar gv(symbol_name);
        gv->checked_type_ = func->checked_type();
        ir_module_->Update(gv, prim_func);
        return Call(gv, call->args, call->attrs, call->type_args);
      }
    }
    return post;
  }

 private:
  IRModule ir_module_;
  String target_name_;
};

tvm::transform::Pass RelayToTIR(String target_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        return RelayToTIRMutator(ir_module, target_name)();
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "relay.contrib.generic.RelayToTIR", {});
}

}  // namespace generic
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
