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
 * \file relay/backend/contrib/ethosu/codegen.cc
 *
 * \brief This file contains the target hooks for Arm(R) Ethos(TM)-U NPU
 * Codegen.
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

#include "../../../op/make_op.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosu {

/*!
 * \brief This mutator lowers each external
 * relay function to a TIR PrimFunc
 */
class RelayToTIRMutator : public MixedModeMutator {
 public:
  explicit RelayToTIRMutator(IRModule ir_module) : ir_module_(ir_module) {}

  IRModule operator()() {
    GlobalVar main_global_var = ir_module_->GetGlobalVar("main");
    Function main_func = Downcast<Function>(ir_module_->Lookup(main_global_var));

    // Copy everything across and mutate the body
    Function mutated_main =
        Function(main_func->params, VisitExpr(main_func->body), main_func->ret_type,
                 main_func->type_params, main_func->attrs, main_func->span);

    ir_module_->Update(main_global_var, mutated_main);
    ir_module_ = WithAttr(ir_module_, "device_contexts", device_contexts_);
    return ir_module_;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    Call call = Downcast<Call>(post);
    if (call->op->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(call->op);
      auto codegen_name = func->GetAttr<String>(attr::kCompiler);
      if (codegen_name.defined() && codegen_name == "ethos-u") {
        auto relay_to_tir_func_pf =
            tvm::runtime::Registry::Get("relay.ext.ethos-u.relay_to_tir_func");
        ICHECK(relay_to_tir_func_pf);
        tir::PrimFunc prim_func = (*relay_to_tir_func_pf)(func);
        prim_func = WithAttr(prim_func, tvm::attr::kTarget, Target("ethos-u"));
        String symbol_name = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
        GlobalVar gv(symbol_name);
        Array<RelayExpr> args = call->args;
        gv->checked_type_ = func->checked_type();
        ir_module_->Update(gv, prim_func);
        device_contexts_.Set(gv, codegen_name.value());
        return Call(gv, args, call->attrs, call->type_args);
      }
    }
    return post;
  }

 private:
  IRModule ir_module_;
  Map<GlobalVar, String> device_contexts_;
};

tvm::transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        return RelayToTIRMutator(ir_module)();
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "relay.contrib.ethos-u.RelayToTIR", {});
}

/*!
 * \brief This function lowers the IRModule with PrimFunc
 * with the target of the microNPU to a C-source runtime module
 */
runtime::Module TIRToRuntime(IRModule mod, Target target) {
  Array<CompilationArtifact> compile_artifacts;
  for (const auto& kv : mod->functions) {
    const tir::PrimFunc& prim_func = Downcast<tir::PrimFunc>(kv.second);
    Optional<Map<Integer, runtime::NDArray>> params =
        prim_func->GetAttr<Map<Integer, runtime::NDArray>>("ethos-u.constants");
    ICHECK(params) << "microNPU params should be present";
    auto primfunc_to_artifact_pf =
        tvm::runtime::Registry::Get("relay.ext.ethos-u.primfunc_to_artifact");
    ICHECK(primfunc_to_artifact_pf);
    CompilationArtifact ca = (*primfunc_to_artifact_pf)(prim_func);
    compile_artifacts.push_back(ca);
  }
  auto ca_to_runtime = tvm::runtime::Registry::Get("runtime.module.ethos-u.create");
  return (*ca_to_runtime)(compile_artifacts);
}

TVM_REGISTER_TARGET_KIND("ethos-u", kDLCPU)
    .set_attr<Bool>("use_device_api", Bool(true))
    .set_attr<FTVMRelayToTIR>("RelayToTIR", RelayToTIR())
    .set_attr<FTVMTIRToRuntime>("TIRToRuntime", TIRToRuntime);

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
