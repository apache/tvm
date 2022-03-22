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

#include "../../../op/contrib/ethosu/op_attrs.h"
#include "../../../op/make_op.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosu {

/*!
 * \brief This mutator outlines functions that are marked with a named
 * "Compiler" attribute. Functions that do not match this condition remain
 * unaltered.
 */
class OutlineCompilerFunctionsMutator : public MixedModeMutator {
 public:
  explicit OutlineCompilerFunctionsMutator(const IRModule& mod, const std::string& compiler_name)
      : mod_(mod), compiler_name_(compiler_name) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    Call call = Downcast<Call>(post);
    if (call->op->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(call->op);
      auto compiler = func->GetAttr<String>(attr::kCompiler);
      if (compiler.defined() && compiler == compiler_name_) {
        auto gv_name = func->GetAttr<String>("global_symbol").value_or("");
        ICHECK_NE(gv_name, "")
            << "Function to be outlined must have global_symbol attribute, but didn't.";
        GlobalVar gv(gv_name);
        if (func->checked_type_.defined()) {
          gv->checked_type_ = func->checked_type();
        }
        mod_->Update(gv, func);
        return Call(gv, call->args, call->attrs, call->type_args);
      }
    }
    return post;
  }

 private:
  IRModule mod_;
  std::string compiler_name_;
};

/*!
 * \brief A pass to outline compiler specific functions.
 */
tvm::transform::Pass OutlineCompilerFunctions(const std::string& compiler_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule mod, transform::PassContext ctx) {
        GlobalVar gv = mod->GetGlobalVar("main");
        Function main_func = Downcast<Function>(mod->Lookup("main"));
        auto new_main_body =
            OutlineCompilerFunctionsMutator(mod, compiler_name).VisitExpr(main_func->body);
        if (!new_main_body.same_as(main_func->body)) {
          Function new_main_func = WithFields(main_func, main_func->params, new_main_body);
          mod->Update(gv, new_main_func);
        }
        return mod;
      };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "relay.backend.contrib.ethos-u.OutlineCompilerFunctions", {});
}

TVM_REGISTER_GLOBAL("relay.ext.ethos-u.OutlineCompilerFunctions")
    .set_body_typed(OutlineCompilerFunctions);

/*!
 * \brief This mutator removes identity operations that are not necessary. Specifically, an
 * identity operation can be removed when it is immediately followed by an NPU compute
 * operation.
 */
class RemoveRedundantIdentities : public MixedModeMutator {
 public:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    Call call = Downcast<Call>(post);

    // only consider rewrite if current op is an NPU compute op.
    if (!call->op->IsInstance<OpNode>()) {
      return post;
    }
    const auto* op = call->op.as<OpNode>();
    std::string op_name = op->name;
    if (op_name.substr(0, 15) != "contrib.ethosu." || op_name == "contrib.ethosu.identity") {
      return post;
    }

    // check if we can rewrite parent identity operations to current call.
    bool needs_rewrite = false;
    Array<Expr> new_args;
    for (const auto& arg : call->args) {
      if (const auto* parent_callnode = arg.as<CallNode>()) {
        if (const auto* parent_op = parent_callnode->op.as<OpNode>()) {
          Call parent_call = GetRef<Call>(parent_callnode);
          if (parent_op->name == "contrib.ethosu.identity" && IdentityDoesNothing(parent_call)) {
            needs_rewrite = true;
            new_args.push_back(parent_call->args[0]);
            continue;
          }
        }
      }
      new_args.push_back(arg);
    }

    if (needs_rewrite) {
      return Call(call->op, new_args, call->attrs, call->type_args);
    }
    return post;
  }

 private:
  bool IdentityDoesNothing(const Call& call) {
    const auto* attrs = call->attrs.as<tvm::relay::op::contrib::ethosu::EthosuIdentityAttrs>();
    bool does_not_requantize = attrs->ifm_scale == 1.0 && attrs->ifm_zero_point == 0 &&
                               attrs->ofm_scale == 1.0 && attrs->ofm_zero_point == 0;
    bool has_no_activation = attrs->activation == "NONE";
    return does_not_requantize && has_no_activation;
  }
};

/*!
 * \brief A pass to remove redundant identity operations.
 */
tvm::transform::Pass IdentityOptimizer() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule mod, transform::PassContext ctx) {
        for (auto gv : mod->GetGlobalVars()) {
          Function func = Downcast<Function>(mod->Lookup(gv));
          auto compiler_name = func->GetAttr<String>(attr::kCompiler);
          if (compiler_name.defined() && compiler_name == "ethos-u") {
            auto new_body = RemoveRedundantIdentities().VisitExpr(func->body);
            if (!new_body.same_as(func->body)) {
              Function new_func = WithFields(func, func->params, new_body);
              mod->Update(gv, new_func);
            }
          }
        }
        return mod;
      };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "relay.backend.contrib.ethos-u.IdentityOptimizer", {});
}

TVM_REGISTER_GLOBAL("relay.ext.ethos-u.IdentityOptimizer").set_body_typed(IdentityOptimizer);

/*!
 * \brief This pass will lower NPU functions in a Relay module to scheduled TIR prim functions.
 */
tvm::transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        auto relay_to_tir_pf = tvm::runtime::Registry::Get("relay.ext.ethos-u.relay_to_tir");
        ICHECK(relay_to_tir_pf);
        ir_module = (*relay_to_tir_pf)(ir_module);
        return ir_module;
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
