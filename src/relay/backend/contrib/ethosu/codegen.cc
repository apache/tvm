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

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/error.h>
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

using FTVMTIRToRuntime = tvm::runtime::TypedPackedFunc<runtime::Module(IRModule, Target)>;

/*!
 * \brief This mutator outlines functions that are marked with a named
 * "Compiler" attribute. Functions that do not match this condition remain
 * unaltered.
 */
class OutlineCompilerFunctionsMutator : public MixedModeMutator {
 public:
  explicit OutlineCompilerFunctionsMutator(const IRModule& mod, const std::string& compiler_name)
      : mod_(mod), compiler_name_(compiler_name) {}

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      Expr var = this->VisitExpr(op->var);
      Expr value = this->VisitExpr(op->value);

      // Outlineable function no longer needs let binding
      if (this->CanOutlineExpr(value)) {
        this->memo_[var] = value;
      }
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);

      // Drop the let binding
      if (this->CanOutlineExpr(value)) {
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

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    Call call = Downcast<Call>(post);
    if (CanOutlineExpr(call->op)) {
      Function func = Downcast<Function>(call->op);
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
    return post;
  }

 private:
  /*!
   * \brief Check if the expr is a function and has the same
   * compiler name as compiler_name_.
   *
   * \param expr The input expr.
   * \return True if is outlineable else False.
   */
  bool CanOutlineExpr(const Expr& expr) {
    if (!expr->IsInstance<FunctionNode>()) {
      return false;
    }
    Function func = Downcast<Function>(expr);
    auto compiler = func->GetAttr<String>(attr::kCompiler);
    if (!compiler.defined()) {
      return false;
    }
    if (compiler != compiler_name_) {
      return false;
    }
    return true;
  }

  /*! \brief The module that the pass will run on. */
  IRModule mod_;
  /*! \brief The name of the compiler to enable outlining on external functions for. */
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

    // don't consider rewrite if current op is an identity or concatenate.
    if (!call->op->IsInstance<OpNode>()) {
      return post;
    }
    const auto* op = call->op.as<OpNode>();
    std::string op_name = op->name;
    if (op_name == "contrib.ethosu.identity" || op_name == "concatenate") {
      return post;
    }

    // check if we can rewrite parent identity operations to current call.
    bool needs_rewrite = false;
    Array<Expr> new_args;
    for (const auto& arg : call->args) {
      Expr current_arg = arg;

      // expand tuple to get parent op if we run into one - nested tuples are not supported.
      if (const auto* tuple_get_item = arg.as<TupleGetItemNode>()) {
        const auto* tuple = tuple_get_item->tuple.as<TupleNode>();
        current_arg = tuple->fields[tuple_get_item->index];
      }

      if (const auto* parent_callnode = current_arg.as<CallNode>()) {
        if (auto parent_op = parent_callnode->op.as<OpNode>()) {
          Call parent_call = GetRef<Call>(parent_callnode);
          if (parent_op->name == "contrib.ethosu.identity" && IdentityDoesNothing(parent_call) &&
              CheckIdentityBetweenTransformOperations(call, parent_call)) {
            needs_rewrite = true;
            new_args.push_back(parent_call->args[0]);
            continue;
          }
        }
      }
      new_args.push_back(arg);
    }

    if (needs_rewrite) {
      Call new_call = Call(call->op, new_args, call->attrs, call->type_args);
      // since we are only removing an identity, we know the type information has not changed
      new_call->checked_type_ = call->checked_type_;
      return new_call;
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

  bool CheckIdentityBetweenTransformOperations(const Call& call, const Call& identity_call) {
    const auto* op = call->op.as<OpNode>();
    std::vector<std::string> nc_ops = {"reshape", "strided_slice"};

    if (op && (std::find(nc_ops.begin(), nc_ops.end(), op->name) != nc_ops.end())) {
      // check if the parent to identity operation is also a non-compute operation,
      // if it isn't we can safely remove the identity in question by returning true.
      const auto* identity_arg = identity_call->args[0].as<CallNode>();
      if (!identity_arg) {
        return true;
      }
      const auto* identity_arg_op = identity_arg->op.as<OpNode>();
      if (!identity_arg_op ||
          !(std::find(nc_ops.begin(), nc_ops.end(), identity_arg_op->name) != nc_ops.end())) {
        return true;
      }

      const auto* call_tt = call->checked_type_.as<TensorTypeNode>();
      const auto* identity_arg_tt = identity_arg->checked_type_.as<TensorTypeNode>();
      ICHECK(call_tt && identity_arg_tt)
          << "InferType should be run before RemoveRedundantIdentities";

      // we can only remove the identity operation if the second non-compute operation
      // in the sequence does not reduce the dimensionality of the output to the first
      // non-compute operation. Doing so could lead to data being accessed incorrectly
      // by the subsequent compute operation due to the reduction in dimensionality.
      size_t first_transform_op_dims = identity_arg_tt->shape.size();
      size_t second_transform_op_dims = call_tt->shape.size();
      if (second_transform_op_dims < first_transform_op_dims) {
        return false;
      }
    }
    return true;
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
  return tvm::transform::CreateModulePass(
      pass_func, 0, "relay.backend.contrib.ethos-u.IdentityOptimizer", {"InferType"});
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
    .set_attr<relay::transform::FTVMRelayToTIR>(tvm::attr::kRelayToTIR, RelayToTIR())
    .set_attr<FTVMTIRToRuntime>("TIRToRuntime", TIRToRuntime);

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
