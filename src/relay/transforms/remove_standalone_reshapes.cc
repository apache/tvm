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
 * \file src/relay/transforms/remove_standalone_reshapes.cc
 * \brief This file contains the Relay pass for removing unfused reshapes from lowered graph.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../op/call/call.h"
#include "../op/memory/on_device.h"

namespace tvm {
namespace relay {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.remove_standalone_reshapes.enable", Bool);
/*! Removes reshapes right after LowerTE. Removes preceding on_device calls
 * while removing reshapes.
 */
class RemoveStandaloneReshapesMutator : public MixedModeMutator {
 public:
  explicit RemoveStandaloneReshapesMutator(IRModule& mod) {}  // NOLINT(runtime/references)

  using MixedModeMutator::VisitExpr_;

  /*!  * \brief Generated map of let variables to preceding CallLowered */
  Expr VisitExpr_(const LetNode* let) final {
    Let ret_let;
    Var var = Downcast<Var>(this->Mutate(let->var));
    auto value = this->Mutate(let->value);
    if (auto* on_device_call = value.as<CallNode>()) {
      OnDeviceProps on_device_props = GetOnDeviceProps(on_device_call);
      if (on_device_props.body.defined() && on_device_props.body->IsInstance<CallNode>()) {
        const Call call_lowered = Downcast<Call>(on_device_props.body);
        if (call_lowered.defined() && call_lowered->op.same_as(CallLoweredOp())) {
          let_var_to_call_lowered_.Set(var, call_lowered);
        }
      }
    }
    auto body = this->Mutate(let->body);
    return WithFields(GetRef<Let>(let), var, value, body);
  }

  /*!  * \brief Returns preceding CallLowered when call is a CallLowered(Reshape) */
  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    /*
    %1 = call_lowered(@tvmgen_default_non_reshape_function, %input, ...);
    let %x: = on_device(%1, ...);
    %2 = (%x,);
    %3 = call_lowered(@tvmgen_default_fused_reshape, %2, ...,
    "relay_attrs"=__dict__="relay.reshape_only"=1, ...);
    */
    const CallNode* post_call = post.as<CallNode>();
    CallLoweredProps call_lowered_props = GetCallLoweredProps(post_call);
    if (call_lowered_props.lowered_func.defined() && IsReshapeOnly(call_lowered_props)) {
      if (!call_lowered_props.arguments.empty() &&
          call_lowered_props.arguments[0]->IsInstance<VarNode>()) {
        Var var = Downcast<Var>(call_lowered_props.arguments[0]);
        if (var.defined() && let_var_to_call_lowered_.find(var) != let_var_to_call_lowered_.end()) {
          return let_var_to_call_lowered_[var];
        }
      }
    }

    return post;
  }

 private:
  /*! \brief Map of LetNode's var to previous call_lowered. */
  Map<Var, Call> let_var_to_call_lowered_;
};

namespace transform {

Pass RemoveStandaloneReshapes() {
  auto pass_func = [=](IRModule mod, const PassContext& pass_ctx) {
    VLOG(1) << "RemoveStandaloneReshapes before:" << std::endl << PrettyPrint(mod);
    RemoveStandaloneReshapesMutator remove_reshapes_mutator(mod);
    Function main_func = Downcast<Function>(mod->Lookup("main"));
    Expr new_main_body = remove_reshapes_mutator.VisitExpr(main_func->body);
    if (!new_main_body.same_as(main_func->body)) {
      auto main_var = mod->GetGlobalVar("main");
      auto new_main_func = Function(main_func->params, new_main_body, main_func->ret_type,
                                    main_func->type_params, main_func->attrs);
      mod->Update(main_var, new_main_func);
    }
    Array<runtime::String> entry_functions{"main"};
    mod = RemoveUnusedFunctions(entry_functions)(mod);

    VLOG(1) << "RemoveStandaloneReshapes after:" << std::endl << PrettyPrint(mod);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "RemoveStandaloneReshapes", {});
}

TVM_REGISTER_GLOBAL("relay._transform.RemoveStandaloneReshapes")
    .set_body_typed(RemoveStandaloneReshapes);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
