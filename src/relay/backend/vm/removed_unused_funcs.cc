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
 * \file tvm/relay/backend/vm/remove_unused_funcs.cc
 * \brief Remove unused global relay functions in a relay module.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <unordered_set>
#include <vector>

#include "../../op/call/call.h"

namespace tvm {
namespace relay {
namespace vm {

/**
 * \brief Detects all the functions that can be possibly called by entry function.
 */
struct CallTracer : ExprVisitor {
  IRModule module_;

  // Record the names of all encountered functions
  std::unordered_set<std::string> called_funcs_;

  // Record the expressions that are being visited
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visiting_;

  explicit CallTracer(const IRModule& module) : module_{module}, called_funcs_{}, visiting_{} {}

  void VisitExpr_(const GlobalVarNode* op) final {
    called_funcs_.insert(op->name_hint);
    auto func = module_->Lookup(op->name_hint);
    if (auto function_node = func.as<Function>()) {
      VisitExpr(function_node.value());
    }
    // else: Don't visit PrimFuncs -- we don't need to collect any tir.Calls therein.
  }

  void VisitExpr_(const CallNode* call_node) final {
    // TODO(mbs): Cleanup shape functions.
    CallLoweredProps props = GetCallLoweredProps(call_node);
    if (props.lowered_func.defined() && props.attrs.metadata.count("prim_shape_fn_var")) {
      auto callee = Downcast<GlobalVar>(props.attrs.metadata["prim_shape_fn_var"]);
      // We are implicitly calling the shape function *in addition to* the callee.
      called_funcs_.insert(callee->name_hint);
    }
    ExprVisitor::VisitExpr_(call_node);
  }

  void VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);
    if (visiting_.find(func) == visiting_.end()) {
      visiting_.insert(func);
      for (auto param : func_node->params) {
        ExprVisitor::VisitExpr(param);
      }
      ExprVisitor::VisitExpr(func_node->body);
    }
  }

  std::unordered_set<std::string> Trace(const std::string& entry) {
    called_funcs_.insert(entry);
    auto main_func = module_->Lookup(entry);
    VisitExpr(main_func);
    return called_funcs_;
  }
};

/*!
 * \brief Remove functions that are not used.
 *
 * \param module The Relay module.
 * \param entry_funcs The set of functions that can be entry function.
 *
 * \return The module with dead functions removed.
 */
IRModule RemoveUnusedFunctions(const IRModule& module, Array<runtime::String> entry_funcs) {
  std::unordered_set<std::string> called_funcs{};
  for (auto entry : entry_funcs) {
    auto funcs = CallTracer(module).Trace(entry);
    called_funcs.insert(funcs.cbegin(), funcs.cend());
  }
  auto existing_functions = module->functions;
  for (auto f : existing_functions) {
    auto it = called_funcs.find(f.first->name_hint);
    if (it == called_funcs.end()) {
      module->Remove(f.first);
    }
  }
  return module;
}

}  // namespace vm

namespace transform {

Pass RemoveUnusedFunctions(Array<runtime::String> entry_functions) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    return relay::vm::RemoveUnusedFunctions(m, entry_functions);
  };
  return CreateModulePass(pass_func, 1, "RemoveUnusedFunctions", {});
}

TVM_REGISTER_GLOBAL("relay._transform.RemoveUnusedFunctions").set_body_typed(RemoveUnusedFunctions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
