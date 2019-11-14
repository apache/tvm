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
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/backend/vm/remove_unused_funcs.cc
 * \brief Remove unused global relay functions in a relay module.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/logging.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relay {
namespace vm {

/**
 * \brief Detects all the functions that can be possibly called by entry function.
 */
struct CallTracer : ExprVisitor {
  Module module_;

  // Record the names of all encountered functions
  std::unordered_set<std::string> called_funcs_;

  // Record the expressions that are being visited
  std::unordered_set<Expr, NodeHash, NodeEqual> visiting_;

  explicit CallTracer(const Module& module)
    : module_{module},
      called_funcs_{},
      visiting_{} {}

  void VisitExpr_(const CallNode* call_node) final {
    Expr op = call_node->op;
    for (auto param : call_node->args) {
      VisitExpr(param);
    }
    if (auto func_node = op.as<FunctionNode>()) {
      auto func = GetRef<Function>(func_node);
      auto it = visiting_.find(func);
      if (it != visiting_.end()) {
        return;
      }
      visiting_.insert(func);
      VisitExpr(func);
    } else if (auto global = op.as<GlobalVarNode>()) {
      called_funcs_.insert(global->name_hint);
      auto func = module_->Lookup(global->name_hint);
      auto it = visiting_.find(func);
      if (it != visiting_.end()) {
        return;
      }
      visiting_.insert(func);
      VisitExpr(func);
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
Module RemoveUnusedFunctions(const Module& module,
                             Array<tvm::Expr> entry_funcs) {
  std::unordered_set<std::string> called_funcs{};
  for (auto entry : entry_funcs) {
    auto* str_name = entry.as<ir::StringImm>();
    auto funcs = CallTracer(module).Trace(str_name->value);
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

Pass RemoveUnusedFunctions(Array<tvm::Expr> entry_functions) {
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func =
    [=](Module m, PassContext pc) {
    return relay::vm::RemoveUnusedFunctions(m, entry_functions);
  };
  return CreateModulePass(pass_func, 1, "RemoveUnusedFunctions", {});
}

TVM_REGISTER_API("relay._transform.RemoveUnusedFunctions")
.set_body_typed(RemoveUnusedFunctions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
