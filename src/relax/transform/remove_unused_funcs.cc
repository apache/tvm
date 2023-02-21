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
 * \file tvm/relax/transform/remove_unused_funcs.cc
 * \brief Remove unused global relax functions in a IRModule.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <unordered_set>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

/**
 * \brief Detects all the functions that can be possibly called by entry function.
 */
class CallTracer : ExprVisitor {
 public:
  explicit CallTracer(IRModule mod_) : mod_{mod_}, called_funcs_{}, visiting_{} {}

  void VisitExpr_(const GlobalVarNode* op) final {
    called_funcs_.insert(GetRef<GlobalVar>(op));
    auto func = mod_->Lookup(op->name_hint);
    if (const auto* function_node = func.as<FunctionNode>()) {
      VisitExpr(GetRef<Function>(function_node));
    }
    // else: Don't visit PrimFuncs -- we don't need to collect any tir.Calls therein.
  }

  void VisitExpr_(const CallNode* call_node) final { ExprVisitor::VisitExpr_(call_node); }

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

  void Trace(std::string entry) {
    called_funcs_.insert(mod_->GetGlobalVar(entry));
    auto main_func = mod_->Lookup(entry);
    VisitExpr(main_func);
  }

  bool check_if_called(GlobalVar gv) { return called_funcs_.count(gv) > 0; }

 private:
  IRModule mod_;

  // Record the names of all encountered functions.
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> called_funcs_;

  // Record the expressions that are being visited.
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visiting_;
};

/*!
 * \brief Remove functions that are not used.
 *
 * \param mod_ IRModule.
 * \param entry_funcs The set of functions that can be entry function.
 *
 * \return The module with dead functions removed.
 */
IRModule RemoveUnusedFunctions(IRModule mod_, Array<runtime::String> entry_funcs) {
  auto tracer = CallTracer(mod_);
  for (auto entry : entry_funcs) {
    tracer.Trace(entry);
  }
  auto existing_functions = mod_->functions;
  for (auto f : existing_functions) {
    // If a function has an external linkage type, we do not remove it.
    // Otherwise, we check the function and remove it if it is not used anywhere.
    if (f.second->GetLinkageType() == LinkageType::kInternal && !tracer.check_if_called(f.first)) {
      mod_->Remove(f.first);
    }
  }
  return mod_;
}

}  // namespace relax

namespace transform {
Pass RemoveUnusedFunctions(Array<runtime::String> entry_functions) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::RemoveUnusedFunctions(m, entry_functions); };
  return CreateModulePass(pass_func, 0, "RemoveUnusedFunctions", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RemoveUnusedFunctions").set_body_typed(RemoveUnusedFunctions);

}  // namespace transform
}  // namespace tvm
