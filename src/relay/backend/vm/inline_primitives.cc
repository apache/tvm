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
 * \file tvm/relay/backend/vm/inline_primitives.cc
 * \brief Ensure that primitives only appear in the call position.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/support/logging.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
#include <iostream>
#include <vector>

using namespace tvm::runtime;

namespace tvm {
namespace relay {
namespace vm {

// TODO(@jroesch): write verifier

/* This pass will eliminate primitives which have been lifted by the ANF
 * transform inlining them directly into call sites.
 *
 * This makes VM related code generation easier as the call target is always
 * a primitive function.
 *
 * let prim = fn(...) { ... };
 * prim(...)
 *
 * will become:
 *
 * (fn(...) { ... })(...)
 */
struct PrimitiveInliner : ExprMutator {
  IRModule module_;
  std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> var_map;

  explicit PrimitiveInliner(const IRModule& module) : module_(module) {}

  Expr VisitExpr_(const LetNode* let_node) {
    var_map.insert({let_node->var, VisitExpr(let_node->value)});
    return ExprMutator::VisitExpr_(let_node);
  }

  Expr VisitExpr_(const CallNode* call) {
    Expr op = call->op;
    // For now just collapse the chain of variables to see if
    // they point to a primitive function.
    const VarNode* var_node;

    // Collapse a chain of let bindings
    //
    // let x = fn (..) { .. };
    // let y = x
    // let w = y
    // in w(...)
    while ((var_node = op.as<VarNode>())) {
      auto var = GetRef<Var>(var_node);
      DLOG(INFO) << "Var: " << var << std::endl;
      auto it = var_map.find(GetRef<Var>(var_node));
      if (it != var_map.end()) {
        op = it->second;
      } else {
        return ExprMutator::VisitExpr_(call);
      }
    }

    if (auto func = op.as<FunctionNode>()) {
      if (func->HasNonzeroAttr(attr::kPrimitive)) {
        tvm::Array<Expr> call_args;
        for (auto arg : call->args) {
          auto new_arg = VisitExpr(arg);
          call_args.push_back(new_arg);
        }
        return Call(GetRef<Function>(func), call_args, call->attrs, call->type_args);
      }
    }

    if (auto global = op.as<GlobalVarNode>()) {
      tvm::Array<Expr> call_args;
      for (auto arg : call->args) {
        auto new_arg = VisitExpr(arg);
        call_args.push_back(new_arg);
      }
      return Call(GetRef<GlobalVar>(global), call_args, call->attrs, call->type_args);
    }

    return ExprMutator::VisitExpr_(call);
  }

  Expr VisitExpr_(const FunctionNode* func) {
    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Function>(func);
    } else {
      return ExprMutator::VisitExpr_(func);
    }
  }

  IRModule Inline() {
    auto gvar_funcs = module_->functions;
    for (auto pair : gvar_funcs) {
      auto global = pair.first;
      auto base_func = pair.second;
      if (auto* n = base_func.as<FunctionNode>()) {
        if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
        auto func = GetRef<Function>(n);

        DLOG(INFO) << "Before inlining primitives: " << global
                   << std::endl << AsText(func, false);

        func = Function(func->params,
                        VisitExpr(func->body),
                        func->ret_type,
                        func->type_params,
                        func->attrs);
        module_->Add(global, func, true);

        DLOG(INFO) << "After inlining primitives: " << global
                   << std::endl << AsText(func, false);
      }
    }
    return module_;
  }
};

}  // namespace vm

namespace transform {

Pass InlinePrimitives() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
    [=](IRModule m, PassContext pc) {
      return relay::vm::PrimitiveInliner(m).Inline();
  };
  auto inline_pass = CreateModulePass(pass_func, 1, "Inline", {});
  // Eliminate dead code for each function after inlining.
  return Sequential({inline_pass, DeadCodeElimination()}, "InlinePrimitives");
}

TVM_REGISTER_GLOBAL("relay._transform.InlinePrimitives")
.set_body_typed(InlinePrimitives);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
