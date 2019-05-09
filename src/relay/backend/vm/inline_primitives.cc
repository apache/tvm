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
 * \file tvm/relay/backend/vm/inline_primitives.cc
 * \brief Ensure that primitives only appear in the call position.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/logging.h>
#include <tvm/relay/pass.h>
#include <tvm/runtime/vm.h>
#include <iostream>
#include <vector>

using namespace tvm::runtime;

namespace tvm {
namespace relay {
namespace vm {

struct PrimitiveInliner : ExprMutator {
  Module module_;
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> var_map;

  explicit PrimitiveInliner(const Module& module) : module_(module) {}

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
      if (func->IsPrimitive()) {
        return CallNode::make(GetRef<Function>(func), call->args, call->attrs, call->type_args);
      }
    }

    if (auto global = op.as<GlobalVarNode>()) {
      return CallNode::make(GetRef<GlobalVar>(global), call->args, call->attrs, call->type_args);
    }

    return ExprMutator::VisitExpr_(call);
  }

  Expr VisitExpr_(const FunctionNode* func) {
    if (func->IsPrimitive()) {
      return GetRef<Function>(func);
    } else {
      return ExprMutator::VisitExpr_(func);
    }
  }

  Function Inline(const Function& func) {
    DLOG(INFO) << "Before inlining primitives: " << std::endl
                    << "func= " << AsText(func, false) << std::endl;

    auto inlined = FunctionNode::make(func->params, VisitExpr(func->body), func->ret_type,
                                      func->type_params, func->attrs);

    inlined = Downcast<Function>(DeadCodeElimination(inlined));

    DLOG(INFO) << "After inlining primitives" << std::endl
                    << "after_func= " << AsText(inlined, false) << std::endl;
    return inlined;
  }
};

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
Module InlinePrimitives(const Module& module) {
  PrimitiveInliner inliner(module);

  tvm::Map<GlobalVar, Function> updates;

  // There is an ordering bug here.
  for (auto pair : module->functions) {
    auto global = pair.first;
    auto func = pair.second;
    updates.Set(global, inliner.Inline(func));
  }

  for (auto pair : updates) {
    module->Add(pair.first, pair.second, true);
  }

  return module;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
