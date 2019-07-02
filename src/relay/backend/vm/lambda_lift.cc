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
 * \file tvm/relay/backend/vm/lambda_lift.cc
 * \brief Lift all local functions into global functions.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/logging.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
#include <iostream>
#include <vector>

using namespace tvm::runtime;

namespace tvm {
namespace relay {
namespace vm {

static const char* kIsClosure = "IsClosure";

inline std::string GenerateName(const Function& func) {
  size_t hash = StructuralHash()(func);
  return std::string("lifted_name") + std::to_string(hash);
}

bool IsClosure(const Function& func) {
  NodeRef res = FunctionGetAttr(func, kIsClosure);
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

Function MarkClosure(const Function& func) {
  return FunctionSetAttr(func, kIsClosure, tvm::Integer(1));
}

/* The goal of this class is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift a function out into a global which takes the set of the free
 * vars and then return the new created function.
 */
struct LambdaLifter : ExprMutator {
  Module module_;
  explicit LambdaLifter(const Module& module) : module_(module) {}

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    // We should not transform primitive functions.
    if (func->IsPrimitive()) {
      return std::move(func);
    }

    auto free_vars = FreeVars(func);
    auto free_type_vars = FreeTypeVars(func, module_);
    auto body = Downcast<Function>(ExprMutator::VisitExpr_(func_node));

    // When performing this optimization there are two cases.
    //
    // The first case in which we have no free variables
    // we can just lift the function into the global
    // environment without needing to allocate a closure.
    //
    //
    // The second case requires that we generate a special
    // function which makes a distinction between allocating
    // a closure, and then the code for the closure.
    //
    // We represent a closure allocation by lifting the
    // closure to a global function which takes its
    // captured arguments and then directly returns
    // the function representing the closure's code.
    //
    // When we generate code later on a call to the "outer"
    // function marked as a closure is used to emit allocation
    // code for the closure's environment.
    //
    // The "inner" function should be used to generate the
    // code for the closure.
    Function lifted_func;
    if (free_vars.size() == 0) {
      lifted_func = FunctionNode::make(body->params, body->body, body->ret_type, free_type_vars);
    } else {
      lifted_func =
          FunctionNode::make(free_vars, body, func->func_type_annotation(), free_type_vars);

      lifted_func = MarkClosure(lifted_func);
    }

    CHECK(lifted_func.defined());

    auto name = GenerateName(lifted_func);
    auto global = GlobalVarNode::make(name);

    // Add the lifted function to the module.
    module_->Add(global, lifted_func);

    if (free_vars.size() == 0) {
      return std::move(global);
    } else {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : free_vars) {
        fvs.push_back(fv);
      }
      return CallNode::make(global, fvs);
    }
  }

  Module Lift() {
    // There is an ordering bug here.
    auto glob_funcs = module_->functions;
    for (auto pair : glob_funcs) {
      auto func = pair.second;
      DLOG(INFO) << "Lifting " << AsText(func, false);
      func = FunctionNode::make(func->params,
                                VisitExpr(func->body),
                                func->ret_type,
                                func->type_params,
                                func->attrs);
      module_->Add(pair.first, func, true);
    }
    return module_;
  }
};

}  // namespace vm

namespace transform {

Pass LambdaLift() {
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func =
    [=](Module m, PassContext pc) {
    return relay::vm::LambdaLifter(m).Lift();
  };
  return CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_API("relay._transform.LambdaLift")
.set_body_typed(LambdaLift);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
