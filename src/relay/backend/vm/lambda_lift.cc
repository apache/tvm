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
 * \file tvm/relay/backend/vm/lambda_lift.cc
 * \brief Lift all local functions into global functions.
 */

#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/support/logging.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
#include <iostream>
#include <vector>

using namespace tvm::runtime;

namespace tvm {
namespace relay {
namespace vm {

inline std::string GenerateName(const Function& func) {
  size_t hash = tvm::StructuralHash()(func);
  return std::string("lifted_name") + std::to_string(hash);
}

bool IsClosure(const Function& func) {
  return func->GetAttr<Integer>(attr::kClosure, 0) != 0;
}

Function MarkClosure(Function func) {
  return WithAttr(std::move(func), attr::kClosure, tvm::Integer(1));
}

/* The goal of this class is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift a function out into a global which takes the set of the free
 * vars and then return the new created function.
 */
class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(const IRModule& module) : module_(module) {}

  Expr VisitExpr_(const LetNode* let_node) final {
    bool is_lambda = false;
    if (auto func = let_node->value.as<FunctionNode>()) {
      if (!func->HasNonzeroAttr(attr::kPrimitive)) {
        is_lambda = true;
        letrec_.push_back(let_node->var);
      }
    }
    auto value = VisitExpr(let_node->value);
    if (is_lambda) {
      letrec_.pop_back();
    }
    auto body = VisitExpr(let_node->body);
    return Let(let_node->var, value, body);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto var_node = call_node->op.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      if (!letrec_.empty() && var == letrec_.back()) {
        auto it = lambda_map_.find(var);
        CHECK(it != lambda_map_.end());
        return Call(it->second, call->args, call_node->attrs,
                              call_node->type_args);
      }
    }
    return std::move(call);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    // We should not transform primitive functions.
    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      return std::move(func);
    }

    auto name = GenerateName(func);
    auto global = GlobalVar(name);
    auto free_vars = FreeVars(func);
    auto free_type_vars = FreeTypeVars(func, module_);

    Array<Var> captured_vars;
    bool recursive = false;
    for (const auto& var : free_vars) {
      if (!letrec_.empty() && var == letrec_.back()) {
        recursive = true;
        continue;
      }
      captured_vars.push_back(var);
    }
    if (recursive) {
      if (!captured_vars.empty()) {
        Array<Expr> fvs;
        for (auto fv : captured_vars) {
          fvs.push_back(fv);
        }
        lambda_map_.emplace(letrec_.back(), Call(global, fvs));
      } else {
        lambda_map_.emplace(letrec_.back(), global);
      }
    }
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
    if (captured_vars.size() == 0 && free_type_vars.size() == 0) {
      lifted_func = Function(body->params, body->body, body->ret_type, body->type_params);
    } else {
      lifted_func =
          Function(captured_vars, body, func->func_type_annotation(), free_type_vars);
      lifted_func = MarkClosure(lifted_func);
    }

    CHECK(lifted_func.defined());


    if (module_->ContainGlobalVar(name)) {
      const auto existing_func = module_->Lookup(name);
      CHECK(tvm::StructuralEqual()(lifted_func, existing_func))
        << "lifted function hash collision";
      // If an identical function already exists, use its global var.
      global = module_->GetGlobalVar(name);
    } else {
      // Add the lifted function to the module.
      module_->Add(global, lifted_func);
    }

    if (captured_vars.size() == 0) {
      return std::move(global);
    } else {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : captured_vars) {
        fvs.push_back(fv);
      }
      return Call(global, fvs);
    }
  }

  IRModule Lift() {
    // There is an ordering bug here.
    auto glob_funcs = module_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
        auto func = GetRef<Function>(n);
        func = Function(func->params,
                        VisitExpr(func->body),
                        func->ret_type,
                        func->type_params,
                        func->attrs);
        module_->Add(pair.first, func, true);
      }
    }
    return module_;
  }

 private:
  std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> lambda_map_;
  std::vector<Var> letrec_;
  IRModule module_;
};

}  // namespace vm

namespace transform {

Pass LambdaLift() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
    [=](IRModule m, PassContext pc) {
    return relay::vm::LambdaLifter(m).Lift();
  };
  return CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_GLOBAL("relay._transform.LambdaLift")
.set_body_typed(LambdaLift);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
