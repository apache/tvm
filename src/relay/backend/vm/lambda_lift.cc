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
class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(const Module& module) : module_(module) {}

  Expr VisitExpr_(const LetNode* let_node) final {
    std::cout << "============================================\n";
    std::cout << "Visit let: " << AsText(GetRef<Let>(let_node), false) << "\n";
    bool is_lambda = false;
    if (auto func = let_node->value.as<FunctionNode>()) {
      if (!func->IsPrimitive()) {
        std::cout << "[Letrec] push back " << let_node->var << "\n";
        is_lambda = true;
        letrec_.push_back(let_node->var);
      }
    }
    auto new_let = ExprMutator::VisitExpr_(let_node);
    if (is_lambda) {
      std::cout << "[Letrec] pop\n";
      letrec_.pop_back();
    }
    std::cout << "[NEW let] " << AsText(new_let, false) << "\n";
    return new_let;
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    std::cout << "============================================\n";
    std::cout << "Visit call: " << AsText(GetRef<Call>(call_node), false) << "\n";
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto var_node = call_node->op.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      auto it = lambda_map_.find(var);
      if (it != lambda_map_.end()) {
        auto new_call = CallNode::make(it->second, call->args, call_node->attrs, call_node->type_args);
        std::cout << "[new call] " << AsText(new_call, false) << "\n";
        return new_call;
        // CHECK_GT(lambda_captured_vars_.count(var), 0);
        // auto args = call->args;
        // for (auto arg : lambda_captured_vars_.at(var)) {
        //   args.push_back(arg);
        // }
        // auto new_call = CallNode::make(it->second, args, call_node->attrs, call_node->type_args);
        // std::cout << "[NEW call] " << AsText(new_call, false) << "\n";
        // return new_call;
      }
    }
    return call;
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);
    std::cout << "============================================\n";
    std::cout << "Visit func: " << AsText(func, false) << "\n";

    // We should not transform primitive functions.
    if (func->IsPrimitive()) {
      return std::move(func);
    }

    auto name = GenerateName(func);
    auto global = GlobalVarNode::make(name);
    
    auto free_vars = FreeVars(func);
    Array<Var> captured_vars;
    bool recursive = false;
    for (const auto& var : free_vars) {
      if (!letrec_.empty() && var == letrec_.back()) {
        std::cout << "recursive: " << var <<"\n";
        recursive = true;
        continue;
      }
      captured_vars.push_back(var);
    }
    if (recursive) {
      lambda_map_.emplace(letrec_.back(), global);
      lambda_captured_vars_.emplace(letrec_.back(), captured_vars);
    }
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
    if (captured_vars.size() == 0 && free_type_vars.size() == 0) {
      lifted_func = FunctionNode::make(body->params, body->body, body->ret_type, body->type_params);
    } else {
      auto params = body->params;
      for (auto var : captured_vars) {
        params.push_back(var);
      }
      // lifted_func = FunctionNode::make(params, body->body, body->ret_type, body->type_params);
      
      lifted_func =
          FunctionNode::make(captured_vars, body, func->func_type_annotation(), free_type_vars);
      lifted_func = MarkClosure(lifted_func);
      std::cout << "lifted function (closure): " << AsText(lifted_func, false) << "\n";
    }

    CHECK(lifted_func.defined());


    if (module_->ContainGlobalVar(name)) {
      const auto existing_func = module_->Lookup(name);
      CHECK(AlphaEqual(lifted_func, existing_func)) << "lifted function hash collision";
      // If an identical function already exists, use its global var.
      global = module_->GetGlobalVar(name);
    } else {
      // Add the lifted function to the module.
      LOG(INFO) << "add lifted function";
      module_->Add(global, lifted_func);
      LOG(INFO) << "After added lifted func: " << AsText(module_, false);
    }

    // return std::move(global);

    if (captured_vars.size() == 0) {
      return std::move(global);
    } else {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : captured_vars) {
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
      LOG(INFO) << "Lifting " << AsText(func, false);
      func = FunctionNode::make(func->params,
                                VisitExpr(func->body),
                                func->ret_type,
                                func->type_params,
                                func->attrs);
      LOG(INFO) << "new func: " << AsText(func, false);
      module_->Add(pair.first, func, true);
    }
    LOG(INFO) << "After lambda lift: " << AsText(module_, false);
    return module_;
  }

 private:
  std::unordered_map<Var, GlobalVar, NodeHash, NodeEqual> lambda_map_;
  std::unordered_map<Var, Array<Var>, NodeHash, NodeEqual> lambda_captured_vars_;
  std::vector<Var> letrec_;
  Module module_;
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
