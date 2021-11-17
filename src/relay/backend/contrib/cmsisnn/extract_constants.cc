
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
 * \file extract_constant.cc
 * \brief Pushes out constants within partitioned functions all the way upto main()
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>

#include "../../../qnn/utils.h"
#include "../../../transforms/pattern_utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief This Mutator finds all functions with constants. Constants are replaced with function
 * parameter variables. Constants are pushed all the way upto main().
 */
class ExtractConstantsMutator : public MixedModeMutator {
 public:
  explicit ExtractConstantsMutator(const IRModule& mod) : mod_(mod) {}

 private:
  String gen_var_name() { return "tvm_var_extract_const_" + std::to_string(var_count_++); }

  Expr VisitExpr_(const FunctionNode* function) final {
    Function func = GetRef<Function>(function);
    function_to_constants_.Set(func, Array<Constant>{});
    functions_.push_back(func);
    auto new_body = VisitExpr(func->body);
    functions_.pop_back();
    if (function_to_constants_[func].size()) {
      func = Function(FreeVars(new_body), new_body, func->ret_type, FreeTypeVars(new_body, mod_),
                      func->attrs);
    }
    return func;
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr final_call = post;
    auto* post_call = post.as<CallNode>();

    // Replace Constant arguments with Vars for ML Operators
    // Perform this for non-main Call Nodes only
    if (!functions_.empty() && call->op.as<OpNode>()) {
      Array<Expr> new_args;
      for (auto& arg : post_call->args) {
        auto* const_arg = arg.as<ConstantNode>();
        if (const_arg && !const_arg->is_scalar()) {
          Var var_arg = Var(gen_var_name(), const_arg->tensor_type());
          new_args.push_back(var_arg);
          const Function& last_func = functions_.back();
          Array<Constant> fconstants(function_to_constants_[last_func]);
          fconstants.push_back(GetRef<Constant>(const_arg));
          function_to_constants_.Set(last_func, fconstants);
        } else {
          new_args.push_back(arg);
        }
      }
      final_call = Call(call->op, new_args, call->attrs, {});
    }

    // Since the constants are kicked out of partitioned functions
    // a new call to global function is needed
    if (auto* glob_var_node = post_call->op.as<GlobalVarNode>()) {
      auto glob_var = GetRef<GlobalVar>(glob_var_node);
      auto glob_func = Downcast<Function>(mod_->Lookup(glob_var));
      auto new_glob_func = VisitExpr(glob_func);
      if (!new_glob_func.same_as(glob_func)) {
        mod_->Update(glob_var, Downcast<Function>(new_glob_func));
        Array<Expr> new_args = post_call->args;
        ICHECK(function_to_constants_.find(glob_func) != function_to_constants_.end());
        for (auto constant : function_to_constants_.at(glob_func)) {
          new_args.push_back(constant);
        }
        final_call = Call(glob_var, new_args);
      }
    }

    // Since the constants are kicked out of the local partitioned functions
    // a new call to local function is needed
    // Also, pass on the constants to the callee of this function to support nested functions
    if (auto* func_node = call->op.as<FunctionNode>()) {
      Function func = GetRef<Function>(func_node);
      auto new_func = VisitExpr(func);
      if (!new_func.same_as(func)) {
        Array<Expr> new_args = post_call->args;
        ICHECK(function_to_constants_.find(func) != function_to_constants_.end());
        const Function& last_func = functions_.back();
        Array<Constant> fconstants(function_to_constants_[last_func]);
        for (auto constant : function_to_constants_.at(func)) {
          fconstants.push_back(constant);
          Var var_arg = Var(gen_var_name(), constant->tensor_type());
          new_args.push_back(var_arg);
        }
        function_to_constants_.Set(last_func, fconstants);
        final_call = Call(new_func, new_args);
      }
    }

    return final_call;
  }

 private:
  /* \brief Updated module where all calls have replaced constants with new variables */
  IRModule mod_;
  /* \brief Maintains mapping of original function to the replaced constants */
  Map<Function, Array<Constant>> function_to_constants_;
  /* \brief Stack of functions to determine scope while filling up function_to_constants_ */
  Array<Function> functions_;
  /* \brief Keeps track of variables being created */
  int var_count_ = 0;
};

/*!  * \brief Kicks out all constants out of the partitioned function into main()  */
IRModule ExtractConstants(const IRModule& mod) {
  String func_name;
  Function func;

  auto extract_constants = ExtractConstantsMutator(mod);
  Function main_func = Downcast<Function>(mod->Lookup("main"));
  auto new_main_body = extract_constants.VisitExpr(main_func->body);
  if (!new_main_body.same_as(main_func->body)) {
    auto main_var = mod->GetGlobalVar("main");
    auto new_main_func = Function(main_func->params, new_main_body, main_func->ret_type,
                                  main_func->type_params, main_func->attrs);
    mod->Update(main_var, new_main_func);
  }
  return mod;
}

transform::Pass ExtractConstantsFromPartitionedFunction() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule m, transform::PassContext pc) { return ExtractConstants(m); };
  return tvm::transform::CreateModulePass(pass_func, 0, "ExtractConstantsFromPartitionedFunction",
                                          {});
}

TVM_REGISTER_GLOBAL("relay.ext.cmsisnn.transform.ExtractConstantsFromPartitionedFunction")
    .set_body_typed(ExtractConstantsFromPartitionedFunction);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
