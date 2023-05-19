
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

  using MixedModeMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionNode* function) final {
    Function func = GetRef<Function>(function);
    auto composite_name = func->GetAttr<String>(attr::kComposite);
    if (composite_name.defined()) {
      std::string name = composite_name.value().operator std::string();
      if (name.find("cmsis-nn") == std::string::npos) {
        return func;
      }
    }
    auto compiler_name = func->GetAttr<String>(::tvm::relay::attr::kCompiler);
    if (compiler_name.defined() && compiler_name != "cmsis-nn") {
      return func;
    }

    function_to_arguments_.Set(func, Array<Expr>{});
    functions_.push_back(func);
    auto new_body = VisitExpr(func->body);
    functions_.pop_back();
    if (function_to_arguments_[func].size()) {
      func = WithFields(func, FreeVars(new_body), new_body, func->ret_type,
                        FreeTypeVars(new_body, mod_), func->attrs);
    }
    return std::move(func);
  }

  // Creates new arguments from current call's arguments
  // Updates constants into the caller arguments: here caller signifies caller that comprises call
  // to func
  Array<Expr> CreateNewCallArgsFromExtractedConstants(Call call, Function func) {
    ICHECK(function_to_arguments_.find(func) != function_to_arguments_.end());
    Array<Expr> function_signature(function_to_arguments_[func]);

    // Is func a global_function?
    // main() is not registered for extracting constants
    bool is_global_function = functions_.empty() ? true : false;

    bool new_constants_added = false;
    // This tracks arguments traversed inside function_signature
    uint32_t function_signature_id = 0;
    // This contains arguments including constants for the caller of this function inside which
    // post_call resides.
    Array<Expr> new_caller_args;
    // New arguments to post_call that includes new variables representing constants extracted from
    // the function
    Array<Expr> new_call_args;
    for (auto& arg : call->args) {
      if (auto* constant = arg.as<ConstantNode>()) {
        new_caller_args.push_back(arg);
        new_call_args.push_back(Var(gen_var_name(), constant->tensor_type()));
        ++function_signature_id;
        new_constants_added = true;
        continue;
      }

      // Push all constants from the function_signature until a variable corresponding to the
      // current argument is hit
      while (function_signature_id < function_signature.size()) {
        auto* constant = function_signature[function_signature_id].as<ConstantNode>();
        if (constant == nullptr) {
          break;
        }
        new_caller_args.push_back(function_signature[function_signature_id++]);
        new_call_args.push_back(Var(gen_var_name(), constant->tensor_type()));
        new_constants_added = true;
      }

      new_call_args.push_back(arg);
      if (is_global_function || arg.as<VarNode>()) {
        new_caller_args.push_back(arg);
      }
      ++function_signature_id;
    }

    // Push remaining constants as new arguments
    for (uint32_t i = function_signature_id; i < function_signature.size(); ++i) {
      auto* constant = function_signature[i].as<ConstantNode>();
      ICHECK(constant)
          << "Rest of the collected arguments should be constant in the partitioned function.";
      new_caller_args.push_back(GetRef<Constant>(constant));
      new_call_args.push_back(Var(gen_var_name(), constant->tensor_type()));
      new_constants_added = true;
    }

    // Update the arguments of caller of local function
    if (new_constants_added && !is_global_function) {
      const Function& last_func = functions_.back();
      Array<Expr> function_constants(function_to_arguments_[last_func]);
      function_to_arguments_.Set(last_func,
                                 tvm::runtime::Concat(function_constants, new_caller_args));
    } else {
      new_call_args = new_caller_args;
    }

    return new_call_args;
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr final_call = post;
    auto* post_call = post.as<CallNode>();

    // Replace Constant arguments with Vars for ML Operators
    // Perform this for non-main Call Nodes only
    if (!functions_.empty() && call->op.as<OpNode>()) {
      Array<Expr> new_args;
      const Function& last_func = functions_.back();
      Array<Expr> function_signature(function_to_arguments_[last_func]);
      for (auto& arg : post_call->args) {
        // Push all arguments including constants to maintain correct order of
        // variables and constants
        auto* const_arg = arg.as<ConstantNode>();
        if (const_arg && !const_arg->is_scalar()) {
          Var var_arg = Var(gen_var_name(), const_arg->tensor_type());
          new_args.push_back(var_arg);
          function_signature.push_back(arg);
        } else {
          if (arg.as<VarNode>()) {
            // Only push if its not already present as multiple consumers of any input var
            // will appear only once in the function signature.
            bool found_in_existing_signature = false;
            for (auto& sign : function_signature) {
              if (arg.same_as(sign)) {
                found_in_existing_signature = true;
                break;
              }
            }
            if (!found_in_existing_signature) {
              function_signature.push_back(arg);
            }
          }
          new_args.push_back(arg);
        }
      }
      function_to_arguments_.Set(last_func, function_signature);
      final_call = Call(call->op, new_args, call->attrs, {});
    }

    // Since the constants are extracted from partitioned functions
    // a new call to global function is needed
    if (auto opt = post_call->op.as<GlobalVar>()) {
      auto glob_var = opt.value();
      auto glob_func = Downcast<Function>(mod_->Lookup(glob_var));
      auto new_glob_func = VisitExpr(glob_func);
      if (!new_glob_func.same_as(glob_func)) {
        mod_->Update(glob_var, Downcast<Function>(new_glob_func));
        auto new_args = CreateNewCallArgsFromExtractedConstants(GetRef<Call>(post_call), glob_func);
        final_call = Call(glob_var, new_args);
      }
    }

    // Since the constants are extracted from the local partitioned functions
    // a new call to local function is needed
    if (auto opt = call->op.as<Function>()) {
      Function func = opt.value();
      auto new_func = VisitExpr(func);
      Array<Expr> new_args = CreateNewCallArgsFromExtractedConstants(GetRef<Call>(post_call), func);
      final_call = Call(new_func, new_args);
    }

    final_call->span = call->span;
    return final_call;
  }

 private:
  /* \brief Updated module where all calls have replaced constants with new variables */
  IRModule mod_;
  /* \brief Maintains mapping of original function to the replaced constants along with other
   * arguments to retain the order in which variables are used within the function */
  Map<Function, Array<Expr>> function_to_arguments_;
  /* \brief Stack of functions to determine scope while filling up function_to_arguments_ */
  Array<Function> functions_;
  /* \brief Keeps track of variables being created */
  int var_count_ = 0;
};

/*!  * \brief Extracts all constants out of the partitioned function into main()  */
IRModule ExtractConstants(const IRModule& mod) {
  String func_name;
  Function func;

  auto extract_constants = ExtractConstantsMutator(mod);
  Function main_func = Downcast<Function>(mod->Lookup("main"));
  auto new_main_body = extract_constants.VisitExpr(main_func->body);
  if (!new_main_body.same_as(main_func->body)) {
    auto main_var = mod->GetGlobalVar("main");
    Function new_main_func = WithFields(main_func, main_func->params, new_main_body);
    mod->Update(main_var, new_main_func);
  }
  return mod;
}

transform::Pass ExtractConstantsFromPartitionedFunction() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule m, transform::PassContext pc) { return ExtractConstants(m); };
  return tvm::transform::CreateModulePass(pass_func, 0, "ExtractConstantsFromPartitionedFunction",
                                          {"InferType"});
}

TVM_REGISTER_GLOBAL("relay.ext.cmsisnn.transform.ExtractConstantsFromPartitionedFunction")
    .set_body_typed(ExtractConstantsFromPartitionedFunction);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
