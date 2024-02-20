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
 * \file tvm/relax/transform/bundle_model_params.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include "utils.h"

namespace tvm {
namespace relax {

class ModelParamBundler : public ExprMutator {
 public:
  explicit ModelParamBundler(Optional<String> param_tuple_name)
      : param_tuple_name_(param_tuple_name) {}

  Expr VisitExpr_(const FunctionNode* op) override {
    Function func = GetRef<Function>(op);
    auto opt_num_input = func->attrs.GetAttr<Integer>(attr::kNumInput);
    if (!opt_num_input) return func;
    auto signed_num_input = opt_num_input.value()->value;

    ICHECK_GE(signed_num_input, 0);
    ICHECK_LE(signed_num_input, func->params.size())
        << "Function was declared to have " << signed_num_input << " runtime inputs, "
        << "but only has " << func->params.size() << " parameters total.";
    size_t num_input = signed_num_input;

    Array<Var> params;
    for (size_t i = 0; i < num_input; i++) {
      params.push_back(func->params[i]);
    }

    Array<StructInfo> param_tuple;
    for (size_t i = num_input; i < func->params.size(); i++) {
      param_tuple.push_back(GetStructInfo(func->params[i]));
    }

    Var var_param_tuple(param_tuple_name_.value_or("model_params"), TupleStructInfo(param_tuple));
    params.push_back(var_param_tuple);

    for (size_t i = num_input; i < func->params.size(); i++) {
      var_to_expr_.Set(func->params[i], TupleGetItem(var_param_tuple, i - num_input));
    }

    func.CopyOnWrite()->params = params;

    return ExprMutator::VisitExpr_(func.get());
  }

  Expr VisitExpr_(const VarNode* op) override {
    auto var = GetRef<Var>(op);
    if (auto it = var_to_expr_.find(var); it != var_to_expr_.end()) {
      return builder_->Emit((*it).second, op->name_hint());
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

 private:
  Optional<String> param_tuple_name_;
  Map<Var, Expr> var_to_expr_;
};

Function BundleModelParams(const Function& func, Optional<String> param_tuple_name) {
  ModelParamBundler mutator(param_tuple_name);
  return Downcast<Function>(mutator(func));
}

namespace transform {
Pass BundleModelParams(Optional<String> param_tuple_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    IRModule updates;

    ModelParamBundler mutator(param_tuple_name);

    for (const auto& [gvar, func] : mod->functions) {
      if (auto opt = func.as<relax::Function>()) {
        auto new_func = Downcast<relax::Function>(mutator(opt.value()));
        if (!new_func.same_as(func)) {
          updates->Add(gvar, new_func);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };
  return CreateModulePass(pass_func, 1, "BundleModelParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.BundleModelParams").set_body_typed(BundleModelParams);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
