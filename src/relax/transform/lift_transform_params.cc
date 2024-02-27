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
 * \file tvm/relax/transform/lift_transform_params.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "../../support/ordered_set.h"
#include "utils.h"

namespace tvm {
namespace relax {

namespace {

struct CollectInfo {
  /* \brief The analyzed function */
  Function orig_func;

  /* \brief The number of parameters unknown until runtime */
  size_t num_runtime_params;

  /*! \brief Bindings that can be lifted out into a pre-processing
   *
   * - All bindings in `computable_at_compile_time` are suitable for
   *   use in a DataflowBlock.
   *
   * - Do not depend on any parameter prior to attr::kNumInput.
   *
   * - Does not include "relax.builtin.stop_lift_params"
   */
  std::vector<Binding> computable_at_compile_time;

  /*! \brief Variables that are required at runtime */
  std::unordered_set<Variant<relax::Var, tir::Var>, ObjectPtrHash, ObjectPtrEqual>
      required_at_runtime;

  Array<Var> GetCompileTimeInputs() const {
    return Array<Var>(orig_func->params.begin() + num_runtime_params, orig_func->params.end());
  }

  Array<Var> GetRuntimeInputs() const {
    return Array<Var>(orig_func->params.begin(), orig_func->params.begin() + num_runtime_params);
  }

  Array<tir::Var> GetPropagatedSymbolicVariables() const {
    auto vars_from_any_param =
        DefinableTIRVarsInStructInfo(TupleStructInfo(orig_func->params.Map(GetStructInfo)));

    auto vars_from_runtime_params =
        [&]() -> std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> {
      auto tir_var_vec =
          DefinableTIRVarsInStructInfo(TupleStructInfo(GetRuntimeInputs().Map(GetStructInfo)));
      return {tir_var_vec.begin(), tir_var_vec.end()};
    }();

    auto vars_from_transformed_params =
        [&]() -> std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> {
      auto tir_var_vec =
          DefinableTIRVarsInStructInfo(TupleStructInfo(GetCompileTimeOutputs().Map(GetStructInfo)));
      return {tir_var_vec.begin(), tir_var_vec.end()};
    }();

    Array<tir::Var> output;
    for (const auto& tir_var : vars_from_any_param) {
      if (required_at_runtime.count(tir_var) && !vars_from_runtime_params.count(tir_var) &&
          !vars_from_transformed_params.count(tir_var)) {
        output.push_back(tir_var);
      }
    }
    return output;
  }

  Array<Var> GetCompileTimeOutputs() const {
    Array<Var> params;

    // Any value that is available at compile-time, but is also
    // required at runtime, must be passed through the compile-time
    // function.
    for (size_t i = num_runtime_params; i < orig_func->params.size(); i++) {
      Var var = orig_func->params[i];
      if (required_at_runtime.count(var)) {
        params.push_back(var);
      }
    }

    // Any variable that is computed at compile-time, but is required
    // at runtime, must be provided as a parameter.
    for (const auto& binding : computable_at_compile_time) {
      if (required_at_runtime.count(binding->var)) {
        params.push_back(binding->var);
      }
    }

    return params;
  }

  Function MakeCompileTimeFunction() const {
    auto compile_time_params = GetCompileTimeInputs();

    Array<Binding> output_var_binding;
    Array<Expr> output_exprs;

    // Any symbolic variables that are inferrable from compile-time
    // parameters, but are not inferrable from run-time parameters,
    // must be propagated to the output.
    if (auto propagated_tir_vars = GetPropagatedSymbolicVariables(); propagated_tir_vars.size()) {
      output_exprs.push_back(
          ShapeExpr(propagated_tir_vars.Map([](tir::Var var) -> PrimExpr { return var; })));
    }

    for (const auto& var : GetCompileTimeOutputs()) {
      Var out_var(var->name_hint() + "_output", GetStructInfo(var));
      output_var_binding.push_back(VarBinding(out_var, var));
      output_exprs.push_back(out_var);
    }

    Var tuple_var("output_tuple", TupleStructInfo(output_exprs.Map(GetStructInfo)));
    output_var_binding.push_back(VarBinding(tuple_var, Tuple(output_exprs)));

    SeqExpr body(
        {
            DataflowBlock(computable_at_compile_time),
            DataflowBlock(output_var_binding),
        },
        tuple_var);

    Function func(compile_time_params, body, GetStructInfo(tuple_var));
    func = WithAttr(func, attr::kNumInput, Integer(0));
    func = CopyWithNewVars(func);
    func = Downcast<Function>(CanonicalizeBindings(func));
    return func;
  }

  Function MakeRuntimeFunction() const {
    Array<Binding> bindings;

    // Any parameter that isn't available until runtime must be an
    // input, along with any output from the compile-time function.
    // Compile-time outputs must have a fresh non-dataflow var to
    // serve as the parameter.  This trivial binding will later be
    // removed with CanonicalizeBindings.
    Array<Var> params = GetRuntimeInputs();
    if (auto propagated_tir_vars = GetPropagatedSymbolicVariables(); propagated_tir_vars.size()) {
      ShapeStructInfo shape_sinfo(
          propagated_tir_vars.Map([](tir::Var var) -> PrimExpr { return var; }));
      Var shape_expr("vars_from_compile_time_params", shape_sinfo);
      params.push_back(shape_expr);
    }
    for (const auto& var : GetCompileTimeOutputs()) {
      Var param_var(var->name_hint(), GetStructInfo(var));
      bindings.push_back(VarBinding(var, param_var));
      params.push_back(param_var);
    }

    // Any binding that is computable at compile-time should be
    // suppressed at run-time.
    struct SuppressCompileTime : ExprMutator {
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> to_suppress;
      explicit SuppressCompileTime(const std::vector<Binding>& bindings) {
        for (const auto& binding : bindings) {
          to_suppress.insert(binding->var);
        }
      }

      void VisitBinding(const Binding& binding) override {
        if (!to_suppress.count(binding->var)) {
          ExprMutator::VisitBinding(binding);
        }
      }

      using ExprMutator::VisitExpr_;
      Expr VisitExpr_(const CallNode* call) override {
        static const Op& stop_lift_params_op = Op::Get("relax.builtin.stop_lift_params");
        if (call->op.same_as(stop_lift_params_op)) {
          return VisitExpr(call->args[0]);
        } else {
          return ExprMutator::VisitExpr_(call);
        }
      }
    };
    Expr body = SuppressCompileTime(computable_at_compile_time)(orig_func->body);
    body = SeqExpr({DataflowBlock(bindings)}, body);

    Function func(params, body, orig_func->ret_struct_info, orig_func->is_pure, orig_func->attrs);
    func = WithoutAttr(func, tvm::attr::kGlobalSymbol);
    func = CopyWithNewVars(func);
    return func;
  }

  Function MakePartitionedFunction() const {
    Array<Binding> inner_func_bindings;
    Var compile_time_func = [&]() {
      auto func = MakeCompileTimeFunction();
      Var var("transform_params", GetStructInfo(func));
      inner_func_bindings.push_back(VarBinding(var, std::move(func)));
      return var;
    }();
    Var runtime_func = [&]() {
      auto func = MakeRuntimeFunction();
      Var var("runtime", GetStructInfo(func));
      inner_func_bindings.push_back(VarBinding(var, std::move(func)));
      return var;
    }();

    Array<Binding> calling_scope;

    Call compile_time_preprocess(
        compile_time_func, GetCompileTimeInputs().Map([](const Var& var) -> Expr { return var; }));

    // Use a fresh variable in case it is passed through unmodified in
    // the compile-time function.
    Array<Var> compile_time_outputs;
    if (auto propagated_tir_vars = GetPropagatedSymbolicVariables(); propagated_tir_vars.size()) {
      ShapeStructInfo shape_sinfo(
          propagated_tir_vars.Map([](tir::Var var) -> PrimExpr { return var; }));
      Var shape_expr("vars_from_compile_time_params", shape_sinfo);
      compile_time_outputs.push_back(shape_expr);
    }
    for (const auto& relax_var : GetCompileTimeOutputs()) {
      compile_time_outputs.push_back(
          Var(relax_var->name_hint(), GetStructInfo(relax_var), relax_var->span));
    }
    {
      Var tuple_output("compile_time_output",
                       TupleStructInfo(compile_time_outputs.Map(GetStructInfo)));
      calling_scope.push_back(VarBinding(tuple_output, compile_time_preprocess));
      for (size_t i = 0; i < compile_time_outputs.size(); i++) {
        calling_scope.push_back(VarBinding(compile_time_outputs[i], TupleGetItem(tuple_output, i)));
      }
    }

    Array<Expr> runtime_args = GetRuntimeInputs().Map([](const Var& var) -> Expr { return var; });
    for (const auto& var : compile_time_outputs) {
      runtime_args.push_back(var);
    }

    Call runtime_execution(runtime_func, runtime_args);
    Var output_var("output", orig_func->ret_struct_info);
    calling_scope.push_back(VarBinding(output_var, runtime_execution));

    SeqExpr body(
        {
            BindingBlock(inner_func_bindings),
            DataflowBlock(calling_scope),
        },
        output_var);

    Function func = orig_func;
    func.CopyOnWrite()->body = body;
    func = Downcast<Function>(CanonicalizeBindings(func));
    return func;
  }
};

class LiftableBindingCollector : ExprVisitor {
 public:
  static CollectInfo Collect(const Function& func) {
    LiftableBindingCollector visitor;
    visitor(func);
    visitor.info_.orig_func = func;
    return visitor.info_;
  }

 private:
  void VisitExpr_(const FunctionNode* func) override {
    size_t num_runtime_params = func->params.size();
    if (auto opt = func->attrs.GetAttr<Integer>(attr::kNumInput)) {
      num_runtime_params = opt.value()->value;
    }

    info_.num_runtime_params = num_runtime_params;

    for (size_t i = num_runtime_params; i < func->params.size(); i++) {
      liftable_vars_.insert(func->params[i]);
      for (const auto& tir_var : DefinableTIRVarsInStructInfo(GetStructInfo(func->params[i]))) {
        liftable_vars_.insert(tir_var);
      }
    }
    ExprVisitor::VisitExpr_(func);
  }

  void VisitBindingBlock_(const DataflowBlockNode* block) final {
    bool cache = is_in_dataflow_block_;
    is_in_dataflow_block_ = true;
    ExprVisitor::VisitBindingBlock_(block);
    is_in_dataflow_block_ = cache;
  }

  void VisitBinding(const Binding& binding) override {
    if (CanLiftBinding(binding)) {
      info_.computable_at_compile_time.push_back(binding);
      liftable_vars_.insert(binding->var);
    } else {
      info_.required_at_runtime.insert(binding->var);
      auto bound_value = GetBoundValue(binding);
      for (const auto& upstream_var : FreeVars(bound_value)) {
        info_.required_at_runtime.insert(upstream_var);
      }
      for (const auto& tir_var : FreeSymbolicVars(bound_value)) {
        info_.required_at_runtime.insert(tir_var);
      }
    }
  }

  bool CanLiftBinding(const Binding& binding) const {
    auto value = GetBoundValue(binding);

    // Cond 1. Do not lift bindings outside dataflow blocks.
    if (!is_in_dataflow_block_) {
      return false;
    }

    // Cond 2. Do not lift regarding the "builtin.stop_lift_params" op.
    if (const auto* call = value.as<CallNode>()) {
      static const Op& stop_lift_params_op = Op::Get("relax.builtin.stop_lift_params");
      if (call->op.same_as(stop_lift_params_op)) {
        return false;
      }
    }

    // Cond 3. Do not lift when involving Vars that are not liftable.
    for (const auto& var : FreeVars(value)) {
      if (!liftable_vars_.count(var)) {
        return false;
      }
    }

    // Cond 4. Do not lift when its struct info contains symbolic variables that do not appear in
    // params.
    for (const auto& var : TIRVarsInStructInfo(GetStructInfo(binding->var))) {
      if (!liftable_vars_.count(var)) {
        return false;
      }
    }

    // Cond 5. Do not lift declarations of external functions
    if (value.as<relax::ExternFuncNode>()) {
      return false;
    }

    return true;
  }

  CollectInfo info_;
  std::unordered_set<Variant<Var, tir::Var>, ObjectPtrHash, ObjectPtrEqual> liftable_vars_;
  bool is_in_dataflow_block_{false};
};

class PreprocessPartitioner : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const FunctionNode* op) override {
    auto func = GetRef<Function>(op);
    if (func->attrs.GetAttr<Integer>(attr::kNumInput)) {
      auto info = LiftableBindingCollector::Collect(func);
      return info.MakePartitionedFunction();
    } else {
      return func;
    }
  }
};

// Adapted from https://stackoverflow.com/a/2072890
inline bool ends_with(const std::string& value, const std::string& ending) {
  return ending.size() <= value.size() &&
         std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

}  // namespace

namespace transform {

Pass PartitionTransformParams() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    PreprocessPartitioner mutator;

    IRModule updates;
    for (const auto& [gvar, func] : mod->functions) {
      if (auto opt = func.as<relax::Function>()) {
        auto new_func = Downcast<Function>(mutator(opt.value()));
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
  return tvm::transform::CreateModulePass(pass_func, 1, "PartitionTransformParams", {});
}

Pass LiftTransformParams() {
  // A post-proc utility as as the third step in LiftTransformParams
  //
  // 1. PartitionTransformParams: Partition each function into a
  // compile-time and run-time lambda functions.
  //
  // 2. LambdaLift: Lift the compile-time and run-time lambda
  // functions out of the end-to-end function.
  //
  // 3. Post-proc: Expose the compile-time and run-time functions for
  // external use, replacing the end-to-end functions.
  auto post_proc_func = [=](IRModule mod, PassContext pc) {
    std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> to_remove;
    std::unordered_map<GlobalVar, Function, ObjectPtrHash, ObjectPtrEqual> to_add;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<Function>()) {
        auto func = opt.value();

        std::string func_name = gvar->name_hint;
        if (ends_with(func_name, "transform_params")) {
          func = WithAttr(func, tvm::attr::kGlobalSymbol, gvar->name_hint);
          func = BundleModelParams(func);
          to_add[gvar] = func;
        } else if (ends_with(func_name, "_runtime")) {
          std::string name(func_name.begin(), func_name.end() - sizeof("_runtime") + 1);
          to_remove.insert(mod->GetGlobalVar(name));
          to_remove.insert(gvar);
          to_add[GlobalVar(name)] = WithAttr(func, tvm::attr::kGlobalSymbol, String(name));
        }
      }
    }

    if (to_remove.size() || to_add.size()) {
      auto write_ptr = mod.CopyOnWrite();
      for (const auto& gvar : to_remove) {
        write_ptr->Remove(gvar);
      }
      for (const auto& [gvar, func] : to_add) {
        write_ptr->Add(gvar, func);
      }
    }

    return mod;
  };
  auto post_proc =
      tvm::transform::CreateModulePass(post_proc_func, 1, "LiftTransformParamsPostProc", {});

  return tvm::transform::Sequential(
      {
          PartitionTransformParams(),
          LambdaLift(),
          post_proc,
      },
      "LiftTransformParams");
}

TVM_REGISTER_GLOBAL("relax.transform.LiftTransformParams").set_body_typed(LiftTransformParams);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
