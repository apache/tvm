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
 * \file tvm/relax/transform/lambda_lift.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

#include "../../support/ordered_set.h"
#include "utils.h"

namespace tvm {
namespace relax {

/*! \brief Plan of lifting transform params */
struct LiftTransformParamsInfoPlan {
  Function f_transform_params;  // the lifted function that transforms the parameters
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
      output_to_index;  // the index of the original bindings in the output tuple
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
      lifted_bindings;  // the bindings of the original function that are lifted
};

/*! \brief Builder of the function that transforms the parameters. */
class TransformParamsFuncBuilder : public ExprMutator {
 public:
  TransformParamsFuncBuilder() { builder_->BeginDataflowBlock(); }

  /*! \brief Add a input parameter. */
  void AddInput(const Var& var) {
    inputs_.push_back(var);
    lifted_binding_lookup_.insert(var);
  }

  void UpdateBasedOnRuntimeInput(const Var& var) {
    for (const auto& var : DefinableTIRVarsInStructInfo(GetStructInfo(var))) {
      known_symbolic_var_during_inference_.insert(var);
    }
    for (const auto& var : TIRVarsInStructInfo(GetStructInfo(var))) {
      required_symbolic_var_during_inference_.insert(var);
    }
  }

  /*! \brief Add a binding to lift. */
  void AddInternalBinding(const VarBinding& binding) {
    bindings_.push_back(binding);
    lifted_binding_lookup_.insert(binding->var);
  }

  /*! \brief Update based on bindings not being lifted. */
  void UpdateBasedOnRuntimeBinding(const VarBinding& binding) {
    for (const auto& producer : FreeVars(binding->value)) {
      // An external value that uses a lifted binding requires the
      // lifted binding to be returned as output.
      if (lifted_binding_lookup_.count(producer)) {
        outputs_.insert(producer);

        for (const auto& var : DefinableTIRVarsInStructInfo(GetStructInfo(producer))) {
          known_symbolic_var_during_inference_.insert(var);
        }
      }
    }

    // All TIR variables used in the binding must be available at runtime.
    for (const auto& var : FreeSymbolicVars(binding->value)) {
      required_symbolic_var_during_inference_.insert(var);
    }
  }

  bool UsesOnlyLiftableProducers(const Expr& expr) {
    auto producers = FreeVars(expr);
    bool uses_only_liftable_producers = [&]() {
      return std::all_of(producers.begin(), producers.end(),
                         [&](const auto& var) { return lifted_binding_lookup_.count(var); });
    }();
    return uses_only_liftable_producers;
  }

  /*!
   * \brief Build the function that transforms the parameters
   * \return The created function, and a map from the variable in the original function to the index
   * of the element of the output tuple
   */
  std::pair<Function, std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>> Build() {
    Array<PrimExpr> extra_symbolic_vars;
    for (const auto& var : required_symbolic_var_during_inference_) {
      if (!known_symbolic_var_during_inference_.count(var)) {
        extra_symbolic_vars.push_back(var);
      }
    }

    Array<StructInfo> input_sinfo;
    Array<Expr> output_vars;
    std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> output_to_index;

    for (const auto& input : inputs_) {
      input_sinfo.push_back(Downcast<StructInfo>(input->struct_info_.value()));
    }
    Var params("params", TupleStructInfo(input_sinfo));

    if (extra_symbolic_vars.size()) {
      output_vars.push_back(builder_->Emit(ShapeExpr(extra_symbolic_vars), "extra_symbolic_vars"));
    }

    // Helper to add a variable to the output tuple
    // original_var: the binding variable in the original function
    // output_var: the variable, which is a binding in the transform_params function, that is added
    // to the output tuple
    auto f_add_output = [&](const Var& original_var, const Var& output_var) -> void {
      output_to_index[original_var] = output_vars.size();
      output_vars.push_back(output_var);
    };

    // Create mapping from the original input variables to the TupleGetItem from the packed
    // parameter tuple Add the parameters that are marked as the output of the function to the
    // output tuple
    for (const auto& input : inputs_) {
      input_remap_.emplace(input.get(), TupleGetItem(params, input_remap_.size()));
      if (outputs_.count(input)) {
        auto output_var = builder_->Emit(input_remap_.at(input.get()));
        f_add_output(input, output_var);
      }
    }

    // Re-emit the bindings that are lifted. Update the output tuple if the binding is marked as the
    // output.
    for (const auto& binding : bindings_) {
      if (outputs_.count(binding->var)) {
        auto output_var = builder_->Emit(VisitExpr(binding->value));
        var_remap_[binding->var->vid] = output_var;
        f_add_output(binding->var, output_var);
      } else {
        VisitBinding(binding);
      }
    }

    // Create the function.
    Expr transformed_params = builder_->EmitOutput(Tuple(output_vars));
    BindingBlock block = builder_->EndBlock();
    Expr body = VisitWithNewScope(SeqExpr({block}, transformed_params), Array<Var>{params});
    Function f_transform_params =
        Function(/*params=*/{params}, /*body=*/body, /*ret_struct_info=*/NullOpt);
    return {f_transform_params, output_to_index};
  }

  Expr VisitExpr_(const VarNode* var) final {
    if (auto it = input_remap_.find(var); it != input_remap_.end()) {
      return builder_->Emit((*it).second);
    } else {
      return ExprMutator::VisitExpr_(var);
    }
  }

  // The input parameters of the function.
  Array<Var> inputs_;
  // Remap from the original input variable to TupleGetItem from the packed parameter tuple, which
  // is the input of the lifted function.
  std::unordered_map<const VarNode*, Expr> input_remap_;
  // The bindings that are lifted.
  Array<VarBinding> bindings_;
  // The variables that are marked as the output of the function.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> outputs_;

  // The bindings that are lifted
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> lifted_binding_lookup_;

  /* Symbolic variables that are known during the transform_params execution.
   *
   * This set is populated based on the variables declared with
   * AddInput, and contains variables that may appear inside the
   * transformation function.  A binding that depends on a symbolic
   * variable not contained in this set may not be lifted.
   */
  support::OrderedSet<tir::Var> known_symbolic_var_during_transform_;

  /* Symbolic variables that are known during the runtime
   *
   * This set is populated based on the variables declared with
   * UpdateBasedOnRuntimeInput, and contains variables that are
   * defined at runtime.  A variable that present in
   * required_symbolic_var_during_inference_, but not present in this
   * set, causes the Build() function to output an additional
   * R.ShapeExpr in order to propagate the symbolic variables.
   */
  support::OrderedSet<tir::Var> known_symbolic_var_during_inference_;

  /* Symbolic variables that must be known at runtime
   *
   * This set is populated based on the variables used in external
   * bindings.  A variable that is present here, but not present in
   * known_symbolic_var_during_inference_, must be provided as an
   * additional R.ShapeExpr parameter from the transform_params
   * function.
   */
  support::OrderedSet<tir::Var> required_symbolic_var_during_inference_;
};

/*!
 * \brief Visitor that creates the plan of lifting transform params.
 *
 * Starting from the parameters of the function (they are the initial set of lifted bindings), we
 * will visit the body of the function to find the bindings that can be lifted. A binding can be
 * lifted if all the variables that it depends on are also lifted.
 *
 * When a binding cannot be lifted, all the variables that 1) it depends on, and 2) have been
 * lifted, will be marked as the boundary variable and will be in the output of the lifted function.
 */
class LiftTransformParamsPlanner : public ExprVisitor {
 public:
  LiftTransformParamsInfoPlan Plan(const Function& function, int num_inputs) {
    for (int i = 0; i < static_cast<int>(function->params.size()); ++i) {
      if (i < num_inputs) {
        builder_.UpdateBasedOnRuntimeInput(function->params[i]);
      } else {
        builder_.AddInput(function->params[i]);
        if (function->params[i]->struct_info_.defined()) {
          Array<tir::Var> symbolic_vars = DefinableTIRVarsInStructInfo(
              Downcast<StructInfo>(function->params[i]->struct_info_.value()));
          for (const auto& var : symbolic_vars) {
            param_symbolic_vars_.insert(var);
          }
        }
      }
    }
    VisitExpr(function->body);

    const auto& [f_transform_params, output_to_index] = builder_.Build();
    return {f_transform_params, output_to_index, std::move(builder_.lifted_binding_lookup_)};
  }

 private:
  void VisitBindingBlock_(const DataflowBlockNode* block) final {
    is_in_dataflow_block_ = true;
    ExprVisitor::VisitBindingBlock_(block);
    is_in_dataflow_block_ = false;
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    bool can_lift = true;

    // Cond 1. Do not lift bindings outside dataflow blocks.
    if (!is_in_dataflow_block_) {
      can_lift = false;
    }

    // Cond 2. Do not lift regarding the "builtin.stop_lift_params" op.
    if (const auto* call = binding->value.as<CallNode>()) {
      static const Op& stop_lift_params_op = Op::Get("relax.builtin.stop_lift_params");
      if (call->op.same_as(stop_lift_params_op)) {
        can_lift = false;
      }
    }

    // Cond 3. Do not lift when involving Vars that are not liftable.
    auto producers = FreeVars(binding->value);
    bool uses_only_liftable_producers = builder_.UsesOnlyLiftableProducers(binding->value);
    if (!uses_only_liftable_producers) {
      can_lift = false;
    }

    // Cond 4. Do not lift when its struct info contains symbolic variables that do not appear in
    // params.
    for (const auto& var : TIRVarsInStructInfo(GetStructInfo(binding->var))) {
      if (!param_symbolic_vars_.count(var)) {
        can_lift = false;
      }
    }

    // Cond 5. Do not lift declarations of external functions
    if (binding->value.as<relax::ExternFuncNode>()) {
      can_lift = false;
    }

    if (can_lift) {
      builder_.AddInternalBinding(GetRef<VarBinding>(binding));
    } else {
      builder_.UpdateBasedOnRuntimeBinding(GetRef<VarBinding>(binding));
    }
  }

  // The builder of the function that transforms the parameters
  TransformParamsFuncBuilder builder_;
  // Whether we are in a dataflow block
  bool is_in_dataflow_block_{false};
  // The symbolic variables in the parameters
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> param_symbolic_vars_;
};

/*!
 *\brief The rewriter that lifts the transform params of a function and updates the original
 * function.
 */
class TransformParamsLifter : ExprMutator {
 public:
  explicit TransformParamsLifter(const IRModule& module) : ExprMutator(module) {}

  Function VisitFunction(GlobalVar gvar, Function func) {
    current_gvar_ = gvar;
    auto out = Downcast<Function>(VisitExpr(std::move(func)));
    current_gvar_ = NullOpt;
    return out;
  }

  Map<GlobalVar, Function> GetTransformParamFunctions() const { return transform_param_funcs_; }

 private:
  Expr VisitExpr_(const FunctionNode* op) override {
    auto func = GetRef<Function>(op);
    Optional<Integer> opt_num_input = func->attrs.GetAttr<Integer>(attr::kNumInput);
    if (!opt_num_input) {
      return func;
    }
    auto signed_num_input = opt_num_input.value()->value;
    ICHECK_GE(signed_num_input, 0);
    ICHECK_LE(signed_num_input, func->params.size());
    size_t num_input = signed_num_input;

    LiftTransformParamsPlanner planner;

    // Step 1: Create the plan of lifting transform params
    lift_plan_ = planner.Plan(func, num_input);

    // Step 2: Stash the lifted function to add to the module
    transform_param_funcs_.Set(current_gvar_.value(), lift_plan_.f_transform_params);

    // Step 3: Update the current function.

    // Step 3.1: Update the function signature
    Array<StructInfo> param_fields =
        Downcast<TupleStructInfo>(lift_plan_.f_transform_params->ret_struct_info)->fields;

    Array<Var> new_params(func->params.begin(), func->params.begin() + num_input);
    for (size_t i = 0; i < param_fields.size(); i++) {
      std::stringstream name;
      name << "transformed_param_" << i;
      Var param(name.str(), param_fields[i]);
      new_params.push_back(param);
    }

    // Step 3.2: Update the function body
    for (const auto& [var, index] : lift_plan_.output_to_index) {
      ICHECK_LT(num_input + index, new_params.size());
      param_remap_[var] = new_params[num_input + index];
    }
    auto new_body = VisitWithNewScope(func->body, new_params);

    return Function(new_params, new_body, func->ret_struct_info, func->is_pure, func->attrs);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    if (lift_plan_.lifted_bindings.count(binding->var)) {
      return;
    }
    if (const auto* call = binding->value.as<CallNode>()) {
      static const Op& stop_lift_params_op = Op::Get("relax.builtin.stop_lift_params");
      if (call->op.same_as(stop_lift_params_op)) {
        var_remap_[binding->var->vid] = Downcast<Var>(VisitExpr(call->args[0]));
        return;
      }
    }
    ExprMutator::VisitBinding_(binding);
  }

  Expr VisitExpr_(const VarNode* var) final {
    auto it = param_remap_.find(GetRef<Var>(var));
    if (it != param_remap_.end()) {
      return builder_->Emit(it->second);
    }
    return ExprMutator::VisitExpr_(var);
  }

  // Remap the original parameters to TupleGetItem from the packed tuple of transformed parameters.
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> param_remap_;
  // The plan of lifting the transform params
  LiftTransformParamsInfoPlan lift_plan_;

  Map<GlobalVar, Function> transform_param_funcs_;
  Optional<GlobalVar> current_gvar_;
};

namespace transform {
Pass LiftTransformParams() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    TransformParamsLifter mutator(mod);

    IRModule updates;
    for (const auto& [gvar, func] : mod->functions) {
      if (auto opt = func.as<relax::Function>()) {
        auto new_func = mutator.VisitFunction(gvar, opt.value());
        if (!new_func.same_as(func)) {
          updates->Add(gvar, new_func);
        }
      }
    }
    for (auto [gvar, transform_func] : mutator.GetTransformParamFunctions()) {
      String name = gvar->name_hint + "_transform_params";
      GlobalVar new_gvar(name);
      new_gvar->struct_info_ = transform_func->struct_info_;

      transform_func = CopyWithNewVars(transform_func);
      transform_func = WithAttr(transform_func, tvm::attr::kGlobalSymbol, name);

      updates->Add(new_gvar, transform_func);
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }

    return mod;
  };
  return CreateModulePass(pass_func, 1, "LiftTransformParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LiftTransformParams").set_body_typed(LiftTransformParams);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
