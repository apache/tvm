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
  void AddInput(const Var& var) { inputs_.push_back(var); }

  /*! \brief Add a binding to lift. */
  void AddBinding(const VarBinding& binding) { bindings_.push_back(binding); }

  /*! \brief Mark a variable as the output of the function. */
  void MarkOutput(const Var& output) { outputs_.insert(output); }

  /*!
   * \brief Build the function that transforms the parameters
   * \return The created function, and a map from the variable in the original function to the index
   * of the element of the output tuple
   */
  std::pair<Function, std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>> Build() {
    Array<StructInfo> input_sinfo;
    Array<Expr> output_vars;
    std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> output_to_index;

    for (const auto& input : inputs_) {
      input_sinfo.push_back(Downcast<StructInfo>(input->struct_info_.value()));
    }
    Var params("params", TupleStructInfo(input_sinfo));

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
    Expr body = builder_->Normalize(SeqExpr({block}, transformed_params));
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
    for (int i = num_inputs; i < static_cast<int>(function->params.size()); ++i) {
      builder_.AddInput(function->params[i]);
      lifted_bindings_.emplace(function->params[i]);
    }
    VisitExpr(function->body);

    const auto& [f_transform_params, output_to_index] = builder_.Build();
    return {f_transform_params, output_to_index, std::move(lifted_bindings_)};
  }

 private:
  void VisitBindingBlock_(const DataflowBlockNode* block) final {
    is_in_dataflow_block_ = true;
    ExprVisitor::VisitBindingBlock_(block);
    is_in_dataflow_block_ = false;
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    std::vector<const VarNode*> producers;
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
    PostOrderVisit(binding->value, [&](const ObjectRef& obj) {
      if (const VarNode* var = obj.as<VarNode>()) {
        producers.push_back(var);
        if (!lifted_bindings_.count(GetRef<Var>(var))) {
          can_lift = false;
        }
      }
    });

    // Cond 4. Do not lift when its struct info contains symbolic variables.
    if (!TIRVarsInStructInfo(GetStructInfo(binding->var)).empty()) {
      can_lift = false;
    }

    if (can_lift) {
      lifted_bindings_.insert(binding->var);
      builder_.AddBinding(GetRef<VarBinding>(binding));
    } else {
      for (const VarNode* producer : producers) {
        if (lifted_bindings_.count(GetRef<Var>(producer))) {
          builder_.MarkOutput(GetRef<Var>(producer));
        }
      }
    }
  }

  // The bindings that are lifted
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> lifted_bindings_;
  // The builder of the function that transforms the parameters
  TransformParamsFuncBuilder builder_;
  // Whether we are in a dataflow block
  bool is_in_dataflow_block_{false};
};

/*!
 *\brief The rewriter that lifts the transform params of a function and updates the original
 * function.
 */
class TransformParamsLifter : public ExprMutator {
 public:
  explicit TransformParamsLifter(const IRModule& module) : ExprMutator(module) {}

  IRModule Lift() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      // Skip non-Relax functions.
      const auto* func_ = base_func.as<FunctionNode>();
      if (func_ == nullptr) {
        continue;
      }
      // Skip functions that do not have the `num_input` attribute.
      Optional<Integer> opt_num_input = func_->attrs.GetAttr<Integer>(attr_num_input_);
      if (!opt_num_input.defined()) {
        continue;
      }
      Function func = RewriteFunc(GetRef<Function>(func_), opt_num_input.value()->value,
                                  gv->name_hint + "_transform_params");
      builder_->UpdateFunction(gv, func);
    }

    return builder_->GetContextIRModule();
  }

 private:
  Function RewriteFunc(const Function& func, int num_input, String new_func_name) {
    LiftTransformParamsPlanner planner;

    // Step 1: Create the plan of lifting transform params
    lift_plan_ = planner.Plan(func, num_input);

    // Step 2: Add the lifted function to the module
    // (The lifted function should be public so we add a global symbol to it)
    auto lift_func =
        WithAttr(lift_plan_.f_transform_params, tvm::attr::kGlobalSymbol, new_func_name);
    builder_->AddFunction(lift_func, new_func_name);

    // Step 3: Update the current function.

    // Step 3.1: Update the function signature
    Var params("params", lift_plan_.f_transform_params->ret_struct_info);
    Array<Var> new_params;
    for (int i = 0; i < num_input; ++i) {
      new_params.push_back(func->params[i]);
    }
    new_params.push_back(params);

    // Step 3.2: Update the function body
    for (const auto& [var, index] : lift_plan_.output_to_index) {
      param_remap_[var] = TupleGetItem(params, index);
    }
    auto new_body = VisitWithNewScope(func->body, new_params);

    // Step 3.3: Remove function attributes that are not needed
    auto new_attrs = func->attrs;
    auto* new_attrs_node = new_attrs.CopyOnWrite();
    new_attrs_node->dict.erase(attr_num_input_);
    if (new_attrs->dict.empty()) {
      new_attrs = NullValue<DictAttrs>();
    }

    Function new_func(new_params, new_body, func->ret_struct_info, func->is_pure, new_attrs);
    return new_func;
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

  Expr VisitExpr_(const DataflowVarNode* var) final {
    return VisitExpr_(static_cast<const VarNode*>(var));
  }

  const char* attr_num_input_ = "num_input";
  // Remap the original parameters to TupleGetItem from the packed tuple of transformed parameters.
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> param_remap_;
  // The plan of lifting the transform params
  LiftTransformParamsInfoPlan lift_plan_;
};

namespace transform {
Pass LiftTransformParams() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return TransformParamsLifter(m).Lift(); };
  return CreateModulePass(pass_func, 1, "LiftTransformParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LiftTransformParams").set_body_typed(LiftTransformParams);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
