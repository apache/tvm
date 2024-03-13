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
#include <optional>
#include <tuple>
#include <vector>

#include "../../support/ordered_set.h"
#include "utils.h"

namespace tvm {
namespace relax {

constexpr const char* kLiftTransformConsumeParams = "relax.lift_transform_params.consume_params";
TVM_REGISTER_PASS_CONFIG_OPTION(kLiftTransformConsumeParams, Bool);

constexpr const char* kLiftTransformGlobal = "relax.lift_transform_params.lift_globally";
TVM_REGISTER_PASS_CONFIG_OPTION(kLiftTransformGlobal, Bool);

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

  /*! \brief Variables that require a compile-time parameter
   *
   * Used to distinguish between computed tensors that depend on the
   * model weights, and computed tensors that require neither model
   * weights nor runtime arguments (e.g. `R.zeros([16], "float16")`).
   */
  std::unordered_set<Variant<relax::Var, tir::Var>, ObjectPtrHash, ObjectPtrEqual>
      requires_compile_time_param;

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
      if (requires_compile_time_param.count(binding->var) &&
          required_at_runtime.count(binding->var)) {
        params.push_back(binding->var);
      }
    }

    return params;
  }

  // TODO(wuwei): this can be an independant function outside the class
  Function MakeCompileTimeFunction(const Array<Binding>& bindings, const Array<Var> params,
                                   const Array<tir::Var>& output_symbolic_vars,
                                   const Array<Var>& outputs) {
    Array<Binding> output_var_binding;
    Array<Expr> output_exprs;
    if (output_symbolic_vars.size()) {
      output_exprs.push_back(
          ShapeExpr(output_symbolic_vars.Map([](tir::Var var) -> PrimExpr { return var; })));
    }

    for (const auto& var : outputs) {
      Var out_var(var->name_hint() + "_output", GetStructInfo(var));
      output_var_binding.push_back(VarBinding(out_var, var));
      output_exprs.push_back(out_var);
    }

    Var tuple_var("output_tuple", TupleStructInfo(output_exprs.Map(GetStructInfo)));
    output_var_binding.push_back(VarBinding(tuple_var, Tuple(output_exprs)));

    SeqExpr body(
        {
            DataflowBlock(bindings),
            DataflowBlock(output_var_binding),
        },
        tuple_var);
    Function func(params, body, GetStructInfo(tuple_var));
    func = WithAttr(func, attr::kNumInput, Integer(0));
    // LOG(INFO) << "MakeCompileTimeFunction: " << func;
    func = CopyWithNewVars(func);
    func = Downcast<Function>(CanonicalizeBindings(func));
    return func;
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
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> to_suppress;
    for (const auto& binding : computable_at_compile_time) {
      if (requires_compile_time_param.count(binding->var)) {
        to_suppress.insert(binding->var);
      }
    }

    class SuppressCompileTime : public ExprMutator {
     public:
      explicit SuppressCompileTime(
          const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& to_suppress)
          : to_suppress_(to_suppress) {}

      void VisitBinding(const Binding& binding) override {
        if (!to_suppress_.count(binding->var)) {
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

     private:
      const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& to_suppress_;
    };
    Expr body = SuppressCompileTime(to_suppress)(orig_func->body);
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

class BaseLiftableBindingCollector : public ExprVisitor {
 protected:
  void VisitBindingBlock_(const DataflowBlockNode* block) final {
    bool cache = is_in_dataflow_block_;
    is_in_dataflow_block_ = true;
    ExprVisitor::VisitBindingBlock_(block);
    is_in_dataflow_block_ = cache;
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

  std::unordered_set<Variant<Var, tir::Var>, ObjectPtrHash, ObjectPtrEqual> liftable_vars_;
  bool is_in_dataflow_block_{false};
};

struct GlobalCollectInfo {
  Map<Var, Expr> var_remap;
  Array<Binding> unified_bindings;
};

class LiftableBindingCollector : public BaseLiftableBindingCollector {
 public:
  static CollectInfo Collect(const Function& func, std::optional<GlobalCollectInfo> global_info) {
    LiftableBindingCollector visitor(global_info);
    visitor(func);
    visitor.info_.orig_func = func;
    return visitor.info_;
  }

 private:
  LiftableBindingCollector(std::optional<GlobalCollectInfo> global_info)
      : global_info_(global_info) {}
  void VisitExpr_(const FunctionNode* func) override {
    size_t num_runtime_params = func->params.size();
    if (auto opt = func->attrs.GetAttr<Integer>(attr::kNumInput)) {
      num_runtime_params = opt.value()->value;
    }

    info_.num_runtime_params = num_runtime_params;

    for (size_t i = num_runtime_params; i < func->params.size(); i++) {
      liftable_vars_.insert(func->params[i]);
      info_.requires_compile_time_param.insert(func->params[i]);
      for (const auto& tir_var : DefinableTIRVarsInStructInfo(GetStructInfo(func->params[i]))) {
        liftable_vars_.insert(tir_var);
      }
    }
    ExprVisitor::VisitExpr_(func);
  }

  void VisitBinding(const Binding& binding) override {
    auto bound_value = GetBoundValue(binding);

    if (CanLiftBinding(binding) &&
        (!global_info_.has_value() || global_info_->var_remap.count(binding->var))) {
      info_.computable_at_compile_time.push_back(binding);
      liftable_vars_.insert(binding->var);

      // There are three type of variables we want to distinguish.
      //
      // 1. Depend on runtime parameters
      //
      //    Must remain within the original function, cannot be
      //    lifted out into the `transform_params` function.
      //
      // 2. Depend on model weights, but not runtime parameters.
      //
      //    Legal to lift out into the `transform_params` function.
      //    Doing so is beneficial, as it reduces the work performed
      //    in the inference function.
      //
      // 3. Depend on neither model weights nor runtime parameters
      //    (e.g. `R.zeros(shape,dtype)`)
      //
      //    Legal to lift out into the `transform_params` function.
      //    However, doing so would increase the memory footprint of
      //    the pre-computed parameters, for little to no benefit.
      //    These may be duplicated between the `transform_params`
      //    function and the original function, as they typically
      //    initialize a tensor to an easy-to-compute state.
      //
      // Tracking whether a variable depends on the model weights,
      // either directly or indirectly, allows us to distinguish
      // between categories (2) and (3).
      auto upstream_vars = FreeVars(bound_value);
      bool depends_on_compile_time_param = std::any_of(
          upstream_vars.begin(), upstream_vars.end(),
          [&](const Var& var) -> bool { return info_.requires_compile_time_param.count(var); });
          LOG(INFO) << "upstream_vars: " << upstream_vars;
      if (depends_on_compile_time_param) {
        LOG(INFO) << "requires_compile_time_param " << binding->var;
        info_.requires_compile_time_param.insert(binding->var);
      }

    } else {
      LOG(INFO) << "requried_at_runtime: " << binding->var << " " << CanLiftBinding(binding) << " "
                << (global_info_.has_value() && global_info_->var_remap.count(binding->var));
      info_.required_at_runtime.insert(binding->var);
      for (const auto& upstream_var : FreeVars(bound_value)) {
        info_.required_at_runtime.insert(upstream_var);
      }
      for (const auto& tir_var : FreeSymbolicVars(bound_value)) {
        info_.required_at_runtime.insert(tir_var);
      }
    }
  }

  std::optional<GlobalCollectInfo> global_info_;
  CollectInfo info_;
};

class ParamVisitor {};

class ParamRemapper : public ExprFunctor<void(const Expr&, const Expr&)> {
 public:
  void VisitExpr_(const VarNode* lhs_var, const Expr& rhs_expr) final {
    auto rhs_var = Downcast<Var>(rhs_expr);
    if (auto it = var_remap_.find(GetRef<Var>(lhs_var)); it != var_remap_.end()) {
      CHECK((*it).second.same_as(rhs_var));
    } else {
      LOG(INFO) << "ParamRemapper: " << GetRef<Var>(lhs_var) << " " << lhs_var << " -> " << rhs_var;
      var_remap_.Set(GetRef<Var>(lhs_var), rhs_var);
    }
  }

  Map<Var, Expr> var_remap_;
};

class GlobalCollector : public BaseLiftableBindingCollector {
 public:
  static GlobalCollectInfo Collect(const Array<Function>& functions,
                                   const Map<Var, Expr>& var_remap) {
    GlobalCollector collector(var_remap);
    for (const auto& func : functions) {
      int num_inputs = func->GetAttr<Integer>(attr::kNumInput).value()->value;
      for (int i = num_inputs; i < static_cast<int>(func->params.size()); i++) {
        collector.liftable_vars_.insert(func->params[i]);
      }
      collector(func);
    }
    GlobalCollectInfo info;
    info.var_remap = collector.var_remap_;
    for (const auto& [normalized_expr, original_bindings] : collector.unified_bindings_) {
      // Note: it is possible that a function has common subexpressions, so it is not necessary to
      // require original_bindings.size() == functions.size().
      if (original_bindings.size() % functions.size() == 0) {
        // All target functions have the same binding
        // var_remap_[normalized_expr] = original_bindings[0].var;
        // var_remap_[original_bindings[0].var] = normalized_expr;
        info.unified_bindings.push_back(original_bindings[0]);
        LOG(INFO) << "GlobalTransform: " << info.unified_bindings.back();
        for (const auto& binding : original_bindings) {
          info.var_remap.Set(binding->var, original_bindings.front()->var);
          LOG(INFO) << "GlobalTransform: " << binding->var << " -> "
                    << original_bindings.front()->var;
        }
      } else {
        LOG(INFO) << "GlobalTransform: discard " << normalized_expr;
      }
    }
    return info;
  }

 private:
  GlobalCollector(const Map<Var, Expr>& var_remap) : var_remap_(var_remap) {}
  void VisitBinding(const Binding& binding) override {
    auto bound_value = GetBoundValue(binding);
    if (CanLiftBinding(binding)) {
      liftable_vars_.insert(binding->var);
      auto new_value = Bind(bound_value, var_remap_);
      unified_bindings_[new_value].push_back(binding);
      if (unified_bindings_[new_value].size() > 1) {
        var_remap_.Set(binding->var, unified_bindings_[new_value].front()->var);
      }
    } else {
      LOG(INFO) << "GlobalTransform: not liftable " << binding->var;
    }
  }

  std::unordered_map<Expr, std::vector<Binding>, StructuralHash, StructuralEqual> unified_bindings_;
  Map<Var, Expr> var_remap_;
};

GlobalCollectInfo GlobalCollect(IRModule mod) {
  // Map<GlobalVar, Function> target_functions;
  std::vector<Function> target_functions;
  for (const auto& [gvar, func] : mod->functions) {
    if (func->IsInstance<FunctionNode>()) {
      auto opt_num_input = func->GetAttr<Integer>(attr::kNumInput);
      if (opt_num_input) {
        target_functions.push_back(Downcast<Function>(func));
      }
    }
  }

  ParamRemapper remapper;
  LOG(INFO) << "GlobalCollect: " << target_functions.size() << " functions found.";
  if (target_functions.size() > 0) {
    int num_inputs_0 = target_functions[0]->GetAttr<Integer>(attr::kNumInput).value()->value;
    int num_params = static_cast<int>(target_functions[0]->params.size()) - num_inputs_0;
    for (int i = 0; i < static_cast<int>(target_functions.size()); i++) {
      int num_inputs_i = target_functions[i]->GetAttr<Integer>(attr::kNumInput).value()->value;
      CHECK_EQ(num_params, static_cast<int>(target_functions[i]->params.size()) - num_inputs_i)
          << "The number of parameters should be the same for all target functions";
      LOG(INFO) << "GlobalCollect: " << target_functions[i]->params.size() << " "
                << target_functions[i]->params;
      for (int j = 0; j < num_params; j++) {
        const auto& rhs_param = target_functions[0]->params[num_inputs_0 + j];
        const auto& lhs_param = target_functions[i]->params[num_inputs_i + j];
        remapper.VisitExpr(lhs_param, rhs_param);
        // remapper.VisitExprDepStructInfoField(lhs_param->struct_info_, rhs_param);
      }
    }
  }
  return GlobalCollector::Collect(target_functions, remapper.var_remap_);
}

// class PreprocessPartitioner : public ExprMutator {
//  public:
//   using ExprMutator::VisitExpr_;
//   Expr VisitExpr_(const FunctionNode* op) override {
//     auto func = GetRef<Function>(op);
//     if (func->attrs.GetAttr<Integer>(attr::kNumInput)) {
//       auto info = LiftableBindingCollector::Collect(func);
//       return info.MakePartitionedFunction();
//     } else {
//       return func;
//     }
//   }
// };

// Adapted from https://stackoverflow.com/a/2072890
inline bool ends_with(const std::string& value, const std::string& ending) {
  return ending.size() <= value.size() &&
         std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

/*!
 * \brief A mutator to rewrite the transform_params functions to release the original weight after
 * use. This is done by using builtin.tuple_reset_item to reset the bundled weight tuple. It
 * requires `BundleModelParams` to be called before this mutator.
 */
class ConsumeBundledParams : public ExprMutator {
 public:
  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* tuple_get_item) final {
    static const auto& call_pure_packed = Op::Get("relax.call_pure_packed");
    static const auto& builtin_tuple_reset_item = ExternFunc("vm.builtin.tuple_reset_item");
    if (tuple_get_item->tuple.same_as(params_)) {
      if (auto it = param_remap_.find(tuple_get_item->index); it != param_remap_.end()) {
        ReEmitBinding(binding, it->second);
        return;
      }
      ExprMutator::VisitBinding_(binding, tuple_get_item);
      auto new_var = VisitExpr(binding->var);
      param_remap_[tuple_get_item->index] = new_var;
      builder_->Emit(
          Call(call_pure_packed,
               {builtin_tuple_reset_item, tuple_get_item->tuple, PrimValue(tuple_get_item->index)},
               tvm::Attrs(), {TupleStructInfo(Array<StructInfo>{})}));
    } else {
      ExprMutator::VisitBinding_(binding, tuple_get_item);
    }
  }

  Expr VisitExpr_(const FunctionNode* func) final {
    auto opt_num_input = func->GetAttr<Integer>(attr::kNumInput);
    ICHECK(opt_num_input.defined());
    auto num_input = opt_num_input.value()->value;
    ICHECK_EQ(func->params.size(), num_input + 1);
    params_ = func->params.back();
    ICHECK(params_->struct_info_.as<TupleStructInfoNode>());
    return ExprMutator::VisitExpr_(func);
  }

 private:
  Var params_;
  std::unordered_map<int, Expr> param_remap_;
};

}  // namespace

namespace transform {

Pass PartitionTransformParams() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    IRModule updates;

    std::optional<GlobalCollectInfo> global_collect_info = std::nullopt;
    if (pc->GetConfig<Bool>(kLiftTransformGlobal).value_or(Bool(false))) {
      global_collect_info = GlobalCollect(mod);
    }
    // PreprocessPartitioner mutator(global_collect_info);

    std::unordered_map<GlobalVar, CollectInfo, ObjectPtrHash, ObjectPtrEqual> local_collect_info;
    for (const auto& [gvar, func] : mod->functions) {
      if (func.as<relax::FunctionNode>() && func->GetAttr<Integer>(attr::kNumInput)) {
        auto info =
            LiftableBindingCollector::Collect(Downcast<relax::Function>(func), global_collect_info);
        local_collect_info[gvar] = info;
      }
    }

    // Combine local collect info. This determines the output of the compile-time functions.
    if (global_collect_info.has_value()) {
    }
    for (const auto& [gvar, info] : local_collect_info) {
      auto new_runtime_func = info.MakeRuntimeFunction();
      new_runtime_func = Downcast<Function>(CanonicalizeBindings(new_runtime_func));
      new_runtime_func.CopyOnWrite()->attrs = info.orig_func->attrs;
      updates->Add(gvar, new_runtime_func);
      if (!global_collect_info.has_value()) {
        Function new_func = info.MakeCompileTimeFunction();
        String global_symbol = gvar->name_hint + "_transform_params";
        new_func = WithAttr(new_func, tvm::attr::kGlobalSymbol, global_symbol);
        updates->Add(GlobalVar(global_symbol), new_func);
      }
    }
    if (global_collect_info.has_value()) {
      Array<Var> outputs = local_collect_info.begin()->second.GetCompileTimeOutputs();
      outputs = outputs.Map([&](const Var& var) -> Var {
        if (!global_collect_info.value().var_remap.count(var)) {
          LOG(FATAL) << "GlobalCollectInfo does not contain " << var << " " << var.get();
        }
        return Downcast<Var>(global_collect_info.value().var_remap[var]);
      });
      Array<Var> inputs;
      inputs = local_collect_info.begin()->second.GetCompileTimeInputs();
      inputs = inputs.Map([&](const Var& var) -> Var {
        if (!global_collect_info.value().var_remap.count(var)) {
          LOG(FATAL) << "GlobalCollectInfo does not contain " << var << " " << var.get();
        }
        return Downcast<Var>(global_collect_info.value().var_remap[var]);
      });
      Array<Binding> bindings = local_collect_info.begin()->second.computable_at_compile_time;
      LOG(INFO) << "ComputableAtCompileTime: " << bindings;
      // FIXME(wuwei): need to consider non-VarBinding cases
      for (int i = 0; i < bindings.size(); i++) {
        auto binding = bindings[i];
        bindings.Set(i, VarBinding(Downcast<Var>(global_collect_info.value().var_remap[binding->var]),
                                   Bind(Downcast<VarBinding>(binding)->value, global_collect_info.value().var_remap)));
      }

      Function global_transform = local_collect_info.begin()->second.MakeCompileTimeFunction(
          global_collect_info.value().unified_bindings, inputs,
          local_collect_info.begin()->second.GetPropagatedSymbolicVariables(), outputs);
      updates->Add(GlobalVar("transform_params"),
                   local_collect_info.begin()->second.MakeCompileTimeFunction());
    }
    //
    // for (const auto& [gvar, info] : local_collect_info) {

    //   updates->Add(gvar, info.MakePartitionedFunction());
    // }

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
          if (pc->GetConfig<Bool>(kLiftTransformConsumeParams).value_or(Bool(false))) {
            func = Downcast<Function>(ConsumeBundledParams()(func));
          }
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
