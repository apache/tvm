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

namespace {
struct BaseCollectInfo {
 public:
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

 protected:
  Array<Var> GetCompileTimeOutputsHelper(const Array<Var>& params) const {
    // The output of the compile-time function is in the following order:
    // 1) Any parameter that is required at runtime in the original order, followed by,
    // 2) Any binding that is computable at compile-time and required at runtime in the original
    // order.
    Array<Var> output;
    for (const auto& param : params) {
      if (required_at_runtime.count(param)) {
        output.push_back(param);
      }
    }
    for (const auto& binding : computable_at_compile_time) {
      if (requires_compile_time_param.count(binding->var) &&
          required_at_runtime.count(binding->var)) {
        output.push_back(binding->var);
      }
    }

    return output;
  }

  Function MakeCompileTimeFunctionHelper(const Array<Var> params, const Array<Binding>& bindings,
                                         const Array<tir::Var>& output_symbolic_vars,
                                         const Array<Var>& outputs) const {
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
    func = CopyWithNewVars(func);
    func = BundleModelParams(func);
    func = Downcast<Function>(CanonicalizeBindings(func));
    func = Downcast<Function>(RemoveAllUnused(func));

    return func;
  }
};

struct GlobalCollectInfo : public BaseCollectInfo {
  // The original functions
  Array<Function> orig_functions;
  // The parameters of the compile-time function.
  Array<Var> params;
  // The cross-function mapping between variables.
  Map<relax::Var, Expr> var_remap;
  // The cross-function between between TIR variables.
  Map<tir::Var, PrimExpr> tir_var_remap;
  Array<tir::Var> GetPropagatedSymbolicVariables() const {
    auto vars_from_original_params =
        DefinableTIRVarsInStructInfo(TupleStructInfo(params.Map(GetStructInfo)));
    auto vars_from_transformed_params = [&]() -> std::unordered_set<tir::Var> {
      auto tir_vars =
          DefinableTIRVarsInStructInfo(TupleStructInfo(GetCompileTimeOutputs().Map(GetStructInfo)));
      return {tir_vars.begin(), tir_vars.end()};
    }();

    Array<tir::Var> output;
    for (const auto& tir_var : vars_from_original_params) {
      if (required_at_runtime.count(tir_var) && !vars_from_transformed_params.count(tir_var)) {
        output.push_back(tir_var);
      }
    }
    return output;
  }

  Function MakeCompileTimeFunc() {
    return MakeCompileTimeFunctionHelper(params, computable_at_compile_time,
                                         GetPropagatedSymbolicVariables(), GetCompileTimeOutputs());
  }
  Array<Var> GetCompileTimeOutputs() const { return GetCompileTimeOutputsHelper(params); }
};
struct LocalCollectInfo : public BaseCollectInfo {
  /* \brief The analyzed function */
  Function orig_func;

  /* \brief The number of parameters unknown until runtime */
  size_t num_runtime_params;

  GlobalCollectInfo* global_info = nullptr;

  Array<Var> GetCompileTimeInputs() const {
    return Array<Var>(orig_func->params.begin() + num_runtime_params, orig_func->params.end());
  }

  Array<Var> GetRuntimeInputs() const {
    return Array<Var>(orig_func->params.begin(), orig_func->params.begin() + num_runtime_params);
  }

  Array<tir::Var> GetPropagatedSymbolicVariables() const {
    auto vars_from_any_param =
        DefinableTIRVarsInStructInfo(TupleStructInfo(orig_func->params.Map(GetStructInfo)));

    auto vars_from_runtime_params = [&]() -> std::unordered_set<tir::Var> {
      auto tir_var_vec =
          DefinableTIRVarsInStructInfo(TupleStructInfo(GetRuntimeInputs().Map(GetStructInfo)));
      return {tir_var_vec.begin(), tir_var_vec.end()};
    }();

    auto vars_from_transformed_params = [&]() -> std::unordered_set<tir::Var> {
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
    return GetCompileTimeOutputsHelper(GetCompileTimeInputs());
  }

  Function MakeCompileTimeFunction() const {
    ICHECK(!global_info);  // This function is only called for local lifting
    return MakeCompileTimeFunctionHelper(GetCompileTimeInputs(), computable_at_compile_time,
                                         GetPropagatedSymbolicVariables(), GetCompileTimeOutputs());
  }

  Function MakeRuntimeFunction() const {
    Array<Binding> bindings;

    // Any parameter that isn't available until runtime must be an
    // input, along with any output from the compile-time function.
    // Compile-time outputs must have a fresh non-dataflow var to
    // serve as the parameter.  This trivial binding will later be
    // removed with CanonicalizeBindings.
    Array<Var> params = GetRuntimeInputs();
    auto propagated_tir_vars = [&]() {
      Array<tir::Var> local_tir_vars = GetPropagatedSymbolicVariables();
      if (!global_info) {
        return local_tir_vars;
      }
      // When global lifting is enabled, the compile-time outputs are the global outputs, but the
      // variables in the global outputs to the local variables.
      Map<tir::Var, tir::Var> reverse_map;
      for (const auto& var : local_tir_vars) {
        if (auto it = global_info->tir_var_remap.find(var);
            it != global_info->tir_var_remap.end()) {
          reverse_map.Set(Downcast<tir::Var>((*it).second), var);
        }
      }
      Array<tir::Var> global_tir_vars = global_info->GetPropagatedSymbolicVariables();
      global_tir_vars = global_tir_vars.Map([&](const tir::Var& var) {
        if (auto it = reverse_map.find(var); it != reverse_map.end()) {
          return Downcast<tir::Var>((*it).second);
        } else {
          // This is the case when the some of the outputs of the shared transform is not used in
          // this function.
          return var;
        }
      });
      return global_tir_vars;
    }();
    if (propagated_tir_vars.size()) {
      ShapeStructInfo shape_sinfo(
          propagated_tir_vars.Map([](tir::Var var) -> PrimExpr { return var; }));
      Var shape_expr("vars_from_compile_time_params", shape_sinfo);
      params.push_back(shape_expr);
    }
    Array<Var> compile_time_outputs = [&]() {
      Array<Var> local_outputs = GetCompileTimeOutputs();
      if (!global_info) {
        return local_outputs;
      }
      // When global lifting is enabled, the compile-time outputs are the global outputs, but the
      // variables in the global outputs to the local variables.
      Map<Var, Var> reverse_map;
      for (const auto& var : local_outputs) {
        if (auto it = global_info->var_remap.find(var); it != global_info->var_remap.end()) {
          reverse_map.Set(Downcast<Var>((*it).second), var);
        }
      }
      Array<Var> global_outputs = global_info->GetCompileTimeOutputs();
      global_outputs = global_outputs.Map([&](const Var& var) {
        if (auto it = reverse_map.find(var); it != reverse_map.end()) {
          return Downcast<Var>((*it).second);
        } else {
          // This is the case when the some of the outputs of the shared transform is not used in
          // this function.
          return var;
        }
      });
      return global_outputs;
    }();
    for (const auto& var : compile_time_outputs) {
      Var param_var(var->name_hint(), GetStructInfo(var));
      bindings.push_back(VarBinding(var, param_var));
      params.push_back(param_var);
    }

    // Any binding that is computable at compile-time should be
    // suppressed at run-time.
    std::unordered_set<Var> to_suppress;
    for (const auto& binding : computable_at_compile_time) {
      if (requires_compile_time_param.count(binding->var)) {
        to_suppress.insert(binding->var);
      }
    }

    class SuppressCompileTime : public ExprMutator {
     public:
      explicit SuppressCompileTime(const std::unordered_set<Var>& to_suppress)
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
      const std::unordered_set<Var>& to_suppress_;
    };
    Expr body = SuppressCompileTime(to_suppress)(orig_func->body);
    body = SeqExpr({DataflowBlock(bindings)}, body);

    Function func(params, body, orig_func->ret_struct_info, orig_func->is_pure, orig_func->attrs);
    func = CopyWithNewVars(func);
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

class LocalLiftableBindingCollector : public BaseLiftableBindingCollector {
 public:
  static LocalCollectInfo Collect(const Function& func, GlobalCollectInfo* global_info) {
    LocalLiftableBindingCollector visitor(global_info);
    visitor(func);
    visitor.info_.orig_func = func;

    auto set_union =
        [&](std::unordered_set<Variant<relax::Var, tir::Var>, ObjectPtrHash, ObjectPtrEqual>&
                target_set,
            const std::unordered_set<Variant<relax::Var, tir::Var>, ObjectPtrHash, ObjectPtrEqual>&
                source_set,
            const Map<relax::Var, Expr>& var_remap, const Map<tir::Var, PrimExpr>& tir_var_remap) {
          // In-place update the set in global info by unioning with the local set, variable
          // mappings are applied.
          for (const auto& relax_or_tir_var : source_set) {
            if (relax_or_tir_var->IsInstance<relax::VarNode>()) {
              if (auto it = var_remap.find(Downcast<Var>(relax_or_tir_var));
                  it != var_remap.end()) {
                target_set.insert(Downcast<relax::Var>((*it).second));
              } else {
                target_set.insert(Downcast<relax::Var>(relax_or_tir_var));
              }
            } else {
              if (auto it = tir_var_remap.find(Downcast<tir::Var>(relax_or_tir_var));
                  it != tir_var_remap.end()) {
                target_set.insert(Downcast<tir::Var>((*it).second));
              } else {
                target_set.insert(Downcast<tir::Var>(relax_or_tir_var));
              }
            }
          }
        };

    if (global_info) {
      set_union(global_info->requires_compile_time_param, visitor.info_.requires_compile_time_param,
                global_info->var_remap, global_info->tir_var_remap);
      set_union(global_info->required_at_runtime, visitor.info_.required_at_runtime,
                global_info->var_remap, global_info->tir_var_remap);
    }
    return visitor.info_;
  }

 private:
  explicit LocalLiftableBindingCollector(GlobalCollectInfo* global_info) {
    info_.global_info = global_info;
  }
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
        (!info_.global_info || info_.global_info->var_remap.count(binding->var))) {
      // The binding is liftable and can be shared with other functions (if global lifting is
      // enabled)
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
      if (depends_on_compile_time_param) {
        info_.requires_compile_time_param.insert(binding->var);
      }

    } else {
      info_.required_at_runtime.insert(binding->var);
      for (const auto& upstream_var : FreeVars(bound_value)) {
        info_.required_at_runtime.insert(upstream_var);
      }
      for (const auto& tir_var : FreeSymbolicVars(bound_value)) {
        info_.required_at_runtime.insert(tir_var);
      }
    }
  }

  LocalCollectInfo info_;
};

/*! \brief Visitor to find the correspondence between parameters in multiple functions. */
class ParamRemapper : private ExprFunctor<void(const Expr&, const Expr&)> {
 public:
  static std::pair<Map<Var, Expr>, Map<tir::Var, PrimExpr>> GetParamMapping(
      const Array<Function>& functions) {
    ParamRemapper mapper;
    if (functions.size()) {
      auto num_inputs_0 = functions[0]->GetAttr<Integer>(attr::kNumInput).value()->value;
      int num_params = static_cast<int>(functions[0]->params.size()) - num_inputs_0;
      for (int i = 0; i < static_cast<int>(functions.size()); i++) {
        auto num_inputs_i = functions[i]->GetAttr<Integer>(attr::kNumInput).value()->value;
        CHECK_EQ(num_params, static_cast<int>(functions[i]->params.size()) - num_inputs_i)
            << "The number of parameters should be the same for all target functions";

        for (int j = 0; j < num_params; j++) {
          // Map the parameters to the first function
          int index_i = j + num_inputs_i;
          int index_0 = j + num_inputs_0;
          mapper.VisitExpr(functions[i]->params[index_i], functions[0]->params[index_0]);
          StructuralEqual eq;
          eq(functions[i]->params[index_i]->struct_info_,
             functions[0]->params[index_0]->struct_info_);
        }
      }
    }
    return {mapper.var_remap_, mapper.tir_var_remap_};
  }

 private:
  void VisitExpr_(const VarNode* lhs_var, const Expr& rhs_expr) final {
    auto rhs_var = Downcast<Var>(rhs_expr);
    if (auto it = var_remap_.find(GetRef<Var>(lhs_var)); it != var_remap_.end()) {
      CHECK((*it).second.same_as(rhs_var));
    } else {
      var_remap_.Set(GetRef<Var>(lhs_var), rhs_var);
    }
    CHECK(structural_equal.Equal(lhs_var->struct_info_, rhs_var->struct_info_,
                                 /*map_free_vars=*/true))
        << "The struct info of the parameters should be the same for all target functions";
    auto lhs_tir_vars = DefinableTIRVarsInStructInfo(GetStructInfo(GetRef<Var>(lhs_var)));
    auto rhs_tir_vars = DefinableTIRVarsInStructInfo(GetStructInfo(rhs_expr));
    ICHECK_EQ(lhs_tir_vars.size(), rhs_tir_vars.size());
    for (size_t i = 0; i < lhs_tir_vars.size(); i++) {
      if (auto it = tir_var_remap_.find(lhs_tir_vars[i]); it != tir_var_remap_.end()) {
        CHECK((*it).second.same_as(rhs_tir_vars[i]));
      } else {
        tir_var_remap_.Set(lhs_tir_vars[i], rhs_tir_vars[i]);
      }
    }
  }

  SEqualHandlerDefault structural_equal{/*assert_mode=*/false, /*first_mismatch=*/nullptr,
                                        /*defer_fail=*/false};
  Map<Var, Expr> var_remap_;
  Map<tir::Var, PrimExpr> tir_var_remap_;
};

class GlobalLiftableBindingCollector : public BaseLiftableBindingCollector {
 public:
  static GlobalCollectInfo Collect(const Array<Function>& functions,
                                   const Map<Var, Expr>& var_remap,
                                   const Map<tir::Var, PrimExpr>& tir_var_remap) {
    GlobalLiftableBindingCollector collector(var_remap, tir_var_remap);
    ICHECK(functions.size());
    for (const auto& func : functions) {
      int num_inputs = func->GetAttr<Integer>(attr::kNumInput).value()->value;
      for (int i = num_inputs; i < static_cast<int>(func->params.size()); i++) {
        collector.liftable_vars_.insert(func->params[i]);
      }
      collector(func);
    }
    Array<Var> params(functions[0]->params.begin() +
                          functions[0]->GetAttr<Integer>(attr::kNumInput).value()->value,
                      functions[0]->params.end());
    // todo(@tvm-team): use c++20 designated initializers when windows CI supports it
    GlobalCollectInfo info = GlobalCollectInfo();
    info.orig_functions = functions;
    info.params = std::move(params);
    info.var_remap = var_remap;
    info.tir_var_remap = tir_var_remap;
    // Find shared bindings among transform_params. Re-compute var_remap based on the shared
    // bindings as collector.var_remap_ may contain invalid mappings.
    for (const auto& unified_binding : collector.unified_bindings_) {
      const auto& original_bindings = collector.original_bindings_[GetBoundValue(unified_binding)];
      // Note: it is possible that one or more functions have common subexpressions such as:
      //
      //   func1:
      //     w1_t = w.transpose
      //     w2_t = w.transpose
      //
      //   func2:
      //     w1_t = w.transpose
      //     w2_t = w.transpose
      //
      // In this case, original_bindings.size() != functions.size() but we should still consider
      // w and w.transpose as a shared binding.

      if (original_bindings.size() == functions.size()) {
        info.computable_at_compile_time.push_back(unified_binding);
        for (const auto& original_binding : original_bindings) {
          info.var_remap.Set(original_binding->var, unified_binding->var);
        }
      }
    }
    return info;
  }

 private:
  GlobalLiftableBindingCollector(const Map<Var, Expr>& var_remap,
                                 const Map<tir::Var, PrimExpr> tir_var_remap)
      : var_remap_(var_remap), tir_var_remap_(tir_var_remap) {}
  void VisitBinding(const Binding& binding) override {
    CHECK(!binding->IsInstance<MatchCastNode>()) << "MatchCast is not supported in global lifting";
    if (CanLiftBinding(binding)) {
      liftable_vars_.insert(binding->var);
      auto bound_value = GetBoundValue(binding);
      auto new_value = Bind(bound_value, var_remap_, tir_var_remap_);
      if (auto it = original_bindings_.find(new_value); it != original_bindings_.end()) {
        it->second.push_back(binding);
      } else {
        unified_bindings_.push_back(binding);
        original_bindings_[new_value].push_back(binding);
      }
      var_remap_.Set(binding->var, original_bindings_[new_value].front()->var);
    }
  }

  // The cross-function mapping between variables. This is initialized with the mapping from the
  // function parameters, and is updated with the mapping between binding variables asthe collector
  // visits the bindings.
  Map<Var, Expr> var_remap_;
  // The cross-function between between TIR variables.
  Map<tir::Var, PrimExpr> tir_var_remap_;
  std::vector<Binding> unified_bindings_;
  // The mapping between the unified bindings and the original bindings in different functions.
  // The unified binding is the binding with all variables replaced by the unified variables as
  // defined in var_remap_.
  std::unordered_map<Expr, std::vector<Binding>, StructuralHash, StructuralEqual>
      original_bindings_;
};  // namespace

GlobalCollectInfo MakeGlobalLiftPlan(const IRModule& mod,
                                     const std::vector<Function>& target_functions) {
  ParamRemapper remapper;
  auto [var_remap, tir_var_remap] = ParamRemapper::GetParamMapping(target_functions);
  return GlobalLiftableBindingCollector::Collect(target_functions, var_remap, tir_var_remap);
}

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

std::vector<std::pair<GlobalVar, Function>> GetTargetFunctions(
    const IRModule& mod, const Variant<Bool, Array<String>>& shared_transform) {
  std::vector<std::pair<GlobalVar, Function>> target_functions;
  if (shared_transform.as<Array<String>>().value_or(Array<String>{}).size()) {
    for (const auto& name : shared_transform.as<Array<String>>().value()) {
      auto gvar = mod->global_var_map_.Get(name);
      CHECK(gvar) << "When LiftTransformParams is called with a list of function names, "
                  << "all function names must occur within the IRModule.  "
                  << "However, the IRModule does not contain a function names '" << name << "'";

      auto base_func = mod->functions.Get(gvar.value());
      ICHECK(base_func) << "Ill-formed IRModule.  "
                        << "The map from name to GlobalVar found " << gvar.value()
                        << " for the function name '" << name
                        << "', but this GlobalVar does not appear in the IRModule";

      auto func = base_func.as<Function>();
      CHECK(func) << "When LiftTransformParams is called with a list of function names, "
                  << "only functions in the list must be relax functions.  "
                  << "However, the function " << name << " is of type " << base_func->GetTypeKey();
      CHECK(func.value()->GetAttr<Integer>(attr::kNumInput))
          << "When LiftTransformParams is called with a list of function names, "
          << "all functions in the list must have the kNumInput ('" << attr::kNumInput
          << "') attribute.  "
          << "However, the function " << name << " does not have the kNumInput attribute";

      target_functions.push_back({gvar.value(), func.value()});
    }
  } else {
    // Get all the functions that have the `num_input` attribute, and
    // are not already the result of `LiftTransformParams`.
    for (const auto& [gvar, func] : mod->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto opt_num_input = func->GetAttr<Integer>(attr::kNumInput);
        if (opt_num_input && !ends_with(gvar->name_hint, "transform_params")) {
          target_functions.emplace_back(gvar, Downcast<Function>(func));
        }
      }
    }
    std::sort(target_functions.begin(), target_functions.end(),
              [](const auto& lhs, const auto& rhs) {
                return lhs.first->name_hint < rhs.first->name_hint;
              });
  }
  return target_functions;
}

}  // namespace

namespace transform {

Pass PartitionTransformParams(Variant<Bool, Array<String>> shared_transform) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    std::optional<GlobalCollectInfo> global_collect_info;

    CHECK(shared_transform.defined()) << "shared_transform is not defined";
    CHECK((shared_transform.as<Bool>() || shared_transform.as<Array<String>>()))
        << "shared_transform should be a boolean or an array of function names";

    auto target_functions = GetTargetFunctions(mod, shared_transform);

    if (shared_transform.as<Bool>().value_or(Bool(true))) {
      std::vector<Function> functions;
      for (const auto& [_, func] : target_functions) {
        functions.push_back(func);
      }
      global_collect_info = MakeGlobalLiftPlan(mod, functions);
    }

    std::unordered_map<GlobalVar, LocalCollectInfo> local_collect_info;
    for (const auto& [gvar, func] : target_functions) {
      auto info = LocalLiftableBindingCollector::Collect(
          func, global_collect_info.has_value() ? &global_collect_info.value() : nullptr);
      local_collect_info[gvar] = info;
    }

    IRModule updated_runtime_functions;

    for (const auto& [gvar, info] : local_collect_info) {
      auto new_runtime_func = info.MakeRuntimeFunction();
      updated_runtime_functions->Add(gvar, new_runtime_func);
    }

    Map<String, Function> lifted_transform_functions;
    if (global_collect_info.has_value()) {
      auto global_transform = global_collect_info.value().MakeCompileTimeFunc();
      lifted_transform_functions.Set("transform_params", global_transform);
    } else {
      for (const auto& [gvar, info] : local_collect_info) {
        // transform_params is emitted for each function if global lifting is not enabled
        lifted_transform_functions.Set(gvar->name_hint + "_transform_params",
                                       info.MakeCompileTimeFunction());
      }
    }

    if (updated_runtime_functions->functions.size() || lifted_transform_functions.size()) {
      auto write_ptr = mod.CopyOnWrite();
      write_ptr->Update(updated_runtime_functions);

      for (auto [name, transform] : lifted_transform_functions) {
        if (auto opt = write_ptr->global_var_map_.Get(name)) {
          auto old_gvar = opt.value();
          auto old_transform = Downcast<Function>(write_ptr->Lookup(old_gvar));
          write_ptr->Remove(old_gvar);

          transform = ComposeFunctions(old_transform, transform);
        }
        GlobalVar new_gvar(name);
        UpdateStructInfo(new_gvar, GetStructInfo(transform));
        write_ptr->Add(new_gvar, transform);
      }
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 1, "PartitionTransformParams", {});
}

Pass LiftTransformParams(Variant<Bool, Array<String>> shared_transform) {
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
    std::unordered_map<GlobalVar, Function> to_add;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<Function>()) {
        auto func = opt.value();

        std::string func_name = gvar->name_hint;
        if (ends_with(func_name, "transform_params")) {
          func = WithAttr(func, tvm::attr::kGlobalSymbol, gvar->name_hint);
          if (pc->GetConfig<Bool>(kLiftTransformConsumeParams).value_or(Bool(false))) {
            func = Downcast<Function>(ConsumeBundledParams()(func));
          }
          to_add[gvar] = func;
        }
      }
    }

    if (to_add.size()) {
      auto write_ptr = mod.CopyOnWrite();
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
          PartitionTransformParams(shared_transform),
          LambdaLift(),
          post_proc,
      },
      "LiftTransformParams");
}

TVM_REGISTER_GLOBAL("relax.transform.LiftTransformParams").set_body_typed(LiftTransformParams);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
