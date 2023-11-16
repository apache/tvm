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
 * \file relax/transform/rewrite_cuda_graph.cc
 * \brief Pass for transforming Relax module to execute with CUDA graph.
 *
 * CUDA graph provides a way to capture a sequence of CUDA kernel launches in the runtime and
 * save them as a graph. The graph can be executed multiple times with less overhead than launching
 * kernels individually. This pass rewrites the Relax module to execute with CUDA graph.
 *
 * The transformation is done in two steps:
 *
 * 1. Identify the regions that can be captured by CUDA graph and create the rewriting plan with
 * `CUDAGraphRewritePlanner`.
 *
 * A region is a subgraph of the Relax function that are executed statically. A region is executed
 * statically if 1) it only depends on the memory allocated internally in the Relax function with
 * constant shapes, 2) it only contains kernel launches (there are no control flow).
 *
 * This transformation is expected to run after `StaticPlanBlockMemory`. After
 * `StaticPlanBlockMemory`, all the tensors that can be statically allocated are allocated with
 * `R.memory.alloc_storage` and `R.memory.alloc_tensor`, while other tensors will be allocated via
 * `R.builtin.alloc_tensor`.
 *
 * `CUDAGraphRewritePlanner` is executed at the level of BindingBlock. It first identify all the
 * storage objects allocated with `R.memory.alloc_storage` within the BindingBlock, and then
 * identify the static regions by propagating starting from the storage objects.
 *
 * All the calls to `R.memory.alloc_storage` within the same BindingBlock are grouped into a single
 * new function. Each of the static regions are lifted to a new function.
 *
 * 2. Lift the regions identified in step 1 to a separate function and rewrite the original function
 * with `CUDAGraphRewriter`.
 */

#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/expr.h>

#include "../../support/arena.h"
#include "../../support/ordered_set.h"
#include "../../support/utils.h"

namespace tvm {
namespace relax {

TVM_REGISTER_PASS_CONFIG_OPTION("relax.backend.use_cuda_graph", Bool);

/*! \brief The rewriting plan of lifting a region for either allocation or capturing for cuda graph
 * execution
 */
struct LiftedFunctionRewritePlan {
  // The lifted function for allocation or capturing
  Function func;
  // Whether the lifted function is for allocation or capturing
  bool is_alloc;
  // The binding var before which the lifted function should be invoked
  const VarNode* launch_point;

  // Variable remappings between the original function and the lifted function

  // The bindings in the original function that are lifted
  std::unordered_set<const VarNode*> lifted_bindings;
  // The corresponding binding vars in the original function of the outputs of the lifted function
  std::vector<const VarNode*> outputs;
  // The corresponding binding vars in the original function of the inputs of the lifted function
  std::vector<const VarNode*> inputs;
};

/*! \brief Builder of the lifted function for cuda graph capturing or allocations */
class FuncBuilder : public ExprMutator {
 public:
  /*!
   * \brief Add a binding to the new function
   * \param binding The binding to add
   */
  void AddBinding(const VarBindingNode* binding) { bindings_.push_back(binding); }

  /*!
   * \brief Mark a variable as the input of the new function.
   * \param var The variable to mark as input
   */
  void MarkInput(const VarNode* var) { inputs_.push_back(var); }
  /*!
   * \brief Mark a variable as the output of the new function. The variable must be the LHS of an
   * existing binding in the new function.
   * \param var The variable to mark as output
   */
  void MarkOutput(const VarNode* var) { outputs_.push_back(var); }

  /*! \brief Get the number of bindings in the new function */
  auto size() const { return bindings_.size(); }

  /*! \brief Build the new function */
  Function Build() {
    Array<Var> params;
    // Set up the parameters
    for (const auto* input : inputs_) {
      auto new_var = Var(input->name_hint(), Downcast<Optional<StructInfo>>(input->struct_info_));
      var_remap_[input->vid] = new_var;
      params.push_back(new_var);
    }
    // Emit the function body
    builder_->BeginBindingBlock();
    for (const auto* binding : bindings_) {
      VisitBinding_(binding);
    }
    // Set up the outputs
    Array<Expr> outputs;
    for (const auto* var : outputs_) {
      outputs.push_back(VisitExpr_(var));
    }
    auto output = builder_->Emit(Tuple(outputs));
    auto block = builder_->EndBlock();
    auto body = builder_->Normalize(SeqExpr({block}, output));
    Map<String, ObjectRef> attrs;
    attrs.Set(relax::attr::kForcePure, Bool(true));
    auto func = Function(params, body, Downcast<StructInfo>(output->struct_info_.value()),
                         /*is_pure=*/true, /*attrs=*/DictAttrs(attrs));
    return func;
  }

  support::OrderedSet<const VarNode*> inputs_;
  support::OrderedSet<const VarNode*> outputs_;
  std::vector<const VarBindingNode*> bindings_;
};

/*!
 * \brief The planner for rewriting the function to enable cuda graph capturing.
 */
class CUDAGraphRewritePlanner : public ExprVisitor {
 public:
  explicit CUDAGraphRewritePlanner(const IRModule& mod) : mod_(mod) {}
  std::vector<LiftedFunctionRewritePlan> Plan() {
    for (const auto& pair : mod_->functions) {
      if (pair.second->IsInstance<FunctionNode>()) {
        // If a function has the num_input attribute, the last func->params.size() - num_inputs
        // inputs are assumed to be fixed and thus they can be captured into a cuda graph.
        const auto& func = Downcast<Function>(pair.second);
        if (auto num_input = func->attrs.GetAttr<Integer>(attr::kNumInput)) {
          for (size_t i = num_input.value().IntValue(); i < func->params.size(); ++i) {
            static_vars_.insert(func->params[i].get());
          }
        }
        VisitExpr(func);
      }
    }
    std::vector<LiftedFunctionRewritePlan> plans;

    auto region_to_plan = [&](FuncBuilder* region, bool is_alloc) -> LiftedFunctionRewritePlan {
      LiftedFunctionRewritePlan plan;
      plan.is_alloc = true;
      plan.func = region->Build();
      ICHECK(region->size());
      plan.launch_point = region->bindings_.front()->var.get();
      plan.is_alloc = is_alloc;
      for (const auto* binding : region->bindings_) {
        plan.lifted_bindings.insert(binding->var.get());
      }
      plan.inputs.assign(region->inputs_.begin(), region->inputs_.end());
      plan.outputs.assign(region->outputs_.begin(), region->outputs_.end());
      return plan;
    };

    for (auto* region : alloc_storages_) {
      plans.push_back(region_to_plan(region, /*is_alloc=*/true));
    }

    for (auto* region : captured_regions_) {
      plans.push_back(region_to_plan(region, /*is_alloc=*/false));
    }
    return plans;
  }

  /*!
   *\brief Start a new static region. This method should be called when encountering a
   * CUDA kernel launch (calls to PrimFunc or ExternFunc) that only depends on static parameters.
   */
  void StartRegion() { current_.capture_builder = arena_.make<FuncBuilder>(); }

  /*!
   * \brief Finish a static region. This method should be called when non-static bindings or
   * unsupported operations are encountered.
   */
  void EndRegion() {
    if (current_.capture_builder && current_.capture_builder->size()) {
      captured_regions_.emplace_back(current_.capture_builder);
    }
    current_.capture_builder = nullptr;
  }

  void VisitBindingBlock_(const BindingBlockNode* binding_block) final {
    Scope new_scope;
    std::swap(new_scope, current_);
    current_.alloc_storage_builder = arena_.make<FuncBuilder>();
    for (const auto& binding : binding_block->bindings) {
      VisitBinding(binding);
    }
    EndRegion();
    if (current_.alloc_storage_builder->outputs_.size()) {
      alloc_storages_.emplace_back(current_.alloc_storage_builder);
    }
    std::swap(new_scope, current_);
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) final {
    static const auto& mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const auto& builtin_alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const auto& call_builtin_with_ctx_op = Op::Get("relax.call_builtin_with_ctx");

    if (call->op.same_as(mem_alloc_storage_op) && IsStaticAllocStorage(binding)) {
      AddStaticBinding(binding, /*is_alloc_storage=*/true);
      return;
    } else if (call->op.same_as(builtin_alloc_tensor_op)) {
      return;
    }

    const auto* call_gv = call->op.as<GlobalVarNode>();
    bool call_prim_func =
        call_gv ? mod_->Lookup(GetRef<GlobalVar>(call_gv))->IsInstance<tir::PrimFuncNode>() : false;

    // Check whether the call can be lifted to the capture function. It requires all the arguments
    // to be static and the call to be a kernel launch or a pure operation (e.g. memory view).
    std::vector<const VarNode*> args;
    bool is_all_static = [&]() {
      if (!IsStatic(call->args, &args)) {
        return false;
      }
      if (call_gv != nullptr && !call_prim_func) {
        // calls to other Relax functions are not allowed
        return false;
      }
      if (const auto* extern_func = call->op.as<ExternFuncNode>();
          extern_func != nullptr && support::StartsWith(extern_func->global_symbol, "vm.builtin")) {
        return false;
      }
      return true;
    }();

    if (is_all_static) {
      bool is_kernel_launch = [&]() {
        static const auto& null_value_op = Op::Get("relax.null_value");

        if (call_prim_func) {
          return true;
        }
        if (call->op.as<ExternFuncNode>()) {
          return true;
        }
        if (const auto* op = call->op.as<OpNode>()) {
          return !support::StartsWith(op->name, "relax.memory") &&
                 !support::StartsWith(op->name, "relax.builtin") && op->name != "relax.reshape" &&
                 !GetRef<Op>(op).same_as(null_value_op) &&
                 !GetRef<Op>(op).same_as(call_builtin_with_ctx_op);
        }
        return false;
      }();
      if (current_.capture_builder == nullptr && is_kernel_launch) {
        StartRegion();
      }
      AddStaticBinding(binding, /*is_alloc_storage=*/false);
      MarkAsFuncInput(args);
    } else {
      EndRegion();
    }

    MarkAsFuncOutput(args);
  }

  void MarkAsFuncInput(const std::vector<const VarNode*>& vars) {
    if (current_.capture_builder == nullptr) {
      return;
    }
    for (const VarNode* var : vars) {
      auto it = binding_to_region_.find(var);
      if (it == binding_to_region_.end() || it->second != current_.capture_builder) {
        current_.capture_builder->MarkInput(var);
      }
    }
  }

  void MarkAsFuncOutput(const std::vector<const VarNode*>& vars) {
    for (const VarNode* var : vars) {
      if (auto it = binding_to_region_.find(var);
          it != binding_to_region_.end() && it->second != current_.capture_builder) {
        it->second->MarkOutput(var);
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) final {
    if (IsStatic(GetRef<Var>(var))) {
      AddStaticBinding(binding, false);
      MarkAsFuncInput({var});
    } else {
      EndRegion();
    }
    MarkAsFuncOutput({var});
  }

  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* constant) final {
    AddStaticBinding(binding, false);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple) final {
    std::vector<const VarNode*> args;
    if (IsStatic(tuple->fields, &args)) {
      AddStaticBinding(binding, false);
      MarkAsFuncInput(args);
    } else {
      EndRegion();
    }
    MarkAsFuncOutput(args);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* tuple_get_item) final {
    const VarNode* tuple = tuple_get_item->tuple.as<VarNode>();
    ICHECK(tuple);
    if (IsStatic(tuple_get_item->tuple)) {
      AddStaticBinding(binding, false);
      MarkAsFuncInput({tuple});
    } else {
      EndRegion();
    }
    MarkAsFuncOutput({tuple});
  }

  bool IsStatic(const PrimExpr& expr,
                [[maybe_unused]] std::vector<const VarNode*>* vars_collector = nullptr) {
    return expr->IsInstance<tir::IntImmNode>() || expr->IsInstance<tir::FloatImmNode>();
  }

  bool IsStatic(const Expr& expr, std::vector<const VarNode*>* vars_collector = nullptr) {
    if (expr->IsInstance<ConstantNode>() || expr->IsInstance<DataTypeImmNode>()) {
      return true;
    }
    if (const auto* prim_value = expr.as<PrimValueNode>()) {
      return IsStatic(prim_value->value, vars_collector);
    }
    if (const auto* var = expr.as<VarNode>()) {
      if (vars_collector != nullptr) {
        vars_collector->push_back(var);
      }
      return static_vars_.count(var);
    }

    if (const auto* shape = expr.as<ShapeExprNode>()) {
      return IsStatic(shape->values, vars_collector);
    }
    if (const auto* tuple = expr.as<TupleNode>()) {
      return IsStatic(tuple->fields, vars_collector);
    }
    return false;
  }

  template <typename T>
  bool IsStatic(const Array<T>& exprs, std::vector<const VarNode*>* vars_collector = nullptr) {
    bool result = true;
    for (const auto& expr : exprs) {
      // If vars_collector is provided, we will collect all the vars in the exprs and we should
      // not perform short-circuiting.
      result &= IsStatic(expr, vars_collector);
      if (!vars_collector && !result) {
        return false;
      }
    }
    return result;
  }

 private:
  bool IsStaticAllocStorage(const VarBindingNode* binding) {
    // Check if the allocation has constant shape
    const auto* alloc_storage_call = binding->value.as<CallNode>();
    auto shape = Downcast<ShapeExpr>(alloc_storage_call->args[0]);
    return std::all_of(shape->values.begin(), shape->values.end(),
                       [](const PrimExpr& expr) { return expr.as<IntImmNode>() != nullptr; });
  }

  /*!
   * \brief Add a static bindings. This is used to mark the bindings that are known to be static
   * and further propagate to other bindings.
   * \param binding the binding to add
   * \param is_alloc_storage whether the binding is call to R.memory.alloc_storage
   */
  void AddStaticBinding(const VarBindingNode* binding, bool is_alloc_storage) {
    if (is_alloc_storage) {
      current_.alloc_storage_builder->AddBinding(binding);
      binding_to_region_[binding->var.get()] = current_.alloc_storage_builder;
    } else if (current_.capture_builder != nullptr) {
      // Add the binding if the capture builder exists. It is possible that capture builder is
      // null when it is not capturing. This is the case that there are not yet any kernel launches
      // encountered, in this case static bindings (e.g. binding of other non-kernel-launch
      // operations) are marked but are not lifted.
      current_.capture_builder->AddBinding(binding);
      binding_to_region_[binding->var.get()] = current_.capture_builder;
    }
    static_vars_.emplace(binding->var.get());
  }

  /*! \brief The states of the current scope (the BindingBlock) which is a pair of FuncBuilder.
   * The FuncBuilder are initialized with nullptr, meaning the planner is currently not doing any
   * lifting. They are initialized lazily when a binding that can be lifted is encountered.
   * They are reset to nullptr when an unsupported operation is encountered.
   */
  struct Scope {
    FuncBuilder* alloc_storage_builder = nullptr;  // The builder for the allocation function
    FuncBuilder* capture_builder = nullptr;        // The builder for the capture function
  };

  // The IRModule
  IRModule mod_;
  // States of the current scope
  Scope current_;
  // Variables whose buffer address is fixed
  std::unordered_set<const VarNode*> static_vars_;
  // Binding to the FuncBuilder if the binding is lifted. This is used to update the inputs/outputs
  // of the lifted function when its binding is used outside.
  std::unordered_map<const VarNode*, FuncBuilder*> binding_to_region_;
  // The regions for capturing.
  std::vector<FuncBuilder*> captured_regions_;
  // The regions for allocation.
  std::vector<FuncBuilder*> alloc_storages_;
  // The arena.
  support::Arena arena_;
};

/*! \brief The rewriter for CUDA graph */
class CUDAGraphRewriter : public ExprMutator {
 public:
  explicit CUDAGraphRewriter(const IRModule& mod) : ExprMutator(mod) {}

  IRModule Rewrite() {
    CUDAGraphRewritePlanner planner(builder_->GetContextIRModule());
    auto plans = planner.Plan();
    for (const auto& plan : plans) {
      subgraph_launches_[plan.launch_point] = plan;
    }

    for (const auto& [gv, func] : builder_->GetContextIRModule()->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto new_func = Downcast<Function>(VisitExpr(func));
        if (!new_func.same_as(func)) {
          builder_->UpdateFunction(gv, new_func);
        }
      }
    }
    return builder_->GetContextIRModule();
  }

  void LaunchSubgraph(const VarBindingNode* op, const LiftedFunctionRewritePlan& plan) {
    static const auto& call_builtin_with_ctx_op = Op::Get("relax.call_builtin_with_ctx");
    static const auto& builtin_run_or_capture = ExternFunc("vm.builtin.cuda_graph.run_or_capture");
    static const auto& builtin_get_cached_alloc =
        ExternFunc("vm.builtin.cuda_graph.get_cached_alloc");

    Expr launch_subgraph;
    auto gv_func =
        builder_->AddFunction(plan.func, plan.is_alloc ? "cuda_graph_alloc" : "cuda_graph_capture");
    if (plan.is_alloc) {
      ICHECK(plan.inputs.empty());
      launch_subgraph =
          Call(call_builtin_with_ctx_op,
               {builtin_get_cached_alloc,
                Tuple({gv_func, PrimValue(IntImm(DataType::Int(64), index_alloc_++))})},
               Attrs(), {plan.func->ret_struct_info});
    } else {
      Array<Expr> args;
      for (const auto& arg : plan.inputs) {
        args.push_back(VisitExpr_(arg));
      }
      launch_subgraph = Call(
          call_builtin_with_ctx_op,
          {builtin_run_or_capture,
           Tuple({gv_func, Tuple(args), PrimValue(IntImm(DataType::Int(64), index_capture_++))})},
          Attrs(), {plan.func->ret_struct_info});
    }
    Expr ret_value = builder_->Emit(launch_subgraph);
    for (int i = 0; i < static_cast<int>(plan.outputs.size()); ++i) {
      // The unpacked result is saved in the var_redef_. It will be emitted when 1) the var
      // definition is the original IR is visited, or 2) the var is used as an input to another
      // lifted function, whichever comes first.
      var_redef_[plan.outputs[i]] = TupleGetItem(ret_value, i);
    }

    lifted_bindings_.insert(plan.lifted_bindings.begin(), plan.lifted_bindings.end());
  }

  void VisitBinding_(const VarBindingNode* op) final {
    if (subgraph_launches_.count(op->var.get())) {
      LaunchSubgraph(op, subgraph_launches_[op->var.get()]);
    }
    if (auto it = var_redef_.find(op->var.get());
        it != var_redef_.end() && !var_remap_.count(op->var->vid)) {
      EmitRedef(op->var.get(), it->second);
      return;
    }
    if (lifted_bindings_.count(op->var.get())) {
      // The binding is lifted to the subgraph and will be removed from the original function.
      return;
    }
    ExprMutator::VisitBinding_(op);
  }

  Expr VisitExpr_(const VarNode* op) final {
    if (auto it = var_remap_.find(op->vid); it != var_remap_.end()) {
      return it->second;
    }
    if (auto it = var_redef_.find(op); it != var_redef_.end()) {
      // This is the case that the var is used as an input to another lifted when
      // the original var definition is not visited yet.
      return EmitRedef(op, it->second);
    }
    return GetRef<Expr>(op);
  }

  Var EmitRedef(const VarNode* var, const Expr& redef) {
    auto new_var = builder_->Emit(redef, var->name_hint());
    var_remap_[var->vid] = new_var;
    return new_var;
  }

  std::unordered_map<const VarNode*, LiftedFunctionRewritePlan> subgraph_launches_;
  std::unordered_map<const VarNode*, Expr> var_redef_;
  std::unordered_set<const VarNode*> lifted_bindings_;
  int index_alloc_ = 0;
  int index_capture_ = 0;
};

IRModule RewriteCUDAGraph(IRModule mod) {
  CUDAGraphRewriter rewriter(mod);
  mod = rewriter.Rewrite();
  return mod;
}

namespace transform {

Pass RewriteCUDAGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule mod, PassContext pc) {
        bool use_cuda_graph =
            pc->GetConfig<Bool>("relax.backend.use_cuda_graph").value_or(Bool(false))->value;
        if (use_cuda_graph) {
          mod = ::tvm::relax::RewriteCUDAGraph(std::move(mod));
        }

        return mod;
      };
  return CreateModulePass(pass_func, 0, "RewriteCUDAGraph", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RewriteCUDAGraph").set_body_typed(RewriteCUDAGraph);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
