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
 * \file tvm/relax/distributed/transform/propagate_sharding.cc
 * \brief Pass for propagating sharding information.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/distributed.h>
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>

#include <numeric>

#include "../../op/distributed/distributed.h"
#include "../../op/distributed/utils.h"

namespace tvm {
namespace relax {
namespace distributed {

void CollectAxisGraphBinary(const VarBindingNode* binding, const CallNode* call,
                            AxisGroupGraph* axis_group_graph) {
  const std::vector<std::string> binary_op_names = {
      "add",     "subtract",      "multiply", "divide",     "power",     "floor_divide", "equal",
      "greater", "greater_equal", "less",     "less_equal", "not_equal", "minimum",      "maximum"};
  for (const auto& op_name : binary_op_names) {
    const Op& binary_op = Op::Get("relax." + op_name);
    if (call->op.same_as(binary_op)) {
      BuildAxisGraphBinary(binding->var, GetRef<Call>(call), axis_group_graph);
      break;
    }
  }
}

void CollectAxisGraphUnary(const VarBindingNode* binding, const CallNode* call,
                           AxisGroupGraph* axis_group_graph) {
  const std::vector<std::string> unary_op_names = {
      "abs",   "acos",     "acosh",    "asin",   "asinh", "atan",
      "atanh", "ceil",     "cos",      "cosh",   "exp",   "floor",
      "log",   "negative", "nn.relu",  "round",  "rsqrt", "sigmoid",
      "sign",  "sin",      "sinh",     "square", "sqrt",  "tan",
      "tanh",  "clip",     "isfinite", "isinf",  "isnan", "dist.annotate_sharding",
      "erf",   "nn.gelu"};
  for (const auto& op_name : unary_op_names) {
    const Op& unary_op = Op::Get("relax." + op_name);
    if (call->op.same_as(unary_op)) {
      BuildAxisGraphUnary(binding->var, GetRef<Call>(call), axis_group_graph);
    }
  }
}

void CollectAxisGraphReduce(const VarBindingNode* binding, const CallNode* call,
                            AxisGroupGraph* axis_group_graph) {
  const std::vector<std::string> reduction_op_names = {"sum",  "max", "min",      "prod",
                                                       "mean", "std", "variance", "nn.softmax"};
  for (const auto& op_name : reduction_op_names) {
    const Op& reduction_op = Op::Get("relax." + op_name);
    if (call->op.same_as(reduction_op)) {
      BuildAxisGraphReduce(binding->var, GetRef<Call>(call), axis_group_graph);
      break;
    }
  }
}

void CollectAxisGraphMatmul(const VarBindingNode* binding, const CallNode* call,
                            AxisGroupGraph* axis_group_graph) {
  static const Op& matmul_op = Op::Get("relax.matmul");
  if (call->op.same_as(matmul_op)) {
    BuildAxisGraphMatmul(binding->var, GetRef<Call>(call), axis_group_graph);
  }
}

void CollectAxisGraphPermuteDims(const VarBindingNode* binding, const CallNode* call,
                                 AxisGroupGraph* axis_group_graph) {
  static const Op& permute_dims_op = Op::Get("relax.permute_dims");
  if (call->op.same_as(permute_dims_op)) {
    BuildAxisGraphPermuteDims(binding->var, GetRef<Call>(call), axis_group_graph);
  }
}

void CollectAxisGraphReshape(const VarBindingNode* binding, const CallNode* call,
                             AxisGroupGraph* axis_group_graph) {
  static const Op& reshape_op = Op::Get("relax.reshape");
  if (call->op.same_as(reshape_op)) {
    BuildAxisGraphReshape(binding->var, GetRef<Call>(call), axis_group_graph);
  }
}

void CollectAxisGraphForDeviceMesh(const VarBindingNode* binding, const CallNode* call,
                                   AxisGroupGraph* axis_group_graph) {
  Array<Expr> tensor_list;
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  Array<Expr> args;
  if (call->op.same_as(call_tir_op)) {
    args = Downcast<Tuple>(call->args[1])->fields;
  } else {
    args = call->args;
  }
  for (const auto& arg : args) {
    if (arg->struct_info_.as<TensorStructInfoNode>()) {
      tensor_list.push_back(arg);
    }
  }
  for (int i = 0; i < static_cast<int>(tensor_list.size()); i++) {
    axis_group_graph->JoinAxis(Axis(tensor_list[i].get(), -1), {binding->var.get(), -1},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
  }
}
/*!
 * \brief Pattern match op to a TIR function and look it up.
 * \return The TIR function, or nullopt if pattern match fails.
 */
Optional<tir::PrimFunc> MatchPrimFunc(const IRModule& mod_, const Expr& op) {
  const GlobalVar& global_var = Downcast<GlobalVar>(op);
  // NOTE: as check works for nullptr(returns null)
  Optional<BaseFunc> base_func = mod_->functions.Get(global_var);
  if (auto* pfunc = base_func.as<tir::PrimFuncNode>()) {
    return GetRef<tir::PrimFunc>(pfunc);
  }
  return NullOpt;
}

/*!
 * \brief Build an axis group graph for propagation.
 */
class AxisGroupGraphBuilder : public ExprVisitor {
 public:
  static void BuildAxisGroupGraph(AxisGroupGraph* axis_group_graph, const Function& func,
                                  const IRModule& mod) {
    AxisGroupGraphBuilder builder(axis_group_graph, mod);
    builder.VisitExpr(func);
  }

 private:
  explicit AxisGroupGraphBuilder(AxisGroupGraph* axis_group_graph, IRModule mod)
      : axis_group_graph_(axis_group_graph), mod_(mod) {}

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) {
    CollectAxisGraphBinary(binding, val, axis_group_graph_);
    CollectAxisGraphUnary(binding, val, axis_group_graph_);
    CollectAxisGraphReduce(binding, val, axis_group_graph_);
    CollectAxisGraphMatmul(binding, val, axis_group_graph_);
    CollectAxisGraphPermuteDims(binding, val, axis_group_graph_);
    CollectAxisGraphReshape(binding, val, axis_group_graph_);
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (val->op.same_as(call_tir_op)) {
      if (Optional<tir::PrimFunc> func = MatchPrimFunc(mod_, val->args[0])) {
        BuildAxisGraphCallTIR(binding->var, GetRef<Call>(val), func.value(), axis_group_graph_);
      }
    }
    CollectAxisGraphForDeviceMesh(binding, val, axis_group_graph_);
    ExprVisitor::VisitBinding_(binding, val);
  }
  AxisGroupGraph* axis_group_graph_;
  IRModule mod_;
};

/*!
 * \brief Collect the sharding annotations and add source sharding spec in axis group graph.
 */
class ShardingAnnotationCollector : public ExprVisitor {
 public:
  static void CollectShardingAnnotation(AxisGroupGraph* axis_group_graph, const Function& func) {
    ShardingAnnotationCollector collector(axis_group_graph);
    collector.VisitExpr(func);
  }

 private:
  explicit ShardingAnnotationCollector(AxisGroupGraph* axis_group_graph)
      : axis_group_graph_(axis_group_graph) {}
  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) {
    static const Op& annotate_sharding_op = Op::Get("relax.dist.annotate_sharding");
    if (val->op.same_as(annotate_sharding_op)) {
      const auto* attrs = val->attrs.as<DistributionAttrs>();
      ICHECK(attrs);

      for (int i = 0; i < static_cast<int>(attrs->placement->dim_specs.size()); i++) {
        const PlacementSpec& placement_spec = attrs->placement->dim_specs[i];
        if (placement_spec->kind == PlacementSpecKind::kSharding) {
          axis_group_graph_->AddSrcShardingPoint({binding->var.get(), placement_spec->axis},
                                                 {attrs->device_mesh, i});
        }
      }
      axis_group_graph_->AddSrcShardingPoint({binding->var.get(), -1}, {attrs->device_mesh, -1});
    }
    ExprVisitor::VisitBinding_(binding, val);
  }

  AxisGroupGraph* axis_group_graph_;
};

/*!
 * \brief Check if the sharding of each tensor is legal.
 */
class ShardingConflictHandler : public ExprVisitor {
 public:
  static void HandleShardingConflict(AxisGroupGraph* axis_group_graph, Function function) {
    axis_group_graph->PropagateShardingSpec();
    ShardingConflictHandler handler(axis_group_graph);
    handler.VisitExpr(function);
    for (const Var& var : function->params) {
      if (GetStructInfoAs<TensorStructInfoNode>(var)) {
        handler.CheckTensorShardingCompatible(var);
      }
    }
    axis_group_graph->PropagateShardingSpec();
  }

 private:
  explicit ShardingConflictHandler(AxisGroupGraph* axis_group_graph)
      : axis_group_graph_(axis_group_graph) {}

  void CheckTensorShardingCompatible(Var var) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(var);
    ICHECK(sinfo);
    const auto* shape = sinfo->shape.as<ShapeExprNode>();
    ICHECK(shape);
    int ndim = sinfo->ndim;
    std::unordered_set<int> sharded_mesh_dim;
    Optional<DeviceMesh> device_mesh;
    for (int i = -1; i < ndim; i++) {
      AxisShardingSpec sharding_spec;
      int has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph_->GetAxisShardingSpec({var.get(), i});
      if (!has_sharding_spec) {
        continue;
      }

      if (device_mesh.defined()) {
        ICHECK(StructuralEqual()(device_mesh.value(), sharding_spec.first))
            << "Sharding conflict detected for tensor " << var->name_hint()
            << ": Device Mesh mismatch"
            << ". Conflict Handling logic will be added in the future.";
      } else {
        device_mesh = sharding_spec.first;
      }
      if (i >= 0) {
        int sharding_dim = sharding_spec.second;
        ICHECK(sharded_mesh_dim.count(sharding_dim) == 0)
            << "Sharding conflict detected for tensor " << var->name_hint()
            << ": Replicate sharding device mesh axis " << sharding_dim
            << ". Conflict Handling logic will be added in the future.";
        sharded_mesh_dim.insert(sharding_dim);
        if (const auto* val = shape->values[i].as<IntImmNode>()) {
          if (val->value < device_mesh.value()->shape[sharding_spec.second]) {
            axis_group_graph_->AddPropagationCutPoint({var.get(), i}, sharding_spec);
          }
        }
      }
    }
  }

  void CheckConstantNoSharding(Constant constant) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(constant);
    for (int i = 0; i < sinfo->ndim; i++) {
      AxisShardingSpec sharding_spec;
      int has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph_->GetAxisShardingSpec({constant.get(), i});
      ICHECK(!has_sharding_spec)
          << "Constant is not allowed to be sharded. Please convert it into an input param.";
    }
  }

  void VisitExpr_(const CallNode* op) final {
    Array<Expr> args = GetCallArgs(GetRef<Call>(op));
    for (const auto& arg : args) {
      if (arg.as<ConstantNode>()) {
        CheckConstantNoSharding(Downcast<Constant>(arg));
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    if (GetStructInfoAs<TensorStructInfoNode>(binding->var)) {
      CheckTensorShardingCompatible(binding->var);
    }
    ExprVisitor::VisitBinding_(binding);
  }

  AxisGroupGraph* axis_group_graph_;
};

/*!
 * \brief Build distributed IR from given sharding annotation
 */
class DistributedIRBuilder : public ExprMutator {
 public:
  explicit DistributedIRBuilder(const IRModule& module) : ExprMutator(module) {}

  IRModule BuildDistributedIR() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionNode>();
      if (func_ == nullptr) {
        continue;
      }
      Function func = RewriteFunction(GetRef<Function>(func_), mod);
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  Expr RewriteInputTensorAndConstant(Expr tensor) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(tensor);
    int ndim = sinfo->ndim;
    DeviceMesh device_mesh =
        std::get<0>(axis_group_graph_.GetAxisShardingSpec({tensor.get(), -1})).first;
    ICHECK(device_mesh.defined()) << tensor << " is not assigned device mesh";
    Array<PlacementSpec> placement_specs(
        std::vector<PlacementSpec>(device_mesh->shape.size(), PlacementSpec::Replica()));
    for (int i = 0; i < ndim; i++) {
      AxisShardingSpec sharding_spec;
      bool has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph_.GetAxisShardingSpec({tensor.get(), i});
      if (has_sharding_spec) {
        int sharding_dim = sharding_spec.second;
        placement_specs.Set(sharding_dim, PlacementSpec::Sharding(i));
      }
    }
    if (const auto* var = tensor.as<VarNode>()) {
      Var new_param(var->name_hint(), DTensorStructInfo(GetRef<TensorStructInfo>(sinfo),
                                                        device_mesh, Placement(placement_specs)));
      return new_param;
    } else if (const auto* constant = tensor.as<ConstantNode>()) {
      Constant new_constant(constant->data,
                            DTensorStructInfo(GetRef<TensorStructInfo>(sinfo), device_mesh,
                                              Placement(placement_specs)));
      return new_constant;
    } else {
      LOG(FATAL) << "Cannot rewrite tensor which is not a Var or Constant";
      throw;
    }
  }

  Function RewriteFunction(Function func, IRModule mod) {
    // Step 1. Construct AxisGroupGraph
    AxisGroupGraphBuilder::BuildAxisGroupGraph(&axis_group_graph_, func, mod);
    // Step 2. Collect Sharding Annotation
    ShardingAnnotationCollector::CollectShardingAnnotation(&axis_group_graph_, func);
    // Step 3. Handle Sharding Conflict
    ShardingConflictHandler::HandleShardingConflict(&axis_group_graph_, func);
    // Step 4. Rewrite Function
    Array<Var> new_params;
    for (const Var& var : func->params) {
      if (GetStructInfoAs<TensorStructInfoNode>(var)) {
        Var new_param = Downcast<Var>(RewriteInputTensorAndConstant(var));
        input_tensor_remap_.Set(var, new_param);
        new_params.push_back(new_param);
      } else {
        new_params.push_back(var);
      }
    }
    auto new_body = VisitWithNewScope(func->body, new_params);
    Function new_func(new_params, new_body, NullOpt, func->is_pure, func->attrs);
    return new_func;
  }

  Expr VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    FBuildAxisGraph f = [&](const Var& var, const Call& call, AxisGroupGraph* axis_group_graph) {
      Optional<tir::PrimFunc> prim_func =
          MatchPrimFunc(this->builder_->GetContextIRModule(), call->args[0]);
      ICHECK(prim_func);
      return BuildAxisGraphCallTIR(var, call, prim_func.value(), axis_group_graph);
    };
    Call new_call = Downcast<Call>(ExprMutator::VisitExpr_(call));
    Array<Expr> args = GetCallArgs(new_call);
    for (int i = 0; i < static_cast<int>(args.size()); i++) {
      if (args[i].as<ConstantNode>()) {
        args.Set(i, RewriteInputTensorAndConstant(args[i]));
      }
    }

    ObjectPtr<CallNode> n = make_object<CallNode>(*new_call.get());
    if (new_call->op.same_as(call_tir_op)) {
      n->args.Set(1, Tuple(args));
      ICHECK(new_call->sinfo_args[0]->IsInstance<TensorStructInfoNode>());
      ShardingSpec sharding_spec = InferShardingSpec(
          Call(n), this->builder_, Downcast<TensorStructInfo>(new_call->sinfo_args[0]), f);
      n->sinfo_args = {DTensorStructInfo(Downcast<TensorStructInfo>(new_call->sinfo_args[0]),
                                         sharding_spec.first, sharding_spec.second)};
    } else {
      n->args = args;
    }

    if (const auto* extern_func = new_call->op.as<ExternFuncNode>()) {
      if (extern_func->global_symbol == "vm.builtin.attention_kv_cache_append") {
        n->op = ExternFunc("vm.builtin.distributed.attention_kv_cache_append");
      } else if (extern_func->global_symbol == "vm.builtin.attention_kv_cache_view") {
        n->op = ExternFunc("vm.builtin.distributed.attention_kv_cache_view");
      }
    }
    return Call(n);
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) {
    const auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(binding->var);
    if (!sinfo) {
      ExprMutator::VisitBinding_(binding, val);
      return;
    }
    int ndim = sinfo->ndim;
    bool insert_redistribute = false;
    // get DTensorStuctInfo from axis group graph
    DeviceMesh device_mesh =
        std::get<0>(axis_group_graph_.GetAxisShardingSpec({binding->var.get(), -1})).first;
    Array<PlacementSpec> placement_specs(
        std::vector<PlacementSpec>(device_mesh->shape.size(), PlacementSpec::Replica()));

    for (int i = 0; i < ndim; i++) {
      AxisShardingSpec sharding_spec;
      bool has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph_.GetAxisShardingSpec({binding->var.get(), i});
      if (has_sharding_spec) {
        placement_specs.Set(sharding_spec.second, PlacementSpec::Sharding(i));
      }
    }
    // get DTensorStuctInfo from struct info deduction
    Call new_call = Downcast<Call>(this->VisitExpr(binding->value));
    if (const auto* extern_func = new_call->op.as<ExternFuncNode>()) {
      if (extern_func->global_symbol == "vm.builtin.distributed.attention_kv_cache_view") {
        ObjectPtr<CallNode> new_call_node = make_object<CallNode>(*new_call.get());
        StructInfo new_dtensor_sinfo =
            DTensorStructInfo(Downcast<TensorStructInfo>(new_call->sinfo_args[0]), device_mesh,
                              Placement(placement_specs));
        new_call_node->sinfo_args = {new_dtensor_sinfo};
        new_call = Call(new_call_node);
        new_call->struct_info_ = new_dtensor_sinfo;
      }
    }
    Expr new_value = builder_->Normalize(new_call);
    const auto* inferred_dtensor_sinfo = GetStructInfoAs<DTensorStructInfoNode>(new_value);
    ICHECK(inferred_dtensor_sinfo);
    if (!StructuralEqual()(DTensorStructInfo(inferred_dtensor_sinfo->tensor_sinfo, device_mesh,
                                             Placement(placement_specs)),
                           GetRef<DTensorStructInfo>(inferred_dtensor_sinfo))) {
      insert_redistribute = true;
    }

    static const Op& annotate_sharding_op = Op::Get("relax.dist.annotate_sharding");
    if (val->op.same_as(annotate_sharding_op)) {
      if (insert_redistribute) {
        Expr redistribute_call =
            redistribute(new_call->args[0], device_mesh, Placement(placement_specs));
        redistribute_call = builder_->Normalize(redistribute_call);
        ReEmitBinding(binding, redistribute_call);
      } else {
        ReEmitBinding(binding, new_call->args[0]);
      }
    } else {
      if (insert_redistribute) {
        Expr redistribute_call = redistribute(new_call, device_mesh, Placement(placement_specs));
        redistribute_call = builder_->Normalize(redistribute_call);
        ReEmitBinding(binding, redistribute_call);
      } else {
        ReEmitBinding(binding, new_call);
      }
    }
  }

  Expr VisitExpr_(const VarNode* var) final {
    auto it = input_tensor_remap_.find(GetRef<Var>(var));
    if (it != input_tensor_remap_.end()) {
      return (*it).second;
    }
    return ExprMutator::VisitExpr_(var);
  }

  Map<Var, Var> input_tensor_remap_;
  AxisGroupGraph axis_group_graph_;
};
namespace transform {

Pass PropagateSharding() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return DistributedIRBuilder(m).BuildDistributedIR(); };
  return CreateModulePass(pass_func, 1, "PropagateSharding", {});
}
TVM_REGISTER_GLOBAL("relax.distributed.transform.PropagateSharding")
    .set_body_typed(PropagateSharding);
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
