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
 * \file tvm/relax/distributed/transform/lower_distir.cc
 * \brief Pass for lowering DistIR into Relax
 *  This pass assumes all the TensorIR functions are in local view,
 *  so the pass only handles sharding relax tensor shape and
 *  inserting necessary broadcast and scatter for inputs.
 */

#include <tvm/relax/attrs/ccl.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/distributed/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../tir/schedule/transform.h"
#include "../../op/ccl/ccl.h"
#include "../../op/tensor/manipulate.h"
#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

class DistIRSharder : public ExprMutator {
 public:
  static IRModule LowerDistIR(IRModule mod) { return DistIRSharder(mod).Lower(); }

 private:
  explicit DistIRSharder(IRModule mod) : ExprMutator(mod) {}

  IRModule Lower() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, base_func] : mod->functions) {
      const auto* func_ = base_func.as<FunctionNode>();
      if (func_ == nullptr || !IsDistIRFunc(GetRef<Function>(func_))) {
        continue;
      }
      Function func = RewriteFunction(GetRef<Function>(func_));
      builder_->UpdateFunction(gv, func);
    }
    return builder_->GetContextIRModule();
  }

  ShapeExpr ShardShape(ShapeExpr orig_shape, DeviceMesh device_mesh, Placement placement) {
    ShapeTuple device_mesh_shape = device_mesh->shape;
    Array<PrimExpr> new_tensor_shape_value = orig_shape->values;
    for (int i = 0; i < static_cast<int>(device_mesh_shape.size()); i++) {
      if (placement->dim_specs[i]->kind == PlacementSpecKind::kSharding) {
        int shard_size = device_mesh_shape[i];
        int axis = placement->dim_specs[i]->axis;
        new_tensor_shape_value.Set(axis, floordiv(orig_shape->values[axis], shard_size));
      }
    }
    return ShapeExpr(new_tensor_shape_value);
  }

  TensorStructInfo ShardDTensorSinfo(DTensorStructInfo orig_sinfo) {
    TensorStructInfo tensor_sinfo = orig_sinfo->tensor_sinfo;
    ICHECK(tensor_sinfo->shape);
    const auto* orig_shape = tensor_sinfo->shape.as<ShapeExprNode>();
    auto new_tensor_sinfo = make_object<TensorStructInfoNode>(*tensor_sinfo.get());
    new_tensor_sinfo->shape =
        ShardShape(GetRef<ShapeExpr>(orig_shape), orig_sinfo->device_mesh, orig_sinfo->placement);
    return TensorStructInfo(new_tensor_sinfo);
  }

  StructInfo ConvertSinfo(StructInfo orig_sinfo, bool shard_shape) {
    if (const auto* dtensor_sinfo = orig_sinfo.as<DTensorStructInfoNode>()) {
      if (shard_shape) {
        return ShardDTensorSinfo(GetRef<DTensorStructInfo>(dtensor_sinfo));
      } else {
        return dtensor_sinfo->tensor_sinfo;
      }
    } else if (const auto* tuple_sinfo = orig_sinfo.as<TupleStructInfoNode>()) {
      Array<StructInfo> new_fields;
      for (const auto& field_sinfo : tuple_sinfo->fields) {
        if (const auto* dtensor_sinfo = field_sinfo.as<DTensorStructInfoNode>()) {
          if (shard_shape) {
            new_fields.push_back(ShardDTensorSinfo(GetRef<DTensorStructInfo>(dtensor_sinfo)));
          } else {
            new_fields.push_back(dtensor_sinfo->tensor_sinfo);
          }
        } else {
          new_fields.push_back(field_sinfo);
        }
      }
      return TupleStructInfo(new_fields);
    } else {
      return orig_sinfo;
    }
  }

  Expr ShardInputParamTensorAndConstant(Expr input) {
    ICHECK(input->struct_info_);
    StructInfo old_sinfo = GetStructInfo(input);
    StructInfo new_sinfo = ConvertSinfo(old_sinfo, false);
    if (const auto* var = input.as<VarNode>()) {
      Var new_param(var->name_hint(), new_sinfo);
      return new_param;
    } else if (const auto* constant = input.as<ConstantNode>()) {
      for (const auto& spec : Downcast<DTensorStructInfo>(old_sinfo)->placement->dim_specs) {
        ICHECK(spec->kind == PlacementSpecKind::kReplica);
      }
      Constant new_constant(constant->data, new_sinfo);
      return new_constant;
    } else {
      LOG(FATAL) << "Cannot shard tensor which is not Var or Constant: " << input;
      throw;
    }
  }

  void EmitBroadcastOrScatter(Expr old_expr, Expr new_expr, DTensorStructInfo dtensor_sinfo) {
    // FIXME: this is a hack that only works for 1d device mesh
    ICHECK(dtensor_sinfo->device_mesh->shape.size() == 1);
    PlacementSpec sharding_spec = dtensor_sinfo->placement->dim_specs[0];
    if (sharding_spec->kind == PlacementSpecKind::kReplica) {
      Var new_var = builder_->Emit(broadcast_from_worker0(new_expr));
      if (const auto* var = old_expr.as<VarNode>()) {
        var_remap_[var->vid] = new_var;
      } else {
        tuple_getitem_remap_[Downcast<TupleGetItem>(old_expr)] = new_var;
      }
    } else if (sharding_spec->kind == PlacementSpecKind::kSharding) {
      Var scatter_var = builder_->Emit(scatter_from_worker0(
          new_expr, dtensor_sinfo->device_mesh->shape[0], sharding_spec->axis));
      if (const auto* var = old_expr.as<VarNode>()) {
        var_remap_[var->vid] = scatter_var;
      } else {
        tuple_getitem_remap_[Downcast<TupleGetItem>(old_expr)] = scatter_var;
      }
    } else {
      LOG(FATAL) << "Unsupported placement spec";
    }
  }

  void InputPreprocessing() {
    for (int i = 0; i < static_cast<int>(func_->params.size()); i++) {
      Var param = func_->params[i];
      if (const auto* dtensor_sinfo = GetStructInfoAs<DTensorStructInfoNode>(param)) {
        EmitBroadcastOrScatter(param, new_params_[i], GetRef<DTensorStructInfo>(dtensor_sinfo));
      } else if (const auto* tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(param)) {
        for (int j = 0; j < static_cast<int>(tuple_sinfo->fields.size()); j++) {
          if (const auto* dtensor_sinfo = tuple_sinfo->fields[j].as<DTensorStructInfoNode>()) {
            EmitBroadcastOrScatter(TupleGetItem(param, j), TupleGetItem(new_params_[i], j),
                                   GetRef<DTensorStructInfo>(dtensor_sinfo));
          }
        }
      }
    }
  }

  Function RewriteFunction(Function func) {
    Array<Var> new_params;
    for (const Var& var : func->params) {
      Var new_param = Downcast<Var>(ShardInputParamTensorAndConstant(var));
      var_remap_[var->vid] = new_param;
      new_params.push_back(new_param);
    }
    func_ = func;
    new_params_ = new_params;
    auto new_body = VisitWithNewScope(func->body, new_params);
    Function new_func(new_params, new_body, NullOpt, func->is_pure, func->attrs);
    return new_func;
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) {
    if (tuple_getitem_remap_.count(GetRef<TupleGetItem>(val))) {
      var_remap_[binding->var->vid] = tuple_getitem_remap_[GetRef<TupleGetItem>(val)];
    } else {
      ExprMutator::VisitBinding_(binding, val);
    }
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) {
    builder_->BeginBindingBlock();
    InputPreprocessing();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) {
    builder_->BeginDataflowBlock();
    InputPreprocessing();
    for (auto binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  Call HandleSpecialCaseinDTensorLowering(const CallNode* call, Var binding_var) {
    static Op reshape_op = Op::Get("relax.reshape");
    static Op call_tir_op = Op::Get("relax.call_tir");
    static Op call_tir_local_view_op = Op::Get("relax.dist.call_tir_local_view");
    if (call->op.same_as(reshape_op)) {
      ICHECK(call->args[1].as<ShapeExprNode>());
      const auto* out_sinfo = GetStructInfoAs<DTensorStructInfoNode>(binding_var);
      ICHECK(out_sinfo);
      auto new_call_node = make_object<CallNode>(*call);
      new_call_node->args.Set(1, ShardShape(Downcast<ShapeExpr>(call->args[1]),
                                            out_sinfo->device_mesh, out_sinfo->placement));
      return Call(new_call_node);
    } else if (call->op.same_as(call_tir_local_view_op)) {
      auto new_call_node = make_object<CallNode>(*call);
      new_call_node->op = call_tir_op;
      new_call_node->sinfo_args = {ConvertSinfo(GetStructInfo(binding_var), true)};
      return Call(new_call_node);
    } else if (call->op.same_as(call_tir_op)) {
      LOG(FATAL) << "call_tir should be lowered to call_tir_local_view before lowering to relax";
    } else if (const auto* extern_func = call->op.as<ExternFuncNode>()) {
      auto new_call_node = make_object<CallNode>(*call);
      if (extern_func->global_symbol == "vm.builtin.distributed.attention_kv_cache_append") {
        new_call_node->op = ExternFunc("vm.builtin.attention_kv_cache_append");
      } else if (extern_func->global_symbol == "vm.builtin.distributed.attention_kv_cache_view") {
        new_call_node->op = ExternFunc("vm.builtin.attention_kv_cache_view");
        auto orig_shape = Downcast<ShapeExpr>(call->args[1]);
        const auto* out_sinfo = GetStructInfoAs<DTensorStructInfoNode>(binding_var);
        ICHECK(out_sinfo);
        ShapeExpr new_shape = ShardShape(orig_shape, out_sinfo->device_mesh, out_sinfo->placement);
        new_call_node->args.Set(1, new_shape);
        new_call_node->sinfo_args = {TensorStructInfo(new_shape, out_sinfo->tensor_sinfo->dtype)};
      }
      return Call(new_call_node);
    }
    return GetRef<Call>(call);
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) {
    Call new_call =
        Downcast<Call>(this->VisitExpr(HandleSpecialCaseinDTensorLowering(val, binding->var)));
    ReEmitBinding(binding, builder_->Normalize(new_call));
  }

  Function func_;
  Array<Var> new_params_;
  std::unordered_map<TupleGetItem, Var, StructuralHash, StructuralEqual> tuple_getitem_remap_;
};

namespace transform {

Pass LowerDistIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return DistIRSharder::LowerDistIR(m); };
  return CreateModulePass(pass_func, 1, "LowerDistIR", {});
}
TVM_REGISTER_GLOBAL("relax.distributed.transform.LowerDistIR").set_body_typed(LowerDistIR);
}  // namespace transform

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
