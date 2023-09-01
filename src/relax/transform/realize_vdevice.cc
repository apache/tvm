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
 * \file tvm/relax/transform/realize_vdevice.cc
 * \brief Propagate virtual device information.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

void UpdateTensorStructInfo(Expr expr, StructInfo struct_info) {
  if (auto* tensor_sinfo = expr->struct_info_.as<TensorStructInfoNode>()) {
    auto* new_tensor_sinfo = struct_info.as<TensorStructInfoNode>();
    if (new_tensor_sinfo != nullptr && new_tensor_sinfo->vdevice.defined() &&
        !tensor_sinfo->vdevice.defined()) {
      expr->struct_info_ = struct_info;
      expr->checked_type_ = GetStaticType(struct_info);
    }
  }
}

void AddVDeviceToStuctInfo(Expr expr, VDevice vdevice) {
  auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(expr);
  if (tinfo != nullptr) {
    if (tinfo->shape.defined()) {
      UpdateTensorStructInfo(
          expr, TensorStructInfo(tinfo->shape.value(), tinfo->dtype, vdevice, tinfo->span));
    } else {
      UpdateTensorStructInfo(expr,
                             TensorStructInfo(tinfo->dtype, tinfo->ndim, vdevice, tinfo->span));
    }
  }
}

class VDeviceRealizer : public ExprMutator {
 public:
  explicit VDeviceRealizer(const IRModule& mod) : ExprMutator(mod), mod_(std::move(mod)) {}

  IRModule Run() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = Downcast<Function>(this->VisitExpr(func));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  void AddToVDeviceMap(Expr expr, VDevice vdevice) {
    ICHECK((vdevice_map_.count(expr) == 0) || (vdevice_map_[expr] == vdevice))
        << "Conflicted vdevice found.";
    vdevice_map_.Set(expr, vdevice);
  }

  Expr VisitExpr(const Expr& expr) {
    auto visited_expr = ExprMutator::VisitExpr(expr);
    if (vdevice_map_.count(visited_expr)) {
      AddVDeviceToStuctInfo(visited_expr, vdevice_map_[visited_expr]);
    }
    return visited_expr;
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    Function func = GetRef<Function>(op);
    auto* finfo = GetStructInfoAs<FuncStructInfoNode>(func);
    if (finfo != nullptr) {
      StructInfo ret = finfo->ret;
      auto* tinfo = finfo->ret.as<TensorStructInfoNode>();
      if (tinfo != nullptr && tinfo->vdevice.defined()) {
        AddToVDeviceMap(op->body, tinfo->vdevice.value());
      }
    }
    Function visited_func = Downcast<Function>(this->VisitExprPostOrder_(op));
    return visited_func;
  }

  Expr VisitExpr_(const SeqExprNode* op) final {
    SeqExpr seq_expr = GetRef<SeqExpr>(op);
    if (vdevice_map_.count(seq_expr)) {
      AddToVDeviceMap(seq_expr->body, vdevice_map_[seq_expr]);
    }
    SeqExpr visited_seqexpr = Downcast<SeqExpr>(this->VisitExprPostOrder_(op));
    return visited_seqexpr;
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) {
    builder_->BeginBindingBlock();
    for (size_t i = block->bindings.size(); i > 0; --i) {
      this->VisitBinding(block->bindings[i - 1]);
    }
    for (size_t i = bindings_.size(); i > 0; --i) {
      builder_->EmitNormalized(bindings_[i - 1]);
    }
    bindings_.clear();
    return builder_->EndBlock();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) {
    builder_->BeginDataflowBlock();
    for (size_t i = block->bindings.size(); i > 0; --i) {
      this->VisitBinding(block->bindings[i - 1]);
    }
    for (size_t i = bindings_.size(); i > 0; --i) {
      builder_->EmitNormalized(bindings_[i - 1]);
    }
    bindings_.clear();
    return builder_->EndBlock();
  }

  void VisitBinding_(const VarBindingNode* binding) {
    if (vdevice_map_.count(binding->var)) {
      AddToVDeviceMap(binding->value, vdevice_map_[binding->var]);
      AddVDeviceToStuctInfo(binding->var, vdevice_map_[binding->var]);
    }
    auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(binding->var);
    if (tinfo != nullptr && tinfo->vdevice.defined()) {
      AddToVDeviceMap(binding->value, tinfo->vdevice.value());
    }
    UpdateTensorStructInfo(binding->value, GetStructInfo(binding->var));
    Expr new_value = this->VisitExpr(binding->value);
    if (!binding->var->struct_info_.defined()) {
      UpdateTensorStructInfo(binding->var, GetStructInfo(new_value));
    }

    if (new_value.same_as(binding->value)) {
      bindings_.push_back(GetRef<VarBinding>(binding));
    } else {
      bindings_.push_back(VarBinding(binding->var, new_value));
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    // Record the vdevice information of each arguments of call
    if (auto* sinfo = call->struct_info_.as<TensorStructInfoNode>()) {
      if (sinfo->vdevice.defined() && call->op != to_vdevice_op_) {
        Array<Expr> call_args;
        for (Expr arg : call->args) {
          AddToVDeviceMap(arg, sinfo->vdevice.value());
        }
      }
    }
    return Downcast<Call>(ExprMutator::VisitExpr_(call));
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
  /*! \brief The bindings in reverse ordering. */
  Array<Binding> bindings_;
  /*! \brief The virtual device map. */
  Map<Expr, VDevice> vdevice_map_;

  const Op& to_vdevice_op_ = Op::Get("relax.to_vdevice");
};

class HintOnDeviceRemover : public ExprMutator {
 public:
  explicit HintOnDeviceRemover(const IRModule& mod) : ExprMutator(mod), mod_(std::move(mod)) {}

  IRModule Run() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = Downcast<Function>(this->VisitExpr(func));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  void AddToVDeviceMap(Expr expr, VDevice vdevice) {
    ICHECK((vdevice_map_.count(expr) == 0) || (vdevice_map_[expr] == vdevice))
        << "Conflicted vdevice found.";
    vdevice_map_.Set(expr, vdevice);
  }

  VDevice LookupVDevice(int32_t device_type, int32_t device_id) {
    Array<GlobalInfo> vdevices = mod_->global_infos["vdevice"];
    if (vdevices.empty() || device_id < 0 || static_cast<size_t>(device_id) >= vdevices.size()) {
      LOG(FATAL) << "ValueError: The target VDevice in the GlobalInfos was not found.";
    }
    for (auto vdev : vdevices) {
      auto vdevice = Downcast<VDevice>(vdev);
      int dev_type = vdevice->target->GetTargetDeviceType();
      if (dev_type == device_type && vdevice->vdevice_id == device_id) {
        return vdevice;
      }
    }
    LOG(WARNING) << "The specified device was not found in the global_infos";
    return VDevice();
  }

  Expr VisitExpr(const Expr& expr) {
    auto visited_expr = ExprMutator::VisitExpr(expr);
    if (vdevice_map_.count(visited_expr)) {
      AddVDeviceToStuctInfo(visited_expr, vdevice_map_[visited_expr]);
    }
    return visited_expr;
  }

  void VisitBinding_(const VarBindingNode* binding) {
    Expr new_value = this->VisitExpr(binding->value);
    UpdateTensorStructInfo(binding->var, GetStructInfo(new_value));
    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      builder_->EmitNormalized(VarBinding(binding->var, new_value));
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    // Replace hint_on_device with to_vdevice
    if (call->op == hint_on_device_op_) {
      // Find out the vdevice from global_infos
      Expr data = call->args[0];
      auto attrs = call->attrs.as<HintOnDeviceAttrs>();
      int32_t device_type = attrs->dev_type;
      int32_t device_id = attrs->dev_id;
      VDevice dst_vdev = LookupVDevice(device_type, device_id);
      // Insert to_vdevice if input are on different device
      auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(data);
      if (tinfo != nullptr) {
        if (!tinfo->vdevice.defined()) {
          // Remove hint_on_device
          AddVDeviceToStuctInfo(data, dst_vdev);
          AddToVDeviceMap(data, dst_vdev);
          return data;
        } else if (tinfo->vdevice.value() != dst_vdev) {
          // Call to_vdevice
          ObjectPtr<ToVDeviceAttrs> attrs = make_object<ToVDeviceAttrs>();
          attrs->dst_vdevice = dst_vdev;
          auto new_call = Call(to_vdevice_op_, {data}, Attrs(attrs), {});
          AddToVDeviceMap(new_call, dst_vdev);
          return new_call;
        }
      }
    }

    auto visited_call = ExprMutator::VisitExpr_(call);
    visited_call->struct_info_ = NullOpt;
    return builder_->Normalize(visited_call);
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
  /*! \brief The virtual device map. */
  Map<Expr, VDevice> vdevice_map_;

  const Op& hint_on_device_op_ = Op::Get("relax.hint_on_device");
  const Op& to_vdevice_op_ = Op::Get("relax.to_vdevice");
};

namespace transform {

Pass RealizeVDevice() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    IRModule new_mod = HintOnDeviceRemover(mod).Run();
    return VDeviceRealizer(new_mod).Run();
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"RealizeVDevice",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.RealizeVDevice").set_body_typed(RealizeVDevice);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
