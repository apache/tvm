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
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <unordered_map>

namespace tvm {
namespace relax {

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

  void AddVDevice(Expr expr, VDevice vdevice) {
    auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(expr);
    if (tinfo != nullptr && !tinfo->vdevice.defined()) {
      if (tinfo->shape.defined()) {
        expr->struct_info_ =
            TensorStructInfo(tinfo->shape.value(), tinfo->dtype, vdevice, tinfo->span);
      } else {
        expr->struct_info_ = TensorStructInfo(tinfo->dtype, tinfo->ndim, vdevice, tinfo->span);
      }
    }
  }

  Expr VisitExpr(const Expr& expr) {
    auto visited_expr = ExprMutator::VisitExpr(expr);
    if (update_map_.count(visited_expr)) {
      AddVDevice(visited_expr, update_map_[visited_expr]);
    }
    return visited_expr;
  }

  void VisitBinding_(const VarBindingNode* binding) {
    if (update_map_.count(binding->var)) {
      update_map_[binding->value] = update_map_[binding->var];
      AddVDevice(binding->var, update_map_[binding->var]);
    }
    auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(binding->var);
    if (tinfo != nullptr && tinfo->vdevice.defined()) {
      update_map_[binding->value] = tinfo->vdevice.value();
    }
    binding->value->struct_info_ = binding->var->struct_info_;
    Expr new_value = this->VisitExpr(binding->value);
    if (!binding->var->struct_info_.defined()) {
      UpdateStructInfo(binding->var, GetStructInfo(new_value));
    }

    if (new_value.same_as(binding->value)) {
      bindings_.push_back(GetRef<VarBinding>(binding));
    } else {
      bindings_.push_back(VarBinding(binding->var, new_value));
    }
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    Function func = GetRef<Function>(op);
    auto* finfo = GetStructInfoAs<FuncStructInfoNode>(func);
    if (finfo != nullptr) {
      StructInfo ret = finfo->ret;
      auto* tinfo = finfo->ret.as<TensorStructInfoNode>();
      if (tinfo != nullptr && tinfo->vdevice.defined()) {
        update_map_[op->body] = tinfo->vdevice.value();
      }
    }
    Function visited_func = Downcast<Function>(this->VisitExprPostOrder_(op));
    return visited_func;
  }

  Expr VisitExpr_(const SeqExprNode* op) final {
    SeqExpr seq_expr = GetRef<SeqExpr>(op);
    if (update_map_.count(seq_expr)) {
      update_map_[seq_expr->body] = update_map_[seq_expr];
    }
    SeqExpr visited_seqexpr = Downcast<SeqExpr>(this->VisitExprPostOrder_(op));
    return visited_seqexpr;
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    DataflowVar visited_dtf_var = Downcast<DataflowVar>(this->VisitExprPostOrder_(op));
    if (update_map_.count(visited_dtf_var)) {
      AddVDevice(visited_dtf_var, update_map_[visited_dtf_var]);
    }
    return visited_dtf_var;
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

  Expr VisitExpr_(const CallNode* call) final {
    if (auto* sinfo = call->struct_info_.as<TensorStructInfoNode>()) {
      if (sinfo->vdevice.defined()) {
        Array<Expr> call_args;
        for (Expr arg : call->args) {
          update_map_[arg] = sinfo->vdevice.value();
        }
      }
    }
    return Downcast<Call>(ExprMutator::VisitExpr_(call));
  }

  /*! \brief The context IRModule. */
  IRModule mod_;
  /*! \brief The bindings. */
  Array<Binding> bindings_;
  /*! \brief */
  std::unordered_map<Expr, VDevice, StructuralHash, StructuralEqual> update_map_;
};

namespace transform {

Pass RealizeVDevice() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return VDeviceRealizer(mod).Run(); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"RealizeVDevice",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.RealizeVDevice").set_body_typed(RealizeVDevice);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
