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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/transform/allocate_workspace.cc
 * \brief Allocate a workspace and append it to the arguments of external functions, to
 * satisfy their temporary storage requirement.
 */

#include <tvm/ir/name_supply.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include "../op/op_common.h"

namespace tvm {
namespace relax {

class ExternFunctionRewriter : ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  ExternFunctionRewriter(IRModule mod, size_t max_workspace_size)
      : ExprMutator(mod), name_sup_(""), max_workspace_size_(max_workspace_size) {}

  std::unordered_map<const GlobalVarNode*, Function> Run() {
    std::unordered_map<const GlobalVarNode*, Function> ret;
    for (const auto& [gvar, f] : builder_->GetContextIRModule()->functions) {
      if (f->GetAttr<Integer>(attr::kWorkspaceSize)) {
        ret[gvar.get()] = Downcast<Function>(VisitExpr(f));
      }
    }
    return ret;
  }

  Expr VisitExpr_(const FunctionNode* func_node) override {
    if (!func_node->GetAttr<String>(attr::kCodegen) &&
        !func_node->GetAttr<String>(attr::kComposite)) {
      return ExprMutator::VisitExpr_(func_node);
    }
    if (auto workspace = func_node->GetAttr<Integer>(attr::kWorkspaceSize)) {
      // Append the workspace parameter to this function.
      Array<Var> new_params = func_node->params;

      auto sinfo = TensorStructInfo(ShapeExpr({Integer(max_workspace_size_)}), DataType::UInt(8));
      Var workspace_param(name_sup_->FreshName("workspace"), sinfo);

      if (func_node->GetAttr<String>(attr::kCodegen)) {
        workspace_var_param_ = workspace_param;
      }

      new_params.push_back(workspace_param);
      return Function(new_params, VisitExpr(func_node->body), func_node->ret_struct_info,
                      func_node->is_pure, func_node->attrs);
    }
    return ExprMutator::VisitExpr_(func_node);
  }

  Expr VisitExpr_(const CallNode* call_node) override {
    auto new_op = VisitExpr(call_node->op);
    if (auto var = new_op.as<Var>()) {
      if (auto callee = builder_->LookupBinding(var.value());
          callee && callee->IsInstance<FunctionNode>() &&
          Downcast<Function>(callee.value())->GetAttr<String>(attr::kComposite)) {
        // Append the workspace argument to this call. The callee should have been updated to accept
        // a workspace as the last parameter.
        auto new_args = call_node->args;
        ICHECK(workspace_var_param_.defined());
        new_args.push_back(workspace_var_param_);
        return Call(new_op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  NameSupply name_sup_;
  /*! \brief A variable that represents the workspace parameter passed from main. */
  Var workspace_var_param_;
  size_t max_workspace_size_ = 0;
};

class WorkspaceProvider : ExprMutator {
 public:
  explicit WorkspaceProvider(IRModule mod) : ExprMutator(mod), mod_(mod) {}
  using ExprMutator::VisitBindingBlock_;
  using ExprMutator::VisitExpr_;

  IRModule Run() {
    for (const auto& [gvar, f] : mod_->functions) {
      if (auto workspace = f->GetAttr<Integer>(relax::attr::kWorkspaceSize)) {
        max_workspace_size_ = std::max<size_t>(max_workspace_size_, workspace.value()->value);
      }
    }

    if (max_workspace_size_ == 0) {
      return mod_;
    }

    auto new_funcs = relax::ExternFunctionRewriter(mod_, max_workspace_size_).Run();

    for (const auto& [gvar, f] : new_funcs) {
      auto new_gvar = builder_->AddFunction(f, gvar->name_hint);
      // This is only required since the well-formed check requires kGlobalSymbol to be the same
      // as the actual name of the global variable.
      builder_->UpdateFunction(new_gvar,
                               WithAttr(f, tvm::attr::kGlobalSymbol, new_gvar->name_hint));
      gvar_map_[gvar] = new_gvar;
      builder_->GetContextIRModule()->Remove(GetRef<GlobalVar>(gvar));
    }

    for (const auto& [gvar, f] : mod_->functions) {
      workspace_var_main_ = Var();
      if (!f->IsInstance<relax::FunctionNode>() || f->GetAttr<String>(attr::kCodegen) ||
          f->GetAttr<String>(attr::kComposite)) {
        continue;
      }
      auto func = Downcast<Function>(mod_->Lookup(gvar));
      auto new_func = Function(func->params, VisitExpr(func->body), func->ret_struct_info,
                               func->is_pure, func->attrs);
      builder_->UpdateFunction(gvar, new_func);
    }
    return builder_->GetContextIRModule();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block_node) final {
    builder_->BeginDataflowBlock();
    if (!workspace_var_main_.defined()) {
      auto shape = ShapeExpr({Integer(max_workspace_size_)});
      auto ty = DataTypeImm(DataType::UInt(8));
      auto storage = MakeVMAllocStorage(shape, PrimValue::Int64(0), ty);
      auto workspace = MakeVMAllocTensor(storage, PrimValue::Int64(0), shape, ty);
      workspace_var_main_ = builder_->Emit(workspace, "workspace_main");
    }
    for (const auto& binding : block_node->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  Expr VisitExpr_(const GlobalVarNode* gvar_node) override {
    if (gvar_map_.count(gvar_node)) {
      return gvar_map_[gvar_node];
    }
    return ExprMutator::VisitExpr_(gvar_node);
  }

  Expr VisitExpr_(const CallNode* call_node) override {
    auto new_op = VisitExpr(call_node->op);

    if (auto gv = new_op.as<GlobalVar>()) {
      auto callee = builder_->GetContextIRModule()->Lookup(gv.value());
      if (callee->HasNonzeroAttr(attr::kWorkspaceSize)) {
        auto new_args = call_node->args;
        ICHECK(workspace_var_main_.defined());
        new_args.push_back(workspace_var_main_);
        return Call(new_op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      }
    }

    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  IRModule mod_;
  /*! \brief A variable that represents the workspace created at the beginning of main. */
  Var workspace_var_main_;
  size_t max_workspace_size_ = 0;
  /*! \brief A map from old global variables representing a function with workspace requirement to
   * the new ones that are transformed to take an additional workspace parameter. This is only
   * needed since the struct info of the global variables changes between transformation. */
  std::unordered_map<const GlobalVarNode*, GlobalVar> gvar_map_;
};

}  // namespace relax

namespace transform {

Pass AllocateWorkspace() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::WorkspaceProvider(m).Run(); };

  return CreateModulePass(pass_func, 0, "AllocateWorkspace", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AllocateWorkspace").set_body_typed(AllocateWorkspace);

}  // namespace transform
}  // namespace tvm
