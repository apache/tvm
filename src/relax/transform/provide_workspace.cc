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

#include <tvm/ir/name_supply.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include "../op/tensor/create.h"
#include "tvm/ir/expr.h"
#include "tvm/ir/module.h"
#include "tvm/relax/struct_info.h"
#include "utils.h"

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
      Array<Var> new_params = func_node->params;

      auto sinfo = TensorStructInfo(ShapeExpr({Integer(max_workspace_size_)}), DataType::UInt(8));
      Var workspace_param(name_sup_->FreshName("workspace"), sinfo);

      if (func_node->GetAttr<String>(attr::kCodegen)) {
        workspace_var_param_ = workspace_param;
      }

      new_params.push_back(workspace_param);
      return Function(new_params, VisitExpr(func_node->body), func_node->ret_struct_info,
                      func_node->attrs);
    }
    return ExprMutator::VisitExpr_(func_node);
  }

  Expr VisitExpr_(const CallNode* call_node) override {
    auto new_op = VisitExpr(call_node->op);
    if (auto var = new_op.as<Var>()) {
      if (auto callee = builder_->LookupBinding(var.value());
          callee && callee->IsInstance<FunctionNode>() &&
          Downcast<Function>(callee.value())->GetAttr<String>(attr::kComposite)) {
        auto new_args = call_node->args;
        new_args.push_back(workspace_var_param_);
        return Call(new_op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  NameSupply name_sup_;
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

    auto new_funcs = relax::ExternFunctionRewriter(mod_, max_workspace_size_).Run();

    for (const auto& [gvar, f] : new_funcs) {
      gvar_map_[gvar] = builder_->AddFunction(f, gvar->name_hint);
      builder_->GetContextIRModule()->Remove(GetRef<GlobalVar>(gvar));
    }

    auto gvar = mod_->GetGlobalVar("main");
    auto func = Downcast<Function>(mod_->Lookup(gvar));
    auto new_func =
        Function(func->params, VisitExpr(func->body), func->ret_struct_info, func->attrs);
    builder_->UpdateFunction(gvar, new_func);
    return builder_->GetContextIRModule();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block_node) final {
    builder_->BeginDataflowBlock();
    auto workspace = zeros(ShapeExpr({Integer(max_workspace_size_)}), DataType::UInt(8));
    workspace_var_main_ = builder_->Emit(workspace, "workspace_main");
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
        new_args.push_back(workspace_var_main_);
        return Call(new_op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      }
    }

    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  IRModule mod_;
  Var workspace_var_main_;
  size_t max_workspace_size_ = 0;
  std::unordered_map<const GlobalVarNode*, GlobalVar> gvar_map_;
};

}  // namespace relax

namespace transform {

Pass ProvideWorkspace() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::WorkspaceProvider(m).Run(); };

  return CreateModulePass(pass_func, 0, "ProvideWorkspace", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ProvideWorkspace").set_body_typed(ProvideWorkspace);

}  // namespace transform
}  // namespace tvm
