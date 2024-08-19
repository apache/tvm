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
 * \file tvm/relax/transform/inject_debug_callback.cc
 * \brief Add a callback that is called after each binding
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <vector>

namespace tvm {
namespace relax {

namespace {

class Mutator : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionNode* func) override {
    if (!func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined()) {
      return GetRef<Function>(func);
    }

    auto callback_signature = FuncStructInfo::OpaqueFunc(TupleStructInfo(Array<StructInfo>{}));
    Var debug_callback("debug_callback", callback_signature);

    Array<Var> new_params;
    new_params.push_back(debug_callback);
    for (Var param : func->params) {
      new_params.push_back(param);
    }

    auto cached = info_;
    info_ = PerFunctionInfo{debug_callback};
    auto new_body = VisitWithNewScope(func->body, new_params);

    ICHECK(info_->callback_invocations.empty());
    bool new_purity =
        Downcast<FuncStructInfo>(func->struct_info_)->purity && !info_->uses_debug_callback;
    info_ = cached;

    FuncStructInfo new_sinfo(new_params.Map(GetStructInfo), func->ret_struct_info, new_purity);

    auto new_attrs = func->attrs;
    if (auto num_input = func->attrs.GetAttr<runtime::Int>(attr::kNumInput)) {
      new_attrs =
          WithAttr(new_attrs, String(attr::kNumInput), runtime::Int(num_input.value()->value + 1));
    }

    return Function(new_params, new_body, func->ret_struct_info, new_purity, new_attrs);
  }

  void VisitBinding(const Binding& binding) override {
    ExprMutator::VisitBinding(binding);
    if (info_ && !binding->var.as<DataflowVarNode>()) {
      info_->uses_debug_callback = true;
      Expr invoke_callback =
          Call(info_->debug_callback, {relax::StringImm(binding->var->name_hint()), binding->var});
      if (builder_->CurrentBlockIsDataFlow()) {
        info_->callback_invocations.push_back(invoke_callback);
      } else {
        builder_->Emit(invoke_callback, "_");
      }
    }
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) override {
    bool made_change = false;
    Array<BindingBlock> new_blocks;

    for (const auto& block : seq_expr->blocks) {
      auto new_block = VisitBindingBlock(block);
      new_blocks.push_back(new_block);
      made_change = made_change || !new_block.same_as(block);

      if (info_ && info_->callback_invocations.size()) {
        builder_->BeginBindingBlock();
        for (Expr invoke_callback : info_->callback_invocations) {
          builder_->Emit(invoke_callback, "_");
        }
        new_blocks.push_back(builder_->EndBlock());
        info_->callback_invocations.clear();
        made_change = true;
      }
    }

    Expr new_body = VisitExpr(seq_expr->body);
    made_change = made_change || !new_body.same_as(seq_expr->body);

    if (made_change) {
      return SeqExpr(new_blocks, new_body);
    } else {
      return GetRef<SeqExpr>(seq_expr);
    }
  }

 private:
  struct PerFunctionInfo {
    Var debug_callback;
    std::vector<Expr> callback_invocations;
    bool uses_debug_callback = false;
  };
  std::optional<PerFunctionInfo> info_;
};

}  // namespace

namespace transform {
Pass InjectDebugCallback() {
  auto pass_func = [=](Function func, IRModule, PassContext) -> Function {
    return Downcast<Function>(Mutator()(func));
  };
  return CreateFunctionPass(pass_func, 0, "InjectDebugCallback", {});
}

TVM_REGISTER_GLOBAL("relax.transform.InjectDebugCallback").set_body_typed(InjectDebugCallback);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
