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
 * \file tvm/relax/transform/normalize.cc
 * \brief Pass for transforming Relax IR to normal form, i.e., the expressions are normalized(no
 * nesting and hence the AST is in ANF), and all checked_type_ and shape_ of expressions are
 * available.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

// TODO(@altanh): LCA binding lifting
class NormalizeMutator : public ExprMutatorBase {
 public:
  NormalizeMutator() { builder_ = BlockBuilder::Create(NullOpt); }

  Expr VisitExpr(const Expr& expr) override {
    return builder_->Normalize(ExprMutatorBase::VisitExpr(expr));
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    Expr body = this->VisitWithNewScope(op->body, op->params);

    if (body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      return Function(op->params, body, op->ret_struct_info, op->is_pure, op->attrs);
    }
  }

  Expr VisitExpr_(const IfNode* op) final {
    Expr guard = this->VisitExpr(op->cond);
    Expr true_b = this->VisitWithNewScope(op->true_branch);
    Expr false_b = this->VisitWithNewScope(op->false_branch);
    if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
        op->false_branch.same_as(false_b)) {
      return GetRef<Expr>(op);
    } else {
      return If(guard, true_b, false_b, op->span);
    }
  }

  Expr VisitWithNewScope(const Expr& expr, Optional<Array<Var>> params = NullOpt) {
    builder_->BeginBindingBlock();
    builder_->BeginScope(params);
    Expr ret = this->VisitExpr(expr);
    BindingBlock prologue = builder_->EndBlock();
    if (!prologue->bindings.empty()) {
      ret = SeqExpr({prologue}, ret);
    }
    builder_->EndScope();
    return ret;
  }

  Expr VisitExpr_(const SeqExprNode* op) final {
    bool all_blocks_unchanged = true;
    Array<BindingBlock> blocks;
    for (auto block : op->blocks) {
      BindingBlock new_block = this->VisitBindingBlock(block);
      if (!new_block->bindings.empty()) {
        blocks.push_back(new_block);
      }
      all_blocks_unchanged &= block.same_as(new_block);
    }

    builder_->BeginBindingBlock();
    Expr body = this->VisitExpr(op->body);
    BindingBlock prologue = builder_->EndBlock();
    if (!prologue->bindings.empty()) {
      blocks.push_back(prologue);
      all_blocks_unchanged = false;
    }

    if (all_blocks_unchanged && body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      return SeqExpr(blocks, body);
    }
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) final {
    BindingBlock ret;
    if (const auto* node = block.as<DataflowBlockNode>()) {
      ret = VisitBindingBlock_(node);
    } else if (const auto* node = block.as<BindingBlockNode>()) {
      ret = VisitBindingBlock_(node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
    }
    return ret;
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) {
    builder_->BeginDataflowBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  void VisitBinding(const Binding& binding) {
    if (const auto* node = binding.as<VarBindingNode>()) {
      VisitBinding_(node);
    } else if (const auto* node = binding.as<MatchCastNode>()) {
      VisitBinding_(node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
    }
  }

  void VisitBinding_(const VarBindingNode* binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (!binding->var->struct_info_.defined()) {
      UpdateStructInfo(binding->var, GetStructInfo(new_value));
    }

    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      builder_->EmitNormalized(VarBinding(binding->var, new_value));
    }
  }

  void VisitBinding_(const MatchCastNode* binding) {
    Expr new_value = this->VisitExpr(binding->value);

    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(GetRef<MatchCast>(binding));
    } else {
      builder_->EmitNormalized(
          MatchCast(binding->var, builder_->NormalizeArgument(new_value), binding->struct_info));
    }
  }

 private:
  /*! \brief Internal block builder to emit bindings during rewriting. */
  BlockBuilder builder_;
};  // namespace relax

Expr Normalize(const Expr& e) { return NormalizeMutator().VisitExpr(e); }

namespace transform {

Pass Normalize() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(Normalize(f)); };
  return CreateFunctionPass(pass_func, 1, "Normalize", {});
}

TVM_REGISTER_GLOBAL("relax.transform.Normalize").set_body_typed(Normalize);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
