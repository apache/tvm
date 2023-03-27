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
 * \file tvm/relax/transform/eliminate_common_subexpr.cc
 * \brief Eliminrate common subexpression pass.
 *
 * Currently it removes common subexpressions within a DataflowBlock.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class SubexprCounter : public ExprVisitor {
 public:
  // overriding VisitExpr ensures we do this for every subexpression
  void VisitExpr(const Expr& e) override {
    // Cases we ignore because we will not substitute them:
    // 1. Vars of all kinds
    // 2. Op nodes (nothing we can do)
    // 3. Scalar constants (not much benefit from binding to a var)
    if (!(e->IsInstance<VarNode>() || e->IsInstance<DataflowVarNode>() ||
          e->IsInstance<GlobalVarNode>() || e->IsInstance<tvm::OpNode>() ||
          (e.as<ConstantNode>() && (e.as<ConstantNode>()->is_scalar())))) {
      int count = 0;
      if (count_map_.count(e)) {
        count = count_map_.at(e);
      }
      count_map_[e] = count + 1;
    }
    ExprVisitor::VisitExpr(e);
  }

  // do not visit inner functions: we will do CSE within those
  void VisitExpr_(const FunctionNode* func) override {}

  // we are not going to do replacements inside struct info to avoid binding lots of reused shapes
  void VisitExprDepStructInfoField(const StructInfo& struct_info) override {}

  std::unordered_map<Expr, int, StructuralHash, StructuralEqual> Count(
      const DataflowBlock& df_block) {
    for (auto binding : df_block->bindings) {
      VisitBinding(binding);
    }
    return count_map_;
  }

 private:
  std::unordered_map<Expr, int, StructuralHash, StructuralEqual> count_map_;
};

// forward declaration
DataflowBlock EliminateCommonSubexpr(const DataflowBlock&);

class CommonSubexprEliminator : public ExprMutator {
 public:
  explicit CommonSubexprEliminator(
      const std::unordered_map<Expr, int, StructuralHash, StructuralEqual>& count_map)
      : count_map_(count_map) {}

  // overriding here ensures we visit every subexpression
  Expr VisitExpr(const Expr& e) override {
    if (count_map_.count(e) && count_map_.at(e) > 1) {
      // if we already have a mapping for it, get it
      if (replacements_.count(e)) {
        return replacements_.at(e);
      }
      // Otherwise, insert a new binding for the current expression.
      // Visit before emitting to do inner replacements
      Expr new_e = ExprMutator::VisitExpr(e);
      Var v = builder_->Emit(new_e);
      replacements_[e] = v;
      return v;
    }
    return ExprMutator::VisitExpr(e);
  }

  // we are not going to do replacements inside struct info to avoid binding lots of reused shapes
  StructInfo VisitExprDepStructInfoField(const StructInfo& struct_info) override {
    return struct_info;
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    // for an inner function, we will do CSE on its body
    Expr new_body = ExprMutator::VisitExpr(func->body);
    if (new_body.same_as(func->body)) {
      return GetRef<Expr>(func);
    }
    return Function(func->params, new_body, func->ret_struct_info, func->attrs, func->span);
  }

  // this should happen only for the inner function case
  Expr VisitExpr_(const SeqExprNode* seq) override {
    bool all_unchanged = true;
    Array<BindingBlock> new_blocks;
    // apply CSE within dataflow blocks only
    for (auto block : seq->blocks) {
      if (const DataflowBlockNode* df_block = block.as<DataflowBlockNode>()) {
        auto new_df_block = EliminateCommonSubexpr(GetRef<DataflowBlock>(df_block));
        if (!new_df_block.same_as(block)) {
          new_blocks.push_back(new_df_block);
          all_unchanged = false;
          continue;
        }
      }
      new_blocks.push_back(block);
    }

    if (all_unchanged) {
      return GetRef<Expr>(seq);
    }
    // do not visit the body
    return SeqExpr(new_blocks, seq->body, seq->span);
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    // no need to visit var def because the struct info isn't going to change
    Expr new_value = RegisterBoundValue(binding->var, binding->value);

    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      // no need to renormalize new_value because all replacements are with vars
      builder_->EmitNormalized(VarBinding(binding->var, new_value, binding->span));
    }
  }

  void VisitBinding_(const MatchCastNode* binding) override {
    // no need to visit var def because the struct info isn't going to change
    Expr new_value = RegisterBoundValue(binding->var, binding->value);

    // re-emit old binding if nothing changes
    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(GetRef<MatchCast>(binding));
    } else {
      // no need to renormalize new_value because all replacements are with vars
      builder_->EmitNormalized(
          MatchCast(binding->var, new_value, binding->struct_info, binding->span));
    }
  }

 private:
  Expr RegisterBoundValue(Var var, Expr bound_value) {
    // special case: if we are processing a binding
    // and this is the first time we've encountered it,
    // we will use the binding's var for the mapping
    bool newly_replaced = false;
    if (count_map_.count(bound_value) && count_map_.at(bound_value) > 1 &&
        !replacements_.count(bound_value)) {
      replacements_[bound_value] = var;
      newly_replaced = true;
    }

    if (newly_replaced) {
      // If we've just added the mapping, using the overridden visitor will
      // just return the var, which we don't want, so we will use
      // the superclass VisitExpr to do inner substitutions
      return ExprMutator::VisitExpr(bound_value);
    }
    return VisitExpr(bound_value);
  }

  const std::unordered_map<Expr, int, StructuralHash, StructuralEqual>& count_map_;
  std::unordered_map<Expr, Var, StructuralHash, StructuralEqual> replacements_;
};

DataflowBlock EliminateCommonSubexpr(const DataflowBlock& df_block) {
  SubexprCounter counter;
  auto count_map = counter.Count(df_block);
  CommonSubexprEliminator eliminator(count_map);
  return Downcast<DataflowBlock>(eliminator.VisitBindingBlock(df_block));
}

namespace transform {

Pass EliminateCommonSubexpr() {
  runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func =
      [=](DataflowBlock df_block, IRModule m, PassContext pc) {
        return Downcast<DataflowBlock>(EliminateCommonSubexpr(df_block));
      };
  return CreateDataflowBlockPass(pass_func, 1, "EliminateCommonSubexpr", {});
}

TVM_REGISTER_GLOBAL("relax.transform.EliminateCommonSubexpr")
    .set_body_typed(EliminateCommonSubexpr);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
