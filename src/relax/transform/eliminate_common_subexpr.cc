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
 * \brief Eliminate common subexpression pass.
 *
 * Currently it removes common subexpressions within a Function.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

namespace tvm {
namespace relax {

class CommonSubexprEliminator : public ExprMutator {
 public:
  explicit CommonSubexprEliminator(bool call_only = false) : call_only_(call_only) {}

  void VisitBinding(const Binding& binding) override {
    Expr bound_value = VisitExpr(GetBoundValue(binding));

    auto it = var_replacements_.find(bound_value);

    if (call_only_ && !bound_value->IsInstance<relax::CallNode>()) {
      // This type may not be eliminated, so we maintain the new
      // binding.
      ExprMutator::VisitBinding(binding);

    } else if (ContainsImpureCall(bound_value)) {
      // This expression is impure, and must be retained.
      ExprMutator::VisitBinding(binding);
    } else if (it == var_replacements_.end()) {
      // This expression could be de-duplicated, but it is the first
      // time we've seen this expression.  Remember it in case we see
      // it again.
      var_replacements_.insert({bound_value, binding->var});
      ExprMutator::VisitBinding(binding);

    } else if (!StructuralEqual()(GetStructInfo(binding->var), GetStructInfo(it->second))) {
      // We've seen this expression before, but it is bound to a
      // different struct info this time.  This is a MatchCast node,
      // and the struct info may be required for downstream usage, or to
      // define symbolic variables.
      ExprMutator::VisitBinding(binding);

    } else {
      // We've seen this expression before, and can re-use the first occurrence.
      var_remap_.insert({binding->var->vid, it->second});
    }
  }

  Expr VisitExpr_(const FunctionNode* op) override {
    // If we have accumulated any state, visit the function in a fresh
    // copy of the mutator, to avoid replacing a child-scope
    // expression with a parent-scope binding, or vice versa.
    if (var_replacements_.size() || var_remap_.size()) {
      return VisitWithCleanScope(GetRef<Expr>(op));
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  Expr VisitExpr_(const IfNode* op) override {
    Expr cond = VisitExpr(op->cond);
    Expr true_branch = VisitWithCleanScope(op->true_branch);
    Expr false_branch = VisitWithCleanScope(op->false_branch);
    if (op->cond.same_as(cond) && op->true_branch.same_as(true_branch) &&
        op->false_branch.same_as(false_branch) &&
        VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
      return GetRef<Expr>(op);
    } else {
      return If(cond, true_branch, false_branch, op->span);
    }
  }

 private:
  Expr VisitWithCleanScope(Expr expr) {
    CommonSubexprEliminator clean_mutator(call_only_);
    return clean_mutator.VisitExpr(expr);
  }

  bool call_only_{false};

  std::unordered_map<Expr, Var, StructuralHash, StructuralEqual> var_replacements_;
};

Expr EliminateCommonSubexpr(const Expr& expr, bool call_only) {
  CommonSubexprEliminator mutator(call_only);
  return mutator(expr);
}

namespace transform {

Pass EliminateCommonSubexpr(bool call_only) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function func, IRModule m, PassContext pc) {
        return Downcast<Function>(EliminateCommonSubexpr(func, call_only));
      };
  return CreateFunctionPass(pass_func, 1, "EliminateCommonSubexpr", {});
}

TVM_REGISTER_GLOBAL("relax.transform.EliminateCommonSubexpr")
    .set_body_typed(EliminateCommonSubexpr);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
