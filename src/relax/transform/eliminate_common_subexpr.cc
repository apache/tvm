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

#include "../../support/utils.h"

namespace tvm {
namespace relax {
namespace {
/* \brief Lookup key for subexpression replacements
 *
 * The lookup key must contain the expression being bound, along with
 * the struct info used for a match cast, if applicable.  Using
 * `MatchCast` with StructuralEqual and StructuralHash would be almost
 * correct, but acts as a point of definition for symbolic variables
 * within the output struct info.  As a result, it would erroneously
 * de-duplicate `R.match_cast(A, R.Tensor([m,n]))` and
 * `R.match_cast(A, R.Tensor([p,q]))`, even though they define
 * different symbolic variables.
 */
struct ReplacementKey {
  tvm::relax::Expr bound_value;
  tvm::Optional<tvm::relax::StructInfo> match_cast = tvm::NullOpt;

  explicit ReplacementKey(const tvm::relax::Binding& binding)
      : bound_value(GetBoundValue(binding)) {
    if (const auto* ptr = binding.as<tvm::relax::MatchCastNode>()) {
      match_cast = ptr->struct_info;
    }
  }

  friend bool operator==(const ReplacementKey& a, const ReplacementKey& b) {
    tvm::StructuralEqual eq;
    return eq(a.bound_value, b.bound_value) && eq(a.match_cast, b.match_cast);
  }
};

}  // namespace
}  // namespace relax
}  // namespace tvm

/* \brief Definition of std::hash<ReplacementKey>
 *
 * Specialization of std::hash must occur outside of tvm::relax
 * namespace, and before its usage in the constructor of
 * `CommonSubexprEliminator`.
 */
template <>
struct std::hash<tvm::relax::ReplacementKey> {
  std::size_t operator()(const tvm::relax::ReplacementKey& key) const {
    tvm::StructuralHash hasher;
    return tvm::support::HashCombine(hasher(key.bound_value), hasher(key.match_cast));
  }
};

namespace tvm {
namespace relax {

namespace {

class CommonSubexprEliminator : public ExprMutator {
 public:
  explicit CommonSubexprEliminator(bool call_only = false) : call_only_(call_only) {}

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) override {
    auto cache_vars = var_remap_;
    auto output = ExprMutator::VisitBindingBlock_(block);

    for (auto& [key, replacements] : expr_replacements_) {
      replacements.erase(
          std::remove_if(replacements.begin(), replacements.end(),
                         [](const Var& var) -> bool { return var->IsInstance<DataflowVarNode>(); }),
          replacements.end());
    }

    var_remap_ = cache_vars;
    return output;
  }

  void VisitBinding(const Binding& binding) override {
    Expr bound_value = VisitExpr(GetBoundValue(binding));

    Binding output_binding = [&]() -> Binding {
      if (binding.as<VarBindingNode>()) {
        return VarBinding(binding->var, bound_value);
      } else if (auto match_cast = binding.as<MatchCastNode>()) {
        return MatchCast(binding->var, bound_value, match_cast->struct_info);
      } else {
        LOG(FATAL) << "Binding must be either VarBinding or MatchCast, "
                   << "but was " << binding->GetTypeKey();
      }
    }();

    ReplacementKey lookup_key(output_binding);

    if (call_only_ && !bound_value->IsInstance<relax::CallNode>()) {
      VLOG(1) << "Since call_only_ is true, it is forbidden to de-duplicate " << bound_value;

    } else if (ContainsImpureCall(bound_value)) {
      VLOG(1) << "Since the expression is impure, cannot de-duplicate " << bound_value;

    } else if (auto it = expr_replacements_.find(lookup_key);
               it != expr_replacements_.end() && it->second.size()) {
      VLOG(1) << "Value " << bound_value << " has previously been bound as " << it->second[0]
              << ".  The duplicate binding of this value to " << binding->var
              << " will be replaced with a trivial binding, "
              << "and occurrences of " << binding->var << " will be replaced with "
              << it->second[0];
      output_binding = VarBinding(binding->var, it->second[0]);
      var_remap_.insert({binding->var->vid, it->second[0]});
      it->second.push_back(binding->var);

    } else {
      VLOG(1) << "Value " << bound_value << " is bound to " << binding->var
              << " and may be de-duplicated if it occurs again.";

      expr_replacements_[lookup_key].push_back(binding->var);
    }

    builder_->EmitNormalized(output_binding);
  }

  Expr VisitExpr_(const FunctionNode* op) override {
    // If we have accumulated any state, visit the function in a fresh
    // copy of the mutator, to avoid replacing a child-scope
    // expression with a parent-scope binding, or vice versa.
    if (expr_replacements_.size() || var_remap_.size()) {
      return VisitWithCleanScope(GetRef<Expr>(op));
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  Expr VisitExpr_(const IfNode* op) override {
    Expr cond = VisitExpr(op->cond);
    Expr true_branch = VisitWithInnerScope(op->true_branch);
    Expr false_branch = VisitWithInnerScope(op->false_branch);
    if (op->cond.same_as(cond) && op->true_branch.same_as(true_branch) &&
        op->false_branch.same_as(false_branch) &&
        VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
      return GetRef<Expr>(op);
    } else {
      return If(cond, true_branch, false_branch, op->span);
    }
  }

 private:
  Expr VisitWithInnerScope(Expr expr) {
    auto cached_vars = var_remap_;
    auto cached_exprs = expr_replacements_;
    auto output = VisitExpr(expr);
    var_remap_ = cached_vars;
    expr_replacements_ = cached_exprs;
    return output;
  }

  Expr VisitWithCleanScope(Expr expr) {
    CommonSubexprEliminator clean_mutator(call_only_);
    return clean_mutator.VisitExpr(expr);
  }

  bool call_only_{false};
  std::unordered_map<ReplacementKey, std::vector<Var>> expr_replacements_;
};

}  // namespace

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
