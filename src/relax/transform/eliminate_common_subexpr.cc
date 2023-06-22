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
 * Currently it removes common subexpressions within a Function.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

namespace tvm {
namespace relax {

// Checks if a given expression contains an impure subexpression
// Caches the results of checks to avoid revisiting subexpressions
class ImpurityDetector : public ExprVisitor {
 public:
  bool Detect(const Expr& expr) {
    impure_found_ = false;
    VisitExpr(expr);
    return impure_found_;
  }

  void VisitExpr(const Expr& expr) {
    // already checked: do not revisit
    if (purity_map_.count(expr)) {
      impure_found_ = impure_found_ || !purity_map_.at(expr);
      return;
    }

    // in principle, we could stop checking once we find an impurity,
    // but not doing so lets us fully populate the cache

    // store the previous state so we could assess the purity of this subexpression alone
    bool prev_state = impure_found_;
    impure_found_ = false;
    ExprVisitor::VisitExpr(expr);
    // if impure_found_ remains false, then the expression is pure
    purity_map_[expr] = !impure_found_;
    impure_found_ = prev_state || impure_found_;
  }

  void VisitExpr_(const CallNode* call) {
    // the only possible impurities can come from call nodes
    bool is_impure = IsImpureCall(GetRef<Call>(call));
    impure_found_ = impure_found_ || is_impure;
    ExprVisitor::VisitExpr_(call);
  }

 private:
  bool impure_found_ = false;
  std::unordered_map<Expr, bool, StructuralHash, StructuralEqual> purity_map_;
};

class SubexprCounter : public ExprVisitor {
 public:
  // overriding VisitExpr ensures we do this for every subexpression
  void VisitExpr(const Expr& e) override {
    // Cases we ignore because we will not substitute them:
    // 1. Vars of all kinds
    // 2. Op nodes (nothing we can do)
    // 3. PrimValue nodes (not much benefit from binding to a var)
    // 4. StringImm nodes (not much benefit from binding to a var)
    // 5. Scalar constants (not much benefit from binding to a var)
    if (!(e->IsInstance<VarNode>() || e->IsInstance<DataflowVarNode>() ||
          e->IsInstance<GlobalVarNode>() || e->IsInstance<tvm::OpNode>() ||
          e->IsInstance<PrimValueNode>() || e->IsInstance<StringImmNode>() ||
          (e.as<ConstantNode>() && (e.as<ConstantNode>()->is_scalar())))) {
      // also if e has an impure subexpression, we will not deduplicate it
      if (!impurity_detector_.Detect(e)) {
        int count = 0;
        if (count_map_.count(e)) {
          count = count_map_.at(e);
        }
        count_map_[e] = count + 1;
      }
    }
    ExprVisitor::VisitExpr(e);
  }

  // do not visit inner functions: we will do CSE within those
  void VisitExpr_(const FunctionNode* func) override {}

  // we are not going to do replacements inside struct info to avoid binding lots of reused shapes
  void VisitExprDepStructInfoField(const StructInfo& struct_info) override {}

  std::unordered_map<Expr, int, StructuralHash, StructuralEqual> Count(const Function& func) {
    VisitExpr(func->body);
    return count_map_;
  }

 private:
  std::unordered_map<Expr, int, StructuralHash, StructuralEqual> count_map_;
  ImpurityDetector impurity_detector_;
};

// forward declaration
Function EliminateCommonSubexpr(const Function&, bool call_only);

class CommonSubexprEliminator : public ExprMutator {
 public:
  explicit CommonSubexprEliminator(
      const std::unordered_map<Expr, int, StructuralHash, StructuralEqual>& count_map,
      bool call_only = false)
      : count_map_(count_map), call_only_(call_only) {}

  // overriding here ensures we visit every subexpression
  Expr VisitExpr(const Expr& e) override {
    if (call_only_ && !e->IsInstance<CallNode>()) {
      return ExprMutator::VisitExpr(e);
    }
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
    // do full CSE within the function
    return EliminateCommonSubexpr(GetRef<Function>(func), call_only_);
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
  bool call_only_{false};
};

Function EliminateCommonSubexpr(const Function& func, bool call_only) {
  SubexprCounter counter;
  auto count_map = counter.Count(func);
  CommonSubexprEliminator eliminator(count_map, call_only);
  return Function(func->params, eliminator.VisitExpr(func->body), func->ret_struct_info,
                  func->is_pure, func->attrs, func->span);
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
