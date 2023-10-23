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
 * \file src/relax/transform/canonicalize_bindings.cc
 * \brief Pass for simplifying modules by folding var bindings and match shape nodes.
 *        May include other forms of simplification in the future.
 *        Ideally should be used before constant folding and eliminating unused bindings.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class BindingCanonicalizer : public ExprMutator {
 public:
  BindingCanonicalizer() {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const TupleGetItemNode* tuple_get_item) override {
    if (auto tuple_var = tuple_get_item->tuple.as<Var>()) {
      if (auto tuple_value = LookupBinding(tuple_var.value())) {
        if (auto explicit_tuple = tuple_value.as<TupleNode>()) {
          CHECK_GE(tuple_get_item->index, 0)
              << "Tuple " << tuple_value << " is accessed at index " << tuple_get_item->index
              << ", but negative indices are not supported in this context.";
          CHECK_LT(tuple_get_item->index, explicit_tuple->fields.size())
              << "Tuple " << tuple_value << " is accessed at index " << tuple_get_item->index
              << ", but the tuple size is only " << explicit_tuple->fields.size();
          return VisitExpr(explicit_tuple->fields[tuple_get_item->index]);
        }
      }
    }
    return ExprMutator::VisitExpr_(tuple_get_item);
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    // Unlike default visitor, we do not permit the struct info to change
    // if the new value's struct info is different (this preserves user annotations)
    Expr new_value = this->VisitExpr(binding->value);
    Var new_var = this->VisitVarDef(binding->var);

    if (auto opt_var = new_value.as<Var>();
        opt_var && CanCanonicalizeVar(new_var, opt_var.value())) {
      var_remap_[new_var->vid] = opt_var.value();
    } else if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
      this->builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      this->builder_->EmitNormalized(VarBinding(new_var, new_value));
    }
  }

  void VisitBinding_(const MatchCastNode* binding) override {
    // If we have a trivial shape check (the struct_info_ of LHS and RHS is the same),
    // we can canonicalize to a var binding
    Expr new_value = this->VisitExpr(binding->value);
    bool has_same_struct_info = StructuralEqual()(binding->struct_info, GetStructInfo(new_value));

    if (has_same_struct_info) {
      if (auto parent = new_value.as<Var>();
          parent && CanCanonicalizeVar(binding->var, parent.value())) {
        // LHS and RHS have the same struct info, and occur in a
        // context where the RHS can replace the LHS.
        var_remap_[binding->var->vid] = parent.value();
      } else {
        // LHS and RHS have the same struct info, but the RHS is not a
        // drop-in replacement for the LHS.
        builder_->EmitNormalized(VarBinding(binding->var, new_value));
      }
    } else if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(GetRef<MatchCast>(binding));
    } else {
      // we can't elide in the same way as with var bindings because
      // the struct info comparison has semantics
      builder_->EmitNormalized(MatchCast(binding->var, new_value, binding->struct_info));
    }
  }

  // Special case: for dataflow blocks, we will check for dataflow vars that solely exist
  // to be bound to the output. In this case, we will get rid of those bindings and
  // use the dataflow var's definition directly
  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) override {
    auto new_block = Downcast<DataflowBlock>(ExprMutator::VisitBindingBlock_(block));
    std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> disqualified_set;
    std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> output_vars;

    std::unordered_map<DataflowVar, Expr, ObjectPtrHash, ObjectPtrEqual> candidates;
    for (int i = new_block->bindings.size() - 1; i >= 0; i--) {
      auto binding = new_block->bindings[i];
      auto var = binding->var;
      auto value = GetBoundValue(binding);

      if (var->IsInstance<DataflowVarNode>()) {
        auto df_var = Downcast<DataflowVar>(var);

        // disqualify any vars that appear in the RHS
        // (for a function literal, consider only free vars)
        Array<Var> rhs_vars;
        if (!value->IsInstance<FunctionNode>()) {
          rhs_vars = FreeVars(value);
        } else {
          rhs_vars = AllVars(value);
        }

        for (auto rhs_var : rhs_vars) {
          if (rhs_var->IsInstance<DataflowVarNode>()) {
            disqualified_set.insert(Downcast<DataflowVar>(rhs_var));
          }
        }

        // if the current var is an output and has not been disqualified,
        // then include it in the candidate map
        if (!disqualified_set.count(df_var) && output_vars.count(df_var)) {
          candidates[df_var] = value;
        }
      } else {
        // The LHS is an output binding.
        // We are looking for cases where the RHS is a single dataflow var;
        // disqualify if the RHS is not a single dataflow var
        // or if the var has been output before
        if (const auto* rhs_var = value.as<DataflowVarNode>()) {
          if (output_vars.count(GetRef<DataflowVar>(rhs_var))) {
            disqualified_set.insert(GetRef<DataflowVar>(rhs_var));
          }
          output_vars.insert(GetRef<DataflowVar>(rhs_var));
        } else {
          Array<Var> disqualified;
          // for function literal, consider only free vars
          if (value->IsInstance<FunctionNode>()) {
            disqualified = FreeVars(value);
          } else {
            disqualified = AllVars(value);
          }

          for (auto rhs_var : disqualified) {
            if (rhs_var->IsInstance<DataflowVarNode>()) {
              disqualified_set.insert(Downcast<DataflowVar>(rhs_var));
            }
          }
        }
      }
    }

    // second pass: for each binding where the LHS is a candidate, remove the binding.
    // If the RHS is a candidate, replace it with the definition
    Array<Binding> new_bindings;
    bool changed = false;
    for (auto binding : new_block->bindings) {
      if (binding->var->IsInstance<DataflowVarNode>() &&
          candidates.count(Downcast<DataflowVar>(binding->var))) {
        changed = true;
        continue;
      } else if (!binding->var->IsInstance<DataflowVarNode>() &&
                 GetBoundValue(binding)->IsInstance<DataflowVarNode>() &&
                 candidates.count(Downcast<DataflowVar>(GetBoundValue(binding)))) {
        changed = true;
        if (auto* match_binding = binding.as<MatchCastNode>()) {
          auto new_binding =
              MatchCast(binding->var, candidates.at(Downcast<DataflowVar>(match_binding->value)),
                        match_binding->struct_info);
          new_bindings.push_back(new_binding);
        } else if (auto* var_binding = binding.as<VarBindingNode>()) {
          auto new_binding =
              VarBinding(binding->var, candidates.at(Downcast<DataflowVar>(var_binding->value)));
          new_bindings.push_back(new_binding);
        } else {
          CHECK(false) << "Invalid binding";  // never happens
        }
      } else {
        new_bindings.push_back(binding);
      }
    }

    if (!changed) {
      return new_block;
    }
    return DataflowBlock(new_bindings);
  }

 private:
  bool AnnotationsDiffer(const ObjectRef& obj1, const ObjectRef& obj2,
                         std::function<bool(const ObjectRef&, const ObjectRef&)> check_eq) {
    // annotations differ if one is present but not the other
    // or they're both present and they differ
    bool both_present = obj1.defined() && obj2.defined();
    bool neither_present = !obj1.defined() && !obj2.defined();
    return !(both_present || neither_present) || (both_present && !check_eq(obj1, obj2));
  }

  bool CanCanonicalizeVar(Var var, Var parent_var) {
    // Cases when we conservatively do not unify:
    // 1. The struct_info_ of the child differs from that of the parent
    //    In this case, we could be overriding user annotations.
    // 2. If the child is a Var and the parent is a DataflowVar.
    //    That could result in a DataflowVar leaving the current DataflowBlock.
    bool annotations_differ = AnnotationsDiffer(var->struct_info_, parent_var->struct_info_,
                                                [&](const ObjectRef& lhs, const ObjectRef& rhs) {
                                                  return tvm::StructuralEqual()(lhs, rhs);
                                                });
    bool var_to_dataflow = (!var.as<DataflowVarNode>() && parent_var.as<DataflowVarNode>());
    return !annotations_differ && !var_to_dataflow;
  }
};

Expr CanonicalizeBindings(const Expr& e) { return BindingCanonicalizer().VisitExpr(e); }

namespace transform {

Pass CanonicalizeBindings() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CanonicalizeBindings(f));
      };
  return CreateFunctionPass(pass_func, 1, "CanonicalizeBindings", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CanonicalizeBindings").set_body_typed(CanonicalizeBindings);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
