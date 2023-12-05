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

namespace {

struct CanonicalizationPlan {
  Map<Id, Var> replace_usage;
  Map<Id, Var> replace_binding;
  std::unordered_set<Id, ObjectPtrHash, ObjectPtrEqual> bindings_to_remove;
};

/*! \brief Utility class to identify usage location
 *
 * Canonicalization of a variable binding may require information from
 * later in the function.  For example, replacing `dataflow_x = expr`
 * with `var_x = expr` to avoid a trivial binding of `var_x =
 * dataflow_x` later in the function.  This utility examines a relax
 * expression, and plans the changes to be made in a mutation pass.
 */
class CanonicalizePlanner : public ExprVisitor {
 public:
  static CanonicalizationPlan Collect(const Expr& expr) {
    CanonicalizePlanner visitor;
    visitor.VisitExpr(expr);

    CanonicalizationPlan plan;

    // If a Var has been defined inside a DataflowBlock, is only used
    // within a DataflowBlock, and is not already handled by removal
    // of trivial bindings, then we can replace it with a DataflowVar.
    for (auto var : visitor.defined_inside_dataflow_) {
      if (!var.as<DataflowVarNode>() && !visitor.used_outside_home_dataflow_.count(var)) {
        DataflowVar new_var(var->name_hint(), GetStructInfo(var));

        plan.replace_binding.Set(var->vid, new_var);
        plan.replace_usage.Set(var->vid, new_var);
      }
    }

    for (const auto& binding_iter : visitor.trivial_bindings_) {
      Var bound_var = binding_iter.first;
      Var bound_to = binding_iter.second;

      while (auto opt = visitor.trivial_bindings_.Get(bound_to)) {
        // This may be a trivial binding into a trivial binding.  In
        // that case, unwrap the bindings until we find the earliest
        // non-trivial binding.
        bound_to = opt.value();
      }
      while (auto opt = plan.replace_binding.Get(bound_to->vid)) {
        // The variable we are binding to may have already been
        // replaced, if it fell into Case 4 (Var = DataflowVar).  In
        // that case, we check against its replacement instead.
        bound_to = opt.value();
      }

      if (bound_var.as<DataflowVarNode>() || !bound_to.as<DataflowVarNode>()) {
        // Case 1: Var = Var
        // Case 2: DataflowVar = Var
        // Case 3: DataflowVar = DataflowVar
        //
        // For these three cases, the trivial binding can be
        // unwrapped, using the bound variable directly at the point
        // of use.
        plan.replace_usage.Set(bound_var->vid, bound_to);
        plan.bindings_to_remove.insert(bound_var->vid);
      } else {
        // Case 4: Var = DataflowVar
        //
        // Replacing a Var with a DataflowVar could result in illegal
        // use of a DataflowVar outside of a DataflowBlock.  Instead,
        // we replace in the opposite direction, replacing the binding
        // of the DataflowVar with a binding of the Var.
        plan.replace_binding.Set(bound_to->vid, bound_var);
        plan.replace_usage.Set(bound_to->vid, bound_var);
        plan.bindings_to_remove.insert(bound_var->vid);
      }
    }

    return plan;
  }

 private:
  void VisitExpr_(const FunctionNode* func) override {
    // for functions, treat any free vars as used outside their home DF block
    auto cache = current_block_;
    current_block_ = Optional<BindingBlock>();
    auto free_vars = FreeVars(GetRef<Function>(func));
    for (auto var : free_vars) {
      used_outside_home_dataflow_.insert(var);
    }
    ExprVisitor::VisitExpr_(func);
    current_block_ = cache;
  }

  void VisitExpr_(const SeqExprNode* seq) override {
    // need to reset current_block_ for nested seq exprs (such as in If nodes)
    auto cache = current_block_;
    current_block_ = Optional<BindingBlock>();
    ExprVisitor::VisitExpr_(seq);
    current_block_ = cache;
  }

  void VisitBindingBlock_(const BindingBlockNode* block) override {
    CHECK(!current_block_.defined()) << "Forgetting to unset current block";
    current_block_ = GetRef<BindingBlock>(block);
    ExprVisitor::VisitBindingBlock_(block);
    current_block_ = Optional<BindingBlock>();
  }

  void VisitBindingBlock_(const DataflowBlockNode* block) override {
    CHECK(!current_block_.defined()) << "Forgetting to unset current block";
    current_block_ = GetRef<DataflowBlock>(block);
    ExprVisitor::VisitBindingBlock_(block);
    current_block_ = Optional<BindingBlock>();
  }

  void VisitBinding(const Binding& binding) override {
    bool has_same_struct_info = true;
    Expr value;
    if (auto ptr = binding.as<VarBindingNode>()) {
      value = ptr->value;
    } else if (auto ptr = binding.as<MatchCastNode>()) {
      has_same_struct_info =
          StructuralEqual()(GetStructInfo(binding->var), GetStructInfo(ptr->value));
      value = ptr->value;
    } else {
      LOG(FATAL) << "Invalid binding type: " << binding->GetTypeKey();
    }

    // Unwrap TupleGetItem, if the Tuple being accessed is known.
    if (auto tuple_get_item = value.as<TupleGetItemNode>()) {
      Expr tuple = tuple_get_item->tuple;
      while (auto tuple_var = tuple.as<Var>()) {
        if (auto opt = known_bindings_.Get(tuple_var.value())) {
          tuple = opt.value();
        } else {
          break;
        }
      }

      if (auto ptr = tuple.as<TupleNode>()) {
        value = ptr->fields[tuple_get_item->index];
      }
    }

    if (auto parent = value.as<Var>(); parent && has_same_struct_info) {
      trivial_bindings_.Set(binding->var, parent.value());
    }

    known_bindings_.Set(binding->var, value);
    def_blocks_.Set(binding->var, current_block_.value());

    ExprVisitor::VisitBinding(binding);
  }

  void VisitVarDef(const Var& var) override {
    if (inside_dataflow()) {
      defined_inside_dataflow_.insert(var);
    }
  }

  void VisitExpr_(const VarNode* var) override {
    auto var_ref = GetRef<Var>(var);
    // if a var is used in a dataflow block but *not* the one
    // where it was defined, it also needs to be exposed, so also we treat that as
    // used outside of a dataflow block
    if (!inside_dataflow() ||
        (def_blocks_.count(var_ref) &&
         (current_block_.defined() && !current_block_.value().same_as(def_blocks_.at(var_ref))))) {
      used_outside_home_dataflow_.insert(GetRef<Var>(var));
    }
  }

  inline bool inside_dataflow() {
    return current_block_.defined() && current_block_.value().as<DataflowBlockNode>();
  }

  Optional<BindingBlock> current_block_;
  Map<Var, BindingBlock> def_blocks_;

  Map<Var, Var> trivial_bindings_;
  Map<Var, Expr> known_bindings_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> defined_inside_dataflow_;
  // Set of vars either used outside a dataflow block altogether or outside their
  // home dataflow block (the one where they were defined)
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> used_outside_home_dataflow_;
};

/*! \brief The mutator class to apply a CanonicalizationPlan */
class BindingCanonicalizer : public ExprMutator {
 public:
  static Expr Apply(Expr expr) {
    auto used_outside_home_dataflow = CanonicalizePlanner::Collect(expr);
    BindingCanonicalizer mutator(std::move(used_outside_home_dataflow));
    return mutator.VisitExpr(expr);
  }

 private:
  explicit BindingCanonicalizer(CanonicalizationPlan plan) : plan_(plan) {}

  void VisitBinding(const Binding& binding) override {
    if (!plan_.bindings_to_remove.count(binding->var->vid)) {
      ExprMutator::VisitBinding(binding);
    }
  }

  Var VisitVarDef(const Var& var) override {
    Var new_var = var;
    while (auto opt = plan_.replace_binding.Get(new_var->vid)) {
      new_var = opt.value();
    }

    return ExprMutator::VisitVarDef(new_var);
  }

  Expr VisitExpr_(const VarNode* var) override {
    Var new_var = GetRef<Var>(var);
    while (auto opt = plan_.replace_usage.Get(new_var->vid)) {
      new_var = opt.value();
    }

    return ExprMutator::VisitExpr_(new_var.get());
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
  CanonicalizationPlan plan_;
};
}  // namespace

Expr CanonicalizeBindings(const Expr& expr) { return BindingCanonicalizer::Apply(expr); }

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
