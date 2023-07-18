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
 * \file tvm/relax/transform/dead_code_elimination.cc
 * \brief Dead code elimination pass.
 * \sa tvm/relax/ir/binding_rewrite.cc
 *
 * Currently it removes:
 *   1. Unused local VarBindings in a DataflowBlock.
 *   2. Unused DataflowBlocks in a function.
 *   3. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

/**
 * \brief Detects all the functions that can be possibly called by entry function.
 */
class CallTracer : ExprVisitor {
 public:
  explicit CallTracer(IRModule mod_) : mod_{mod_}, called_funcs_{}, visiting_{} {}

  void VisitExpr_(const GlobalVarNode* op) final {
    called_funcs_.insert(GetRef<GlobalVar>(op));
    auto func = mod_->Lookup(op->name_hint);
    if (const auto* function_node = func.as<FunctionNode>()) {
      VisitExpr(GetRef<Function>(function_node));
    }
    // else: Don't visit PrimFuncs -- we don't need to collect any tir.Calls therein.
  }

  void VisitExpr_(const CallNode* call_node) final { ExprVisitor::VisitExpr_(call_node); }

  void VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);
    if (visiting_.find(func) == visiting_.end()) {
      visiting_.insert(func);
      for (auto param : func_node->params) {
        ExprVisitor::VisitExpr(param);
      }
      ExprVisitor::VisitExpr(func_node->body);
    }
  }

  void Trace(std::string entry) {
    called_funcs_.insert(mod_->GetGlobalVar(entry));
    auto main_func = mod_->Lookup(entry);
    VisitExpr(main_func);
  }

  bool check_if_called(GlobalVar gv) { return called_funcs_.count(gv) > 0; }

 private:
  IRModule mod_;

  // Record the names of all encountered functions.
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> called_funcs_;

  // Record the expressions that are being visited.
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visiting_;
};

IRModule RemoveUnusedFunctions(IRModule mod_, Array<runtime::String> entry_funcs) {
  auto tracer = CallTracer(mod_);
  for (auto entry : entry_funcs) {
    tracer.Trace(entry);
  }
  auto existing_functions = mod_->functions;
  for (auto f : existing_functions) {
    // If a function has an external linkage type, we do not remove it.
    // Otherwise, we check the function and remove it if it is not used anywhere.
    if (f.second->GetLinkageType() == LinkageType::kInternal && !tracer.check_if_called(f.first)) {
      mod_->Remove(f.first);
    }
  }
  return mod_;
}

// Checks the function for the following condition:
// If a dataflow var is used *only* as the LHS of a binding to the dataflow block output
// (i.e., an ordinary var), then we can get rid of that dataflow var and bind the DF var's
// definition directly to the output.
Function ElideIntermediateDataflowVars(const Function& func) {
  // helper: gather all dataflow vars inside an expression
  class DataflowVarGatherer : public ExprVisitor {
   public:
    // ignore inner functions
    void VisitExpr_(const FunctionNode* _) override {}

    void VisitExpr_(const DataflowVarNode* var) override { vars_.insert(GetRef<DataflowVar>(var)); }

    std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> Gather(const Expr& expr) {
      VisitExpr(expr);
      return vars_;
    }

    std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> vars_;
  };

  // first we search for dataflow vars for which the condition is met:
  // exclude if found anywhere other than RHS of a binding to an ordinary var (or more than once)
  // candidate set -> eliminate if we find somewhere it's not supposed to be
  class CandidateFinder : public ExprVisitor {
   public:
    // ignore non-DF blocks
    void VisitBindingBlock_(const BindingBlockNode* block) override {}

    void VisitBinding_(const VarBindingNode* binding) override {
      ProcessBinding(binding->var, binding->value);
    }

    void VisitBinding_(const MatchCastNode* binding) override {
      ProcessBinding(binding->var, binding->value);
    }

    void ProcessBinding(const Var& var, const Expr& value) {
      if (var.as<DataflowVarNode>()) {
        // add definition to binding map
        candidate_map_[Downcast<DataflowVar>(var)] = value;

        // disqualify any dataflow vars in the RHS (since the LHS isn't an ordinary var)
        DataflowVarGatherer gatherer;
        auto disqualified = gatherer.Gather(value);
        for (auto var : disqualified) {
          candidate_map_.erase(var);
        }
      } else {
        // the LHS is an output, so disqualify if the RHS is not a single dataflow var
        // or if the var has been output before
        if (const auto* rhs_var = value.as<DataflowVarNode>()) {
          if (output_vars_.count(GetRef<DataflowVar>(rhs_var))) {
            candidate_map_.erase(GetRef<DataflowVar>(rhs_var));
          }
          output_vars_.insert(GetRef<DataflowVar>(rhs_var));
        } else {
          DataflowVarGatherer gatherer;
          auto disqualified = gatherer.Gather(value);
          for (auto var : disqualified) {
            candidate_map_.erase(var);
          }
        }
      }
    }

    std::unordered_map<DataflowVar, Expr, ObjectPtrHash, ObjectPtrEqual> candidate_map_;
    std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> output_vars_;
  };

  // given a candidate map (dataflow vars that should be eliminated mapped to their definitions),
  // remove the bindings corresponding to those DF vars and replace the vars with their definitions
  // when the appear on the RHS of a binding to an output var (non-DF var)
  class BindingUpdater : public ExprMutator {
   public:
    explicit BindingUpdater(
        const std::unordered_map<DataflowVar, Expr, ObjectPtrHash, ObjectPtrEqual>& candidate_map)
        : candidate_map_(candidate_map) {}

    // skip non-DF blocks
    BindingBlock VisitBindingBlock_(const BindingBlockNode* block) override {
      return GetRef<BindingBlock>(block);
    }

    void VisitBinding_(const VarBindingNode* binding) override {
      // case 1: if the LHS is a DF node in the candidate map, erase the binding
      if (binding->var.as<DataflowVarNode>() &&
          candidate_map_.count(Downcast<DataflowVar>(binding->var))) {
        return;
      }
      // case 2: if the RHS consists only of a DF node in the candidate map, replace the value
      //   with the definition from the candidate map
      if (!binding->var.as<DataflowVarNode>() && binding->value.as<DataflowVarNode>() &&
          candidate_map_.count(Downcast<DataflowVar>(binding->value))) {
        builder_->EmitNormalized(
            VarBinding(binding->var, candidate_map_.at(Downcast<DataflowVar>(binding->value))));
        return;
      }
      // case 3: if neither, use the default logic
      ExprMutator::VisitBinding_(binding);
    };

    void VisitBinding_(const MatchCastNode* binding) {
      // case 1: if the LHS is a DF node in the candidate map, erase the binding
      if (binding->var.as<DataflowVarNode>() &&
          candidate_map_.count(Downcast<DataflowVar>(binding->var))) {
        return;
      }
      // case 2: if the RHS consists only of a DF node in the candidate map, replace the value
      //   with the definition from the candidate map
      if (!binding->var.as<DataflowVarNode>() && binding->value.as<DataflowVarNode>() &&
          candidate_map_.count(Downcast<DataflowVar>(binding->value))) {
        builder_->EmitNormalized(MatchCast(binding->var,
                                           candidate_map_.at(Downcast<DataflowVar>(binding->value)),
                                           binding->struct_info));
        return;
      }
      // case 3: if neither, use the default logic
      ExprMutator::VisitBinding_(binding);
    }

    const std::unordered_map<DataflowVar, Expr, ObjectPtrHash, ObjectPtrEqual>& candidate_map_;
  };

  CandidateFinder finder;
  finder.VisitExpr(func->body);
  auto candidate_map = finder.candidate_map_;
  BindingUpdater updater(candidate_map);
  auto new_body = updater.VisitExpr(func->body);
  return Function(func->params, new_body, func->ret_struct_info, func->is_pure, func->attrs);
}

IRModule DeadCodeElimination(const IRModule& mod, Array<runtime::String> entry_functions) {
  // S1: remove unused functions to reduce the number of functions to be analyzed.
  IRModule tmp_mod = RemoveUnusedFunctions(mod, entry_functions);
  // S2: remove unused variables in each function.
  for (const auto& gv : tmp_mod->GetGlobalVars()) {
    auto func = tmp_mod->Lookup(gv);
    if (func->IsInstance<FunctionNode>()) {
      auto new_func = ElideIntermediateDataflowVars(RemoveAllUnused(Downcast<Function>(func)));
      tmp_mod->Update(gv, new_func);
    }
  }
  // S3: remove unused functions again as some callers may be removed in S2.
  return RemoveUnusedFunctions(tmp_mod, entry_functions);
}

namespace transform {

Pass DeadCodeElimination(Array<runtime::String> entry_functions) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::DeadCodeElimination(m, entry_functions); };
  return CreateModulePass(pass_func, 1, "DeadCodeElimination", {});
}

TVM_REGISTER_GLOBAL("relax.transform.DeadCodeElimination").set_body_typed(DeadCodeElimination);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
