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
 * \file tvm/relax/transform/fold_dataflow_block_output.cc
 * \brief Pass that folds dataflow vars used only for binding
 *   the dataflow block output
 *   directly into the output
 *
 * If a dataflow var is used only in a binding to the dataflow block's
 * output var (a non-dataflow var), this pass removes the dataflow var
 * binding from the block and uses the dataflow var's definition
 * directly in the output binding.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

// If a dataflow var is used *only* as the RHS of a binding to the dataflow block output
// (i.e., an ordinary var), then we can get rid of that dataflow var and bind the DF var's
// definition directly to the output.
DataflowBlock FoldDataflowBlockOutput(const DataflowBlock& block) {
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
    void VisitBinding_(const VarBindingNode* binding) override {
      ProcessBinding(binding->var, binding->value);
    }

    void VisitBinding_(const MatchCastNode* binding) override {
      ProcessBinding(binding->var, binding->value);
    }

    void ProcessBinding(const Var& var, const Expr& value) {
      if (var.as<DataflowVarNode>()) {
        // add definition to binding map
        binding_map_[Downcast<DataflowVar>(var)] = value;

        // disqualify any dataflow vars in the RHS (since the LHS isn't an ordinary var)
        DataflowVarGatherer gatherer;
        auto disqualified = gatherer.Gather(value);
        for (auto var : disqualified) {
          disqualified_set_.insert(var);
        }
      } else {
        // the LHS is an output, so disqualify if the RHS is not a single dataflow var
        // or if the var has been output before
        if (const auto* rhs_var = value.as<DataflowVarNode>()) {
          if (output_vars_.count(GetRef<DataflowVar>(rhs_var))) {
            disqualified_set_.insert(GetRef<DataflowVar>(rhs_var));
          }
          output_vars_.insert(GetRef<DataflowVar>(rhs_var));
        } else {
          DataflowVarGatherer gatherer;
          auto disqualified = gatherer.Gather(value);
          for (auto var : disqualified) {
            disqualified_set_.insert(var);
          }
        }
      }
    }

    std::unordered_map<DataflowVar, Expr, ObjectPtrHash, ObjectPtrEqual> FindCandidates(
        const DataflowBlock& block) {
      VisitBindingBlock(block);
      // candidates: the output vars that are not in the disqualified set
      std::unordered_map<DataflowVar, Expr, ObjectPtrHash, ObjectPtrEqual> ret;
      for (auto var : output_vars_) {
        if (!disqualified_set_.count(var)) {
          ret[var] = binding_map_.at(var);
        }
      }
      return ret;
    }

    std::unordered_map<DataflowVar, Expr, ObjectPtrHash, ObjectPtrEqual> binding_map_;
    std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> disqualified_set_;
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
  auto candidate_map = finder.FindCandidates(block);
  BindingUpdater updater(candidate_map);
  auto new_block = updater.VisitBindingBlock(block);
  return Downcast<DataflowBlock>(new_block);
}

namespace transform {

Pass FoldDataflowBlockOutput() {
  const runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)>& pass_func =
      [=](DataflowBlock b, IRModule m, PassContext pc) {
        return relax::FoldDataflowBlockOutput(b);
      };
  return CreateDataflowBlockPass(pass_func, 1, "FoldDataflowBlockOutput", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FoldDataflowBlockOutput")
    .set_body_typed(FoldDataflowBlockOutput);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
