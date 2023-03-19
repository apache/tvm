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
 * Currently it removes:
 *   1. Unused local VarBindings in a DataflowBlock.
 *      The used var set is set to empty at the beginning of each DataflowBlock.
 *      We reverse scan the DataflowBlock, if a VarBinding
 *        - bindings to a DataflowVar, or
 *        - is used in the used var set
 *      We keep it and add its var to the used var set. Otherwise, we remove it.
 *   2. Unused Relax functions in the module.
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

class DeadCodeEliminator : public ExprMutator {
 private:
  Expr VisitExpr_(const VarNode* op) final {
    ICHECK(!used_vars_.empty());
    used_vars_.back().insert(GetRef<Var>(op));
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    ICHECK(!used_vars_.empty());
    used_vars_.back().insert(GetRef<Var>(op));
    return GetRef<Expr>(op);
  }

  void VisitBinding_(const VarBindingNode* binding) { this->VisitExpr(binding->value); }

  void VisitBinding_(const MatchCastNode* binding) {
    this->VisitExpr(binding->value);
    this->VisitAndCheckStructInfoFieldUnchanged(binding->struct_info);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    // reverse scan the data flow block to find the used vars
    used_vars_.push_back({});

    std::vector<Binding> new_bindings;
    for (auto rit = block->bindings.rbegin(); rit != block->bindings.rend(); rit++) {
      const Var& var = (*rit)->var;
      // only keep the used bindings or non dataflow var bindings
      if (used_vars_.back().count(var) || !var->IsInstance<DataflowVarNode>()) {
        new_bindings.push_back(*rit);
        // collect the used vars
        this->VisitBinding((*rit));
      }
    }

    used_vars_.pop_back();
    // reverse the bindings
    std::reverse(new_bindings.begin(), new_bindings.end());
    if (new_bindings.size() == block->bindings.size()) {
      return GetRef<BindingBlock>(block);
    } else {
      auto n = make_object<DataflowBlockNode>(*block);
      n->bindings = std::move(new_bindings);
      return BindingBlock(n);
    }
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) final {
    return GetRef<BindingBlock>(block);
  }

  std::vector<std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>> used_vars_{{}};
};

IRModule DeadCodeElimination(const IRModule& mod, Array<runtime::String> entry_functions) {
  // S1: remove unused functions to reduce the number of functions to be analyzed.
  IRModule tmp_mod = RemoveUnusedFunctions(mod, entry_functions);
  // S2: remove unused variables in each function.
  for (const auto& gv : tmp_mod->GetGlobalVars()) {
    auto func = tmp_mod->Lookup(gv);
    if (func->IsInstance<FunctionNode>()) {
      tmp_mod->Update(gv, RemoveAllUnused(Downcast<Function>(func)));
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
