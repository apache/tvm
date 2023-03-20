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
 * \file tvm/relax/transform/infer_purity.cc
 * \brief Insert annotations for function purity for any unannotated function in the module.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

void UpdatePurityAnnotation(FunctionNode* func, bool pure) {
  auto attrs = func->attrs;
  auto dict = attrs->dict;
  dict.Set("fPure", (pure) ? Integer(1) : Integer(2));
  func->attrs = DictAttrs(dict);
}

bool DetectPurity(const IRModule& mod_, const GlobalVar& gv, Map<GlobalVar, bool> global_purity_map,
                  Map<Var, bool> local_purity_map) {
  // look up the func
  // go through the body:
  //   track purity of local bindings (need to track closures and tuples)
  //   for each local binding, determine if it denotes a pure/impure closure or tuple member purity
  //   if there is an impure call (call to impure local func, call to impure global func, call to
  //   impure op, call to packed func)
  //     consider this function impure
  return true;
}

Expr UpdateLocalFunctions(const Expr& body, const Map<Var, bool>& local_purity_map) {
  class LocalFuncUpdater : public ExprMutator {
   public:
    explicit LocalFuncUpdater(const Map<Var, bool>& local_purity_map) : map_(local_purity_map) {}

    void VisitBinding_(const VarBindingNode* binding, const FunctionNode* func) override {
      // if func is not already annotated, add an annotation
      Var v = binding->var;
      Function f = Downcast<Function>(this->VisitExpr(GetRef<Function>(func)));
      if (!f->HasNonzeroAttr("fPure")) {
        auto new_func = f.CopyOnWrite();
        UpdatePurityAnnotation(new_func, map_.Get(v).value_or(true));
      }
      ReEmitBinding(binding, f);
    }

   private:
    const Map<Var, bool>& map_;
  };

  LocalFuncUpdater updater(local_purity_map);
  return updater.VisitExpr(body);
}

IRModule InferPurity(IRModule mod_) {
  /* Overall approach: A fixpoint algorithm
   *
   * Keep a map of global funcs -> bool, where the bool indicates whether the func is pure
   * Initially set all global funcs in the map to be pure
   *   (or use the annotated purity if present)
   * Worklist = [all global funcs except those that already have annotations]
   * Until the map stops changing:
   *   Next round worklist = []
   *   For each global func in the worklist:
   *     Check the func body to see if there is any impure call
   *     If the check finds an impure call,
   *       add all callers of the func to the next round worklist
   *     Update the global function mapping with the updated purity
   *     (maybe just update the annotations and struct info for the global var now???)
   *   Worklist = next round worklist
   * Insert annotations corresponding to the values in the final mapping
   * 
   * Worklist = [all global funcs]
   * Repeat until we have no changes in an iteration:
   *   Next worklist = []
   *   For each global func in the worklist:
   *     Check the body to see if there is any impure call and update local bindings
   *     Update the func with an annotation corresponding to the detected purity
   *     If there was any change made, add all callers of the func to the next worklist
   *   Worklist = next worklist
   */

  Map<GlobalVar, bool> global_purity_map;
  Map<Var, bool> local_purity_map;

  // the keys are sets of global vars
  // (the bool value is unused, but there is no tvm::Set)
  Map<GlobalVar, Map<GlobalVar, bool>> callers;

  Array<GlobalVar> worklist;

  // initialize maps: treat all global functions as pure
  // also treat func parameters as pure (TODO: use annotation for those)
  for (auto gv : mod_->GetGlobalVars()) {
    // only consider relax vars
    auto func = mod_->Lookup(gv);
    if (!func->IsInstance<relax::Function>()) {
      continue;
    }

    // if it's not already annotated, include it in the initial worklist
    if (!func->HasNonzeroAttr("fPure")) {
      worklist.push_back(gv);
    }

    // if it wasn't already set up, add a mapping
    if (!callers.count(gv)) {
      callers.Set(gv, {});
    }

    auto relax_func = Downcast<relax::Function>(func);
    for (const Var& param : relax_func->params) {
      if (GetStructInfo(param)->IsInstance<FuncStructInfo>()) {
        // TODO: Use the StructInfo annotation
        local_purity_map.Set(param, true);
      }
    }

    // if there is already an annotation, use that
    // 0 -> unspecified, 1 -> pure, 2 -> impure
    int purity = relax_func->GetAttr<Integer>("fPure", Integer(1)).value_or(1).IntValue();
    global_purity_map.Set(gv, (purity == 1) ? true : false);

    // update the set of called functions
    auto called_gvs = AllGlobalVars(relax_func);
    for (auto called_gv : called_gvs) {
      // ignore those that aren't Relax funcs
      if (!mod_->Lookup(called_gv)->IsInstance<relax::Function>()) {
        continue;
      }
      // also ignore simple recursion (there is no need to re-visit the same func)
      // todo: think about this case. You may need to revisit local funcs
      // if (called_gv.same_as(gv)) {
      //   continue;
      // }

      // make a new called set if one hasn't been initialized yet
      auto called_set = callers.Get(called_gv).value_or({});
      called_set.Set(gv, true);
      callers.Set(called_gv, called_set);
    }
  }

  bool changed = false;
  do {
    changed = false;
    Array<GlobalVar> next_worklist;
    for (auto gv : worklist) {
      // ignore those that have already been annotated or are marked impure
      auto relax_func = Downcast<relax::Function>(mod_->Lookup(gv));
      // first update local defs if needed

      // then check the purity
      bool checked_purity = DetectPurity(mod_, gv, global_purity_map, local_purity_map);
      if (!checked_purity) {
        changed = true;
        for (auto caller_gv : callers.Get(gv).value()) {
          next_worklist.push_back(caller_gv.first);
        }
        global_purity_map.Set(gv, checked_purity);
      }
    }
    worklist = std::move(next_worklist);
  } while (changed);

  // when the map stops changing, insert annotations and return new mod
  for (auto mapping : global_purity_map) {
    auto gv = mapping.first;
    auto relax_func = Downcast<relax::Function>(mod_->Lookup(gv));
    auto new_func = relax_func.CopyOnWrite();
    new_func->body = UpdateLocalFunctions(relax_func->body, local_purity_map);
    if (!relax_func->HasNonzeroAttr("fPure")) {
      UpdatePurityAnnotation(new_func, mapping.second);
    }
    mod_->functions.Set(gv, relax_func);
  }

  return mod_;
}

}  // namespace relax

namespace transform {

Pass InferPurity() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::InferPurity(m); };
  return CreateModulePass(pass_func, 0, "InferPurity", {});
}

TVM_REGISTER_GLOBAL("relax.transform.InferPurity").set_body_typed(InferPurity);
}  // namespace transform
}  // namespace tvm
