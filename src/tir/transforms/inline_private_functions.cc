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
 * \file inline_private_functions.cc
 * \brief Inline private functions to their callsite
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace transform {

namespace {

template <typename T>
using PSet = std::unordered_set<T, ObjectPtrHash, ObjectPtrEqual>;

template <typename T, typename U>
using PMap = std::unordered_map<T, U, ObjectPtrHash, ObjectPtrEqual>;

PMap<GlobalVar, PSet<GlobalVar>> CollectCallMap(const IRModule& mod) {
  struct Visitor : StmtExprVisitor {
    GlobalVar current;
    PMap<GlobalVar, PSet<GlobalVar>> caller_lookup;

    void VisitExpr_(const CallNode* op) {
      if (auto gvar = op->op.as<GlobalVar>()) {
        caller_lookup[gvar.value()].insert(current);
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  } visitor;

  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto prim_func = base_func.as<PrimFuncNode>()) {
      visitor.current = gvar;
      visitor(prim_func->body);
    }
  }

  return visitor.caller_lookup;
}

PSet<GlobalVar> CollectRecursiveFunctions(const IRModule& mod) {
  // Collect all direct callers
  auto call_map = CollectCallMap(mod);

  // Propagate to find all indirect callers
  while (true) {
    bool made_change = false;
    for (const auto& [callee, callers] : call_map) {
      for (const auto& caller : callers) {
        if (auto it = call_map.find(caller); it != call_map.end()) {
          PSet<GlobalVar>& indirect_callers = it->second;

          auto res = indirect_callers.insert(callee);
          made_change = made_change || res.second;
        }
      }
    }
    if (!made_change) {
      break;
    }
  }

  // Filter all GlobalVars that can be called by themselves, either
  // directly or indirectly.
  PSet<GlobalVar> recursive_funcs;
  for (const auto& [caller, callees] : call_map) {
    if (callees.count(caller)) {
      recursive_funcs.insert(caller);
    }
  }
  return recursive_funcs;
}

bool IsInlinablePrimFunc(const GlobalVar& gvar, const PrimFunc& prim_func,
                         const PSet<GlobalVar>& recursive_functions) {
  // Only inline private functions.  Externally-exposed functions
  // must be preserved so to avoid breaking callsites outside of
  // the IRModule.
  bool is_exposed = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined();
  if (is_exposed) return false;

  // We do not currently implement any analysis for termination of
  // a function.  If a recursive function requires runtime checks
  // in order to terminate, we would keep inlining until the
  // recursive visits segfault.
  bool is_recursive = recursive_functions.count(gvar);
  if (is_recursive) return false;

  // We do not currently support inlining of functions that accept
  // buffer arguments.
  bool has_buffer_arguments = prim_func->buffer_map.size();
  if (has_buffer_arguments) return false;

  // We do not currently support inlining of schedulable TIR
  // functions.  To support this use case, repeated names in
  // `tir::Block` nodes resulting from multiple calls to the same
  // inlined function will need to be de-duplicated.
  bool has_block_node = prim_func->body.as<BlockRealizeNode>();
  if (has_block_node) return false;

  return true;
}

Map<GlobalVar, PrimFunc> CollectInlinablePrimFuncs(const IRModule& mod) {
  auto recursive_functions = CollectRecursiveFunctions(mod);

  Map<GlobalVar, PrimFunc> output;
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto opt = base_func.as<PrimFunc>()) {
      auto prim_func = opt.value();
      if (IsInlinablePrimFunc(gvar, prim_func, recursive_functions)) {
        output.Set(gvar, prim_func);
      }
    }
  }

  return output;
}

class PrimFuncInliner : StmtExprMutator {
 public:
  explicit PrimFuncInliner(Map<GlobalVar, PrimFunc> inlinable_funcs)
      : inlinable_funcs_(inlinable_funcs) {
    for (const auto& [gvar, callee] : inlinable_funcs_) {
      removable_funcs_.insert(gvar);
    }
  }

  PrimFunc VisitFunc(PrimFunc func) {
    current_target_ = func->GetAttr<Target>(tvm::attr::kTarget);
    auto new_body = VisitStmt(func->body);
    current_target_ = NullOpt;

    if (!new_body.same_as(func->body)) {
      func.CopyOnWrite()->body = new_body;
    }

    return func;
  }

  PSet<GlobalVar> GetRemovableFunctions() const { return removable_funcs_; }

 private:
  Stmt VisitStmt_(const EvaluateNode* eval) override {
    if (auto inlined = GetInlinedFunction(eval)) {
      return inlined.value();
    } else {
      return StmtExprMutator::VisitStmt_(eval);
    }
  }

  Optional<Stmt> GetInlinedFunction(const EvaluateNode* eval) {
    auto call = eval->value.as<CallNode>();
    if (!call) return NullOpt;

    auto gvar = call->op.as<GlobalVar>();
    if (!gvar) return NullOpt;

    auto opt_callee = inlinable_funcs_.Get(gvar.value());
    if (!opt_callee) return NullOpt;
    auto callee = opt_callee.value();

    bool is_same_target = [&]() -> bool {
      auto callee_target = callee->GetAttr<Target>(tvm::attr::kTarget);
      if (current_target_ && callee_target) {
        return callee_target.value()->str() == current_target_.value()->str();
      } else {
        return true;
      }
    }();
    if (!is_same_target) return NullOpt;

    Stmt inlined = InlineArguments(gvar.value(), callee, call->args);
    return VisitStmt(inlined);
  }

  PrimExpr VisitExpr_(const CallNode* call) override {
    // Because the current implementation inlines a subroutine inserts
    // the `tir::Stmt` body at the point of use, replacement must
    // occur in a context where a `tir::Stmt` can be returned. Support
    // of subroutines that are called within an expression
    // (e.g. Replacing func in `Buf[0] = func(1) + func(2)`) would
    // require hoisting preprocessing done in the subroutine to the
    // parent `tir::Stmt`.
    //
    // See `TestInlineCallOccurringInExpression` in
    // `test_tir_inline_private_functions.py` for a test of this
    // behavior, currently marked with `pytest.mark.xfail`.
    //
    // Any callee that hasn't been inlined at this point must be kept
    // in the output IRModule.
    if (auto gvar = call->op.as<GlobalVar>()) {
      removable_funcs_.erase(gvar.value());
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  Stmt InlineArguments(const GlobalVar& gvar, PrimFunc callee, const Array<PrimExpr>& args) const {
    CHECK_EQ(callee->params.size(), args.size())
        << "Callee " << gvar << " accepts " << callee->params.size() << " parameters ("
        << callee->params << "), but is called with " << args.size() << " arguments (" << args
        << ")";

    ICHECK(callee->buffer_map.empty())
        << "Inlining of PrimFuncs with buffer arguments is not yet supported, "
        << "but callee " << gvar << " has non-empty buffer map " << callee->buffer_map;

    Map<Var, ObjectRef> param_map;
    for (size_t i = 0; i < callee->params.size(); i++) {
      param_map.Set(callee->params[i], args[i]);
    }

    callee = Specialize(callee, param_map);

    return callee->body;
  }

  // Map from GlobalVar to PrimFuncs which may be inlined.
  Map<GlobalVar, PrimFunc> inlinable_funcs_;

  /* \brief Set of callees that may be removed
   *
   * Some constructs may not be inlined (e.g. if the call site occurs
   * outside of an Evaluate node).  For these cases, the output
   * IRModule must still contain the callee.
   */
  PSet<GlobalVar> removable_funcs_;

  Optional<Target> current_target_ = NullOpt;
};

}  // namespace

Pass InlinePrivateFunctions() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    auto inlinable_prim_funcs = CollectInlinablePrimFuncs(mod);

    if (inlinable_prim_funcs.empty()) {
      // Early bail-out if the module has no inlinable PrimFuncs.
      return mod;
    }

    PrimFuncInliner mutator(std::move(inlinable_prim_funcs));
    IRModule updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto updated = mutator.VisitFunc(opt.value());
        if (!updated.same_as(base_func)) {
          updates->Add(gvar, updated);
        }
      }
    }

    if (updates->functions.size()) {
      auto write_ptr = mod.CopyOnWrite();
      write_ptr->Update(updates);
      for (const auto& gvar : mutator.GetRemovableFunctions()) {
        write_ptr->Remove(gvar);
      }
      mod = ConvertSSA()(mod);
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.InlinePrivateFunctions", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InlinePrivateFunctions").set_body_typed(InlinePrivateFunctions);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
