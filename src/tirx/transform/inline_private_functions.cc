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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {
namespace transform {

namespace {

template <typename T>
using PSet = std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

template <typename T, typename U>
using PMap = std::unordered_map<T, U, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

PMap<GlobalVar, PSet<GlobalVar>> CollectCallMap(const IRModule& mod) {
  struct Visitor : StmtExprVisitor {
    GlobalVar current{ffi::UnsafeInit{}};
    PMap<GlobalVar, PSet<GlobalVar>> caller_lookup;

    ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
      if (auto gvar = op->op.as<GlobalVar>()) {
        caller_lookup[gvar.value()].insert(current);
      }
      return StmtExprVisitor::Visit_(op);
    }
  };
  auto visitor = ffi::make_object<Visitor>();

  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto function = base_func.as<FunctionNode>()) {
      visitor->current = gvar;
      visitor->Visit(function->body);
    }
  }

  return visitor->caller_lookup;
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

bool IsInlinableFunction(const GlobalVar& gvar, const Function& function,
                         const PSet<GlobalVar>& recursive_functions) {
  // Only inline private functions.  Externally-exposed functions
  // must be preserved so to avoid breaking callsites outside of
  // the IRModule.
  bool is_exposed = function->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value();
  if (is_exposed) return false;

  // We do not currently implement any analysis for termination of
  // a function.  If a recursive function requires runtime checks
  // in order to terminate, we would keep inlining until the
  // recursive visits segfault.
  bool is_recursive = recursive_functions.count(gvar);
  if (is_recursive) return false;

  // We do not currently support inlining of functions that accept
  // buffer arguments.
  for (const Var& param : function->params) {
    if (param->ty.as<TensorTypeNode>()) return false;
  }

  // Generalize the old SBlockRealize exclusion to all non-native statement roots:
  // they may introduce binder or naming rules that this pass cannot preserve.
  // Only inline roots supported by native TIRX traversal.
  struct NativeStmtTable : StmtExprVisitor {
    static VTable Make() {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }
  };
  static const auto native_stmts = NativeStmtTable::Make();
  if (!function->body.has_value() || !native_stmts.CanDispatch(function->body.get())) {
    return false;
  }

  return true;
}

ffi::Map<GlobalVar, Function> CollectInlinableFunctions(const IRModule& mod) {
  auto recursive_functions = CollectRecursiveFunctions(mod);

  ffi::Map<GlobalVar, Function> output;
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto opt = base_func.as<Function>()) {
      auto function = opt.value();
      if (IsInlinableFunction(gvar, function, recursive_functions)) {
        output.Set(gvar, function);
      }
    }
  }

  return output;
}

class FunctionInliner : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit FunctionInliner(ffi::Map<GlobalVar, Function> inlinable_funcs)
      : inlinable_funcs_(inlinable_funcs) {
    for (const auto& [gvar, callee] : inlinable_funcs_) {
      removable_funcs_.insert(gvar);
    }
  }

  Function VisitFunc(Function func) {
    current_target_ = func->GetAttr<Target>(tvm::attr::kTarget);
    auto new_body_result = Mutate(func->body, InplaceMode::kDisallow);
    bool new_body_unchanged = new_body_result.UnchangedOrSameAs(func->body);
    auto new_body = std::move(new_body_result).ValueOrUnchanged(func->body);
    current_target_ = std::nullopt;

    if (!new_body_unchanged) {
      func.CopyOnWrite()->body = new_body;
    }

    return func;
  }

  PSet<GlobalVar> GetRemovableFunctions() const { return removable_funcs_; }

 private:
  UnchangedOr<Stmt> Mutate_(const EvaluateNode* eval, InplaceMode inplace_mode) override {
    if (auto inlined = GetInlinedFunction(eval)) {
      return inlined.value();
    } else {
      return StmtExprMutator::Mutate_(eval, inplace_mode);
    }
  }

  ffi::Optional<Stmt> GetInlinedFunction(const EvaluateNode* eval) {
    auto call = eval->value.as<CallNode>();
    if (!call) return std::nullopt;

    auto gvar = call->op.as<GlobalVar>();
    if (!gvar) return std::nullopt;

    auto opt_callee = inlinable_funcs_.Get(gvar.value());
    if (!opt_callee) return std::nullopt;
    auto callee = opt_callee.value();

    bool is_same_target = [&]() -> bool {
      auto callee_target = callee->GetAttr<Target>(tvm::attr::kTarget);
      if (current_target_ && callee_target) {
        return callee_target.value()->str() == current_target_.value()->str();
      } else {
        return true;
      }
    }();
    if (!is_same_target) return std::nullopt;

    Stmt inlined = InlineArguments(gvar.value(), callee, call->args);
    return Mutate(inlined, InplaceMode::kDisallow).ValueOrUnchanged(inlined);
  }

  UnchangedOr<Expr> Mutate_(const CallNode* call, InplaceMode inplace_mode) override {
    // Because the current implementation inlines a subroutine inserts
    // the `Stmt` body at the point of use, replacement must
    // occur in a context where a `Stmt` can be returned. Support
    // of subroutines that are called within an expression
    // (e.g. Replacing func in `Buf[0] = func(1) + func(2)`) would
    // require hoisting preprocessing done in the subroutine to the
    // parent `Stmt`.
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
    return StmtExprMutator::Mutate_(call, inplace_mode);
  }

  Stmt InlineArguments(const GlobalVar& gvar, Function callee, const ffi::Array<Expr>& args) const {
    TVM_FFI_ICHECK_EQ(callee->params.size(), args.size())
        << "Callee " << gvar << " accepts " << callee->params.size() << " parameters ("
        << callee->params << "), but is called with " << args.size() << " arguments (" << args
        << ")";

    for (const Var& param : callee->params) {
      TVM_FFI_ICHECK(!param->ty.as<TensorTypeNode>())
          << "Inlining of Functions with buffer arguments is not yet supported, "
          << "but callee " << gvar << " has TensorType-annotated parameter " << param;
    }

    ffi::Map<Var, ffi::Variant<tirx::TensorVar, tvm::Expr>> param_map;
    for (size_t i = 0; i < callee->params.size(); i++) {
      param_map.Set(callee->params[i], args[i]);
    }

    callee = Specialize(callee, param_map);

    return callee->body.value();
  }

  // Map from GlobalVar to Functions which may be inlined.
  ffi::Map<GlobalVar, Function> inlinable_funcs_;

  /* \brief Set of callees that may be removed
   *
   * Some constructs may not be inlined (e.g. if the call site occurs
   * outside of an Evaluate node).  For these cases, the output
   * IRModule must still contain the callee.
   */
  PSet<GlobalVar> removable_funcs_;

  ffi::Optional<Target> current_target_ = std::nullopt;
};

}  // namespace

Pass InlinePrivateFunctions() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    auto inlinable_functions = CollectInlinableFunctions(mod);

    if (inlinable_functions.empty()) {
      // Early bail-out if the module has no inlinable Functions.
      return mod;
    }

    auto mutator = ffi::make_object<FunctionInliner>(std::move(inlinable_functions));
    IRModule updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<Function>()) {
        auto updated = mutator->VisitFunc(opt.value());
        if (!updated.same_as(base_func)) {
          updates->Add(gvar, updated);
        }
      }
    }

    if (updates->functions.size()) {
      auto write_ptr = mod.CopyOnWrite();
      write_ptr->Update(updates);
      for (const auto& gvar : mutator->GetRemovableFunctions()) {
        write_ptr->Remove(gvar);
      }
      mod = ConvertSSA()(mod);
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tirx.InlinePrivateFunctions");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.InlinePrivateFunctions", InlinePrivateFunctions);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
