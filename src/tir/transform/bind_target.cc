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
 * \file bind_target.cc
 * \brief Pass to bind target to primfunc for heterogeneous compilation.
 *
 * This pass analyzes function call patterns in an IRModule and binds appropriate
 * targets (host/device) to each PrimFunc based on where they are called from.
 *
 * The pass handles the following scenarios:
 * 1. Functions called from host code (CPU)
 * 2. Functions called from device code (GPU/accelerator)
 * 3. Functions called from both host and device
 * 4. Externally exposed functions (entry points)
 *
 * For functions called from both host and device, the pass creates duplicates
 * with appropriate targets and updates call sites accordingly.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "tvm/ir/attrs.h"

namespace tvm {
namespace tir {

/*!
 * \brief Visitor class to classify function calls as host or device calls.
 *
 * This visitor traverses the IRModule to identify which functions are called
 * from host code vs device code. It tracks GPU scopes (thread binding loops
 * and thread extent attributes) to determine the calling context.
 */
class FunctionClassifierVisitor : public StmtExprVisitor {
 public:
  /*!
   * \brief Analyze function call patterns in the IRModule.
   * \param mod The IRModule to analyze
   * \return A tuple containing:
   *         - Set of GlobalVarNodes called from host code
   *         - Set of GlobalVarNodes called from device code
   * \note A single function can be called by both host and device contexts.
   */
  static std::tuple<std::unordered_set<const GlobalVarNode*>,
                    std::unordered_set<const GlobalVarNode*>>
  GetFunctionCallers(const IRModule& mod) {
    FunctionClassifierVisitor visitor;

    // Only analyze externally exposed functions as potential callers
    // since they represent the entry points where host/device calls originate
    for (const auto& [gvar, func] : mod->functions) {
      bool is_externally_exposed = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value();
      const auto* prim_func = func.as<PrimFuncNode>();

      if (is_externally_exposed && prim_func != nullptr) {
        visitor.VisitStmt(prim_func->body);
      }
    }

    return std::make_tuple(visitor.host_called_global_vars_, visitor.device_called_global_vars_);
  }

 private:
  using StmtExprVisitor::VisitStmt_;

  void VisitExpr_(const CallNode* op) final {
    const auto* global_var = op->op.as<GlobalVarNode>();
    if (global_var != nullptr) {
      // Classify the call based on current scope
      if (is_under_gpu_scope_) {
        device_called_global_vars_.insert(global_var);
      } else {
        host_called_global_vars_.insert(global_var);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kThreadBinding) {
      // Enter GPU scope for thread binding loops
      bool last_is_under_gpu_scope = is_under_gpu_scope_;
      is_under_gpu_scope_ = true;
      StmtExprVisitor::VisitStmt_(op);
      is_under_gpu_scope_ = last_is_under_gpu_scope;
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      // Enter GPU scope for thread extent and virtual thread attributes
      bool last_is_under_gpu_scope = is_under_gpu_scope_;
      is_under_gpu_scope_ = true;
      StmtExprVisitor::VisitStmt_(op);
      is_under_gpu_scope_ = last_is_under_gpu_scope;
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

 private:
  /*! \brief Whether the current statement is under a GPU scope */
  bool is_under_gpu_scope_ = false;
  /*! \brief Set of functions called from host code */
  std::unordered_set<const GlobalVarNode*> host_called_global_vars_;
  /*! \brief Set of functions called from device code */
  std::unordered_set<const GlobalVarNode*> device_called_global_vars_;
};

/*!
 * \brief Mutator class to substitute function calls in host contexts.
 *
 * This mutator replaces calls to functions that have been duplicated for
 * host/device contexts. It only performs substitutions when not under
 * GPU scope to ensure device calls remain unchanged.
 */
class CallSubstitutor : public StmtExprMutator {
 public:
  /*!
   * \brief Constructor with function replacement mapping.
   * \param replacements Map from original GlobalVar to host-specific GlobalVar
   */
  explicit CallSubstitutor(const ffi::Map<GlobalVar, GlobalVar>& replacements)
      : replacements_(replacements) {}

  /*!
   * \brief Substitute function calls in a PrimFunc.
   * \param func The PrimFunc to process
   * \return The modified PrimFunc with updated calls
   */
  PrimFunc Substitute(PrimFunc func) {
    auto f = func.CopyOnWrite();
    auto body = VisitStmt(f->body);

    // Only update if the body actually changed
    if (body.same_as(func->body)) {
      return func;
    }

    f->body = std::move(body);
    return func;
  }

 private:
  using StmtExprMutator::VisitStmt_;

  PrimExpr VisitExpr_(const CallNode* op) final {
    auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    // Only substitute calls when not under GPU scope
    if (!is_under_gpu_scope_) {
      if (auto old_gvar = call->op.as<GlobalVar>()) {
        if (auto new_gvar = replacements_.Get(old_gvar.value())) {
          call.CopyOnWrite()->op = new_gvar.value();
        }
      }
    }
    return call;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kThreadBinding) {
      // Enter GPU scope for thread binding loops
      bool last_is_under_gpu_scope = is_under_gpu_scope_;
      is_under_gpu_scope_ = true;
      auto stmt = StmtExprMutator::VisitStmt_(op);
      is_under_gpu_scope_ = last_is_under_gpu_scope;
      return stmt;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      // Enter GPU scope for thread extent and virtual thread attributes
      bool last_is_under_gpu_scope = is_under_gpu_scope_;
      is_under_gpu_scope_ = true;
      auto stmt = StmtExprMutator::VisitStmt_(op);
      is_under_gpu_scope_ = last_is_under_gpu_scope;
      return stmt;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

 private:
  /*! \brief Whether the current statement is under a GPU scope */
  bool is_under_gpu_scope_ = false;
  /*! \brief Mapping from original functions to host-specific duplicates */
  ffi::Map<GlobalVar, GlobalVar> replacements_;
};

/*!
 * \brief Bind appropriate targets to functions in an IRModule.
 *
 * This function analyzes the call patterns in the module and binds appropriate
 * targets to each PrimFunc based on where they are called from. The binding
 * follows these rules:
 *
 * 1. Externally exposed functions (with global symbol) get the full target
 * 2. Functions called only from host get the host target
 * 3. Functions called only from device get the device target
 * 4. Functions called from both contexts get the device target, and a duplicate
 *    is created with the host target for host callers
 *
 * \param mod The IRModule to process
 * \param target The target to bind (should include both host and device)
 * \return The modified IRModule with targets bound to functions
 */
IRModule BindTarget(IRModule mod, const Target& target) {
  // Extract host and device targets
  auto target_host = Downcast<Target>(target->host.value_or(Target("llvm")));
  auto target_without_host = target.WithoutHost();

  auto mod_copy_on_write = mod.CopyOnWrite();
  auto new_mod = ffi::GetRef<IRModule>(mod_copy_on_write);

  // Step 1: Analyze function call patterns
  auto [host_called_global_vars, device_called_global_vars] =
      FunctionClassifierVisitor::GetFunctionCallers(mod);

  // Step 2: Bind target to functions with the following rules:
  //  1. If the function has a target, and the target has a host, and the function does not have a
  //     host, then add the host to the function target
  //  2. If the function is marked as host function, bind the host target to the function
  //  3. If the function is externally exposed (with global symbol), bind the full target
  //  4. If the function is not externally exposed:
  //    2.1 If the function is called by both host and device, bind the device target to the current
  //        function and duplicate the function with the host target.
  //    2.2 If the function is called by host only, bind the host target to the current function
  //    2.3 If the function is called by device only, bind the device target to the current function
  //    2.4 If the function is not called by any host or device, skip binding

  // Track duplicated functions for call replacement
  ffi::Map<GlobalVar, GlobalVar> host_function_replacements;
  GlobalVarSupply gvar_supply(new_mod);

  for (auto [gvar, func] : mod->functions) {
    const auto* prim_func_node = func.as<PrimFuncNode>();
    if (prim_func_node == nullptr) {
      // Skip non-PrimFunc entries
      continue;
    }
    auto prim_func = ffi::GetRef<PrimFunc>(prim_func_node);

    bool is_externally_exposed =
        prim_func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value();

    if (auto func_target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      // Rule 1: If the function has a target, and the target has a host, and the function does not
      // have a host, then add the host to the function target
      auto func_target_host = func_target.value()->GetHost();
      auto target_host = target->GetHost();

      if (target_host && !func_target_host && is_externally_exposed) {
        auto new_target = Target::WithHost(func_target.value(), target_host.value());
        new_mod->Update(gvar, WithAttr(std::move(prim_func), tvm::attr::kTarget, new_target));
      }
      continue;
    }

    if (prim_func->HasNonzeroAttr(tvm::tir::attr::kIsHostFunc)) {
      // Rule 2: If the function is marked as host function, bind the host target to the function
      prim_func = WithAttr(std::move(prim_func), tvm::attr::kTarget,
                           Target::WithHost(target_host, target_host));
      new_mod->Update(gvar, WithoutAttr(std::move(prim_func), tvm::tir::attr::kIsHostFunc));
      continue;
    }

    if (is_externally_exposed) {
      // Rule 3: Externally exposed functions get the full target
      new_mod->Update(gvar, WithAttr(std::move(prim_func), tvm::attr::kTarget, target));
    } else {
      const auto* gvar_node = gvar.get();
      bool called_by_host = host_called_global_vars.count(gvar_node);
      bool called_by_device = device_called_global_vars.count(gvar_node);

      if (called_by_host && called_by_device) {
        // Rule 4.1: Called by both host and device
        // Bind device target to current function
        PrimFunc host_func = RenewDefs(prim_func);
        new_mod->Update(gvar,
                        WithAttr(std::move(prim_func), tvm::attr::kTarget, target_without_host));

        // Create duplicate with host target for host callers
        host_func = WithAttr(std::move(host_func), tvm::attr::kTarget, target_host);
        ffi::String host_func_name = gvar->name_hint + "_host";
        GlobalVar host_gvar = gvar_supply->FreshGlobal(host_func_name, false);

        new_mod->Add(host_gvar, host_func);
        host_function_replacements.Set(gvar, host_gvar);

      } else if (called_by_host) {
        // Rule 4.2: Called by host only
        new_mod->Update(gvar, WithAttr(std::move(prim_func), tvm::attr::kTarget, target_host));
      } else if (called_by_device) {
        // Rule 4.3: Called by device only
        new_mod->Update(gvar,
                        WithAttr(std::move(prim_func), tvm::attr::kTarget, target_without_host));
      } else {
        // Rule 4.4: Not called by any context
        // NOTE: To keep the current behavior, we bind the target to the full target, but it needs
        // further check
        new_mod->Update(gvar,
                        WithAttr(std::move(prim_func), tvm::attr::kTarget, target_without_host));
      }
    }
  }

  // Step 3: Update call sites in externally exposed functions
  if (!host_function_replacements.empty()) {
    CallSubstitutor substitutor(host_function_replacements);

    for (auto [gvar, func] : mod->functions) {
      const auto* prim_func = func.as<PrimFuncNode>();
      if (prim_func == nullptr) {
        continue;
      }

      bool is_externally_exposed =
          prim_func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value();
      if (is_externally_exposed) {
        // Update calls in externally exposed functions to use host duplicates
        PrimFunc new_func = substitutor.Substitute(Downcast<PrimFunc>(func));
        new_mod->Update(gvar, new_func);
      }
    }
  }

  return new_mod;
}

namespace transform {

/*!
 * \brief Create a pass that binds targets to functions in an IRModule.
 *
 * This pass analyzes the call patterns in the module and binds appropriate
 * targets (host/device) to each PrimFunc based on where they are called from.
 *
 * \param target The target to bind (should include both host and device)
 * \return A transform pass that performs target binding
 */
transform::Pass BindTarget(Target target) {
  auto fpass = [target](IRModule mod, transform::PassContext ctx) {
    return tvm::tir::BindTarget(mod, target);
  };
  return tir::transform::CreateModulePass(fpass, 0, "tir.BindTarget", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.BindTarget", BindTarget);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
