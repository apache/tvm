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
 * \file tir/analysis/verify_tirx_well_formed.cc
 * \brief Check if the TIRX program is well-formed.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/tirx_op.h>

#include <exception>
#include <optional>
#include <tuple>
#include <variant>

#include "../ir/functor_common.h"
#include "../ir/tir_visitor_with_path.h"
#include "tvm/ir/module.h"

namespace tvm {
namespace tirx {

class ExecScopeVerifier : public Verifier<ExecScopeVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlock is not allowed in tirx=True mode at " << path
                  << ". Use ExecScopeStmt with T.attr() instead.";
  }

  void VisitStmt_(const SBlockRealizeNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlockRealize is not allowed in tirx=True mode at " << path
                  << ". Use ExecScopeStmt with T.attr() instead.";
  }

  void VisitStmt_(const tirx::TilePrimitiveCallNode* op,
                  ffi::reflection::AccessPath path) override {
    static const tvm::OpAttrMap<bool>& tirx_op_map_ = Op::GetAttrMap<bool>("TIsTIRxOp");
    Verify(tirx_op_map_.count(op->op))
        << "TIRxError: TilePrimitiveCall at " << path << " has unknown TIRX op " << op->op;
  }

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    auto scope = op->exec_scope;
    // C1: exec_scope is valid
    // ExecScope ctor FATALs on unknown name, so a constructed scope is
    // always valid; nothing to re-check structurally here.
    bool is_root = false;
    if (!root_.has_value()) {
      root_ = scope;
      is_root = true;
    }
    if (!scope_stack_.empty()) {
      TVM_FFI_ICHECK(root_.has_value()) << "TIRxError: root scope should be the highest scope";
      Verify(!ScopeKindHigher(scope->kind, root_.value()->kind))
          << "TIRxError: ExecScopeStmt at " << path << " has invalid exec_scope " << scope->name()
          << " under " << root_.value()->name();
    }
    scope_stack_.push_back(scope);
    Verifier::VisitStmt_(op, path);
    scope_stack_.pop_back();
    if (is_root) root_ = std::nullopt;
  }

  ffi::Optional<ExecScope> root_ = std::nullopt;
  std::vector<ExecScope> scope_stack_;
};

class ScopeIdVerifier : public Verifier<ScopeIdVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    const auto& scope = op->exec_scope;
    auto it = scope_id_def_.end();
    scope_id_def_.insert(it, scope->scope_id_def.begin(), scope->scope_id_def.end());
    Verifier::VisitStmt_(op, path);
    if (!scope->scope_id_def.empty()) {
      ScopeIdDefVerifier verifier;
      // Relaxed: PrimFunc construction allows deferred (extent=NullOpt) defs.
      // Strict resolution is enforced later at LowerTIRx entry.
      Verify(verifier.Verify(scope_id_def_, ScopeIdDefVerifier::Mode::kRelaxed))
          << "TIRxError: Scope at " << path << " has invalid scope_id_def";
      // At kernel scope, enforce launch-parameter sanity. The thread count
      // (kCtaThread) must be positive; if the kernel uses any warp-granular
      // binding (warp_id / lane_id / warpgroup_id / warp_id_in_wg), it must
      // additionally be a multiple of warp size 32. Pure thread-flat kernels
      // (only kCtaThread declared, e.g. single-thread tests) are unconstrained.
      // When kCtaThread is deferred and not yet resolvable from siblings,
      // skip the sanity check -- LowerTIRx will catch unresolved cases.
      if (scope->kind == ScopeKind::kKernel) {
        auto cta_thread_it = verifier.id_set.find(ScopeBinding::kCtaThread);
        if (cta_thread_it != verifier.id_set.end() && !(*cta_thread_it).second.is_deferred()) {
          PrimExpr ext = (*cta_thread_it).second.fused_extent();
          if (const auto* imm = ext.as<IntImmNode>()) {
            Verify(imm->value > 0) << "TIRxError: kernel at " << path
                                   << " has non-positive thread count " << imm->value;
            bool needs_warp_align = verifier.id_set.count(ScopeBinding::kCtaWarp) ||
                                    verifier.id_set.count(ScopeBinding::kWarpThread) ||
                                    verifier.id_set.count(ScopeBinding::kCtaWarpgroup) ||
                                    verifier.id_set.count(ScopeBinding::kWarpgroupWarp);
            if (needs_warp_align) {
              Verify(imm->value % 32 == 0)
                  << "TIRxError: kernel at " << path << " uses warp-granular bindings"
                  << " but has thread count " << imm->value << " not a multiple of 32";
            }
          }
        }
      }
    }
    scope_id_def_.erase(scope_id_def_.end() - scope->scope_id_def.size(), scope_id_def_.end());
  }

  Array<ScopeIdDef> scope_id_def_;
  arith::Analyzer ana_;
};

class LayoutVerifier : public Verifier<LayoutVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlock is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const SBlockRealizeNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlockRealize is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    // Check buffer layouts in alloc_buffers that appear as AllocBuffer stmts
    Verifier::VisitStmt_(op, path);
  }
};

class AsyncStructsVerifier : public Verifier<AsyncStructsVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlock is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const SBlockRealizeNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlockRealize is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    scope_stack_.push_back(op->exec_scope);
    Verifier::VisitStmt_(op, path);
    scope_stack_.pop_back();
  }

  std::vector<ExecScope> scope_stack_;
};

class DeviceFuncVerifier : public Verifier<DeviceFuncVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlock is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const SBlockRealizeNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlockRealize is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const ExecScopeStmtNode* op, ffi::reflection::AccessPath path) override {
    if (!inside_root_scope_) {
      // At the top level: only one root scope is allowed
      Verify(!root_.has_value()) << "TIRxError: Only one root scope is allowed in device function";
      root_ = op->exec_scope;
      Verify(ScopeKindHigher(ScopeKind::kKernel, root_.value()->kind))
          << "TIRxError: Root scope of device function at " << path
          << " is higher than kernel scope";
      inside_root_scope_ = true;
      Verifier::VisitStmt_(op, path);
      inside_root_scope_ = false;
    } else {
      // Already inside a root scope: nested scopes are allowed
      Verifier::VisitStmt_(op, path);
    }
  }

  ffi::Optional<ExecScope> root_ = std::nullopt;
  bool inside_root_scope_ = false;
};

bool VerifyTIRxWellFormed(const PrimFunc& func, bool assert_mode, bool device_func) {
  if (!ExecScopeVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (!ScopeIdVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (!LayoutVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (!AsyncStructsVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (device_func) {
    if (!DeviceFuncVerifier::Verify(func, assert_mode)) {
      return false;
    }
  }
  return true;
}

bool VerifyTIRxWellFormed(const IRModule& mod, bool assert_mode, bool device_func) {
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto prim_func = base_func.as<PrimFunc>()) {
      // s_tir=True PrimFuncs use s_tir semantics — defer to VerifyWellFormed.
      if (prim_func.value()->attrs->dict.count(tvm::attr::kSTir)) {
        if (!VerifyWellFormed(prim_func.value(), assert_mode)) return false;
        continue;
      }
      bool res = VerifyTIRxWellFormed(prim_func.value(), assert_mode, device_func);
      if (!res) {
        return false;
      }
    }
  }
  return true;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.analysis.VerifyTIRxWellFormed",
                        [](const ffi::ObjectRef& obj, bool assert_mode, bool device_func) {
                          if (auto n = obj.as<PrimFunc>()) {
                            return VerifyTIRxWellFormed(n.value(), assert_mode, device_func);
                          } else if (auto n = obj.as<IRModule>()) {
                            return VerifyTIRxWellFormed(n.value(), assert_mode, device_func);
                          } else {
                            LOG(FATAL) << "Expects PrimFunc or IRModule,  but get "
                                       << obj->GetTypeKey() << " instead.";
                            return false;
                          }
                        });
}
}  // namespace tirx
}  // namespace tvm
