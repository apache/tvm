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
    Verify(false) << "TIRxError: SBlock is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const SBlockRealizeNode* op, ffi::reflection::AccessPath path) override {
    Verify(false) << "TIRxError: SBlockRealize is not allowed in tirx=True mode at " << path;
  }

  void VisitStmt_(const tirx::TilePrimitiveCallNode* op,
                  ffi::reflection::AccessPath path) override {
    static const tvm::OpAttrMap<bool>& tirx_op_map_ = Op::GetAttrMap<bool>("TIsTIRxOp");
    Verify(tirx_op_map_.count(op->op))
        << "TIRxError: TilePrimitiveCall at " << path << " has unknown TIRX op " << op->op;
  }
};

class ScopeIdVerifier : public Verifier<ScopeIdVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void VisitStmt_(const AttrStmtNode* op, ffi::reflection::AccessPath path) override {
    if (op->attr_key == tvm::tirx::attr::kDeviceEntry) {
      // Device-region marker: defs gathered from the body are verified when
      // the AttrStmt exits, with launch-param sanity enforced as ``is_root``.
      size_t baseline = scope_id_def_.size();
      Verifier::VisitStmt_(op, path);
      size_t total = scope_id_def_.size();
      if (total > baseline) {
        RunScopeIdVerify(path, baseline, /*is_root=*/true);
      }
      while (scope_id_def_.size() > baseline) {
        scope_id_def_.pop_back();
      }
      return;
    }
    Verifier::VisitStmt_(op, path);
  }

  void RunScopeIdVerify(ffi::reflection::AccessPath path, size_t baseline, bool is_root) {
    ScopeIdDefVerifier verifier;
    Verify(verifier.Verify(scope_id_def_, ScopeIdDefVerifier::Mode::kRelaxed))
        << "TIRxError: Scope at " << path << " has invalid scope_id_def";
    if (is_root) {
      // Enforce launch-parameter sanity at the device-region root.
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

  void VisitStmt_(const ScopeIdDefStmtNode* op, ffi::reflection::AccessPath path) override {
    scope_id_def_.push_back(op->def);
    Verifier::VisitStmt_(op, path);
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
