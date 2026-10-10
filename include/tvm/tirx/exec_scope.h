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
 * \file tvm/tirx/block_scope.h
 * \brief Definition of execution scope
 */

#ifndef TVM_TIRX_EXEC_SCOPE_H_
#define TVM_TIRX_EXEC_SCOPE_H_

#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>

#include <string>
#include <utility>

namespace tvm {
namespace tirx {

/*!
 * \brief The target execution scope kind of a tile primitive call.
 *
 * Identifies the granularity at which an op executes (the per-call
 * ``scope`` on tensor Call attributes, e.g. ``T.cuda.tile.ld(..., scope="warp")``).
 * Ordered from coarsest to finest; smaller integer = wider scope, so
 * ``ScopeKindHigher`` is a plain ``<``.
 */
enum class ScopeKind : int {
  kCluster = 2,
  kCta = 3,
  kWarpgroup = 4,
  kWarp = 5,
  kThread = 6,
};

/*! \brief Convert a ScopeKind to its string name (e.g. kThread -> "thread"). */
TVM_DLL std::string ScopeKindToString(ScopeKind kind);

/*! \brief Parse a string name to a ScopeKind. FATAL if unknown. */
TVM_DLL ScopeKind StringToScopeKind(const ffi::String& name);

/*!
 * \brief The binding between a parent scope and a child scope as used by a
 * CUDA index call. The closed enum of valid (parent -> cur) pairs.
 *
 * Single-axis bindings (target one ActiveSet box axis -- ``laneid`` /
 * ``warpid`` / ``cta_id``, possibly via a warpid factor lane):
 *   kKernelCta, kClusterCta -> cta_id (flat)
 *   kCtaWarp                -> warpid (flat)
 *   kCtaWarpgroup           -> warpid (outer factor; warpgroup index)
 *   kWarpgroupWarp          -> warpid (inner factor; warp-within-wg index)
 *   kWarpThread             -> laneid (flat)
 *   kKernelCluster          -> not a filter target (cluster_id by design)
 *   kClusterCtaPair         -> hardware CTA pair id (cluster CTA rank % 2)
 *
 * Multi-axis (flat-thread) bindings -- linearize across two ActiveSet
 * axes; a flat ``lo <= var and var < hi`` predicate cannot narrow them as a
 * contiguous box range, so they fall back to plain predicate semantics:
 *   kCtaThread       -> threadIdx.x within a CTA          (laneid * warpid)
 *   kWarpgroupThread -> threadIdx.x within a warpgroup    (laneid * wid_in_wg)
 */
enum class ScopeBinding : int {
  kKernelCluster = 0,
  kKernelCta = 1,
  kClusterCta = 2,
  kCtaWarpgroup = 3,
  kCtaWarp = 4,
  kWarpgroupWarp = 5,
  kWarpThread = 6,
  kCtaThread = 7,
  kWarpgroupThread = 8,
  kClusterCtaPair = 9,
};

/*! \brief Convert a ScopeBinding to its (parent, cur) string pair. */
TVM_DLL std::pair<ffi::String, ffi::String> ScopeBindingToStringPair(ScopeBinding binding);

/*! \brief Parse a (parent, cur) string pair to a ScopeBinding. FATAL if unknown. */
TVM_DLL ScopeBinding StringPairToScopeBinding(const ffi::String& parent, const ffi::String& cur);

/*!
 * \brief Strict-weak "a is wider than b" on scope kinds: ``world > kernel >
 * cluster > cta > warpgroup > warp > thread``. Only used by axe-layout
 * scope-chain validity (the rest of the codebase compares scope identities
 * with ==).
 */
inline bool ScopeKindHigher(ScopeKind a, ScopeKind b) {
  return static_cast<int>(a) < static_cast<int>(b);
}

/*! \brief String-keyed convenience over ScopeKindHigher. FATALs on bad name. */
TVM_DLL bool ScopeNameHigher(const ffi::String& a, const ffi::String& b);

/******** Definition of Execution Scope ********/
class ExecScopeNode : public ffi::Object {
 public:
  /*! \brief scope identity; one of the closed ScopeKind values. */
  ScopeKind kind = ScopeKind::kThread;

  /*! \brief Human-readable name derived from ``kind`` (for printing / errors). */
  ffi::String name() const { return ScopeKindToString(kind); }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExecScopeNode>().def_ro("kind", &ExecScopeNode::kind);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.ExecScope", ExecScopeNode, ffi::Object);
};

class ExecScope : public ffi::ObjectRef {
 public:
  /*! \brief Construct from a ScopeKind (canonical). */
  TVM_DLL explicit ExecScope(ScopeKind kind);
  /*! \brief Construct from a name string (FATALs on unknown name). */
  TVM_DLL explicit ExecScope(const ffi::String& name) : ExecScope(StringToScopeKind(name)) {}

  explicit ExecScope(ffi::ObjectPtr<ExecScopeNode> node) : ffi::ObjectRef(std::move(node)) {
    TVM_FFI_ICHECK(data_ != nullptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExecScope, ffi::ObjectRef, ExecScopeNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_SCOPE_H_
