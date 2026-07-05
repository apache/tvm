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

#include <tvm/ffi/container/variant.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/var.h>

#include <string>
#include <utility>

namespace tvm {
namespace tirx {

/*!
 * \brief The target execution scope kind of a tile primitive call.
 *
 * Identifies the granularity at which an op executes (the per-call
 * ``scope`` on a ``TilePrimitiveCall``, e.g. ``Tx.warp.copy(...)``).
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
 * `ScopeIdDef`. The closed enum of valid (parent -> cur) pairs.
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

/******** Definition of ScopeId ********/
class ScopeIdDefNode : public ffi::Object {
 public:
  /*! \brief The ScopeId defined */
  ffi::Array<PrimVar> def_ids;
  /*!
   * \brief The extents of the ScopeId.
   *
   * NullOpt means the extent is *deferred*: the user wrote e.g.
   * ``bx = T.cta_id()`` without specifying the extent, and the value will be
   * inferred from sibling ScopeIdDefs at LowerTIRx entry via the verifier's
   * BFS closure. Deferred form requires ``def_ids.size() == 1`` (single axis
   * only -- multi-axis defers have no well-defined recovery).
   *
   * Explicit (Some) form preserves the per-axis shape, e.g. ``[3, 4, 5]``
   * for ``T.cta_id([3, 4, 5])``.
   */
  ffi::Optional<ffi::Array<PrimExpr>> extents;
  /*! \brief The (parent, cur) binding of this scope id as a closed enum. */
  ScopeBinding scope;
  /*!
   * \brief Optional preferred extents (cluster→cta only).
   * Maps to cudaLaunchAttributePreferredClusterDimension (CUDA 12.8+).
   */
  ffi::Optional<ffi::Array<PrimExpr>> preferred_extents;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScopeIdDefNode>()
        .def_ro("def_ids", &ScopeIdDefNode::def_ids, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("extents", &ScopeIdDefNode::extents)
        .def_ro("scope", &ScopeIdDefNode::scope)
        .def_ro("preferred_extents", &ScopeIdDefNode::preferred_extents);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.ScopeIdDef", ScopeIdDefNode, ffi::Object);
};

class ScopeIdDef : public ffi::ObjectRef {
 public:
  TVM_DLL explicit ScopeIdDef(ffi::Array<PrimVar> def_ids,
                              ffi::Optional<ffi::Array<PrimExpr>> extents, ScopeBinding scope,
                              ffi::Optional<ffi::Array<PrimExpr>> preferred_extents =
                                  ffi::Optional<ffi::Array<PrimExpr>>(std::nullopt));

  /*! \brief Whether this def has a deferred (unknown) extent. */
  bool is_deferred() const { return !get()->extents.has_value(); }

  /*! \brief Product of all extent dimensions. PRECONDITION: !is_deferred(). */
  PrimExpr fused_extent() const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ScopeIdDef, ffi::ObjectRef, ScopeIdDefNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScopeIdDefNode);
};

class ScopeIdDefVerifier {
 public:
  using ScopeIdSet = std::unordered_map<ScopeBinding, ScopeIdDef>;

  /*!
   * \brief Verification mode.
   *
   * - kRelaxed: tolerate deferred (extent=None) ScopeIdDefs. Used for partial
   *   programs in the well-formedness check at PrimFunc construction time.
   * - kStrict: every original ScopeIdDef must end with a resolved extent
   *   (either explicit at construction, or inferred via closure). Used at
   *   LowerTIRx entry where downstream resolve/codegen needs concrete values.
   */
  enum class Mode { kRelaxed, kStrict };

  /*! \brief Verify the scope id definitions are well formed. */
  bool Verify(const ffi::Array<ScopeIdDef>& defs, Mode mode = Mode::kStrict);

  /*!
   * \brief The resolved scope id set; ``id_set[binding]`` is the best-known
   * def for that binding (extents filled in from closure when possible).
   */
  ScopeIdSet id_set;
};

/*!
 * \brief Static resolver for ScopeIdDef values. Replaces the former
 * ScopeIdResolveTable runtime registry with a closed-enum switch.
 */
class ScopeIdResolve {
 public:
  using LaunchParams = std::unordered_map<ffi::String, IterVar>;

  /*! \brief Resolve a ScopeIdDef for a given canonical binding + target. */
  TVM_DLL static ffi::Array<PrimExpr> Resolve(ScopeBinding binding,
                                              const ffi::Optional<ffi::Array<PrimExpr>>& extents,
                                              int out_dim, const ffi::String& target_kind,
                                              const LaunchParams& params);

  /*! \brief Compute the warp_id_in_cta shuffle expression from threadIdx in launch params */
  TVM_DLL static PrimExpr ComputeWarpIdInCta(const LaunchParams& params);
};

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

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ExecScope, ffi::ObjectRef, ExecScopeNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_SCOPE_H_
