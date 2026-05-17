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
 * \file tvm/tirx/exec_context.h
 * \brief Compile-time ExecContext state: the active thread set ``A`` as a
 * TileLayout and the (inter, intra) split under the current scope kind,
 * threaded through the IR walker so per-op lowerers see the precise execution
 * shape at each site.
 *
 * Mirrors the pure-Python implementation in python/tvm/tirx/exec_context.py.
 */
#ifndef TVM_TIRX_EXEC_CONTEXT_H_
#define TVM_TIRX_EXEC_CONTEXT_H_

#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/var.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tirx {

/*! \brief Warpgroup size in warps (hardware-fixed). */
constexpr int kWgSize = 4;

/*! \brief Active slice offset + stride * [0, extent) encoded on one TileLayout axis. */
struct AxisRange {
  PrimExpr extent;
  PrimExpr offset;
  PrimExpr stride;

  /*! \brief Intersect with [lo, hi). Returns false if the result is empty. */
  bool Intersect(int64_t lo, int64_t hi, AxisRange* out) const;

  /*! \brief Intersect with values satisfying axis % modulus == residue. */
  bool Modulo(int64_t modulus, int64_t residue, AxisRange* out) const;
};

/*!
 * \brief Active thread set A.
 * The source of truth is ``layout``:
 *   shard  = active axes with extents
 *   offset = per-axis lower bound, possibly a selector PrimExpr
 */
struct ActiveSet {
  TileLayout layout;

  int64_t size() const;
  bool GetAxis(const std::string& axis, AxisRange* out) const;
  bool HasAxis(const std::string& axis) const;
  ActiveSet WithAxis(const std::string& axis, const AxisRange& range) const;
  std::vector<std::string> AxisNames() const;
};

/*!
 * \brief One scope_switch split. Fields are sparse dicts keyed by active-set
 * axis name, e.g. laneid/warpid/cta_id/wid_in_wg/wgid or factorized CTA axes
 * such as cbx/cby/cbz. An empty map denotes the empty layout (e.g. intra under
 * scope_kind=thread).
 */
struct ExecSplit {
  std::unordered_map<std::string, AxisRange> inter;
  std::unordered_map<std::string, AxisRange> intra;
};

/*! \brief Initial A at T.kernel() entry: all threads active, offsets zero. */
TVM_DLL ActiveSet InitialActiveSet(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext);
TVM_DLL ActiveSet InitialActiveSet(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext,
                                   const std::vector<std::pair<std::string, int64_t>>& cta_axes);

/*!
 * \brief Narrow A on the lane bound to ``binding``.
 *
 * The ScopeBinding maps directly to which native axis (laneid/warpid/cta_id)
 * to narrow, and for warpid whether to narrow the full axis (kCtaWarp), the
 * outer factor (kCtaWarpgroup), or the inner factor (kWarpgroupWarp).
 *
 * Bindings with no single-lane representation are conservative: cluster_id is
 * not a filter target; flat thread ids are accepted only when the range can be
 * represented as a rectangular lane/warp active set.
 */
TVM_DLL bool FilterNarrow(const ActiveSet& A, ScopeBinding binding, int64_t lo, int64_t hi,
                          ActiveSet* out, std::string* err);

/*!
 * \brief Factor A into (inter, intra) for target scope_kind.
 *
 * Returns false on factoring failure (warpgroup with warpid lane that
 * crosses a warpgroup boundary unaligned) and writes reason to *err.
 */
TVM_DLL bool ScopeSwitch(const ActiveSet& A, ScopeKind scope_kind, ExecSplit* out,
                         std::string* err);

/*! \brief Per-program-point ExecContext: active set + scope kind + split. */
struct ExecContext {
  ActiveSet A;
  ScopeKind scope_kind = ScopeKind::kKernel;
  ExecSplit split;  // (inter, intra) of current A under current scope_kind

  /*! \brief Kernel-entry ctor. */
  static ExecContext AtKernelEntry(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext);
  static ExecContext AtKernelEntry(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext,
                                   const std::vector<std::pair<std::string, int64_t>>& cta_axes);

  /*! \brief Apply filter; scope_kind preserved, split recomputed. */
  bool WithFilter(ScopeBinding binding, int64_t lo, int64_t hi, ExecContext* out,
                  std::string* err) const;

  /*! \brief Apply a unique-value selector filter on one scope id Var. */
  bool WithSelector(ScopeBinding binding, PrimExpr selector, ExecContext* out,
                    std::string* err) const;

  /*! \brief Apply filter on a factorized CTA axis such as cbx/cby/cbz. */
  bool WithCtaAxisFilter(const std::string& axis, int64_t lo, int64_t hi, ExecContext* out,
                         std::string* err) const;

  /*! \brief Apply modulo filter on a factorized CTA axis such as cbx/cby/cbz. */
  bool WithCtaAxisModulo(const std::string& axis, int64_t modulus, int64_t residue,
                         ExecContext* out, std::string* err) const;

  /*! \brief Apply scope_switch; A preserved, split recomputed for new scope_kind. */
  bool WithScopeSwitch(ScopeKind new_scope_kind, ExecContext* out, std::string* err) const;
};

/*!
 * \brief Encode one side of an ExecSplit (inter or intra) as the FFI map used
 * by ``DispatchContextNode::{inter, intra}``: axis name -> [extent, offset]
 * for unit-stride axes, or [extent, offset, stride] for strided axes.
 */
TVM_DLL ffi::Map<ffi::String, ffi::Array<PrimExpr>> EncodeSplitSide(
    const std::unordered_map<std::string, AxisRange>& side);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_EXEC_CONTEXT_H_
