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
 * \file exec_context.cc
 * \brief Compile-time active-thread state backed by TileLayout.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/exec_context.h>
#include <tvm/tirx/expr.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <utility>

namespace tvm {
namespace tirx {

namespace {

constexpr int kWarpSize = 32;

PrimExpr I64(int64_t value) { return IntImm(DataType::Int(64), value); }

AxisRange MakeRange(int64_t extent, int64_t offset = 0, int64_t stride = 1) {
  return AxisRange{I64(extent), I64(offset), I64(stride)};
}

bool TryAsInt64(const PrimExpr& expr, int64_t* value) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    *value = imm->value;
    return true;
  }
  return false;
}

bool IsZero(const PrimExpr& expr) {
  arith::Analyzer analyzer;
  return analyzer.CanProveEqual(expr, 0);
}

ActiveSet MakeActiveSet(const std::vector<std::pair<std::string, AxisRange>>& axes) {
  ffi::Array<Iter> shard;
  ffi::Map<Axis, PrimExpr> offset;
  for (const auto& [name, range] : axes) {
    Axis axis = Axis::Get(name);
    shard.push_back(Iter(range.extent, range.stride, axis));
    if (!IsZero(range.offset)) {
      offset.Set(axis, range.offset);
    }
  }
  return ActiveSet{TileLayout(shard, {}, offset)};
}

std::vector<std::pair<std::string, AxisRange>> AxisRanges(const ActiveSet& A) {
  std::vector<std::pair<std::string, AxisRange>> axes;
  for (const auto& iter : A.layout->shard) {
    AxisRange range;
    TVM_FFI_ICHECK(A.GetAxis(iter->axis->name.operator std::string(), &range));
    axes.push_back({iter->axis->name.operator std::string(), range});
  }
  return axes;
}

bool NarrowAxis(const ActiveSet& A, const std::string& axis, int64_t lo, int64_t hi, ActiveSet* out,
                std::string* err) {
  AxisRange cur;
  if (!A.GetAxis(axis, &cur)) {
    *err = "unknown active-set axis: " + axis;
    return false;
  }
  AxisRange narrowed;
  if (!cur.Intersect(lo, hi, &narrowed)) {
    *err = "filter produces empty or non-structural active-set range on axis " + axis;
    return false;
  }
  *out = A.WithAxis(axis, narrowed);
  return true;
}

bool ModuloAxis(const ActiveSet& A, const std::string& axis, int64_t modulus, int64_t residue,
                ActiveSet* out, std::string* err) {
  AxisRange cur;
  if (!A.GetAxis(axis, &cur)) {
    *err = "unknown active-set axis: " + axis;
    return false;
  }
  AxisRange narrowed;
  if (!cur.Modulo(modulus, residue, &narrowed)) {
    *err = "modulo filter produces empty or non-structural active-set slice on axis " + axis;
    return false;
  }
  *out = A.WithAxis(axis, narrowed);
  return true;
}

void AddCtaAxes(const ActiveSet& A, std::unordered_map<std::string, AxisRange>* side) {
  AxisRange cta_id;
  if (A.GetAxis("cta_id", &cta_id)) {
    (*side)["cta_id"] = cta_id;
    return;
  }
  for (const std::string& axis : A.AxisNames()) {
    if (axis == "laneid" || axis == "warpid") continue;
    AxisRange range;
    TVM_FFI_ICHECK(A.GetAxis(axis, &range));
    (*side)[axis] = range;
  }
}

// Factor warpid into (wid_in_wg, wgid). Returns false on case 3 or symbolic offset.
bool FactorWarpid(const AxisRange& wp, AxisRange* wid_in_wg, AxisRange* wgid) {
  int64_t off = 0;
  int64_t ext = 0;
  int64_t stride = 0;
  if (!TryAsInt64(wp.offset, &off) || !TryAsInt64(wp.extent, &ext) ||
      !TryAsInt64(wp.stride, &stride) || stride != 1) {
    return false;
  }
  int64_t wid_off = off % kWgSize;
  int64_t wgid_off = off / kWgSize;

  if (wid_off == 0 && ext % kWgSize == 0) {
    *wid_in_wg = MakeRange(kWgSize, 0);
    *wgid = MakeRange(ext / kWgSize, wgid_off);
    return true;
  }
  if (ext <= kWgSize - wid_off) {
    *wid_in_wg = MakeRange(ext, wid_off);
    *wgid = MakeRange(1, wgid_off);
    return true;
  }
  return false;
}

int64_t FloorDivInt(int64_t a, int64_t b);
int64_t CeilDivInt(int64_t a, int64_t b);

bool SameIntRange(const AxisRange& lhs, const AxisRange& rhs) {
  int64_t lhs_ext = 0;
  int64_t lhs_off = 0;
  int64_t lhs_stride = 0;
  int64_t rhs_ext = 0;
  int64_t rhs_off = 0;
  int64_t rhs_stride = 0;
  return TryAsInt64(lhs.extent, &lhs_ext) && TryAsInt64(lhs.offset, &lhs_off) &&
         TryAsInt64(lhs.stride, &lhs_stride) && TryAsInt64(rhs.extent, &rhs_ext) &&
         TryAsInt64(rhs.offset, &rhs_off) && TryAsInt64(rhs.stride, &rhs_stride) &&
         lhs_ext == rhs_ext && lhs_off == rhs_off && lhs_stride == rhs_stride;
}

bool NarrowFlatProductRange(const AxisRange& major, const AxisRange& lane, int64_t lo, int64_t hi,
                            AxisRange* new_major, AxisRange* new_lane, std::string* err) {
  int64_t major_off = 0;
  int64_t major_ext = 0;
  int64_t major_stride = 0;
  int64_t lane_off = 0;
  int64_t lane_ext = 0;
  int64_t lane_stride = 0;
  if (!TryAsInt64(major.offset, &major_off) || !TryAsInt64(major.extent, &major_ext) ||
      !TryAsInt64(major.stride, &major_stride) || !TryAsInt64(lane.offset, &lane_off) ||
      !TryAsInt64(lane.extent, &lane_ext) || !TryAsInt64(lane.stride, &lane_stride) ||
      major_ext <= 0 || lane_ext <= 0 || major_stride <= 0 || lane_stride <= 0) {
    *err = "flat thread range requires structural lane and warp axes";
    return false;
  }

  int64_t active_min = major_off * kWarpSize + lane_off;
  int64_t active_max = (major_off + major_stride * (major_ext - 1)) * kWarpSize +
                       (lane_off + lane_stride * (lane_ext - 1)) + 1;
  if (lo <= active_min && active_max <= hi) {
    *new_major = major;
    *new_lane = lane;
    return true;
  }

  if (major_stride != 1 || lane_stride != 1) {
    *err = "flat thread range narrowing requires unit-stride lane and warp axes";
    return false;
  }

  int64_t lane_hi = lane_off + lane_ext;
  int64_t major_hi = major_off + major_ext;
  int64_t hit_lo = std::max(major_off, FloorDivInt(lo - lane_hi, kWarpSize) + 1);
  int64_t hit_hi = std::min(major_hi, CeilDivInt(hi - lane_off, kWarpSize));
  if (hit_hi <= hit_lo) {
    *err = "flat thread range produces empty active set";
    return false;
  }

  if (hit_hi == hit_lo + 1) {
    int64_t m = hit_lo;
    int64_t new_lane_lo = std::max(lane_off, lo - m * kWarpSize);
    int64_t new_lane_hi = std::min(lane_hi, hi - m * kWarpSize);
    if (new_lane_hi <= new_lane_lo) {
      *err = "flat thread range produces empty lane range";
      return false;
    }
    *new_major = MakeRange(1, m);
    *new_lane = MakeRange(new_lane_hi - new_lane_lo, new_lane_lo);
    return true;
  }

  if (lo <= hit_lo * kWarpSize + lane_off && (hit_hi - 1) * kWarpSize + lane_hi <= hi) {
    *new_major = MakeRange(hit_hi - hit_lo, hit_lo);
    *new_lane = lane;
    return true;
  }

  *err = "flat thread range would require a non-rectangular lane/warp active set";
  return false;
}

bool NarrowFlatCtaThreadRange(const ActiveSet& A, int64_t lo, int64_t hi, ActiveSet* out,
                              std::string* err) {
  AxisRange lane;
  AxisRange warpid;
  if (!A.GetAxis("laneid", &lane) || !A.GetAxis("warpid", &warpid)) {
    *err = "active set has no laneid/warpid axes";
    return false;
  }
  AxisRange new_lane;
  AxisRange new_warpid;
  if (!NarrowFlatProductRange(warpid, lane, lo, hi, &new_warpid, &new_lane, err)) {
    return false;
  }
  *out = A.WithAxis("laneid", new_lane).WithAxis("warpid", new_warpid);
  return true;
}

bool NarrowFlatWarpgroupThreadRange(const ActiveSet& A, int64_t lo, int64_t hi, ActiveSet* out,
                                    std::string* err) {
  AxisRange lane;
  AxisRange warpid;
  if (!A.GetAxis("laneid", &lane) || !A.GetAxis("warpid", &warpid)) {
    *err = "active set has no laneid/warpid axes";
    return false;
  }
  AxisRange wid_in_wg;
  AxisRange wgid;
  if (!FactorWarpid(warpid, &wid_in_wg, &wgid)) {
    *err = "filter on flat warpgroup-thread range requires factorable warpid axis";
    return false;
  }

  AxisRange new_lane;
  AxisRange new_wid_in_wg;
  if (!NarrowFlatProductRange(wid_in_wg, lane, lo, hi, &new_wid_in_wg, &new_lane, err)) {
    return false;
  }

  int64_t wgid_ext = 0;
  int64_t wgid_off = 0;
  if (!TryAsInt64(wgid.extent, &wgid_ext) || !TryAsInt64(wgid.offset, &wgid_off)) {
    *err = "filter on flat warpgroup-thread range requires structural warpgroup id";
    return false;
  }
  if (wgid_ext != 1) {
    if (SameIntRange(new_lane, lane) && SameIntRange(new_wid_in_wg, wid_in_wg)) {
      *out = A;
      return true;
    }
    *err = "flat warpgroup-thread range across multiple warpgroups is not representable";
    return false;
  }

  int64_t wid_ext = 0;
  int64_t wid_off = 0;
  if (!TryAsInt64(new_wid_in_wg.extent, &wid_ext) || !TryAsInt64(new_wid_in_wg.offset, &wid_off)) {
    *err = "filter on flat warpgroup-thread range requires structural warp id";
    return false;
  }
  *out = A.WithAxis("laneid", new_lane)
             .WithAxis("warpid", MakeRange(wid_ext, wgid_off * kWgSize + wid_off));
  return true;
}

int64_t FloorDivInt(int64_t a, int64_t b) {
  TVM_FFI_ICHECK_GT(b, 0);
  if (a >= 0) return a / b;
  return -static_cast<int64_t>((static_cast<uint64_t>(-a) + b - 1) / b);
}

int64_t CeilDivInt(int64_t a, int64_t b) { return -FloorDivInt(-a, b); }

int64_t NormalizeMod(int64_t value, int64_t modulus) {
  int64_t ret = value % modulus;
  if (ret < 0) ret += modulus;
  return ret;
}

int64_t ExtendedGcd(int64_t a, int64_t b, int64_t* x, int64_t* y) {
  if (b == 0) {
    *x = 1;
    *y = 0;
    return a;
  }
  int64_t x1 = 0;
  int64_t y1 = 0;
  int64_t g = ExtendedGcd(b, a % b, &x1, &y1);
  *x = y1;
  *y = x1 - (a / b) * y1;
  return g;
}

int64_t ModularInverse(int64_t value, int64_t modulus) {
  int64_t x = 0;
  int64_t y = 0;
  int64_t g = ExtendedGcd(NormalizeMod(value, modulus), modulus, &x, &y);
  TVM_FFI_ICHECK_EQ(g, 1);
  return NormalizeMod(x, modulus);
}

}  // namespace

bool AxisRange::Intersect(int64_t lo, int64_t hi, AxisRange* out) const {
  int64_t cur_off = 0;
  int64_t cur_ext = 0;
  int64_t cur_stride = 0;
  if (!TryAsInt64(offset, &cur_off) || !TryAsInt64(extent, &cur_ext) ||
      !TryAsInt64(stride, &cur_stride) || cur_stride <= 0) {
    return false;
  }
  int64_t i_lo = std::max<int64_t>(0, CeilDivInt(lo - cur_off, cur_stride));
  int64_t i_hi = std::min<int64_t>(cur_ext, FloorDivInt(hi - 1 - cur_off, cur_stride) + 1);
  if (i_hi <= i_lo) return false;
  out->extent = I64(i_hi - i_lo);
  out->offset = I64(cur_off + cur_stride * i_lo);
  out->stride = I64(cur_stride);
  return true;
}

bool AxisRange::Modulo(int64_t modulus, int64_t residue, AxisRange* out) const {
  if (modulus <= 0) return false;
  int64_t cur_off = 0;
  int64_t cur_ext = 0;
  int64_t cur_stride = 0;
  if (!TryAsInt64(offset, &cur_off) || !TryAsInt64(extent, &cur_ext) ||
      !TryAsInt64(stride, &cur_stride) || cur_stride <= 0) {
    return false;
  }
  residue = NormalizeMod(residue, modulus);
  int64_t rhs = NormalizeMod(residue - cur_off, modulus);
  int64_t g = std::gcd(std::llabs(cur_stride), std::llabs(modulus));
  if (rhs % g != 0) return false;
  int64_t reduced_stride = cur_stride / g;
  int64_t reduced_rhs = rhs / g;
  int64_t reduced_modulus = modulus / g;
  int64_t period = reduced_modulus;
  int64_t i0 =
      NormalizeMod(reduced_rhs * ModularInverse(reduced_stride, reduced_modulus), reduced_modulus);
  if (i0 >= cur_ext) return false;
  int64_t new_ext = (cur_ext - 1 - i0) / period + 1;
  out->extent = I64(new_ext);
  out->offset = I64(cur_off + cur_stride * i0);
  out->stride = I64(cur_stride * period);
  return true;
}

bool ActiveSet::GetAxis(const std::string& axis, AxisRange* out) const {
  if (!layout.defined()) return false;
  for (const auto& iter : layout->shard) {
    if (iter->axis->name != axis) continue;
    PrimExpr off = I64(0);
    for (const auto& kv : layout->offset) {
      if (kv.first->name == axis) {
        off = kv.second;
        break;
      }
    }
    *out = AxisRange{iter->extent, off, iter->stride};
    return true;
  }
  return false;
}

bool ActiveSet::HasAxis(const std::string& axis) const {
  AxisRange ignored;
  return GetAxis(axis, &ignored);
}

ActiveSet ActiveSet::WithAxis(const std::string& axis, const AxisRange& range) const {
  std::vector<std::pair<std::string, AxisRange>> axes = AxisRanges(*this);
  bool found = false;
  for (auto& entry : axes) {
    if (entry.first == axis) {
      entry.second = range;
      found = true;
      break;
    }
  }
  TVM_FFI_ICHECK(found) << "Internal Error: unknown active-set axis " << axis;
  return MakeActiveSet(axes);
}

std::vector<std::string> ActiveSet::AxisNames() const {
  std::vector<std::string> names;
  if (!layout.defined()) return names;
  for (const auto& iter : layout->shard) {
    names.push_back(iter->axis->name.operator std::string());
  }
  return names;
}

int64_t ActiveSet::size() const {
  int64_t size = 1;
  for (const auto& iter : layout->shard) {
    int64_t extent = 0;
    if (!TryAsInt64(iter->extent, &extent)) return 0;
    size *= extent;
  }
  return size;
}

ActiveSet InitialActiveSet(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext) {
  return InitialActiveSet(lane_ext, warp_ext, cta_ext, {});
}

ActiveSet InitialActiveSet(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext,
                           const std::vector<std::pair<std::string, int64_t>>& cta_axes) {
  std::vector<std::pair<std::string, AxisRange>> axes = {{"laneid", MakeRange(lane_ext)},
                                                         {"warpid", MakeRange(warp_ext)}};
  if (cta_axes.empty()) {
    axes.push_back({"cta_id", MakeRange(cta_ext)});
  } else {
    for (const auto& [axis, extent] : cta_axes) {
      axes.push_back({axis, MakeRange(extent)});
    }
  }
  return MakeActiveSet(axes);
}

bool FilterNarrow(const ActiveSet& A, ScopeBinding binding, int64_t lo, int64_t hi, ActiveSet* out,
                  std::string* err) {
  if (lo >= hi) {
    *err = "filter range is empty or inverted";
    return false;
  }

  switch (binding) {
    case ScopeBinding::kWarpThread:
      return NarrowAxis(A, "laneid", lo, hi, out, err);
    case ScopeBinding::kCtaWarp:
      return NarrowAxis(A, "warpid", lo, hi, out, err);
    case ScopeBinding::kKernelCta:
    case ScopeBinding::kClusterCta:
      return NarrowAxis(A, "cta_id", lo, hi, out, err);
    case ScopeBinding::kCtaWarpgroup: {
      AxisRange wp;
      if (!A.GetAxis("warpid", &wp)) {
        *err = "active set has no warpid axis";
        return false;
      }
      int64_t wp_off = 0;
      int64_t wp_ext = 0;
      if (!TryAsInt64(wp.offset, &wp_off) || !TryAsInt64(wp.extent, &wp_ext)) {
        *err = "filter on warpgroup_id requires structural warpid offset";
        return false;
      }
      if (wp_off % kWgSize != 0 || wp_ext % kWgSize != 0) {
        *err = "filter on warpgroup_id requires warpid axis aligned to WG_SIZE";
        return false;
      }
      AxisRange cur_outer = MakeRange(wp_ext / kWgSize, wp_off / kWgSize);
      AxisRange new_outer;
      if (!cur_outer.Intersect(lo, hi, &new_outer)) {
        *err = "filter on warpgroup_id produces empty range";
        return false;
      }
      int64_t outer_ext = 0;
      int64_t outer_off = 0;
      TVM_FFI_ICHECK(TryAsInt64(new_outer.extent, &outer_ext));
      TVM_FFI_ICHECK(TryAsInt64(new_outer.offset, &outer_off));
      *out = A.WithAxis("warpid", MakeRange(outer_ext * kWgSize, outer_off * kWgSize));
      return true;
    }
    case ScopeBinding::kWarpgroupWarp: {
      AxisRange wp;
      if (!A.GetAxis("warpid", &wp)) {
        *err = "active set has no warpid axis";
        return false;
      }
      int64_t wp_off = 0;
      int64_t wp_ext = 0;
      if (!TryAsInt64(wp.offset, &wp_off) || !TryAsInt64(wp.extent, &wp_ext)) {
        *err = "filter on warp_id_in_wg requires structural warpid offset";
        return false;
      }
      int64_t cur_inner_off = wp_off % kWgSize;
      if (wp_ext > kWgSize - cur_inner_off) {
        *err = "filter on warp_id_in_wg would break active-set TileLayout box";
        return false;
      }
      AxisRange cur_inner = MakeRange(wp_ext, cur_inner_off);
      AxisRange new_inner;
      if (!cur_inner.Intersect(lo, hi, &new_inner)) {
        *err = "filter on warp_id_in_wg produces empty range";
        return false;
      }
      int64_t inner_ext = 0;
      int64_t inner_off = 0;
      TVM_FFI_ICHECK(TryAsInt64(new_inner.extent, &inner_ext));
      TVM_FFI_ICHECK(TryAsInt64(new_inner.offset, &inner_off));
      int64_t outer_base = (wp_off / kWgSize) * kWgSize;
      *out = A.WithAxis("warpid", MakeRange(inner_ext, outer_base + inner_off));
      return true;
    }
    case ScopeBinding::kKernelCluster:
      *err = "filter on cluster_id is not supported";
      return false;
    case ScopeBinding::kClusterCtaPair:
      *err = "filter on cta_id_in_pair must be lowered through CTA pair modulo analysis";
      return false;
    case ScopeBinding::kCtaThread:
      return NarrowFlatCtaThreadRange(A, lo, hi, out, err);
    case ScopeBinding::kWarpgroupThread:
      return NarrowFlatWarpgroupThreadRange(A, lo, hi, out, err);
  }
  *err = "unknown ScopeBinding";
  return false;
}

bool ScopeSwitch(const ActiveSet& A, ScopeKind scope_kind, ExecSplit* out, std::string* err) {
  out->inter.clear();
  out->intra.clear();
  AxisRange laneid;
  AxisRange warpid;
  TVM_FFI_ICHECK(A.GetAxis("laneid", &laneid));
  TVM_FFI_ICHECK(A.GetAxis("warpid", &warpid));

  switch (scope_kind) {
    case ScopeKind::kThread:
      out->inter["laneid"] = laneid;
      out->inter["warpid"] = warpid;
      AddCtaAxes(A, &out->inter);
      return true;
    case ScopeKind::kWarp:
      out->intra["laneid"] = laneid;
      out->inter["warpid"] = warpid;
      AddCtaAxes(A, &out->inter);
      return true;
    case ScopeKind::kCta:
      out->intra["laneid"] = laneid;
      out->intra["warpid"] = warpid;
      AddCtaAxes(A, &out->inter);
      return true;
    case ScopeKind::kCluster:
      out->intra["laneid"] = laneid;
      out->intra["warpid"] = warpid;
      AddCtaAxes(A, &out->intra);
      return true;
    case ScopeKind::kWarpgroup: {
      AxisRange wid_in_wg;
      AxisRange wgid;
      if (!FactorWarpid(warpid, &wid_in_wg, &wgid)) {
        std::ostringstream os;
        os << "scope_switch(warpgroup) failed: warpid TileLayout axis crosses warpgroup boundary "
              "or has symbolic offset";
        *err = os.str();
        return false;
      }
      out->intra["laneid"] = laneid;
      out->intra["wid_in_wg"] = wid_in_wg;
      out->inter["wgid"] = wgid;
      AddCtaAxes(A, &out->inter);
      return true;
    }
    case ScopeKind::kKernel:
      out->inter["laneid"] = laneid;
      out->inter["warpid"] = warpid;
      AddCtaAxes(A, &out->inter);
      return true;
    case ScopeKind::kWorld:
      *err = "scope_switch(world) is not a valid ExecContext transition";
      return false;
  }
  *err = "unknown ScopeKind";
  return false;
}

ExecContext ExecContext::AtKernelEntry(int64_t lane_ext, int64_t warp_ext, int64_t cta_ext) {
  return AtKernelEntry(lane_ext, warp_ext, cta_ext, {});
}

ExecContext ExecContext::AtKernelEntry(
    int64_t lane_ext, int64_t warp_ext, int64_t cta_ext,
    const std::vector<std::pair<std::string, int64_t>>& cta_axes) {
  ExecContext ctx;
  ctx.A = InitialActiveSet(lane_ext, warp_ext, cta_ext, cta_axes);
  ctx.scope_kind = ScopeKind::kKernel;
  std::string err;
  bool ok = ScopeSwitch(ctx.A, ctx.scope_kind, &ctx.split, &err);
  (void)ok;
  return ctx;
}

bool ExecContext::WithFilter(ScopeBinding binding, int64_t lo, int64_t hi, ExecContext* out,
                             std::string* err) const {
  ActiveSet new_A;
  if (!FilterNarrow(A, binding, lo, hi, &new_A, err)) return false;
  ExecSplit new_split;
  if (!ScopeSwitch(new_A, scope_kind, &new_split, err)) return false;
  out->A = new_A;
  out->scope_kind = scope_kind;
  out->split = std::move(new_split);
  return true;
}

bool ExecContext::WithSelector(ScopeBinding binding, PrimExpr selector, ExecContext* out,
                               std::string* err) const {
  if (binding != ScopeBinding::kWarpThread) {
    *err = "selector filter currently requires a lane_id / warp->thread binding";
    return false;
  }
  ActiveSet new_A = A.WithAxis("laneid", AxisRange{I64(1), selector, I64(1)});
  ExecSplit new_split;
  if (!ScopeSwitch(new_A, scope_kind, &new_split, err)) return false;
  out->A = std::move(new_A);
  out->scope_kind = scope_kind;
  out->split = std::move(new_split);
  return true;
}

bool ExecContext::WithCtaAxisFilter(const std::string& axis, int64_t lo, int64_t hi,
                                    ExecContext* out, std::string* err) const {
  if (lo >= hi) {
    *err = "filter range is empty or inverted";
    return false;
  }
  ActiveSet new_A;
  if (!NarrowAxis(A, axis, lo, hi, &new_A, err)) return false;
  ExecSplit new_split;
  if (!ScopeSwitch(new_A, scope_kind, &new_split, err)) return false;
  out->A = std::move(new_A);
  out->scope_kind = scope_kind;
  out->split = std::move(new_split);
  return true;
}

bool ExecContext::WithCtaAxisModulo(const std::string& axis, int64_t modulus, int64_t residue,
                                    ExecContext* out, std::string* err) const {
  ActiveSet new_A;
  if (!ModuloAxis(A, axis, modulus, residue, &new_A, err)) return false;
  ExecSplit new_split;
  if (!ScopeSwitch(new_A, scope_kind, &new_split, err)) return false;
  out->A = std::move(new_A);
  out->scope_kind = scope_kind;
  out->split = std::move(new_split);
  return true;
}

bool ExecContext::WithScopeSwitch(ScopeKind new_scope_kind, ExecContext* out,
                                  std::string* err) const {
  ExecSplit new_split;
  if (!ScopeSwitch(A, new_scope_kind, &new_split, err)) return false;
  out->A = A;
  out->scope_kind = new_scope_kind;
  out->split = std::move(new_split);
  return true;
}

ffi::Map<ffi::String, ffi::Array<PrimExpr>> EncodeSplitSide(
    const std::unordered_map<std::string, AxisRange>& side) {
  ffi::Map<ffi::String, ffi::Array<PrimExpr>> out;
  for (const auto& kv : side) {
    if (IsZero(kv.second.stride - I64(1))) {
      out.Set(ffi::String(kv.first), ffi::Array<PrimExpr>{kv.second.extent, kv.second.offset});
    } else {
      out.Set(ffi::String(kv.first),
              ffi::Array<PrimExpr>{kv.second.extent, kv.second.offset, kv.second.stride});
    }
  }
  return out;
}

}  // namespace tirx
}  // namespace tvm
