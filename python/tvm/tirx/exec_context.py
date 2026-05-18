# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""ExecContext: per-program-point active-thread state.

The active thread set is represented as a ``TileLayout``: active axes live in
``layout.shard`` and per-axis lower bounds live in ``layout.offset``. Filters
narrow that layout; scope switches derive the current ``inter``/``intra`` view.
"""

from __future__ import annotations

from dataclasses import dataclass

from tvm.tirx.layout import Axis, Iter, TileLayout

WG_SIZE = 4

KERNEL = "kernel"
CLUSTER = "cluster"
CTA = "cta"
WARPGROUP = "warpgroup"
WARP = "warp"
THREAD = "thread"

SCOPE_KINDS = (KERNEL, CLUSTER, CTA, WARPGROUP, WARP, THREAD)

LANE_FLAT = "flat"
LANE_WG_OUTER = "wg_outer"
LANE_W_INNER = "w_inner"
LANE_CTA_THREAD = "cta_thread"
LANE_WG_THREAD = "wg_thread"


class ExecContextError(Exception):
    """Raised on structural violations of the ExecContext model."""


def _ceildiv(lhs: int, rhs: int) -> int:
    return -((-lhs) // rhs)


def _gcd(lhs: int, rhs: int) -> int:
    while rhs:
        lhs, rhs = rhs, lhs % rhs
    return abs(lhs)


def _extended_gcd(lhs: int, rhs: int) -> tuple[int, int, int]:
    if rhs == 0:
        return lhs, 1, 0
    gcd, x1, y1 = _extended_gcd(rhs, lhs % rhs)
    return gcd, y1, x1 - (lhs // rhs) * y1


def _mod_inverse(value: int, modulus: int) -> int:
    if modulus == 1:
        return 0
    gcd, inv, _ = _extended_gcd(value % modulus, modulus)
    if gcd != 1:
        raise ExecContextError(f"{value} has no inverse modulo {modulus}")
    return inv % modulus


@dataclass(frozen=True)
class AxisRange:
    """An active slice offset + stride * [0, extent) on one TileLayout axis."""

    extent: int
    offset: int = 0
    stride: int = 1

    def intersect(self, lo: int, hi: int) -> AxisRange:
        i_lo = max(0, _ceildiv(lo - self.offset, self.stride))
        i_hi = min(self.extent, (hi - 1 - self.offset) // self.stride + 1)
        if i_hi <= i_lo:
            raise ExecContextError(
                f"filter produces empty range: current=[{self.offset},"
                f" {self.offset + self.extent}) ∩ [{lo}, {hi})"
            )
        return AxisRange(
            extent=i_hi - i_lo, offset=self.offset + self.stride * i_lo, stride=self.stride
        )

    def modulo(self, modulus: int, residue: int) -> AxisRange:
        residue %= modulus
        rhs = (residue - self.offset) % modulus
        g = _gcd(self.stride, modulus)
        if rhs % g != 0:
            raise ExecContextError(
                f"modulo filter produces empty range: {self.offset} + {self.stride} * i"
                f" == {residue} mod {modulus}"
            )
        reduced_stride = self.stride // g
        reduced_rhs = rhs // g
        reduced_modulus = modulus // g
        period = reduced_modulus
        i0 = (reduced_rhs * _mod_inverse(reduced_stride, reduced_modulus)) % reduced_modulus
        if i0 >= self.extent:
            raise ExecContextError(
                f"modulo filter produces empty range: {self.offset} + {self.stride} * i"
                f" == {residue} mod {modulus}"
            )
        return AxisRange(
            extent=(self.extent - 1 - i0) // period + 1,
            offset=self.offset + self.stride * i0,
            stride=self.stride * period,
        )


@dataclass(frozen=True)
class ActiveSet:
    """Active thread set represented by a TileLayout."""

    layout: TileLayout

    @staticmethod
    def from_axes(axes: list[tuple[str, AxisRange]]) -> ActiveSet:
        shard = [Iter(axis_range.extent, axis_range.stride, name) for name, axis_range in axes]
        offset = {
            Axis.get(name): axis_range.offset for name, axis_range in axes if axis_range.offset != 0
        }
        return ActiveSet(TileLayout.from_iters(shard, [], offset))

    @property
    def size(self) -> int:
        result = 1
        for it in self.layout.shard:
            result *= int(it.extent)
        return result

    @property
    def axis_names(self) -> list[str]:
        return [str(it.axis.name) for it in self.layout.shard]

    def axis(self, name: str) -> AxisRange:
        for it in self.layout.shard:
            if str(it.axis.name) != name:
                continue
            offset = 0
            for axis, value in self.layout.offset.items():
                if str(axis.name) == name:
                    offset = int(value)
                    break
            return AxisRange(int(it.extent), offset, int(it.stride))
        raise ValueError(f"unknown active-set axis: {name!r}")

    def replace_axis(self, axis: str, axis_range: AxisRange) -> ActiveSet:
        axes: list[tuple[str, AxisRange]] = []
        found = False
        for name in self.axis_names:
            if name == axis:
                axes.append((name, axis_range))
                found = True
            else:
                axes.append((name, self.axis(name)))
        if not found:
            raise ValueError(f"unknown active-set axis: {axis!r}")
        return ActiveSet.from_axes(axes)

    @property
    def laneid(self) -> AxisRange:
        return self.axis("laneid")

    @property
    def warpid(self) -> AxisRange:
        return self.axis("warpid")

    @property
    def cta_id(self) -> AxisRange:
        return self.axis("cta_id")


@dataclass(frozen=True)
class LaneBinding:
    """Resolution of a user-declared ScopeIdDef Var to one active-set axis."""

    axis: str
    kind: str
    declared_extent: int


def initial_A(*, lane_ext: int = 32, warp_ext: int, cta_ext: int = 1) -> ActiveSet:
    """Build A at T.kernel() entry: all threads active, offsets all zero."""
    return ActiveSet.from_axes(
        [
            ("laneid", AxisRange(lane_ext, 0)),
            ("warpid", AxisRange(warp_ext, 0)),
            ("cta_id", AxisRange(cta_ext, 0)),
        ]
    )


def filter_narrow(A: ActiveSet, binding: LaneBinding, lo: int, hi: int) -> ActiveSet:
    """Intersect A's binding axis with [lo, hi)."""
    if lo >= hi:
        raise ExecContextError(f"filter range [{lo}, {hi}) is empty or inverted")

    if binding.kind == LANE_CTA_THREAD:
        new_warpid, new_laneid = _flat_product_range(A.warpid, A.laneid, lo, hi)
        return A.replace_axis("laneid", new_laneid).replace_axis("warpid", new_warpid)

    if binding.kind == LANE_WG_THREAD:
        factored = _factor_warpid(A.warpid)
        if factored is None:
            raise ExecContextError(
                "filter on flat warpgroup-thread range requires factorable warpid axis"
            )
        wid_in_wg, wgid = factored
        new_wid_in_wg, new_laneid = _flat_product_range(wid_in_wg, A.laneid, lo, hi)
        if wgid.extent != 1:
            if new_wid_in_wg == wid_in_wg and new_laneid == A.laneid:
                return A
            raise ExecContextError(
                "flat warpgroup-thread range across multiple warpgroups is not representable"
            )
        new_warpid = AxisRange(
            extent=new_wid_in_wg.extent, offset=wgid.offset * WG_SIZE + new_wid_in_wg.offset
        )
        return A.replace_axis("laneid", new_laneid).replace_axis("warpid", new_warpid)

    if binding.kind == LANE_FLAT:
        new_axis = A.axis(binding.axis).intersect(lo, hi)
        return A.replace_axis(binding.axis, new_axis)

    if binding.axis != "warpid":
        raise ExecContextError(
            f"kind={binding.kind!r} only valid for axis='warpid'; got {binding.axis!r}"
        )

    wp = A.warpid
    if wp.stride != 1:
        raise ExecContextError(
            f"kind={binding.kind!r} requires unit-stride warpid axis; got stride={wp.stride}"
        )
    if binding.kind == LANE_WG_OUTER:
        if wp.offset % WG_SIZE != 0 or wp.extent % WG_SIZE != 0:
            raise ExecContextError(
                f"filter on wg_outer requires warpid axis aligned to WG_SIZE={WG_SIZE};"
                f" got extent={wp.extent}, offset={wp.offset}"
            )
        cur_outer = AxisRange(extent=wp.extent // WG_SIZE, offset=wp.offset // WG_SIZE)
        new_outer = cur_outer.intersect(lo, hi)
        return A.replace_axis(
            "warpid",
            AxisRange(extent=new_outer.extent * WG_SIZE, offset=new_outer.offset * WG_SIZE),
        )

    if binding.kind == LANE_W_INNER:
        cur_inner_off = wp.offset % WG_SIZE
        if wp.extent > WG_SIZE - cur_inner_off:
            raise ExecContextError(
                "filter on w_inner would break A's TileLayout box: warpid spans multiple"
                f" warpgroups (extent={wp.extent}, offset={wp.offset})"
            )
        cur_inner = AxisRange(extent=wp.extent, offset=cur_inner_off)
        new_inner = cur_inner.intersect(lo, hi)
        outer_base = (wp.offset // WG_SIZE) * WG_SIZE
        return A.replace_axis(
            "warpid", AxisRange(extent=new_inner.extent, offset=outer_base + new_inner.offset)
        )

    raise ValueError(f"unknown axis kind: {binding.kind!r}")


def filter_modulo(A: ActiveSet, axis: str, modulus: int, residue: int) -> ActiveSet:
    """Intersect an active-set axis with ``axis % modulus == residue``."""
    if modulus <= 0:
        raise ExecContextError(f"modulus must be positive, got {modulus}")
    new_axis = A.axis(axis).modulo(modulus, residue)
    return A.replace_axis(axis, new_axis)


@dataclass(frozen=True)
class Split:
    """A scope_switch split of A."""

    inter: dict[str, AxisRange]
    intra: dict[str, AxisRange]


def _factor_warpid(warp: AxisRange) -> tuple[AxisRange, AxisRange] | None:
    if warp.stride != 1:
        return None
    off = warp.offset
    ext = warp.extent
    wid_off = off % WG_SIZE
    wgid_off = off // WG_SIZE

    if wid_off == 0 and ext % WG_SIZE == 0:
        return (
            AxisRange(extent=WG_SIZE, offset=0),
            AxisRange(extent=ext // WG_SIZE, offset=wgid_off),
        )
    if ext <= WG_SIZE - wid_off:
        return (AxisRange(extent=ext, offset=wid_off), AxisRange(extent=1, offset=wgid_off))
    return None


def _flat_product_range(
    major: AxisRange, lane: AxisRange, lo: int, hi: int
) -> tuple[AxisRange, AxisRange]:
    active_min = major.offset * 32 + lane.offset
    active_max = (
        (major.offset + major.stride * (major.extent - 1)) * 32
        + lane.offset
        + lane.stride * (lane.extent - 1)
        + 1
    )
    if lo <= active_min and active_max <= hi:
        return major, lane

    if major.stride != 1 or lane.stride != 1:
        raise ExecContextError("flat thread range narrowing requires unit-stride axes")

    lane_hi = lane.offset + lane.extent
    major_hi = major.offset + major.extent
    hit_lo = max(major.offset, (lo - lane_hi) // 32 + 1)
    hit_hi = min(major_hi, _ceildiv(hi - lane.offset, 32))
    if hit_hi <= hit_lo:
        raise ExecContextError("flat thread range produces empty active set")

    if hit_hi == hit_lo + 1:
        new_lane_lo = max(lane.offset, lo - hit_lo * 32)
        new_lane_hi = min(lane_hi, hi - hit_lo * 32)
        if new_lane_hi <= new_lane_lo:
            raise ExecContextError("flat thread range produces empty lane range")
        return AxisRange(1, hit_lo), AxisRange(new_lane_hi - new_lane_lo, new_lane_lo)

    if lo <= hit_lo * 32 + lane.offset and (hit_hi - 1) * 32 + lane_hi <= hi:
        return AxisRange(hit_hi - hit_lo, hit_lo), lane

    raise ExecContextError("flat thread range would require a non-rectangular lane/warp active set")


def scope_switch(A: ActiveSet, scope_kind: str) -> Split:
    """Split A into (inter, intra) for the target scope kind."""
    if scope_kind == THREAD:
        return Split(inter={"laneid": A.laneid, "warpid": A.warpid, "cta_id": A.cta_id}, intra={})
    if scope_kind == WARP:
        return Split(inter={"warpid": A.warpid, "cta_id": A.cta_id}, intra={"laneid": A.laneid})
    if scope_kind == CTA:
        return Split(inter={"cta_id": A.cta_id}, intra={"laneid": A.laneid, "warpid": A.warpid})
    if scope_kind == CLUSTER:
        return Split(inter={}, intra={"laneid": A.laneid, "warpid": A.warpid, "cta_id": A.cta_id})
    if scope_kind == WARPGROUP:
        factored = _factor_warpid(A.warpid)
        if factored is None:
            raise ExecContextError(
                "scope_switch(warpgroup) failed: warpid axis"
                f" (extent={A.warpid.extent}, offset={A.warpid.offset})"
                " crosses warpgroup boundary and is not aligned"
            )
        wid_in_wg, wgid = factored
        return Split(
            inter={"wgid": wgid, "cta_id": A.cta_id},
            intra={"laneid": A.laneid, "wid_in_wg": wid_in_wg},
        )
    if scope_kind == KERNEL:
        return Split(inter={"laneid": A.laneid, "warpid": A.warpid, "cta_id": A.cta_id}, intra={})
    raise ValueError(f"unknown scope kind: {scope_kind!r}")


@dataclass(frozen=True)
class ExecContext:
    """Per-program-point compiler state: active set + scope kind + split."""

    A: ActiveSet
    scope_kind: str
    inter: dict[str, AxisRange]
    intra: dict[str, AxisRange]

    @staticmethod
    def at_kernel_entry(*, lane_ext: int = 32, warp_ext: int, cta_ext: int = 1) -> ExecContext:
        A = initial_A(lane_ext=lane_ext, warp_ext=warp_ext, cta_ext=cta_ext)
        split = scope_switch(A, KERNEL)
        return ExecContext(A=A, scope_kind=KERNEL, inter=split.inter, intra=split.intra)

    def with_filter(self, binding: LaneBinding, lo: int, hi: int) -> ExecContext:
        new_A = filter_narrow(self.A, binding, lo, hi)
        split = scope_switch(new_A, self.scope_kind)
        return ExecContext(
            A=new_A, scope_kind=self.scope_kind, inter=split.inter, intra=split.intra
        )

    def with_cta_axis_modulo(self, axis: str, modulus: int, residue: int) -> ExecContext:
        new_A = filter_modulo(self.A, axis, modulus, residue)
        split = scope_switch(new_A, self.scope_kind)
        return ExecContext(
            A=new_A, scope_kind=self.scope_kind, inter=split.inter, intra=split.intra
        )

    def with_scope_switch(self, scope_kind: str) -> ExecContext:
        split = scope_switch(self.A, scope_kind)
        return ExecContext(A=self.A, scope_kind=scope_kind, inter=split.inter, intra=split.intra)
