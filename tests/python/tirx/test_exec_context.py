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
"""Unit tests for ExecContext (RFC v3 §6). Cases mirror RFC §8.1 -- §8.10."""

from __future__ import annotations

import pytest

from tvm.tirx.exec_context import (
    CLUSTER,
    CTA,
    LANE_CTA_THREAD,
    LANE_FLAT,
    LANE_W_INNER,
    LANE_WG_OUTER,
    LANE_WG_THREAD,
    THREAD,
    WARP,
    WARPGROUP,
    AxisRange,
    ExecContext,
    ExecContextError,
    LaneBinding,
    filter_modulo,
    filter_narrow,
    initial_A,
    scope_switch,
)

# -- canonical bindings declared at kernel entry (see RFC §8 naming conv) --
WARP_FLAT = LaneBinding(axis="warpid", kind=LANE_FLAT, declared_extent=16)
WG_OUTER = LaneBinding(axis="warpid", kind=LANE_WG_OUTER, declared_extent=4)
W_INNER = LaneBinding(axis="warpid", kind=LANE_W_INNER, declared_extent=4)
LANE_BIND = LaneBinding(axis="laneid", kind=LANE_FLAT, declared_extent=32)
CTA_BIND = LaneBinding(axis="cta_id", kind=LANE_FLAT, declared_extent=1)
CTA_THREAD_BIND = LaneBinding(axis="thread", kind=LANE_CTA_THREAD, declared_extent=256)
WG_THREAD_BIND = LaneBinding(axis="thread", kind=LANE_WG_THREAD, declared_extent=128)


# ---------------------------------------------------------------------------
# §3 scope_switch: split table
# ---------------------------------------------------------------------------


def test_initial_A_single_cta():
    A = initial_A(warp_ext=16)
    assert A.laneid == AxisRange(32, 0)
    assert A.warpid == AxisRange(16, 0)
    assert A.cta_id == AxisRange(1, 0)
    assert A.size == 512


def test_initial_A_cluster():
    A = initial_A(warp_ext=16, cta_ext=4)
    assert A.cta_id == AxisRange(4, 0)
    assert A.size == 2048


def test_axis_modulo_filter_uses_stride():
    A = initial_A(warp_ext=16, cta_ext=4)
    A = filter_modulo(A, "cta_id", 2, 0)
    assert A.cta_id == AxisRange(2, 0, 2)
    A = filter_narrow(A, CTA_BIND, 1, 4)
    assert A.cta_id == AxisRange(1, 2, 2)


def test_axis_modulo_filter_two_cta_pair_residues():
    A = initial_A(warp_ext=16, cta_ext=2)
    assert filter_modulo(A, "cta_id", 2, 0).cta_id == AxisRange(1, 0, 2)
    assert filter_modulo(A, "cta_id", 2, 1).cta_id == AxisRange(1, 1, 2)


@pytest.mark.parametrize(
    "kappa,expected_inter_axes,expected_intra_axes",
    [
        (THREAD, {"laneid", "warpid", "cta_id"}, set()),
        (WARP, {"warpid", "cta_id"}, {"laneid"}),
        (CTA, {"cta_id"}, {"laneid", "warpid"}),
        (CLUSTER, set(), {"laneid", "warpid", "cta_id"}),
    ],
)
def test_scope_switch_trivial(kappa, expected_inter_axes, expected_intra_axes):
    A = initial_A(warp_ext=16, cta_ext=4)
    split = scope_switch(A, kappa)
    assert set(split.inter) == expected_inter_axes
    assert set(split.intra) == expected_intra_axes


def test_scope_switch_warpgroup_aligned():
    A = initial_A(warp_ext=16)
    split = scope_switch(A, WARPGROUP)
    assert split.inter["wgid"] == AxisRange(4, 0)
    assert split.inter["cta_id"] == AxisRange(1, 0)
    assert split.intra["laneid"] == AxisRange(32, 0)
    assert split.intra["wid_in_wg"] == AxisRange(4, 0)


# ---------------------------------------------------------------------------
# §4.2 warpgroup factoring: 3 cases
# ---------------------------------------------------------------------------


def test_factor_case1_aligned():
    A = initial_A(warp_ext=8)  # ext=8, off=0 -- aligned
    split = scope_switch(A, WARPGROUP)
    assert split.inter["wgid"] == AxisRange(2, 0)
    assert split.intra["wid_in_wg"] == AxisRange(4, 0)


def test_factor_case2_fits_in_one_wg():
    # warpid ext=2, off=0 -- fits in one wg
    A = initial_A(warp_ext=16)
    A = filter_narrow(A, WARP_FLAT, 0, 2)
    split = scope_switch(A, WARPGROUP)
    assert split.inter["wgid"] == AxisRange(1, 0)
    assert split.intra["wid_in_wg"] == AxisRange(2, 0)


def test_factor_case2_offset():
    # warpid ext=2, off=6 -> wid_off=2, fits (2 <= 4-2)
    A = initial_A(warp_ext=16)
    A = filter_narrow(A, WARP_FLAT, 6, 8)
    split = scope_switch(A, WARPGROUP)
    assert split.inter["wgid"] == AxisRange(1, 1)
    assert split.intra["wid_in_wg"] == AxisRange(2, 2)


def test_factor_case3_fails():
    # RFC §8.6: warpid[2:6] crosses wg boundary unaligned
    A = initial_A(warp_ext=16)
    A = filter_narrow(A, WARP_FLAT, 2, 6)
    assert A.warpid == AxisRange(4, 2)
    with pytest.raises(ExecContextError, match="crosses warpgroup boundary"):
        scope_switch(A, WARPGROUP)


# ---------------------------------------------------------------------------
# §8.1 -- Pure narrowing CTA -> WG -> W
# ---------------------------------------------------------------------------


def test_ex_8_1_cta_wg_warp():
    ctx = ExecContext.at_kernel_entry(warp_ext=16)
    # with T.cta()
    ctx = ctx.with_scope_switch(CTA)
    assert ctx.inter == {"cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0), "warpid": AxisRange(16, 0)}
    # with T.warpgroup()
    ctx = ctx.with_scope_switch(WARPGROUP)
    assert ctx.inter == {"wgid": AxisRange(4, 0), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0), "wid_in_wg": AxisRange(4, 0)}
    # with T.warp()
    ctx = ctx.with_scope_switch(WARP)
    assert ctx.inter == {"warpid": AxisRange(16, 0), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0)}


# ---------------------------------------------------------------------------
# §8.2 -- Filter + scope_switch
# ---------------------------------------------------------------------------


def test_ex_8_2_filter_then_warpgroup():
    ctx = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)
    ctx = ctx.with_filter(WARP_FLAT, 0, 8)
    assert ctx.A.warpid == AxisRange(8, 0)
    # recompute at cta: intra=(lane:32, warp:8)
    assert ctx.intra == {"laneid": AxisRange(32, 0), "warpid": AxisRange(8, 0)}
    # enter warpgroup: factor(8, 0) -> case 1
    ctx = ctx.with_scope_switch(WARPGROUP)
    assert ctx.inter == {"wgid": AxisRange(2, 0), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0), "wid_in_wg": AxisRange(4, 0)}


# ---------------------------------------------------------------------------
# §8.3 -- Sugar form T.warp(warpid[2:4])
# ---------------------------------------------------------------------------


def test_ex_8_3_sugar_warp_range():
    ctx = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)
    # desugar: filter warpid[2:4], then warp
    ctx = ctx.with_filter(WARP_FLAT, 2, 4).with_scope_switch(WARP)
    assert ctx.A.warpid == AxisRange(2, 2)
    assert ctx.inter == {"warpid": AxisRange(2, 2), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0)}


# ---------------------------------------------------------------------------
# §8.4 -- Widen after filter (warp -> warpgroup)
# ---------------------------------------------------------------------------


def test_ex_8_4_widen_warp_to_wg():
    ctx = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)
    ctx = ctx.with_filter(WARP_FLAT, 0, 4).with_scope_switch(WARP)
    # widen to warpgroup
    ctx = ctx.with_scope_switch(WARPGROUP)
    assert ctx.inter == {"wgid": AxisRange(1, 0), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0), "wid_in_wg": AxisRange(4, 0)}


# ---------------------------------------------------------------------------
# §8.5 -- Partial warp selection -> warpgroup (partial intra)
# ---------------------------------------------------------------------------


def test_ex_8_5_partial_wg():
    ctx = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)
    ctx = ctx.with_filter(WARP_FLAT, 0, 2).with_scope_switch(WARPGROUP)
    # case 2: 2 <= 4-0
    assert ctx.inter == {"wgid": AxisRange(1, 0), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0), "wid_in_wg": AxisRange(2, 0)}


# ---------------------------------------------------------------------------
# §8.6 -- Cross warpgroup boundary (factor fails)
# ---------------------------------------------------------------------------


def test_ex_8_6_factor_fail():
    ctx = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)
    # with_filter recomputes (inter, intra) for current scope_kind=cta -- still OK
    ctx2 = ctx.with_filter(WARP_FLAT, 2, 6)
    assert ctx2.A.warpid == AxisRange(4, 2)
    # scope_switch to warpgroup is the one that must fail
    with pytest.raises(ExecContextError, match="crosses warpgroup boundary"):
        ctx2.with_scope_switch(WARPGROUP)


# ---------------------------------------------------------------------------
# §8.7 -- Deep mixed nesting
# ---------------------------------------------------------------------------


def test_ex_8_7_deep_nested():
    ctx = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)
    ctx = ctx.with_filter(WARP_FLAT, 0, 8).with_scope_switch(WARPGROUP)
    assert ctx.inter == {"wgid": AxisRange(2, 0), "cta_id": AxisRange(1, 0)}
    ctx = ctx.with_filter(WARP_FLAT, 0, 2)
    # recompute at warpgroup: factor(2, 0) -> case 2
    assert ctx.inter == {"wgid": AxisRange(1, 0), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0), "wid_in_wg": AxisRange(2, 0)}
    ctx = ctx.with_scope_switch(WARP)
    assert ctx.inter == {"warpid": AxisRange(2, 0), "cta_id": AxisRange(1, 0)}
    assert ctx.intra == {"laneid": AxisRange(32, 0)}
    ctx = ctx.with_filter(LANE_BIND, 0, 8)
    assert ctx.intra == {"laneid": AxisRange(8, 0)}
    assert ctx.inter == {"warpid": AxisRange(2, 0), "cta_id": AxisRange(1, 0)}


# ---------------------------------------------------------------------------
# §8.8 -- FA4 pattern: 3 sibling filter branches
# ---------------------------------------------------------------------------


def test_ex_8_8_fa4_pattern():
    root = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)

    # Branch 1: warp 12 (single warp, tcgen05 MMA elected)
    b1 = root.with_filter(WARP_FLAT, 12, 13)
    assert b1.A.warpid == AxisRange(1, 12)
    assert b1.intra == {"laneid": AxisRange(32, 0), "warpid": AxisRange(1, 12)}

    # Branch 2: softmax warpgroups (warps 0-7)
    b2 = root.with_filter(WARP_FLAT, 0, 8).with_scope_switch(WARPGROUP)
    assert b2.inter == {"wgid": AxisRange(2, 0), "cta_id": AxisRange(1, 0)}
    assert b2.intra == {"laneid": AxisRange(32, 0), "wid_in_wg": AxisRange(4, 0)}

    # Branch 3: correction warpgroup (warps 8-11 = wg2)
    b3 = root.with_filter(WARP_FLAT, 8, 12)
    assert b3.A.warpid == AxisRange(4, 8)
    assert b3.intra == {"laneid": AxisRange(32, 0), "warpid": AxisRange(4, 8)}
    # And should factor cleanly when entering warpgroup
    b3wg = b3.with_scope_switch(WARPGROUP)
    assert b3wg.inter == {"wgid": AxisRange(1, 2), "cta_id": AxisRange(1, 0)}
    assert b3wg.intra == {"laneid": AxisRange(32, 0), "wid_in_wg": AxisRange(4, 0)}


# ---------------------------------------------------------------------------
# §8.9 -- Cross-CTA with widening to cluster
# ---------------------------------------------------------------------------


def test_ex_8_9_cross_cta_cluster():
    ctx = ExecContext.at_kernel_entry(warp_ext=16, cta_ext=4).with_scope_switch(CTA)
    assert ctx.inter == {"cta_id": AxisRange(4, 0)}
    # filter to warp 0, then warp
    w = ctx.with_filter(WARP_FLAT, 0, 1).with_scope_switch(WARP)
    assert w.inter == {"warpid": AxisRange(1, 0), "cta_id": AxisRange(4, 0)}
    assert w.intra == {"laneid": AxisRange(32, 0)}

    # back at cta scope, enter warpgroup
    wg = ctx.with_scope_switch(WARPGROUP)
    assert wg.inter == {"wgid": AxisRange(4, 0), "cta_id": AxisRange(4, 0)}
    # widen to cluster
    cl = wg.with_scope_switch(CLUSTER)
    assert cl.inter == {}
    assert cl.intra == {
        "laneid": AxisRange(32, 0),
        "warpid": AxisRange(16, 0),
        "cta_id": AxisRange(4, 0),
    }


# ---------------------------------------------------------------------------
# §8.10 -- identical to 8.3 modulo prose; covered above
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rule 1 & 5: filter can only shrink A; saved/restored across scope exit
# (Restoration is the caller's (IR walker's) responsibility -- ExecContext
# is immutable, each with_filter returns a fresh ctx. Test that the parent
# is untouched.)
# ---------------------------------------------------------------------------


def test_filter_is_pure():
    ctx = ExecContext.at_kernel_entry(warp_ext=16).with_scope_switch(CTA)
    child = ctx.with_filter(WARP_FLAT, 0, 8)
    assert ctx.A.warpid == AxisRange(16, 0)  # parent not mutated
    assert child.A.warpid == AxisRange(8, 0)


def test_filter_empty_range_rejected():
    A = initial_A(warp_ext=16)
    with pytest.raises(ExecContextError, match="empty or inverted"):
        filter_narrow(A, WARP_FLAT, 5, 5)


def test_filter_out_of_range_rejected():
    A = initial_A(warp_ext=16)
    A = filter_narrow(A, WARP_FLAT, 0, 4)
    with pytest.raises(ExecContextError, match="empty range"):
        filter_narrow(A, WARP_FLAT, 8, 12)  # disjoint from [0, 4)


def test_filter_flat_cta_thread_full_warp_range():
    A = initial_A(warp_ext=8)
    A = filter_narrow(A, CTA_THREAD_BIND, 0, 128)
    assert A.warpid == AxisRange(4, 0)
    assert A.laneid == AxisRange(32, 0)


def test_filter_flat_cta_thread_single_warp_lane_range():
    A = initial_A(warp_ext=8)
    A = filter_narrow(A, CTA_THREAD_BIND, 34, 40)
    assert A.warpid == AxisRange(1, 1)
    assert A.laneid == AxisRange(6, 2)


def test_filter_flat_cta_thread_nonrectangular_rejected():
    A = initial_A(warp_ext=8)
    with pytest.raises(ExecContextError, match="non-rectangular"):
        filter_narrow(A, CTA_THREAD_BIND, 20, 50)


def test_filter_flat_warpgroup_thread_range_inside_one_warpgroup():
    A = initial_A(warp_ext=8)
    A = filter_narrow(A, WG_OUTER, 1, 2)
    A = filter_narrow(A, WG_THREAD_BIND, 32, 64)
    assert A.warpid == AxisRange(1, 5)
    assert A.laneid == AxisRange(32, 0)


def test_filter_flat_warpgroup_thread_full_range_across_warpgroups_is_noop():
    A = initial_A(warp_ext=8)
    A2 = filter_narrow(A, WG_THREAD_BIND, 0, 128)
    assert A2.warpid == AxisRange(8, 0)
    assert A2.laneid == AxisRange(32, 0)


def test_filter_flat_warpgroup_thread_partial_range_across_warpgroups_rejected():
    A = initial_A(warp_ext=8)
    with pytest.raises(ExecContextError, match="multiple warpgroups"):
        filter_narrow(A, WG_THREAD_BIND, 0, 64)


# ---------------------------------------------------------------------------
# Factor-lane bindings: wg_outer and w_inner
# ---------------------------------------------------------------------------


def test_filter_wg_outer():
    A = initial_A(warp_ext=16)
    A2 = filter_narrow(A, WG_OUTER, 1, 3)  # wg 1..2 -> warps 4..11
    assert A2.warpid == AxisRange(8, 4)


def test_filter_wg_outer_unaligned_rejected():
    A = initial_A(warp_ext=16)
    A = filter_narrow(A, WARP_FLAT, 2, 6)  # warp offset 2 (not WG-aligned)
    with pytest.raises(ExecContextError, match="aligned to WG_SIZE"):
        filter_narrow(A, WG_OUTER, 0, 1)


def test_filter_w_inner():
    A = initial_A(warp_ext=16)
    # First narrow into a single warpgroup, then inner filter is valid
    A = filter_narrow(A, WARP_FLAT, 4, 8)  # wg1: warps 4..7
    A2 = filter_narrow(A, W_INNER, 1, 3)  # pick inner lanes 1..2
    assert A2.warpid == AxisRange(2, 5)


def test_filter_w_inner_spanning_wg_rejected():
    A = initial_A(warp_ext=16)  # spans all 4 wgs
    with pytest.raises(ExecContextError, match="spans multiple warpgroups"):
        filter_narrow(A, W_INNER, 0, 2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
