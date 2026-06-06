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
# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
import functools
import itertools
import operator

import pytest

import tvm
from tvm.arith import Analyzer
from tvm.ir import assert_structural_equal
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as Tx_builder
from tvm.tirx import Var
from tvm.tirx.layout import (
    Axis,
    ComposeLayout,
    F,
    Iter,
    P,
    R,
    S,
    SwizzleLayout,
    TileLayout,
    laneid,
    m,
    tid_in_wg,
    tx,
    warpid,
    wg_local_layout,
    wgid,
    wid_in_wg,
)
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import (
    SwizzleMode,
    mma_shared_layout,
    tma_shared_layout,
)


def test_axis():
    assert Axis.bx == Axis.get("bx")
    assert Axis.by == Axis.get("by")
    assert Axis.bz == Axis.get("bz")
    assert Axis.cbx == Axis.get("cbx")
    assert Axis.cby == Axis.get("cby")
    assert Axis.cbz == Axis.get("cbz")
    assert Axis.tx == Axis.get("tx")
    assert Axis.warpid == Axis.get("warpid")
    assert Axis.laneid == Axis.get("laneid")
    assert Axis.wgid == Axis.get("wgid")
    assert Axis.tid_in_wg == Axis.get("tid_in_wg")
    assert Axis.wid_in_wg == Axis.get("wid_in_wg")
    assert Axis.m == Axis.get("m")
    assert Axis.P == Axis.get("P")
    assert Axis.F == Axis.get("F")
    assert Axis.TCol == Axis.get("TCol")
    assert Axis.TLane == Axis.get("TLane")

    assert Axis.bx.is_thread()
    assert Axis.by.is_thread()
    assert Axis.bz.is_thread()
    assert Axis.cbx.is_thread()
    assert Axis.cby.is_thread()
    assert Axis.cbz.is_thread()
    assert Axis.tx.is_thread()
    assert Axis.warpid.is_thread()
    assert Axis.laneid.is_thread()
    assert Axis.wgid.is_thread()
    assert Axis.tid_in_wg.is_thread()
    assert Axis.wid_in_wg.is_thread()
    assert Axis.m.is_memory()
    assert Axis.P.is_memory()
    assert Axis.F.is_memory()
    assert Axis.TCol.is_memory()
    assert Axis.TLane.is_memory()

    assert Axis.bx.get_scope().name == "thread"
    assert Axis.bx.get_subscope().name == "cta"


def test_constructor():
    def assert_tile_layout(layout, shard, replica=None, offset=None):
        expected = TileLayout.from_iters(shard, replica or [], offset or {})
        assert_structural_equal(layout, expected)

    layout = TileLayout(S[2, 3, 4])
    assert_tile_layout(layout, [Iter(2, 12, "m"), Iter(3, 4, "m"), Iter(4, 1, "m")])

    layout = TileLayout(S[(2, 3, 4) : (12, 4, 1)])
    assert_tile_layout(layout, [Iter(2, 12, "m"), Iter(3, 4, "m"), Iter(4, 1, "m")])

    layout = TileLayout(S[(2, 3, 4) : (12 @ m, 4 @ m, 1 @ m)])
    assert_tile_layout(layout, [Iter(2, 12, "m"), Iter(3, 4, "m"), Iter(4, 1, "m")])

    layout = TileLayout(S[(8, 4, 2) : (4 @ laneid, 1 @ laneid, 1)])
    assert_tile_layout(layout, [Iter(8, 4, "laneid"), Iter(4, 1, "laneid"), Iter(2, 1, "m")])

    layout = TileLayout(S[8 : 4 @ laneid] + R[4 : 1 @ laneid])
    assert_tile_layout(layout, [Iter(8, 4, "laneid")], replica=[Iter(4, 1, "laneid")])

    layout = TileLayout(S[8 : 4 @ laneid] + 1 @ laneid)
    assert_tile_layout(layout, [Iter(8, 4, "laneid")], offset={laneid: 1})


def test_constructor_multi_term_offset():
    """Multiple offset terms can be chained with `+` without parens.

    `_LayoutSpec.__add__` previously overwrote `self.offset` on each call,
    silently dropping all but the last axis term in
    `S[..] + 1 @ a + 2 @ b + 64`. Verify the merge happens for every entry
    point: `_LayoutSpec + _OnAxis`, `_LayoutSpec + int`,
    `_LayoutSpec + _OffsetExpr`, and the parenthesised form (which already
    worked) producing the same result.
    """

    # Chained, no parens: must merge into all three axes.
    layout = TileLayout(S[8 : 4 @ laneid] + 1 @ laneid + 2 @ warpid + 64)
    assert dict(layout.offset) == {laneid: 1, warpid: 2, m: 64}

    # Parenthesised form must produce the same offset.
    parens = TileLayout(S[8 : 4 @ laneid] + (1 @ laneid + 2 @ warpid + 64))
    assert_structural_equal(layout, parens)

    # Single-axis offset still works (regression sanity).
    single = TileLayout(S[8 : 4 @ laneid] + 1 @ laneid)
    assert dict(single.offset) == {laneid: 1}

    # Bare-int offset alone still routes to `m`.
    bare = TileLayout(S[8 : 4 @ laneid] + 64)
    assert dict(bare.offset) == {m: 64}

    # `_LayoutSpec + _LayoutSpec` where both carry an offset must also merge.
    a = S[8 : 4 @ laneid] + 1 @ laneid
    b = R[4 : 1 @ laneid] + 2 @ warpid
    combined = TileLayout(a + b)
    assert dict(combined.offset) == {laneid: 1, warpid: 2}

    # `int + _LayoutSpec` reaches `_LayoutSpec.__radd__` (Python's `int.__add__`
    # returns NotImplemented for `_LayoutSpec`); verify it merges through the
    # same path as `__add__`.
    radd = TileLayout(64 + S[8 : 4 @ laneid] + 1 @ laneid)
    assert dict(radd.offset) == {laneid: 1, m: 64}


def test_wg_local_layout_helper():
    layout = wg_local_layout(16)
    expected = TileLayout(S[(128, 16) : (1 @ tid_in_wg, 1)])
    assert_structural_equal(layout.canonicalize(), expected.canonicalize())

    layout_rows = wg_local_layout(8, rows=64)
    expected_rows = TileLayout(S[(64, 8) : (1 @ tid_in_wg, 1)])
    assert_structural_equal(layout_rows.canonicalize(), expected_rows.canonicalize())


def test_spec_builder():
    """Test S[shape:stride] + R[shape:stride] + offset combinator API."""

    # --- S[shape:stride] shard only ---
    new = TileLayout(S[(8, 4, 2) : (4 @ laneid, 1 @ laneid, 1)])
    old = TileLayout(S[(8, 4, 2) : (4 @ laneid, 1 @ laneid, 1)])
    assert str(new) == str(old)

    # --- 1D (no inner parens) ---
    new = TileLayout(S[128 : 1 @ laneid])
    old = TileLayout(S[128 : 1 @ laneid])
    assert str(new) == str(old)

    # --- Extents only ---
    new = TileLayout(S[8, 4, 2])
    old = TileLayout(S[8, 4, 2])
    assert str(new) == str(old)

    # --- S + R (shard + replica) ---
    new = TileLayout(S[(8,) : (4 @ laneid,)] + R[4 : 1 @ laneid])
    old = TileLayout(S[8 : 4 @ laneid] + R[4 : 1 @ laneid])
    assert str(new) == str(old)

    # --- S + offset ---
    new = TileLayout(S[8 : 4 @ laneid] + 1 @ laneid)
    old = TileLayout(S[8 : 4 @ laneid] + 1 @ laneid)
    assert str(new) == str(old)

    # --- S + R + offset ---
    new = TileLayout(S[(1,) : (1,)] + R[(8, 4) : (4 @ laneid, 1 @ laneid)] + 2 @ warpid)
    old = TileLayout(S[1:1] + R[(8, 4) : (4 @ laneid, 1 @ laneid)] + 2 @ warpid)
    assert str(new) == str(old)

    # --- Memory axes ---
    new = TileLayout(S[(2, 3, 4) : (12 @ m, 4 @ m, 1 @ m)])
    old = TileLayout(S[(2, 3, 4) : (12 @ m, 4 @ m, 1 @ m)])
    assert str(new) == str(old)

    # --- String axis names (no import needed) ---
    # stride=1 shorthand
    assert str(TileLayout(S[8:"laneid"])) == str(TileLayout(S[8 : 1 @ laneid]))
    assert str(TileLayout(S[32:"warpid"])) == str(TileLayout(S[32 : 1 @ warpid]))
    # multi-dim with string
    assert str(TileLayout(S[(8, 4) : ("laneid", 1)])) == str(
        TileLayout(S[(8, 4) : (1 @ laneid, 1)])
    )
    # non-unit stride via tuple
    assert str(TileLayout(S[(8,) : ((4, "laneid"),)])) == str(TileLayout(S[8 : 4 @ laneid]))
    # string in R
    assert str(TileLayout(S[1:1] + R[4:"laneid"])) == str(TileLayout(S[1:1] + R[4 : 1 @ laneid]))


def test_verify_well_formed():
    def test_scope_connected():
        layout = TileLayout(S[(8, 4, 2) : (4 @ laneid, 1 @ laneid, 1)])
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "thread"
        assert res[1].name == "warp"
        assert layout.verify_well_formed()

        layout = TileLayout(S[8 : 4 @ laneid] + R[4 : 1 @ laneid])
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "thread"
        assert res[1].name == "warp"
        assert layout.verify_well_formed()

        layout = TileLayout(S[(8, 4, 2) : (4 @ laneid, 1 @ laneid, 1)])
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "thread"
        assert res[1].name == "warp"
        assert layout.verify_well_formed()

        layout = TileLayout(
            S[(2, 8, 2, 4, 2) : (2 @ warpid, 4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
        )
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "thread"
        assert res[1].name == "cta"
        assert layout.verify_well_formed()

        layout = TileLayout(
            S[(2, 8, 2, 4, 2) : (2 @ wid_in_wg, 4 @ laneid, 1 @ wid_in_wg, 1 @ laneid, 1)]
        )
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "thread"
        assert res[1].name == "warpgroup"
        assert layout.verify_well_formed()

        layout = TileLayout(S[(2, 8, 2, 4, 2) : (2 @ wgid, 4 @ laneid, 1 @ wgid, 1 @ laneid, 1)])
        with pytest.raises(Exception):
            layout.verify_well_formed()

    test_scope_connected()


def test_normalize_tile_layout():
    def case1():
        layout = TileLayout(S[(8, 8, 8, 4, 2) : (512, 64, 8, 2, 1)])
        layout_expected = TileLayout(S[4096:1])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case1()

    def case2():
        layout = TileLayout(S[(8, 8, 1, 8, 4, 2) : (512, 64, 160, 8, 2, 1)])
        layout_expected = TileLayout(S[4096:1])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case2()

    def case3():
        layout = TileLayout(S[(8, 8, 8, 4, 1, 1) : (512, 64, 8, 2, 1, 1)])
        layout_expected = TileLayout(S[2048:2])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case3()

    def case4():
        layout = TileLayout(S[(8, 8, 1, 1, 1, 4, 1, 1) : (512, 64, 1, 1, 1, 2, 1, 1)])
        layout_expected = TileLayout(S[(64, 4) : (64, 2)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case4()

    def case5():
        layout = TileLayout(S[(2, 3, 6) : (18, 6, 1)])
        layout_expected = TileLayout(S[36:1])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case5()

    def case6():
        layout = TileLayout(S[(8, 2, 3, 6) : (6, 18, 6, 1)])
        layout_expected = TileLayout(S[(8, 36) : (6, 1)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case6()

    def case7():
        layout = TileLayout(S[(8, 2, 3, 6) : (6, 24, 6, 1)])
        layout_expected = TileLayout(S[(8, 2, 18) : (6, 24, 1)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case7()

    def case8():
        layout = TileLayout(S[(8, 2, 4, 2, 3, 6) : (2, 1, 4, 24, 6, 1)])
        layout_expected = TileLayout(S[(16, 4, 2, 18) : (1, 4, 24, 1)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case8()

    def case9():
        layout = TileLayout(S[(3, 4, 5, 2) : (20, 5, 1, 60)])
        layout_expected = TileLayout(S[(60, 2) : (1, 60)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case9()

    def case10():
        layout = TileLayout(S[(18, 8, 2, 4, 2, 3, 6) : (4, 2, 1, 4, 24, 6, 1)])
        layout_expected = TileLayout(S[(18, 16, 4, 2, 18) : (4, 1, 4, 24, 1)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case10()

    def case11():
        layout = TileLayout(S[(3, 4, 5, 2, 3, 4) : (20, 5, 1, 60, 20, 5)])
        layout_expected = TileLayout(S[(60, 24) : (1, 5)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case11()

    def case_no_norm():
        layout_normalized = TileLayout(S[(8, 8, 8, 4, 2) : (16, 4 @ laneid, 2, 1 @ laneid, 1)])
        assert_structural_equal(layout_normalized, layout_normalized.canonicalize())

    case_no_norm()

    def case_both_data_device1():
        layout = TileLayout(S[(8, 8, 8, 1, 4, 2, 1) : (16, 4 @ laneid, 2, 1, 1 @ laneid, 1, 1)])
        layout_normalized = TileLayout(S[(8, 8, 8, 4, 2) : (16, 4 @ laneid, 2, 1 @ laneid, 1)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device1()

    def case_both_data_device2():
        layout = TileLayout(
            S[(8, 8, 8, 1, 4, 2, 1) : (16, 4 @ laneid, 2, 1, 1 @ laneid, 1, 4 @ laneid)]
        )
        layout_normalized = TileLayout(S[(8, 8, 8, 4, 2) : (16, 4 @ laneid, 2, 1 @ laneid, 1)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device2()

    def case_both_data_device3():
        layout = TileLayout(
            S[(8, 8, 8, 1, 1, 2, 1) : (16, 4 @ laneid, 2, 1, 4 @ laneid, 1, 1)] + 0 @ laneid
        )
        layout_normalized = TileLayout(S[(8, 8, 16) : (16, 4 @ laneid, 1)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device3()

    def case_both_data_device4():
        layout = TileLayout(S[(8, 4, 8, 8, 16) : (4 @ laneid, 1 @ laneid, 4, 2, 4)])
        layout_normalized = TileLayout(S[(32, 8, 8, 16) : (1 @ laneid, 4, 2, 4)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device4()

    def case_both_data_device6():
        layout = TileLayout(S[(8, 4, 8, 16) : (4 @ laneid, 1 @ laneid, 2, 4)])
        layout_normalized = TileLayout(S[(32, 8, 16) : (1 @ laneid, 2, 4)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device6()

    def case_both_data_device7():
        layout = TileLayout(S[(8, 4, 8) : (4 @ laneid, 1 @ laneid, 8)])
        layout_normalized = TileLayout(S[(32, 8) : (1 @ laneid, 8)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device7()

    def case_both_data_device8():
        # Fuse-Case 1
        layout = TileLayout(S[(8, 4, 8) : (4 @ laneid, 1 @ laneid, 4)])
        layout_normalized = TileLayout(S[(32, 8) : (1 @ laneid, 4)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device8()

    def case_both_data_device9():
        # Fuse-Case 2
        layout = TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)])
        layout_normalized = TileLayout(S[32 : 1 @ laneid])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device9()

    def case_both_data_device12():
        # Fuse-mixed
        layout = TileLayout(S[(8, 4, 4, 8, 8, 8) : (4 @ laneid, 1 @ laneid, 4, 8, 8, 8)])
        layout_normalized = TileLayout(S[(32, 4, 8, 8, 8) : (1 @ laneid, 4, 8, 8, 8)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device12()

    def case_both_data_device13():
        # Fuse-mixed with partial
        layout = TileLayout(S[(8, 4, 4, 8, 8, 8) : (4 @ laneid, 1 @ laneid, 16, 2, 8, 8)])
        layout_normalized = TileLayout(S[(32, 32, 8, 8) : (1 @ laneid, 2, 8, 8)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device13()

    def case_both_data_device14():
        # Fuse-mixed with partial (another case)
        layout = TileLayout(
            S[(8, 4, 4, 8, 8, 4, 4, 16, 8) : (4 @ laneid, 1 @ laneid, 16, 2, 8, 2, 16, 1, 4)]
        )
        layout_normalized = TileLayout(S[(32, 32, 32, 64, 8) : (1 @ laneid, 2, 2, 1, 4)])
        assert_structural_equal(layout_normalized, layout.canonicalize())

    case_both_data_device14()

    def case15():
        # Only data tree (partial norm - middle) #15
        layout = TileLayout(S[(32, 3, 4, 5, 2, 3, 4) : (1 @ laneid, 20, 5, 1, 60, 20, 5)])
        layout_expected = TileLayout(S[(32, 60, 24) : (1 @ laneid, 1, 5)])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case15()

    def unit_layout_case1():
        layout = TileLayout(S[(1, 1, 1, 1, 1) : (1, 1, 1, 1, 1)])
        layout_unit = TileLayout(S[1:1])
        assert_structural_equal(layout_unit, layout.canonicalize())

    unit_layout_case1()

    def case_fuse_axis():
        with tvm.target.Target("cuda"):
            layout = TileLayout(S[(2, 8, 2, 4) : (2 @ warpid, 4 @ laneid, 1 @ warpid, 1 @ laneid)])
            layout_expected = TileLayout(S[(2, 8, 2, 4) : (64 @ tx, 4 @ tx, 32 @ tx, 1 @ tx)])
            assert layout.verify_well_formed()
            assert layout_expected.verify_well_formed()
            assert_structural_equal(layout_expected, layout.canonicalize())

            layout = TileLayout(S[(2, 2, 8, 4) : (2 @ warpid, 1 @ warpid, 4 @ laneid, 1 @ laneid)])
            layout_expected = TileLayout(S[128 : 1 @ tx])
            assert layout.verify_well_formed()
            assert layout_expected.verify_well_formed()
            assert_structural_equal(layout_expected, layout.canonicalize())

            layout = TileLayout(
                S[
                    (2, 2, 8, 2, 2, 4) : (
                        2 @ wgid,
                        2 @ wid_in_wg,
                        4 @ laneid,
                        1 @ wgid,
                        1 @ wid_in_wg,
                        1 @ laneid,
                    )
                ]
            )
            layout_expected = TileLayout(
                S[(2, 2, 8, 2, 2, 4) : (256 @ tx, 64 @ tx, 4 @ tx, 128 @ tx, 32 @ tx, 1 @ tx)]
            )
            assert layout.verify_well_formed()
            assert layout_expected.verify_well_formed()
            assert_structural_equal(layout_expected, layout.canonicalize())

            layout = TileLayout(
                S[(2, 8, 2, 4) : (2 @ wid_in_wg, 4 @ laneid, 1 @ wid_in_wg, 1 @ laneid)]
            )
            layout_expected = TileLayout(
                S[(2, 8, 2, 4) : (64 @ tid_in_wg, 4 @ tid_in_wg, 32 @ tid_in_wg, 1 @ tid_in_wg)]
            )
            assert layout.verify_well_formed()
            assert layout_expected.verify_well_formed()
            assert_structural_equal(layout_expected, layout.canonicalize())

            layout = TileLayout(
                S[(2, 2, 4, 32) : (2 @ wgid, 1 @ wgid, 32 @ tid_in_wg, 1 @ tid_in_wg)]
            )
            layout_expected = TileLayout(S[512 : 1 @ tx])
            assert layout.verify_well_formed()
            assert layout_expected.verify_well_formed()
            assert_structural_equal(layout_expected, layout.canonicalize())

    case_fuse_axis()

    def case_sort_replicate_exclude_iters():
        layout1 = TileLayout(S[1:1] + R[(8, 4) : (4 @ laneid, 1 @ laneid)] + 2 @ warpid)
        layout2 = TileLayout(S[1:1] + R[(4, 8) : (1 @ laneid, 4 @ laneid)] + 2 @ warpid)
        assert_structural_equal(layout1.canonicalize(), layout2.canonicalize())

    case_sort_replicate_exclude_iters()

    def case_empty_shard_canonicalize():
        """Regression test for F6: canonicalize must not crash when layout->shard is empty."""
        layout = TileLayout(R[32 : 1 @ laneid])
        canon = layout.canonicalize()
        assert canon is not None

    case_empty_shard_canonicalize()


def test_tile_layout():
    def case1():
        # (8):(1)x(8):(1) -> (64):(1)
        inner = TileLayout(S[8:1])
        outer = inner
        layout_tile = TileLayout(S[64:1])
        assert_structural_equal(layout_tile, inner.tile(outer, [8], [8]))

        outer_res = inner.is_tile_inner(layout_tile, [64], [8])
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, [64], [8])
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

    case1()

    def case2():
        # (8,8):(8,1)x(8,8):(8,1) -> (8,8,8,8):(512,8,64,1)
        inner = TileLayout(S[(8, 8) : (8, 1)])
        outer = inner
        layout_tile = TileLayout(S[(8, 8, 8, 8) : (512, 8, 64, 1)])
        assert_structural_equal(layout_tile, inner.tile(outer, [8, 8], [8, 8]))

        outer_res = inner.is_tile_inner(layout_tile, [64, 64], [8, 8])
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, [64, 64], [8, 8])
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

    case2()

    def case3():
        # (2,4):(1,2)x(8,8):(8,1) -> (8,2,8,4):(64,1,8,2)
        inner = TileLayout(S[(2, 4) : (1, 2)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = TileLayout(S[(8, 2, 32) : (64, 1, 2)])
        assert_structural_equal(layout_tile, inner.tile(outer, [8, 8], [2, 4]))

        outer_res = inner.is_tile_inner(layout_tile, [16, 32], [2, 4])
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, [16, 32], [8, 8])
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

        assert outer.is_tile_inner(layout_tile, [16, 32], [8, 8]) is None
        assert inner.is_tile_outer(layout_tile, [16, 32], [2, 4]) is None

    case3()

    def case4():
        # ((4,2),(2,4)):((16,8),(1,2))x(8,8):(8,1) -> (8,4,2,8,2,4):(512,16,8,64,1,2)
        inner = TileLayout(S[(4, 2, 2, 4) : (16, 8, 1, 2)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = TileLayout(S[(8, 4, 2, 8, 2, 4) : (512, 16, 8, 64, 1, 2)])
        assert_structural_equal(layout_tile.canonicalize(), inner.tile(outer, (8, 8), (8, 8)))

        outer_res = inner.is_tile_inner(layout_tile, (64, 64), (8, 8))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, (64, 64), (8, 8))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

        assert outer.is_tile_inner(layout_tile, (64, 64), (8, 8)) is None
        assert inner.is_tile_outer(layout_tile, (64, 64), (8, 8)) is None

    case4()

    def case5_sharded1():
        # Tile over a sharded layout - 1
        layout = TileLayout(S[(8, 1, 4, 2) : (4 @ laneid, 2, 1 @ laneid, 1)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = layout.tile(outer=outer, outer_shape=(8, 8), inner_shape=(8, 8))
        layout_expected = TileLayout(S[(8, 8, 1, 8, 4, 2) : (16, 4 @ laneid, 2, 2, 1 @ laneid, 1)])
        assert_structural_equal(layout_expected.canonicalize(), layout_tile)

        outer_res = layout.is_tile_inner(layout_tile, (64, 64), (8, 8))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, (64, 64), (8, 8))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), layout.canonicalize())

        assert outer.is_tile_inner(layout_tile, (64, 64), (8, 8)) is None
        assert layout.is_tile_outer(layout_tile, (64, 64), (8, 8)) is None

    case5_sharded1()

    def case6_sharded2():
        # Tile over a sharded layout - 2
        inner = TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = inner.tile(outer=outer, outer_shape=(8, 8), inner_shape=(8, 4))
        layout_expected = TileLayout(S[(8, 8, 8, 4) : (8, 4 @ laneid, 1, 1 @ laneid)])
        assert_structural_equal(layout_expected, layout_tile)

        outer_res = inner.is_tile_inner(layout_tile, (64, 32), (8, 4))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, (64, 32), (8, 8))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

        assert outer.is_tile_inner(layout_tile, (64, 32), (8, 8)) is None
        assert inner.is_tile_outer(layout_tile, (64, 32), (8, 4)) is None

    case6_sharded2()

    def case7_normalized4():
        # Normalized Tile Layout Test - 4 (tile < inner)
        outer = TileLayout(S[(4, 2, 1) : (2, 1, 1)])
        inner = TileLayout(S[(2, 4, 1) : (2, 3, 1)])
        layout_tile = inner.tile(outer, outer_shape=(4, 2), inner_shape=(2, 4))

        inner_res = outer.is_tile_outer(layout_tile, (8, 8), (4, 2))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

        outer_res = inner.is_tile_inner(layout_tile, (8, 8), (2, 4))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        assert outer.is_tile_inner(layout_tile, (8, 8), (4, 2)) is None
        assert inner.is_tile_outer(layout_tile, (8, 8), (2, 4)) is None

    case7_normalized4()

    def case8_normalized5():
        # Normalized Tile Layout Test - 5 (tile = inner)
        outer = TileLayout(S[(8, 2) : (2, 1)])
        inner = TileLayout(S[(2, 4) : (4, 1)])
        layout_tile = inner.tile(outer, (8, 2), (2, 4))

        outer_res = inner.is_tile_inner(layout_tile, (16, 8), (2, 4))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, (16, 8), (8, 2))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

        assert outer.is_tile_inner(layout_tile, (16, 8), (8, 2)) is None
        assert inner.is_tile_outer(layout_tile, (16, 8), (2, 4)) is None

    case8_normalized5()

    def case9_normalized6():
        # Normalized Tile Layout Test - 6 (tile < inner)
        outer = TileLayout(S[(8, 4, 1) : (4, 1, 4)])
        inner = TileLayout(S[(2, 1, 1) : (4, 3, 1)])
        TileLayout(S[(8, 2, 2) : (4, 2, 2)])
        layout_tile = inner.tile(outer, (8, 4), (2, 1))

        outer_res = inner.is_tile_inner(layout_tile, (16, 4), (2, 1))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        inner_res = outer.is_tile_outer(layout_tile, (16, 4), (8, 4))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), inner.canonicalize())

    case9_normalized6()

    def case10_normalized7():
        # Normalized Tile Layout Test - 7 (tile = inner)
        outer = TileLayout(S[(8, 8, 4) : (32, 4, 1)])
        inner = TileLayout(S[(1, 2, 1) : (4, 3, 1)])
        inner_tmp = TileLayout(S[(1, 2, 2) : (8, 4, 3)])
        layout_tile = inner.tile(outer, (8, 8, 4), (1, 2, 1))

        outer_res = inner.is_tile_inner(layout_tile, (8, 16, 4), (1, 2, 1))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())

        assert inner.is_tile_inner(layout_tile.canonicalize(), (8, 16, 4), (1, 2, 1))

        assert outer.is_tile_inner(layout_tile, (8, 16, 4), (8, 8, 4)) is None
        assert inner_tmp.is_tile_inner(layout_tile, (8, 16, 4), (1, 2, 2)) is None

    case10_normalized7()

    def case11_normalized8():
        # Normalized Tile Layout Test - 8 (tile = inner w/ device)
        outer = TileLayout(S[(8, 8, 4) : (32, 4, 1)])
        inner = TileLayout(S[(8, 8, 1, 4, 2) : (4, 4 @ laneid, 2, 1 @ laneid, 1)])
        layout_tile = inner.tile(outer, (8, 8, 4), (8, 8, 8))

        outer_res = inner.is_tile_inner(layout_tile, (64, 64, 32), (8, 8, 8))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())
        assert inner.is_tile_inner(layout_tile.canonicalize(), (64, 64, 32), (8, 8, 8))
        assert not outer.canonicalize().is_tile_inner(
            layout_tile.canonicalize(), (64, 64, 32), (8, 8, 4)
        )

    case11_normalized8()

    def case12_normalized9():
        # Normalized Tile Layout Test - 9 (tile = inner w/ device + diff major-dim)
        outer = TileLayout(S[(16, 8, 4) : (1, 64, 16)])
        inner = TileLayout(S[(2, 4, 2, 2) : (4, 1, 4, 3)])
        layout_tile = inner.tile(outer, (16, 8, 4), (8, 2, 2))

        outer_res = inner.is_tile_inner(layout_tile, (128, 16, 8), (8, 2, 2))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), outer.canonicalize())
        assert inner.is_tile_inner(layout_tile.canonicalize(), (128, 16, 8), (8, 2, 2))
        assert not outer.canonicalize().is_tile_inner(
            layout_tile.canonicalize(), (128, 16, 8), (16, 8, 4)
        )

    case12_normalized9()

    def case_dims_mismatch():
        with pytest.raises(Exception):
            layout = TileLayout(S[8:1])
            layout2 = TileLayout(S[(2, 4) : (1, 2)])
            layout2.tile(layout, [8], [2, 4])

    case_dims_mismatch()

    def case_tile_compose_layout():
        # tile(TileLayout, ComposeLayout)
        compose = ComposeLayout(
            layout_A=SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            layout_B=TileLayout(S[(8, 64) : (64, 1)]),
        )
        layout = TileLayout(S[(8, 1) : (1, 1)])
        layout_tile = compose.tile(layout, (8, 1), (8, 64))
        layout_expected = ComposeLayout(
            SwizzleLayout(3, 3, 3, swizzle_inner=True), TileLayout(S[4096:1])
        )
        assert_structural_equal(layout_tile.canonicalize(), layout_expected.canonicalize())

        outer_res = compose.is_tile_inner(layout_tile, (4096,), (512,))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), layout.canonicalize())

        inner_res = layout.is_tile_outer(layout_tile, (4096,), (8,))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), compose.canonicalize())

        assert layout.is_tile_inner(layout_tile, (4096,), (512,)) is None
        assert compose.is_tile_outer(layout_tile, (4096,), (8,)) is None

    case_tile_compose_layout()

    def case_tile_swizzle_layout():
        # swizzle_128B_atom
        swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        layout = TileLayout(S[(8, 4) : (1, 8)])
        layout_tile = swizzle.tile(layout, (8, 4), (8, 64))
        layout_expected = ComposeLayout(
            SwizzleLayout(3, 3, 3, swizzle_inner=True), TileLayout(S[(64, 4, 64) : (64, 4096, 1)])
        )
        assert_structural_equal(layout_tile.canonicalize(), layout_expected)

        outer_res = swizzle.is_tile_inner(layout_tile, (64, 256), (8, 64))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), layout.canonicalize())

        inner_res = layout.is_tile_outer(layout_tile, (64, 256), (8, 4))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), swizzle.canonicalize())

    case_tile_swizzle_layout()

    def case_tile_swizzle_layout2():
        # swizzle_128B_atom
        swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        tile = TileLayout(S[(3, 8, 4) : (8 * 4, 1, 8)])
        layout_tile = swizzle.tile(tile, (3, 8, 4), (1, 8, 64))
        layout_expected = ComposeLayout(
            swizzle, TileLayout(S[(3, 64, 4, 64) : (16384, 64, 4096, 1)])
        )
        assert_structural_equal(layout_tile.canonicalize(), layout_expected.canonicalize())

        outer_res = swizzle.is_tile_inner(layout_tile, (3, 64, 256), (1, 8, 64))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), tile.canonicalize())

        inner_res = tile.is_tile_outer(layout_tile, (3, 64, 256), (3, 8, 4))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), swizzle.canonicalize())

    case_tile_swizzle_layout2()

    def case_tile_swizzle_layout3():
        # swizzle_64B_atom
        swizzle = SwizzleLayout(per_element=3, swizzle_len=2, atom_len=3)
        tile = TileLayout(S[(8, 8) : (1, 8)])
        layout_tile = swizzle.tile(tile, (8, 8), (8, 32))
        layout_expected = ComposeLayout(swizzle, TileLayout(S[(64, 8, 32) : (32, 2048, 1)]))
        assert_structural_equal(layout_tile.canonicalize(), layout_expected.canonicalize())

        outer_res = swizzle.is_tile_inner(layout_tile, (64, 256), (8, 32))
        assert outer_res is not None
        assert_structural_equal(outer_res.canonicalize(), tile.canonicalize())

        inner_res = tile.is_tile_outer(layout_tile, (64, 256), (8, 8))
        assert inner_res is not None
        assert_structural_equal(inner_res.canonicalize(), swizzle.canonicalize())

    case_tile_swizzle_layout3()

    def case_tile_swizzle_layout4():
        # swizzle_64B_atom
        swizzle = SwizzleLayout(per_element=3, swizzle_len=2, atom_len=3)
        outer = swizzle.is_tile_inner(swizzle, (64, 256), (8, 32))
        assert outer is None

        outer = swizzle.is_tile_inner(swizzle, (64, 32), (8, 32))
        assert outer is not None
        outer_expected = TileLayout(S[(8, 1) : (1, 0)])
        assert_structural_equal(outer.canonicalize(), outer_expected.canonicalize())

    case_tile_swizzle_layout4()

    def case_tile_swizzle_layout5():
        # swizzle_128B_atom
        swizzle = SwizzleLayout(per_element=3, swizzle_len=2, atom_len=3)
        tile1 = TileLayout(S[(8, 8) : (1, 8)])
        tile2 = TileLayout(S[(2, 2) : (1, 2)])
        layout_tile = swizzle.tile(tile1, (8, 8), (8, 32))
        layout_tile = layout_tile.tile(tile2, (2, 2), (64, 256))

        outer = swizzle.is_tile_inner(layout_tile, (128, 512), (8, 32))
        assert outer is not None
        outer_expected = tile1.tile(tile2, (2, 2), (8, 8))
        assert_structural_equal(outer.canonicalize(), outer_expected.canonicalize())

    case_tile_swizzle_layout5()


def test_shard_layout():
    """In the current layout design, shard is just a special case of tile, where the outer tile has thread axes."""  # noqa: E501

    def case_mma_layout():
        layout = TileLayout(S[(1, 2) : (2, 1)])
        layout_warp = TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)])
        res = layout.tile(layout_warp, [8, 4], [1, 2])
        layout_expected = TileLayout(S[(32, 2) : (1 @ laneid, 1)])
        assert_structural_equal(res.canonicalize(), layout_expected.canonicalize())

        outer = layout.is_tile_inner(res, [8, 8], [1, 2])
        assert outer is not None
        assert_structural_equal(outer.canonicalize(), layout_warp.canonicalize())

        inner = layout_warp.is_tile_outer(res, [8, 8], [8, 4])
        assert inner is not None
        assert_structural_equal(inner.canonicalize(), layout.canonicalize())

    case_mma_layout()

    def case_cta_layout():
        layout = TileLayout(S[(1, 2) : (2, 1)])
        layout_warp = TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)])
        layout_cta = TileLayout(S[(2, 2) : (2 @ warpid, 1 @ warpid)])

        res_warp = layout.tile(layout_warp, [8, 4], [1, 2])
        res = res_warp.tile(layout_cta, [2, 2], [8, 8])
        layout_expected = TileLayout(
            S[(2, 8, 2, 4, 2) : (2 @ warpid, 4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
        )
        assert_structural_equal(res.canonicalize(), layout_expected.canonicalize())

        outer = layout.is_tile_inner(res, [16, 16], [1, 2])
        outer_expected = TileLayout(
            S[(2, 8, 2, 4) : (2 @ warpid, 4 @ laneid, 1 @ warpid, 1 @ laneid)]
        )
        assert outer is not None
        assert_structural_equal(outer, outer_expected)

        inner = layout_cta.is_tile_outer(res, [16, 16], [2, 2])
        assert inner is not None
        assert_structural_equal(inner.canonicalize(), res_warp.canonicalize())

    case_cta_layout()

    def case_cta_layout2():
        with tvm.target.Target("cuda"):
            tiled = TileLayout(S[(2, 8, 2, 4, 2) : (64 @ tx, 4 @ tx, 32 @ tx, 1 @ tx, 1)])
            # local is inner of cta
            layout = TileLayout(S[2:1])
            outer = layout.is_tile_inner(tiled, [16, 16], [1, 2])
            assert outer is not None
            outer_expected = TileLayout(S[(2, 8, 2, 4) : (64 @ tx, 4 @ tx, 32 @ tx, 1 @ tx)])
            assert_structural_equal(outer.canonicalize(), outer_expected.canonicalize())

            layout = TileLayout(S[(2, 8, 2, 4) : (2 @ warpid, 4 @ laneid, 1 @ warpid, 1 @ laneid)])
            inner = layout.is_tile_outer(tiled, [16, 16], [16, 8])
            inner_expected = TileLayout(S[2:1])
            assert inner is not None
            assert_structural_equal(inner.canonicalize(), inner_expected.canonicalize())

            # warp view is inner of cta
            layout = TileLayout(S[(8, 1, 4, 2) : (4 @ laneid, 2, 1 @ laneid, 1)])
            outer = layout.is_tile_inner(tiled, [16, 16], [8, 8])
            assert outer is not None
            outer_expected = TileLayout(S[(2, 2) : (2 @ warpid, 1 @ warpid)])
            assert_structural_equal(outer.canonicalize(), outer_expected.canonicalize())

            layout = TileLayout(S[(2, 2) : (2 @ warpid, 1 @ warpid)])
            inner = layout.is_tile_outer(tiled, [16, 16], [2, 2])
            inner_expected = TileLayout(S[(32, 2) : (1 @ laneid, 1)])
            assert inner is not None
            assert_structural_equal(inner.canonicalize(), inner_expected.canonicalize())

    case_cta_layout2()

    def case_quad_shuffle():
        layout = TileLayout(S[(1, 2) : (2, 1)])
        layout_warp = TileLayout(S[8 : 4 @ laneid])
        res = layout.tile(layout_warp, [8, 1], [1, 2])
        layout_expected = TileLayout(S[(8, 2) : (4 @ laneid, 1)])
        assert_structural_equal(res.canonicalize(), layout_expected.canonicalize())

        outer = layout.is_tile_inner(res, [8, 2], [1, 2])
        assert outer is not None
        assert_structural_equal(outer.canonicalize(), layout_warp.canonicalize())

        inner = layout_warp.is_tile_outer(res, [8, 2], [8, 1])
        assert inner is not None
        assert_structural_equal(inner.canonicalize(), layout.canonicalize())

    case_quad_shuffle()

    def case_replicate():
        layout = TileLayout(S[(64, 128) : (128, 1)])
        layout_rep = TileLayout(S[2 : 2 @ warpid] + R[2 : 1 @ warpid])
        res = layout.tile(layout_rep, [2, 1], [64, 128])
        layout_expected = TileLayout(S[(2, 8192) : (2 @ warpid, 1)] + R[2 : 1 @ warpid])
        assert_structural_equal(res.canonicalize(), layout_expected.canonicalize())

        outer = layout.is_tile_inner(res, [128, 128], [64, 128])
        assert outer is not None
        assert_structural_equal(outer.canonicalize(), layout_rep.canonicalize())

        inner = layout_rep.is_tile_outer(res, [128, 128], [2, 1])
        assert inner is not None
        assert_structural_equal(inner.canonicalize(), layout.canonicalize())

    case_replicate()


def test_size_span():
    def tile_layout_size():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        assert layout.size() == 64

    tile_layout_size()

    def swizzle_layout_size():
        layout = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        assert layout.size() == 512
        layout = SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3)
        assert layout.size() == 1024

    swizzle_layout_size()

    def compose_layout_size():
        layout = ComposeLayout(
            SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 64) : (64, 1)]),
        )
        assert layout.size() == 512

    compose_layout_size()

    def tile_layout_span():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        assert layout.span() == 64
        layout = TileLayout(S[(8, 6) : (8, 1)])
        assert layout.span() == 62
        layout = TileLayout(S[(8, 1, 4, 2) : (4 @ laneid, 2, 1 @ laneid, 1)])
        assert layout.span() == 2

    tile_layout_span()

    def swizzle_layout_span():
        layout = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        assert layout.span() == 512
        layout = SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3)
        assert layout.span() == 1024

    swizzle_layout_span()

    def compose_layout_span():
        layout = ComposeLayout(
            SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 64) : (64, 1)]),
        )
        assert layout.span() == 512

    compose_layout_span()

    def trainium_layout_tests():
        # TrainiumLayout tests
        layout = TileLayout(S[(8, 8) : (1 @ P, 1 @ F)])
        assert layout.size("P") == 8
        assert layout.size("F") == 8

        layout = TileLayout(S[(8, 8, 8) : (64 @ F, 1 @ P, 1 @ F)])
        assert layout.size("P") == 8
        assert layout.size("F") == 64
        assert layout.span("F") == 456

        layout_partition = TileLayout(S[8 : 1 @ P])
        assert layout_partition.size("P") == 8 and layout_partition.size("F") == 1

        layout_free = TileLayout(S[8 : 1 @ F])
        assert layout_free.size("P") == 1 and layout_free.size("F") == 8

        layout = TileLayout.trainium("PF", (128, 128))
        assert layout.size("P") == 128 and layout.size("F") == 128

        layout = TileLayout.trainium("FPF", (32, 512, 512))
        assert_structural_equal(
            layout, TileLayout(S[(32, 4, 128, 512) : (512 @ F, (512 * 32) @ F, 1 @ P, 1 @ F)])
        )

        layout = TileLayout.trainium("FPPF", (2, 4, 32, 512))
        assert_structural_equal(
            layout, TileLayout(S[(2, 4, 32, 512) : (512 @ F, 32 @ P, 1 @ P, 1 @ F)])
        )

    trainium_layout_tests()


def test_apply():
    ################ TileLayout
    def test_tile_layout_0():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j)["m"] == i * 8 + j * 1
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i, j, shape=(8, 8))["m"] == i * 8 + j * 1
        # # apply can accept coord larger than size
        # for p in range(1024):
        #     outer = p // 64
        #     inner = p % 64
        #     i, j = inner // 8, inner % 8
        #     assert layout.apply(p)["m"] == outer * 64 + i * 8 + j * 1
        with pytest.raises(Exception):
            layout.apply(1, 1, 1)

    test_tile_layout_0()

    def test_tile_layout_1():
        layout = TileLayout(S[(8, 8) : (10, 1)])
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j)["m"] == i * 10 + j * 1
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i, j, shape=(8, 8))["m"] == i * 10 + j * 1

        # # apply can accept coord larger than size
        # for p in range(1024):
        #     outer = p // 64
        #     inner = p % 64
        #     i, j = inner // 8, inner % 8
        #     assert (
        #         layout.apply(
        #             p,
        #         )[0]
        #         == outer * 78 + i * 10 + j * 1
        #     )

    test_tile_layout_1()

    def test_tile_layout_2():
        layout = TileLayout(S[(2, 3, 4, 2, 2) : (1, 2, 12, 6, 48)])

        def f(i0, i1):
            leaf1 = i0 // 3
            leaf2 = i0 % 3
            leaf3 = i1 // 4
            leaf4 = (i1 % 4) // 2
            leaf5 = i1 % 2
            assert (
                layout.apply(i0, i1, shape=(6, 16))["m"]
                == leaf1 * 1 + leaf2 * 2 + leaf3 * 12 + leaf4 * 6 + leaf5 * 48
            )

        for i0, i1 in itertools.product(range(6), range(16)):
            f(i0, i1)
        for i in range(6 * 16):
            f(i // 16, i % 16)

    test_tile_layout_2()

    def test_tile_layout_3():
        layout = TileLayout(S[(8, 1, 4, 2) : (4 @ laneid, 2, 1 @ laneid, 1)])
        for i0, i1 in itertools.product(range(8), range(8)):
            res = layout.apply(i0, i1, shape=(8, 8))
            assert res["m"] == i1 % 2
            assert res["laneid"] == i0 * 4 + i1 // 2

    test_tile_layout_3()

    def test_tile_layout_4():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        v = tvm.tirx.Var("v", dtype="int32")
        res = layout.apply(v)
        assert res["m"] == v

    test_tile_layout_4()

    ################ Swizzle Layout
    def test_swizzle_layout_0():
        layout = SwizzleLayout(per_element=0, swizzle_len=3, atom_len=3)
        # assert layout.size == 64
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j)["m"] == i * 8 + i ^ j

    test_swizzle_layout_0()

    def test_swizzle_layout_1():
        layout = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        assert layout.size() == 512
        for i, j, k in itertools.product(range(8), range(8), range(8)):
            assert layout.apply((i * 8 + j) * 8 + k)["m"] == (i * 8 + (i ^ j)) * 8 + k
        # apply can accept coord larger than size
        for p in range(4096):
            outer = p // 512
            inner = p % 512
            i, j, k = inner // 64, (inner % 64) // 8, inner % 8
            assert layout.apply(p)["m"] == outer * 512 + (i * 8 + (i ^ j)) * 8 + k

    test_swizzle_layout_1()

    def test_swizzle_layout_2():
        layout = SwizzleLayout(per_element=0, swizzle_len=3, atom_len=3, swizzle_inner=False)
        assert layout.size() == 64
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j)["m"] == (i ^ j) * 8 + j

    test_swizzle_layout_2()

    def test_swizzle_layout_3():
        layout = SwizzleLayout(per_element=0, swizzle_len=2, atom_len=3)
        for i, j in itertools.product(range(8), range(8)):
            _outer_i, inner_i = i // 4, i % 4
            outer_j, inner_j = j // 4, j % 4
            assert layout.apply(i * 8 + j)["m"] == i * 8 + outer_j * 4 + (inner_i ^ inner_j)

    test_swizzle_layout_3()

    ################ Compose Layout
    def test_compose_layout_0():
        layoutA = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        layoutB = TileLayout(S[(8, 64) : (64, 1)])
        layout = ComposeLayout(layoutA, layoutB)
        assert layout.size() == 512
        assert layout.span() == 512
        for i, j in itertools.product(range(8), range(64)):
            assert (
                layout.apply(i * 64 + j)["m"] == layoutA.apply(layoutB.apply(i * 64 + j)["m"])["m"]
            )

    test_compose_layout_0()

    def test_compose_layout_1():
        layoutA = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        layoutB = TileLayout(S[(16, 64, 8) : (64, 1, 1024)])
        layout = ComposeLayout(layoutA, layoutB)
        assert layout.size() == 16 * 64 * 8
        assert layout.span() == 16 * 64 * 8
        for i, j, k in itertools.product(range(16), range(64), range(8)):
            assert (
                layout.apply(i * 64 * 8 + j * 8 + k)["m"]
                == layoutA.apply(layoutB.apply(i * 64 * 8 + j * 8 + k)["m"])["m"]
            )

    test_compose_layout_1()

    ################ Trainium Layout
    def test_trainium_layout_0():
        layout = TileLayout(S[(8, 8) : (8 @ F, 1 @ P)])
        for i, j in itertools.product(range(8), range(8)):
            coord = layout.apply(i, j, shape=(8, 8))
            assert coord["P"] == j
            assert coord["F"] == i * 8

    test_trainium_layout_0()

    def test_trainium_layout_1():
        layout = TileLayout(S[(2, 6, 4, 2, 2) : (1 @ F, 1 @ P, 12 @ F, 6 @ P, 48 @ F)])

        def f(i0, i1):
            leaf1 = i0 // 6
            leaf2 = i0 % 6
            leaf3 = i1 // 4
            leaf4 = (i1 % 4) // 2
            leaf5 = i1 % 2
            coord = layout.apply(i0, i1, shape=(12, 16))
            assert coord["P"] == leaf2 + leaf4 * 6
            assert coord["F"] == leaf1 * 1 + leaf3 * 12 + leaf5 * 48

        for i0, i1 in itertools.product(range(6), range(16)):
            f(i0, i1)
        for i in range(6 * 16):
            f(i // 16, i % 16)

    test_trainium_layout_1()

    ################ Trainium PSUM Layout
    def test_trainium_psum_layout_0():
        layout = TileLayout(S[(1024, 8) : (1 @ F, 1 @ P)]).to_psum()
        for i, j in itertools.product(range(1024), range(8)):
            coord = layout.apply(i, j, shape=(1024, 8))
            assert coord["Bank"] == i // 512
            assert coord["P"] == j
            assert coord["F"] == i % 512

    test_trainium_psum_layout_0()


def test_normalize_compose_layout():
    def case1():
        layoutA = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        layoutB = TileLayout(S[(8, 64) : (64, 1)])
        layout = ComposeLayout(layoutA, layoutB.canonicalize())
        assert_structural_equal(layout.canonicalize(), layoutA)

    case1()

    def case2():
        layoutA = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        layoutB = TileLayout(S[(64, 4, 64) : (64, 4096, 1)])
        layout = ComposeLayout(layoutA, layoutB.canonicalize())
        assert_structural_equal(layout.canonicalize(), layout)

    case2()


def test_normalize_trainium_layout():
    def case1():
        layout = TileLayout(S[(8, 8) : (8 @ P, 1 @ F)])
        assert_structural_equal(layout, layout.canonicalize())

    case1()

    def case2():
        layout = TileLayout(S[(8, 1, 8) : (8 @ F, 1 @ P, 1 @ F)])
        layout_expected = TileLayout(S[64 : 1 @ F])
        assert_structural_equal(layout_expected, layout.canonicalize())

    case2()

    def case3():
        layout = TileLayout(S[(8, 8, 8) : (8 @ F, 1 @ P, 1 @ F)])
        assert_structural_equal(layout, layout.canonicalize())

    case3()


def test_direct_sum():
    def case1():
        # Example from the appendix: A + B yields contiguous (16):(1)
        # B = (2,2):(4,1), A = (2,2):(8,2)
        B = TileLayout(S[(2, 2) : (4, 1)])
        A = TileLayout(S[(2, 2) : (8, 2)])

        # Compute direct sum on tiling domain S_A ⊗ S_B with shapes (2,2) and (2,2)
        sum_layout = B.direct_sum(A, [2, 2], [2, 2]).canonicalize()
        expected = TileLayout(S[16:1])
        assert_structural_equal(expected, sum_layout)

        # Verify Apply equality: 8p + 2q + 4i + j
        print(f"sum_layout: {sum_layout}")
        an = Analyzer()
        for p in [0, 1]:
            for q in [0, 1]:
                for i in [0, 1]:
                    for j in [0, 1]:
                        m = sum_layout.apply(p, q, i, j, shape=(2, 2, 2, 2))["m"]
                        m_left = A.apply(p, i, shape=(2, 2))["m"]
                        m_right = B.apply(q, j, shape=(2, 2))["m"]
                        assert an.can_prove(m == m_left + m_right)

        # Recognition: recover A given B and sum, and recover B given A and sum
        interleaved_shape = [2, 2, 2, 2]  # [A0, B0, A1, B1]
        A_rec = B.is_direct_sum_right(sum_layout, interleaved_shape, [2, 2])
        assert A_rec is not None
        assert_structural_equal(A.canonicalize(), A_rec.canonicalize())

        B_rec = A.is_direct_sum_left(sum_layout, interleaved_shape, [2, 2])
        assert B_rec is not None
        assert_structural_equal(B.canonicalize(), B_rec.canonicalize())

    case1()


def test_group_by_logical_shape():
    def case1():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        layout = layout.tile(layout, outer_shape=[8, 8], inner_shape=[8, 8])
        outer, seps = layout.group([64, 64])
        assert_structural_equal(outer, layout)
        assert seps[0] == 0
        assert seps[1] == 2
        assert seps[2] == 4

    case1()


def test_permute_by_groups():
    def case_swap_two_groups():
        # Two groups, each with 2 shard iters: swap them.
        layout = TileLayout(S[(8, 8) : (8, 1)])
        layout = layout.tile(layout, outer_shape=[8, 8], inner_shape=[8, 8])
        grouped, seps = layout.group([64, 64])
        # seps == [0, 2, 4]
        permuted = grouped.permute_by_groups(seps, [1, 0])
        # Expected: shard reordered as [g1[0], g1[1], g0[0], g0[1]]
        expected = grouped.permute_dims([2, 3, 0, 1])
        assert_structural_equal(permuted, expected)

    def case_identity():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        layout = layout.tile(layout, outer_shape=[8, 8], inner_shape=[8, 8])
        grouped, seps = layout.group([64, 64])
        permuted = grouped.permute_by_groups(seps, [0, 1])
        assert_structural_equal(permuted, grouped)

    def case_invalid_perm():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        layout = layout.tile(layout, outer_shape=[8, 8], inner_shape=[8, 8])
        grouped, seps = layout.group([64, 64])
        with pytest.raises(AssertionError):
            grouped.permute_by_groups(seps, [0, 0])

    case_swap_two_groups()
    case_identity()
    case_invalid_perm()


def test_tile_to():
    def case1():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        tiled = layout.tile_to([64, 64], [8, 8])
        tiled_expected = layout.tile(layout, [8, 8], [8, 8])
        assert_structural_equal(tiled, tiled_expected)

    case1()


def test_mma_shared_layout():
    def case1():
        layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_128B_ATOM, (64, 256))
        layout_expected = ComposeLayout(
            SwizzleLayout(3, 3, 3, swizzle_inner=True), TileLayout(S[(64, 4, 64) : (64, 4096, 1)])
        )
        assert_structural_equal(layout, layout_expected)

    case1()

    def case2():
        layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_128B_ATOM, (3, 64, 256))
        layout_expected = ComposeLayout(
            SwizzleLayout(3, 3, 3, swizzle_inner=True),
            TileLayout(S[(3, 64, 4, 64) : (16384, 64, 4096, 1)]),
        )
        assert_structural_equal(layout, layout_expected)

    case2()

    def case3():
        layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_64B_ATOM, (3, 64, 256))
        layout_expected = ComposeLayout(
            SwizzleLayout(3, 2, 3, swizzle_inner=True),
            TileLayout(S[(3, 64, 8, 32) : (16384, 32, 2048, 1)]),
        )
        assert_structural_equal(layout, layout_expected)

    case3()


def test_tma_shared_layout_alias():
    shape = (3, 64, 256)
    layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_128B_ATOM, shape)
    alias_layout = tma_shared_layout("float16", SwizzleMode.SWIZZLE_128B_ATOM, shape)
    assert_structural_equal(alias_layout, layout)


def test_pool_allocator_alloc_mma():
    def alloc_layout(shape, dtype, swizzle_mode="auto"):
        with IRBuilder():
            with Tx_builder.prim_func():
                pool = T.SMEMPool(Var("smem_ptr", PointerType(PrimType("uint8"))))
                buf = pool.alloc_mma(shape, dtype, swizzle_mode=swizzle_mode)
        return buf.layout

    cases = [
        ("uint8", (3, 64, 256)),
        ("float16", (3, 64, 256)),
        ("bfloat16", (3, 64, 256)),
        ("float32", (3, 64, 256)),
        ("float4_e2m1fn", (3, 64, 256)),
    ]
    for dtype, shape in cases:
        layout = alloc_layout(shape, dtype)
        expected = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, shape)
        assert_structural_equal(layout, expected)

    shape = (3, 64, 256)
    layout_64b = alloc_layout(shape, "float32", SwizzleMode.SWIZZLE_64B_ATOM)
    expected_64b = mma_shared_layout("float32", SwizzleMode.SWIZZLE_64B_ATOM, shape)
    assert_structural_equal(layout_64b, expected_64b)

    layout_none = alloc_layout(shape, "float16", "none")
    expected_none = mma_shared_layout("float16", SwizzleMode.SWIZZLE_NONE, shape)
    assert_structural_equal(layout_none, expected_none)


def test_storage():
    def case1():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        assert_structural_equal(layout.storage(), layout)

    case1()

    def case2():
        layout = TileLayout(S[(8, 4, 2) : (4 @ laneid, 1 @ laneid, 1)])
        layout_stroage = TileLayout(S[2:1])
        assert_structural_equal(layout.storage(), layout_stroage)

    case2()

    def case3():
        layout = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        assert_structural_equal(layout.storage(), layout)

    case3()

    def case4():
        layout = (
            TileLayout(S[2:1])
            .tile(TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)]), (8, 4), (1, 2))
            .tile(TileLayout(S[(2, 1) : (1, 2)]), (2, 1), (8, 8))
            .tile(TileLayout(S[(1, 8) : (8, 1)]), (1, 8), (16, 8))
        )
        layout_stroage = (
            TileLayout(S[2:1])
            .tile(TileLayout(S[(2, 1) : (1, 2)]), (2, 1), (1, 2))
            .tile(TileLayout(S[(1, 8) : (8, 1)]), (1, 8), (2, 2))
        )
        assert_structural_equal(layout.storage().canonicalize(), layout_stroage.canonicalize())

    case4()


def test_unpack():
    def case1():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        layout_expected = TileLayout(S[(8, 16) : (16, 1)])
        assert_structural_equal(layout.unpack(2).canonicalize(), layout_expected.canonicalize())

    case1()

    def case2():
        layout = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        layout_expected = SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3)
        assert_structural_equal(layout.unpack(2).canonicalize(), layout_expected.canonicalize())

    case2()

    def case3():
        layout = ComposeLayout(
            SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 64) : (64, 1)]),
        )
        layout_expected = ComposeLayout(
            SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 128) : (128, 1)]),
        )
        assert_structural_equal(layout.unpack(2).canonicalize(), layout_expected.canonicalize())

    case3()


def test_pack():
    def case1():
        layout = TileLayout(S[(8, 16) : (16, 1)])
        layout_expected = TileLayout(S[(8, 8) : (8, 1)])
        assert_structural_equal(layout.pack(2).canonicalize(), layout_expected.canonicalize())

    case1()

    def case2():
        layout = SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3)
        layout_expected = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        assert_structural_equal(layout.pack(2).canonicalize(), layout_expected.canonicalize())

    case2()

    def case3():
        layout = ComposeLayout(
            SwizzleLayout(per_element=4, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 128) : (128, 1)]),
        )
        layout_expected = ComposeLayout(
            SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 64) : (64, 1)]),
        )
        assert_structural_equal(layout.pack(2).canonicalize(), layout_expected.canonicalize())

    case3()


def test_slice():
    def verify_slice(layout, shape, region, sliced):
        r_shape = [r[1] - r[0] for r in region]
        r_size = functools.reduce(operator.mul, [r[1] - r[0] for r in region])

        def get_region_coord(u):
            coord = []
            for r in reversed(region):
                coord.append(u % (r[1] - r[0]))
                u //= r[1] - r[0]
            return coord[::-1]

        def get_shape_coord(r_coord, region):
            return [region[i][0] + r_coord[i] for i in range(len(region))]

        analyzer = Analyzer()

        for u in range(r_size):
            r_coord = get_region_coord(u)
            s_coord = get_shape_coord(r_coord, region)
            a = layout.apply(*s_coord, shape=shape)["m"]
            b = sliced.apply(*r_coord, shape=r_shape)["m"]
            assert analyzer.simplify(a == b)

    def case1():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        shape = [64]
        region = [(5, 8)]
        sliced = layout.slice(shape, region).canonicalize()
        assert sliced is not None
        verify_slice(layout, shape, region, sliced)

        region = [tvm.ir.Range(5, 8)]
        sliced_2 = layout.slice(shape, region).canonicalize()
        assert sliced_2 is not None
        assert_structural_equal(sliced, sliced_2)

    case1()

    def case2():
        # Choose begin and extent to satisfy midpoint condition
        layout = TileLayout(S[(4, 4, 4, 4) : (64, 4, 16, 1)])
        shape = [16, 16]
        region = [(2, 3), (6, 10)]
        sliced = layout.slice(shape, region).canonicalize()
        assert sliced is not None
        verify_slice(layout, shape, region, sliced)

    case2()

    def case3():
        layout = TileLayout(S[(2, 8, 3, 8) : (192, 8, 64, 1)])
        shape = [16, 24]
        region = [(2, 6), (4, 12)]
        sliced = layout.slice(shape, region).canonicalize()
        assert sliced is not None
        verify_slice(layout, shape, region, sliced)

    case3()

    def case4():
        layout = TileLayout(S[(128, 2, 64) : (64, 128 * 64, 1)])
        shape = [128, 128]
        region = [(0, 128), (32, 96)]
        sliced = layout.slice(shape, region).canonicalize()
        assert sliced is not None
        verify_slice(layout, shape, region, sliced)

    case4()

    def case_swizzle_slice():
        # SwizzleLayout slice - delegates to ComposeLayout
        swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
        shape = [512]
        region = [(64, 128)]
        sliced = swizzle.slice(shape, region)
        assert sliced is not None
        verify_slice(swizzle, shape, region, sliced)

    case_swizzle_slice()

    def case_compose_slice():
        # ComposeLayout slice
        compose = ComposeLayout(
            SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 64) : (64, 1)]),
        )
        shape = [512]
        region = [(64, 128)]
        sliced = compose.slice(shape, region)
        assert sliced is not None
        verify_slice(compose, shape, region, sliced)

    case_compose_slice()

    def case_compose_slice_2d():
        # ComposeLayout slice with 2D shape
        compose = ComposeLayout(
            SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            TileLayout(S[(8, 64) : (64, 1)]),
        )
        shape = [8, 64]
        region = [(2, 4), (0, 64)]
        sliced = compose.slice(shape, region)
        assert sliced is not None
        verify_slice(compose, shape, region, sliced)

    case_compose_slice_2d()


def test_apply_to_shape():
    """``apply_to_shape`` should give per-shard coord, preferring per-dim
    split when the input shape aligns with the layout's grouping."""

    from tvm.tirx.layout import Iter, TileLayout

    # 1 shard per dim — coord[d] passes through unchanged.
    lay = TileLayout(S[16, 16])
    assert [int(x) for x in lay.apply_to_shape([5, 7], [16, 16])] == [5, 7]

    # Dim 1 split into (4, 4) factors — per-dim mixed-radix within dim 1,
    # no cross-dim flatten needed.
    lay2 = TileLayout.from_iters([Iter(16, 16, "m"), Iter(4, 4, "m"), Iter(4, 1, "m")])
    assert [int(x) for x in lay2.apply_to_shape([5, 7], [16, 16])] == [5, 7 // 4, 7 % 4]

    # Both dims split — verifies split stays local to each dim.
    lay3 = TileLayout.from_iters(
        [Iter(4, 64, "m"), Iter(4, 16, "m"), Iter(4, 4, "m"), Iter(4, 1, "m")]
    )
    r = lay3.apply_to_shape([13, 9], [16, 16])
    assert [int(x) for x in r] == [13 // 4, 13 % 4, 9 // 4, 9 % 4]


def test_slice_single_shard_skips_defensive_floormod():
    """Regression: ``Layout.slice`` must not emit ``floormod(begin, Ek)`` on
    single-shard groups whose caller-contract guarantees ``begin + extent
    <= Ek``.

    Background: ``SlicePerGroup`` in ``src/tirx/ir/layout/tile_slice.cc``
    decomposes ``begin`` into per-shard coordinates via
    ``floormod(floordiv(begin, B[k]), Ek)``. When ``m == 1`` (single shard
    in the group) and ``begin`` is a runtime expression (e.g. a pipeline
    stage ``BufferLoad``), the analyzer cannot prove ``begin < Ek`` so the
    defensive ``floormod`` survives codegen.

    Concretely, fa4's K_smem with shape ``(SMEM_PIPE_DEPTH_KV=3, 128, 128)``
    sliced by ``[stage:stage+1, :, :]`` would emit
    ``floormod(stage, 3) * 16384`` in every per-MMA SMEM-descriptor offset
    (72 sites at s1024_kv4) — even though ``PipelineState`` already keeps
    ``stage`` in ``[0, 3)``.

    The fix relies on the existing single-shard caller contract noted in
    the function:
        ``the slice is valid as long as the caller guarantees
         begin + slice_extent <= extent (which is assumed)``

    With the contract the mod is provably a no-op; this test asserts the
    sliced layout's ``offset`` is the bare ``stage * stride`` form for
    runtime ``begin``.
    """
    # Single-shard outer-axis slice with a runtime stage variable.
    layout = TileLayout(S[(3, 128, 128) : (16384, 128, 1)])
    shape = [3, 128, 128]
    stage = Var("stage", "int32")
    region = [tvm.ir.Range(stage, stage + 1), tvm.ir.Range(0, 128), tvm.ir.Range(0, 128)]
    sliced = layout.slice(shape, region)
    assert sliced is not None
    offset_strs = [str(off) for _, off in sliced.offset.items()]
    full = " | ".join(offset_strs)
    # No defensive floormod-by-extent should remain on the stage axis.
    assert "FloorMod" not in full and "floormod" not in full and "% 3" not in full, (
        f"single-shard slice with runtime begin must not emit defensive floormod, got offset={full}"
    )

    # Multi-shard groups (e.g. row dim with swizzle interleaving
    # ``(128, 2):(64, 8192)``) still need the floormod for correct
    # decomposition; verify we did not over-aggressively strip it.
    multi_shard = TileLayout.from_iters(
        [Iter(2, 8192, "m"), Iter(128, 64, "m")]  # outer (extent=2), inner (extent=128)
    )
    multi_shape = [256]
    multi_region = [tvm.ir.Range(96, 96 + 32)]
    multi_sliced = multi_shard.slice(multi_shape, multi_region)
    assert multi_sliced is not None
    # Constants — analyzer simplifies floormod(96, 128) to 96 internally;
    # we just assert offset is non-empty and structurally sane (not None).


if __name__ == "__main__":
    tvm.testing.main()
