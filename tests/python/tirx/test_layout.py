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
import itertools

import pytest

import tvm
from tvm.backend.cuda.axis import Axis as CUDAAxis
from tvm.backend.cuda.axis import TCol, TLane, laneid, tid_in_wg, tx, warpid, wgid, wid_in_wg
from tvm.backend.cuda.lang import SMEMPool
from tvm.backend.cuda.layout import (
    SwizzleMode,
    mma_shared_layout,
    tmem_datapath_layout,
    tmem_mma_operand_layout,
)
from tvm.backend.trn.axis import Axis as TRNAxis
from tvm.backend.trn.axis import Bank, F, P
from tvm.ir import assert_structural_equal
from tvm.ir.type import PrimType, PtrType
from tvm.script.ir_builder import IRBuilder
from tvm.tirx import Var
from tvm.tirx.layout import Axis, ComposeLayout, Iter, R, S, TileLayout, m
from tvm.tirx.script import ir_builder as Tx_builder


def test_axis():
    assert CUDAAxis.bx == Axis.get("bx")
    assert CUDAAxis.by == Axis.get("by")
    assert CUDAAxis.bz == Axis.get("bz")
    assert CUDAAxis.cbx == Axis.get("cbx")
    assert CUDAAxis.cby == Axis.get("cby")
    assert CUDAAxis.cbz == Axis.get("cbz")
    assert CUDAAxis.tx == Axis.get("tx")
    assert CUDAAxis.warpid == Axis.get("warpid")
    assert CUDAAxis.laneid == Axis.get("laneid")
    assert CUDAAxis.wgid == Axis.get("wgid")
    assert CUDAAxis.tid_in_wg == Axis.get("tid_in_wg")
    assert CUDAAxis.wid_in_wg == Axis.get("wid_in_wg")
    assert Axis.m == Axis.get("m")
    assert TRNAxis.P == Axis.get("P")
    assert TRNAxis.F == Axis.get("F")
    assert CUDAAxis.TCol == Axis.get("TCol")
    assert CUDAAxis.TLane == Axis.get("TLane")

    assert CUDAAxis.bx.is_thread()
    assert CUDAAxis.by.is_thread()
    assert CUDAAxis.bz.is_thread()
    assert CUDAAxis.cbx.is_thread()
    assert CUDAAxis.cby.is_thread()
    assert CUDAAxis.cbz.is_thread()
    assert CUDAAxis.tx.is_thread()
    assert CUDAAxis.warpid.is_thread()
    assert CUDAAxis.laneid.is_thread()
    assert CUDAAxis.wgid.is_thread()
    assert CUDAAxis.tid_in_wg.is_thread()
    assert CUDAAxis.wid_in_wg.is_thread()
    assert Axis.m.is_memory()
    assert TRNAxis.P.is_memory()
    assert TRNAxis.F.is_memory()
    assert CUDAAxis.TCol.is_memory()
    assert CUDAAxis.TLane.is_memory()

    assert CUDAAxis.bx.get_scope().name == "cuda.kernel"
    assert CUDAAxis.bx.get_subscope().name == "cuda.cta"


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
        assert res[0].name == "cuda.thread"
        assert res[1].name == "cuda.warp"
        assert layout.verify_well_formed()

        layout = TileLayout(S[8 : 4 @ laneid] + R[4 : 1 @ laneid])
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "cuda.thread"
        assert res[1].name == "cuda.warp"
        assert layout.verify_well_formed()

        layout = TileLayout(S[(8, 4, 2) : (4 @ laneid, 1 @ laneid, 1)])
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "cuda.thread"
        assert res[1].name == "cuda.warp"
        assert layout.verify_well_formed()

        layout = TileLayout(
            S[(2, 8, 2, 4, 2) : (2 @ warpid, 4 @ laneid, 1 @ warpid, 1 @ laneid, 1)]
        )
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "cuda.thread"
        assert res[1].name == "cuda.cta"
        assert layout.verify_well_formed()

        layout = TileLayout(
            S[(2, 8, 2, 4, 2) : (2 @ wid_in_wg, 4 @ laneid, 1 @ wid_in_wg, 1 @ laneid, 1)]
        )
        res = layout.get_scope()
        assert res is not None
        assert res[0].name == "cuda.thread"
        assert res[1].name == "cuda.warpgroup"
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
        inner = TileLayout(S[8:1])
        outer = inner
        layout_tile = TileLayout(S[64:1])
        assert_structural_equal(layout_tile, inner.tile(outer, [8], [8]))

    case1()

    def case2():
        inner = TileLayout(S[(8, 8) : (8, 1)])
        outer = inner
        layout_tile = TileLayout(S[(8, 8, 8, 8) : (512, 8, 64, 1)])
        assert_structural_equal(layout_tile, inner.tile(outer, [8, 8], [8, 8]))

    case2()

    def case3():
        inner = TileLayout(S[(2, 4) : (1, 2)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = TileLayout(S[(8, 2, 32) : (64, 1, 2)])
        assert_structural_equal(layout_tile, inner.tile(outer, [8, 8], [2, 4]))

    case3()

    def case4():
        inner = TileLayout(S[(4, 2, 2, 4) : (16, 8, 1, 2)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = TileLayout(S[(8, 4, 2, 8, 2, 4) : (512, 16, 8, 64, 1, 2)])
        assert_structural_equal(layout_tile.canonicalize(), inner.tile(outer, (8, 8), (8, 8)))

    case4()

    def case5_sharded1():
        layout = TileLayout(S[(8, 1, 4, 2) : (4 @ laneid, 2, 1 @ laneid, 1)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = layout.tile(outer=outer, outer_shape=(8, 8), inner_shape=(8, 8))
        layout_expected = TileLayout(S[(8, 8, 1, 8, 4, 2) : (16, 4 @ laneid, 2, 2, 1 @ laneid, 1)])
        assert_structural_equal(layout_expected.canonicalize(), layout_tile)

    case5_sharded1()

    def case6_sharded2():
        inner = TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)])
        outer = TileLayout(S[(8, 8) : (8, 1)])
        layout_tile = inner.tile(outer=outer, outer_shape=(8, 8), inner_shape=(8, 4))
        layout_expected = TileLayout(S[(8, 8, 8, 4) : (8, 4 @ laneid, 1, 1 @ laneid)])
        assert_structural_equal(layout_expected, layout_tile)

    case6_sharded2()

    def case_dims_mismatch():
        with pytest.raises(Exception):
            layout = TileLayout(S[8:1])
            layout2 = TileLayout(S[(2, 4) : (1, 2)])
            layout2.tile(layout, [8], [2, 4])

    case_dims_mismatch()

    def case_tile_compose_layout():
        compose = ComposeLayout(3, 3, 3, TileLayout(S[(8, 64) : (64, 1)]))
        layout = TileLayout(S[(8, 1) : (1, 1)])
        layout_tile = compose.tile(layout, (8, 1), (8, 64))
        layout_expected = ComposeLayout(3, 3, 3, TileLayout(S[4096:1]))
        assert_structural_equal(layout_tile.canonicalize(), layout_expected.canonicalize())

    case_tile_compose_layout()

    def case_tile_swizzle_layout():
        swizzle = ComposeLayout(3, 3, 3, TileLayout(S[512,]))
        layout = TileLayout(S[(8, 4) : (1, 8)])
        layout_tile = swizzle.tile(layout, (8, 4), (8, 64))
        layout_expected = ComposeLayout(3, 3, 3, TileLayout(S[(64, 4, 64) : (64, 4096, 1)]))
        assert_structural_equal(layout_tile.canonicalize(), layout_expected)

    case_tile_swizzle_layout()

    def case_tile_swizzle_layout2():
        swizzle = ComposeLayout(3, 3, 3, TileLayout(S[512,]))
        tile = TileLayout(S[(3, 8, 4) : (8 * 4, 1, 8)])
        layout_tile = swizzle.tile(tile, (3, 8, 4), (1, 8, 64))
        layout_expected = ComposeLayout(
            swizzle.per_element,
            swizzle.swizzle_len,
            swizzle.atom_len,
            TileLayout(S[(3, 64, 4, 64) : (16384, 64, 4096, 1)]),
            swizzle.swizzle_inner,
        )
        assert_structural_equal(layout_tile.canonicalize(), layout_expected.canonicalize())

    case_tile_swizzle_layout2()

    def case_tile_swizzle_layout3():
        swizzle = ComposeLayout(3, 2, 3, TileLayout(S[256,]))
        tile = TileLayout(S[(8, 8) : (1, 8)])
        layout_tile = swizzle.tile(tile, (8, 8), (8, 32))
        layout_expected = ComposeLayout(
            swizzle.per_element,
            swizzle.swizzle_len,
            swizzle.atom_len,
            TileLayout(S[(64, 8, 32) : (32, 2048, 1)]),
            swizzle.swizzle_inner,
        )
        assert_structural_equal(layout_tile.canonicalize(), layout_expected.canonicalize())

    case_tile_swizzle_layout3()


def test_shard_layout():
    """Sharding is tiling with thread axes on the outer tile."""

    def case_mma_layout():
        layout = TileLayout(S[(1, 2) : (2, 1)])
        layout_warp = TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)])
        res = layout.tile(layout_warp, [8, 4], [1, 2])
        layout_expected = TileLayout(S[(32, 2) : (1 @ laneid, 1)])
        assert_structural_equal(res.canonicalize(), layout_expected.canonicalize())

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

    case_cta_layout()

    def case_quad_shuffle():
        layout = TileLayout(S[(1, 2) : (2, 1)])
        layout_warp = TileLayout(S[8 : 4 @ laneid])
        res = layout.tile(layout_warp, [8, 1], [1, 2])
        layout_expected = TileLayout(S[(8, 2) : (4 @ laneid, 1)])
        assert_structural_equal(res.canonicalize(), layout_expected.canonicalize())

    case_quad_shuffle()

    def case_replicate():
        layout = TileLayout(S[(64, 128) : (128, 1)])
        layout_rep = TileLayout(S[2 : 2 @ warpid] + R[2 : 1 @ warpid])
        res = layout.tile(layout_rep, [2, 1], [64, 128])
        layout_expected = TileLayout(S[(2, 8192) : (2 @ warpid, 1)] + R[2 : 1 @ warpid])
        assert_structural_equal(res.canonicalize(), layout_expected.canonicalize())

    case_replicate()


def test_size_span():
    def tile_layout_size():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        assert layout.size() == 64

    tile_layout_size()

    def swizzle_layout_size():
        layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        assert layout.size() == 512
        layout = ComposeLayout(4, 3, 3, TileLayout(S[(1024,)]))
        assert layout.size() == 1024

    swizzle_layout_size()

    def compose_layout_size():
        layout = ComposeLayout(3, 3, 3, TileLayout(S[(8, 64) : (64, 1)]))
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
        layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        assert layout.span() == 512
        layout = ComposeLayout(4, 3, 3, TileLayout(S[(1024,)]))
        assert layout.span() == 1024

    swizzle_layout_span()

    def compose_layout_span():
        layout = ComposeLayout(3, 3, 3, TileLayout(S[(8, 64) : (64, 1)]))
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

        layout = TileLayout(
            S[
                (
                    128,
                    128,
                ) : (
                    1 @ P,
                    1 @ F,
                )
            ]
        )
        assert layout.size("P") == 128 and layout.size("F") == 128

        layout = TileLayout(
            S[
                (
                    32,
                    4,
                    128,
                    512,
                ) : (
                    512 @ F,
                    16384 @ F,
                    1 @ P,
                    1 @ F,
                )
            ]
        )
        assert_structural_equal(
            layout, TileLayout(S[(32, 4, 128, 512) : (512 @ F, (512 * 32) @ F, 1 @ P, 1 @ F)])
        )

        layout = TileLayout(
            S[
                (
                    2,
                    4,
                    32,
                    512,
                ) : (
                    512 @ F,
                    32 @ P,
                    1 @ P,
                    1 @ F,
                )
            ]
        )
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
        v = tvm.tirx.Var("v", ty="int32")
        res = layout.apply(v)
        assert res["m"] == v

    test_tile_layout_4()

    ################ Swizzle Layout
    def test_swizzle_layout_0():
        layout = ComposeLayout(0, 3, 3, TileLayout(S[(64,)]))
        # assert layout.size == 64
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j)["m"] == i * 8 + i ^ j

    test_swizzle_layout_0()

    def test_swizzle_layout_1():
        layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
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
        layout = ComposeLayout(0, 3, 3, TileLayout(S[(64,)]), swizzle_inner=False)
        assert layout.size() == 64
        for i, j in itertools.product(range(8), range(8)):
            assert layout.apply(i * 8 + j)["m"] == (i ^ j) * 8 + j

    test_swizzle_layout_2()

    def test_swizzle_layout_3():
        layout = ComposeLayout(0, 2, 3, TileLayout(S[(32,)]))
        for i, j in itertools.product(range(8), range(8)):
            _outer_i, inner_i = i // 4, i % 4
            outer_j, inner_j = j // 4, j % 4
            assert layout.apply(i * 8 + j)["m"] == i * 8 + outer_j * 4 + (inner_i ^ inner_j)

    test_swizzle_layout_3()

    ################ Compose Layout
    def test_compose_layout_0():
        layoutA = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        layoutB = TileLayout(S[(8, 64) : (64, 1)])
        layout = ComposeLayout(
            layoutA.per_element,
            layoutA.swizzle_len,
            layoutA.atom_len,
            layoutB,
            layoutA.swizzle_inner,
        )
        assert layout.size() == 512
        assert layout.span() == 512
        for i, j in itertools.product(range(8), range(64)):
            assert (
                layout.apply(i * 64 + j)["m"] == layoutA.apply(layoutB.apply(i * 64 + j)["m"])["m"]
            )

    test_compose_layout_0()

    def test_compose_layout_1():
        layoutA = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        layoutB = TileLayout(S[(16, 64, 8) : (64, 1, 1024)])
        layout = ComposeLayout(
            layoutA.per_element,
            layoutA.swizzle_len,
            layoutA.atom_len,
            layoutB,
            layoutA.swizzle_inner,
        )
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
        layout = TileLayout(
            S[
                (
                    2,
                    512,
                    8,
                ) : (
                    1 @ Bank,
                    1 @ F,
                    1 @ P,
                )
            ]
        )
        for i, j in itertools.product(range(1024), range(8)):
            coord = layout.apply(i, j, shape=(1024, 8))
            assert coord["Bank"] == i // 512
            assert coord["P"] == j
            assert coord["F"] == i % 512

    test_trainium_psum_layout_0()


def test_normalize_compose_layout():
    def case1():
        layoutA = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        layoutB = TileLayout(S[(8, 64) : (64, 1)])
        layout = ComposeLayout(
            layoutA.per_element,
            layoutA.swizzle_len,
            layoutA.atom_len,
            layoutB.canonicalize(),
            layoutA.swizzle_inner,
        )
        assert_structural_equal(layout.canonicalize(), layoutA)

    case1()

    def case2():
        layoutA = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        layoutB = TileLayout(S[(64, 4, 64) : (64, 4096, 1)])
        layout = ComposeLayout(
            layoutA.per_element,
            layoutA.swizzle_len,
            layoutA.atom_len,
            layoutB.canonicalize(),
            layoutA.swizzle_inner,
        )
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
        layout_expected = ComposeLayout(3, 3, 3, TileLayout(S[(64, 4, 64) : (64, 4096, 1)]))
        assert_structural_equal(layout, layout_expected)

    case1()

    def case2():
        layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_128B_ATOM, (3, 64, 256))
        layout_expected = ComposeLayout(
            3, 3, 3, TileLayout(S[(3, 64, 4, 64) : (16384, 64, 4096, 1)])
        )
        assert_structural_equal(layout, layout_expected)

    case2()

    def case3():
        layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_64B_ATOM, (3, 64, 256))
        layout_expected = ComposeLayout(
            3, 2, 3, TileLayout(S[(3, 64, 8, 32) : (16384, 32, 2048, 1)])
        )
        assert_structural_equal(layout, layout_expected)

    case3()


def test_pool_allocator_alloc_mma():
    def alloc_layout(shape, dtype, swizzle_mode="auto"):
        with IRBuilder():
            with Tx_builder.function():
                pool = SMEMPool(Var("smem_ptr", PtrType(PrimType("uint8"))))
                buf = pool.alloc_tcgen05_mma_AB(shape, dtype, swizzle_mode=swizzle_mode)
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


def test_tmem_mma_operand_layout_grouped_d():
    head64_o = tmem_mma_operand_layout(
        "D", (64, 512), "float32", M=64, cta_group=1, ws=True, group=(2, 2, 128)
    )
    expected_head64_o = TileLayout(
        S[(64, 2, 2, 128) : (1 @ TLane, 128 @ TCol, 64 @ TLane, 1 @ TCol)]
    ).canonicalize()
    assert_structural_equal(head64_o, expected_head64_o)

    head128_o = tmem_mma_operand_layout(
        "D", (64, 512), "float32", M=128, cta_group=2, group=(2, 2, 128)
    )
    assert_structural_equal(head128_o, expected_head64_o)

    wide_ws = tmem_mma_operand_layout("D", (128, 256), "float32", M=128, cta_group=1, ws=True)
    assert_structural_equal(wide_ws, tmem_datapath_layout("D", 128, 256))

    with pytest.raises(ValueError, match="ws=True requires cta_group=1"):
        tmem_mma_operand_layout("D", (64, 256), "float32", M=128, cta_group=2, ws=True)

    with pytest.raises(ValueError, match="group must be"):
        tmem_mma_operand_layout(
            "D", (64, 512), "float32", M=64, cta_group=1, ws=True, group=(4, 128)
        )

    with pytest.raises(ValueError, match="identity needs 2D"):
        tmem_mma_operand_layout("A", (1, 128, 64), "bfloat16", M=128, cta_group=1)

    # M=64 non-.ws A occupies lanes 0..63 identically (Layout F datapath is the
    # accumulator's scatter, not the A operand's).
    a_f = tmem_mma_operand_layout("A", (64, 64), "float32", M=64, cta_group=1)
    assert_structural_equal(a_f, TileLayout(S[(64, 64) : (1 @ TLane, 1 @ TCol)]).canonicalize())


def test_unpack():
    def case1():
        layout = TileLayout(S[(8, 8) : (8, 1)])
        layout_expected = TileLayout(S[(8, 16) : (16, 1)])
        assert_structural_equal(layout.unpack(2).canonicalize(), layout_expected.canonicalize())

    case1()

    def case2():
        layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        layout_expected = ComposeLayout(4, 3, 3, TileLayout(S[(1024,)]))
        assert_structural_equal(layout.unpack(2).canonicalize(), layout_expected.canonicalize())

    case2()

    def case3():
        layout = ComposeLayout(3, 3, 3, TileLayout(S[(8, 64) : (64, 1)]))
        layout_expected = ComposeLayout(4, 3, 3, TileLayout(S[(8, 128) : (128, 1)]))
        assert_structural_equal(layout.unpack(2).canonicalize(), layout_expected.canonicalize())

    case3()


def test_pack():
    def case1():
        layout = TileLayout(S[(8, 16) : (16, 1)])
        layout_expected = TileLayout(S[(8, 8) : (8, 1)])
        assert_structural_equal(layout.pack(2).canonicalize(), layout_expected.canonicalize())

    case1()

    def case2():
        layout = ComposeLayout(4, 3, 3, TileLayout(S[(1024,)]))
        layout_expected = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        assert_structural_equal(layout.pack(2).canonicalize(), layout_expected.canonicalize())

    case2()

    def case3():
        layout = ComposeLayout(4, 3, 3, TileLayout(S[(8, 128) : (128, 1)]))
        layout_expected = ComposeLayout(3, 3, 3, TileLayout(S[(8, 64) : (64, 1)]))
        assert_structural_equal(layout.pack(2).canonicalize(), layout_expected.canonicalize())

    case3()


@pytest.mark.parametrize("storage_axis, owner_axis", [(m, laneid), (TCol, TLane)])
def test_pack_preserves_ownership_and_storage_offsets(storage_axis, owner_axis):
    layout = TileLayout.from_iters(
        [Iter(8, 1, owner_axis), Iter(4, 2, storage_axis), Iter(2, 1, storage_axis)],
        [Iter(2, 16, storage_axis)],
        {owner_axis: 8, storage_axis: 4},
    )
    packed = layout.pack(2, storage_axis.name)
    for owner, word in itertools.product(range(8), range(4)):
        before = layout.apply(owner, word, 0)
        after = packed.apply(owner, word, 0)
        assert int(after[owner_axis.name]) == int(before[owner_axis.name])
        assert int(after[storage_axis.name]) * 2 == int(before[storage_axis.name])
    assert int(packed.replica[0].stride) == 8
    assert_structural_equal(
        packed.unpack(2, storage_axis.name).canonicalize(), layout.canonicalize()
    )


def _compose_swizzle(value, per_element, swizzle_len, atom_len, swizzle_inner=True):
    chunk = 1 << per_element
    index = value // chunk
    mask = (1 << swizzle_len) - 1
    if swizzle_inner:
        index ^= (index & (mask << atom_len)) >> atom_len
    else:
        index ^= (index & mask) << atom_len
    return index * chunk + value % chunk


def _evaluate_layout_expr(expr, values):
    node_type = type(expr).__name__
    if node_type == "IntImm":
        return int(expr.value)
    if node_type == "Var":
        return values[expr]
    if node_type == "Let":
        bindings = dict(values)
        bindings[expr.var] = _evaluate_layout_expr(expr.value, bindings)
        return _evaluate_layout_expr(expr.body, bindings)
    if node_type in ("Add", "Sub", "Mul", "FloorDiv", "FloorMod"):
        lhs = _evaluate_layout_expr(expr.a, values)
        rhs = _evaluate_layout_expr(expr.b, values)
        if node_type == "Add":
            return lhs + rhs
        if node_type == "Sub":
            return lhs - rhs
        if node_type == "Mul":
            return lhs * rhs
        if node_type == "FloorDiv":
            return lhs // rhs
        return lhs % rhs
    if node_type == "Cast":
        return _evaluate_layout_expr(expr.value, values)
    if node_type in ("BitwiseXor", "BitwiseAnd", "LShift", "RShift"):
        lhs = _evaluate_layout_expr(expr.a, values)
        rhs = _evaluate_layout_expr(expr.b, values)
        if node_type == "BitwiseXor":
            return lhs ^ rhs
        if node_type == "BitwiseAnd":
            return lhs & rhs
        if node_type == "LShift":
            return lhs << rhs
        return lhs >> rhs
    raise AssertionError(f"Cannot evaluate node type {node_type}")


def _naive_compose_expr(tile, coords, per_element=3, swizzle_len=3, atom_len=3, inner=True):
    period = 1 << (per_element + swizzle_len + atom_len)
    swizzle = ComposeLayout(
        per_element,
        swizzle_len,
        atom_len,
        TileLayout(S[(period,)]),
        inner,
    )
    return swizzle.apply(tile.apply(*coords)["m"])["m"]


@pytest.mark.parametrize("swizzle_len", [3, 2, 1, 0])
def test_compose_structured_apply_uses_bounded_normal_form(swizzle_len):
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    tile = TileLayout(S[(8, 8) : (64, 1)] + 72)
    layout = ComposeLayout(3, swizzle_len, 3, tile)

    actual = layout.apply(x, y)["m"]
    shaped = layout.apply(x, y, shape=(8, 8))["m"]
    if swizzle_len == 0:
        expected = x * 64 + y + 72
    else:
        mask = (1 << swizzle_len) - 1
        base = x * 64 + (y ^ (((x + 1) & mask) << 3)) + 64
        expected = base ^ 8
    assert_structural_equal(actual, expected)
    assert_structural_equal(shaped, expected)

    for xv, yv in itertools.product(range(8), repeat=2):
        logical = xv * 64 + yv + 72
        assert _evaluate_layout_expr(actual, {x: xv, y: yv}) == _compose_swizzle(
            logical, 3, swizzle_len, 3
        )


def test_compose_rank_one_shape_aware_apply_is_structured():
    flat = Var("flat", ty="int32")
    tile = TileLayout(S[(8, 8) : (64, 1)] + 72)
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(flat, shape=(64,))["m"]
    x = flat // 8
    y = flat % 8
    expected = (x * 64 + (y ^ (((x + 1) & 7) << 3)) + 64) ^ 8
    assert_structural_equal(actual, expected)
    assert type(actual).__name__ != "Let"
    for value in range(64):
        logical = value // 8 * 64 + value % 8 + 72
        assert _evaluate_layout_expr(actual, {flat: value}) == _compose_swizzle(logical, 3, 3, 3)


def test_compose_scalar_apply_keeps_full_swizzle_semantics():
    coord = Var("coord", ty="int32")
    tile = TileLayout(S[(8, 8) : (64, 1)] + 72)
    layout = ComposeLayout(3, 3, 3, tile)
    actual = layout.apply(coord)["m"]

    assert "//" in str(actual)
    assert "%" in str(actual)
    for value in range(256):
        logical = value // 8 * 64 + value % 8 + 72
        assert _evaluate_layout_expr(actual, {coord: value}) == _compose_swizzle(logical, 3, 3, 3)


def test_compose_shape_aware_apply_decomposes_each_logical_coordinate():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    tile = TileLayout(S[(2, 4, 8, 8) : (256, 64, 8, 1)])
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(x, y, shape=(8, 64))["m"]
    expected = x * 64 + (y ^ ((x & 7) << 3))
    assert_structural_equal(actual, expected)
    for xv, yv in itertools.product(range(8), range(64)):
        assert _evaluate_layout_expr(actual, {x: xv, y: yv}) == _compose_swizzle(
            xv * 64 + yv, 3, 3, 3
        )


def test_compose_structured_apply_splits_one_cross_atom_term():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    tile = TileLayout(S[(128, 8) : (8, 1)])
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(x, y)["m"]
    high = x // 8 * 64
    low = x % 8 * 8 + y
    expected = high + (low ^ (((x // 8) & 7) << 3))
    assert_structural_equal(actual, expected)
    assert type(actual).__name__ != "Let"
    for xv, yv in itertools.product(range(128), range(8)):
        logical = xv * 8 + yv
        assert _evaluate_layout_expr(actual, {x: xv, y: yv}) == _compose_swizzle(logical, 3, 3, 3)


def test_compose_structured_apply_handles_uint32_casts_per_term():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    x_u32 = tvm.tirx.Cast(x, "uint32")
    y_u32 = tvm.tirx.Cast(y, "uint32")
    layout = ComposeLayout(3, 3, 3, TileLayout(S[(8, 8) : (64, 1)] + 72))

    actual = layout.apply(x_u32, y_u32, shape=(8, 8))["m"]
    high = x_u32 * 64 + 64
    expected = (high + (y_u32 ^ (((x_u32 + 1) & 7) << 3))) ^ 8
    assert_structural_equal(actual, expected)
    assert "// T.uint32(64)" not in str(actual)

    for xv, yv in itertools.product(range(8), repeat=2):
        logical = xv * 64 + yv + 72
        assert _evaluate_layout_expr(actual, {x: xv, y: yv}) == _compose_swizzle(logical, 3, 3, 3)


def test_compose_structured_apply_classifies_dynamic_offsets():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    high = Var("high", ty="int32")
    low = Var("low", ty="int32")
    tile = TileLayout(S[(8, 8) : (64, 1)] + high * 64 + low % 8 + 16)
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(x, y)["m"]
    expected = (x * 64 + high * 64 + ((y + low % 8) ^ (((x + high) & 7) << 3))) ^ 16
    assert_structural_equal(actual, expected)
    for xv, yv, hv, lv in itertools.product(range(2), range(8), range(2), range(8)):
        logical = xv * 64 + yv + hv * 64 + lv % 8 + 16
        assert _evaluate_layout_expr(actual, {x: xv, y: yv, high: hv, low: lv}) == (
            _compose_swizzle(logical, 3, 3, 3)
        )


def test_compose_extent_one_coordinate_is_zero_before_classification():
    unused = Var("unused", ty="int32")
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    tile = TileLayout(S[(1, 8, 8) : (-1, 64, 1)] + 8)
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(unused, x, y)["m"]
    expected = (x * 64 + (y ^ ((x & 7) << 3))) ^ 8
    assert_structural_equal(actual, expected)
    assert "unused" not in str(actual)


@pytest.mark.parametrize(
    "case",
    ["outer_swizzle", "negative_stride", "unbounded_low", "carry", "symbolic_stride"],
)
def test_compose_structured_apply_falls_back_to_full_swizzle(case):
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    value = Var("value", ty="int32")
    inner = True
    if case == "outer_swizzle":
        tile = TileLayout(S[(8, 8) : (64, 1)] + 8)
        inner = False
    elif case == "negative_stride":
        tile = TileLayout(S[(8, 8) : (-64, 1)] + 8)
    elif case == "unbounded_low":
        tile = TileLayout(S[(8, 8) : (64, 1)] + value)
    elif case == "carry":
        tile = TileLayout(S[(8, 8) : (64, 1)] + 4)
    else:
        tile = TileLayout(S[(8, 8) : (value, 1)] + 8)

    layout = ComposeLayout(3, 3, 3, tile, inner)
    actual = layout.apply(x, y)["m"]
    expected = _naive_compose_expr(tile, (x, y), inner=inner)

    assert type(actual).__name__ == "Let"
    assert_structural_equal(actual.value, tile.apply(x, y)["m"])
    quotient = actual.body
    assert type(quotient).__name__ == "Let"
    assert_structural_equal(quotient.value, actual.var // 8)
    body_vars = tvm.tirx.analysis.undefined_vars(quotient.body)
    assert all(var.same_as(actual.var) or var.same_as(quotient.var) for var in body_vars)

    for xv, yv, vv in itertools.product(range(3), range(8), [-3, 0, 5, 64]):
        values = {x: xv, y: yv, value: vv}
        assert _evaluate_layout_expr(actual, values) == _evaluate_layout_expr(expected, values)


def test_compose_structured_apply_emits_shared_base_address_chain():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    constants = [0, 8, 16, 24, 512, 520, 528, 536]
    addresses = [
        ComposeLayout(3, 3, 3, TileLayout(S[(8, 8) : (64, 1)] + constant)).apply(x, y)["m"]
        for constant in constants
    ]

    base = addresses[0]
    high_base = base + 512
    expected = [
        base,
        base ^ 8,
        base ^ 16,
        base ^ 24,
        high_base,
        high_base ^ 8,
        high_base ^ 16,
        high_base ^ 24,
    ]
    for actual, expected_expr in zip(addresses, expected):
        assert_structural_equal(actual, expected_expr)

    for xv, yv in itertools.product(range(8), repeat=2):
        for constant, address in zip(constants, addresses):
            logical = xv * 64 + yv + constant
            assert _evaluate_layout_expr(address, {x: xv, y: yv}) == _compose_swizzle(
                logical, 3, 3, 3
            )

    # Constants — analyzer simplifies floormod(96, 128) to 96 internally;
    # we just assert offset is non-empty and structurally sane (not None).


if __name__ == "__main__":
    tvm.testing.main()
