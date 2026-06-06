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
"""Tests for T.hint() — universal directive primitive for TIRx sketch language."""

import tvm
import tvm.script
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.script import tirx as T
from tvm.tirx import AttrStmt


def from_source(code):
    return tvm.script.from_source(code)


def test_hint_statement():
    """T.hint("msg") as a bare statement produces an AttrStmt with attr_key=tirx_hint."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        _A = T.match_buffer(A_ptr, (64,), "float32", scope="global")
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        T.hint("persistent tile scheduler with L2 swizzle")
        T.evaluate(0)

    # Walk the IR to find the AttrStmt with tirx_hint
    found = [False]

    def visit(stmt):
        if isinstance(stmt, AttrStmt) and stmt.attr_key == "tirx_hint":
            # node is now a Map with "message" key
            assert isinstance(stmt.node, tvm.ir.Map)
            assert str(stmt.node["message"]) == "persistent tile scheduler with L2 swizzle"
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected AttrStmt with attr_key='tirx_hint' not found"


def test_hint_context_manager():
    """with T.hint("msg"): scopes its body inside the AttrStmt."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        _A = T.match_buffer(A_ptr, (64,), "float32", scope="global")
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        with T.hint("software pipeline, depth 4"):
            T.evaluate(0)

    found = [False]

    def visit(stmt):
        if isinstance(stmt, AttrStmt) and stmt.attr_key == "tirx_hint":
            assert isinstance(stmt.node, tvm.ir.Map)
            assert str(stmt.node["message"]) == "software pipeline, depth 4"
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected AttrStmt with attr_key='tirx_hint' not found"


def test_hint_with_attrs():
    """T.hint("msg", key="value") passes structured attrs in Map node."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        _A = T.match_buffer(A_ptr, (64,), "float32", scope="global")
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        T.hint("scheduler", mode="persistent", depth="4")
        T.evaluate(0)

    found = [False]

    def visit(stmt):
        if isinstance(stmt, AttrStmt) and stmt.attr_key == "tirx_hint":
            assert isinstance(stmt.node, tvm.ir.Map)
            assert str(stmt.node["message"]) == "scheduler"
            assert str(stmt.node["mode"]) == "persistent"
            assert str(stmt.node["depth"]) == "4"
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected AttrStmt with attr_key='tirx_hint' not found"


def test_hint_printer_roundtrip_statement():
    """Verify T.hint("msg") prints as T.hint("msg") and roundtrips through script/parse."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        _A = T.match_buffer(A_ptr, (64,), "float32", scope="global")
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        T.hint("persistent tile scheduler with L2 swizzle")
        T.evaluate(0)

    code = func.script()
    assert 'hint("persistent tile scheduler with L2 swizzle")' in code
    reparsed = from_source(code)
    assert_structural_equal(func, reparsed)


def test_hint_printer_roundtrip_context_manager():
    """Verify with T.hint("msg"): prints correctly and roundtrips."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        _A = T.match_buffer(A_ptr, (64,), "float32", scope="global")
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        with T.hint("software pipeline, depth 4"):
            T.evaluate(0)

    code = func.script()
    assert 'hint("software pipeline, depth 4")' in code
    reparsed = from_source(code)
    assert_structural_equal(func, reparsed)


def test_hint_printer_roundtrip_with_attrs():
    """Verify T.hint("msg", key="val") prints with kwargs and roundtrips."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        _A = T.match_buffer(A_ptr, (64,), "float32", scope="global")
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        T.hint("scheduler", mode="persistent")
        T.evaluate(0)

    code = func.script()
    assert 'hint("scheduler"' in code
    assert 'mode="persistent"' in code
    reparsed = from_source(code)
    assert_structural_equal(func, reparsed)


def test_hint_keyword_arg_on_tx_op():
    """Tx.op(..., hint="msg") stores hint in TilePrimitiveCall.config."""
    from tvm.tirx.buffer import decl_buffer
    from tvm.tirx.stmt import TilePrimitiveCall

    A = decl_buffer((64, 64), "float32", scope="global")
    A_sm = decl_buffer((64, 64), "float32", scope="shared")

    op_call = TilePrimitiveCall(
        A[0:64, 0:64],
        A_sm[0:64, 0:64],
        op=tvm.ir.Op.get("tirx.tile.copy"),
        workspace={},
        config={"hint": "3-input ptx"},
    )
    assert "hint" in op_call.config
    assert str(op_call.config["hint"]) == "3-input ptx"


def test_hint_keyword_arg_on_tx_op_roundtrip():
    """Tx.op(..., hint="msg") roundtrips through printer/parser."""
    from tvm.script.tirx import tile as Tx

    @T.prim_func
    def func(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [10], "float32", scope="global")
        B = T.match_buffer(B_ptr, [10], "float32", scope="global")
        Tx.add(B, A, T.float32(1), hint="use_fast_math")

    code = func.script()
    assert 'hint="use_fast_math"' in code
    reparsed = from_source(code)
    assert reparsed.script() == code
    assert_structural_equal(func, reparsed)


def test_hint_no_message():
    """T.hint(access=...) with no message string."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        T.hint(access=A[0:64])
        T.evaluate(0)

    found = [False]

    def visit(stmt):
        if isinstance(stmt, AttrStmt) and stmt.attr_key == "tirx_hint":
            assert isinstance(stmt.node, tvm.ir.Map)
            # Should have "access" key but no "message" key
            assert "access" in stmt.node
            assert "message" not in stmt.node
            from tvm.tirx import BufferRegion

            assert isinstance(stmt.node["access"], BufferRegion)
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected AttrStmt with attr_key='tirx_hint' containing access not found"


def test_hint_access_buffer_region():
    """T.hint(access=A[region]) stores the BufferRegion structurally in the IR."""

    @T.prim_func
    def func(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, 64), "float32", scope="global")
        bx, by, bz = T.cta_id([2, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        T.hint("partition", access=A[bx * 64 : (bx + 1) * 64, 0:64])
        T.evaluate(0)

    found = [False]

    def visit(stmt):
        if isinstance(stmt, AttrStmt) and stmt.attr_key == "tirx_hint":
            assert isinstance(stmt.node, tvm.ir.Map)
            assert str(stmt.node["message"]) == "partition"
            assert "access" in stmt.node
            from tvm.tirx import BufferRegion

            assert isinstance(stmt.node["access"], BufferRegion)
            br = stmt.node["access"]
            assert br.buffer.name == "A"
            assert len(br.region) == 2
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert found[0], "Expected AttrStmt with structured BufferRegion access not found"


if __name__ == "__main__":
    test_hint_statement()
    test_hint_context_manager()
    test_hint_with_attrs()
    test_hint_printer_roundtrip_statement()
    test_hint_printer_roundtrip_context_manager()
    test_hint_printer_roundtrip_with_attrs()
    test_hint_keyword_arg_on_tx_op()
    test_hint_keyword_arg_on_tx_op_roundtrip()
    test_hint_no_message()
    test_hint_access_buffer_region()
