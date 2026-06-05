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
"""Tests for TIRx op namespace split between T, T.tile, and device namespaces."""

import pytest

import tvm
from tvm.ir import Op, assert_structural_equal
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.stmt import TilePrimitiveCall


def _tile_calls(func):
    calls = []

    def visit(stmt):
        if isinstance(stmt, TilePrimitiveCall):
            calls.append(stmt)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _expr_calls(func):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def _op_attr(op_name, attr_name):
    return Op.get(op_name).get_attr(attr_name)


def _has_path(root, path):
    cur = root
    for part in path.split("."):
        if not hasattr(cur, part):
            return False
        cur = getattr(cur, part)
    return True


def test_tx_is_tile_shorthand_only():
    assert T.tile is Tx
    assert T.tile.copy is Tx.copy
    assert not hasattr(T, "copy")
    assert not hasattr(Tx, "SMEMPool")
    assert not hasattr(Tx, "ScopedOp")
    assert not hasattr(Tx, "meta_class")
    assert T.cast is not Tx.cast
    assert T.sqrt is not Tx.sqrt


def test_tx_rejects_expression_overloads():
    x = tvm.tirx.Var("x", "float32")
    y = tvm.tirx.Var("y", "int32")

    with pytest.raises(TypeError, match="tile-only"):
        Tx.sqrt(x)
    with pytest.raises(TypeError, match="tile-only"):
        T.tile.sqrt(x)
    with pytest.raises(TypeError, match="tile-only"):
        Tx.cast(y, "float32")
    with pytest.raises(TypeError, match="tile-only"):
        T.tile.cast(y, "float32")


def test_builtin_expression_ops_are_not_tile_primitives():
    x = tvm.tirx.Var("x", "int32")
    y = tvm.tirx.Var("y", "float32")

    cast = T.cast(x, "float32")
    assert isinstance(cast, tvm.tirx.Cast)
    assert cast.dtype == "float32"

    sqrt = T.sqrt(y)
    assert sqrt.op.name == "tirx.sqrt"

    fma = T.fma(y, y, y)
    assert fma.op.name == "tirx.fma"


def test_tile_shorthand_and_scoped_aliases_use_tile_ops():
    @T.prim_func(check_well_formed=False)
    def tile_aliases(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (16,), "float32")
        B = T.match_buffer(b, (16,), "float32")
        T.tile.copy(A[0:16], B[0:16])
        Tx.cast(A[0:16], B[0:16])
        T.cta.cast(A[0:16], B[0:16])
        Tx.cta.sqrt(A[0:16], B[0:16])

    calls = _tile_calls(tile_aliases)
    assert [call.op.name for call in calls] == [
        "tirx.tile.copy",
        "tirx.tile.cast",
        "tirx.tile.cast",
        "tirx.tile.sqrt",
    ]
    assert [call.scope.name for call in calls] == ["thread", "thread", "cta", "cta"]


def test_device_intrinsic_namespaces_are_canonical_and_classified():
    buffer = tvm.tirx.decl_buffer((1,), "float32")
    calls = [
        T.ptx.elect_sync(),
        T.cuda.thread_fence(),
        T.nvshmem.fence(),
        T.nki.identity(buffer[0:1], 1),
    ]

    expected = [
        ("tirx.ptx.elect_sync", "ptx"),
        ("tirx.cuda.thread_fence", "cuda"),
        ("tirx.nvshmem.fence", "nvshmem"),
        ("tirx.nki.identity", "nki"),
    ]
    assert [
        (call.op.name, _op_attr(call.op.name, "TDeviceIntrinsicNamespace")) for call in calls
    ] == expected
    for op_name, namespace in expected:
        assert _op_attr(op_name, "TIRxOpCategory") == "device_intrin"
        assert _op_attr(op_name, "TDeviceIntrinsicNamespace") == namespace
        assert _op_attr(op_name, "TTilePrimitiveKind") is None


def test_device_intrinsic_printer_roundtrips_canonical_namespaces():
    @T.prim_func
    def device_namespaces(dst: T.handle, src: T.handle):
        A = T.match_buffer(src, (1,), "float32")
        R = T.alloc_buffer((1,), "float32", scope="local")
        T.cuda.copy_bytes(dst, src, 16)
        T.ptx.ldg32(R[0], 1, A[0], 0)
        T.metal.simd_shuffle(A[0], 0)

    calls = _expr_calls(device_namespaces)
    assert [call.op.name for call in calls] == [
        "tirx.cuda.copy_bytes",
        "tirx.ptx.ldg32",
        "tirx.metal.simd_shuffle",
    ]
    for op_name, namespace in [
        ("tirx.cuda.copy_bytes", "cuda"),
        ("tirx.ptx.ldg32", "ptx"),
        ("tirx.metal.simd_shuffle", "metal"),
    ]:
        assert _op_attr(op_name, "TIRxOpCategory") == "device_intrin"
        assert _op_attr(op_name, "TDeviceIntrinsicNamespace") == namespace
        assert _op_attr(op_name, "TCallEffectKind") in (1, 3)

    code = device_namespaces.script()
    assert "T.cuda.copy_bytes(" in code
    assert "T.ptx.ldg32(" in code
    assert "T.metal.simd_shuffle(" in code
    assert "T.tirx." not in code
    reparsed = tvm.script.from_source(code)
    assert reparsed.script() == code
    assert_structural_equal(device_namespaces, reparsed)


def test_registered_tirx_ops_have_exactly_one_category():
    if _op_attr("tirx.sqrt", "TIRxOpCategory") is None:
        pytest.skip("TIRx op categories require a rebuilt C++ runtime")

    categories = {"builtin", "tile_primitive", "device_intrin"}
    tile_kinds = {"dispatch", "compose", "async", "marker"}
    device_namespaces = {"cuda", "ptx", "nvshmem", "nki", "metal", "webgpu"}
    flat_tile_only_names = {
        "tirx.add",
        "tirx.binary_chain",
        "tirx.binary_reduce",
        "tirx.compose_op",
        "tirx.copy",
        "tirx.copy_async",
        "tirx.fdiv",
        "tirx.fill",
        "tirx.gemm",
        "tirx.gemm_async",
        "tirx.maximum",
        "tirx.memset",
        "tirx.minimum",
        "tirx.mul",
        "tirx.permute_layout",
        "tirx.reduce_negate",
        "tirx.select",
        "tirx.sub",
        "tirx.sum",
        "tirx.unary_reduce",
        "tirx.zero",
    }

    missing = []
    invalid = []
    lingering_flat_tile = []
    for op_name in sorted(name for name in Op.list_op_names() if name.startswith("tirx.")):
        category = _op_attr(op_name, "TIRxOpCategory")
        tile_kind = _op_attr(op_name, "TTilePrimitiveKind")
        device_namespace = _op_attr(op_name, "TDeviceIntrinsicNamespace")

        if category is None:
            missing.append(op_name)
            continue
        if category not in categories:
            invalid.append((op_name, category))
            continue
        if op_name in flat_tile_only_names:
            lingering_flat_tile.append(op_name)

        if category == "tile_primitive":
            if not op_name.startswith("tirx.tile."):
                lingering_flat_tile.append(op_name)
            assert tile_kind in tile_kinds, op_name
            assert device_namespace is None, op_name
        elif category == "device_intrin":
            assert tile_kind is None, op_name
            assert device_namespace in device_namespaces, op_name
            printer_name = _op_attr(op_name, "TScriptPrinterName")
            assert printer_name is not None, op_name
            assert printer_name.startswith(device_namespace + "."), op_name
            assert _has_path(T, printer_name), op_name
        else:
            assert category == "builtin"
            assert tile_kind is None, op_name
            assert device_namespace is None, op_name

    assert not missing
    assert not invalid
    assert not lingering_flat_tile
