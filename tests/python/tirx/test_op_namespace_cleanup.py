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

import importlib
import sys
import types

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
        if isinstance(node, tvm.ir.Call):
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
    assert cast.ty.dtype == "float32"

    sqrt = T.sqrt(y)
    assert sqrt.op.name == "tirx.sqrt"

    fma = T.fma(y, y, y)
    assert fma.op.name == "tirx.fma"


def test_kernel_replace_point_is_builtin_marker_not_tile_primitive():
    assert _op_attr("tirx.tvm_kernel_replace_point", "TIRxOpCategory") == "builtin"
    assert "tirx.tile.tvm_kernel_replace_point" not in Op.list_op_names()
    assert hasattr(T, "tvm_kernel_replace_point")
    assert not hasattr(Tx, "tvm_kernel_replace_point")

    @T.prim_func(check_well_formed=False)
    def marker():
        T.tvm_kernel_replace_point()

    calls = _expr_calls(marker)
    assert [call.op.name for call in calls] == ["tirx.tvm_kernel_replace_point"]
    assert _tile_calls(marker) == []

    code = marker.script()
    assert "T.tvm_kernel_replace_point()" in code
    assert "tvm_kernel_replace_point" in code
    assert "T.tile.tvm_kernel_replace_point" not in code
    assert "Tx.tvm_kernel_replace_point" not in code
    reparsed = tvm.script.from_source(code)
    assert_structural_equal(marker, reparsed)


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
    from tvm.backend.cuda.script import (
        CUDANamespace as BackendCUDANamespace,
    )
    from tvm.backend.cuda.script import (
        NVSHMEMNamespace as BackendNVSHMEMNamespace,
    )
    from tvm.backend.cuda.script import (
        PTXNamespace as BackendPTXNamespace,
    )
    from tvm.backend.metal.script import MetalNamespace as BackendMetalNamespace
    from tvm.backend.trn.script import NKINamespace as BackendNKINamespace
    from tvm.tirx.script.builder import ir as builder_ir

    assert isinstance(builder_ir.cuda, BackendCUDANamespace)
    assert isinstance(builder_ir.ptx, BackendPTXNamespace)
    assert isinstance(builder_ir.nvshmem, BackendNVSHMEMNamespace)
    assert isinstance(builder_ir.metal, BackendMetalNamespace)
    assert isinstance(builder_ir.nki, BackendNKINamespace)
    assert T.cuda is builder_ir.cuda
    assert T.ptx is builder_ir.ptx
    assert T.nvshmem is builder_ir.nvshmem
    assert T.metal is builder_ir.metal
    assert T.nki is builder_ir.nki

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


def test_backend_specific_wrappers_are_not_root_exports():
    from tvm.tirx.cuda import op as cuda_op
    from tvm.tirx.metal import op as metal_op
    from tvm.tirx.trn import op as trn_op

    backend_only_names = [
        "ptx_mma",
        "mma_store",
        "cuda_thread_fence",
        "nvshmem_fence",
        "make_filled_simdgroup_matrix",
        "simdgroup_load",
        "nki_load",
    ]
    for name in backend_only_names:
        assert not hasattr(tvm.tirx.op, name)
        assert not hasattr(tvm.tirx, name)
        assert not hasattr(T, name)

    assert cuda_op.ptx_mma
    assert cuda_op.mma_store
    assert cuda_op.cuda_thread_fence
    assert cuda_op.nvshmem_fence
    assert metal_op.make_filled_simdgroup_matrix
    assert metal_op.simdgroup_load
    assert trn_op.nki_load
    assert hasattr(T, "cuda")
    assert hasattr(T, "ptx")
    assert hasattr(T, "nvshmem")
    assert hasattr(T, "metal")
    assert hasattr(T, "nki")


def test_backend_load_updates_tirx_alias_and_script_facades(monkeypatch):
    from tvm.tirx.script import builder, parser
    from tvm.tirx.script.builder import ir as builder_ir

    backend_name = "unit_test_backend"
    backend_module_name = f"tvm.backend.{backend_name}"
    public_module_name = f"tvm.tirx.{backend_name}"
    public_op_module_name = f"{public_module_name}.op"
    namespace_name = "unit_test_backend_ns"
    register_calls = []

    class UnitTestNamespace:
        pass

    module = types.ModuleType(backend_module_name)
    module.__path__ = []
    module.__package__ = backend_module_name
    op_module = types.ModuleType(f"{backend_module_name}.op")
    op_module.marker = object()

    def register_backend():
        register_calls.append(True)
        builder_ir.register_script_namespace(namespace_name, UnitTestNamespace())

    module.register_backend = register_backend
    monkeypatch.setitem(sys.modules, backend_module_name, module)
    monkeypatch.setitem(sys.modules, op_module.__name__, op_module)
    sys.modules.pop(public_module_name, None)
    sys.modules.pop(public_op_module_name, None)
    tvm.backend.loader._LOADED_BACKENDS.pop(backend_name, None)
    if hasattr(tvm.tirx, backend_name):
        delattr(tvm.tirx, backend_name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(public_op_module_name)

    try:
        assert tvm.backend.load(backend_name) is None
        assert tvm.backend.load(backend_name) is None
        assert register_calls == [True]
        assert tvm.backend.is_loaded(backend_name)
        assert getattr(tvm.tirx, backend_name) is module
        assert sys.modules[public_module_name] is module
        public_op_module = importlib.import_module(public_op_module_name)
        assert public_op_module.__tvm_backend_module__ is op_module
        assert public_op_module.marker is op_module.marker

        namespace = getattr(builder_ir, namespace_name)
        assert isinstance(namespace, UnitTestNamespace)
        assert getattr(builder, namespace_name) is namespace
        assert getattr(parser, namespace_name) is namespace
        assert getattr(T, namespace_name) is namespace
    finally:
        tvm.backend.loader._LOADED_BACKENDS.pop(backend_name, None)
        if hasattr(tvm.tirx, backend_name):
            delattr(tvm.tirx, backend_name)
        sys.modules.pop(public_module_name, None)
        sys.modules.pop(public_op_module_name, None)


def test_device_intrinsic_printer_roundtrips_canonical_namespaces():
    @T.prim_func
    def device_namespaces(dst: T.handle, src: T.handle):
        A = T.match_buffer(src, (1,), "float32")
        R = T.alloc_buffer((1,), "float32", scope="local")
        T.cuda.cta_sync()
        T.ptx.ldg32(R[0], 1, A[0], 0)
        T.metal.simd_shuffle(A[0], 0)
        T.metal.simd_shuffle_up(A[0], 1)
        T.metal.simd_shuffle_down(A[0], 1)

    calls = _expr_calls(device_namespaces)
    assert [call.op.name for call in calls] == [
        "tirx.cuda.cta_sync",
        "tirx.ptx.ldg32",
        "tirx.metal.simd_shuffle",
        "tirx.metal.simd_shuffle_up",
        "tirx.metal.simd_shuffle_down",
    ]
    for op_name, namespace in [
        ("tirx.cuda.cta_sync", "cuda"),
        ("tirx.ptx.ldg32", "ptx"),
        ("tirx.metal.simd_shuffle", "metal"),
        ("tirx.metal.simd_shuffle_up", "metal"),
        ("tirx.metal.simd_shuffle_down", "metal"),
    ]:
        assert _op_attr(op_name, "TIRxOpCategory") == "device_intrin"
        assert _op_attr(op_name, "TDeviceIntrinsicNamespace") == namespace
        assert _op_attr(op_name, "TCallEffectKind") in (1, 3)

    code = device_namespaces.script()
    assert "T.cuda.cta_sync(" in code
    assert "T.ptx.ldg32(" in code
    assert "T.metal.simd_shuffle(" in code
    assert "T.metal.simd_shuffle_up(" in code
    assert "T.metal.simd_shuffle_down(" in code
    assert "T.tirx." not in code
    reparsed = tvm.script.from_source(code)
    assert reparsed.script() == code
    assert_structural_equal(device_namespaces, reparsed)


def test_registered_tirx_ops_have_exactly_one_category():
    if _op_attr("tirx.sqrt", "TIRxOpCategory") is None:
        pytest.skip("TIRx op categories require a rebuilt C++ runtime")

    categories = {"builtin", "tile_primitive", "device_intrin"}
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
            assert device_namespace is None, op_name
        elif category == "device_intrin":
            assert device_namespace in device_namespaces, op_name
            printer_name = _op_attr(op_name, "TScriptPrinterName")
            assert printer_name is not None, op_name
            assert printer_name.startswith(device_namespace + "."), op_name
            assert _has_path(T, printer_name), op_name
        else:
            assert category == "builtin"
            assert device_namespace is None, op_name

    assert not missing
    assert not invalid
    assert not lingering_flat_tile
