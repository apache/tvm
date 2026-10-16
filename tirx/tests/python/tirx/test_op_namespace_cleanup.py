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
import tvm_ffi

import tvm_tirx
from tvm_tirx.ir import Op, assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _tile_calls(func):
    calls = []

    def visit(stmt):
        if (
            isinstance(stmt, tvm_tirx.ir.Call)
            and isinstance(stmt.op, Op)
            and stmt.op.get_attr("FTileOpDispatch") is not None
        ):
            calls.append(stmt)

    tvm_ffi.structural_walk(func.body, visit)
    return calls


def _expr_calls(func):
    calls = []

    def visit(node):
        if isinstance(node, tvm_tirx.ir.Call):
            calls.append(node)

    tvm_ffi.structural_walk(func.body, visit)
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


def test_builtin_expression_ops_are_not_tile_ops():
    x = tvm_tirx.tirx.Var("x", "int32")
    y = tvm_tirx.tirx.Var("y", "float32")

    cast = T.cast(x, ty="float32")
    assert isinstance(cast, tvm_tirx.tirx.Cast)
    assert cast.ty.dtype == "float32"

    sqrt = T.sqrt(y)
    assert sqrt.op.name == "prim.sqrt"

    fma = T.fma(y, y, y)
    assert fma.op.name == "prim.fma"


def test_tensormap_encoding_accepts_attribute_keywords_and_defaults():
    @T.function
    def encode(descriptor: T.Ptr(T.TensorMap), data: T.handle):
        T.cuda.tensormap_encode_tiled(
            descriptor,
            data,
            64,
            32,
            128,
            64,
            32,
            1,
            1,
            descriptor_dtype="bfloat16",
            rank=2,
            swizzle=3,
        )

    (call,) = [c for c in _expr_calls(encode) if c.op.name == "tirx.cuda.tensormap_encode_tiled"]
    assert call.op.attrs_type_key == "tirx.cuda.TensorMapEncodeTiledAttr"
    assert str(call.attrs.descriptor_dtype) == "bfloat16"
    assert call.attrs.rank == 2
    assert call.attrs.swizzle == 3
    assert call.attrs.interleave == call.attrs.l2_promotion == call.attrs.oob_fill == 0
    assert call.attrs.force_cu_dtype == -1
    call.validate()
    reparsed = tvm_tirx.script.from_source(encode.script(), extra_vars={"I": I, "T": T})
    assert_structural_equal(encode, reparsed)


def test_kernel_replace_point_is_builtin_marker_not_tile_op():
    assert _op_attr("tirx.kernel_replace_point", "FTileOpDispatch") is None
    assert "tirx.tile.kernel_replace_point" not in Op.list_op_names()
    assert hasattr(T, "kernel_replace_point")
    assert not hasattr(T.cuda.tile, "kernel_replace_point")

    @T.function(check_well_formed=False)
    def marker():
        T.kernel_replace_point()

    calls = _expr_calls(marker)
    assert [call.op.name for call in calls] == ["tirx.kernel_replace_point"]
    assert _tile_calls(marker) == []

    code = marker.script()
    assert "T.kernel_replace_point()" in code
    assert "kernel_replace_point" in code
    assert "T.tile.kernel_replace_point" not in code
    assert "Tx.kernel_replace_point" not in code
    reparsed = tvm_tirx.script.from_source(
        code, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
    )
    assert_structural_equal(marker, reparsed)


def test_device_intrinsic_namespaces_are_canonical_and_classified():
    cuda_script = importlib.import_module("tvm_tirx.backend.cuda.script")
    from tvm_tirx.backend.cuda.script import (
        NVSHMEMNamespace as BackendNVSHMEMNamespace,
    )
    from tvm_tirx.backend.cuda.script import (
        STIRNamespace as BackendSTIRNamespace,
    )
    from tvm_tirx.backend.metal.script import MetalNamespace as BackendMetalNamespace
    from tvm_tirx.backend.trn.script import NKINamespace as BackendNKINamespace
    from tvm_tirx.tirx.script.ir_builder import op as builder_op

    assert builder_op.cuda is cuda_script
    assert isinstance(builder_op.s_tir, BackendSTIRNamespace)
    assert isinstance(builder_op.nvshmem, BackendNVSHMEMNamespace)
    assert isinstance(builder_op.metal, BackendMetalNamespace)
    assert isinstance(builder_op.nki, BackendNKINamespace)
    assert T.cuda is builder_op.cuda
    assert T.s_tir is builder_op.s_tir
    assert T.nvshmem is builder_op.nvshmem
    assert T.metal is builder_op.metal
    assert T.nki is builder_op.nki

    tensor = tvm_tirx.tirx.decl_tensor((1,), "float32")
    calls = [
        T.cuda.elect_sync(),
        T.cuda.thread_fence(),
        T.nvshmem.fence(),
        T.nki.identity(tensor[0], 1),
    ]

    expected = [
        ("tirx.cuda.elect_sync", "cuda"),
        ("tirx.cuda.thread_fence", "cuda"),
        ("tirx.nvshmem.fence", "nvshmem"),
        ("tirx.nki.identity", "nki"),
    ]
    assert [(call.op.name, call.op.name.split(".")[1]) for call in calls] == expected
    for op_name, _ in expected:
        assert _op_attr(op_name, "FTileOpDispatch") is None


def test_backend_specific_wrappers_are_not_root_exports():
    from tvm_tirx.tirx.cuda import op as cuda_op
    from tvm_tirx.tirx.metal import op as metal_op
    from tvm_tirx.tirx.trn import op as trn_op

    backend_only_names = [
        "ptx_legacy_mma",
        "mma_store",
        "thread_fence",
        "nvshmem_fence",
        "make_filled_simdgroup_matrix",
        "simdgroup_load",
        "nki_load",
    ]
    for name in backend_only_names:
        assert not hasattr(tvm_tirx.tirx.op, name)
        assert not hasattr(tvm_tirx.tirx, name)
        assert not hasattr(T, name)

    assert cuda_op.ptx_legacy_mma
    assert cuda_op.mma_store
    assert cuda_op.thread_fence
    assert cuda_op.nvshmem_fence
    assert metal_op.make_filled_simdgroup_matrix
    assert metal_op.simdgroup_load
    assert trn_op.nki_load
    assert hasattr(T, "cuda")
    assert hasattr(T, "ptx_legacy")
    assert hasattr(T, "nvshmem")
    assert hasattr(T, "metal")
    assert hasattr(T, "nki")


def test_backend_load_updates_tirx_alias_and_script_facades(monkeypatch):
    from tvm_tirx.script.parser import tirx as parser
    from tvm_tirx.tirx.script import ir_builder as builder
    from tvm_tirx.tirx.script.ir_builder import op as builder_op

    backend_name = "unit_test_backend"
    backend_module_name = f"tvm_tirx.backend.{backend_name}"
    public_module_name = f"tvm_tirx.tirx.{backend_name}"
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
        builder_op.register_script_namespace(namespace_name, UnitTestNamespace())

    module.register_backend = register_backend
    monkeypatch.setitem(sys.modules, backend_module_name, module)
    monkeypatch.setitem(sys.modules, op_module.__name__, op_module)
    sys.modules.pop(public_module_name, None)
    sys.modules.pop(public_op_module_name, None)
    tvm_tirx.backend.loader._LOADED_BACKENDS.pop(backend_name, None)
    if hasattr(tvm_tirx.tirx, backend_name):
        delattr(tvm_tirx.tirx, backend_name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(public_op_module_name)

    try:
        assert tvm_tirx.backend.load(backend_name) is None
        assert tvm_tirx.backend.load(backend_name) is None
        assert register_calls == [True]
        assert tvm_tirx.backend.is_loaded(backend_name)
        assert getattr(tvm_tirx.tirx, backend_name) is module
        assert sys.modules[public_module_name] is module
        public_op_module = importlib.import_module(public_op_module_name)
        assert public_op_module.__tvm_backend_module__ is op_module
        assert public_op_module.marker is op_module.marker

        namespace = getattr(builder_op, namespace_name)
        assert isinstance(namespace, UnitTestNamespace)
        assert getattr(builder, namespace_name) is namespace
        assert getattr(parser, namespace_name) is namespace
        assert getattr(T, namespace_name) is namespace
    finally:
        tvm_tirx.backend.loader._LOADED_BACKENDS.pop(backend_name, None)
        if hasattr(tvm_tirx.tirx, backend_name):
            delattr(tvm_tirx.tirx, backend_name)
        sys.modules.pop(public_module_name, None)
        sys.modules.pop(public_op_module_name, None)


def test_device_intrinsic_printer_roundtrips_canonical_namespaces():
    @T.function
    def device_namespaces(dst: T.handle, A: T.Tensor((1,), "float32")):
        T.cuda.cta_sync()
        T.metal.simd_shuffle(A[0], 0)
        T.metal.simd_shuffle_up(A[0], 1)
        T.metal.simd_shuffle_down(A[0], 1)

    calls = _expr_calls(device_namespaces)
    assert [call.op.name for call in calls] == [
        "tirx.cuda.cta_sync",
        "tirx.metal.simd_shuffle",
        "tirx.metal.simd_shuffle_up",
        "tirx.metal.simd_shuffle_down",
    ]
    for op_name, namespace in [
        ("tirx.cuda.cta_sync", "cuda"),
        ("tirx.metal.simd_shuffle", "metal"),
        ("tirx.metal.simd_shuffle_up", "metal"),
        ("tirx.metal.simd_shuffle_down", "metal"),
    ]:
        assert _op_attr(op_name, "FTileOpDispatch") is None
        assert op_name.split(".")[1] == namespace
        assert _op_attr(op_name, "TCallEffectKind") in (
            tvm_tirx.ir.CallEffectKind.Pure,
            tvm_tirx.ir.CallEffectKind.UpdateState,
        )

    code = device_namespaces.script()
    assert "T.cuda.cta_sync(" in code
    assert "T.metal.simd_shuffle(" in code
    assert "T.metal.simd_shuffle_up(" in code
    assert "T.metal.simd_shuffle_down(" in code
    assert "T.tirx." not in code
    reparsed = tvm_tirx.script.from_source(
        code, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
    )
    assert reparsed.script() == code
    assert_structural_equal(device_namespaces, reparsed)


def test_registered_tirx_ops_use_dispatch_attributes():
    from tvm_tirx.backend.cuda.ptx.table import TABLE, escape_token

    assert not Op.get("prim.sqrt").has_attr("TIRxOpCategory")
    device_namespaces = {
        "cuda",
        "ptx",
        # `ptx_legacy` holds the Apache-compatible spellings, `s_tir` the
        # nodes that pipeline's own passes build; neither is a dialect surface.
        "ptx_legacy",
        "s_tir",
        "nvshmem",
        "nki",
        "metal",
        "webgpu",
    }
    flat_tile_only_names = {
        "tirx.add",
        "tirx.binary_chain",
        "tirx.binary_reduce",
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

    lingering_flat_tile = []
    for op_name in sorted(name for name in Op.list_op_names() if name.startswith("tirx.")):
        dispatcher = _op_attr(op_name, "FTileOpDispatch")

        if op_name in flat_tile_only_names:
            lingering_flat_tile.append(op_name)

        if dispatcher is not None:
            assert callable(dispatcher), op_name
            if not op_name.startswith(("tirx.cuda.tile.", "tirx.trn.tile.")):
                lingering_flat_tile.append(op_name)
        else:
            assert not op_name.startswith(("tirx.cuda.tile.", "tirx.trn.tile.")), op_name
            if op_name.split(".")[1] in device_namespaces:
                surface_name = op_name.removeprefix("tirx.")
                if op_name.startswith("tirx.ptx."):
                    entry = TABLE.get(op_name.removeprefix("tirx.ptx."))
                    if entry is not None:
                        surface_name = "ptx." + escape_token(entry.family)
                assert _has_path(T, surface_name), op_name

    assert not lingering_flat_tile


def test_tile_helpers_are_separate_from_backend_instructions():
    assert T.tile.DispatchContext
    assert T.tile.Instruction
    assert not hasattr(T.tile, "copy")
    assert not hasattr(tvm_tirx.tirx, "TilePrimitiveCall")
    assert not hasattr(tvm_tirx.tirx, "TileOpCall")
    assert not any(name.startswith("tirx.tile.") for name in Op.list_op_names())
    for name in (
        "copy",
        "copy_async",
        "sum",
        "min",
        "permute_layout",
        "ld",
        "st",
        "compose",
        "tcgen05",
    ):
        assert not hasattr(T.cuda.tile, name)
    assert T.cuda.tile.apply
    assert T.cuda.tile.ewise
