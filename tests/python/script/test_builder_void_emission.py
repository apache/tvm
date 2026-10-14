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
"""Canonical void operator calls participate in native builder construction."""

import importlib.util
from types import ModuleType, SimpleNamespace

import pytest
import tvm_ffi

import tvm
from tvm import ir, tirx
from tvm.backend import cuda
from tvm.ir import builder_base
from tvm.ir.builder_base import AlreadyEmitted, builder_emit_if_void
from tvm.ir.op import _init_op_api, _make_op_api
from tvm.script import ir_builder as I
from tvm.script import tirx as T
from tvm.script.ir_builder import base


@pytest.fixture(scope="module", autouse=True)
def register_cuda():
    cuda.register_backend()


@pytest.mark.parametrize("ty", [ir.PrimType("void"), ir.TupleType([])])
def test_native_void_forms_and_frameless_identity(ty):
    call = ir.Call("tirx.cpu_parallel_barrier", [], ty=ty)
    assert builder_emit_if_void(call) is call
    with I.IRBuilder() as builder:
        receipt = builder_emit_if_void(call)
        assert isinstance(receipt, AlreadyEmitted)
        assert receipt.__tvm_ffi_object__() is receipt.value
        assert T.evaluate(receipt) is receipt
        T.emit_(receipt)
    assert builder.get().same_as(receipt.value)
    assert receipt.value.value.same_as(call)
    assert receipt.value.value.ty.same_as(ty)
    assert tvm_ffi.convert(receipt).same_as(receipt.value)
    assert tvm_ffi.convert([receipt])[0].same_as(receipt.value)
    assert base.AlreadyEmitted is AlreadyEmitted
    assert I.AlreadyEmitted is AlreadyEmitted


@pytest.mark.parametrize("ty", [ir.Type.missing(), ir.PrimType("int32"), ir.PtrType("void")])
def test_nonvoid_and_raw_constructors_do_not_emit(ty):
    with I.IRBuilder() as builder:
        call = ir.Call("tirx.cpu_parallel_barrier", [], ty=ty)
        assert builder_emit_if_void(call) is call
        raw_void = tirx.call_intrin("void", "tirx.cpu_parallel_barrier")
        assert isinstance(raw_void, ir.Call)
        assert builder.result is None
        receipt = T.evaluate(raw_void)
    assert builder.get().same_as(receipt.value)
    assert receipt.value.value.same_as(raw_void)


def test_fixed_void_exposures_and_explicit_type():
    generated = _make_op_api(ir.Op.get("tirx.cpu_parallel_barrier"), __name__)
    for api in (generated, tirx.cpu_parallel_barrier):
        assert isinstance(api(), ir.Call)
        with I.IRBuilder() as builder:
            assert isinstance(api(ty="int32"), ir.Call)
            assert isinstance(api(ty=ir.Type.missing()), ir.Call)
            assert builder.result is None
            emitted = api()
        assert builder.get().same_as(emitted.value)
    assert T.cpu_parallel_barrier is tirx.cpu_parallel_barrier


def test_existing_exposures_stay_owned_by_their_module(monkeypatch):
    import sys

    module = ModuleType("void_test_exposure")
    original = module.cpu_parallel_barrier = lambda: "handwritten"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ir.Op, "list_op_names", lambda: ["tirx.cpu_parallel_barrier"])
    _init_op_api("tirx", module.__name__)
    assert module.cpu_parallel_barrier is original


def test_statement_frame_and_invalid_frame():
    with I.IRBuilder():
        with T.function_(private=True) as frame:
            T.func_name_("emission")
            receipt = tirx.cpu_parallel_barrier()
            assert T.evaluate(receipt) is receipt
    assert len(frame.function.body.seq) == 1
    assert frame.function.body.seq[0].same_as(receipt.value)
    with I.IRBuilder():
        with I.ir_module():
            with pytest.raises(TypeError, match="Unsupported frame type"):
                tirx.cpu_parallel_barrier()


def test_nested_builder_scope():
    with I.IRBuilder() as outer:
        first = tirx.cpu_parallel_barrier()
        with I.IRBuilder() as inner:
            middle = tirx.cpu_parallel_barrier()
        last = tirx.cpu_parallel_barrier()
    assert inner.get().same_as(middle.value)
    assert len(outer.get().seq) == 2
    assert outer.get().seq[0].same_as(first.value)
    assert outer.get().seq[1].same_as(last.value)


def test_optional_hook_cached_without_importing_a_builder(monkeypatch):
    import builtins

    lookups = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("tvm.script")
        return original_import(name, *args, **kwargs)

    def missing(name, allow_missing=False):
        lookups.append((name, allow_missing))
        return None

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(tvm_ffi, "get_global_func", missing)
    spec = importlib.util.spec_from_file_location("isolated_builder_base", builder_base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    call = ir.Call("tirx.cpu_parallel_barrier", [])
    with I.IRBuilder() as builder:
        assert module.builder_emit_if_void(call) is call
        assert module.builder_emit_if_void(call) is call
        assert builder.result is None
    assert lookups == [("ir.builder_emit_if_void", True)]


def test_present_hook_cached(monkeypatch):
    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("per-call hook lookup")

    monkeypatch.setattr(tvm_ffi, "get_global_func", unexpected_lookup)
    with I.IRBuilder() as builder:
        first = tirx.cpu_parallel_barrier()
        second = tirx.cpu_parallel_barrier()
    assert len(builder.get().seq) == 2
    assert first.value.same_as(builder.get().seq[0])
    assert second.value.same_as(builder.get().seq[1])


def test_ptx_namespace_candidate_selection_and_printer():
    from tvm.backend.cuda.ptx.engine import _InstrChain
    from tvm.tirx.script.ir_builder import op as builder_op

    assert T.ptx is builder_op.ptx
    assert T.ptx is builder_op._get_script_namespace("ptx")
    instruction = T.ptx.barrier.sync.aligned
    raw = instruction(0)
    assert isinstance(raw, ir.Call)
    with I.IRBuilder() as builder:
        # An ambiguous selection must not leave speculative statements behind.
        ambiguous = _InstrChain(instruction._cands + instruction._cands)
        with pytest.raises(AssertionError, match="ambiguous ptx table"):
            ambiguous(0)
        with pytest.raises(ValueError):
            instruction()
        assert builder.result is None
        # Printing also reconstructs candidates and must remain side-effect free.
        raw.script()
        assert builder.result is None
        receipt = instruction(0)
        assert T.evaluate(receipt) is receipt
        T.emit_(receipt)
    assert builder.get().same_as(receipt.value)
    ir.assert_structural_equal(receipt.value.value, raw)


def test_parser_bare_and_explicit_evaluate_emit_once():
    @T.function
    def function():
        T.ptx.barrier.sync.aligned(0)
        T.evaluate(T.ptx.barrier.sync.aligned(0))
        T.cpu_parallel_barrier()
        T.evaluate(T.cpu_parallel_barrier())

    assert len(function.body.seq) == 4
    assert all(isinstance(stmt, ir.Evaluate) for stmt in function.body.seq)
    assert all(not isinstance(stmt.loc, ir.UnknownLoc) for stmt in function.body.seq)
    assert all(not isinstance(stmt.value.loc, ir.UnknownLoc) for stmt in function.body.seq)


def test_explicit_call_location_and_receipt_attachment():
    explicit = ir.SourceLoc(ir.SourceName("explicit.py"), 1, 1, 1, 8)
    attached = ir.SourceLoc(ir.SourceName("source.py"), 2, 1, 2, 8)
    with I.IRBuilder() as builder:
        receipt = tirx.cpu_parallel_barrier(loc=explicit)
        assert receipt.value.value.loc.same_as(explicit)
        assert base.at(attached, receipt) is receipt
    assert builder.get().same_as(receipt.value)
    assert receipt.value.loc.same_as(attached)
    assert receipt.value.value.loc.same_as(explicit)


def test_fixed_void_backend_helpers():
    from tvm.backend.cuda import op as cuda_op
    from tvm.backend.metal import op as metal_op
    from tvm.backend.trn import op as trn_op

    # Raw calls preserve provisional operands; these checks require no device.
    calls = [
        lambda: cuda_op.cuda_func_call("noop", "void noop() {}"),
        lambda: cuda_op.nvshmem_quiet(),
        lambda: metal_op.cooperative_tensor_fill(0, 0, 0, 16, 16),
        lambda: trn_op.nki_memset(0, 0),
    ]
    for make in calls:
        assert isinstance(make(), ir.Call)
        with I.IRBuilder() as builder:
            receipt = make()
        assert isinstance(receipt, AlreadyEmitted)
        assert builder.get().same_as(receipt.value)


def test_generated_region_api_stays_a_frame():
    generated = _make_op_api(ir.Op.get("tirx.parallel_launch"), __name__)
    with I.IRBuilder() as builder:
        frame = generated()
        assert isinstance(frame, base.IRBuilderFrame)
        assert builder.result is None
        with frame:
            receipt = tirx.cpu_parallel_barrier()
    assert isinstance(builder.get(), ir.RegionStmt)
    assert builder.get().body.seq[0].same_as(receipt.value)


def test_internal_ptx_codegen_does_not_emit():
    from tvm.backend.cuda.codegen.registry import CODEGEN_REGISTRY

    raw = T.ptx.barrier.sync.aligned(0)
    with I.IRBuilder() as builder:
        helper, _ = CODEGEN_REGISTRY[raw.op.name](raw.args)
        assert isinstance(helper, ir.Call)
        assert helper.op.name == "tirx.cuda.func_call"
        assert builder.result is None


def test_internal_tile_lowering_does_not_emit():
    from tvm.tirx.layout import S, TileLayout
    from tvm.tirx.tile.dispatch import run_dispatch

    shape = (2, 4)
    layout = TileLayout(S[shape])
    src = ir.Var("src", T.Tensor(shape, "uint32", layout=layout, align=16))
    dst = ir.Var("dst", T.Tensor(shape, "uint32", scope="local", layout=layout))
    call = T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], dst, src, scope="thread")
    context = SimpleNamespace(
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        scope_kind="thread",
        launch_params={},
        var_range_map={},
        intra={},
    )
    with I.IRBuilder() as builder:
        lowered = run_dispatch(call, context)
        assert lowered is not None
        assert builder.result is None


def test_stir_async_helpers_emit_once():
    from tvm.script import s_tir as S

    assert isinstance(S.async_commit(0), ir.Call)
    assert isinstance(S.async_wait(0, 1), ir.Call)
    with I.IRBuilder() as builder:
        commit = S.async_commit(0)
        wait = S.async_wait(0, 1)
        assert T.evaluate(commit) is commit
        assert T.evaluate(wait) is wait
    assert len(builder.get().seq) == 2
    assert builder.get().seq[0].same_as(commit.value)
    assert builder.get().seq[1].same_as(wait.value)
