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
"""Registered operator exposure and canonical CUDA script construction."""

import importlib
import os
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import tvm
from tvm import ir
from tvm.ir.op import _init_op_api
from tvm.script import tirx as T


def register(name, args=(), **kwargs):
    ir._ffi_api.RegisterOp(name, "Initializer regression")
    op = ir.Op.get(name)
    op.set_signature(args, **kwargs)
    op.set_attr("TCallEffectKind", 3)
    if name.startswith("tirx.cuda."):
        op.set_attr("TIRxOpCategory", "device_intrin")
        op.set_attr("TDeviceIntrinsicNamespace", "cuda")
    elif name.startswith("tirx."):
        op.set_attr("TIRxOpCategory", "builtin")
    return op


@pytest.fixture
def target(request, monkeypatch):
    name = "test_op_api_" + re.sub(r"\W", "_", request.node.name)
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_prefix_nested_reinitialization(target):
    first = register(target.__name__ + ".__first")
    register(target.__name__ + "_other.ignored")
    nested = register(target.__name__ + ".nested.leaf")
    target.nested = SimpleNamespace()
    assert _init_op_api(target.__name__) is None
    assert target.__first.__tvm_op__.same_as(first)
    assert target.nested.leaf.__tvm_op__.same_as(nested)
    assert not hasattr(target, "ignored")
    original = target.__first
    late = register(target.__name__ + ".late")
    _init_op_api(target.__name__)
    assert target.__first is original
    assert target.late.__tvm_op__.same_as(late)


def test_explicit_module_and_wrapper(target):
    op = register(target.__name__ + ".wrapped")

    def wrapper():
        return ir.Call(op, [])

    wrapper.__tvm_op__ = op
    target.wrapped = wrapper
    _init_op_api(target.__name__, target.__name__)
    assert target.wrapped is wrapper
    other = ModuleType(target.__name__ + "_target")
    sys.modules[other.__name__] = other
    try:
        _init_op_api(target.__name__, other.__name__)
        assert other.wrapped.__tvm_op__.same_as(op)
    finally:
        del sys.modules[other.__name__]


@pytest.mark.parametrize("occupant", [42, lambda: None])
def test_collision_preflight(target, occupant):
    register(target.__name__ + ".a_new")
    register(target.__name__ + ".z_conflict")
    target.z_conflict = occupant
    with pytest.raises(ValueError, match="conflicts"):
        _init_op_api(target.__name__)
    assert not hasattr(target, "a_new")


def test_wrong_identity(target):
    register(target.__name__ + ".leaf")
    target.leaf = lambda: None
    target.leaf.__tvm_op__ = ir.Op.get("tirx.cuda.clock64")
    with pytest.raises(ValueError, match="conflicts"):
        _init_op_api(target.__name__)


@pytest.mark.parametrize("suffix", ["nested.leaf", "bad..leaf", "class"])
def test_missing_or_malformed_container(target, suffix):
    register(target.__name__ + ".a_new")
    register(target.__name__ + "." + suffix)
    with pytest.raises(ValueError):
        _init_op_api(target.__name__)
    assert not hasattr(target, "a_new")


def test_aliased_container_collision(target):
    register(target.__name__ + ".a.leaf")
    register(target.__name__ + ".b.leaf")
    target.a = target.b = SimpleNamespace()
    with pytest.raises(ValueError, match="aliases"):
        _init_op_api(target.__name__)
    assert not vars(target.a)


def test_leaf_container_conflict(target):
    register(target.__name__ + ".leaf")
    register(target.__name__ + ".leaf.child")
    with pytest.raises(ValueError, match="existing namespace"):
        _init_op_api(target.__name__)
    assert not hasattr(target, "leaf")


def test_missing_target():
    with pytest.raises(KeyError):
        _init_op_api("test_op_api_unloaded")
    with pytest.raises(ValueError, match="Invalid"):
        _init_op_api("invalid..prefix")


def test_call_fields_and_customization(target):
    op = register(target.__name__ + ".call", ["x"], ty_args=["T"])
    _init_op_api(target.__name__)
    x = tvm.tirx.Var("x", "int16")
    ty = ir.PrimType("int16")
    span = ir.Span(ir.SourceName("api"), 1, 1, 1, 2)
    call = target.call(x, attrs={"flag": 1}, ty_args=[ty], span=span)
    assert call.ty.is_missing()
    assert call.args[0].same_as(x)
    assert call.span.same_as(span)
    assert call.attrs["flag"] == 1
    ir.assert_structural_equal(call.ty_args[0], ty)
    op.set_attr("FInferType", lambda c: c.args[0].ty)
    ir.assert_structural_equal(target.call(x, ty_args=[ty]).ty, ty)
    op.set_attr("FInferType", lambda c: ir.PrimType("int64"), override=True)
    assert target.call(x=x, ty_args=[ty]).ty.dtype == "int64"
    assert target.call(x, ty_args=[ty], ret_ty="float32").ty.dtype == "float32"
    op.set_attr("TFixedReturnType", ir.PrimType("uint32"))
    assert target.call(x, ty_args=[ty]).ty.dtype == "uint32"
    with pytest.raises(TypeError):
        target.call(x)  # Required type argument is still validated.
    with pytest.raises(TypeError, match="multiple values"):
        target.call(x, x=x)
    with pytest.raises(TypeError, match="unexpected"):
        target.call(x, unknown=x)


def test_inference_failure_propagates(target):
    op = register(target.__name__ + ".call")

    def fail(call):
        raise ValueError("inference sentinel")

    op.set_attr("FInferType", fail)
    _init_op_api(target.__name__)
    with pytest.raises(ValueError, match="inference sentinel"):
        target.call()


def test_cuda_inventory_and_import_identity():
    script = importlib.import_module("tvm.backend.cuda.script")

    assert T.cuda is script
    for name in ir.Op.list_op_names():
        if name.startswith("tirx.cuda."):
            current = script
            for part in name.removeprefix("tirx.cuda.").split("."):
                current = getattr(current, part)
            assert callable(current)
            assert current.__tvm_op__.same_as(ir.Op.get(name)), name
    first = script.clock64
    wrapper = script.func_call
    _init_op_api("tirx.cuda", script.__name__)
    assert script.clock64 is first
    assert script.func_call is wrapper
    assert script.clock64().ty.dtype == "uint64"
    assert callable(script.iket.mark)
    assert callable(script.wgmma.encode_matrix_descriptor)
    assert callable(script.tcgen05.encode_instr_descriptor)
    assert script.sm100_2sm_leader_smem_addr(0).op.name == "tirx.cuda.sm100_2sm_leader_smem_addr"


@pytest.mark.parametrize("first", ["tvm.backend.cuda.script", "tvm.script.tirx"])
def test_import_order(first):
    code = f"""
import importlib
importlib.import_module({first!r})
script = importlib.import_module("tvm.backend.cuda.script")
from tvm.script import tirx as T
assert T.cuda is script
assert T.cuda.clock64().ty.dtype == 'uint64'
"""
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        check=True,
    )


def roundtrip(call, params=()):
    func = tvm.tirx.PrimFunc(list(params), tvm.tirx.Evaluate(call))
    code = func.script()
    reparsed = tvm.script.from_source(code, extra_vars={"T": T, "I": tvm.script.ir})
    ir.assert_structural_equal(func, reparsed)
    return code


@pytest.mark.parametrize(
    "name,args",
    [
        ("clock64", ()),
        ("iket_mark", ("mark", 1, 2, 3)),
        ("iket_range_start", ("range", 1, 2)),
        ("iket_range_end", (1, 2, 3)),
        ("iket_range_push", ("range", 1, 2)),
        ("iket_range_pop", ()),
        ("iket_sentinel_token", ("token",)),
        ("iket_official_event", (1, "source", 2, 3)),
        ("wgmma_noop_barrier", (1,)),
        ("__shfl_sync", (1, tvm.tirx.const(2, "int32"), 0, 32)),
    ],
)
def test_canonical_roundtrip(name, args):
    call = getattr(T.cuda, name)(*args)
    code = roundtrip(call)
    assert "T.cuda." + name + "(" in code


def test_retained_wrappers_and_raw_fields():
    p = tvm.tirx.Var("p", "handle")
    for call in [
        T.cuda.atomic_add(p, 1),
        T.cuda.atomic_cas(p, 1, 2),
        T.cuda.ldg(p, "float32"),
        T.cuda.func_call(
            "f", 1, source_code="__device__ int f(int x) { return x; }", return_type="int32"
        ),
        T.cuda.tcgen05_encode_matrix_descriptor(p, p, 1, 2, 0),
        T.cuda.wgmma_encode_matrix_descriptor(p, p, 1, 2, 0),
    ]:
        roundtrip(call, [p])
    for kwargs in [{"attrs": {"flag": 1}}, {"ret_ty": "float32"}, {"ret_ty": ir.Type.missing()}]:
        assert "I.Call(" in roundtrip(T.cuda.clock64(**kwargs))


def test_wrapper_custom_inference_is_lossless():
    op = ir.Op.get("tirx.cuda.atomic_add")
    original = op.get_attr("FInferType")
    p = tvm.tirx.Var("p", "handle")
    try:
        op.set_attr("FInferType", lambda c: ir.PrimType("int64"), override=True)
        call = ir.Call(op, [p, tvm.tirx.const(1, "int32")], ret_ty="int64")
        assert "I.Call(" in roundtrip(call, [p])
    finally:
        op.set_attr("FInferType", original, override=True)


def test_generated_full_call_roundtrip(target):
    op = register(target.__name__ + ".call", ["x"], ty_args=["T"])
    _init_op_api(target.__name__)
    call = target.call(1, ty_args=[ir.PrimType("int16")], attrs={"flag": 1}, ret_ty="int32")
    assert "I.Call(" in roundtrip(call)
    assert op.get_attr("TScriptPrinterName") is None


def test_explicit_printer_alias_and_late_cuda_registration():
    module = importlib.import_module("tvm.backend.cuda.script")
    op = register("tirx.cuda.test_op_api_alias")
    op.set_attr("TFixedReturnType", ir.PrimType("int32"))
    _init_op_api("tirx.cuda", module.__name__)
    generated = module.test_op_api_alias
    assert "T.cuda.test_op_api_alias(" in roundtrip(generated())
    module.test_op_api_custom_name = generated
    op.set_attr("TScriptPrinterName", "tirx.cuda.test_op_api_custom_name")
    try:
        assert "T.cuda.test_op_api_custom_name(" in roundtrip(generated())
    finally:
        op.set_attr("TScriptPrinterName", op.name, override=True)
        del module.test_op_api_custom_name


def test_legacy_module_namespace_printer_discovery():
    from tvm.tirx.script.ir_builder.op import register_script_namespace

    op = register("tirx.test_op_api_legacy")
    module = ModuleType("test_op_api_legacy")
    module.wrapper = lambda: None
    module.wrapper.__tir_op_name__ = "test_op_api_legacy"
    register_script_namespace("test_op_api_legacy_namespace", module)
    assert op.get_attr("TScriptPrinterName") == "tirx.test_op_api_legacy_namespace.wrapper"


def test_wait_and_vector_load_fallbacks():
    p = tvm.tirx.Var("p", "handle")
    calls = [
        T.cuda.wait_until(p, "eq", 1),
        T.cuda.ldg(p, "float32", dst=[p, p], vec="v2"),
    ]
    for call in calls:
        assert "I.Call(" in roundtrip(call, [p])


@pytest.mark.parametrize(
    "name,args,ret_ty",
    [
        ("clock64", [1], "uint64"),
        ("atomic_add", [1, 2, 3], "int32"),
        ("__shfl_sync", [], "int32"),
    ],
)
def test_unchecked_calls_keep_lossless_fallback(name, args, ret_ty):
    call = ir.Call.unchecked("tirx.cuda." + name, args, ret_ty=ret_ty)
    assert "I.Call(" in roundtrip(call)


def test_retained_wrapper_tensor_region_fallback():
    buffer = tvm.tirx.decl_buffer((4,), "int32", name="A")
    call = ir.Call("tirx.cuda.atomic_add", [buffer[0:1], 1], ret_ty="int32")
    assert "I.Call(" in roundtrip(call, [buffer])


def test_shuffle_buffer_operand_fallback():
    buffer = tvm.tirx.decl_buffer((1,), "int32", name="A")
    call = ir.Call("tirx.cuda.__shfl_sync", [1, buffer, 0, 32], ret_ty=buffer.ty)
    assert "I.Call(" in roundtrip(call, [buffer])
