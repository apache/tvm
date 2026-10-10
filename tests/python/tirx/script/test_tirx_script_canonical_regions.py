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
"""Canonical region construction and recursive expression printing."""

import ast

import pytest
import tvm_ffi

import tvm
from tvm import ir, tirx
from tvm.ir.op import _make_op_api, register_op_attr
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder


def _roundtrip_expr(value):
    # Standalone printers declare free variables before the final expression.
    source = value.script()
    tree = ast.parse(source)
    env = {"I": I, "T": T, "R": R}
    tvm_ffi.structural_walk(value, (ir.Var, lambda var: env.update({var.name: var})))
    exec(compile(ast.Module(tree.body[:-1], []), "<script>", "exec"), env)
    restored = eval(compile(ast.Expression(tree.body[-1].value), "<script>", "eval"), env)
    ir.assert_structural_equal(value, restored, map_free_vars=True)
    return source, restored


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("extents", [(1, 1), (1, 4), (3, 1)])
def test_region_expression_kind_and_nested_call_tuple(dtype, extents):
    a = tirx.decl_tensor((16, 16), "float32", name="A")
    i, j = tirx.Var("i", dtype), tirx.Var("j", dtype)
    region = tirx.BufferRegion(
        a, [ir.Range.from_min_extent(x, tirx.IntImm(dtype, n)) for x, n in zip((i, j), extents)]
    )
    source, restored = _roundtrip_expr(region)
    assert isinstance(restored, ir.TensorRegion)
    assert "I.TensorRegion" not in source
    assert ":" in source.splitlines()[-1]
    call = ir.Call("prim.exp", [(region, (region,))], ty="void")
    source, restored = _roundtrip_expr(call)
    assert "I.Tuple(" not in source
    assert isinstance(restored.args[0], ir.Tuple)
    assert isinstance(restored.args[0].fields[1].fields[0], ir.TensorRegion)


@pytest.mark.parametrize(
    "kind", ["unusual_type", "unusual_source", "zero_rank", "unsimplified_extent"]
)
def test_region_explicit_fallback(kind):
    a = tirx.decl_tensor(() if kind == "zero_rank" else (16,), "float32", name="A")
    source = ir.Var("source", "handle") if kind == "unusual_source" else a
    ty = ir.Type.missing() if kind == "unusual_type" else ir.TensorRegionType()
    extent = tirx.Add(tirx.IntImm("int32", 1), tirx.IntImm("int32", 1))
    ranges = (
        []
        if kind == "zero_rank"
        else [ir.Range.from_min_extent(0, extent if kind == "unsimplified_extent" else 1)]
    )
    region = ir.TensorRegion(source, ranges, ty=ty)
    printed, _ = _roundtrip_expr(region)
    assert "I.TensorRegion(" in printed


def test_typed_load_conversion_and_raw_call_boundary():
    a = tirx.decl_tensor((8, 8), "float32", name="A", scope="local")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    load = a[i, j]
    region = tvm_ffi.get_global_func("tirx.AsTensorRegion")(load)
    assert isinstance(region, ir.TensorRegion)
    assert region.source.same_as(a)
    assert region.ty == ir.TensorRegionType()
    for actual, index in zip(region.region, (i, j)):
        assert actual.min.same_as(index)
        assert actual.extent.value == 1
    assert ir.Call("prim.exp", [load]).args[0].same_as(load)
    call = T.cuda.tile.sqrt(load, load)
    assert all(isinstance(arg, ir.TensorRegion) for arg in call.args)
    call.validate()
    raw = ir.Call(call.op, [load, load], attrs=call.attrs, ty=call.ty)
    with pytest.raises(TypeError, match="TensorRegion"):
        raw.validate()


def test_instruction_canonical_empty_tuple_and_attrs():
    a = tirx.decl_tensor((16, 16), "float32", name="A", scope="shared")
    original = T.cuda.tile.cp_async_bulk_tensor_load(a, a, 0, descriptor_mode="explicit")
    values = list(original.args)
    canonical = T.cuda.tile.cp_async_bulk_tensor_load(*values, attrs=original.attrs, ty=original.ty)
    ir.assert_structural_equal(original, canonical)
    source, _ = _roundtrip_expr(original)
    assert "I.Call(" not in source
    assert "()" in source
    assert original.op.get_attr("__tvm_doc_translate_op_call__") is None


@pytest.mark.parametrize(
    "name,args",
    [
        ("device_entry", []),
        ("device_context", [1, 0]),
        ("compute_scope", ["compute"]),
        ("parallel_launch", []),
        ("launch_thread", ["threadIdx.x", tirx.IntImm("int64", 8)]),
    ],
)
def test_named_region_canonical_builders(name, args):
    op = ir.Op.get("tirx." + name)
    params = [tirx.Var("thread", "int64")] if name == "launch_thread" else []
    expected = ir.RegionStmt(
        op, args, params, {}, ir.SeqStmt([ir.Evaluate(params[0] if params else 0)])
    )
    generated = _make_op_api(op, __name__)
    with IRBuilder() as builder:
        with generated(*args, attrs={}, body_params=params):
            T.evaluate(params[0] if params else 0)
    ir.assert_structural_equal(expected, builder.get())
    with IRBuilder() as builder:
        with getattr(T, name)(*args, attrs={}, body_params=params):
            T.evaluate(params[0] if params else 0)
    ir.assert_structural_equal(expected, builder.get())
    function = tirx.Function([], ir.SeqStmt([expected]))
    printed = function.script()
    assert f"with T.{name}(" in printed
    restored = tvm.script.from_source(printed, extra_vars={"T": T, "I": I})
    ir.assert_structural_equal(function, restored)
    if params:
        region = restored.body[0]
        assert region.body[0].value.same_as(region.body_params[0])
    with pytest.raises(TypeError, match="region builders"):
        generated(*args, ty="void")


def test_generated_custom_region_and_unnamed_fallback(monkeypatch):
    name = "test.canonical_region"
    register_op_attr(
        name,
        "FRegionGetBodyParams",
        ir.Op.get("tirx.device_entry").get_attr("FRegionGetBodyParams"),
    )
    op = ir.Op.get(name)
    op.set_signature(["value"])
    region = ir.RegionStmt(op, [(1, (2,))], [], {"tag": "kept"}, ir.SeqStmt([ir.Evaluate(0)]))
    function = tirx.Function([], ir.SeqStmt([region]))
    raw = function.script()
    assert 'T.region("test.canonical_region"' in raw
    ir.assert_structural_equal(function, tvm.script.from_source(raw, extra_vars={"T": T, "I": I}))
    op.set_attr("TScriptPrinterName", "tirx.test_canonical_region")
    generated = _make_op_api(op, __name__)
    monkeypatch.setattr(T, "test_canonical_region", generated, raising=False)
    named = function.script()
    assert "T.test_canonical_region((" in named
    assert 'attrs={"tag": "kept"}' in named
    ir.assert_structural_equal(function, tvm.script.from_source(named, extra_vars={"T": T, "I": I}))


@pytest.mark.parametrize("names", [("n", "m", "k"), ("z", "a", "q")])
def test_signature_symbols_follow_first_appearance(names):
    n, m, k = [T.dynamic(name) for name in names]

    @T.function
    def function(A: T.Tensor((n, m), "float32"), B: T.Tensor((m, k), "float32")):
        T.evaluate(0)

    source = function.script()
    assert "[" + ", ".join(names) + "](" in source
    ir.assert_structural_equal(
        function, tvm.script.from_source(source, extra_vars={"T": T, "I": I})
    )


def test_relax_packed_call_recurses_through_canonical_tuple():
    fields = ir.Tuple([tirx.IntImm("int32", 1), ir.Tuple([tirx.FloatImm("float32", 2)])])
    call = tvm.relax.call_dps_packed(
        "packed", fields, ty_args=[tvm.relax.TensorType((1,), "float32")]
    )
    source, _ = _roundtrip_expr(call)
    assert "R.call_dps_packed(" in source
    assert "I.Tuple(" not in source


@pytest.mark.parametrize("backend", ["cuda", "trn"])
def test_registered_instruction_families_use_default_call_printing(backend):
    tvm.backend.load(backend)
    from tvm.tirx.tensor_instruction import SPECS

    a = tirx.decl_tensor((16, 16), "float32", name="A", scope="shared")
    for name, spec in SPECS.items():
        if spec.backend != backend:
            continue
        values = []
        for operand in spec.operands:
            if operand.role == "coordinates":
                value = (0, 1, 2, 3)
            elif operand.role == "selectors":
                value = [(True, a[:, :])]
            elif operand.role in ("scalar", "address"):
                value = 1
            else:
                value = a
            values.append(value)
        call = spec.make(*values)
        source, restored = _roundtrip_expr(call)
        assert "I.Call(" not in source, name
        assert call.op.get_attr("__tvm_doc_translate_op_call__") is None, name
        restored.validate()


@pytest.mark.parametrize("nested", [False, True])
def test_llvm_intrinsic_region_operand_reconstruction(nested):
    lookup = tvm_ffi.get_global_func("target.llvm_lookup_intrinsic_id", allow_missing=True)
    if lookup is None:
        pytest.skip("LLVM support is unavailable")
    a = tirx.decl_tensor((4,), "float32", name="A")
    region = a[0:1]
    arg = ir.Tuple([region]) if nested else region
    call = ir.Call(
        "tirx.call_llvm_intrin", [tirx.IntImm("int32", lookup("llvm.prefetch")), arg], ty="void"
    )
    source, _ = _roundtrip_expr(call)
    assert ("T.call_llvm_intrin(" if nested else "I.Call(") in source
    assert "I.Tuple(" not in source


def test_region_and_tuple_diagnostic_origins():
    a = tirx.decl_tensor((4,), "float32", name="A")
    region = a[0:1]
    call = ir.Call("prim.exp", [(region,)], ty="void")
    source = call.script(obj_to_underline=[region])
    assert "^" in source
    assert "A[0:1]" in source


def test_typed_region_binding_preserves_span_and_source():
    a = tirx.decl_tensor((8,), "float32", name="A")
    b = tirx.decl_tensor((1,), "float32", name="B")
    span = ir.Span(ir.SourceName("typed-region"), 1, 1, 1, 5)
    load = tirx.TensorLoad(a, [2], span=span)
    match = tvm.s_tir.MatchBufferRegion(b, load)
    assert isinstance(match.source, ir.TensorRegion)
    assert match.source.source.same_as(a)
    assert match.source.span.same_as(span)
    assert match.source.ty == ir.TensorRegionType()
    call = T.cuda.tile.sqrt(load, load, span=span)
    assert call.span.same_as(span)
    assert all(arg.span.same_as(span) for arg in call.args)
