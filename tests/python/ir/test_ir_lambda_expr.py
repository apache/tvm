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
"""Test shared typed staging lambdas."""

import json

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import ir
from tvm.script import ir as I
from tvm.script import tirx as T


def test_typed_construction_and_single_evaluation():
    calls = []

    def body(x, y):
        calls.append((x, y))
        return [x + y, (True, 1)]

    value = ir.LambdaExpr([T.float32, ir.PrimType("float32")], body)
    assert isinstance(value, ir.StagingExpr)
    assert len(calls) == 1
    assert value.vars[0].same_as(calls[0][0])
    expected = ir.FuncType(
        [ir.PrimType("float32")] * 2,
        ir.TupleType(
            [ir.PrimType("float32"), ir.TupleType([ir.PrimType("bool"), ir.PrimType("int32")])]
        ),
    )
    ir.assert_structural_equal(value.ty, expected)
    assert isinstance(value.body, ir.Tuple)
    assert isinstance(value.body[1], ir.Tuple)
    assert T.TypedLambda is I.TypedLambda


def test_generic_expr_apply_and_simultaneous_substitution():
    tensor_ty = tvm.relax.TensorType([4], "float32")
    identity = ir.LambdaExpr([tensor_ty], lambda tensor: tensor)
    arg = ir.Var("tensor", tensor_ty)
    assert identity.apply([arg]).same_as(arg)

    pair = ir.LambdaExpr(["int32", "int32"], lambda x, y: (x, y))
    swapped = pair.apply([pair.vars[1], pair.vars[0]])
    assert swapped[0].same_as(pair.vars[1])
    assert swapped[1].same_as(pair.vars[0])


def test_nested_capture_and_shadowing():
    capture = ir.Var("x", "int32")
    outer = ir.LambdaExpr(
        ["int32"],
        lambda x: ir.LambdaExpr(["int32"], lambda y: (x, y, capture)),
    )
    inner = outer.apply([11])
    result = inner.apply([22])
    assert result[0].value == 11
    assert result[1].value == 22
    assert result[2].same_as(capture)

    shadowed = ir.LambdaExpr(["int32"], lambda x: ir.LambdaExpr(["int32"], lambda x: x))
    assert shadowed.apply([11]).apply([22]).value == 22


def test_apply_avoids_capture_and_respects_reused_binder_identity():
    construct = tvm_ffi.get_global_func("ir.LambdaExpr")
    outer_var = ir.Var("x", "int32")
    inner_var = ir.Var("y", "int32")
    inner = construct([inner_var], ir.Tuple([outer_var, inner_var]))
    outer = construct([outer_var], inner)
    applied = outer.apply([inner_var])
    assert not applied.vars[0].same_as(inner_var)
    assert applied.body[0].same_as(inner_var)
    assert applied.body[1].same_as(applied.vars[0])

    shadowed = construct([outer_var], construct([outer_var], outer_var))
    result = shadowed.apply([11])
    assert result.body.same_as(result.vars[0])
    assert result.apply([22]).value == 22


def test_apply_avoids_capture_by_let_binder():
    construct = tvm_ffi.get_global_func("ir.LambdaExpr")
    parameter = ir.Var("x", "int32")
    binder = ir.Var("y", "int32")
    value = construct([parameter], ir.prim.Let(binder, parameter, parameter + binder))
    applied = value.apply([binder])
    assert not applied.var.same_as(binder)
    assert applied.value.same_as(binder)
    assert applied.body.a.same_as(binder)
    assert applied.body.b.same_as(applied.var)
    assert applied.ty == ir.PrimType("int32")


@pytest.mark.parametrize("dataflow", [False, True])
def test_apply_avoids_capture_by_relax_binding(dataflow):
    construct = tvm_ffi.get_global_func("ir.LambdaExpr")
    parameter = ir.Var("x", "int32")
    var_class = tvm.relax.DataflowVar if dataflow else ir.Var
    block_class = tvm.relax.DataflowBlock if dataflow else tvm.relax.BindingBlock
    binder = var_class("y", ir.PrimType("int32"))
    binding = tvm.relax.VarBinding(binder, tvm.runtime.const(1))
    body = tvm.relax.SeqExpr([block_class([binding])], parameter)
    applied = construct([parameter], body).apply([binder])
    assert type(applied.blocks[0].bindings[0].var) is var_class
    assert not applied.blocks[0].bindings[0].var.same_as(binder)
    assert applied.body.same_as(binder)


def test_apply_restores_remaps_between_sibling_binding_scopes():
    construct = tvm_ffi.get_global_func("ir.LambdaExpr")
    parameter = ir.Var("x", "int32")
    binder = ir.Var("y", "int32")
    first = tvm.relax.SeqExpr(
        [tvm.relax.BindingBlock([tvm.relax.VarBinding(binder, tvm.runtime.const(1))])], binder
    )
    second = tvm.relax.SeqExpr(
        [tvm.relax.BindingBlock([tvm.relax.VarBinding(binder, tvm.runtime.const(2))])],
        ir.Tuple([parameter, binder]),
    )
    applied = construct([parameter], ir.Tuple([first, second, binder])).apply([binder])
    first_binder = applied[0].blocks[0].bindings[0].var
    second_binder = applied[1].blocks[0].bindings[0].var
    assert not first_binder.same_as(binder)
    assert not second_binder.same_as(binder)
    assert not first_binder.same_as(second_binder)
    assert applied[0].body.same_as(first_binder)
    assert applied[1].body[0].same_as(binder)
    assert applied[1].body[1].same_as(second_binder)
    assert applied[2].same_as(binder)


def test_apply_rewrites_dependent_nested_parameter_types():
    construct = tvm_ffi.get_global_func("ir.LambdaExpr")
    extent = ir.Var("n", "int64")
    parameter = ir.Var("x", tvm.relax.TensorType([extent], "float32"))
    value = construct([extent], construct([parameter], parameter))
    applied = value.apply([tvm.runtime.const(4, "int64")])
    expected_type = tvm.relax.TensorType([4], "float32")
    assert applied.vars[0].ty.shape[0].value == 4
    assert applied.body.same_as(applied.vars[0])
    ir.assert_structural_equal(applied.vars[0].ty, expected_type)
    ir.assert_structural_equal(applied.ty, ir.FuncType([expected_type], expected_type))


def test_apply_checks_dependent_parameter_types_after_substitution():
    construct = tvm_ffi.get_global_func("ir.LambdaExpr")
    extent = ir.Var("n", "int64")
    parameter = ir.Var("x", tvm.relax.TensorType([extent], "float32"))
    value = construct([extent, parameter], parameter)
    argument = ir.Var("arg", tvm.relax.TensorType([4], "float32"))
    assert value.apply([tvm.runtime.const(4, "int64"), argument]).same_as(argument)
    with pytest.raises(TypeError, match="argument type mismatch at index 1"):
        value.apply([tvm.runtime.const(5, "int64"), argument])


def test_native_validation():
    construct = tvm_ffi.get_global_func("ir.LambdaExpr")
    var = ir.Var("x", "int32")
    with pytest.raises((ValueError, TypeError), match="[Dd]uplicate|distinct|unique"):
        construct([var, var], var)
    with pytest.raises((ValueError, TypeError), match="[Mm]issing|type"):
        construct([ir.Var("untyped")], var)
    with pytest.raises(TypeError, match="[Mm]issing|type"):
        construct([ir.Var("partially_typed", ir.TupleType([ir.Type.missing()]))], var)
    value = ir.LambdaExpr(["int32"], lambda x: x)
    with pytest.raises((ValueError, TypeError), match="argument|arity|parameter"):
        value.apply([])
    with pytest.raises((ValueError, TypeError), match="type"):
        value.apply([tvm.runtime.const(1, "int64")])


@pytest.mark.parametrize(
    "function,types",
    [(lambda *args: 0, []), (lambda **kwargs: 0, []), (lambda *, x: x, ["int32"])],
)
def test_reject_nonpositional_signature(function, types):
    with pytest.raises(TypeError, match="fixed positional"):
        ir.LambdaExpr(types, function)


def test_explicit_types_and_return_annotation():
    with pytest.raises(ValueError, match="one explicit type"):
        ir.LambdaExpr([], lambda x: x)
    with pytest.raises(TypeError, match="callable"):
        ir.LambdaExpr([], 0)
    with pytest.raises(TypeError, match="MissingType"):
        ir.LambdaExpr([ir.Type.missing()], lambda x: x)
    with pytest.raises(TypeError, match="does not match"):
        ir.LambdaExpr(["int32"], lambda x: x, ret_type="int64")
    with pytest.raises(TypeError, match="known body type"):
        ir.LambdaExpr([], lambda: ir.Var("unknown"), ret_type="int32")
    unknown = ir.LambdaExpr([], lambda: ir.Var("unknown"))
    assert isinstance(unknown.ty, ir.FuncType)
    assert isinstance(unknown.ty.ret_type, ir.MissingType)
    typed = ir.LambdaExpr([T.f32], lambda x: (x,), ret_type=T.Tuple(T.f32))
    assert isinstance(typed.body, ir.Tuple)
    assert typed.body[0].same_as(typed.vars[0])


@pytest.mark.parametrize("inplace", [False, True])
def test_rewrite_refreshes_func_type_and_bound_uses(inplace):
    value = ir.LambdaExpr(["int32"], lambda x: (x,))
    if inplace:
        value = value._move()
    rewritten = tvm_ffi.structural_map(
        value,
        with_def_region_kind=(ir.Var, lambda var, kind: ir.Var(var.name, "int64") if kind else var),
    )
    assert rewritten.vars[0].ty == ir.PrimType("int64")
    assert rewritten.body[0].same_as(rewritten.vars[0])
    ir.assert_structural_equal(
        rewritten.ty, ir.FuncType([ir.PrimType("int64")], ir.TupleType([ir.PrimType("int64")]))
    )


@pytest.mark.parametrize("inplace", [False, True])
def test_rewrite_refreshes_tuple_projection_and_func_type(inplace):
    value = ir.LambdaExpr([ir.TupleType([ir.PrimType("int32")])], lambda x: ir.TupleGetItem(x, 0))
    if inplace:
        value = value._move()
    rewritten = tvm_ffi.structural_map(
        value,
        with_def_region_kind=(
            ir.Var,
            lambda var, kind: ir.Var(var.name, ir.TupleType([ir.PrimType("int64")]))
            if kind
            else var,
        ),
    )
    assert rewritten.body.tuple_value.same_as(rewritten.vars[0])
    assert rewritten.body.ty == ir.PrimType("int64")
    ir.assert_structural_equal(
        rewritten.ty,
        ir.FuncType([ir.TupleType([ir.PrimType("int64")])], ir.PrimType("int64")),
    )


def test_alpha_equivalence_and_json_roundtrip():
    first = ir.LambdaExpr(["int32"], lambda x: (x, x))
    second = ir.LambdaExpr(["int32"], lambda renamed: (renamed, renamed))
    ir.assert_structural_equal(first, second)
    assert tvm_ffi.structural_hash(first) == tvm_ffi.structural_hash(second)
    decoded = ir.load_json(ir.save_json(first))
    ir.assert_structural_equal(decoded, first)
    assert decoded.body[0].same_as(decoded.vars[0])
    assert decoded.body[1].same_as(decoded.vars[0])


def test_dependent_alpha_equivalence_and_json_roundtrip():
    first = ir.LambdaExpr(
        ["int64"], lambda n: ir.LambdaExpr([tvm.relax.TensorType([n], "float32")], lambda x: x)
    )
    second = ir.LambdaExpr(
        ["int64"],
        lambda extent: ir.LambdaExpr([tvm.relax.TensorType([extent], "float32")], lambda y: y),
    )
    ir.assert_structural_equal(first, second)
    assert tvm_ffi.structural_hash(first) == tvm_ffi.structural_hash(second)
    decoded = ir.load_json(ir.save_json(first))
    ir.assert_structural_equal(decoded, first)
    assert tvm_ffi.structural_hash(decoded) == tvm_ffi.structural_hash(first)
    assert decoded.body.vars[0].ty.shape[0].same_as(decoded.vars[0])
    assert decoded.body.body.same_as(decoded.body.vars[0])
    assert decoded.ty.ret_type.arg_types[0].shape[0].same_as(decoded.vars[0])
    applied = decoded.apply([tvm.runtime.const(4, "int64")])
    assert applied.vars[0].ty.shape[0].value == 4
    assert applied.body.same_as(applied.vars[0])


def test_parameter_type_free_variables_are_captures():
    first_extent = ir.Var("n", "int64")
    second_extent = ir.Var("n", "int64")
    first = ir.LambdaExpr([tvm.relax.TensorType([first_extent], "float32")], lambda x: x)
    renamed = ir.LambdaExpr([tvm.relax.TensorType([first_extent], "float32")], lambda y: y)
    different_capture = ir.LambdaExpr(
        [tvm.relax.TensorType([second_extent], "float32")], lambda x: x
    )
    ir.assert_structural_equal(first, renamed)
    assert tvm_ffi.structural_hash(first) == tvm_ffi.structural_hash(renamed)
    assert not tvm_ffi.structural_equal(first, different_capture)


def test_legacy_json_preserves_binder_and_capture_identity():
    capture = ir.Var("capture", "int32")
    value = ir.LambdaExpr(["int32"], lambda x: x + capture)
    graph = json.loads(ir.save_json(ir.Tuple([value, capture])))
    for node in graph["nodes"]:
        if node.get("type") == "ir.LambdaExpr":
            node["type"] = "tirx.LambdaExpr"
            node["data"]["pred"] = node["data"].pop("body")
            node["data"]["ty"] = len(graph["nodes"])
            graph["nodes"].append({"type": "ir.MissingType", "data": {}})
            break
    decoded = ir.load_json(json.dumps(graph))
    ir.assert_structural_equal(decoded[0], value, map_free_vars=True)
    assert decoded[0].body.a.same_as(decoded[0].vars[0])
    assert decoded[0].body.b.same_as(decoded[1])
    assert isinstance(decoded[0].ty, ir.FuncType)
    assert "tirx.LambdaExpr" not in ir.save_json(decoded)


@pytest.mark.parametrize(
    "types,body,message",
    [
        (["int32"], lambda x: x < 1, "per destination axis"),
        (["int64", "int64"], lambda x, y: x < y, "int32"),
        (["int32", "int32"], lambda x, y: (x < y,), "scalar boolean"),
        (["int32", "int32"], lambda x, y: x + y, "scalar boolean"),
    ],
)
def test_tile_select_rejects_nonpredicate_lambda(types, body, message):
    from tvm.tirx.script.ir_builder import tile

    dst = tvm.tirx.decl_tensor((2, 3), "float32")
    predicate = ir.LambdaExpr(types, body)
    with pytest.raises(TypeError, match=message):
        tile.select(dst, 1.0, 0.0, predicate)


def test_typed_lambda_script_roundtrip():
    value = ir.LambdaExpr(["float32", "float32"], lambda x, y: (x + y,))
    script = value.script()
    assert "TypedLambda" in script
    assert "float32" in script
    restored = eval(script, {"I": I, "T": T})  # pylint: disable=eval-used
    ir.assert_structural_equal(value, restored)

    call = ir.Call("tirx.call_extern", [ir.StringImm("consume"), value], ty="int32")
    func = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(call))
    parsed = tvm.script.from_source(func.script(), extra_vars={"I": I, "T": T})
    ir.assert_structural_equal(func, parsed)


if __name__ == "__main__":
    tvm.testing.main()
