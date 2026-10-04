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
# ruff: noqa: F401, F821
"""Test type nodes in the IR"""

import pytest
import tvm_ffi

import tvm
from tvm.script import tirx as T


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    tvm.ir.assert_structural_equal(back, node, map_free_vars=True)


def test_missing_type():
    missing = tvm.ir.MissingType()
    span = tvm.ir.Span(tvm.ir.SourceName("missing.py"), 1, 1, 1, 2)
    reconstructed = tvm.ir.make_node("ir.MissingType", span=span)
    assert not missing.same_as(reconstructed)
    for value in [
        missing,
        reconstructed,
        tvm.ir.Type.missing(),
        tvm.ir.Type.Missing(),
        tvm.ir.load_json(tvm.ir.save_json(missing)),
    ]:
        assert isinstance(value, tvm.ir.MissingType)
        assert isinstance(value, tvm.ir.Type)
        assert value.is_missing()
        assert value == missing
        assert tvm_ffi.structural_hash(value) == tvm_ffi.structural_hash(missing)

    for concrete in [tvm.ir.AnyType(), tvm.ir.PrimType("void"), tvm.ir.TupleType([])]:
        assert not concrete.is_missing()
        assert concrete != missing
    check_json_roundtrip(tvm.ir.FuncType([missing], tvm.ir.TupleType([missing])))


def test_missing_type_structural_traversal():
    missing = tvm.ir.make_node(
        "ir.MissingType", span=tvm.ir.Span(tvm.ir.SourceName("missing.py"), 1, 1, 1, 2)
    )
    visited = []
    tvm_ffi.structural_walk(missing, lambda value: visited.append(value))
    assert len(visited) == 1
    assert visited[0].same_as(missing)
    assert tvm_ffi.structural_map(missing).same_as(missing)
    assert tvm_ffi.structural_map(tvm.ir.MissingType()._move()).is_missing()


def test_missing_type_script():
    from tvm.script import ir as I

    missing = tvm.ir.MissingType()
    assert missing.script() == "I.MissingType()"
    assert I.MissingType().is_missing()
    nested = tvm.ir.FuncType([missing], missing)
    assert nested.script() == "I.FuncType([I.MissingType()], I.MissingType())"


@pytest.mark.parametrize("roundtrip", [False, True])
def test_missing_type_rejected_at_typed_boundaries(roundtrip):
    missing = tvm.ir.MissingType()
    if roundtrip:
        missing = tvm.ir.load_json(tvm.ir.save_json(missing))
    with pytest.raises(tvm.error.InternalError, match="element_type cannot be"):
        tvm.ir.PointerType(missing)
    with pytest.raises(TypeError, match="requires an expression type"):
        tvm.ir.GenericConst(0, missing)
    with pytest.raises(tvm.error.InternalError, match="type is not populated"):
        tvm.ir.reinfer_type(tvm.ir.Call("relax.abs", [tvm.ir.Var("untyped", missing)]))
    untyped = tvm.ir.Var("untyped", missing)
    with pytest.raises(tvm.error.InternalError, match="requires params to contain ty"):
        tvm.relax.Function([untyped], untyped, tvm.ir.AnyType())

    name = f"test.missing_type_inference_{roundtrip}"
    tvm.ir.register_op_attr(name, "FInferType", lambda call: missing)
    with pytest.raises(tvm.error.InternalError, match="returned Type::Missing"):
        tvm.ir.reinfer_type(tvm.ir.Call(name, []))


@pytest.mark.parametrize(
    "analyze", [tvm.relax.analysis.type_base_check, tvm.relax.analysis.type_lca]
)
def test_missing_type_analysis_rejects_missing_operands(analyze):
    missing = tvm.ir.MissingType()
    restored = tvm.ir.load_json(tvm.ir.save_json(missing))
    for other in [
        missing,
        tvm.ir.MissingType(),
        restored,
        tvm.ir.AnyType(),
        tvm.ir.PrimType("void"),
    ]:
        with pytest.raises(tvm.error.InternalError, match="requires populated types"):
            analyze(missing, other)
        with pytest.raises(tvm.error.InternalError, match="requires populated types"):
            analyze(other, restored)


def test_prim_type():
    x = tvm.ir.PrimType("int32")
    assert isinstance(x, tvm.ir.PrimType)
    assert x.dtype == "int32"
    with pytest.raises(TypeError, match="PointerType::VoidPointerTy"):
        tvm.ir.PrimType("handle")


def test_func_type():
    arg_types = tvm.runtime.convert([])
    ret_type = tvm.ir.PrimType("float32")
    tf = tvm.ir.FuncType(arg_types, ret_type)
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert tf.span is None
    # TODO make sure we can set span
    str(tf)
    check_json_roundtrip(tf)


def test_tuple_type():
    tf = tvm.ir.FuncType([], tvm.ir.TupleType([]))
    tt = tvm.ir.PrimType("float32")
    fields = tvm.runtime.convert([tf, tt])

    tup_ty = tvm.ir.TupleType(fields)
    assert tup_ty.fields == fields
    str(tup_ty)
    check_json_roundtrip(tup_ty)


if __name__ == "__main__":
    test_tensor_type_bad_constructor()
    test_func_type()
    test_tuple_type()
