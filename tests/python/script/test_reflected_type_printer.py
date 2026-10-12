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
"""Reflected Type constructors use field metadata without construction side effects."""

from types import SimpleNamespace

import pytest
from tvm_ffi.dataclasses import field, py_class

import tvm
import tvm.testing
from tvm import ir, relax, tirx
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


@py_class("ir.ReflectedPrinterType")
class ReflectedPrinterType(ir.Type):
    second: int
    first: int
    element: ir.Type = field(default=ir.PrimType("float32"))
    tag: str = "default"
    diagnostic: int = field(default=71, init=False)


@py_class("ir.KeywordPrinterType")
class KeywordPrinterType(ir.Type):
    positional: int
    required: int = field(kw_only=True)
    optional: int = field(default=5, kw_only=True)


_factory_calls = []


def _factory():
    _factory_calls.append(True)
    return 17


@py_class("ir.FactoryPrinterType")
class FactoryPrinterType(ir.Type):
    value: int = field(default_factory=_factory)


def _roundtrip(value):
    source = value.script()
    restored = eval(source, {"I": I, "T": T, "R": R})
    ir.assert_structural_equal(value, restored)
    assert restored.script() == source
    return source


def test_required_order_and_structural_default_omission():
    # A separately constructed default compares structurally, not by identity.
    value = ReflectedPrinterType(2, 1, element=ir.PrimType("float32"))
    assert value.script() == "I.ReflectedPrinter(2, 1)"
    value = ReflectedPrinterType(2, 1, element=ir.StringType(), tag="changed")
    source = value.script()
    assert source == 'I.ReflectedPrinter(2, 1, element=I.String(), tag="changed")'
    namespace = SimpleNamespace(ReflectedPrinter=ReflectedPrinterType, String=ir.StringType)
    ir.assert_structural_equal(value, eval(source, {"I": namespace}))
    assert "diagnostic" not in source and "loc=" not in source


def test_keyword_only_fields_preserve_constructor_contract():
    value = KeywordPrinterType(1, required=2)
    assert value.script() == "I.KeywordPrinter(1, required=2)"
    value = KeywordPrinterType(1, required=2, optional=3)
    source = value.script()
    assert source == "I.KeywordPrinter(1, required=2, optional=3)"
    namespace = SimpleNamespace(KeywordPrinter=KeywordPrinterType)
    ir.assert_structural_equal(value, eval(source, {"I": namespace}))


def test_default_factory_is_never_run_by_printer():
    value = FactoryPrinterType()
    before = len(_factory_calls)
    # Factory-produced values are always explicit even when they equal the default.
    assert value.script() == "I.FactoryPrinter(value=17)"
    assert FactoryPrinterType(value=19).script() == "I.FactoryPrinter(value=19)"
    assert len(_factory_calls) == before


@pytest.mark.parametrize(
    "value, spelling",
    [
        (ir.AnyType(), "I.Any"),
        (ir.MissingType(), "I.Missing"),
        (ir.OpaqueType(), "I.Opaque"),
        (ir.StringType(), "I.String"),
        (tirx.TensorMapType(), "T.TensorMap"),
        (relax.PackedFuncType(), "R.PackedFunc"),
    ],
)
def test_nullary_annotation_and_explicit_nested_value(value, spelling):
    assert value.script() == spelling
    # Func's reflected list and return fields require Type instances, not classes.
    wrapped = ir.FuncType([value], value)
    source = _roundtrip(wrapped)
    assert f"[{spelling}()]" in source
    assert source.endswith(f", {spelling}())")


def test_tensor_region_constructor_collision():
    assert I.TensorRegion is ir.TensorRegion
    assert _roundtrip(ir.TensorRegionType()) == "I.TensorRegionType()"
    wrapped = ir.FuncType([ir.TensorRegionType()], ir.TensorRegionType())
    assert _roundtrip(wrapped) == ("I.Func([I.TensorRegionType()], I.TensorRegionType())")


def test_namespace_alias_and_diagnostic_location():
    loc = ir.SourceLoc(ir.SourceName("reflected"), 1, 1, 1, 5)
    value = ir.AnyType(loc=loc)
    assert value.script(extra_config={"ir.prefix": "IR"}) == "IR.Any"
    assert "loc" not in value.script()


def test_pointer_fixed_default_and_nested_primitive():
    value = ir.PtrType(ir.PrimType("float32"))
    assert _roundtrip(value) == 'I.Ptr(I.PrimType("float32"))'
    value = ir.PtrType(ir.PrimType("int32"), "shared")
    assert _roundtrip(value) == 'I.Ptr(I.PrimType("int32"), storage_scope="shared")'


def test_specialized_hooks_precede_reflection():
    assert ir.PrimType("int32").script() == "T.int32"
    assert ir.TupleType([ir.PrimType("int32"), ir.PrimType("float32")]).script() == (
        "T.Tuple(T.int32, T.float32)"
    )
    assert relax.TensorType([2, 3], "float32").script().startswith("R.Tensor(")


def test_tensor_reflected_defaults_and_nondefaults():
    default = tirx.TensorType([2, 3])
    source = _roundtrip(default)
    assert source.startswith("T.Tensor(")
    for omitted in (
        "dtype=",
        "storage_scope=",
        "elem_offset=",
        "data_alignment=",
        "offset_factor=",
    ):
        assert omitted not in source
    explicit = tirx.TensorType(
        [2, 3], "int8", storage_scope="shared", data_alignment=32, offset_factor=4
    )
    source = _roundtrip(explicit)
    assert 'dtype=I.PrimType("int8")' in source
    assert 'storage_scope="shared"' in source
    assert "data_alignment=32" in source
    assert "offset_factor=4" in source
    assert "elem_offset=" not in source


@pytest.mark.parametrize(
    "fields",
    [
        {"storage_scope": ""},
        {"data_alignment": 0},
        {"data_alignment": -1},
        {"offset_factor": 0},
    ],
)
def test_noncanonical_tensor_fields_roundtrip_without_normalization(fields):
    value = ir.make_node("tirx.TensorType", shape=[], **fields)
    assert _roundtrip(value).startswith('I.make_node("tirx.TensorType"')
    assert "I.make_node" in _roundtrip(ir.FuncType([value], value))


def test_tensor_offset_is_fixed_and_explicit_offsets_roundtrip():
    int32_shape = [tirx.IntImm("int32", 2)]
    plain = T.Tensor(int32_shape)
    factored = T.Tensor(int32_shape, offset_factor=8)
    ir.assert_structural_equal(plain.elem_offset, factored.elem_offset)
    assert int(factored.elem_offset.value) == 0
    assert "elem_offset=" not in _roundtrip(factored)
    # An explicit zero of a different index dtype is semantic state, not a default.
    other_dtype = "int32" if str(plain.elem_offset.ty.dtype) == "int64" else "int64"
    explicit = T.Tensor(int32_shape, elem_offset=tirx.IntImm(other_dtype, 0))
    assert "elem_offset=" in _roundtrip(explicit)
    declared = tirx.decl_tensor(int32_shape, offset_factor=8)
    ir.assert_structural_equal(declared.elem_offset, plain.elem_offset)


def test_tensor_symbolic_offset_requires_explicit_input():
    offset = tirx.Var("offset", "int64")
    ty = T.Tensor([2], elem_offset=offset, offset_factor=4)
    assert ty.elem_offset.same_as(offset)
    assert not isinstance(T.Tensor([2], offset_factor=4).elem_offset, tirx.Var)


def test_tensor_legacy_and_canonical_keywords_agree():
    legacy = T.Tensor([2], T.float32, scope="shared", align=32, byte_offset=8)
    canonical = tirx.TensorType(
        [2], "float32", storage_scope="shared", data_alignment=32, elem_offset=2
    )
    ir.assert_structural_equal(legacy, canonical)
    assert T.Tensor is tirx.TensorType


def test_raw_value_context_keeps_nullary_instances():
    call = ir.Call(
        "prim.exp", [], attrs={"type": ir.StringType()}, ty_args=[ir.AnyType()], ty=ir.MissingType()
    )
    source = _roundtrip(call)
    assert "I.String()" in source
    assert "I.Any()" in source
    assert "I.Missing()" in source


if __name__ == "__main__":
    tvm.testing.main()
