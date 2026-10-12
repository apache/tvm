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
"""Canonical type aliases and normalization at explicit type constructors."""

from types import ModuleType, SimpleNamespace

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import ir, relax, tirx
from tvm.ir.type import _init_type_api
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.script.ir_builder import ir as builder_ir
from tvm.tirx.script import ir_builder as builder_tirx


def test_alias_identity_and_existing_exports():
    assert I.Ptr is T.Ptr is R.Ptr is ir.PtrType
    assert T.Tensor is builder_tirx.Tensor is tirx.TensorType
    assert T.TensorMap is tirx.TensorMapType
    assert I.Func is ir.FuncType
    assert R.Func is relax.FuncType
    assert R.PackedFunc is relax.PackedFuncType
    assert I.String is T.String is R.String is ir.StringType
    assert I.Missing is ir.MissingType
    assert builder_ir.Ptr is ir.PtrType
    # Expression constructors and handwritten dialect adapters retain their names.
    assert I.Tuple is ir.Tuple
    assert I.TensorRegion is ir.TensorRegion
    assert R.Tensor is not relax.TensorType
    assert R.Callable is not relax.FuncType
    assert R.Any is R.Object
    assert ir.PtrType.__name__ == "PtrType"
    assert tirx.TensorType.__name__ == "TensorType"


def test_namespace_filter_collision_and_exports():
    root = ModuleType("root_types")
    root.__all__ = []
    _init_type_api("ir", root)
    assert root.Ptr is ir.PtrType
    assert root.TensorRegion is ir.TensorRegionType
    assert "Type" not in vars(root)
    assert "Tensor" not in vars(root)
    assert "TensorMap" not in vars(root)
    assert "Var" not in vars(root)
    assert set(root.__all__) == set(vars(root)) - {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__all__",
        "_generated_type_aliases",
    }

    dialect = ModuleType("dialect_types")
    _init_type_api("relax", dialect)
    assert dialect.Func is relax.FuncType  # Dialect precedes root ir.FuncType.
    assert dialect.Ptr is ir.PtrType
    assert "TensorMap" not in vars(dialect)

    existing = object()
    dialect.Func = existing
    _init_type_api("relax", dialect)
    assert dialect.Func is existing


def test_unbound_dialect_alias_reserves_root_collision(monkeypatch):
    module = ModuleType("pending_dialect")
    explicit = ModuleType("explicit_export")
    explicit.Func = ir.FuncType
    lookup = tvm_ffi.core._lookup_or_register_type_info_from_type_key
    info = lookup("relax.FuncType")

    def unavailable(type_key):
        if type_key == "relax.FuncType":
            return SimpleNamespace(type_ancestors=info.type_ancestors, type_cls=None)
        return lookup(type_key)

    monkeypatch.setattr(tvm_ffi.core, "_lookup_or_register_type_info_from_type_key", unavailable)
    _init_type_api("relax", module)
    _init_type_api("relax", explicit)
    assert "Func" not in vars(module)
    assert explicit.Func is ir.FuncType
    monkeypatch.setattr(tvm_ffi.core, "_lookup_or_register_type_info_from_type_key", lookup)
    _init_type_api("relax", module)
    _init_type_api("relax", explicit)
    assert module.Func is relax.FuncType
    assert explicit.Func is ir.FuncType


def test_new_binding_is_discovered_without_a_type_suffix():
    module = ModuleType("type_api_bindings")
    _init_type_api("type_api", module)
    assert "LateBound" not in vars(module)

    @tvm_ffi.dataclasses.py_class("type_api.LateBound")
    class LateBound(ir.Type):
        value: int

    _init_type_api("type_api", module)
    assert module.LateBound is LateBound
    assert module.LateBound.__name__ == "LateBound"
    assert module.LateBound(7).value == 7
    root = ModuleType("root_without_dialect")
    _init_type_api("ir", root)
    assert "LateBound" not in vars(root)


def test_late_dialect_registration_replaces_only_generated_root_alias():
    module = ModuleType("late_collision_types")
    module.__all__ = []
    explicit = ModuleType("late_explicit_types")
    explicit.Func = ir.FuncType
    overwritten = ModuleType("late_overwritten_types")
    for target in (module, explicit, overwritten):
        _init_type_api("type_api_late_collision", target)
        assert target.Func is ir.FuncType
    sentinel = object()
    overwritten.Func = sentinel

    @tvm_ffi.dataclasses.py_class("type_api_late_collision.FuncType")
    class FuncType(ir.Type):
        value: int

    for target in (module, explicit, overwritten):
        _init_type_api("type_api_late_collision", target)
    assert module.Func is FuncType
    assert module.__all__.count("Func") == 1
    assert explicit.Func is ir.FuncType
    assert overwritten.Func is sentinel


def test_late_binding_and_repeat_do_not_instantiate(monkeypatch):
    module = ModuleType("late_types")
    module.__all__ = []
    lookup = tvm_ffi.core._lookup_or_register_type_info_from_type_key
    info = lookup("ir.MissingType")

    def unavailable(type_key):
        if type_key == "ir.MissingType":
            return SimpleNamespace(type_ancestors=info.type_ancestors, type_cls=None)
        return lookup(type_key)

    def no_construction(*args, **kwargs):
        raise AssertionError("alias discovery must not instantiate a type")

    monkeypatch.setattr(ir.MissingType, "__init__", no_construction)
    monkeypatch.setattr(tvm_ffi.core, "_lookup_or_register_type_info_from_type_key", unavailable)
    _init_type_api("ir", module)
    assert "Missing" not in vars(module)
    monkeypatch.setattr(tvm_ffi.core, "_lookup_or_register_type_info_from_type_key", lookup)
    _init_type_api("ir", module)
    _init_type_api("ir", module)
    assert module.Missing is ir.MissingType
    assert module.__all__.count("Missing") == 1


@pytest.mark.parametrize("element", ["float32", tvm.DataType("float32"), T.float32])
def test_primitive_type_arguments(element):
    ir.assert_structural_equal(I.Ptr(element), ir.PtrType(ir.PrimType("float32")))
    ir.assert_structural_equal(
        I.Func([element], I.String), ir.FuncType([ir.PrimType("float32")], ir.StringType())
    )
    ir.assert_structural_equal(
        ir.TupleType([element, I.String]), ir.TupleType([ir.PrimType("float32"), ir.StringType()])
    )


def test_nested_bare_types_and_legacy_adapters():
    ir.assert_structural_equal(
        T.Ptr(T.TensorMap, "shared"), ir.PtrType(tirx.TensorMapType(), "shared")
    )
    ir.assert_structural_equal(
        I.Func([I.Ptr(I.String), I.Missing], ir.TupleType([I.String, T.TensorMap])),
        ir.FuncType(
            [ir.PtrType(ir.StringType()), ir.MissingType()],
            ir.TupleType([ir.StringType(), tirx.TensorMapType()]),
        ),
    )
    ir.assert_structural_equal(T.Tuple(I.String, T.int32), ir.TupleType([I.String, "int32"]))
    ir.assert_structural_equal(R.Tuple(I.String, T.int32), ir.TupleType([I.String, "int32"]))
    # Existing Relax adapters may still evaluate their explicitly supported factories.
    ir.assert_structural_equal(R.Tuple(lambda: ir.StringType()), ir.TupleType([ir.StringType()]))


@pytest.mark.parametrize("adapter", [T.Tuple, R.Tuple])
def test_legacy_adapters_preserve_known_type_values(adapter):
    class DtypeSentinel:
        _dtype_str = "float32"

        def __call__(self):
            raise AssertionError("known dtype constructors must not create expressions")

    class CallableString(ir.StringType):
        def __call__(self):
            raise AssertionError("a Type instance must not be treated as a factory")

    ir.assert_structural_equal(
        adapter(DtypeSentinel(), CallableString()),
        ir.TupleType([ir.PrimType("float32"), ir.StringType()]),
    )
    attempts = []

    class InvalidString(ir.StringType):
        def __init__(self):
            attempts.append(True)
            raise TypeError("invalid type constructor")

    with pytest.raises(TypeError, match="invalid type constructor"):
        adapter(InvalidString)
    assert len(attempts) == 1


def test_relax_function_type_arguments():
    ir.assert_structural_equal(
        R.Func([I.String, T.float32], I.String),
        relax.FuncType([ir.StringType(), ir.PrimType("float32")], ir.StringType()),
    )
    ir.assert_structural_equal(
        R.Func.opaque_func(ret=I.String), relax.FuncType.opaque_func(ret=ir.StringType())
    )
    ir.assert_structural_equal(R.Func.opaque_func(), relax.FuncType.opaque_func())


def test_canonical_constructors_do_not_invoke_arbitrary_factories():
    def factory():
        raise AssertionError("an arbitrary callable must not be evaluated")

    for construct in (
        lambda: I.Ptr(factory),
        lambda: I.Func([factory], I.String),
        lambda: ir.TupleType([factory]),
        lambda: R.Func([factory], I.String),
        lambda: R.Func.opaque_func(ret=factory),
        lambda: I.Ptr(ir.Type),
    ):
        with pytest.raises(TypeError, match="Expected an IR type"):
            construct()


def test_nullary_annotations_and_raw_value_instances():
    @T.function
    def function(descriptor: T.Ptr(T.TensorMap)):
        T.evaluate(0)

    ir.assert_structural_equal(function.params[0].ty, ir.PtrType(tirx.TensorMapType()))
    # Raw IR operands remain instances; aliases do not change the value boundary.
    value = I.Var("descriptor", T.Ptr(T.TensorMap))
    ir.assert_structural_equal(value.ty, function.params[0].ty)


if __name__ == "__main__":
    tvm.testing.main()
