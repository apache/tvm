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

import json

import pytest
import tvm_ffi

import tvm
import tvm.testing


def test_tuple_core_construction_and_relax_aliases():
    x = tvm.ir.Var("x", "int64")
    y = tvm.ir.Var("y", "float32")
    value = tvm.ir.Tuple([x, tvm.ir.Tuple([y])])

    assert tvm.relax.Tuple is tvm.ir.Tuple
    assert tvm.relax.TupleGetItem is tvm.ir.TupleGetItem
    assert isinstance(value, tvm.relax.Tuple)
    assert value[0].same_as(x)
    assert value[-1].fields[0].same_as(y)
    tvm.ir.assert_structural_equal(
        value.ty,
        tvm.ir.TupleType(
            [tvm.ir.PrimType("int64"), tvm.ir.TupleType([tvm.ir.PrimType("float32")])]
        ),
    )


def test_tuple_projection_and_bounds():
    value = tvm.ir.Tuple([tvm.ir.Var("x", "int64"), tvm.ir.Var("y", "float32")])
    projection = tvm.ir.TupleGetItem(value, 1)

    assert projection.tuple_value.same_as(value)
    assert projection.index == 1
    tvm.ir.assert_structural_equal(projection.ty, tvm.ir.PrimType("float32"))
    assert str(value) == "(x, y)"
    assert str(projection) == "(x, y)[1]"
    assert str(tvm.ir.Tuple([])) == "R.tuple()"

    with pytest.raises(IndexError, match="Tuple index out of range"):
        value[2]
    with pytest.raises(tvm.error.InternalError, match="Index out of bounds"):
        tvm.ir.TupleGetItem(value, -1)
    with pytest.raises(tvm.error.InternalError, match="Index out of bounds"):
        tvm.ir.TupleGetItem(value, 2)


def test_tuple_type_requires_known_field_types():
    unknown = tvm.ir.Var("unknown")
    value = tvm.ir.Tuple([tvm.ir.Var("known", "int32"), unknown])

    assert value.ty.is_missing()
    assert tvm.ir.TupleGetItem(value, 0).ty.is_missing()


def test_tuple_reflection_uses_core_type_key():
    value = tvm.ir.TupleGetItem(tvm.ir.Tuple([tvm.ir.Var("x", "int64")]), 0)
    serialized = tvm.ir.save_json(value)
    restored = tvm.ir.load_json(serialized)

    type_keys = {node["type"] for node in json.loads(serialized)["nodes"]}
    assert {"ir.Tuple", "ir.TupleGetItem"}.issubset(type_keys)
    assert isinstance(restored, tvm.ir.TupleGetItem)
    tvm.ir.assert_structural_equal(restored, value, map_free_vars=True)


def test_tuple_legacy_and_core_ffi_constructors():
    x = tvm.ir.Var("x", "int64")
    legacy_tuple = tvm_ffi.get_global_func("relax.Tuple")([x], None)
    core_tuple = tvm_ffi.get_global_func("ir.Tuple")([x], None)

    assert isinstance(legacy_tuple, tvm.ir.Tuple)
    assert isinstance(core_tuple, tvm.relax.Tuple)
    tvm.ir.assert_structural_equal(legacy_tuple, core_tuple, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
