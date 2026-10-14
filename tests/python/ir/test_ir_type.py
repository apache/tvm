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

import uuid

import pytest

import tvm
from tvm.runtime import MemoryScope
from tvm.script import tirx as T


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    tvm.ir.assert_structural_equal(back, node, map_free_vars=True)


def test_missing_type():
    missing = tvm.ir.Type.missing()

    assert isinstance(missing, tvm.ir.Type)
    assert isinstance(missing, tvm.ir.MissingType)


def test_prim_type():
    x = tvm.ir.PrimType("int32")
    assert isinstance(x, tvm.ir.PrimType)
    assert x.dtype == "int32"
    with pytest.raises(TypeError, match="PtrType::VoidPtrType"):
        tvm.ir.PrimType("handle")


def test_func_type():
    arg_types = tvm.runtime.convert([])
    ret_type = tvm.ir.PrimType("float32")
    tf = tvm.ir.FuncType(arg_types, ret_type)
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert isinstance(tf.loc, tvm.ir.UnknownLoc)
    assert tf.loc.same_as(tvm.ir.UnknownLoc())
    # TODO make sure we can set loc
    str(tf)
    check_json_roundtrip(tf)


def test_tuple_type():
    tf = tvm.ir.FuncType([], tvm.ir.TupleType([]))
    tt = tvm.ir.PrimType("float32")
    fields = tvm.runtime.convert([tf, tt])

    tup_ty = tvm.ir.TupleType(fields)
    assert tup_ty.loc.same_as(tvm.ir.UnknownLoc())
    assert tup_ty.fields == fields
    str(tup_ty)
    check_json_roundtrip(tup_ty)


def test_memory_scope_json_roundtrip():
    custom = MemoryScope.register("test.ir.scope." + uuid.uuid4().hex)
    for scope in (MemoryScope("global"), MemoryScope("shared.dyn"), custom):
        assert tvm.ir.load_json(tvm.ir.save_json(scope)).same_as(scope)
        ty = tvm.tirx.TensorType((4,), "float32", scope=scope)
        restored = tvm.ir.load_json(tvm.ir.save_json(ty))
        assert restored.scope.same_as(scope)
        tvm.ir.assert_structural_equal(ty, restored)


if __name__ == "__main__":
    test_tensor_type_bad_constructor()
    test_func_type()
    test_tuple_type()
