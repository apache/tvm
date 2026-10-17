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
"""Test type nodes in the IR"""

import uuid

import pytest
import tvm_ffi

import tvm_tirx
from tvm_tirx.runtime import MemoryScope


def check_json_roundtrip(node):
    json_str = tvm_tirx.ir.save_json(node)
    back = tvm_tirx.ir.load_json(json_str)
    tvm_tirx.ir.assert_structural_equal(back, node, map_free_vars=True)


def test_missing_type():
    missing = tvm_tirx.ir.Type.missing()

    assert isinstance(missing, tvm_tirx.ir.Type)
    assert isinstance(missing, tvm_tirx.ir.MissingType)


def test_prim_type():
    x = tvm_tirx.ir.PrimType("int32")
    assert isinstance(x, tvm_tirx.ir.PrimType)
    assert x.dtype == "int32"
    with pytest.raises(TypeError, match="PtrType::VoidPtrType"):
        tvm_tirx.ir.PrimType("handle")


def test_func_type():
    arg_types = tvm_ffi.convert([])
    ret_type = tvm_tirx.ir.PrimType("float32")
    tf = tvm_tirx.ir.FuncType(arg_types, ret_type)
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert isinstance(tf.loc, tvm_tirx.ir.UnknownLoc)
    assert tf.loc.same_as(tvm_tirx.ir.UnknownLoc())
    # TODO make sure we can set loc
    str(tf)
    check_json_roundtrip(tf)


def test_tuple_type():
    tf = tvm_tirx.ir.FuncType([], tvm_tirx.ir.TupleType([]))
    tt = tvm_tirx.ir.PrimType("float32")
    fields = tvm_ffi.convert([tf, tt])

    tup_ty = tvm_tirx.ir.TupleType(fields)
    assert tup_ty.loc.same_as(tvm_tirx.ir.UnknownLoc())
    assert tup_ty.fields == fields
    str(tup_ty)
    check_json_roundtrip(tup_ty)


def test_memory_scope_json_roundtrip():
    custom = MemoryScope.register("test.ir.scope." + uuid.uuid4().hex)
    for scope in (MemoryScope("global"), MemoryScope("shared.dyn"), custom):
        assert tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(scope)).same_as(scope)


if __name__ == "__main__":
    test_func_type()
    test_tuple_type()
