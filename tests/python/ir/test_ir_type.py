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
import tvm
from tvm.script import tir as T


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    tvm.ir.assert_structural_equal(back, node, map_free_vars=True)


def test_prim_type():
    x = tvm.ir.PrimType("int32")
    assert isinstance(x, tvm.ir.PrimType)
    assert x.dtype == "int32"


def test_func_type():
    arg_types = tvm.runtime.convert([])
    ret_type = tvm.ir.PrimType("float32")
    tf = tvm.ir.FuncType(arg_types, ret_type)
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert tf.span == None
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
