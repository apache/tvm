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


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    assert tvm.ir.structural_equal(back, node, map_free_vars=True)


def test_prim_type():
    x = tvm.ir.PrimType("int32")
    assert isinstance(x, tvm.ir.PrimType)
    assert x.dtype == "int32"


def test_tensor_type_bad_constructor():
    try:
        x = tvm.ir.TensorType("xx", "xx")
    except tvm.error.TVMError:
        pass


def test_tensor_type():
    shape = tvm.runtime.convert([1, 2, 3])
    dtype = "float32"
    tt = tvm.ir.TensorType(shape, dtype)
    assert tt.dtype == dtype
    assert tt.shape == shape
    assert tt.span == None
    str(tt)
    check_json_roundtrip(tt)


def test_type_param():
    tp = tvm.ir.TypeVar("name", tvm.ir.TypeKind.Type)
    assert tp.kind == tvm.ir.TypeKind.Type
    # assert tp.span  # TODO allow us to set span
    str(tp)
    check_json_roundtrip(tp)


def test_func_type():
    type_params = tvm.runtime.convert([])
    type_constraints = tvm.runtime.convert([])  # TODO: fill me in
    arg_types = tvm.runtime.convert([])
    ret_type = tvm.ir.TensorType((1, 2, 3), "float32")
    tf = tvm.ir.FuncType(arg_types, ret_type, type_params, type_constraints)
    assert tf.type_params == type_params
    assert tf.type_constraints == type_constraints
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert tf.span == None
    # TODO make sure we can set span
    str(tf)
    check_json_roundtrip(tf)


def test_tuple_type():
    tp = tvm.ir.TypeVar("tp", tvm.ir.TypeKind.Type)
    tf = tvm.ir.FuncType([], tvm.ir.TupleType([]), [], [])
    tt = tvm.ir.TensorType(tvm.runtime.convert([1, 2, 3]), "float32")
    fields = tvm.runtime.convert([tp, tf, tt])

    tup_ty = tvm.ir.TupleType(fields)
    assert tup_ty.fields == fields
    str(tup_ty)
    check_json_roundtrip(tup_ty)


def test_type_relation():
    tp = tvm.ir.TypeVar("tp", tvm.ir.TypeKind.Type)
    tf = tvm.ir.FuncType([], None, [], [])
    tt = tvm.ir.TensorType(tvm.runtime.convert([1, 2, 3]), "float32")
    args = tvm.runtime.convert([tp, tf, tt])

    num_inputs = 2
    func = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Broadcast")
    attrs = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3, 4))

    tr = tvm.ir.TypeRelation(func, args, num_inputs, attrs)
    assert tr.args == args
    assert tr.num_inputs == num_inputs
    str(tr)
    check_json_roundtrip(tr)


if __name__ == "__main__":
    test_tensor_type_bad_constructor()
    test_tensor_type()
    test_type_param()
    test_func_type()
    test_tuple_type()
    test_type_relation()
