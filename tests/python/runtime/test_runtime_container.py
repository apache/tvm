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

import pickle
import random

import numpy as np

import tvm
import tvm.testing
from tvm import nd, relay
from tvm.runtime import container as _container


def test_adt_constructor():
    arr = nd.array([1, 2, 3])
    fields = [arr, arr]
    y = _container.ADT(0, [arr, arr])

    assert len(y) == 2
    assert isinstance(y, _container.ADT)
    y[0:1][-1] == arr
    assert y.tag == 0
    assert isinstance(arr, nd.NDArray)


def test_tuple_object():
    x = relay.var(
        "x",
        type_annotation=relay.ty.TupleType(
            [relay.ty.TensorType((), "int32"), relay.ty.TensorType((), "int32")]
        ),
    )

    fn = relay.Function([x], relay.expr.TupleGetItem(x, 0))
    mod = tvm.IRModule.from_expr(fn)

    f = relay.create_executor(kind="vm", mod=mod, device=nd.cpu(), target="llvm").evaluate()
    value_tuple = _container.tuple_object([nd.array(np.array(11)), nd.array(np.array(12))])
    # pass an ADT object to evaluate
    out = f(value_tuple)
    tvm.testing.assert_allclose(out.numpy(), np.array(11))


def test_string():
    s = tvm.runtime.String("xyz")

    assert isinstance(s, tvm.runtime.String)
    assert isinstance(s, str)
    assert s.startswith("xy")
    assert s + "1" == "xyz1"
    y = tvm.testing.echo(s)
    assert isinstance(y, tvm.runtime.String)
    assert s.__tvm_object__.same_as(y.__tvm_object__)
    assert s == y

    x = tvm.ir.load_json(tvm.ir.save_json(y))
    assert isinstance(x, tvm.runtime.String)
    assert x == y

    # test pickle
    z = pickle.loads(pickle.dumps(s))
    assert isinstance(z, tvm.runtime.String)
    assert s == z


def test_shape_tuple():
    shape = [random.randint(-10, 10) for _ in range(5)]
    stuple = _container.ShapeTuple(shape)
    len(stuple) == len(shape)
    for a, b in zip(stuple, shape):
        assert a == b
    # ShapleTuple vs. list
    assert stuple == list(shape)
    # ShapleTuple vs. tuple
    assert stuple == tuple(shape)
    # ShapleTuple vs. ShapeTuple
    assert stuple == _container.ShapeTuple(shape)

    # test pickle
    z = pickle.loads(pickle.dumps(stuple))
    assert isinstance(z, tvm.runtime.ShapeTuple)
    assert stuple == z


def test_bool_argument():
    """Boolean objects are currently stored as int"""
    func = tvm.get_global_func("testing.AcceptsBool")

    assert isinstance(func(True), bool)
    assert isinstance(func(1), bool)
    assert isinstance(func(0), bool)


def test_int_argument():
    func = tvm.get_global_func("testing.AcceptsInt")

    assert isinstance(func(True), int)
    assert isinstance(func(1), int)
    assert isinstance(func(0), int)


def test_object_ref_argument():
    func = tvm.get_global_func("testing.AcceptsObjectRef")

    assert isinstance(func(True), bool)
    assert isinstance(func(1), int)
    assert isinstance(func(3.5), float)
    assert func(3.5) == 3.5


def test_object_ref_array_argument():
    func = tvm.get_global_func("testing.AcceptsObjectRefArray")

    assert isinstance(func([True, 17, "hello"]), bool)
    assert isinstance(func([True]), bool)
    assert isinstance(func([17]), int)
    assert isinstance(func(["hello"]), str)


def test_map_argument_returns_value():
    func = tvm.get_global_func("testing.AcceptsMapReturnsValue")

    res = func({"a": 1, "b": 2}, "a")
    assert isinstance(res, int)
    assert res == 1

    res = func({"a": True, "b": False}, "a")
    assert isinstance(res, bool)
    assert res == True


def test_map_argument_returns_map():
    func = tvm.get_global_func("testing.AcceptsMapReturnsMap")

    res = func({"a": 1, "b": 2})
    for key, value in res.items():
        assert isinstance(key, str)
        assert isinstance(value, int)

    res = func({"a": False, "b": True})
    for key, value in res.items():
        assert isinstance(key, str)
        assert isinstance(value, bool)


if __name__ == "__main__":
    tvm.testing.main()
