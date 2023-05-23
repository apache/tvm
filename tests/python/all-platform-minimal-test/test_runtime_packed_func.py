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
"""Test packed function FFI."""
import tvm
from tvm import te
import tvm.testing
import numpy as np


def test_get_global():
    targs = (10, 10.0, "hello")
    # register into global function table
    @tvm.register_func
    def my_packed_func(*args):
        assert tuple(args) == targs
        return 10

    # get it out from global function table
    f = tvm.get_global_func("my_packed_func")
    assert isinstance(f, tvm.runtime.PackedFunc)
    y = f(*targs)
    assert y == 10


def test_get_callback_with_node():
    x = tvm.runtime.convert(10)

    def test(y):
        assert y.handle != x.handle
        return y

    f2 = tvm.runtime.convert(test)
    # register into global function table
    @tvm.register_func
    def my_callback_with_node(y, f):
        assert y == x
        return f(y)

    # get it out from global function table
    f = tvm.get_global_func("my_callback_with_node")
    assert isinstance(f, tvm.runtime.PackedFunc)
    y = f(x, f2)
    assert y.value == 10


def test_return_func():
    def addy(y):
        def add(x):
            return tvm.runtime.convert(x + y)

        return add

    myf = tvm.runtime.convert(addy)
    f = myf(10)
    assert f(11).value == 21


def test_convert():
    # convert a function to tvm function
    targs = (10, 10.0, "hello", 10)

    def myfunc(*args):
        assert tuple(args) == targs

    f = tvm.runtime.convert(myfunc)
    assert isinstance(f, tvm.runtime.PackedFunc)


def test_byte_array():
    s = "hello"
    a = bytearray(s, encoding="ascii")

    def myfunc(ss):
        assert ss == a

    f = tvm.runtime.convert(myfunc)
    f(a)


def test_empty_array():
    def myfunc(ss):
        assert tuple(ss) == ()

    x = tvm.runtime.convert(())
    tvm.runtime.convert(myfunc)(x)


def test_device():
    def test_device_func(dev):
        assert tvm.cuda(7) == dev
        return tvm.cpu(0)

    x = test_device_func(tvm.cuda(7))
    assert x == tvm.cpu(0)
    x = tvm.opencl(10)
    x = tvm.testing.device_test(x, x.device_type, x.device_id)
    assert x == tvm.opencl(10)


def test_rvalue_ref():
    def callback(x, expected_count):
        assert expected_count == tvm.testing.object_use_count(x)
        return x

    f = tvm.runtime.convert(callback)

    def check0():
        x = tvm.tir.Var("x", "int32")
        assert tvm.testing.object_use_count(x) == 1
        f(x, 2)
        y = f(x._move(), 1)
        assert x.handle.value == None

    def check1():
        x = tvm.tir.Var("x", "int32")
        assert tvm.testing.object_use_count(x) == 1
        y = f(x, 2)
        z = f(x._move(), 2)
        assert x.handle.value == None
        assert y.handle.value is not None

    check0()
    check1()


def test_numpy_scalar():
    maxint = (1 << 63) - 1
    assert tvm.testing.echo(np.int64(maxint)) == maxint


def test_ndarray_args():
    def check(arr):
        assert not arr.is_view
        assert tvm.testing.object_use_count(arr) == 2

    fcheck = tvm.runtime.convert(check)
    x = tvm.nd.array([1, 2, 3])
    fcheck(x)
    assert tvm.testing.object_use_count(x) == 1


def test_dict_function_value_type():
    from tvm import tir  # pylint: disable=import-outside-toplevel

    te_func_dict = {"add": lambda a, b: a + b}

    converted_dict = tvm.runtime.convert(te_func_dict)
    f = converted_dict["add"]
    a = tir.Var("a", "float32")
    b = tir.Var("b", "float32")
    tvm.ir.assert_structural_equal(f(a, b), tir.Add(a, b))


if __name__ == "__main__":
    test_ndarray_args()
    test_numpy_scalar()
    test_rvalue_ref()
    test_empty_array()
    test_get_global()
    test_get_callback_with_node()
    test_convert()
    test_return_func()
    test_byte_array()
    test_device()
    test_dict_function_value_type()
