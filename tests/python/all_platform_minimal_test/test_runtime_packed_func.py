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
import numpy as np
import tvm
import tvm.testing


def test_get_global():
    """Test getting a global funtion."""
    targs = (10, 10.0, "hello")

    # register into global function table
    @tvm.register_func
    def my_packed_func(*args):  # pylint: disable=unused-variable
        assert tuple(args) == targs
        return 10

    # get it out from global function table
    func = tvm.get_global_func("my_packed_func")
    assert isinstance(func, tvm.runtime.PackedFunc)
    res = func(*targs)
    assert res == 10


def test_get_callback_with_node():
    """Test callback function with a node."""
    x = tvm.runtime.convert(10)

    def test(y):
        assert y.handle != x.handle
        return y

    test_func = tvm.runtime.convert(test)
    # register into global function table
    @tvm.register_func
    def my_callback_with_node(y, f):  # pylint: disable=unused-variable
        assert y == x
        return f(y)

    # get it out from global function table
    func = tvm.get_global_func("my_callback_with_node")
    assert isinstance(func, tvm.runtime.PackedFunc)
    res = func(x, test_func)
    assert res.value == 10


def test_return_func():
    """Test returning a function across ffi."""

    def add_y(y_in):
        def add(x_in):
            return tvm.runtime.convert(x_in + y_in)

        return add

    function_getter = tvm.runtime.convert(add_y)
    func = function_getter(10)
    assert func(11).value == 21


def test_convert():
    """convert a function to tvm function"""
    targs = (10, 10.0, "hello", 10)

    def myfunc(*args):
        assert tuple(args) == targs

    f = tvm.runtime.convert(myfunc)
    assert isinstance(f, tvm.runtime.PackedFunc)


def test_byte_array():
    """Test using byte arrays."""
    expected = "hello"
    byte_arr = bytearray(expected, encoding="ascii")

    def myfunc(input_byte_arr):
        assert input_byte_arr == byte_arr

    f = tvm.runtime.convert(myfunc)
    f(byte_arr)


def test_empty_array():
    """Test edge case of empty array."""

    def myfunc(input_arr):
        assert tuple(input_arr) == ()

    in_x = tvm.runtime.convert(())
    tvm.runtime.convert(myfunc)(in_x)


def test_device():
    """Test device use."""

    def test_device_func(dev):
        assert tvm.cuda(7) == dev
        return tvm.cpu(0)

    dev = test_device_func(tvm.cuda(7))
    assert dev == tvm.cpu(0)
    dev = tvm.opencl(10)
    dev = tvm.testing.device_test(dev, dev.device_type, dev.device_id)
    assert dev == tvm.opencl(10)


def test_rvalue_ref():
    """Test rvalue reference mechanism."""

    def callback(x, expected_count):
        assert expected_count == tvm.testing.object_use_count(x)
        return x

    callback_func = tvm.runtime.convert(callback)

    def check0():
        var_x = tvm.tir.Var("x", "int32")
        assert tvm.testing.object_use_count(var_x) == 1
        callback_func(var_x, 2)
        callback_func(var_x._move(), 1)
        assert var_x.handle.value is None

    def check1():
        var_x = tvm.tir.Var("x", "int32")
        assert tvm.testing.object_use_count(var_x) == 1
        res1 = callback_func(var_x, 2)
        callback_func(var_x._move(), 2)
        assert var_x.handle.value is None
        assert res1.handle.value is not None

    check0()
    check1()


def test_numpy_scalar():
    """Test comparing tvm against scalar types."""
    maxint = (1 << 63) - 1
    assert tvm.testing.echo(np.int64(maxint)) == maxint


def test_ndarray_args():
    """Test ndaryy args."""

    def check(arr):
        assert not arr.is_view
        assert tvm.testing.object_use_count(arr) == 2

    fcheck = tvm.runtime.convert(check)
    array = tvm.nd.array([1, 2, 3])
    fcheck(array)
    assert tvm.testing.object_use_count(array) == 1


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
