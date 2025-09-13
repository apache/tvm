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

import gc
import ctypes
import sys
import numpy as np
import tvm_ffi


def test_echo():
    fecho = tvm_ffi.get_global_func("testing.echo")
    assert isinstance(fecho, tvm_ffi.Function)
    # test each type
    assert fecho(None) is None

    # test bool
    bool_result = fecho(True)
    assert isinstance(bool_result, bool)
    assert bool_result is True
    bool_result = fecho(False)
    assert isinstance(bool_result, bool)
    assert bool_result is False

    # test int/float
    assert fecho(1) == 1
    assert fecho(1.2) == 1.2

    # test str
    str_result = fecho("hello")
    assert isinstance(str_result, str)
    assert str_result == "hello"

    # test bytes
    bytes_result = fecho(b"abc")
    assert isinstance(bytes_result, bytes)
    assert bytes_result == b"abc"

    # test dtype
    dtype_result = fecho(tvm_ffi.dtype("float32"))
    assert isinstance(dtype_result, tvm_ffi.dtype)
    assert dtype_result == tvm_ffi.dtype("float32")

    # test device
    device_result = fecho(tvm_ffi.device("cuda:1"))
    assert isinstance(device_result, tvm_ffi.Device)
    assert device_result.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLCUDA
    assert device_result.index == 1
    assert str(device_result) == "cuda:1"
    assert device_result.__repr__() == "device(type='cuda', index=1)"

    # test c_void_p
    c_void_p_result = fecho(ctypes.c_void_p(0x12345678))
    assert isinstance(c_void_p_result, ctypes.c_void_p)
    assert c_void_p_result.value == 0x12345678

    # test function: aka object
    fadd = tvm_ffi.convert(lambda a, b: a + b)
    fadd1 = fecho(fadd)
    assert fadd1(1, 2) == 3
    assert fadd1.same_as(fadd)

    def check_tensor():
        np_data = np.arange(10, dtype="int32")
        if not hasattr(np_data, "__dlpack__"):
            return
        # test Tensor
        x = tvm_ffi.from_dlpack(np_data)
        assert isinstance(x, tvm_ffi.Tensor)
        tensor_result = fecho(x)
        assert isinstance(tensor_result, tvm_ffi.Tensor)
        assert tensor_result.shape == (10,)
        assert tensor_result.dtype == tvm_ffi.dtype("int32")
        assert tensor_result.device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLCPU
        assert tensor_result.device.index == 0

    check_tensor()


def test_return_raw_str_bytes():
    assert tvm_ffi.convert(lambda: "hello")() == "hello"
    assert tvm_ffi.convert(lambda: b"hello")() == b"hello"
    assert tvm_ffi.convert(lambda: bytearray(b"hello"))() == b"hello"


def test_string_bytes_passing():
    fecho = tvm_ffi.get_global_func("testing.echo")
    use_count = tvm_ffi.get_global_func("testing.object_use_count")
    # small string
    assert fecho("hello") == "hello"
    # large string
    x = "hello" * 100
    y = fecho(x)
    assert y == x
    assert y.__tvm_ffi_object__ is not None
    use_count(y) == 1
    # small bytes
    assert fecho(b"hello") == b"hello"
    # large bytes
    x = b"hello" * 100
    y = fecho(x)
    assert y == x
    assert y.__tvm_ffi_object__ is not None
    fecho(y) == 1


def test_nested_container_passing():
    # test and make sure our ref counting is correct
    fecho = tvm_ffi.get_global_func("testing.echo")
    use_count = tvm_ffi.get_global_func("testing.object_use_count")
    obj = tvm_ffi.convert((1, 2, 3))
    assert use_count(obj) == 1
    y = fecho([obj, {"a": 1, "b": obj}])
    assert use_count(y) == 1
    assert use_count(obj) == 3
    assert use_count(y[1]) == 2


def test_pyfunc_convert():
    def add(a, b):
        return a + b

    fadd = tvm_ffi.convert(add)
    assert isinstance(fadd, tvm_ffi.Function)
    assert fadd(1, 2) == 3

    def fapply(f, *args):
        return f(*args)

    fapply = tvm_ffi.convert(fapply)
    assert fapply(add, 1, 3.3) == 4.3


def test_global_func():
    @tvm_ffi.register_global_func("mytest.echo")
    def echo(x):
        return x

    f = tvm_ffi.get_global_func("mytest.echo")
    assert f.same_as(echo)
    assert f(1) == 1

    assert "mytest.echo" in tvm_ffi.registry.list_global_func_names()

    tvm_ffi.registry.remove_global_func("mytest.echo")
    assert "mytest.echo" not in tvm_ffi.registry.list_global_func_names()
    assert tvm_ffi.get_global_func("mytest.echo", allow_missing=True) is None


def test_rvalue_ref():
    use_count = tvm_ffi.get_global_func("testing.object_use_count")

    def callback(x, expected_count):
        # The use count of TVM FFI objects is decremented as part of
        # `ObjectRef.__del__`, which runs when the Python object is
        # destructed.  However, Python object destruction is not
        # deterministic, and even CPython's reference-counting is
        # considered an implementation detail.  Therefore, to ensure
        # correct results from this test, `gc.collect()` must be
        # explicitly called.
        gc.collect()
        assert expected_count == use_count(x)
        return x._move()

    f = tvm_ffi.convert(callback)

    def check0():
        x = tvm_ffi.convert([1, 2])
        assert use_count(x) == 1
        f(x, 2)
        y = f(x._move(), 1)
        assert x.__ctypes_handle__().value == None

    def check1():
        x = tvm_ffi.convert([1, 2])
        assert use_count(x) == 1
        y = f(x, 2)
        z = f(x._move(), 2)
        assert x.__ctypes_handle__().value == None
        assert y.__ctypes_handle__().value is not None

    check0()
    check1()


def test_echo_with_opaque_object():
    class MyObject:
        def __init__(self, value):
            self.value = value

    fecho = tvm_ffi.get_global_func("testing.echo")
    x = MyObject("hello")
    assert sys.getrefcount(x) == 2
    y = fecho(x)
    assert isinstance(y, MyObject)
    assert y is x
    assert sys.getrefcount(x) == 3

    def py_callback(z):
        """python callback with opaque object"""
        assert z is x
        return z

    fcallback = tvm_ffi.convert(py_callback)
    z = fcallback(x)
    assert z is x
    assert sys.getrefcount(x) == 4
