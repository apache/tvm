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

import ctypes
from tvm import ffi as tvm_ffi


def test_echo():
    fecho = tvm_ffi.get_global_func("testing.echo")
    assert isinstance(fecho, tvm_ffi.Function)
    # test each type
    assert fecho(None) is None

    # test bool
    assert fecho(True) is True
    assert fecho(False) is False

    # test int float
    assert fecho(1) == 1
    assert fecho(1.2) == 1.2

    # test str
    str_result = fecho("hello")
    assert isinstance(str_result, str)
    assert str_result == "hello"
    assert isinstance(str_result, tvm_ffi.String)

    # test bytes
    bytes_result = fecho(b"abc")
    assert isinstance(bytes_result, bytes)
    assert bytes_result == b"abc"
    assert isinstance(bytes_result, tvm_ffi.Bytes)
    # test dtype
    dtype_result = fecho(tvm_ffi.dtype("float32"))
    assert isinstance(dtype_result, tvm_ffi.dtype)
    assert dtype_result == tvm_ffi.dtype("float32")

    # test c_void_p
    c_void_p_result = fecho(ctypes.c_void_p(0x12345678))
    assert isinstance(c_void_p_result, ctypes.c_void_p)
    assert c_void_p_result.value == 0x12345678


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
