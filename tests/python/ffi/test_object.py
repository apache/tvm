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
import pytest

from tvm import ffi as tvm_ffi


def test_make_object():
    # with default values
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase")
    assert obj0.v_i64 == 10
    assert obj0.v_f64 == 10.0
    assert obj0.v_str == "hello"


def test_method():
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=12)
    assert obj0.add_i64(1) == 13
    assert type(obj0).add_i64.__doc__ == "add_i64 method"
    assert type(obj0).v_i64.__doc__ == "i64 field"


def test_setter():
    # test setter
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=10, v_str="hello")
    assert obj0.v_i64 == 10
    obj0.v_i64 = 11
    assert obj0.v_i64 == 11
    obj0.v_str = "world"
    assert obj0.v_str == "world"

    with pytest.raises(TypeError):
        obj0.v_str = 1

    with pytest.raises(TypeError):
        obj0.v_i64 = "hello"


def test_derived_object():
    with pytest.raises(TypeError):
        obj0 = tvm_ffi.testing.create_object("testing.TestObjectDerived")

    v_map = tvm_ffi.convert({"a": 1})
    v_array = tvm_ffi.convert([1, 2, 3])

    obj0 = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived", v_i64=20, v_map=v_map, v_array=v_array
    )
    assert obj0.v_map.same_as(v_map)
    assert obj0.v_array.same_as(v_array)
    assert obj0.v_i64 == 20
    assert obj0.v_f64 == 10.0
    assert obj0.v_str == "hello"

    obj0.v_i64 = 21
    assert obj0.v_i64 == 21
