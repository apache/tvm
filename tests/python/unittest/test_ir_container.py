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
import tvm
from tvm import te
import numpy as np


def test_array():
    a = tvm.runtime.convert([1, 2, 3])
    assert len(a) == 3
    assert a[-1].value == 3
    a_slice = a[-3:-1]
    assert (a_slice[0].value, a_slice[1].value) == (1, 2)


def test_array_save_load_json():
    a = tvm.runtime.convert([1, 2, 3])
    json_str = tvm.ir.save_json(a)
    a_loaded = tvm.ir.load_json(json_str)
    assert a_loaded[1].value == 2


def test_dir_array():
    a = tvm.runtime.convert([1, 2, 3])
    dir(a)


def test_getattr_array():
    a = tvm.runtime.convert([1, 2, 3])
    type_key = getattr(a, "type_key")
    assert type_key == "Array"
    try:
        getattr(a, "test_key")
    except AttributeError:
        pass


def test_map():
    a = te.var("a")
    b = te.var("b")
    amap = tvm.runtime.convert({a: 2, b: 3})
    assert a in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert a in dd
    assert b in dd
    assert a + 1 not in amap
    assert {x for x in amap} == {a, b}
    assert set(amap.keys()) == {a, b}
    assert set(amap.values()) == {2, 3}


def test_str_map():
    amap = tvm.runtime.convert({"a": 2, "b": 3})
    assert "a" in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert amap["a"].value == 2
    assert "a" in dd
    assert "b" in dd


def test_map_save_load_json():
    a = te.var("a")
    b = te.var("b")
    amap = tvm.runtime.convert({a: 2, b: 3})
    json_str = tvm.ir.save_json(amap)
    amap = tvm.ir.load_json(json_str)
    assert len(amap) == 2
    dd = {kv[0].name: kv[1].value for kv in amap.items()}
    assert dd == {"a": 2, "b": 3}


def test_dir_map():
    a = te.var("a")
    b = te.var("b")
    amap = tvm.runtime.convert({a: 2, b: 3})
    dir(amap)


def test_getattr_map():
    a = te.var("a")
    b = te.var("b")
    amap = tvm.runtime.convert({a: 2, b: 3})
    type_key = getattr(amap, "type_key")
    assert type_key == "Map"
    try:
        getattr(amap, "test_key")
    except AttributeError:
        pass


def test_in_container():
    arr = tvm.runtime.convert(["a", "b", "c"])
    assert "a" in arr
    assert tvm.tir.StringImm("a") in arr
    assert "d" not in arr


def test_ndarray_container():
    x = tvm.nd.array([1, 2, 3])
    arr = tvm.runtime.convert([x, x])
    assert arr[0].same_as(x)
    assert arr[1].same_as(x)
    assert isinstance(arr[0], tvm.nd.NDArray)


if __name__ == "__main__":
    test_str_map()
    test_array()
    test_map()
    test_array_save_load_json()
    test_map_save_load_json()
    test_dir_array()
    test_dir_map()
    test_getattr_array()
    test_getattr_map()
    test_in_container()
    test_ndarray_container()
    
