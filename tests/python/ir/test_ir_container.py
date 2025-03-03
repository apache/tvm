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
import tvm
from tvm import te
import numpy as np


def test_array():
    a = tvm.runtime.convert([1, 2, 3])
    assert len(a) == 3
    assert a[-1] == 3
    a_slice = a[-3:-1]
    assert (a_slice[0], a_slice[1]) == (1, 2)


def test_array_save_load_json():
    a = tvm.runtime.convert([1, 2, 3.5, True])
    json_str = tvm.ir.save_json(a)
    a_loaded = tvm.ir.load_json(json_str)
    assert a_loaded[1] == 2
    assert a_loaded[2] == 3.5
    assert a_loaded[3] == True
    assert isinstance(a_loaded[3], bool)


def test_dir_array():
    a = tvm.runtime.convert([1, 2, 3])
    assert dir(a)


def test_getattr_array():
    a = tvm.runtime.convert([1, 2, 3])
    assert getattr(a, "type_key") == "Array"
    assert not hasattr(a, "test_key")


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
    assert amap["a"] == 2
    assert "a" in dd
    assert "b" in dd


def test_map_save_load_json():
    a = te.var("a")
    b = te.var("b")
    amap = tvm.runtime.convert({a: 2, b: 3})
    json_str = tvm.ir.save_json(amap)
    amap = tvm.ir.load_json(json_str)
    assert len(amap) == 2
    dd = {kv[0].name: kv[1] for kv in amap.items()}
    assert dd == {"a": 2, "b": 3}


def test_dir_map():
    a = te.var("a")
    b = te.var("b")
    amap = tvm.runtime.convert({a: 2, b: 3})
    assert dir(amap)


def test_getattr_map():
    a = te.var("a")
    b = te.var("b")
    amap = tvm.runtime.convert({a: 2, b: 3})
    assert getattr(amap, "type_key") == "Map"
    assert not hasattr(amap, "test_key")


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


def test_return_variant_type():
    func = tvm.get_global_func("testing.ReturnsVariant")
    res_even = func(42)
    assert isinstance(res_even, tvm.tir.IntImm)
    assert res_even == 21

    res_odd = func(17)
    assert isinstance(res_odd, tvm.runtime.String)
    assert res_odd == "argument was odd"


def test_pass_variant_type():
    func = tvm.get_global_func("testing.AcceptsVariant")

    assert func("string arg") == "runtime.String"
    assert func(17) == "IntImm"


def test_pass_incorrect_variant_type():
    func = tvm.get_global_func("testing.AcceptsVariant")
    float_arg = tvm.tir.FloatImm("float32", 0.5)

    with pytest.raises(Exception):
        func(float_arg)


if __name__ == "__main__":
    tvm.testing.main()
