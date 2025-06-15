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
import tvm.ffi as tvm_ffi


def test_array():
    a = tvm_ffi.convert([1, 2, 3])
    assert isinstance(a, tvm_ffi.Array)
    assert len(a) == 3
    assert a[-1] == 3
    a_slice = a[-3:-1]
    assert (a_slice[0], a_slice[1]) == (1, 2)


def test_bad_constructor_init_state():
    """Test when error is raised before __init_handle_by_constructor

    This case we need the FFI binding to gracefully handle both repr
    and dealloc by ensuring the chandle is initialized and there is
    proper repr code
    """
    with pytest.raises(TypeError):
        tvm_ffi.Array(1)

    with pytest.raises(AttributeError):
        tvm_ffi.Map(1)


def test_array_of_array_map():
    a = tvm_ffi.convert([[1, 2, 3], {"A": 5, "B": 6}])
    assert isinstance(a, tvm_ffi.Array)
    assert len(a) == 2
    assert isinstance(a[0], tvm_ffi.Array)
    assert isinstance(a[1], tvm_ffi.Map)
    assert tuple(a[0]) == (1, 2, 3)
    assert a[1]["A"] == 5
    assert a[1]["B"] == 6


def test_int_map():
    amap = tvm_ffi.convert({3: 2, 4: 3})
    assert 3 in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert 3 in dd
    assert 4 in dd
    assert 5 not in amap
    assert tuple(amap.items()) == ((3, 2), (4, 3))
    assert tuple(amap.keys()) == (3, 4)
    assert tuple(amap.values()) == (2, 3)


def test_str_map():
    data = []
    for i in reversed(range(10)):
        data.append((f"a{i}", i))
    amap = tvm_ffi.convert({k: v for k, v in data})
    assert tuple(amap.items()) == tuple(data)
    for k, v in data:
        assert k in amap
        assert amap[k] == v
        assert amap.get(k) == v

    assert tuple(k for k in amap) == tuple(k for k, _ in data)


def test_key_not_found():
    amap = tvm_ffi.convert({3: 2, 4: 3})
    with pytest.raises(KeyError):
        amap[5]


def test_repr():
    a = tvm_ffi.convert([1, 2, 3])
    assert str(a) == "[1, 2, 3]"
    amap = tvm_ffi.convert({3: 2, 4: 3})
    assert str(amap) == "{3: 2, 4: 3}"

    smap = tvm_ffi.convert({"a": 1, "b": 2})
    assert str(smap) == "{'a': 1, 'b': 2}"
