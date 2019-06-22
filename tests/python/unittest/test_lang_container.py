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

def test_array():
    a = tvm.convert([1,2,3])
    assert len(a) == 3
    assert a[-1].value == 3
    a_slice = a[-3:-1]
    assert (a_slice[0].value, a_slice[1].value) == (1, 2)

def test_array_save_load_json():
    a = tvm.convert([1,2,3])
    json_str = tvm.save_json(a)
    a_loaded = tvm.load_json(json_str)
    assert(a[1].value == 2)


def test_map():
    a = tvm.var('a')
    b = tvm.var('b')
    amap = tvm.convert({a: 2,
                        b: 3})
    assert a in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert a in dd
    assert b in dd
    assert a + 1 not in amap


def test_str_map():
    amap = tvm.convert({'a': 2, 'b': 3})
    assert 'a' in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert amap['a'].value == 2
    assert 'a' in dd
    assert 'b' in dd


def test_map_save_load_json():
    a = tvm.var('a')
    b = tvm.var('b')
    amap = tvm.convert({a: 2,
                        b: 3})
    json_str = tvm.save_json(amap)
    amap = tvm.load_json(json_str)
    assert len(amap) == 2
    dd = {kv[0].name : kv[1].value for kv in amap.items()}
    assert(dd == {"a": 2, "b": 3})


def test_in_container():
    arr = tvm.convert(['a', 'b', 'c'])
    assert 'a' in arr
    assert tvm.make.StringImm('a') in arr
    assert 'd' not in arr

if __name__ == "__main__":
    test_str_map()
    test_array()
    test_map()
    test_array_save_load_json()
    test_map_save_load_json()
    test_in_container()
