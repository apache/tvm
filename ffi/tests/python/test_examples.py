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
# testcases appearing in example docstrings
import tvm_ffi


def test_register_global_func():
    # we can use decorator to register a function
    @tvm_ffi.register_global_func("example.echo")
    def echo(x):
        return x

    # After registering, we can get the function by its name
    f = tvm_ffi.get_global_func("example.echo")
    assert f(1) == 1
    # we can also directly register a function
    tvm_ffi.register_global_func("example.add_one", lambda x: x + 1)
    f = tvm_ffi.get_global_func("example.add_one")
    assert f(1) == 2


def test_array():
    a = tvm_ffi.convert([1, 2, 3])
    assert isinstance(a, tvm_ffi.Array)
    assert len(a) == 3


def test_map():
    amap = tvm_ffi.convert({"a": 1, "b": 2})
    assert isinstance(amap, tvm_ffi.Map)
    assert len(amap) == 2
    assert amap["a"] == 1
    assert amap["b"] == 2
