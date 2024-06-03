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
from tvm.script import tir as T


@tvm.testing.parametrize_targets("c")
def test_buffer_store_predicate_not_supported(target):
    @T.prim_func
    def func(b: T.handle):
        B = T.match_buffer(b, (8,), "float32")
        B.vstore([T.Ramp(0, 2, 4)], T.Broadcast(1.0, 4), predicate=T.Broadcast(T.bool(True), 4))

    err_msg = "Predicated buffer store is not supported."
    with pytest.raises(tvm.TVMError, match=err_msg):
        with tvm.target.Target(target):
            tvm.build(func)


@tvm.testing.parametrize_targets("cuda", "opencl", "metal", "rocm", "vulkan -from_device=0")
def test_buffer_store_predicate_not_supported_gpu(target):
    @T.prim_func
    def func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (2, 3), "float32")
        B = T.match_buffer(b, (6,), "float32")
        T.func_attr({"global_symbol": "main"})
        for i_0 in T.thread_binding(3, thread="threadIdx.x"):
            B.vstore(
                [T.Ramp(i_0, 1, 4)], T.Broadcast(1.0, 4), predicate=T.Broadcast(T.bool(True), 4)
            )

    err_msg = "Predicated buffer store is not supported."
    with pytest.raises(tvm.TVMError, match=err_msg):
        with tvm.target.Target(target):
            tvm.build(func)


@tvm.testing.parametrize_targets("c")
def test_buffer_load_predicate_not_supported(target):
    @T.prim_func
    def func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (8,), "float32")
        B = T.match_buffer(b, (8,), "float32")
        for i_0 in range(4):
            B.vstore(
                [T.Ramp(0, 2, 4)],
                A.vload([T.Ramp(i_0, 1, 4)], predicate=T.Broadcast(T.bool(True), 4)),
            )

    err_msg = "Predicated buffer load is not supported."
    with pytest.raises(tvm.TVMError, match=err_msg):
        with tvm.target.Target(target):
            tvm.build(func)


@tvm.testing.parametrize_targets("cuda", "opencl", "metal", "rocm", "vulkan -from_device=0")
def test_buffer_load_predicate_not_supported_gpu(target):
    @T.prim_func
    def func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (8,), "float32")
        B = T.match_buffer(b, (8,), "float32")
        for i_0 in T.thread_binding(3, thread="threadIdx.x"):
            B.vstore(
                [T.Ramp(0, 2, 4)],
                A.vload([T.Ramp(i_0, 1, 4)], predicate=T.Broadcast(T.bool(True), 4)),
            )

    err_msg = "Predicated buffer load is not supported."
    with pytest.raises(tvm.TVMError, match=err_msg):
        with tvm.target.Target(target):
            tvm.build(func)


if __name__ == "__main__":
    tvm.testing.main()
