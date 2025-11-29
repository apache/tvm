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
import numpy as np
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
            tvm.compile(func)


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
            tvm.compile(func)


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
            tvm.compile(func)


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
            tvm.compile(func)


@tvm.testing.parametrize_targets("c", "llvm")
def test_codegen_loop_step(target):
    @T.prim_func
    def test_loop_step(
        A: T.Buffer((1024,), "float32"),
        B: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
    ):
        for i in T.serial(3, 1024, step=96):
            C[i] = A[i] + B[i]

    with tvm.transform.PassContext(disabled_pass=["tir.CanonicalizeLoop"]):
        lib = tvm.compile(test_loop_step, target=target)

    src = lib.mod.inspect_source()
    if target == "c":
        assert src.find("for (int32_t i = 3; i < 1024; i += 96)") >= 0

    dev = tvm.device(target, 0)
    a_np = np.random.rand(1024).astype("float32")
    b_np = np.random.rand(1024).astype("float32")
    c_np = np.zeros(1024, dtype="float32")
    a_tvm = tvm.runtime.tensor(a_np, dev)
    b_tvm = tvm.runtime.tensor(b_np, dev)
    c_tvm = tvm.runtime.tensor(c_np, dev)

    lib(a_tvm, b_tvm, c_tvm)

    c_result = c_tvm.numpy()

    # Check that the loop executes at positions 3, 99, 195, 291, 387, 483, 579, 675, 771, 867, 963
    for i in range(3, 1024, 96):
        tvm.testing.assert_allclose(c_result[i], a_np[i] + b_np[i], rtol=1e-5)

    # Assert non-touched positions remain zero
    for i in range(0, 3):
        assert c_result[i] == 0.0
    for i in range(4, 1024):
        if (i - 3) % 96 != 0:
            assert c_result[i] == 0.0


if __name__ == "__main__":
    tvm.testing.main()
