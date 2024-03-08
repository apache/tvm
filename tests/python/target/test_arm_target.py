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

import subprocess
import tempfile
import re

import pytest
import numpy as np

import tvm
from tvm.script import tir as T
from tvm.topi.arm_cpu.conv2d_int8 import is_int8_hw_support
from tvm.target import codegen

llvm_version, arm_target, input_dtype, kernel_dtype, is_supported = tvm.testing.parameters(
    # Testing mcpu type
    (8, "c -mcpu=cortex-m4", "int8", "int8", False),
    (8, "c -mcpu=cortex-m7", "int8", "int8", False),
    (8, "c -mcpu=cortex-m33", "int8", "int8", False),
    (8, "c -mcpu=cortex-m55", "int8", "int8", False),
    (8, "c -mcpu=cortex-m3", "int8", "int8", False),
    (7, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", False),
    (8, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", True),
    (9, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", True),
    (8, "llvm -mtriple=arm-linux-gnueabi", "int8", "int8", False),
    (7, "llvm -mtriple=aarch64-linux-gnu -mattr=+v8.4a,+dotprod", "int8", "int8", False),
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+v8.4a,+dotprod", "int8", "int8", True),
    (9, "llvm -mtriple=arm-linux-gnueabi -mattr=+neon", "int8", "int8", True),
    (8, "llvm -mtriple=aarch64-linux-gnu", "int8", "int8", True),
    # Testing dtype
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+neon", "int16", "int8", False),
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+neon", "int8", "int16", False),
    (8, "llvm -mtriple=aarch64-linux-gnu -mattr=+neon", "int16", "int16", False),
)


def test_arm_conv2d_int8_support(
    monkeypatch, llvm_version, arm_target, input_dtype, kernel_dtype, is_supported
):
    """Test ARM conv2d int8 support for different targets.

    Parameters
    ----------
    arm_target : str
        ARM CPU target.
    input_dtype : str
        Conv2d input data type.
    kernel_dtype : Session
        Conv2d kernel data type.
    is_supported : bool
        Expected result.
    """
    with tvm.target.Target(arm_target):
        monkeypatch.setattr(codegen, "llvm_version_major", lambda: llvm_version)
        assert is_int8_hw_support(input_dtype, kernel_dtype) == is_supported


@pytest.fixture(scope="session")
def sve_device_vector_length():
    c_code = r"""
    #include <stdio.h>
    #include <arm_sve.h>

    int main() {
        printf("%ld\n", svcntb() * 8);
    }
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        c_path = f"{tmp_dir}/vl.c"
        o_path = f"{tmp_dir}/out.o"
        with open(c_path, "w") as f:
            f.write(c_code)
        tvm.contrib.cc.create_executable(o_path, c_path, ["-march=native"])
        out = subprocess.check_output(o_path, shell=True).strip().decode()

    return int(out)


@tvm.testing.requires_aarch64_sve
def test_scalable_div(sve_device_vector_length):
    np.random.seed(0)
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(a: T.handle):
        A = T.match_buffer(a, (1,), "int32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[0] = T.Div(10000, 4 * T.vscale())

    mod = tvm.build(my_func, target=target)

    A_nd = tvm.nd.array(np.empty((1,), dtype="int32"), device=dev)
    mod(A_nd)

    ref = 10000 // (sve_device_vector_length // 32)
    tvm.testing.assert_allclose(A_nd.numpy()[0], ref)


@tvm.testing.requires_aarch64_sve
def test_scalable_buffer_load_store(sve_device_vector_length):
    np.random.seed(0)
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (num_elements,), "float32")
        B = T.match_buffer(b, (num_elements,), "float32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        B[T.ramp(0, 1, 4 * T.vscale())] = A[T.ramp(0, 1, 4 * T.vscale())]

    mod = tvm.build(my_func, target=target)

    A_np = np.random.uniform(size=(num_elements,)).astype("float32")
    B_np = np.zeros((num_elements,)).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)

    tvm.testing.assert_allclose(B_nd.numpy(), A_np)


@tvm.testing.requires_aarch64_sve
def test_scalable_loop_bound(sve_device_vector_length):
    np.random.seed(0)

    dtype = "float32"
    num_elements = sve_device_vector_length // 32
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (num_elements,), "float32")
        B = T.match_buffer(b, (num_elements,), "float32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        for i in T.serial(0, 4 * T.vscale()):
            B[i] = A[i]

    mod = tvm.build(my_func, target=target)

    A_np = np.random.uniform(size=(num_elements,)).astype(dtype)
    B_np = np.zeros((num_elements,)).astype(dtype)
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)

    tvm.testing.assert_allclose(B_nd.numpy(), A_np)


@tvm.testing.requires_aarch64_sve
def test_scalable_broadcast(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(a: T.handle):
        A = T.match_buffer(a, (num_elements,), "float32")
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[T.ramp(0, 1, 4 * T.vscale())] = T.broadcast(1, 4 * T.vscale())

    mod = tvm.build(my_func, target=target)

    A_np = np.zeros((num_elements,)).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    mod(A_nd)

    ref = np.ones((num_elements,))
    tvm.testing.assert_allclose(A_nd.numpy(), ref)
