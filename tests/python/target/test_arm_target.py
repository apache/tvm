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

import pytest
import numpy as np

import tvm
from tvm.topi.arm_cpu.conv2d_int8 import is_int8_hw_support
from tvm.target import codegen
from tvm.script import tir as T

import re


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
def test_scalable_vectorize(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    vscale = tvm.tir.vscale()

    @T.prim_func
    def main(A: T.Buffer((num_elements,), "float32"), B: T.Buffer((num_elements,), "float32")):
        for i_0 in range(T.ceildiv(num_elements, 4 * vscale)):
            for i_1 in T.vectorized(4 * vscale):
                A_1 = T.Buffer((num_elements,), data=A.data)
                B_1 = T.Buffer((num_elements,), data=B.data)
                B_1[i_0 * 4 * vscale + i_1] = A_1[i_0 * 4 * vscale + i_1]

    build_mod = tvm.build(main, target=target)

    llvm = build_mod.get_source()
    sve_vec_instrs = re.findall(r"\<vscale x 4 x float\>", llvm)
    assert len(sve_vec_instrs) > 0, "No scalable vectors in assembly"

    dev = tvm.cpu(0)
    np_zeros = np.zeros((num_elements,)).astype("float32")
    np_ones = np.ones((num_elements,)).astype("float32")

    input_buf = tvm.nd.array(np_ones, device=dev)
    output_buf = tvm.nd.array(np_zeros, device=dev)

    build_mod(input_buf, output_buf)
    tvm.testing.assert_allclose(output_buf.numpy(), np_ones)


@tvm.testing.requires_aarch64_sve
def test_scalable_div(sve_device_vector_length):
    np.random.seed(0)
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(A: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[0] = T.Div(10000, 4 * T.vscale())

    mod = tvm.build(my_func, target=target)

    A_nd = tvm.nd.array(np.empty((1,), dtype="int32"), device=dev)
    mod(A_nd)

    ref = 10000 // (sve_device_vector_length // 32)
    tvm.testing.assert_allclose(A_nd.numpy()[0], ref)


@tvm.testing.requires_aarch64_sve
def test_scalable_buffer_load_store(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(A: T.Buffer((num_elements,), "float32"), B: T.Buffer((num_elements,), "float32")):
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
    def my_func(A: T.Buffer((num_elements,), "float32"), B: T.Buffer((num_elements,), "float32")):
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
    def my_func(A: T.Buffer((num_elements,), "float32")):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[T.ramp(0, 1, 4 * T.vscale())] = T.broadcast(1, 4 * T.vscale())

    mod = tvm.build(my_func, target=target)

    A_np = np.zeros((num_elements,)).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    mod(A_nd)

    ref = np.ones((num_elements,))
    tvm.testing.assert_allclose(A_nd.numpy(), ref)


@tvm.testing.requires_aarch64_sve
def test_scalable_ptrue_predicate(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(A: T.Buffer((num_elements,), "float32")):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[T.ramp(0, 1, 4 * T.vscale())] = T.broadcast(T.IntImm("int1", 1), 4 * T.vscale())

    mod = tvm.build(my_func, target=target)

    A_np = np.zeros((num_elements,)).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    mod(A_nd)

    ref = np.ones((num_elements,))
    tvm.testing.assert_allclose(A_nd.numpy(), ref)


@pytest.mark.skip(reason="Currently don't support scalable gathers in codegen")
@tvm.testing.requires_aarch64_sve
def test_scalable_gather(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)
    stride = 2

    @T.prim_func
    def my_func(
        A: T.Buffer((stride * num_elements,), "float32"), B: T.Buffer((num_elements,), "float32")
    ):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        B[T.ramp(0, 1, 4 * T.vscale())] = A[T.ramp(0, stride, 4 * T.vscale())]

    mod = tvm.build(my_func, target=target)

    A_np = np.random.uniform(size=(stride * num_elements,)).astype("float32")
    B_np = np.zeros((num_elements,)).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)

    ref = A_np[::stride]
    tvm.testing.assert_allclose(B_nd.numpy(), ref)


@pytest.mark.skip(reason="Currently don't support scalable scatters in codegen")
@tvm.testing.requires_aarch64_sve
def test_scalable_scatter(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)
    stride = 2

    @T.prim_func
    def my_func(
        A: T.Buffer((num_elements,), "float32"), B: T.Buffer((stride * num_elements,), "float32")
    ):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        B[T.ramp(0, stride, 4 * T.vscale())] = A[T.ramp(0, 1, 4 * T.vscale())]

    mod = tvm.build(my_func, target=target)

    A_np = np.random.uniform(size=(num_elements,)).astype("float32")
    B_np = np.zeros((stride * num_elements,)).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)

    ref = B_np
    ref[::stride] = A_np
    tvm.testing.assert_allclose(B_nd.numpy(), ref)


@pytest.mark.skip(reason="Currently don't support scalable gathers in codegen")
@tvm.testing.requires_aarch64_sve
def test_scalable_complex_gather(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(A: T.Buffer((num_elements,), "float32"), B: T.Buffer((num_elements,), "float32")):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        B[T.ramp(0, 1, 4 * T.vscale())] = A[2 * T.ramp(0, 1, 4 * T.vscale()) % 4]

    mod = tvm.build(my_func, target=target)

    A_np = np.random.uniform(size=(num_elements,)).astype("float32")
    B_np = np.zeros((num_elements,)).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)

    tvm.testing.assert_allclose(B_nd.numpy(), A_np)


@pytest.mark.skip(reason="Currently don't support scalable ramps in codegen")
@tvm.testing.requires_aarch64_sve
def test_scalable_ramp(sve_device_vector_length):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    num_elements = sve_device_vector_length // 32
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(A: T.Buffer((num_elements,), "int32")):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        A[T.ramp(0, 1, 4 * T.vscale())] = T.ramp(11, 1, 4 * T.vscale())

    mod = tvm.build(my_func, target=target)

    A_np = np.zeros((num_elements,)).astype("int32")
    A_nd = tvm.nd.array(A_np, device=dev)
    mod(A_nd)

    ref = np.arange(11, 11 + num_elements)
    tvm.testing.assert_allclose(A_nd.numpy(), ref)


@tvm.testing.requires_aarch64_sve
@pytest.mark.parametrize("disable_predication", [True, False])
def test_schedule_split_vectorized(disable_predication):
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    dev = tvm.cpu(0)

    @T.prim_func
    def my_func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        T.func_attr({"global_symbol": "my_module", "tir.noalias": True})
        for i in T.serial(128):
            with T.block("A"):
                v_i = T.axis.remap("S", [i])
                B[v_i] = A[v_i] + 1.0

    sch = tvm.tir.Schedule(my_func)
    (a,) = sch.get_loops("A")

    with tvm.target.Target(target):
        _, a1 = sch.split(
            a,
            factors=[T.ceildiv(128, 4 * T.vscale()), 4 * T.vscale()],
            disable_predication=disable_predication,
        )

    sch.vectorize(a1)
    mod = tvm.build(sch.mod["main"], target=target)

    A_np = np.arange(128).astype("float32")
    A_nd = tvm.nd.array(A_np, device=dev)
    B_np = np.zeros(128).astype("float32")
    B_nd = tvm.nd.array(B_np, device=dev)
    mod(A_nd, B_nd)

    ref = A_np + 1.0
    tvm.testing.assert_allclose(B_nd.numpy(), ref)


def _test_accuracy(input_values, output_values, build_mod):
    dev = tvm.cpu(0)

    input_buf = tvm.nd.array(input_values, device=dev)

    np_zeros = np.zeros(output_values.shape).astype("float32")
    output_buf = tvm.nd.array(np_zeros, device=dev)

    build_mod(input_buf, output_buf)
    tvm.testing.assert_allclose(output_buf.numpy(), output_values)


@tvm.testing.skip_if_32bit(reason="Skipping test for i386 due to old version of LLVM")
def test_vectorize_to_sve():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    vscale = tvm.tir.vscale()
    buffer_size = 128

    @T.prim_func
    def main(A: T.Buffer((buffer_size,), "float32"), B: T.Buffer((buffer_size,), "float32")):
        for i_0 in range(tvm.tir.ceildiv(128, 4 * vscale)):
            for i_1 in T.vectorized(4 * vscale):
                A_1 = T.Buffer((128,), data=A.data)
                B_1 = T.Buffer((128,), data=B.data)
                B_1[i_0 * 4 * vscale + i_1] = A_1[i_0 * 4 * vscale + i_1]

    build_mod = tvm.build(main, target=target)

    llvm = build_mod.get_source()

    assert re.findall(r"\<vscale x 4 x float\>", llvm), "No scalable vectors in assembly"

    if tvm.testing.has_cpu_feat("sve"):
        print("running on an SVE enabled machine...")

        np_ones = np.ones((buffer_size,)).astype("float32")
        _test_accuracy(np_ones, np_ones, build_mod)


@tvm.testing.skip_if_32bit(reason="Skipping test for i386 due to old version of LLVM")
def test_vectorize_to_sve_with_broadcast():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    vscale = tvm.tir.vscale()
    buffer_size = 128

    @T.prim_func
    def main(A: T.Buffer((buffer_size,), "float32"), B: T.Buffer((buffer_size,), "float32")):
        for i_0 in range(tvm.tir.ceildiv(128, 4 * vscale)):
            for i_1 in T.vectorized(4 * vscale):
                A_1 = T.Buffer((128,), data=A.data)
                B_1 = T.Buffer((128,), data=B.data)
                B_1[i_0 * 4 * vscale + i_1] = A_1[i_0 * 4 * vscale + i_1] * 5

    build_mod = tvm.build(main, target=target)

    llvm = build_mod.get_source()

    assert re.findall(r"\<vscale x 4 x float\>", llvm), "No scalable vectors in assembly"

    if tvm.testing.has_cpu_feat("sve"):
        print("running on an SVE enabled machine...")

        np_ones = np.ones((buffer_size,)).astype("float32")
        output_values = np_ones * 5

        _test_accuracy(np_ones, output_values, build_mod)


@tvm.testing.skip_if_32bit(reason="Skipping test for i386 due to old version of LLVM")
def test_sve_full_stack():
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
    vscale = tvm.tir.vscale()
    buffer_size = 130

    @T.prim_func
    def main(A: T.Buffer((buffer_size,), "float32"), B: T.Buffer((buffer_size,), "float32")):
        for i in range(buffer_size):
            with T.block("A"):
                B[i] = A[i]

    # Schedule
    with tvm.target.Target(target):
        sch = tvm.tir.Schedule(main)
        (l,) = sch.get_loops("A")

        _, l1 = sch.split(l, factors=[T.ceildiv(buffer_size, 4 * vscale), 4 * vscale])

        sch.vectorize(l1)

    with tvm.transform.PassContext(config={"tir.enable_buffer_predication": True}):
        build_mod = tvm.build(sch.mod["main"], target=target)

    llvm = build_mod.get_source()

    assert re.findall(r"\<vscale x 4 x float\>", llvm), "No scalable vectors in llvm"
    assert re.findall(r"llvm.masked", llvm), "No masked instructions in llvm"

    if tvm.testing.has_cpu_feat("sve"):
        print("running on an SVE enabled machine...")

        np_ones = np.ones((buffer_size,)).astype("float32")
        _test_accuracy(np_ones, np_ones, build_mod)


if __name__ == "__main__":
    tvm.testing.main()
