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

"""
Test parallelizing HVX workloads and compare them to single thread examples.
"""
import numpy as np
from numpy.random import default_rng

import tvm
from tvm.script import tir as T

from .infrastructure import get_hexagon_target

TEST_OUTPUT_TEMPLATE = "Test {} with {} operations... \n    -Single Thread: {} ms \n    -Parallel: {} ms\n    -Speedup: {}x\n"


def get_vrmpy_shape_dtypes(operations):
    return ((operations, 128), "uint8", (operations, 128), "uint8", (operations, 32), "int32")


def get_vmpy_vadd_shape_dtype(operations):
    return ((operations, 128), "uint8", (operations, 128), "uint8", (operations, 128), "int16")


def vmpy_expected_producer(shape, a, b):
    expected = np.zeros(shape, dtype="int16")
    for n in range(shape[0]):
        for i in range(0, 128, 2):
            expected[n, i // 2] = np.int16(a[n, i]) * np.int16(b[n, i])
        for i in range(1, 128, 2):
            expected[n, i // 2 + 64] = np.int16(a[n, i]) * np.int16(b[n, i])
    return expected


def vadd_expected_producer(shape, a, b):
    expected = np.zeros(shape, dtype="int16")
    for n in range(shape[0]):
        for i in range(0, 128, 2):
            expected[n, i // 2] = np.int16(a[n, i]) + np.int16(b[n, i])
        for i in range(1, 128, 2):
            expected[n, i // 2 + 64] = np.int16(a[n, i]) + np.int16(b[n, i])
    return expected


def vrmpy_expected_producer(shape, a, b):
    expected = np.zeros(shape, dtype="int32")
    for n in range(shape[0]):
        for i in range(32):
            for r in range(4):
                expected[n, i] = expected[n, i] + np.uint32(a[n, i * 4 + r]) * np.uint32(
                    b[n, i * 4 + r]
                )
    return expected


def get_vmpy_operator(operations):
    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [operations, 128], dtype="uint8")
        B = T.match_buffer(b, [operations, 128], dtype="uint8")
        C = T.match_buffer(c, [operations, 128], dtype="int16")
        for n in T.grid(operations):
            with T.block("C"):
                vn = T.axis.remap("S", [n])
                C[vn, T.ramp(0, 1, 128)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vmpybusv.128B"),
                    T.uint32(2),
                    T.reinterpret(A[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    T.reinterpret(B[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    dtype="int16x128",
                )

    return operator


def get_vadd_operator(operations):
    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [operations, 128], dtype="uint8")
        B = T.match_buffer(b, [operations, 128], dtype="uint8")
        C = T.match_buffer(c, [operations, 128], dtype="int16")
        for n in T.grid(operations):
            with T.block("C"):
                vn = T.axis.remap("S", [n])
                C[vn, T.ramp(0, 1, 128)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vaddubh.128B"),
                    T.uint32(2),
                    T.reinterpret(A[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    T.reinterpret(B[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    dtype="int16x128",
                )

    return operator


def get_vrmpy_operator(operations):
    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [operations, 128], dtype="uint8")
        B = T.match_buffer(b, [operations, 128], dtype="uint8")
        C = T.match_buffer(c, [operations, 32], dtype="int32")
        for n in T.grid(operations):
            with T.block("C"):
                vn = T.axis.remap("S", [n])
                C[vn, T.ramp(0, 1, 32)] = T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyubv.128B"),
                    T.uint32(2),
                    T.reinterpret(A[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    T.reinterpret(B[vn, T.ramp(0, 1, 128)], dtype="int32x32"),
                    dtype="int32x32",
                )

    return operator


def evaluate(hexagon_session, shape_dtypes, expected_output_producer, sch):
    a_shape, a_dtype, b_shape, b_dtype, c_shape, c_dtype = shape_dtypes

    func_tir = tvm.build(sch.mod["main"], target=get_hexagon_target("v68"))
    module = hexagon_session.load_module(func_tir)

    rng = default_rng()
    a = rng.integers(0, 16, a_shape, dtype=a_dtype)
    b = rng.integers(0, 16, b_shape, dtype=b_dtype)
    c = np.zeros(c_shape, dtype=c_dtype)

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device)
    b_hexagon = tvm.runtime.ndarray.array(b, device=hexagon_session.device)
    c_hexagon = tvm.runtime.ndarray.array(c, device=hexagon_session.device)

    # These are reduced for CI but number=100 and repeat=10 does a good job of removing noise.
    number = 1
    repeat = 1

    timer = module.time_evaluator(
        "__tvm_main__", hexagon_session.device, number=number, repeat=repeat
    )
    runtime = timer(a_hexagon, b_hexagon, c_hexagon)
    tvm.testing.assert_allclose(c_hexagon.asnumpy(), expected_output_producer(c_shape, a, b))

    return round(runtime.mean * 1000, 6)


class TestMatMulVec:

    (
        operation_name,
        operator_producer,
        shape_dtypes_producer,
        expected_output_producer,
    ) = tvm.testing.parameters(
        ("vrmpy", get_vrmpy_operator, get_vrmpy_shape_dtypes, vrmpy_expected_producer),
        ("vmpy", get_vmpy_operator, get_vmpy_vadd_shape_dtype, vmpy_expected_producer),
        ("vadd", get_vadd_operator, get_vmpy_vadd_shape_dtype, vadd_expected_producer),
    )

    # Experimentally best split factor but all multiples of 4 perform pretty well.
    # This is because there are 4 HVX untis available on the device and pipelining
    # works best with parallels of the number of available HVX.
    split_factor = tvm.testing.parameter(4)

    # Removed most of these to speedup CI.
    operation_count = tvm.testing.parameter(
        128,
        # 256,
        # 512,
        # 1024,  # Single thread runs faster since L2 cache can handle the entire request quickly
        # 2048,
        # 4096,  # Significant performance degredation once the inputs and outputs cannot all fit in L2
        # 8192,
        # 16384,
    )

    @tvm.testing.requires_hexagon
    def test(
        self,
        hexagon_session,
        operation_count,
        operation_name,
        operator_producer,
        shape_dtypes_producer,
        expected_output_producer,
        split_factor,
    ):

        sch = tvm.tir.Schedule(operator_producer(operation_count))
        single_thread_runtime = evaluate(
            hexagon_session, shape_dtypes_producer(operation_count), expected_output_producer, sch
        )

        sch = tvm.tir.Schedule(operator_producer(operation_count))
        block = sch.get_block("C")
        b = sch.get_loops(block)
        bo, _ = sch.split(b[0], factors=[split_factor, None])
        sch.parallel(bo)

        parallel_runtime = evaluate(
            hexagon_session, shape_dtypes_producer(operation_count), expected_output_producer, sch
        )

        speedup = round(single_thread_runtime / parallel_runtime, 2)

        print(
            TEST_OUTPUT_TEMPLATE.format(
                operation_name, operation_count, single_thread_runtime, parallel_runtime, speedup
            )
        )


if __name__ == "__main__":
    tvm.testing.main()
