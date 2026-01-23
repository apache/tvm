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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)
import numpy as np

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul_bias_before(
    A: T.Buffer((16, 16), "int8"),
    B: T.Buffer((16, 16), "int8"),
    C: T.Buffer((16, 16), "int32"),
    D: T.Buffer((16, 16), "int32"),
) -> None:
    temp = T.alloc_buffer((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with T.block("multiply"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                temp[vi, vj] = T.int32(0)
            temp[vi, vj] = temp[vi, vj] + T.cast(A[vi, vk], "int32") * T.cast(B[vj, vk], "int32")
    for i, j in T.grid(16, 16):
        with T.block("add"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj] + C[vi, vj]


@T.prim_func
def matmul_bias_expected(
    A: T.Buffer((16, 16), "int8"),
    B: T.Buffer((16, 16), "int8"),
    C: T.Buffer((16, 16), "int32"),
    D: T.Buffer((16, 16), "int32"),
) -> None:
    temp = T.alloc_buffer((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with T.block("multiply"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(D[vi, vj])
            with T.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + T.cast(A[vi, vk], "int32") * T.cast(B[vj, vk], "int32")


@T.prim_func
def matmul_bias_fp32_before(
    A: T.Buffer((32, 32), "float32"),
    B: T.Buffer((32, 32), "float32"),
    C: T.Buffer((32, 32), "float32"),
    D: T.Buffer((32, 32), "float32"),
) -> None:
    temp = T.alloc_buffer((32, 32), dtype="float32")
    for i, j, k in T.grid(32, 32, 32):
        with T.block("multiply"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]
    for i, j in T.grid(32, 32):
        with T.block("add"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj] + C[vi, vj]


@T.prim_func
def matmul_bias_fp32_expected(
    A: T.Buffer((32, 32), "float32"),
    B: T.Buffer((32, 32), "float32"),
    C: T.Buffer((32, 32), "float32"),
    D: T.Buffer((32, 32), "float32"),
) -> None:
    temp = T.alloc_buffer((32, 32), dtype="float32")
    for i, j, k in T.grid(32, 32, 32):
        with T.block("multiply"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(D[vi, vj])
            with T.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_bias_multiple_epilogue_before(
    A: T.Buffer((16, 16), "int8"),
    B: T.Buffer((16, 16), "int8"),
    C: T.Buffer((16, 16), "int32"),
    D: T.Buffer((16, 16), "int32"),
    E: T.Buffer((16, 16), "int32"),
) -> None:
    temp = T.alloc_buffer((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with T.block("multiply"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                temp[vi, vj] = T.int32(0)
            temp[vi, vj] = temp[vi, vj] + T.cast(A[vi, vk], "int32") * T.cast(B[vj, vk], "int32")
    for i, j in T.grid(16, 16):
        with T.block("add"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj] + C[vi, vj]
    for i, j in T.grid(16, 16):
        with T.block("add2"):
            vi, vj = T.axis.remap("SS", [i, j])
            E[vi, vj] = temp[vi, vj] + C[vi, vj]


@T.prim_func
def matmul_bias_multiple_epilogue_expected(
    A: T.Buffer((16, 16), "int8"),
    B: T.Buffer((16, 16), "int8"),
    C: T.Buffer((16, 16), "int32"),
    D: T.Buffer((16, 16), "int32"),
    E: T.Buffer((16, 16), "int32"),
) -> None:
    temp = T.alloc_buffer((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with T.block("multiply"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(D[vi, vj])
            with T.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + T.cast(A[vi, vk], "int32") * T.cast(B[vj, vk], "int32")
    for i, j in T.grid(16, 16):
        with T.block("add2"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(temp[vi, vj], C[vi, vj])
            T.writes(E[vi, vj])
            E[vi, vj] = temp[vi, vj] + C[vi, vj]


def test_fuse_reduction_epilogue_basic():
    sch = tir.Schedule(matmul_bias_before, debug_mask="all")
    sch.fuse_reduction_epilogue("multiply", "add")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_before)


def test_fuse_reduction_epilogue_fp32():
    sch = tir.Schedule(matmul_bias_fp32_before, debug_mask="all")
    sch.fuse_reduction_epilogue("multiply", "add")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_fp32_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_fp32_before)


def test_fuse_reduction_epilogue_numerical_correctness():
    sch_original = tir.Schedule(matmul_bias_before, debug_mask="all")
    mod_original = tvm.compile(sch_original.mod["main"], target="llvm")

    sch_fused = tir.Schedule(matmul_bias_before, debug_mask="all")
    sch_fused.fuse_reduction_epilogue("multiply", "add")
    mod_fused = tvm.compile(sch_fused.mod["main"], target="llvm")

    A_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")
    B_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")
    C_np = np.random.randint(-1000, 1000, size=(16, 16), dtype="int32")

    expected = (A_np.astype("int32") @ B_np.T.astype("int32")) + C_np

    D_original_tvm = tvm.runtime.tensor(np.zeros((16, 16), dtype="int32"))
    D_fused_tvm = tvm.runtime.tensor(np.zeros((16, 16), dtype="int32"))

    mod_original(
        tvm.runtime.tensor(A_np), tvm.runtime.tensor(B_np), tvm.runtime.tensor(C_np), D_original_tvm
    )

    mod_fused(
        tvm.runtime.tensor(A_np), tvm.runtime.tensor(B_np), tvm.runtime.tensor(C_np), D_fused_tvm
    )

    D_original = D_original_tvm.numpy()
    D_fused = D_fused_tvm.numpy()

    tvm.testing.assert_allclose(D_original, expected, rtol=1e-5)
    tvm.testing.assert_allclose(D_fused, expected, rtol=1e-5)
    tvm.testing.assert_allclose(D_fused, D_original, rtol=1e-5)


def test_fuse_reduction_epilogue_multiple_epilogue():
    sch = tir.Schedule(matmul_bias_multiple_epilogue_before, debug_mask="all")
    sch.fuse_reduction_epilogue("multiply", "add")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], matmul_bias_multiple_epilogue_expected
    )
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_multiple_epilogue_before)

    mod = tvm.compile(sch.mod["main"], target="llvm")
    assert mod is not None


if __name__ == "__main__":
    tvm.testing.main()
