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

# pylint: disable=no-member,invalid-name,unused-variable


########## Test cases for fuse_reduction_epilogue ##########


@T.prim_func
def matmul_bias_before(
    A: T.Buffer((16, 16), "int8"),
    B: T.Buffer((16, 16), "int8"),
    C: T.Buffer((16, 16), "int32"),
    D: T.Buffer((16, 16), "int32"),
) -> None:
    """Original function with separate reduction and epilogue blocks."""
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
    """Expected function after fusing epilogue into reduction init."""
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
    """Float32 version for additional coverage."""
    temp = T.alloc_buffer((32, 32), dtype="float32")
    for i, j, k in T.grid(32, 32, 32):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]
    for i, j in T.grid(32, 32):
        with T.block("bias"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj] + C[vi, vj]


@T.prim_func
def matmul_bias_fp32_expected(
    A: T.Buffer((32, 32), "float32"),
    B: T.Buffer((32, 32), "float32"),
    C: T.Buffer((32, 32), "float32"),
    D: T.Buffer((32, 32), "float32"),
) -> None:
    """Expected float32 version after fusion."""
    for i, j, k in T.grid(32, 32, 32):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(D[vi, vj])
            with T.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]


def test_fuse_reduction_epilogue_basic():
    """Test basic fusion of epilogue into reduction init."""
    sch = tir.Schedule(matmul_bias_before, debug_mask="all")
    sch.fuse_reduction_epilogue("multiply", "add")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_before)


def test_fuse_reduction_epilogue_fp32():
    """Test fusion with float32 data type."""
    sch = tir.Schedule(matmul_bias_fp32_before, debug_mask="all")
    sch.fuse_reduction_epilogue("matmul", "bias")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_fp32_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_fp32_before)


def test_fuse_reduction_epilogue_numerical_correctness():
    """Test that fusion preserves numerical correctness."""
    import numpy as np

    # Generate random test data
    np.random.seed(0)
    A_np = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)
    B_np = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)
    C_np = np.random.randint(-1000, 1000, size=(16, 16), dtype=np.int32)
    D_original = np.zeros((16, 16), dtype=np.int32)
    D_fused = np.zeros((16, 16), dtype=np.int32)

    # Run original version
    mod_original = tvm.build(matmul_bias_before, target="llvm")
    A_tvm = tvm.runtime.tensor(A_np)
    B_tvm = tvm.runtime.tensor(B_np)
    C_tvm = tvm.runtime.tensor(C_np)
    D_tvm_original = tvm.runtime.tensor(D_original)
    mod_original(A_tvm, B_tvm, C_tvm, D_tvm_original)

    # Run fused version
    sch = tir.Schedule(matmul_bias_before)
    sch.fuse_reduction_epilogue("multiply", "add")
    mod_fused = tvm.build(sch.mod["main"], target="llvm")
    D_tvm_fused = tvm.runtime.tensor(D_fused)
    mod_fused(A_tvm, B_tvm, C_tvm, D_tvm_fused)

    # Verify results match
    tvm.testing.assert_allclose(D_tvm_original.numpy(), D_tvm_fused.numpy())


if __name__ == "__main__":
    tvm.testing.main()
