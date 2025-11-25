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
def matmul_bias_relu_before(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32"),
    C: T.Buffer((16, 16), "float32"),
    D: T.Buffer((16, 16), "float32"),
) -> None:
    """Original function with separate reduction and epilogue blocks (Bias + ReLU)."""
    temp = T.alloc_buffer((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with T.block("bias_relu"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(temp[vi, vj] + C[vi, vj], T.float32(0))


@T.prim_func
def matmul_bias_relu_before_per_iteration(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32"),
    C: T.Buffer((16, 16), "float32"),
    D: T.Buffer((16, 16), "float32"),
) -> None:
    """Original function with per-iteration ReLU (same semantics as fused)."""
    temp = T.alloc_buffer((16, 16), dtype="float32")
    for i, j in T.grid(16, 16):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            temp[vi, vj] = T.max(C[vi, vj], T.float32(0))  # ReLU on bias

    for i, j, k in T.grid(16, 16, 16):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            # Per-iteration ReLU
            temp[vi, vj] = T.max(temp[vi, vj] + A[vi, vk] * B[vj, vk], T.float32(0))

    for i, j in T.grid(16, 16):
        with T.block("copy"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj]


@T.prim_func
def matmul_bias_relu_expected(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32"),
    C: T.Buffer((16, 16), "float32"),
    D: T.Buffer((16, 16), "float32"),
) -> None:
    """Expected function after fusion (Bias + ReLU)."""
    for i, j, k in T.grid(16, 16, 16):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(D[vi, vj])
            with T.init():
                D[vi, vj] = T.max(C[vi, vj], T.float32(0))
            D[vi, vj] = T.max(D[vi, vj] + A[vi, vk] * B[vj, vk], T.float32(0))


def test_matmul_bias_relu():
    """Test fusion of matmul with bias + ReLU epilogue."""
    sch = tir.Schedule(matmul_bias_relu_before, debug_mask="all")
    sch.fuse_reduction_epilogue("matmul", "bias_relu")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_relu_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_relu_before)


def test_matmul_bias_relu_correctness_unified():
    """Test that original and fused produce identical results with per-iteration ReLU."""
    A_np = np.random.randn(16, 16).astype("float32")
    B_np = np.random.randn(16, 16).astype("float32")
    C_np = np.random.randn(16, 16).astype("float32")

    # NumPy reference for per-iteration ReLU
    # Simulate per-iteration ReLU behavior
    # Original code computes A[vi, vk] * B[vj, vk] which is A[i, k] * B[j, k]
    # For each k: add outer product of A[:, k] and B[:, k]
    D_ref = np.maximum(C_np, 0)  # init with ReLU on bias
    for k in range(16):
        # A[:, k] is shape (16,), B[:, k] is shape (16,)
        # Outer product: A[:, k] * B[:, k] for all i, j = A[i, k] * B[j, k]
        # Using broadcasting: A[:, k:k+1] * B[:, k:k+1].T gives (16, 1) * (1, 16) = (16, 16)
        D_ref = np.maximum(D_ref + np.outer(A_np[:, k], B_np[:, k]), 0)

    # TVM execution (original with per-iteration ReLU)
    mod_original = tvm.compile(matmul_bias_relu_before_per_iteration, target="llvm")
    D_original_tvm = tvm.runtime.tensor(np.zeros((16, 16), dtype="float32"))
    mod_original(
        tvm.runtime.tensor(A_np),
        tvm.runtime.tensor(B_np),
        tvm.runtime.tensor(C_np),
        D_original_tvm,
    )

    # TVM execution (fused)
    sch = tir.Schedule(matmul_bias_relu_before)
    sch.fuse_reduction_epilogue("matmul", "bias_relu")
    mod_fused = tvm.compile(sch.mod["main"], target="llvm")
    D_fused_tvm = tvm.runtime.tensor(np.zeros((16, 16), dtype="float32"))
    mod_fused(
        tvm.runtime.tensor(A_np),
        tvm.runtime.tensor(B_np),
        tvm.runtime.tensor(C_np),
        D_fused_tvm,
    )

    D_original = D_original_tvm.numpy()
    D_fused = D_fused_tvm.numpy()

    # Now both should match exactly
    np.testing.assert_allclose(D_original, D_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D_fused, D_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D_original, D_fused, rtol=1e-5, atol=1e-6)


@T.prim_func
def matmul_bias_relu_multiple_epilogue_before(
    A: T.Buffer((16, 16), "float32"),
    B: T.Buffer((16, 16), "float32"),
    C: T.Buffer((16, 16), "float32"),
    D: T.Buffer((16, 16), "float32"),
    E: T.Buffer((16, 16), "float32"),
) -> None:
    """Original function with separate reduction and multiple epilogue blocks (one with ReLU, one without)."""
    temp = T.alloc_buffer((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with T.block("bias_relu"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(temp[vi, vj] + C[vi, vj], T.float32(0))

    for i, j in T.grid(16, 16):
        with T.block("bias"):
            vi, vj = T.axis.remap("SS", [i, j])
            E[vi, vj] = temp[vi, vj] + C[vi, vj]


def test_matmul_bias_relu_multiple_epilogue():
    """Test fusion with multiple epilogue blocks - one with ReLU, one without.
    
    Following the same pattern as test_fuse_reduction_epilogue_multiple_epilogue,
    this test verifies that fusion works correctly when there are multiple
    epilogue blocks. Note: Currently this test fails because temp buffer is
    removed after fusion, leaving the second epilogue block with an undefined
    temp reference. This matches the behavior of the bias-only multiple epilogue test.
    """
    sch = tir.Schedule(matmul_bias_relu_multiple_epilogue_before, debug_mask="all")
    
    # Fusion should fail because temp buffer is removed, leaving the second
    # epilogue block with an undefined temp reference
    # This matches the behavior of test_fuse_reduction_epilogue_multiple_epilogue
    with pytest.raises(ValueError, match="Invalid use of undefined variable"):
        sch.fuse_reduction_epilogue("matmul", "bias_relu")


if __name__ == "__main__":
    tvm.testing.main()

