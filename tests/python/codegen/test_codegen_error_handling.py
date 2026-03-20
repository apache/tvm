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
"""Runtime error message tests for MakePackedAPI + TVMFFIABIBuilder.

All tests compile TVMScript functions and verify the correct Python exception
type and exact error message at runtime.
"""

import re

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.script import tirx as T

# Parameterize over both LLVM and C backends
codegen_target = tvm.testing.parameter("llvm", "c")


# ── Argument count errors ────────────────────────────────────


def test_wrong_argument_count_error(codegen_target):
    """Wrong argument count produces TypeError with function signature."""

    @T.prim_func
    def func(a: T.handle, b: T.handle):
        n0 = T.int64()
        A = T.match_buffer(a, (n0,), "float32")
        B = T.match_buffer(b, (n0,), "float32")
        for i in range(n0):
            B[i] = A[i] + T.float32(1)

    lib = tvm.compile(func, target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    lib(a, b)  # correct input should pass

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Expected 2 arguments when calling:\n"
            "  `func(A: Tensor([n0], float32), B: Tensor([n0], float32))`"
        ),
    ):
        lib()


# ── Type mismatch errors (tensor parameters) ────────────────


def test_type_mismatch_non_tensor(codegen_target):
    """Passing a non-tensor where a tensor is expected raises TypeError."""

    @T.prim_func
    def func(a: T.handle, b: T.handle):
        n0 = T.int64()
        A = T.match_buffer(a, (n0,), "float32")
        B = T.match_buffer(b, (n0,), "float32")
        for i in range(n0):
            B[i] = A[i] + T.float32(1)

    lib = tvm.compile(func, target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    lib(a, b)  # correct input should pass

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched type on argument #1 when calling:\n"
            "  `func(A: Tensor([n0], float32), B: Tensor([n0], float32))`,\n"
            "  expected Tensor"
        ),
    ):
        lib(a, 1)


# ── Shape mismatch errors ───────────────────────────────────


def test_shape_mismatch_shared_variable(codegen_target):
    """b has different shape than a when they share symbolic variable n0."""

    @T.prim_func
    def func(a: T.handle, b: T.handle):
        n0 = T.int64()
        A = T.match_buffer(a, (n0,), "float32")
        B = T.match_buffer(b, (n0,), "float32")
        for i in range(n0):
            B[i] = A[i] + T.float32(1)

    lib = tvm.compile(func, target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    lib(a, b)  # correct input should pass

    b_short = tvm.runtime.tensor(np.zeros(126, dtype="float32"))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mismatched B.shape[0] on argument #1 when calling:\n"
            "  `func(A: Tensor([n0], float32), B: Tensor([n0], float32))`,\n"
            "  expected to match A.shape[0]"
        ),
    ):
        lib(a, b_short)


def test_invalid_shape_fixed(codegen_target):
    """Passing wrong shape for a fixed buffer dimension raises ValueError."""

    @T.prim_func
    def func(a: T.Buffer((128,), "float32"), b: T.Buffer((128,), "float32")):
        for i in range(128):
            b[i] = a[i] + T.float32(1)

    lib = tvm.compile(func, target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    lib(a, b)  # correct input should pass

    a_wrong = tvm.runtime.tensor(np.zeros(256, dtype="float32"))
    b_wrong = tvm.runtime.tensor(np.zeros(256, dtype="float32"))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid a.shape[0] on argument #0 when calling:\n"
            "  `func(a: Tensor([128], float32), b: Tensor([128], float32))`,\n"
            "  expected 128"
        ),
    ):
        lib(a_wrong, b_wrong)


# ── ndim mismatch errors ────────────────────────────────────


def test_ndim_mismatch_error(codegen_target):
    """ndim mismatch produces ValueError with function signature."""

    @T.prim_func
    def func(a: T.Buffer((4, 8), "float32"), b: T.Buffer((4, 8), "float32")):
        for i, j in T.grid(4, 8):
            b[i, j] = a[i, j]

    lib = tvm.compile(func, target=codegen_target)
    a_ok = tvm.runtime.tensor(np.zeros((4, 8), dtype="float32"))
    b_ok = tvm.runtime.tensor(np.zeros((4, 8), dtype="float32"))
    lib(a_ok, b_ok)  # correct input should pass

    a = tvm.runtime.tensor(np.zeros(4, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(4, dtype="float32"))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mismatched a.ndim on argument #0 when calling:\n"
            "  `func(a: Tensor([4, 8], float32), b: Tensor([4, 8], float32))`,\n"
            "  expected 2"
        ),
    ):
        lib(a, b)


# ── dtype mismatch errors ───────────────────────────────────


def test_dtype_mismatch_error(codegen_target):
    """dtype mismatch produces TypeError with function signature."""

    @T.prim_func
    def func(a: T.Buffer((8,), "float32"), b: T.Buffer((8,), "float32")):
        for i in range(8):
            b[i] = a[i]

    lib = tvm.compile(func, target=codegen_target)
    a_ok = tvm.runtime.tensor(np.zeros(8, dtype="float32"))
    b_ok = tvm.runtime.tensor(np.zeros(8, dtype="float32"))
    lib(a_ok, b_ok)  # correct input should pass

    a = tvm.runtime.tensor(np.zeros(8, dtype="int32"))
    b = tvm.runtime.tensor(np.zeros(8, dtype="float32"))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched a.dtype on argument #0 when calling:\n"
            "  `func(a: Tensor([8], float32), b: Tensor([8], float32))`,\n"
            "  expected float32"
        ),
    ):
        lib(a, b)


# ── Data alignment errors ──────────────────────────────────


@pytest.mark.skip(reason="alignment check disabled for now, revisit after merge")
def test_data_alignment_error(codegen_target):
    """Misaligned buffer data pointer raises ValueError."""

    @T.prim_func
    def func(a: T.Buffer((128,), "float32"), b: T.Buffer((128,), "float32")):
        for i in range(128):
            b[i] = a[i] + T.float32(1)

    lib = tvm.compile(func, target=codegen_target)
    a_ok = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    b_ok = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    lib(a_ok, b_ok)  # correct input should pass

    # Slice off first element of a 129-element array to create misaligned data pointer
    np_arr = np.zeros(129, dtype="float32")
    a_misaligned = tvm_ffi.from_dlpack(np_arr[1:])
    b = tvm.runtime.tensor(np.zeros(128, dtype="float32"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Misaligned Tensor data on argument #0 when calling:\n"
            "  `func(a: Tensor([128], float32), b: Tensor([128], float32))`,\n"
            "  expected data alignment=64 bytes"
        ),
    ):
        lib(a_misaligned, b)


# ── Compact strides mismatch errors ────────────────────────


def test_strides_mismatch_transposed(codegen_target):
    """Transposed (non-compact) strides raise ValueError."""

    @T.prim_func
    def func(a: T.Buffer((128, 128), "float32"), b: T.Buffer((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            b[i, j] = a[i, j] + T.float32(1)

    lib = tvm.compile(func, target=codegen_target)
    a_ok = tvm.runtime.tensor(np.zeros((128, 128), dtype="float32"))
    b_ok = tvm.runtime.tensor(np.zeros((128, 128), dtype="float32"))
    lib(a_ok, b_ok)  # correct input should pass

    # Use Fortran-order array to get non-compact (non-C-contiguous) strides
    np_arr = np.asfortranarray(np.zeros((128, 128), dtype="float32"))
    a_transposed = tvm_ffi.from_dlpack(np_arr)
    b = tvm.runtime.tensor(np.zeros((128, 128), dtype="float32"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mismatched a.strides on argument #0 when calling:\n"
            "  `func(a: Tensor([128, 128], float32), b: Tensor([128, 128], float32))`,\n"
            "  expected to be compact array"
        ),
    ):
        lib(a_transposed, b)


# ── Device mismatch errors ─────────────────────────────────


@tvm.testing.requires_cuda
def test_device_mismatch_error():
    """Passing GPU tensor to CPU function raises ValueError."""

    @T.prim_func
    def func(a: T.Buffer((128,), "float32"), b: T.Buffer((128,), "float32")):
        for i in range(128):
            b[i] = a[i] + T.float32(1)

    lib = tvm.compile(func, target="llvm")
    a_ok = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    b_ok = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    lib(a_ok, b_ok)  # correct input should pass

    a_gpu = tvm.runtime.tensor(np.zeros(128, dtype="float32"), device=tvm.cuda(0))
    b = tvm.runtime.tensor(np.zeros(128, dtype="float32"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mismatched a.device_type on argument #0 when calling:\n"
            "  `func(a: Tensor([128], float32), b: Tensor([128], float32))`,\n"
            "  expected cpu"
        ),
    ):
        lib(a_gpu, b)


# ── Scalar type mismatch errors ─────────────────────────────


def test_type_mismatch_int_parameter(codegen_target):
    """Passing a tensor where an int is expected raises TypeError."""

    @T.prim_func
    def func(x: T.int32) -> T.int32:
        if x > 0:
            return 10
        else:
            return 20

    lib = tvm.compile(func, target=codegen_target)
    assert lib(5) == 10  # correct input should pass

    a = tvm.runtime.tensor(np.zeros(8, dtype="float32"))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched type on argument #0 when calling:\n  `func(x: int32)`,\n  expected int"
        ),
    ):
        lib(a)


def test_type_mismatch_float_parameter(codegen_target):
    """Passing a tensor where a float is expected raises TypeError."""

    @T.prim_func
    def func(x: T.float32) -> T.int32:
        if x > T.float32(0):
            return 1
        else:
            return 0

    lib = tvm.compile(func, target=codegen_target)
    assert lib(1.0) == 1  # correct input should pass

    a = tvm.runtime.tensor(np.zeros(8, dtype="float32"))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched type on argument #0 when calling:\n  `func(x: float32)`,\n  expected float"
        ),
    ):
        lib(a)


def test_type_mismatch_bool_parameter(codegen_target):
    """Passing a tensor where a bool is expected raises TypeError."""

    @T.prim_func
    def func(x: T.bool) -> T.int32:
        if x:
            return 1
        else:
            return 0

    lib = tvm.compile(func, target=codegen_target)
    assert lib(True) == 1  # correct input should pass

    a = tvm.runtime.tensor(np.zeros(8, dtype="float32"))
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched type on argument #0 when calling:\n  `func(x: bool)`,\n  expected boolean"
        ),
    ):
        lib(a)


# ── Forward-reference symbolic variable ────────────────────


def test_forward_reference_symbolic_shape(codegen_target):
    """Buffers sharing a symbolic var with forward reference compile and run correctly.

    When buffer A has shape (batch_size+1,) and buffer B has shape (batch_size,),
    batch_size is referenced in A's shape assertion before it is defined from B.
    The three-sequence separation ensures this works. Also verifies the error
    message uses rendered access paths (e.g. "B.shape[0] + 1") for shape checks.
    """

    @T.prim_func
    def func(a: T.handle, b: T.handle):
        batch_size = T.int64()
        A = T.match_buffer(a, (batch_size + 1,), "int32")
        B = T.match_buffer(b, (batch_size,), "int32")
        for i in range(batch_size):
            B[i] = A[i] + A[i + 1]

    lib = tvm.compile(func, target=codegen_target)
    # Correct inputs: A has shape (5,), B has shape (4,)
    a = tvm.runtime.tensor(np.array([1, 2, 3, 4, 5], dtype="int32"))
    b = tvm.runtime.tensor(np.zeros(4, dtype="int32"))
    lib(a, b)
    np.testing.assert_array_equal(b.numpy(), [3, 5, 7, 9])

    # Wrong shape: A has shape (10,) but B has shape (4,), so batch_size=4 but A needs 5
    a_wrong = tvm.runtime.tensor(np.zeros(10, dtype="int32"))
    b_ok = tvm.runtime.tensor(np.zeros(4, dtype="int32"))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid A.shape[0] on argument #0 when calling:\n"
            "  `func(A: Tensor([batch_size + T.int64(1)], int32),"
            " B: Tensor([batch_size], int32))`,\n"
            "  expected B.shape[0] + 1"
        ),
    ):
        lib(a_wrong, b_ok)


# ── Mixed parameter type errors ────────────────────────────


def test_invalid_arguments_mixed_params(codegen_target):
    """Mixed bool + tensor function: type, dtype, and shape errors."""

    @T.prim_func
    def func(a0: T.bool, a1: T.Buffer([10], "float32")) -> T.int32:
        return 0

    lib = tvm.compile(func, target=codegen_target)
    lib(True, tvm.runtime.tensor(np.zeros(10, dtype="float32")))  # correct input should pass

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched type on argument #1 when calling:\n"
            "  `func(a0: bool, a1: Tensor([10], float32))`,\n"
            "  expected Tensor"
        ),
    ):
        lib(1, 1)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched a1.dtype on argument #1 when calling:\n"
            "  `func(a0: bool, a1: Tensor([10], float32))`,\n"
            "  expected float32"
        ),
    ):
        lib(1, tvm.runtime.empty([10], "int32"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid a1.shape[0] on argument #1 when calling:\n"
            "  `func(a0: bool, a1: Tensor([10], float32))`,\n"
            "  expected 10"
        ),
    ):
        lib(False, tvm.runtime.empty([11], "float32"))


if __name__ == "__main__":
    tvm.testing.main()
