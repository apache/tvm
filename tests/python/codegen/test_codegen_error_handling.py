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

import tvm
import tvm.testing
from tvm.script import tir as T

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


# ── Null data pointer errors ────────────────────────────────


def test_null_data_pointer(codegen_target):
    """Null data pointer in tensor raises TypeError."""

    @T.prim_func
    def func(a: T.Buffer((16, 16), "int32"), b: T.Buffer((16, 16), "int32")):
        for i, j in T.grid(16, 16):
            b[i, j] = a[i, j]

    lib = tvm.compile(func, target=codegen_target)
    a = tvm.runtime.tensor(np.zeros([16], dtype="int32"))
    b = tvm.runtime.empty([16, 16], "int32", tvm.cpu())

    with pytest.raises(tvm.TVMError):
        lib(a, b)


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
    a = tvm.runtime.tensor(np.zeros(8, dtype="float32"))

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched type on argument #0 when calling:\n  `func(x: bool)`,\n  expected boolean"
        ),
    ):
        lib(a)


if __name__ == "__main__":
    tvm.testing.main()
