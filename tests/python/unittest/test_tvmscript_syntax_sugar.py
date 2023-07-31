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
# pylint: disable=missing-function-docstring,missing-module-docstring,invalid-name,pointless-string-statement
import sys
from typing import Any

import pytest
import tvm.testing
from tvm.script import from_source
from tvm.script import tir as T
from tvm.tir.schedule.testing import assert_structural_equal_ignore_global_symbol


@T.prim_func
def transformed_matmul_no_syntax_sugar(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i0, i1])
            vk = T.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            T.reads([C[vi, vj], A[vi, vk], B[vj, vk]])
            T.writes([C[vi, vj], A[vi, vk]])
            with T.init():
                C[vi, vj] = 0.0
            A[vi, vk] = A[vi, vk] + B[vj, vk]
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def transformed_matmul_syntax_sugar(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i0, i1])
            vk = T.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj], A[vi, vk])
            with T.init():
                C[vi, vj] = 0.0
            A[vi, vk] = A[vi, vk] + B[vj, vk]
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


def test_reads_writes_syntax_sugar():
    assert_structural_equal_ignore_global_symbol(
        transformed_matmul_no_syntax_sugar, transformed_matmul_syntax_sugar
    )


@T.prim_func
def loop_no_syntax_sugar(a: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    for i in T.serial(0, 128):
        for j in T.parallel(0, 128):
            for k in T.vectorized(0, 128):
                for x in T.unroll(0, 128):
                    for y in T.thread_binding(0, 128, thread="threadIdx.x"):
                        for z in T.thread_binding(0, 128, thread="threadIdx.x"):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


@T.prim_func
def loop_syntax_sugar(a: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    for i in T.serial(128):
        for j in T.parallel(128):
            for k in T.vectorized(128):
                for x in T.unroll(128):
                    for y in T.thread_binding(128, "threadIdx.x"):
                        for z in T.thread_binding(128, thread="threadIdx.x"):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


def test_loop_syntax_sugar():
    assert_structural_equal_ignore_global_symbol(loop_no_syntax_sugar, loop_syntax_sugar)


# match buffer - use kwargs
@T.prim_func
def elementwise_handle(
    a: T.handle,
    b: T.handle,
) -> None:
    A = T.match_buffer(a, (128, 128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128, 128))
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


# match buffer - use buffer with kwargs
@T.prim_func
def elementwise_buffer_kwargs(
    a: T.Buffer(shape=(128, 128, 128, 128), dtype="float32"),
    b: T.Buffer(shape=(128, 128, 128, 128), dtype="float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


# match buffer - use buffer without kwargs
@T.prim_func
def elementwise_buffer_no_kwargs(
    a: T.Buffer((128, 128, 128, 128), "float32"),
    b: T.Buffer((128, 128, 128, 128), "float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with T.block("B"):
            vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


def test_match_buffer_syntax_sugar():
    # with kwargs
    assert_structural_equal_ignore_global_symbol(elementwise_handle, elementwise_buffer_kwargs)
    # without kwargs
    assert_structural_equal_ignore_global_symbol(elementwise_handle, elementwise_buffer_no_kwargs)


def test_match_buffer_1d():
    @T.prim_func
    def func_no_sugar(a: T.handle):
        A = T.match_buffer(a, shape=(16,))
        for i in T.serial(16):
            A[i] = 0.0

    @T.prim_func
    def func_with_sugar(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    assert_structural_equal_ignore_global_symbol(func_no_sugar, func_with_sugar)


# dynamic shape gemm
@T.prim_func
def gemm_dyn_shape(a: T.handle, b: T.handle, c: T.handle):
    N = T.int32()
    M = T.int32()
    K = T.int32()
    A = T.match_buffer(a, (N, K), "float32")
    B = T.match_buffer(b, (K, M), "float32")
    C = T.match_buffer(c, (N, M), "float32")
    for i, j, k in T.grid(N, M, K):
        with T.block("gemm"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_dynamic_shape_gemm():
    gemm_dyn_shape_roundtrip = from_source(gemm_dyn_shape.script())
    assert_structural_equal_ignore_global_symbol(gemm_dyn_shape, gemm_dyn_shape_roundtrip)


@T.prim_func
def match_buffer_int64(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (T.int64(128), T.int64(128)), dtype="float32")
    B = T.alloc_buffer((T.int64(128), T.int64(128)), dtype="float32")
    C = T.match_buffer(c, (T.int64(128), T.int64(128)), dtype="float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def match_buffer_int64_after_roundtrip(
    A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    C: T.Buffer((T.int64(128), T.int64(128)), "float32"),
) -> None:
    B = T.alloc_buffer((T.int64(128), T.int64(128)), dtype="float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


def test_match_buffer_int64():
    original = match_buffer_int64
    after_roundtrip = match_buffer_int64_after_roundtrip
    assert_structural_equal_ignore_global_symbol(original, after_roundtrip, True)


def test_match_buffer_region_has_implicit_shape_dtype():
    @T.prim_func
    def explicit_shape_dtype(A: T.Buffer((16, 64), "int32")):
        with T.block():
            B = T.match_buffer(A[8:16, 32:64], shape=(8, 32), dtype="int32")
            T.evaluate(0)

    @T.prim_func
    def implicit_shape_dtype(A: T.Buffer((16, 64), "int32")):
        with T.block():
            B = T.match_buffer(A[8:16, 32:64])
            T.evaluate(0)

    assert_structural_equal_ignore_global_symbol(explicit_shape_dtype, implicit_shape_dtype)


def test_match_buffer_input_requires_shape_arg():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def func(a: T.handle):
            A = T.match_buffer(a, dtype="int32")
            T.evaluate(0)


def test_letstmt_bufferload_without_type_annotation():
    # Variable assignment of PrimExpr types uses the dtype of the
    # PrimExpr to determine the variable's dtype.  Parsing of
    # buf[indices] is done by generating a BufferSlice object, which
    # handles both store and load cases.  BufferSlice is not a
    # PrimExpr, and implements BufferSlice.dtype explicitly.

    # Failure occurred during parsing of the tvmscript.
    @T.prim_func
    def func_without_type_annotation(A: T.Buffer((1,), "int32")):
        x = A[0]
        T.evaluate(x)


def test_letstmt_bind_with_constant():
    @T.prim_func
    def constant_binds():
        x = T.meta_var(1)
        y = T.meta_var(42.0)
        T.evaluate(T.cast(x, "float32") + y)

    @T.prim_func
    def constant_binds_wrapped():
        x = T.meta_var(T.int32(1))
        y = T.meta_var(T.float32(42.0))
        T.evaluate(T.cast(x, "float32") + y)

    assert_structural_equal_ignore_global_symbol(constant_binds, constant_binds_wrapped)


def test_func_call():
    def shared_16x16_to_ldmatrix_32x8_layout(i, j):
        thread_id = (i % 8) * 4 + (j % 8) // 2
        return T.meta_var((thread_id, (j // 8) * 4 + (i // 8) * 2 + (j % 2)))

    @T.prim_func
    def mma_sync_m16n16k16_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (32, 8), "float16", align=64, offset_factor=16, scope="warp")
        B = T.match_buffer(b, (32, 8), "float16", align=64, offset_factor=16, scope="warp")
        C = T.match_buffer(c, (32, 8), "float16", align=64, offset_factor=16, scope="warp")

        with T.block("root"):
            T.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
            T.writes(C[0:32, 0:8])
            for i, j, k in T.grid(16, 16, 16):
                with T.block("C"):
                    i, j, k = T.axis.remap("SSR", [i, j, k])
                    thread_id_C, local_id_C = shared_16x16_to_ldmatrix_32x8_layout(i, j)
                    thread_id_A, local_id_A = shared_16x16_to_ldmatrix_32x8_layout(i, k)
                    thread_id_B, local_id_B = shared_16x16_to_ldmatrix_32x8_layout(k, j)

                    T.reads(
                        C[thread_id_C, local_id_C],
                        A[thread_id_A, local_id_A],
                        B[thread_id_B, local_id_B],
                    )
                    T.writes(C[thread_id_C, local_id_C])

                    C[thread_id_C, local_id_C] += (
                        A[thread_id_A, local_id_A] * B[thread_id_B, local_id_B]
                    )

    @T.prim_func
    def mma_sync_m16n16k16_desc_manual(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (32, 8), "float16", align=64, offset_factor=16, scope="warp")
        B = T.match_buffer(b, (32, 8), "float16", align=64, offset_factor=16, scope="warp")
        C = T.match_buffer(c, (32, 8), "float16", align=64, offset_factor=16, scope="warp")

        with T.block("root"):
            T.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
            T.writes(C[0:32, 0:8])
            for i, j, k in T.grid(16, 16, 16):
                with T.block("C"):
                    i, j, k = T.axis.remap("SSR", [i, j, k])
                    T.reads(
                        C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2],
                        A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2],
                        B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2],
                    )
                    T.writes(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2])
                    C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] = (
                        C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2]
                        + A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2]
                        * B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2]
                    )

    assert_structural_equal_ignore_global_symbol(
        mma_sync_m16n16k16_desc, mma_sync_m16n16k16_desc_manual
    )

    # The following is an example of an error message from calling an invalid function

    # error: Error occurred when invoking the function sqrt:
    # loop of ufunc does not support argument 0 of type Var which has no callable sqrt method
    #  --> test_tvmscript_syntax_sugar.py:334:19
    #      |
    #  334 |              ind = sqrt(i)
    #      |                    ^^^^^^^
    # note: run with `TVM_BACKTRACE=1` environment variable to display a backtrace.

    # Uncomment to see the error above.
    # def sqrt(x):
    #     import numpy as np
    #     return np.sqrt(x)

    # @T.prim_func
    # def loop(a: T.handle) -> None:
    #     A = T.match_buffer(a, (128,))
    #     for i in T.serial(128):
    #         ind = sqrt(i)
    #         A[i] = A[ind]


def test_int64_loop():
    @T.prim_func
    def int64_grid(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + 1.0

    @T.prim_func
    def int64_grid_expanded(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i in range(T.int64(0), T.int64(128)):
            for j in range(T.int64(0), T.int64(128)):
                with T.block("C"):
                    vi = T.axis.spatial(T.int64(128), i)
                    vj = T.axis.spatial(T.int64(128), j)
                    B[vi, vj] = A[vi, vj] + 1.0

    assert_structural_equal_ignore_global_symbol(int64_grid, int64_grid_expanded)


def test_implicit_evaluate_assume():
    @T.prim_func
    def explicit(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 5))
        A[0] = 10

    @T.prim_func
    def implicit(A: T.Buffer(1, "int32")):
        T.assume(A[0] == 5)
        A[0] = 10

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_implicit_evaluate_call_extern():
    @T.prim_func
    def explicit(A: T.Buffer(1, "int32")):
        T.evaluate(T.call_extern("extern_func", A.data, dtype="int32"))

    @T.prim_func
    def implicit(A: T.Buffer(1, "int32")):
        T.call_extern("extern_func", A.data, dtype="int32")

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_preserve_trivial_let_binding():
    @T.prim_func
    def explicit(i: T.int32):
        j = T.int32()
        T.LetStmt(i, var=j)
        T.evaluate(j)

    @T.prim_func
    def implicit(i: T.int32):
        j = i
        T.evaluate(j)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_preserve_trivial_let_binding_of_value():
    @T.prim_func
    def explicit(i: T.int32):
        j = T.int32()
        T.LetStmt(42, var=j)
        T.evaluate(j)

    @T.prim_func
    def implicit(i: T.int32):
        j = 42
        T.evaluate(j)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_preserve_parameter_name():
    @T.prim_func
    def func(i: T.int32):
        j = i
        T.evaluate(j)

    param_name = func.params[0].name
    assert param_name == "i"


def test_preserve_variable_name():
    """Use variable name when generating tir::LetStmt"""

    @T.prim_func
    def func():
        for i in T.serial(16):
            j = i // 4
            T.evaluate(j)

    var_name = func.body.body.var.name
    assert var_name == "j"


def test_boolean_constant():
    """Python booleans should become T.Bool objects"""

    @T.prim_func
    def explicit():
        T.evaluate(T.bool(True))

    @T.prim_func
    def implicit():
        T.evaluate(True)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_foldable_boolean_in_assert():
    """Foldable booleans T.Bool objects

    The condition of an assert statement should be a boolean
    expression.  Previously, this test failed because the FFI does not
    distinguish between integer primitives and boolean primitives.
    """

    @T.prim_func
    def explicit():
        assert T.bool(False), "Message"
        T.evaluate(0)

    @T.prim_func
    def implicit():
        assert 0 == 1, "Message"
        T.evaluate(0)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


if __name__ == "__main__":
    tvm.testing.main()
