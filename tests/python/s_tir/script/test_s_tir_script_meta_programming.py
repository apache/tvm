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
# ruff: noqa: F841
"""S-TIR script meta programming."""

from __future__ import annotations

import numpy
import pytest

import tvm
import tvm.testing
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_meta_programming_matmul():
    def matmul_generator(M: int, N: int, K: int, dtype: str):
        @Ts.prim_func
        def matmul(
            A: T.Buffer([M, K], dtype=dtype),
            B: T.Buffer([N, K], dtype=dtype),
            C: T.Buffer([M, N], dtype=dtype),
        ) -> None:
            for i, j, k in T.grid(M, N, K):
                with Ts.sblock():
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        return matmul

    @Ts.prim_func
    def matmul_128_128_128_fp16(
        A: T.Buffer([128, 128], dtype="float16"),
        B: T.Buffer([128, 128], dtype="float16"),
        C: T.Buffer([128, 128], dtype="float16"),
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock():
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    f = matmul_generator(128, 128, 128, "float16").with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(f, matmul_128_128_128_fp16.with_attr("global_symbol", "main"))


def test_meta_programming_uncaptured_var():
    def generate_erf(dtype):
        @Ts.prim_func
        def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
            for i in range(1):
                with Ts.sblock("C"):
                    C[i] = T.erf(A[i])

        return main

    @Ts.prim_func
    def fp32(A: T.Buffer((1,), "float32"), C: T.Buffer((1,), "float32")):
        for i in range(1):
            with Ts.sblock("C"):
                C[i] = T.erf(A[i])

    @Ts.prim_func
    def fp16(A: T.Buffer((1,), "float16"), C: T.Buffer((1,), "float16")):
        for i in range(1):
            with Ts.sblock("C"):
                C[i] = T.erf(A[i])

    f1 = generate_erf("float32").with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(f1, fp32.with_attr("global_symbol", "main"))
    f2 = generate_erf("float16").with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(f2, fp16.with_attr("global_symbol", "main"))


def test_tir_macro_decorator_signature():
    @Ts.prim_func(private=True)
    def evaluate0():
        T.evaluate(0)

    # Ok, no parentheses
    @T.inline
    def func1():
        T.evaluate(0)

    @Ts.prim_func(private=True)
    def use1():
        func1()

    tvm.ir.assert_structural_equal(use1, evaluate0)

    # Ok, empty parentheses
    @T.inline()
    def func2():
        T.evaluate(0)

    @Ts.prim_func(private=True)
    def use2():
        func2()

    tvm.ir.assert_structural_equal(use1, evaluate0)

    with pytest.raises(ValueError):
        # Wrong: non-keyword argument
        @T.inline(True)
        def func3():
            T.evaluate()


def test_tir_macro_signature():
    @T.inline
    def assign(i, *args, t1, **kwargs):
        vi, vj, vk = Ts.axis.remap("SSR", [i, args[0], args[1]])
        kwargs["t3"][vi, vj] = kwargs["t3"][vi, vj] + t1[vi, vk] * kwargs["t2"][vj, vk]

    @Ts.prim_func(private=True)
    def matmul_w_macro(
        A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                assign(i, j, k, t1=A, t2=B, t3=C)

    @Ts.prim_func(private=True)
    def matmul_no_macro(
        A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    tvm.ir.assert_structural_equal(matmul_no_macro, matmul_w_macro)


def test_tir_macro_hygienic():
    x_value = 128

    @T.inline
    def static_capture(A, B):
        B[()] = A[x_value]

    @Ts.prim_func(private=True)
    def use_hygienic(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in T.serial(10):
            static_capture(A, B)

    @Ts.prim_func(private=True)
    def expected_hygienic(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in range(10):
            B[()] = A[128]

    tvm.ir.assert_structural_equal(use_hygienic, expected_hygienic)


def test_tir_inline_late_binding():
    """Inline defined inside prim_func uses LEGB late binding:
    it sees the current value of variables from its enclosing scope at call time."""

    @Ts.prim_func(private=True)
    def use_late_binding(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in T.serial(10):

            @T.inline
            def capture(A, B):
                B[()] = A[x_value]

            capture(A, B)

    @Ts.prim_func(private=True)
    def expected(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in range(10):
            B[()] = A[x_value]

    tvm.ir.assert_structural_equal(use_late_binding, expected)


def test_tir_macro_in_class():
    class Object:
        def __init__(self, x: T.Buffer):
            self.local_x = Ts.sblock_alloc_buffer(x.shape, x.dtype)

        @T.inline
        def load(self, x: T.Buffer):
            N, M = T.meta_var(self.local_x.shape)
            for i, j in T.grid(N, M):
                with Ts.sblock("update"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    self.local_x[vi, vj] = x[vi, vj]

    @Ts.prim_func(private=True)
    def func_w_macro(A: T.Buffer([128, 128])):
        o1 = T.meta_var(Object(A))
        o1.load(A)
        o2 = T.meta_var(Object(A))
        o2.load(o1.local_x)

    @Ts.prim_func(private=True)
    def func_no_macro(A: T.Buffer([128, 128])):
        local_a = Ts.sblock_alloc_buffer([128, 128])
        N, M = local_a.shape
        for i, j in T.grid(N, M):
            with Ts.sblock("update"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                local_a[vi, vj] = A[vi, vj]
        local_b = Ts.sblock_alloc_buffer([128, 128])
        N, M = local_b.shape
        for i, j in T.grid(N, M):
            with Ts.sblock("update"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                local_b[vi, vj] = local_a[vi, vj]

    tvm.ir.assert_structural_equal(func_no_macro, func_w_macro)


def test_tir_starred_expression():
    dims = (128, 128)

    @Ts.prim_func(private=True)
    def starred(A: T.Buffer([128, *dims], "int32")) -> None:
        for i, j, k in T.grid(128, *dims):
            A[i, j, k] = T.int32(1)

    @Ts.prim_func(private=True)
    def non_starred(A: T.Buffer([128, 128, 128], "int32")) -> None:
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_tir_dynamic_for_loop():
    dims = (128, 128)

    @Ts.prim_func(private=True)
    def starred(A: T.Buffer([128, *dims], "int32")) -> None:
        for (*iters,) in T.grid(*A.shape):
            A[iters] = T.int32(1)

    @Ts.prim_func(private=True)
    def non_starred(A: T.Buffer([128, 128, 128], "int32")) -> None:
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_tir_starred_for_loop():
    dims = (128, 128)

    @Ts.prim_func(private=True)
    def starred(A: T.Buffer([*dims, 128], "int32"), B: T.Buffer(dims, "int32")):
        for *spatial, reduction in T.grid(*A.shape):
            with Ts.sblock("reduce"):
                with Ts.init():
                    B[spatial] = T.int32(0)
                B[spatial] = B[spatial] + A[(*spatial, reduction)]

    @Ts.prim_func(private=True)
    def non_starred(A: T.Buffer([128, 128, 128], "int32"), B: T.Buffer([128, 128], "int32")):
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("reduce"):
                with Ts.init():
                    B[i, j] = T.int32(0)
                B[i, j] = B[i, j] + A[i, j, k]

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_tir_builtin_expression():
    dims = (128, 128)

    @Ts.prim_func(private=True)
    def with_builtin(A: T.Buffer([len(dims), *dims], "int32")) -> None:
        for i, j, k in T.grid(*A.shape):
            A[i, j, k] = T.int32(1 + len(A.shape))

    @Ts.prim_func(private=True)
    def evaluated(A: T.Buffer((2, 128, 128), "int32")):
        for i, j, k in T.grid(2, 128, 128):
            A[i, j, k] = 4

    tvm.ir.assert_structural_equal(with_builtin, evaluated)


def test_deterministic_branch():
    """Test deterministic branch"""

    def create_func(predicate: bool):
        @Ts.prim_func(private=True)
        def func() -> None:
            if T.constexpr(predicate):
                T.evaluate(0)
            else:
                T.evaluate(1)

        return func

    def create_expected(value):
        @Ts.prim_func(private=True)
        def expected() -> None:
            T.evaluate(value)

        return expected

    tvm.ir.assert_structural_equal(create_func(True), create_expected(0))
    tvm.ir.assert_structural_equal(create_func(False), create_expected(1))


def test_tir_macro_block_name_suffix():
    @T.inline
    def operation(A, idx):
        with Ts.sblock("op"):
            v = Ts.axis.remap("S", [idx])
            A[v] = A[v] * T.float32(2)

    @Ts.prim_func(private=True)
    def func_w_macro(A: T.Buffer([10])) -> None:
        for i in T.serial(0, 10):
            operation(A, i)
            operation(A, i)
            operation(A, i)

    @Ts.prim_func(private=True)
    def expected(A: T.Buffer([10])) -> None:
        for i in T.serial(0, 10):
            with Ts.sblock("op"):
                v = Ts.axis.remap("S", [i])
                A[v] = A[v] * T.float32(2)
            with Ts.sblock("op_1"):
                v = Ts.axis.remap("S", [i])
                A[v] = A[v] * T.float32(2)
            with Ts.sblock("op_2"):
                v = Ts.axis.remap("S", [i])
                A[v] = A[v] * T.float32(2)

    tvm.ir.assert_structural_equal(func_w_macro, expected)


def _normalize(func):
    """Strip the global_symbol so function names do not affect structural equality."""
    return func.with_attr("global_symbol", "")


def test_prim_func_closure_shape():
    """Closure variable used in Buffer shape annotation."""

    def f(M=16):
        @Ts.prim_func
        def func(A: T.Buffer((M,), "float32")):
            T.evaluate(0)

        return func

    @Ts.prim_func
    def expected_16(A: T.Buffer((16,), "float32")):
        T.evaluate(0)

    @Ts.prim_func
    def expected_32(A: T.Buffer((32,), "float32")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(f(16)), _normalize(expected_16))
    tvm.ir.assert_structural_equal(_normalize(f(32)), _normalize(expected_32))


def test_prim_func_closure_dtype():
    """Closure variable used as Buffer dtype."""

    def f(dtype="float32"):
        @Ts.prim_func
        def func(A: T.Buffer((16,), dtype)):
            T.evaluate(0)

        return func

    @Ts.prim_func
    def expected_f32(A: T.Buffer((16,), "float32")):
        T.evaluate(0)

    @Ts.prim_func
    def expected_f16(A: T.Buffer((16,), "float16")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(f("float32")), _normalize(expected_f32))
    tvm.ir.assert_structural_equal(_normalize(f("float16")), _normalize(expected_f16))


def test_prim_func_nested_closure():
    """Variables from enclosing scope active on the call stack (grandparent frame fallback).

    With PEP 563, closure-only variables are missing from __closure__ unless they
    appear in the function body. The ChainMap fallback walks the live call stack,
    so this works when the enclosing frames are still active (outer calls middle
    which applies the decorator, keeping outer's frame alive on the stack).
    """

    def outer(M=16):
        def middle(N=8):
            @Ts.prim_func
            def func(A: T.Buffer((M, N), "float32")):
                T.evaluate(0)

            return func

        return middle()

    @Ts.prim_func
    def expected_16_8(A: T.Buffer((16, 8), "float32")):
        T.evaluate(0)

    @Ts.prim_func
    def expected_32_8(A: T.Buffer((32, 8), "float32")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(outer(16)), _normalize(expected_16_8))
    tvm.ir.assert_structural_equal(_normalize(outer(32)), _normalize(expected_32_8))


def test_ir_module_closure():
    """Closure variable in @I.ir_module class method."""

    def f(M=16):
        @I.ir_module
        class Mod:
            @Ts.prim_func
            def main(A: T.Buffer((M,), "float32")):
                T.evaluate(0)

        return Mod

    @Ts.prim_func
    def expected_16(A: T.Buffer((16,), "float32")):
        T.evaluate(0)

    @Ts.prim_func
    def expected_32(A: T.Buffer((32,), "float32")):
        T.evaluate(0)

    tvm.ir.assert_structural_equal(_normalize(f(16)["main"]), _normalize(expected_16))
    tvm.ir.assert_structural_equal(_normalize(f(32)["main"]), _normalize(expected_32))


def test_mixed_closure_usage():
    """Closure var used in both annotation AND body -- regression check."""

    def f(M=16):
        @Ts.prim_func
        def func(A: T.Buffer((M,), "float32")):
            T.evaluate(M)

        return func

    @Ts.prim_func
    def expected_16(A: T.Buffer((16,), "float32")):
        T.evaluate(16)

    @Ts.prim_func
    def expected_32(A: T.Buffer((32,), "float32")):
        T.evaluate(32)

    tvm.ir.assert_structural_equal(_normalize(f(16)), _normalize(expected_16))
    tvm.ir.assert_structural_equal(_normalize(f(32)), _normalize(expected_32))


np_array = numpy.array([0, 1, 2, 3])


@Ts.prim_func
def matmul(A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_multi_element_array_in_outmost_namespace():
    func = matmul
    rt_func = tvm.script.from_source(
        func.script(), extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir}
    )
    tvm.ir.assert_structural_equal(func, rt_func)


def test_var_capturing_order():
    b = 2

    @Ts.prim_func
    def test_case():
        k: T.let[T.int32] = b

    @Ts.prim_func
    def func_ref():
        k: T.let[T.int32] = 2
        T.evaluate(0)

    tvm.ir.assert_structural_equal(
        test_case.with_attr("global_symbol", "main"), func_ref.with_attr("global_symbol", "main")
    )


def test_bind_with_constant():
    @Ts.prim_func
    def constant_binds():
        x = T.meta_var(1)
        y = T.meta_var(42.0)
        T.evaluate(T.cast(x, "float32") + y)

    @Ts.prim_func
    def constant_binds_wrapped():
        x = T.meta_var(T.int32(1))
        y = T.meta_var(T.float32(42.0))
        T.evaluate(T.cast(x, "float32") + y)

    assert_structural_equal_ignore_global_symbol(constant_binds, constant_binds_wrapped)


def test_func_call():
    def shared_16x16_to_ldmatrix_32x8_layout(i, j):
        thread_id = (i % 8) * 4 + (j % 8) // 2
        return thread_id, (j // 8) * 4 + (i // 8) * 2 + (j % 2)

    @Ts.prim_func
    def mma_sync_m16n16k16_desc(
        A: T.Buffer((32, 8), "float16", align=64, offset_factor=16, scope="warp"),
        B: T.Buffer((32, 8), "float16", align=64, offset_factor=16, scope="warp"),
        C: T.Buffer((32, 8), "float16", align=64, offset_factor=16, scope="warp"),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
            Ts.writes(C[0:32, 0:8])
            for i, j, k in T.grid(16, 16, 16):
                with Ts.sblock("C"):
                    i, j, k = Ts.axis.remap("SSR", [i, j, k])
                    indices_C = shared_16x16_to_ldmatrix_32x8_layout(i, j)
                    indices_A = shared_16x16_to_ldmatrix_32x8_layout(i, k)
                    indices_B = shared_16x16_to_ldmatrix_32x8_layout(k, j)

                    Ts.reads(
                        C[indices_C[0], indices_C[1]],
                        A[indices_A[0], indices_A[1]],
                        B[indices_B[0], indices_B[1]],
                    )
                    Ts.writes(C[indices_C[0], indices_C[1]])

                    C[indices_C[0], indices_C[1]] += (
                        A[indices_A[0], indices_A[1]] * B[indices_B[0], indices_B[1]]
                    )

    @Ts.prim_func
    def mma_sync_m16n16k16_desc_manual(
        A: T.Buffer((32, 8), "float16", align=64, offset_factor=16, scope="warp"),
        B: T.Buffer((32, 8), "float16", align=64, offset_factor=16, scope="warp"),
        C: T.Buffer((32, 8), "float16", align=64, offset_factor=16, scope="warp"),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
            Ts.writes(C[0:32, 0:8])
            for i, j, k in T.grid(16, 16, 16):
                with Ts.sblock("C"):
                    i, j, k = Ts.axis.remap("SSR", [i, j, k])
                    Ts.reads(
                        C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2],
                        A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2],
                        B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2],
                    )
                    Ts.writes(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2])
                    C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] = (
                        C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2]
                        + A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2]
                        * B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2]
                    )

    assert_structural_equal_ignore_global_symbol(
        mma_sync_m16n16k16_desc, mma_sync_m16n16k16_desc_manual
    )


if __name__ == "__main__":
    tvm.testing.main()
