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
"""Unittests for tvm.script.parser.tir"""

import pytest
import tvm.testing
from tvm.script.parser import tir as T
from tvm import ir, tir


def test_tir_buffer_proxy():
    buffer_0 = T.Buffer((128, 128), "float32")
    assert (
        isinstance(buffer_0, tir.Buffer)
        and list(buffer_0.shape) == [128, 128]
        and buffer_0.dtype == "float32"
    )

    buffer_1 = T.Buffer((64, 64, 64), "int32")
    assert (
        isinstance(buffer_1, tir.Buffer)
        and list(buffer_1.shape) == [64, 64, 64]
        and buffer_1.dtype == "int32"
    )


def test_tir_ptr_proxy():
    ptr_0 = T.handle("int32", "global")
    assert (
        isinstance(ptr_0, tir.Var)
        and ptr_0.dtype == "handle"
        and isinstance(ptr_0.type_annotation, ir.PointerType)
        and ptr_0.type_annotation.element_type == ir.PrimType("int32")
        and ptr_0.type_annotation.storage_scope == "global"
    )

    ptr_1 = T.handle("float32", "shared")
    assert (
        isinstance(ptr_1, tir.Var)
        and ptr_1.dtype == "handle"
        and isinstance(ptr_1.type_annotation, ir.PointerType)
        and ptr_1.type_annotation.element_type == ir.PrimType("float32")
        and ptr_1.type_annotation.storage_scope == "shared"
    )


def test_tir_func_name():
    @T.prim_func
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    assert matmul.__name__ == "matmul"
    assert matmul.attrs["global_symbol"] == "matmul"


def test_tir_func_private_attrs():
    @T.prim_func(private=True)
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"attr": "value"})
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    assert "global_symbol" not in matmul.attrs


def test_tir_func_private_manual_global_symbol_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func(private=True)
        def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "matmul"})
            A = T.match_buffer(a, [128, 128])
            B = T.match_buffer(b, [128, 128])
            C = T.match_buffer(c, [128, 128])
            for i, j, k in T.grid(128, 128, 128):
                with T.block("update"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        # should not execute
        assert matmul.__name__ == "matmul"


def test_tir_macro_decorator_signature():
    @T.prim_func(private=True)
    def evaluate0():
        T.evaluate(0)

    # Ok, no parentheses
    @T.macro
    def func1():
        T.evaluate(0)

    @T.prim_func(private=True)
    def use1():
        func1()

    tvm.ir.assert_structural_equal(use1, evaluate0)

    # Ok, empty parentheses
    @T.macro()
    def func2():
        T.evaluate(0)

    @T.prim_func(private=True)
    def use2():
        func2()

    tvm.ir.assert_structural_equal(use1, evaluate0)

    with pytest.raises(ValueError):
        # Wrong: non-keyword argument
        @T.macro(True)
        def func3():
            T.evaluate()


def test_tir_macro_signature():
    @T.macro
    def assign(i, *args, t1, **kwargs):
        vi, vj, vk = T.axis.remap("SSR", [i, args[0], args[1]])
        kwargs["t3"][vi, vj] = kwargs["t3"][vi, vj] + t1[vi, vk] * kwargs["t2"][vj, vk]

    @T.prim_func(private=True)
    def matmul_w_macro(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                assign(i, j, k, t1=A, t2=B, t3=C)

    @T.prim_func(private=True)
    def matmul_no_macro(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    tvm.ir.assert_structural_equal(matmul_no_macro, matmul_w_macro)


def test_tir_macro_hygienic():
    x_value = 128

    @T.macro(hygienic=True)
    def static_capture(A, B):
        B[()] = A[x_value]

    @T.prim_func(private=True)
    def use_hygienic(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in T.serial(10):
            static_capture(A, B)

    @T.prim_func(private=True)
    def expected_hygienic(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in range(10):
            B[()] = A[128]

    tvm.ir.assert_structural_equal(use_hygienic, expected_hygienic)


def test_tir_macro_non_hygienic():
    x_value = 128

    @T.macro(hygienic=False)
    def dynamic_capture(A, B):
        B[()] = A[x_value]

    @T.prim_func(private=True)
    def use_non_hygienic(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in T.serial(10):
            dynamic_capture(A, B)

    @T.prim_func(private=True)
    def expected_non_hygienic(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
        for x_value in range(10):
            B[()] = A[x_value]

    tvm.ir.assert_structural_equal(use_non_hygienic, expected_non_hygienic)


def test_tir_macro_in_class():
    class Object:
        def __init__(self, x: T.Buffer):
            self.local_x = T.alloc_buffer(x.shape, x.dtype)

        @T.macro
        def load(self, x: T.Buffer):
            N, M = T.meta_var(self.local_x.shape)
            for i, j in T.grid(N, M):
                with T.block("update"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    self.local_x[vi, vj] = x[vi, vj]

    @T.prim_func(private=True)
    def func_w_macro(a: T.handle):
        A = T.match_buffer(a, [128, 128])
        o1 = T.meta_var(Object(A))
        o1.load(A)
        o2 = T.meta_var(Object(A))
        o2.load(o1.local_x)

    @T.prim_func(private=True)
    def func_no_macro(a: T.handle):
        A = T.match_buffer(a, [128, 128])
        local_a = T.alloc_buffer([128, 128])
        for i, j in T.grid(128, 128):
            with T.block("update"):
                vi, vj = T.axis.remap("SS", [i, j])
                local_a[vi, vj] = A[vi, vj]
        local_b = T.alloc_buffer([128, 128])
        for i, j in T.grid(128, 128):
            with T.block("update"):
                vi, vj = T.axis.remap("SS", [i, j])
                local_b[vi, vj] = local_a[vi, vj]

    tvm.ir.assert_structural_equal(func_no_macro, func_w_macro)


def test_tir_starred_expression():
    dims = (128, 128)

    @T.prim_func(private=True)
    def starred(a: T.handle) -> None:
        A = T.match_buffer(a, [128, *dims], "int32")
        for i, j, k in T.grid(128, *dims):
            A[i, j, k] = T.int32(1)

    @T.prim_func(private=True)
    def non_starred(a: T.handle) -> None:
        A = T.match_buffer(a, [128, 128, 128], "int32")
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_tir_starred_shape_expression():
    dims = (128, 128)

    @T.prim_func(private=True)
    def starred(a: T.handle) -> None:
        A = T.match_buffer(a, [128, *dims], "int32")
        for i, j, k in T.grid(*A.shape):
            A[i, j, k] = T.int32(1)

    @T.prim_func(private=True)
    def non_starred(a: T.handle) -> None:
        A = T.match_buffer(a, [128, 128, 128], "int32")
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_tir_dynamic_for_loop():
    dims = (128, 128)

    @T.prim_func(private=True)
    def starred(a: T.handle) -> None:
        A = T.match_buffer(a, [128, *dims], "int32")
        for iters in T.grid(*A.shape):
            A[iters] = T.int32(1)

    @T.prim_func(private=True)
    def non_starred(a: T.handle) -> None:
        A = T.match_buffer(a, [128, 128, 128], "int32")
        for i, j, k in T.grid(128, 128, 128):
            A[i, j, k] = T.int32(1)

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_tir_starred_for_loop():
    dims = (128, 128)

    @T.prim_func(private=True)
    def starred(a: T.handle, b: T.handle):
        A = T.match_buffer(a, [*dims, 128], "int32")
        B = T.match_buffer(b, dims, "int32")
        for *spatial, reduction in T.grid(*A.shape):
            with T.block("reduce"):
                with T.init():
                    B[spatial] = T.int32(0)
                B[spatial] = B[spatial] + A[(*spatial, reduction)]

    @T.prim_func(private=True)
    def non_starred(a: T.handle, b: T.handle):
        A = T.match_buffer(a, [128, 128, 128], "int32")
        B = T.match_buffer(b, [128, 128], "int32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("reduce"):
                with T.init():
                    B[i, j] = T.int32(0)
                B[i, j] = B[i, j] + A[i, j, k]

    tvm.ir.assert_structural_equal(starred, non_starred)


def test_tir_empty_tuple_index():
    @T.macro
    def bar(val):
        T.evaluate(val)

    @T.prim_func(private=True)
    def func_with_empty_tuple(A: T.Buffer((), "int32"), B: T.Buffer((), "int32")):
        bar(val=A[()])

    @T.prim_func(private=True)
    def expected(A: T.Buffer((), "int32"), B: T.Buffer((), "int32")):
        T.evaluate(A[()])

    tvm.ir.assert_structural_equal(func_with_empty_tuple, expected)


def test_tir_builtin_expression():
    dims = (128, 128)

    @T.prim_func(private=True)
    def with_builtin(a: T.handle) -> None:
        A = T.match_buffer(a, [len(dims), *dims], "int32")
        for i, j, k in T.grid(*A.shape):
            A[i, j, k] = T.int32(1 + len(A.shape))

    @T.prim_func(private=True)
    def evaluated(A: T.Buffer((2, 128, 128), "int32")):
        for i, j, k in T.grid(2, 128, 128):
            A[i, j, k] = 4

    tvm.ir.assert_structural_equal(with_builtin, evaluated)


def test_thread_binding_dtype():
    @T.prim_func(private=True)
    def func(A: T.Buffer((128, 128)), B: T.Buffer((128, 128))):
        for i in T.thread_binding(T.int64(128), "threadIdx.x"):
            for j in T.thread_binding(128, "threadIdx.y"):
                B[i, j] = A[i, j]

    loop_i = func.body
    loop_j = loop_i.body
    assert loop_i.loop_var.dtype == "int64"
    assert loop_i.thread_binding.var.dtype == "int64"
    assert loop_j.loop_var.dtype == "int32"
    assert loop_j.thread_binding.var.dtype == "int32"


def test_inferred_sinfo_with_prim_args():
    """A PrimFunc may have inferred StructInfo"""

    @T.prim_func
    def func(M: T.int32, N: T.int32) -> T.int32:
        T.ret(M * N)

    expected = tvm.relax.FuncStructInfo(
        [
            tvm.relax.PrimStructInfo("int32"),
            tvm.relax.PrimStructInfo("int32"),
        ],
        tvm.relax.PrimStructInfo("int32"),
        purity=True,
    )
    tvm.ir.assert_structural_equal(func.struct_info, expected)


def test_inferred_sinfo_with_buffer_args():
    """PrimFunc buffer arguments are inferred as R.Tensor"""

    @T.prim_func
    def func(A: T.Buffer([16, 16], "float32"), B: T.Buffer([256], "int32")) -> T.float32:
        T.ret(T.float32(42.0))

    expected = tvm.relax.FuncStructInfo(
        [
            tvm.relax.TensorStructInfo([16, 16], "float32"),
            tvm.relax.TensorStructInfo([256], "int32"),
        ],
        tvm.relax.PrimStructInfo("float32"),
        purity=True,
    )
    tvm.ir.assert_structural_equal(func.struct_info, expected)


def test_inferred_sinfo_with_internal_allocation():
    """A pure function may still write to internal allocations.

    Whether a function writes to internal allocations is not a visible
    effect, and does not impact the purity of a function.
    """

    @T.prim_func
    def func(A: T.Buffer([16, 16], "float32")) -> T.float32:
        Sum = T.decl_buffer([], "float32")
        Sum[()] = 0.0
        for i, j in T.grid(16, 16):
            Sum[()] = Sum[()] + A[i, j]

        T.ret(Sum[()])

    expected = tvm.relax.FuncStructInfo(
        [
            tvm.relax.TensorStructInfo([16, 16], "float32"),
        ],
        tvm.relax.PrimStructInfo("float32"),
        purity=True,
    )
    tvm.ir.assert_structural_equal(func.struct_info, expected)


def test_inferred_sinfo_with_output_buffer():
    """A pure function may not write to an argument buffer

    If an argument buffer is written to, the function must be impure.
    """

    @T.prim_func
    def func(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
        for i in range(16):
            B[i] = A[i]

    expected = tvm.relax.FuncStructInfo(
        [
            tvm.relax.TensorStructInfo([16], "float32"),
            tvm.relax.TensorStructInfo([16], "float32"),
        ],
        tvm.relax.TupleStructInfo([]),
        purity=False,
    )
    tvm.ir.assert_structural_equal(func.struct_info, expected)


def test_inferred_sinfo_with_dynamic_buffer():
    """The inferred StructInfo may contain dynamic shapes"""

    @T.prim_func
    def func(a_handle: T.handle, b_handle: T.handle):
        M = T.int64()
        N = T.int64()
        A = T.match_buffer(a_handle, [M, N], "float32")
        B = T.match_buffer(b_handle, [M * N], "float32")
        for i, j in T.grid(M, N):
            B[i * N + j] = A[i, j]

    M = tvm.tir.Var("M", "int64")
    N = tvm.tir.Var("N", "int64")
    expected = tvm.relax.FuncStructInfo(
        [
            tvm.relax.TensorStructInfo([M, N], "float32"),
            tvm.relax.TensorStructInfo([M * N], "float32"),
        ],
        tvm.relax.TupleStructInfo([]),
        purity=False,
    )
    tvm.ir.assert_structural_equal(func.struct_info, expected)


def test_reinterpret_nop():
    """Test builtin reinterpret op"""

    @T.prim_func
    def func(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")) -> None:
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 32):
            with T.block():
                vi = T.axis.remap("S", [i])
                B[vi] = T.reinterpret("float32", A[vi])

    @T.prim_func
    def expected(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")) -> None:
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 32):
            with T.block():
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    tvm.ir.assert_structural_equal(func, expected)


def test_launch_thread_i64():
    """Test launching thread with int64"""

    @T.prim_func
    def func() -> None:
        blockIdx_x = T.launch_thread("blockIdx.x", T.int64(1))
        if blockIdx_x == T.int64(0):
            T.evaluate(T.int64(0))
        else:
            T.evaluate(T.int64(1))

    assert func.body.node.dom.min.dtype == "int64"
    assert func.body.node.dom.extent.dtype == "int64"


def test_deterministic_branch():
    """Test deterministic branch"""

    def create_func(predicate: bool):
        @T.prim_func(private=True)
        def func() -> None:
            if predicate:
                T.evaluate(0)
            else:
                T.evaluate(1)

        return func

    def create_expected(value):
        @T.prim_func(private=True)
        def expected() -> None:
            T.evaluate(value)

        return expected

    tvm.ir.assert_structural_equal(create_func(True), create_expected(0))
    tvm.ir.assert_structural_equal(create_func(False), create_expected(1))


if __name__ == "__main__":
    tvm.testing.main()
