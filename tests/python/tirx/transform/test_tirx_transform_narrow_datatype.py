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
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def _index_module(m, n, input_dtype, loop_dtype, kind, index_dtype="int32"):
    input_scalar, loop_scalar = getattr(T, input_dtype), getattr(T, loop_dtype)
    shape = input_scalar(m) * input_scalar(n)
    loop_m, loop_n = loop_scalar(m), loop_scalar(n)
    index_scalar = getattr(T, index_dtype)
    lanes = 2 if kind == "vector" else 1
    dtype = "float32x2" if kind == "vector" else "float32"
    output_shape = shape * 2 if kind == "slice" else shape

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((shape,), dtype), B: T.Tensor((output_shape,), dtype)):
            if T.constexpr(kind == "thread"):
                bx = T.launch_thread("blockIdx.x", loop_m)
                tx = T.launch_thread("threadIdx.x", loop_n)
                B[bx * loop_n + tx] = A[bx * loop_n + tx] + T.float32(1)
            elif T.constexpr(kind == "vector"):
                for i in T.serial(loop_m):
                    B[i] = A[i] + T.Broadcast(T.float32(1), lanes)
                A[index_scalar(0)] = B[index_scalar(1)]
            else:
                for i in T.serial(loop_m):
                    for j in T.serial(loop_n):
                        if T.constexpr(kind == "slice"):
                            A[i * loop_n + j] = B[i * 2 * loop_n + 2 * j] + T.float32(1)
                        else:
                            B[i * loop_n + j] = A[i * loop_n + j] + T.float32(1)

    return Module


@pytest.mark.parametrize("m,n,input_dtype,target_bits,expected_dtype", [
    (2, 2, "int32", 32, "int32"),
    (2, 2, "int64", 32, "int32"),
    (2**16, 2**16, "int64", 32, "int64"),
    (2, 2, "int32", 16, "int16"),
    (2**10, 2**10, "int32", 16, "int32"),
])
def test_basic(m, n, input_dtype, target_bits, expected_dtype):
    Before = _index_module(m, n, input_dtype, input_dtype, "serial")
    Expected = _index_module(m, n, input_dtype, expected_dtype, "serial")
    After = tvm.tirx.transform.NarrowDataType(target_bits)(Before)
    tvm.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_symbolic_extent(dtype):
    scalar = getattr(T, dtype)

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.handle("float32"), B: T.handle("float32"), m: scalar, n: scalar):
            A_buf = T.decl_tensor(A, ty_args=[T.Tensor((m * n,), "float32", data=A)])
            B_buf = T.decl_tensor(B, ty_args=[T.Tensor((m * n,), "float32", data=B)])
            for i in T.serial(m):
                for j in T.serial(n):
                    B_buf[i * n + j] = A_buf[i * n + j] + T.float32(1)

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.handle("float32"), B: T.handle("float32"), m: scalar, n: scalar):
            A_buf = T.decl_tensor(A, ty_args=[T.Tensor((m * n,), "float32", data=A, elem_offset=T.int32(0))])
            B_buf = T.decl_tensor(B, ty_args=[T.Tensor((m * n,), "float32", data=B, elem_offset=T.int32(0))])
            for i in T.serial(m):
                for j in T.serial(n):
                    B_buf[i * n + j] = A_buf[i * n + j] + T.float32(1)

    After = tvm.tirx.transform.NarrowDataType(32)(Before)
    tvm.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("m,n,input_dtype,target_bits,expected_dtype", [
    (2, 32, "int32", 32, "int32"),
    (2, 32, "int64", 32, "int32"),
    (2**30, 32, "int64", 32, "int64"),
    (2, 32, "int32", 16, "int16"),
    (2**14, 32, "int32", 16, "int32"),
])
def test_thread_axis(m, n, input_dtype, target_bits, expected_dtype):
    Before = _index_module(m, n, input_dtype, input_dtype, "thread")
    Expected = _index_module(m, n, input_dtype, expected_dtype, "thread")
    After = tvm.tirx.transform.NarrowDataType(target_bits)(Before)
    tvm.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("m,input_dtype,target_bits,expected_dtype", [
    (2**10, "int32", 32, "int32"),
    (2**10, "int64", 32, "int32"),
    (2**32, "int64", 32, "int64"),
    (2**10, "int32", 16, "int16"),
    (2**16, "int32", 16, "int32"),
])
def test_multilanes(m, input_dtype, target_bits, expected_dtype):
    Before = _index_module(m, 1, input_dtype, input_dtype, "vector")
    Expected = _index_module(m, 1, input_dtype, expected_dtype, "vector", f"int{target_bits}")
    After = tvm.tirx.transform.NarrowDataType(target_bits)(Before)
    tvm.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("n,expected_dtype", [(2**15, "int32"), (2**15 + 1, "int64")])
def test_slice(n, expected_dtype):
    Before = _index_module(2**15, n, "int64", "int64", "slice")
    Expected = _index_module(2**15, n, "int64", expected_dtype, "slice")
    After = tvm.tirx.transform.NarrowDataType(32)(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_condition():
    @T.function
    def before(A: T.Tensor((128,), "float32"), B: T.Tensor((130,), "float32")):
        for i, j in T.grid(T.int64(2), T.int64(65)):
            if i * T.int64(65) + j >= T.int64(0) and i * T.int64(65) + j < T.int64(128):
                A[i * T.int64(65) + j] = 0.0
        for i, j in T.grid(T.int64(2), T.int64(65)):
            B[i * T.int64(65) + j] = T.if_then_else(
                i * T.int64(65) + j >= T.int64(0) and i * T.int64(65) + j < T.int64(128),
                A[i * T.int64(65) + j],
                0.0,
            )

    @T.function
    def expected_after(A: T.Tensor(128, "float32"), B: T.Tensor(130, "float32")):
        for i, j in T.grid(2, 65):
            if i * 65 + j >= 0 and i * 65 + j < 128:
                A[i * 65 + j] = T.float32(0)
        for i, j in T.grid(2, 65):
            B[i * 65 + j] = T.if_then_else(
                i * 65 + j >= 0 and i * 65 + j < 128, A[i * 65 + j], T.float32(0)
            )

    @I.ir_module
    class Before:
        main = before.with_attr("global_symbol", "main")

    after = tvm.tirx.transform.NarrowDataType(32)(
        Before
    )["main"]
    tvm.ir.assert_structural_equal(after, expected_after.with_attr("global_symbol", "main"))


def test_block():
    @T.function
    def before(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        for i in T.serial(0, T.int64(16)):
            for j in T.serial(0, T.int64(8)):
                B[i * T.int64(8) + j] = A[i * T.int64(8) + j] + T.float32(1)

    @T.function
    def expected_after(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        for i in T.serial(0, T.int32(16)):
            for j in T.serial(0, T.int32(8)):
                B[i * T.int32(8) + j] = A[i * T.int32(8) + j] + T.float32(1)

    @I.ir_module
    class Before:
        main = before.with_attr("global_symbol", "main")

    after = tvm.tirx.transform.NarrowDataType(32)(
        Before
    )["main"]
    tvm.ir.assert_structural_equal(after, expected_after.with_attr("global_symbol", "main"))


def test_avg_pool2d():
    @T.function
    def before(PSUM: T.Tensor((313600,), "int32"), PAVG: T.Tensor((313600,), "int32")):
        for j in T.parallel(T.int64(0), T.int64(280)):
            for i in T.serial(T.int64(0), T.int64(35)):
                for vi in T.vectorized(T.int64(0), T.int64(32)):
                    PAVG[(((j * T.int64(1120)) + (i * T.int64(32))) + vi)] = T.cast(
                        T.Div(
                            T.cast(PSUM[(((j * T.int64(1120)) + (i * T.int64(32))) + vi)], "int64"),
                            T.max(
                                (
                                    (
                                        (
                                            T.min(
                                                T.int64(1),
                                                (T.int64(34) - T.floormod(j, T.int64(35))),
                                            )
                                            + T.int64(2)
                                        )
                                        - T.max(
                                            (T.int64(1) - T.floormod(j, T.int64(35))), T.int64(0)
                                        )
                                    )
                                    * (
                                        (T.min(T.int64(1), (T.int64(34) - i)) + T.int64(2))
                                        - T.max((T.int64(1) - i), T.int64(0))
                                    )
                                ),
                                T.int64(1),
                            ),
                        ),
                        "int32",
                    )

    @T.function
    def expected_after(PSUM: T.Tensor((313600,), "int32"), PAVG: T.Tensor((313600,), "int32")):
        for j in T.parallel(T.int32(0), T.int32(280)):
            for i in T.serial(T.int32(0), T.int32(35)):
                for vi in T.vectorized(T.int32(0), T.int32(32)):
                    PAVG[(((j * T.int32(1120)) + (i * T.int32(32))) + vi)] = T.Div(
                        PSUM[(((j * T.int32(1120)) + (i * T.int32(32))) + vi)],
                        (
                            (
                                (
                                    T.min(T.int32(1), (T.int32(34) - T.floormod(j, T.int32(35))))
                                    + T.int32(2)
                                )
                                - T.max((T.int32(1) - T.floormod(j, T.int32(35))), T.int32(0))
                            )
                            * (
                                (T.min(T.int32(1), (T.int32(34) - i)) + T.int32(2))
                                - T.max((T.int32(1) - i), T.int32(0))
                            )
                        ),
                    )

    @I.ir_module
    class Before:
        main = before.with_attr("global_symbol", "main")

    after = tvm.tirx.transform.NarrowDataType(32)(
        Before
    )
    @I.ir_module
    class Before:
        main = before.with_attr("global_symbol", "main")

    after = tvm.tirx.transform.StmtSimplify()(after)
    tvm.ir.assert_structural_equal(after["main"], expected_after.with_attr("global_symbol", "main"))


def test_narrow_i64_valued_tensorload_index_to_i32():
    @T.function
    def before(A: T.Tensor((16,), "int64")):
        for i in range(T.int64(15)):
            A[i + T.int64(1)] = A[i] + T.int64(1)

    @T.function
    def expect(A: T.Tensor((16,), "int64")):
        for i in range(15):
            A[i + 1] = A[i] + T.int64(1)

    @I.ir_module
    class Before:
        main = before.with_attr("global_symbol", "main")

    after = tvm.tirx.transform.NarrowDataType(32)(
        Before
    )["main"]
    tvm.ir.assert_structural_equal(after, expect.with_attr("global_symbol", "main"))


if __name__ == "__main__":
    tvm.testing.main()
