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
from tvm import TVMError
from tvm.script import tir as T
import tvm.testing


def test_thread_axis1():
    @T.prim_func(private=True)
    def before(A: T.Buffer((T.int64(64),), "float32"), B: T.Buffer((T.int64(64),), "float32")):
        blockIdx_x = T.env_thread("blockIdx.x")
        T.launch_thread(blockIdx_x, T.int64(2))
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, T.int64(32))
        B[T.Cast("int64", blockIdx_x) * T.int64(32) + T.Cast("int64", threadIdx_x)] = A[
            T.Cast("int64", blockIdx_x) * T.int64(32) + T.Cast("int64", threadIdx_x)
        ] + T.float32(1)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        blockIdx_x = T.env_thread("blockIdx.x")
        T.launch_thread(blockIdx_x, 2)
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
        B[blockIdx_x * 32 + threadIdx_x] = A[blockIdx_x * 32 + threadIdx_x] + T.float32(1)

    mod = tvm.IRModule.from_expr(before)
    func = tvm.tir.transform.ForceNarrowIndexToInt32()(mod)["main"]
    tvm.ir.assert_structural_equal(func, expected)


def test_thread_axis2():
    @T.prim_func
    def before(
        T_reshape: T.Buffer((1, 12, 384, 384), "float32"),
        placeholder_1: T.Buffer((T.int64(1), T.int64(12), T.int64(384), 384), "bool"),
        T_where: T.Buffer((T.int64(1), T.int64(12), T.int64(384), 384), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in T.serial(T.int64(7)):
                    with T.block("T_where"):
                        ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        ax1 = T.axis.spatial(
                            T.int64(12),
                            (
                                (i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1)
                                * T.int64(1024)
                                + i0_i1_i2_i3_fused_2
                            )
                            % T.int64(1769472)
                            // T.int64(147456),
                        )
                        ax2 = T.axis.spatial(
                            T.int64(384),
                            (
                                (i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1)
                                * T.int64(1024)
                                + i0_i1_i2_i3_fused_2
                            )
                            % T.int64(147456)
                            // T.int64(384),
                        )
                        ax3 = T.axis.spatial(
                            384,
                            T.cast(
                                (
                                    (i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1)
                                    * T.int64(1024)
                                    + i0_i1_i2_i3_fused_2
                                )
                                % T.int64(384),
                                "int32",
                            ),
                        )
                        T.where(
                            (i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1)
                            * T.int64(1024)
                            + i0_i1_i2_i3_fused_2
                            < T.int64(1769472)
                        )
                        T.reads(placeholder_1[ax0, ax1, ax2, ax3], T_reshape[ax0, ax1, ax2, ax3])
                        T.writes(T_where[ax0, ax1, ax2, ax3])
                        T_where[ax0, ax1, ax2, ax3] = T.Select(
                            T.cast(placeholder_1[ax0, ax1, ax2, ax3], "int32") != 0,
                            T.float32(-1000000000),
                            T_reshape[ax0, ax1, ax2, ax3],
                        )

    @T.prim_func
    def expected(
        T_reshape: T.Buffer((1, 12, 384, 384), "float32"),
        placeholder_1: T.Buffer((1, 12, 384, 384), "bool"),
        T_where: T.Buffer((1, 12, 384, 384), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0_i1_i2_i3_fused_1 in T.thread_binding(256, thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(1024, thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(7):
                    with T.block("T_where"):
                        ax0 = T.axis.spatial(1, 0)
                        ax1 = T.axis.spatial(
                            12,
                            (
                                (i0_i1_i2_i3_fused_0 * 256 + i0_i1_i2_i3_fused_1) * 1024
                                + i0_i1_i2_i3_fused_2
                            )
                            % 1769472
                            // 147456,
                        )
                        ax2 = T.axis.spatial(
                            384,
                            (
                                (i0_i1_i2_i3_fused_0 * 256 + i0_i1_i2_i3_fused_1) * 1024
                                + i0_i1_i2_i3_fused_2
                            )
                            % 147456
                            // 384,
                        )
                        ax3 = T.axis.spatial(
                            384,
                            (
                                (i0_i1_i2_i3_fused_0 * 256 + i0_i1_i2_i3_fused_1) * 1024
                                + i0_i1_i2_i3_fused_2
                            )
                            % 384,
                        )
                        T.where(
                            (i0_i1_i2_i3_fused_0 * 256 + i0_i1_i2_i3_fused_1) * 1024
                            + i0_i1_i2_i3_fused_2
                            < 1769472
                        )
                        T.reads(placeholder_1[ax0, ax1, ax2, ax3], T_reshape[ax0, ax1, ax2, ax3])
                        T.writes(T_where[ax0, ax1, ax2, ax3])
                        T_where[ax0, ax1, ax2, ax3] = T.Select(
                            T.Cast("int32", placeholder_1[ax0, ax1, ax2, ax3]) != 0,
                            T.float32(-1000000000),
                            T_reshape[ax0, ax1, ax2, ax3],
                        )

    mod = tvm.IRModule.from_expr(before)
    func = tvm.tir.transform.ForceNarrowIndexToInt32()(mod)["main"]
    tvm.ir.assert_structural_equal(func, expected)


def test_block():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.serial(0, T.int64(16)):
            for j in T.serial(0, T.int64(8)):
                with T.block():
                    vi = T.axis.spatial(T.int64(128), i * T.int64(8) + j)
                    B[vi] = A[vi] + T.float32(1)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.serial(0, T.int32(16)):
            for j in T.serial(0, T.int32(8)):
                with T.block():
                    vi = T.axis.spatial(T.int32(128), i * T.int32(8) + j)
                    B[vi] = A[vi] + T.float32(1)

    mod = tvm.IRModule.from_expr(before)
    func = tvm.tir.transform.ForceNarrowIndexToInt32()(mod)["main"]
    tvm.ir.assert_structural_equal(func, expected)


def test_i16_buffer():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128,), "int16"), B: T.Buffer((128,), "int16")):
        for i in T.serial(0, T.int64(16)):
            for j in T.serial(0, T.int64(16)):
                with T.block():
                    vi = T.axis.spatial(T.int64(128), i * 8 + j)
                    B[vi] = A[vi] + T.int16(1)

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128,), "int16"), B: T.Buffer((128,), "int16")):
        for i in T.serial(0, 16):
            for j in T.serial(0, 16):
                with T.block():
                    vi = T.axis.spatial(128, i * 8 + j)
                    B[vi] = A[vi] + T.int16(1)

    mod = tvm.IRModule.from_expr(before)
    after = tvm.tir.transform.ForceNarrowIndexToInt32()(mod)["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_fail_on_buffer_map():
    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int64"), B: T.Buffer((128,), "int64")):
        for i in T.serial(0, 16):
            for j in T.serial(0, 8):
                with T.block():
                    vi = T.axis.spatial(128, i * 8 + j)
                    B[vi] = A[vi] + T.int64(1)

    mod = tvm.IRModule.from_expr(func)
    with pytest.raises(TVMError):
        tvm.tir.transform.ForceNarrowIndexToInt32()(mod)["main"]


def test_fail_on_buffer_map():
    @T.prim_func(private=True)
    def func(A: T.Buffer((128,), "int32"), B: T.Buffer((128,), "int32")):
        C = T.alloc_buffer((128,), "int64")
        for i in T.serial(0, 16):
            for j in T.serial(0, 8):
                with T.block():
                    vi = T.axis.spatial(128, i * 8 + j)
                    C[vi] = T.cast(A[vi], "int64") + T.int64(1)
        for i in T.serial(0, 16):
            for j in T.serial(0, 8):
                with T.block():
                    vi = T.axis.spatial(128, i * 8 + j)
                    B[vi] = T.cast(C[vi] + T.int64(1), "int32")

    mod = tvm.IRModule.from_expr(func)
    with pytest.raises(TVMError):
        tvm.tir.transform.ForceNarrowIndexToInt32()(mod)["main"]


def test_pod_params_and_select():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            A: T.Buffer((T.int64(4),), "float32"), B: T.Buffer((T.int64(4),), "float32"), n: T.int64
        ):
            for i in T.serial(T.int64(4)):
                B[i] = T.Select(T.int64(1) <= i, A[i + n], T.Cast("float32", i))

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32"), n: T.int32):
            for i in range(4):
                B[i] = T.Select(1 <= i, A[i + n], T.Cast("float32", i))

    after = tvm.tir.transform.ForceNarrowIndexToInt32()(Before)
    tvm.ir.assert_structural_equal(Expected, after)


if __name__ == "__main__":
    tvm.testing.main()
