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
from tvm import te, tir
from tvm.script import tir as T
from tvm.tir.expr import IntImm
from tvm.tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def elementwise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_symbolic(a: T.handle, b: T.handle, n: T.int32) -> None:
    A = T.match_buffer(a, (128, 128, n))
    B = T.match_buffer(b, (128, 128, n))
    for i, j, k in T.grid(128, 128, n):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_anno(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128, annotations={"useless_annotation": True}):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_thread_binding(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j in T.grid(128, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_with_opaque_block(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128, 128))
    B = T.match_buffer(b, (128, 128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("opaque"):
            T.reads([A[i, j, k]])
            T.writes([B[i, j, k]])
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                T.reads([A[vi, vj, vk]])
                T.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@T.prim_func
def elementwise_partition_with_opaque_block(a: T.handle, b: T.handle) -> None:
    B = T.match_buffer(b, [128, 128, 128])
    A = T.match_buffer(a, [128, 128, 128])
    with T.block("root"):
        T.reads()
        T.writes()
        with T.block("opaque_i_common"):
            T.reads()
            T.writes()
            with T.block("opaque_i0_partition"):
                T.reads()
                T.writes()
                for i0, j, k in T.grid(112, 128, 128):
                    with T.block("opaque_i0"):
                        T.reads(A[i0, j, k])
                        T.writes(B[i0, j, k])
                        with T.block("B_i0"):
                            vi, vj, vk = T.axis.remap("SSS", [i0, j, k])
                            T.reads(A[0:112, 0:128, 0:128])
                            T.writes(B[0:112, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with T.block("opaque_i1_partition"):
                T.reads()
                T.writes()
                for i1 in range(112, 128):
                    for j, k in T.grid(128, 128):
                        with T.block("opaque_i1"):
                            T.reads(A[i1, j, k])
                            T.writes(B[i1, j, k])
                            with T.block("B_i1"):
                                vi, vj, vk = T.axis.remap("SSS", [i1, j, k])
                                T.reads(A[112:128, 0:128, 0:128])
                                T.writes(B[112:128, 0:128, 0:128])
                                B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)


@T.prim_func
def elementwise_loop_partition_case0(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128, 128])
    B = T.match_buffer(b, [128, 128, 128])
    with T.block("root"):
        T.reads()
        T.writes()
        with T.block("B_i_common"):
            T.reads()
            T.writes()
            with T.block("B_i0_partition"):
                T.reads()
                T.writes()
                for i0 in range(2):
                    with T.block("B_i0_j_common"):
                        T.reads()
                        T.writes()
                        with T.block("B_i0_j0_partition"):
                            T.reads()
                            T.writes()
                            for j0, k in T.grid(4, 128):
                                with T.block("B_i0_j0"):
                                    vi, vj, vk = T.axis.remap("SSS", [i0, j0, k])
                                    T.reads(A[0:2, 0:4, 0:128])
                                    T.writes(B[0:2, 0:4, 0:128])
                                    B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                        with T.block("B_i0_j1_partition"):
                            T.reads()
                            T.writes()
                            for j1 in range(4, 36):
                                for k in range(128):
                                    with T.block("B_i0_j1"):
                                        vi, vj, vk = T.axis.remap("SSS", [i0, j1, k])
                                        T.reads(A[0:2, 4:36, 0:128])
                                        T.writes(B[0:2, 4:36, 0:128])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                        with T.block("B_i0_j2_partition"):
                            T.reads()
                            T.writes()
                            for j2 in range(36, 128):
                                for k in range(128):
                                    with T.block("B_i0_j2"):
                                        vi, vj, vk = T.axis.remap("SSS", [i0, j2, k])
                                        T.reads(A[0:2, 36:128, 0:128])
                                        T.writes(B[0:2, 36:128, 0:128])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with T.block("B_i1_partition"):
                T.reads()
                T.writes()
                for i1 in range(2, 3):
                    for j, k in T.grid(128, 128):
                        with T.block("B_i1"):
                            vi, vj, vk = T.axis.remap("SSS", [i1, j, k])
                            T.reads(A[2, 0:128, 0:128])
                            T.writes(B[2, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with T.block("B_i2_partition"):
                T.reads()
                T.writes()
                for i2 in range(3, 67):
                    for j, k in T.grid(128, 128):
                        with T.block("B_i2"):
                            vi, vj, vk = T.axis.remap("SSS", [i2, j, k])
                            T.reads(A[3:67, 0:128, 0:128])
                            T.writes(B[3:67, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with T.block("B_i3_partition"):
                T.reads()
                T.writes()
                for i3 in range(67, 128):
                    for j, k in T.grid(128, 128):
                        with T.block("B_i3"):
                            vi, vj, vk = T.axis.remap("SSS", [i3, j, k])
                            T.reads(A[67:128, 0:128, 0:128])
                            T.writes(B[67:128, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)


@T.prim_func
def elementwise_loop_partition_case1(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128, 128])
    B = T.match_buffer(b, [128, 128, 128])
    with T.block("root"):
        T.reads()
        T.writes()
        with T.block("B_i_common"):
            T.reads()
            T.writes()
            with T.block("B_i0_partition"):
                T.reads()
                T.writes()
                for i0, j, k in T.grid(63, 128, 128):
                    with T.block("B_i0"):
                        vi, vj, vk = T.axis.remap("SSS", [i0, j, k])
                        T.reads(A[0:63, 0:128, 0:128])
                        T.writes(B[0:63, 0:128, 0:128])
                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with T.block("B_i1_partition"):
                T.reads()
                T.writes()
                for i1 in range(63, 64):
                    for j in range(128):
                        with T.block("B_i1_k_common"):
                            T.reads()
                            T.writes()
                            with T.block("B_i1_k0_partition"):
                                T.reads()
                                T.writes()
                                for k0 in range(1):
                                    with T.block("B_i1_k0"):
                                        vi, vj, vk = T.axis.remap("SSS", [i1, j, k0])
                                        T.reads(A[63, 0:128, 0])
                                        T.writes(B[63, 0:128, 0])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                            with T.block("B_i1_k1_partition"):
                                T.reads()
                                T.writes()
                                for k1 in range(1, 65):
                                    with T.block("B_i1_k1"):
                                        vi, vj, vk = T.axis.remap("SSS", [i1, j, k1])
                                        T.reads(A[63, 0:128, 1:65])
                                        T.writes(B[63, 0:128, 1:65])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                            with T.block("B_i1_k2_partition"):
                                T.reads()
                                T.writes()
                                for k2 in range(65, 128):
                                    with T.block("B_i1_k2"):
                                        vi, vj, vk = T.axis.remap("SSS", [i1, j, k2])
                                        T.reads(A[63, 0:128, 65:128])
                                        T.writes(B[63, 0:128, 65:128])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with T.block("B_i2_partition"):
                T.reads()
                T.writes()
                for i2 in range(64, 128):
                    for j, k in T.grid(128, 128):
                        with T.block("B_i2"):
                            vi, vj, vk = T.axis.remap("SSS", [i2, j, k])
                            T.reads(A[64:128, 0:128, 0:128])
                            T.writes(B[64:128, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)


@T.prim_func
def opaque_access(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16], "float32")
    B = T.match_buffer(b, [16, 16], "float32")
    for i, j in T.grid(16, 16):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i, j in T.grid(16, 16):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([])
            T.writes([B[0:16, 0:16]])
            T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj, dtype="handle"))


@T.prim_func
def opaque_access_loop_partition(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (16, 16))
    B = T.match_buffer(b, (16, 16))
    for i in range(16):
        with T.block("A_j_common"):
            T.reads()
            T.writes()
            with T.block("A_j0_partition"):
                T.reads()
                T.writes()
                for j0 in range(12):
                    with T.block("A_j0"):
                        vi, vj = T.axis.remap("SS", [i, j0])
                        T.reads()
                        T.writes(A[0:16, 0:12])
                        A[vi, vj] = T.float32(1)
            with T.block("A_j1_partition"):
                T.reads()
                T.writes()
                for j1 in range(12, 16):
                    with T.block("A_j1"):
                        vi, vj = T.axis.remap("SS", [i, j1])
                        T.reads()
                        T.writes(A[0:16, 12:16])
                        A[vi, vj] = T.float32(1)
    for i in range(16):
        with T.block("B_j_common"):
            T.reads()
            T.writes()
            with T.block("B_j0_partition"):
                T.reads()
                T.writes()
                for j0 in range(12):
                    with T.block("B_j0"):
                        vi, vj = T.axis.remap("SS", [i, j0])
                        T.reads()
                        T.writes(B[0:16, 0:16])
                        T.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj)
            with T.block("B_j1_partition"):
                T.reads()
                T.writes()
                for j1 in range(12, 16):
                    with T.block("B_j1"):
                        vi, vj = T.axis.remap("SS", [i, j1])
                        T.reads()
                        T.writes(B[0:16, 0:16])
                        T.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj)


# pylint: enable=no-member,invalid-name,unused-variable


def test_loop_partition():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.loop_partition(i, factors=[2, 1, 64])

    block_b_partition = sch.get_block("B_i0")
    i, j, k = sch.get_loops(block_b_partition)
    loops = sch.loop_partition(j, factors=[4, 32])

    assert_structural_equal_ignore_global_symbol(elementwise_loop_partition_case0, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_partition_with_inferred_factor():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.loop_partition(i, factors=[None, 1, 64])

    block_b_partition = sch.get_block("B_i1")
    i, j, k = sch.get_loops(block_b_partition)
    sch.loop_partition(k, factors=[1, 64, None])

    assert_structural_equal_ignore_global_symbol(elementwise_loop_partition_case1, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_partition_with_opaque_block():
    sch = tir.Schedule(elementwise_with_opaque_block, debug_mask="all")
    block_opaque = sch.get_block("opaque")
    i, _, _ = sch.get_loops(block_opaque)
    sch.loop_partition(i, factors=[None, 16])
    assert_structural_equal_ignore_global_symbol(
        elementwise_partition_with_opaque_block, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_with_opaque_block)


def test_partition_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_block("A")
    _, j = sch.get_loops(block_a)
    sch.loop_partition(j, factors=[None, 4])
    block_b = sch.get_block("B")
    _, j = sch.get_loops(block_b)
    sch.loop_partition(j, factors=[None, 4])
    assert_structural_equal_ignore_global_symbol(opaque_access_loop_partition, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_partition_int64_extent_with_mixed_factors():
    def _create_prim_func():
        m = te.const(384, "int64")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")
        return te.create_prim_func([A, B])

    mod = _create_prim_func()
    sch = tir.Schedule(mod, debug_mask="all")
    (i,) = sch.get_loops(sch.get_block("B"))
    sch.loop_partition(
        i,
        factors=[
            te.const(1, "int64"),
            te.const(51, "int32"),
        ],
    )


def test_partition_fail_symbolic():
    sch = tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_block("B")
    _, _, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[10, None])


def test_partition_fail_out_of_bound():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(i, factors=[1000, 2, 3])


def test_partition_with_non_positive_factors():
    sch = tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(i, factors=[-2, -64])
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(j, factors=[0, None])
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[None, -16])


def test_partition_fail_with_annotation():
    sch = tir.Schedule(elementwise_with_anno, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[None, 10])


def test_partition_fail_with_thread_binding():
    sch = tir.Schedule(elementwise_with_thread_binding, debug_mask="all")
    block_b = sch.get_block("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.loop_partition(k, factors=[None, 10])


if __name__ == "__main__":
    tvm.testing.main()
