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
from tvm import tir
from tvm.script import ty

# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def elementwise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
        B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_symbolic(a: ty.handle, b: ty.handle, n: ty.int32) -> None:
    A = tir.match_buffer(a, (128, 128, n))
    B = tir.match_buffer(b, (128, 128, n))
    for i, j, k in tir.grid(128, 128, n):
        with tir.block([128, 128, n], "B") as [vi, vj, vk]:
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_symbolic_fused(a: ty.handle, b: ty.handle, n: ty.int32) -> None:
    A = tir.match_buffer(a, (128, 128, n))
    B = tir.match_buffer(b, (128, 128, n))
    for i_j_k_fused in tir.serial(0, (n * 16384)):
        with tir.block([128, 128, n], "B") as [vi, vj, vk]:
            tir.bind(vi, tir.floordiv(i_j_k_fused, (n * 128)))
            tir.bind(vj, tir.floormod(tir.floordiv(i_j_k_fused, n), 128))
            tir.bind(vk, tir.floormod(i_j_k_fused, n))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_symbolic_split(a: ty.handle, b: ty.handle, n: ty.int32) -> None:
    A = tir.match_buffer(a, (128, 128, n))
    B = tir.match_buffer(b, (128, 128, n))
    for i, j, k0, k1 in tir.grid(128, 128, 10, tir.floordiv((n + 9), 10)):
        with tir.block([128, 128, n], "B") as [vi, vj, vk]:
            tir.where((((k0 * tir.floordiv((n + 9), 10)) + k1) < n))
            tir.bind(vi, i)
            tir.bind(vj, j)
            tir.bind(vk, ((k0 * tir.floordiv((n + 9), 10)) + k1))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_seq(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    C = tir.alloc_buffer((128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.serial(0, 128):
            with tir.block([128, 128, 128], "C") as [vi, vj, vk]:
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in tir.serial(0, 128):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_anno(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.serial(0, 128, annotations={"useless_annotation": True}):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_thread_binding(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.thread_binding(0, 128, thread="threadIdx.x"):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_starting_point(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.serial(10, 128):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_opaque_block(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j, k in tir.grid(128, 128, 128):
        with tir.block([], "opaque"):
            tir.reads([A[i, j, k]])
            tir.writes([B[i, j, k]])
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_fused(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for fused in tir.serial(0, 2097152):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.bind(vi, tir.floordiv(fused, 16384))
            tir.bind(vj, tir.floormod(tir.floordiv(fused, 128), 128))
            tir.bind(vk, tir.floormod(fused, 128))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_split_case0(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128, 128])
    B = tir.match_buffer(b, [128, 128, 128])
    for i1, i2, i3, j1, j2, k1, k2 in tir.grid(2, 1, 64, 4, 32, 16, 8):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.bind(vi, ((i1 * 64) + i3))
            tir.bind(vj, ((j1 * 32) + j2))
            tir.bind(vk, ((k1 * 8) + k2))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_split_case1(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128, 128])
    B = tir.match_buffer(b, [128, 128, 128])
    for i1, i2, i3, j1, j2, j3, k1, k2, k3 in tir.grid(2, 1, 64, 2, 1, 64, 2, 1, 64):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.bind(vi, i1 * 64 + i3)
            tir.bind(vj, j1 * 64 + j3)
            tir.bind(vk, k1 * 64 + k3)
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_split_with_predicate(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 128, 128])
    A = tir.match_buffer(a, [128, 128, 128])
    for i0, i1, i2, j0, j1, k0, k1 in tir.grid(1000, 2, 3, 1, 129, 3, 43):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.where(
                (
                    ((((((i0 * 2) + i1) * 3) + i2) < 128) and (((j0 * 129) + j1) < 128))
                    and (((k0 * 43) + k1) < 128)
                )
            )
            tir.bind(vi, (((i0 * 6) + (i1 * 3)) + i2))
            tir.bind(vj, j1)
            tir.bind(vk, ((k0 * 43) + k1))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_fuse_with_opaque_block(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 128, 128])
    A = tir.match_buffer(a, [128, 128, 128])
    for i_j_k_fused in tir.serial(0, 2097152):
        with tir.block([], "opaque"):
            tir.reads(
                [
                    A[
                        tir.floormod(tir.floordiv(tir.floordiv(i_j_k_fused, 128), 128), 128),
                        tir.floormod(tir.floordiv(i_j_k_fused, 128), 128),
                        tir.floormod(i_j_k_fused, 128),
                    ]
                ]
            )
            tir.writes(
                [
                    B[
                        tir.floormod(tir.floordiv(tir.floordiv(i_j_k_fused, 128), 128), 128),
                        tir.floormod(tir.floordiv(i_j_k_fused, 128), 128),
                        tir.floormod(i_j_k_fused, 128),
                    ]
                ]
            )
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, tir.floordiv(i_j_k_fused, 16384))
                tir.bind(vj, tir.floormod(tir.floordiv(i_j_k_fused, 128), 128))
                tir.bind(vk, tir.floormod(i_j_k_fused, 128))
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_split_with_opaque_block(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 128, 128])
    A = tir.match_buffer(a, [128, 128, 128])

    for i0, i1, j, k in tir.grid(8, 16, 128, 128):
        with tir.block([], "opaque"):
            tir.reads([A[i0 * 16 + i1, j, k]])
            tir.writes([B[i0 * 16 + i1, j, k]])
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i0 * 16 + i1)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16], "float32")
    B = tir.match_buffer(b, [16, 16], "float32")
    with tir.block([16, 16], "A") as [vi, vj]:
        tir.reads([])
        tir.writes([A[0:16, 0:16]])
        tir.store(A.data, vi * 16 + vj, 1)
    with tir.block([16, 16], "B") as [vi, vj]:
        tir.reads([])
        tir.writes([B[0:16, 0:16]])
        tir.evaluate(tir.tvm_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj, dtype="handle"))


@tvm.script.tir
def opaque_access_fused(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16])
    B = tir.match_buffer(b, [16, 16])
    for i_j_fused in tir.serial(0, 256):
        with tir.block([16, 16], "A") as [vi, vj]:
            tir.bind(vi, tir.floordiv(i_j_fused, 16))
            tir.bind(vj, tir.floormod(i_j_fused, 16))
            tir.reads([])
            tir.writes([A[0:16, 0:16]])
            tir.store(A.data, ((vi * 16) + vj), 1, 1)
    for i_j_fused in tir.serial(0, 256):
        with tir.block([16, 16], "B") as [vi, vj]:
            tir.bind(vi, tir.floordiv(i_j_fused, 16))
            tir.bind(vj, tir.floormod(i_j_fused, 16))
            tir.reads([])
            tir.writes([B[0:16, 0:16]])
            tir.evaluate(
                tir.tvm_fill_fragment(B.data, 16, 16, 16, 0, ((vi * 16) + vj), dtype="handle")
            )


@tvm.script.tir
def opaque_access_split(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16))
    B = tir.match_buffer(b, (16, 16))
    for i, j0, j1 in tir.grid(16, 4, 4):
        with tir.block([16, 16], "A") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, ((j0 * 4) + j1))
            tir.reads([])
            tir.writes([A[0:16, 0:16]])
            tir.store(A.data, ((vi * 16) + vj), 1, 1)
    for i, j0, j1 in tir.grid(16, 4, 4):
        with tir.block([16, 16], "B") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, ((j0 * 4) + j1))
            tir.reads([])
            tir.writes([B[0:16, 0:16]])
            tir.evaluate(
                tir.tvm_fill_fragment(B.data, 16, 16, 16, 0, ((vi * 16) + vj), dtype="handle")
            )


# pylint: enable=no-member,invalid-name,unused-variable


def test_fuse():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.fuse(i, j, k)
    tvm.ir.assert_structural_equal(elementwise_fused, sch.mod["main"])


def test_split():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[2, 1, 64])
    sch.split(j, factors=[4, 32])
    sch.split(k, factors=[16, 8])
    tvm.ir.assert_structural_equal(elementwise_split_case0, sch.mod["main"])


def test_split_with_inferred_factor():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[None, 1, 64])
    sch.split(j, factors=[2, None, 64])
    sch.split(k, factors=[2, 1, None])
    tvm.ir.assert_structural_equal(elementwise_split_case1, sch.mod["main"])


def test_split_with_predicate():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[1000, 2, 3])
    sch.split(j, factors=[None, 129])
    sch.split(k, factors=[3, None])
    tvm.ir.assert_structural_equal(elementwise_split_with_predicate, sch.mod["main"])


def test_fuse_fail_not_only_child():
    sch = tir.Schedule(elementwise_with_seq, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)


def test_fuse_split_fail_with_annotation():
    sch = tir.Schedule(elementwise_with_anno, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_split_fail_not_start_with_zero():
    sch = tir.Schedule(elementwise_with_anno, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_with_opaque_block():
    sch = tir.Schedule(elementwise_with_opaque_block, debug_mode=True)
    block_opaque = sch.get_block("opaque")
    i, j, k = sch.get_loops(block_opaque)
    sch.fuse(i, j, k)
    tvm.ir.assert_structural_equal(elementwise_fuse_with_opaque_block, sch.mod["main"])


def test_fuse_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mode=True)
    block_a = sch.get_block("A")
    i, j = sch.get_loops(block_a)
    sch.fuse(i, j)
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.fuse(i, j)
    tvm.ir.assert_structural_equal(opaque_access_fused, sch.mod["main"])


def test_split_with_opaque_block():
    sch = tir.Schedule(elementwise_with_opaque_block, debug_mode=True)
    block_opaque = sch.get_block("opaque")
    i, j, k = sch.get_loops(block_opaque)
    sch.split(i, factors=[None, 16])
    tvm.ir.assert_structural_equal(elementwise_split_with_opaque_block, sch.mod["main"])


def test_split_with_opaque_access():
    sch = tir.Schedule(opaque_access, debug_mode=True)
    block_a = sch.get_block("A")
    i, j = sch.get_loops(block_a)
    sch.split(j, factors=[None, 4])
    block_b = sch.get_block("B")
    i, j = sch.get_loops(block_b)
    sch.split(j, factors=[None, 4])
    tvm.ir.assert_structural_equal(opaque_access_split, sch.mod["main"])


def test_fuse_split_fail_with_thread_binding():
    sch = tir.Schedule(elementwise_with_thread_binding, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_symbolic():
    sch = tir.Schedule(elementwise_symbolic, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.fuse(i, j, k)
    tvm.ir.assert_structural_equal(elementwise_symbolic_fused, sch.mod["main"])


def test_split_symbolic():
    sch = tir.Schedule(elementwise_symbolic, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(k, factors=[10, None])
    tvm.ir.assert_structural_equal(elementwise_symbolic_split, sch.mod["main"])


if __name__ == "__main__":
    pytest.main([__file__])
