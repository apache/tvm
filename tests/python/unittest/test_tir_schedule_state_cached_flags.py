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
import sys

import pytest
import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.state import CachedFlags
from tvm.tir.stmt_functor import post_order_visit

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    with T.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = 0.0
        for k in range(0, 128):
            with T.block([128, 128, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def block_in_opaque_block(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.match_buffer(b, (128, 128), "float32")
    with T.block([128], "B") as vi:
        T.reads([A[0:128, 0:128]])
        T.writes([B[0:128, 0:128]])
        B[vi, 0] = A[vi, 0]
        if A[vi, 0] == 0.0:
            with T.block([], "C"):
                T.reads([A[0:128, 0:128]])
                T.writes([B[0:128, 0:128]])
                with T.block([128], "D") as vj:
                    B[vi, vj] = A[vi, vj] * 3.0
        else:
            with T.block([], "E"):
                T.reads([A[0:128, 0:128]])
                T.writes([B[0:128, 0:128]])
                with T.block([128], "F") as vj:
                    B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def write_after_read(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0
    with T.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def loop_carried_dependency(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    C = T.match_buffer(c, (128,))
    for i in range(0, 128):
        with T.block([128], "B") as vi:
            B[vi] = A[vi] * 2.0
        with T.block([128], "C") as vi:
            C[vi] = T.if_then_else(vi >= 1, B[vi - 1] + 1.0, 0.0, dtype="float32")


@T.prim_func
def concatenate_multi_producer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    for i in range(0, 64):
        with T.block([64], "A_0") as vi:
            A[vi] = vi + 1
    for i in range(0, 64):
        with T.block([64], "A_1") as vi:
            T.bind(vi, i + 64)
            A[vi] = vi + 2
    with T.block([128], "B") as vi:
        B[vi] = A[vi] * 2.0


@T.prim_func
def concatenate_multi_producer_uncovered(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    for i in range(0, 63):
        with T.block([63], "A_0") as vi:
            A[vi] = vi + 1
    for i in range(0, 64):
        with T.block([64], "A_1") as vi:
            T.bind(vi, i + 64)
            A[vi] = vi + 2
    with T.block([128], "B") as vi:
        B[vi] = A[vi] * 2.0


@T.prim_func
def lca_at_loop(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    C = T.match_buffer(c, (128,))
    for i in range(0, 128):
        with T.block([128], "B") as vi:
            B[vi] = A[vi] * 2.0
        with T.block([128], "C") as vi:
            C[vi] = B[vi] + 1.0


@T.prim_func
def multi_producer_consumer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    for i in range(0, 64):
        with T.block([64], "A_0") as vi:
            A[vi] = vi + 1
    for i in range(0, 64):
        with T.block([64], "A_1") as vi:
            T.bind(vi, i + 64)
            A[vi] = vi + 2
    for i in range(0, 64):
        with T.block([64], "B_0") as vi:
            B[vi] = A[vi] + 2.0
    for i in range(0, 64):
        with T.block([64], "B_1") as vi:
            T.bind(vi, i + 64)
            B[vi] = A[vi] + 3.0


@T.prim_func
def elementwise_affine_producer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    for i, j, k, l in T.grid(16, 2, 32, 16):
        with T.block([128, 128], "B") as [vi, vj]:
            T.bind(vi, i * 8 + j * 4 + k // 8)
            T.bind(vj, k % 8 * 16 + l)
            B[vi, vj] = A[vi, vj] * 2.0
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_subblock(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    with T.block([32, 32], "B") as [vi, vj]:
        T.reads([A[vi * 4 : vi * 4 + 4, vj * 4 : vj * 4 + 4]])
        T.writes([B[vi * 4 : vi * 4 + 4, vj * 4 : vj * 4 + 4]])
        with T.block([4, 4], "B_sub") as [vi_i, vj_i]:
            B[vi * 4 + vi_i, vj * 4 + vj_i] = A[vi * 4 + vi_i, vj * 4 + vj_i] * 2.0
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_subblock_uncovered(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    with T.block([32, 32], "B") as [vi, vj]:
        T.reads([A[vi * 4 : vi * 4 + 2, vj * 4 : vj * 4 + 2]])
        T.writes([B[vi * 4 : vi * 4 + 2, vj * 4 : vj * 4 + 2]])
        with T.block([2, 2], "B_sub") as [vi_i, vj_i]:
            B[vi * 4 + vi_i, vj * 4 + vj_i] = A[vi * 4 + vi_i, vj * 4 + vj_i] * 2.0
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def bound_to_thread(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 128], scope="shared")
    for i in T.thread_binding(0, 128, thread="threadIdx.x"):
        for j in T.serial(0, 128):
            with T.block([128, 128], "B") as [vi, vj]:
                B[vi, vj] = A[vi, vj] * 2.0
        for j in T.serial(0, 128):
            with T.block([128, 128], "C") as [vi, vj]:
                C[vj, vi] = B[vj, vi] + 1.0


@T.prim_func
def equal_ranked_threads(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 128], scope="shared")
    for i_o in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i_i in T.thread_binding(0, 8, thread="threadIdx.y"):
            for j in T.serial(0, 128):
                with T.block([128, 128], "B") as [vi, vj]:
                    T.bind(vi, i_o * 8 + i_i)
                    T.bind(vj, j)
                    B[vi, vj] = A[vi, vj] * 2.0
            for j in T.serial(0, 128):
                with T.block([128, 128], "C") as [vi, vj]:
                    T.bind(vi, i_o * 8 + i_i)
                    T.bind(vj, j)
                    C[vj, vi] = B[vj, vi] + 1.0


@T.prim_func
def warp_memory(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 4, 32], scope="warp")
    for i_o in T.thread_binding(0, 4, thread="threadIdx.y"):
        for i_i in T.thread_binding(0, 32, thread="threadIdx.x"):
            for j in T.serial(0, 128):
                with T.block([4, 32, 128], "B") as [warp_id, lane_id, vj]:
                    B[vj, warp_id, lane_id] = A[warp_id * 32 + lane_id, vj] * 2.0
            for j in T.serial(0, 128):
                with T.block([4, 32, 128], "C") as [warp_id, lane_id, vj]:
                    C[warp_id * 32 + lane_id, vj] = B[vj, warp_id, lane_id] + 1.0


@T.prim_func
def warp_memory_negative(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 4, 32], scope="warp")
    for i_o in T.thread_binding(0, 4, thread="threadIdx.y"):
        for i_i in T.thread_binding(0, 32, thread="threadIdx.x"):
            for j in T.serial(0, 128):
                with T.block([4, 32, 128], "B") as [warp_id, lane_id, vj]:
                    B[vj, warp_id, lane_id] = A[warp_id * 32 + lane_id, vj] * 2.0
            for i_o_prime in T.thread_binding(0, 4, thread="threadIdx.y"):
                for j in T.serial(0, 128):
                    with T.block([4, 32, 4, 128], "C") as [_warp_id, lane_id, warp_id, vj]:
                        C[warp_id * 32 + lane_id, vj] = B[vj, warp_id, lane_id] + 1.0


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def _get_block(s: tir.ScheduleState, name_hint: str) -> tir.StmtSRef:
    result = None

    def f_visit(node):
        nonlocal result
        if isinstance(node, tvm.tir.Block) and node.name_hint == name_hint:
            result = node

    func = s.mod["main"]
    post_order_visit(func.body, f_visit)
    assert result is not None and isinstance(result, tvm.tir.Block)
    return s.get_sref(result)


def test_elementwise():
    s = tir.ScheduleState(elementwise, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_matmul():
    s = tir.ScheduleState(matmul, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "init")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "update")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_block_in_opaque_block():
    s = tir.ScheduleState(block_in_opaque_block, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "E")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "F")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_write_after_read():
    s = tir.ScheduleState(write_after_read, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=False,
    )
    # pylint: enable=protected-access


def test_loop_carried_dependency():
    s = tir.ScheduleState(loop_carried_dependency, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=False,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=False,
    )
    # pylint: enable=protected-access


def test_concatenate_multi_producer_covered():  # pylint: disable=invalid-name
    s = tir.ScheduleState(concatenate_multi_producer, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "A_0")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "A_1")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_concatenate_multi_producer_uncovered():  # pylint: disable=invalid-name
    s = tir.ScheduleState(concatenate_multi_producer_uncovered, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "A_0")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "A_1")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=False,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=False,
    )
    # pylint: enable=protected-access


def test_lca_at_loop():
    s = tir.ScheduleState(lca_at_loop, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_multi_producer_consumer():
    s = tir.ScheduleState(multi_producer_consumer, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "A_0")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "A_1")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B_0")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B_1")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_elementwise_affine_producer():
    s = tir.ScheduleState(elementwise_affine_producer, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_subblock():
    s = tir.ScheduleState(elementwise_subblock, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B_sub")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_subblock_uncovered():
    s = tir.ScheduleState(elementwise_subblock_uncovered, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=False,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B_sub")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=False,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_thread_binding():
    s = tir.ScheduleState(bound_to_thread, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_equal_ranked_threads():
    s = tir.ScheduleState(equal_ranked_threads, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_warp_memory():
    s = tir.ScheduleState(warp_memory, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_warp_memory_negative():
    s = tir.ScheduleState(warp_memory_negative, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=False,
    )
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=False,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
