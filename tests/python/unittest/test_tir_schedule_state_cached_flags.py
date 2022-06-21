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
import tvm.testing
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
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = 0.0
        for k in range(0, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def block_in_opaque_block(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.match_buffer(b, (128, 128), "float32")
    for i in range(128):
        with T.block("B"):
            vi = T.axis.S(128, i)
            T.reads([A[0:128, 0:128]])
            T.writes([B[0:128, 0:128]])
            B[vi, 0] = A[vi, 0]
            if A[vi, 0] == 0.0:
                with T.block("C"):
                    T.reads([A[0:128, 0:128]])
                    T.writes([B[0:128, 0:128]])
                    for j in range(128):
                        with T.block("D"):
                            vj = T.axis.S(128, j)
                            B[vi, vj] = A[vi, vj] * 3.0
            else:
                with T.block("E"):
                    T.reads([A[0:128, 0:128]])
                    T.writes([B[0:128, 0:128]])
                    for j in range(128):
                        with T.block("F"):
                            vj = T.axis.S(128, j)
                            B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def write_after_read(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def loop_carried_dependency(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    C = T.match_buffer(c, (128,))
    for i in range(0, 128):
        with T.block("B"):
            vi = T.axis.S(128, i)
            B[vi] = A[vi] * 2.0
        with T.block("C"):
            vi = T.axis.S(128, i)
            C[vi] = T.if_then_else(vi >= 1, B[vi - 1] + 1.0, 0.0, dtype="float32")


@T.prim_func
def concatenate_multi_producer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    for i in range(0, 64):
        with T.block("A_0"):
            vi = T.axis.S(64, i)
            A[vi] = vi + 1
    for i in range(0, 64):
        with T.block("A_1"):
            vi = T.axis.S(64, i + 64)
            A[vi] = vi + 2
    for i in range(0, 128):
        with T.block("B"):
            vi = T.axis.S(128, i)
            B[vi] = A[vi] * 2.0


@T.prim_func
def concatenate_multi_producer_uncovered(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    for i in range(0, 63):
        with T.block("A_0"):
            vi = T.axis.S(63, i)
            A[vi] = vi + 1
    for i in range(0, 64):
        with T.block("A_1"):
            vi = T.axis.S(64, i + 64)
            A[vi] = vi + 2
    for i in range(0, 128):
        with T.block("B"):
            vi = T.axis.S(128, i)
            B[vi] = A[vi] * 2.0


@T.prim_func
def lca_at_loop(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    C = T.match_buffer(c, (128,))
    for i in range(0, 128):
        with T.block("B"):
            vi = T.axis.S(128, i)
            B[vi] = A[vi] * 2.0
        with T.block("C"):
            vi = T.axis.S(128, i)
            C[vi] = B[vi] + 1.0


@T.prim_func
def multi_producer_consumer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    for i in range(0, 64):
        with T.block("A_0"):
            vi = T.axis.S(64, i)
            A[vi] = vi + 1
    for i in range(0, 64):
        with T.block("A_1"):
            vi = T.axis.S(64, i + 64)
            A[vi] = vi + 2
    for i in range(0, 64):
        with T.block("B_0"):
            vi = T.axis.S(64, i)
            B[vi] = A[vi] + 2.0
    for i in range(0, 64):
        with T.block("B_1"):
            vi = T.axis.S(64, i + 64)
            B[vi] = A[vi] + 3.0


@T.prim_func
def elementwise_affine_producer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    for i, j, k, l in T.grid(16, 2, 32, 16):
        with T.block("B"):
            vi = T.axis.S(128, i * 8 + j * 4 + k // 8)
            vj = T.axis.S(128, k % 8 * 16 + l)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_subblock(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(32, 32):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([A[vi * 4 : vi * 4 + 4, vj * 4 : vj * 4 + 4]])
            T.writes([B[vi * 4 : vi * 4 + 4, vj * 4 : vj * 4 + 4]])
            for ii, jj in T.grid(4, 4):
                with T.block("B_sub"):
                    vi_i, vj_i = T.axis.remap("SS", [ii, jj])
                    B[vi * 4 + vi_i, vj * 4 + vj_i] = A[vi * 4 + vi_i, vj * 4 + vj_i] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def elementwise_subblock_uncovered(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(32, 32):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads([A[vi * 4 : vi * 4 + 2, vj * 4 : vj * 4 + 2]])
            T.writes([B[vi * 4 : vi * 4 + 2, vj * 4 : vj * 4 + 2]])
            for ii, jj in T.grid(2, 2):
                with T.block("B_sub"):
                    vi_i, vj_i = T.axis.remap("SS", [ii, jj])
                    B[vi * 4 + vi_i, vj * 4 + vj_i] = A[vi * 4 + vi_i, vj * 4 + vj_i] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def bound_to_thread(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 128], scope="shared")
    for i in T.thread_binding(0, 128, thread="threadIdx.x"):
        for j in T.serial(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for j in T.serial(0, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vj, vi] = B[vj, vi] + 1.0


@T.prim_func
def equal_ranked_threads(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 128], scope="shared")
    for i_o in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i_i in T.thread_binding(0, 8, thread="threadIdx.y"):
            for j in T.serial(0, 128):
                with T.block("B"):
                    vi = T.axis.S(128, i_o * 8 + i_i)
                    vj = T.axis.S(128, j)
                    B[vi, vj] = A[vi, vj] * 2.0
            for j in T.serial(0, 128):
                with T.block("C"):
                    vi = T.axis.S(128, i_o * 8 + i_i)
                    vj = T.axis.S(128, j)
                    C[vj, vi] = B[vj, vi] + 1.0


@T.prim_func
def warp_memory(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 4, 32], scope="warp")
    for i_o in T.thread_binding(0, 4, thread="threadIdx.y"):
        for i_i in T.thread_binding(0, 32, thread="threadIdx.x"):
            for j in T.serial(0, 128):
                with T.block("B"):
                    warp_id, lane_id, vj = T.axis.remap("SSS", [i_o, i_i, j])
                    B[vj, warp_id, lane_id] = A[warp_id * 32 + lane_id, vj] * 2.0
            for j in T.serial(0, 128):
                with T.block("C"):
                    warp_id, lane_id, vj = T.axis.remap("SSS", [i_o, i_i, j])
                    C[warp_id * 32 + lane_id, vj] = B[vj, warp_id, lane_id] + 1.0


@T.prim_func
def warp_memory_negative(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    C = T.match_buffer(c, [128, 128])
    B = T.alloc_buffer([128, 4, 32], scope="warp")
    for i_o in T.thread_binding(0, 4, thread="threadIdx.y"):
        for i_i in T.thread_binding(0, 32, thread="threadIdx.x"):
            for j in T.serial(0, 128):
                with T.block("B"):
                    warp_id, lane_id, vj = T.axis.remap("SSS", [i_o, i_i, j])
                    B[vj, warp_id, lane_id] = A[warp_id * 32 + lane_id, vj] * 2.0
            for i_o_prime in T.thread_binding(0, 4, thread="threadIdx.y"):
                for j in T.serial(0, 128):
                    with T.block("C"):
                        _warp_id, warp_id, lane_id, vj = T.axis.remap(
                            "SSSS", [i_o, i_i, i_o_prime, j]
                        )
                        C[warp_id * 32 + lane_id, vj] = B[vj, warp_id, lane_id] + 1.0


@T.prim_func
def non_perfect_tiling_cache(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [224, 224], dtype="float32")
    Y = T.match_buffer(b, [224, 224], dtype="float32")
    cache = T.alloc_buffer([224, 224], dtype="float32")
    for hh_0, ww_0 in T.grid(28, 28):
        for ax0 in T.serial(0, 10):
            for ax1 in T.serial(0, 10):
                with T.block("cache"):
                    h = T.axis.spatial(224, hh_0 * 8 - 1 + ax0)
                    w = T.axis.spatial(224, ww_0 * 8 - 1 + ax1)
                    T.where(
                        1 <= hh_0 * 8 + ax0
                        and hh_0 * 8 + ax0 < 225
                        and 1 <= ww_0 * 8 + ax1
                        and ww_0 * 8 + ax1 < 225
                    )
                    cache[h, w] = X[h, w]
        for hh_1, ww_1, khh, kww in T.grid(8, 8, 3, 3):
            with T.block("compute"):
                h = T.axis.spatial(224, hh_0 * 8 + hh_1)
                w = T.axis.spatial(224, ww_0 * 8 + ww_1)
                kh, kw = T.axis.remap("RR", [khh, kww])
                with T.init():
                    Y[h, w] = 0.0
                Y[h, w] = T.max(
                    Y[h, w],
                    T.if_then_else(
                        T.likely(1 <= h + kh, dtype="bool")
                        and T.likely(h + kh < 225, dtype="bool")
                        and T.likely(1 <= w + kw, dtype="bool")
                        and T.likely(w + kw < 225, dtype="bool"),
                        cache[h + kh - 1, w + kw - 1],
                        0.0,
                        dtype="float32",
                    ),
                )


@T.prim_func
def uncovered_producer_region(A: T.Buffer[(128,), "float32"], B: T.Buffer[(128,), "float32"]):
    for i in range(120):
        with T.block("producer"):
            vi = T.axis.S((0, 120), i)
            A[vi] = 1.0
    for i in range(120):
        with T.block("consumer"):
            vi = T.axis.S((8, 128), i + 8)
            B[vi] = A[vi]


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


def test_non_perfect_tiling_cache():
    s = tir.ScheduleState(non_perfect_tiling_cache, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "cache")) == CachedFlags(
        affine_binding=False,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "compute")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


def test_uncovered_producer_region():
    s = tir.ScheduleState(uncovered_producer_region, debug_mask="all")
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "consumer")) == CachedFlags(
        affine_binding=True,
        region_cover=False,
        stage_pipeline=True,
    )
    # pylint: enable=protected-access


if __name__ == "__main__":
    tvm.testing.main()
