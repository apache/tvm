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
import sys

import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original)
    mod = tvm.tir.transform.LowerCrossThreadReduction()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


def _check_fail(original):
    mod = tvm.IRModule.from_expr(original)
    with pytest.raises(ValueError):
        tvm.tir.transform.LowerCrossThreadReduction()(mod)


@T.prim_func
def loop_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i, ko in T.grid(128, 4):
        for ki in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("B"):
                vi = T.axis.S(128, i)
                vk = T.axis.R(128, ko * 32 + ki)
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def lowered_loop_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    normal_reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.serial(0, 128):
        for ki in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("B_in_thread_init"):
                T.reads([])
                T.writes([normal_reduce_temp0[0]])
                normal_reduce_temp0[0] = T.float32(0)
            for ko in T.serial(0, 4):
                with T.block("B_normal_reduction"):
                    vi = T.axis.S(128, i)
                    vk = T.axis.R(128, ko * 32 + ki)
                    T.reads([A[vi, vk], normal_reduce_temp0[0]])
                    T.writes([normal_reduce_temp0[0]])
                    normal_reduce_temp0[0] = normal_reduce_temp0[0] + A[vi, vk]
            with T.block("B_cross_thread_reduction"):
                T.reads([normal_reduce_temp0[0]])
                T.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp0[0],
                        True,
                        reduce_temp0[0],
                        ki,
                        dtype="handle",
                    )
                )
            with T.block("B_write_back"):
                vi = T.axis.S(128, i)
                T.reads([reduce_temp0[0]])
                T.writes([B[vi]])
                B[vi] = reduce_temp0[0]


@T.prim_func
def no_normal_reduction(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def lowered_no_normal_reduction(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B_cross_thread_reduction"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([A[vi, vk]])
                T.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), A[vi, vk], True, reduce_temp0[0], k, dtype="handle"
                    )
                )
            with T.block("B_write_back"):
                vi = T.axis.spatial(128, i)
                T.reads([reduce_temp0[0]])
                T.writes([B[vi]])
                B[vi] = reduce_temp0[0]


@T.prim_func
def two_bound_loops(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i in T.serial(0, 128):
        for ko in T.thread_binding(0, 4, thread="threadIdx.x"):
            for ki in T.thread_binding(0, 32, thread="threadIdx.y"):
                with T.block("B"):
                    vi = T.axis.spatial(128, i)
                    vk = T.axis.reduce(128, ko * 32 + ki)
                    T.reads([B[vi], A[vi, vk]])
                    T.writes([B[vi]])
                    with T.init():
                        B[vi] = T.float32(0)
                    B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def lowered_two_bound_loops(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.serial(0, 128):
        for ko in T.thread_binding(0, 4, thread="threadIdx.x"):
            for ki in T.thread_binding(0, 32, thread="threadIdx.y"):
                with T.block("B_cross_thread_reduction"):
                    vi = T.axis.spatial(128, i)
                    vk = T.axis.reduce(128, ko * 32 + ki)
                    T.reads([A[vi, vk]])
                    T.writes([reduce_temp0[0]])
                    T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    )
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1), A[vi, vk], True, reduce_temp0[0], ko, ki, dtype="handle"
                        )
                    )
                with T.block("B_write_back"):
                    vi = T.axis.spatial(128, i)
                    T.reads([reduce_temp0[0]])
                    T.writes([B[vi]])
                    B[vi] = reduce_temp0[0]


@T.prim_func
def multiple_blocks_under_reduction_loop(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16, 16], dtype="float32")
    B = T.match_buffer(b, [16], dtype="float32")
    B_rf_local = T.alloc_buffer([16, 16], dtype="float32", scope="local")
    for i in T.thread_binding(0, 16, thread="blockIdx.x"):
        for k0o in T.thread_binding(0, 4, thread="threadIdx.x"):
            for k0i0, k1 in T.grid(4, 16):
                with T.block("B_rf"):
                    vk0 = T.axis.spatial(16, k0o * 4 + k0i0)
                    vi, vk1 = T.axis.remap("SR", [i, k1])
                    T.reads([B_rf_local[vk0, vi], A[vi, vk0, vk1]])
                    T.writes([B_rf_local[vk0, vi]])
                    with T.init():
                        B_rf_local[vk0, vi] = T.float32(0)
                    B_rf_local[vk0, vi] = B_rf_local[vk0, vi] + A[vi, vk0, vk1]
            for k0i1 in T.serial(0, 4):
                with T.block("B"):
                    vk0 = T.axis.reduce(16, k0o * 4 + k0i1)
                    vi = T.axis.spatial(16, i)
                    T.reads([B[vi], B_rf_local[vk0, vi]])
                    T.writes([B[vi]])
                    with T.init():
                        B[vi] = T.float32(0)
                    B[vi] = B[vi] + B_rf_local[vk0, vi]


@T.prim_func
def lowered_multiple_blocks_under_reduction_loop(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16, 16], dtype="float32")
    B = T.match_buffer(b, [16], dtype="float32")
    B_rf_local = T.alloc_buffer([16, 16], dtype="float32", scope="local")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    normal_reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.thread_binding(0, 16, thread="blockIdx.x"):
        for k0o in T.thread_binding(0, 4, thread="threadIdx.x"):
            with T.block("B_in_thread_init"):
                T.reads([])
                T.writes([normal_reduce_temp0[0]])
                normal_reduce_temp0[0] = T.float32(0)
            for k0i0, k1 in T.grid(4, 16):
                with T.block("B_rf"):
                    vk0 = T.axis.spatial(16, k0o * 4 + k0i0)
                    vi, vk1 = T.axis.remap("SR", [i, k1])
                    T.reads([B_rf_local[vk0, vi], A[vi, vk0, vk1]])
                    T.writes([B_rf_local[vk0, vi]])
                    with T.init():
                        B_rf_local[vk0, vi] = T.float32(0)
                    B_rf_local[vk0, vi] = B_rf_local[vk0, vi] + A[vi, vk0, vk1]
            for k0i1 in T.serial(0, 4):
                with T.block("B_normal_reduction"):
                    vk0 = T.axis.reduce(16, k0o * 4 + k0i1)
                    vi = T.axis.spatial(16, i)
                    T.reads([B_rf_local[vk0, vi], normal_reduce_temp0[0]])
                    T.writes([normal_reduce_temp0[0]])
                    normal_reduce_temp0[0] = normal_reduce_temp0[0] + B_rf_local[vk0, vi]
            with T.block("B_cross_thread_reduction"):
                T.reads([normal_reduce_temp0[0]])
                T.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp0[0],
                        True,
                        reduce_temp0[0],
                        k0o,
                        dtype="handle",
                    )
                )
            with T.block("B_write_back"):
                vi = T.axis.spatial(16, i)
                T.reads([reduce_temp0[0]])
                T.writes([B[vi]])
                B[vi] = reduce_temp0[0]


@T.prim_func
def with_block_predicate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 120], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i, ko in T.grid(128, 4):
        for ki in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("B"):
                vi = T.axis.spatial(128, i)
                vk = T.axis.reduce(120, ko * 32 + ki)
                T.where(ko * 32 + ki < 120)
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def lowered_with_block_predicate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 120], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    normal_reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.serial(0, 128):
        for ki in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("B_in_thread_init"):
                T.reads([])
                T.writes([normal_reduce_temp0[0]])
                normal_reduce_temp0[0] = T.float32(0)
            for ko in T.serial(0, 4):
                with T.block("B_normal_reduction"):
                    vi = T.axis.spatial(128, i)
                    vk = T.axis.reduce(120, ko * 32 + ki)
                    T.where(ko * 32 + ki < 120)
                    T.reads([A[vi, vk], normal_reduce_temp0[0]])
                    T.writes([normal_reduce_temp0[0]])
                    normal_reduce_temp0[0] = normal_reduce_temp0[0] + A[vi, vk]
            with T.block("B_cross_thread_reduction"):
                T.reads([normal_reduce_temp0[0]])
                T.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp0[0],
                        True,
                        reduce_temp0[0],
                        ki,
                        dtype="handle",
                    )
                )
            with T.block("B_write_back"):
                vi = T.axis.spatial(128, i)
                T.reads([reduce_temp0[0]])
                T.writes([B[vi]])
                B[vi] = reduce_temp0[0]


@T.prim_func
def single_reduction_loop_with_block_predicate(
    A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]
) -> None:
    T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    for i0 in T.serial(256):
        for ax0, ax1_0 in T.grid(1, 1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_maxelem"):
                    i0_1 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax1_1)
                    T.where(ax1_0 * 512 + ax1_1 < 256)
                    T.reads(T_softmax_maxelem_shared[i0_1], A[i0_1, k])
                    T.writes(T_softmax_maxelem_shared[i0_1])
                    with T.init():
                        T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem_shared[i0_1] = T.max(
                        T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                    )
        for ax0, ax1_0 in T.grid(1, 1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_expsum"):
                    i0_2 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax1_1)
                    T.where(ax1_0 * 512 + ax1_1 < 256)
                    T.reads(
                        T_softmax_expsum_shared[i0_2], A[i0_2, k], T_softmax_maxelem_shared[i0_2]
                    )
                    T.writes(T_softmax_expsum_shared[i0_2])
                    with T.init():
                        T_softmax_expsum_shared[i0_2] = T.float32(0)
                    T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                        A[i0_2, k] - T_softmax_maxelem_shared[i0_2], dtype="float32"
                    )
        for i1_0 in T.serial(1):
            for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_norm"):
                    i0_3 = T.axis.spatial(256, i0)
                    i1 = T.axis.spatial(256, i1_1)
                    T.where(i1_0 * 512 + i1_1 < 256)
                    T.reads(
                        A[i0_3, i1], T_softmax_maxelem_shared[i0_3], T_softmax_expsum_shared[i0_3]
                    )
                    T.writes(T_softmax_norm[i0_3, i1])
                    T.block_attr({"axis": 1})
                    T_softmax_norm[i0_3, i1] = (
                        T.exp(A[i0_3, i1] - T_softmax_maxelem_shared[i0_3], dtype="float32")
                        / T_softmax_expsum_shared[i0_3]
                    )


@T.prim_func
def lowered_single_reduction_loop_with_block_predicate(
    A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]
) -> None:
    T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    cross_thread_0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    cross_thread_1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i0 in T.serial(256):
        for ax0, ax1_0 in T.grid(1, 1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_maxelem_in_thread_init"):
                    T.reads()
                    T.writes(in_thread_0[0])
                    in_thread_0[0] = T.float32(-3.4028234663852886e38)
                with T.block("T_softmax_maxelem_in_thread"):
                    i0_1 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax1_1)
                    T.where(ax1_0 * 512 + ax1_1 < 256)
                    T.reads(A[i0_1, k], in_thread_0[0])
                    T.writes(in_thread_0[0])
                    in_thread_0[0] = T.max(in_thread_0[0], A[i0_1, k])
                with T.block("T_softmax_maxelem_cross_thread"):
                    T.reads(in_thread_0[0])
                    T.writes(cross_thread_0[0])
                    T.attr(
                        T.comm_reducer(
                            lambda x, y: T.max(x, y), [T.float32(-3.4028234663852886e38)]
                        ),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    )
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            in_thread_0[0],
                            True,
                            cross_thread_0[0],
                            ax1_1,
                            dtype="handle",
                        )
                    )
                with T.block("T_softmax_maxelem_write_back"):
                    i0_2 = T.axis.spatial(256, i0)
                    T.reads(cross_thread_0[0])
                    T.writes(T_softmax_maxelem_shared[i0_2])
                    T_softmax_maxelem_shared[i0_2] = cross_thread_0[0]
        for ax0, ax1_0 in T.grid(1, 1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_expsum_in_thread_init"):
                    T.reads()
                    T.writes(in_thread_1[0])
                    in_thread_1[0] = T.float32(0)
                with T.block("T_softmax_expsum_in_thread"):
                    i0_3 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax1_1)
                    T.where(ax1_0 * 512 + ax1_1 < 256)
                    T.reads(A[i0_3, k], T_softmax_maxelem_shared[i0_3], in_thread_1[0])
                    T.writes(in_thread_1[0])
                    in_thread_1[0] = in_thread_1[0] + T.exp(
                        A[i0_3, k] - T_softmax_maxelem_shared[i0_3], dtype="float32"
                    )
                with T.block("T_softmax_expsum_cross_thread"):
                    T.reads(in_thread_1[0])
                    T.writes(cross_thread_1[0])
                    T.attr(
                        T.comm_reducer(lambda x_1, y_1: x_1 + y_1, [T.float32(0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    )
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            in_thread_1[0],
                            True,
                            cross_thread_1[0],
                            ax1_1,
                            dtype="handle",
                        )
                    )
                with T.block("T_softmax_expsum_write_back"):
                    i0_4 = T.axis.spatial(256, i0)
                    T.reads(cross_thread_1[0])
                    T.writes(T_softmax_expsum_shared[i0_4])
                    T_softmax_expsum_shared[i0_4] = cross_thread_1[0]
        for i1_0 in T.serial(1):
            for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_norm"):
                    i0_5 = T.axis.spatial(256, i0)
                    i1 = T.axis.spatial(256, i1_1)
                    T.where(i1_0 * 512 + i1_1 < 256)
                    T.reads(
                        A[i0_5, i1], T_softmax_maxelem_shared[i0_5], T_softmax_expsum_shared[i0_5]
                    )
                    T.writes(T_softmax_norm[i0_5, i1])
                    T.block_attr({"axis": 1})
                    T_softmax_norm[i0_5, i1] = (
                        T.exp(A[i0_5, i1] - T_softmax_maxelem_shared[i0_5], dtype="float32")
                        / T_softmax_expsum_shared[i0_5]
                    )


@T.prim_func
def reducer_max(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.min_value("float32")
                B[vi] = T.max(B[vi], A[vi, vk])


@T.prim_func
def lowered_reducer_max(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B_cross_thread_reduction"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([A[vi, vk]])
                T.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: T.max(x, y), [T.min_value("float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), A[vi, vk], True, reduce_temp0[0], k, dtype="handle"
                    )
                )
            with T.block("B_write_back"):
                vi = T.axis.spatial(128, i)
                T.reads([reduce_temp0[0]])
                T.writes([B[vi]])
                B[vi] = reduce_temp0[0]


@T.prim_func
def zero_rank_buffer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="float32")
    B = T.match_buffer(b, [], dtype="float32")
    for k in T.thread_binding(0, 128, thread="threadIdx.x"):
        with T.block("B"):
            vk = T.axis.reduce(128, k)
            T.reads([B[()], A[vk]])
            T.writes([B[()]])
            with T.init():
                B[()] = T.float32(0)
            B[()] = B[()] + A[vk]


@T.prim_func
def lowered_zero_rank_buffer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="float32")
    B = T.match_buffer(b, [], dtype="float32")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for k in T.thread_binding(0, 128, thread="threadIdx.x"):
        with T.block("B_cross_thread_reduction"):
            vk = T.axis.reduce(128, k)
            T.reads([A[vk]])
            T.writes([reduce_temp0[0]])
            T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            )
            T.evaluate(
                T.tvm_thread_allreduce(T.uint32(1), A[vk], True, reduce_temp0[0], k, dtype="handle")
            )
        with T.block("B_write_back"):
            T.reads([reduce_temp0[0]])
            T.writes([B[()]])
            B[()] = reduce_temp0[0]


@T.prim_func
def multiple_bufferstore(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    C = T.alloc_buffer([], dtype="float32")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([A[vi, vk], B[vi], C[()]])
                T.writes([B[vi], C[()]])
                with T.init():
                    B[vi] = T.float32(0)
                C[()] = A[vi, vk]
                B[vi] = B[vi] + C[()]


@T.prim_func
def reduction_loop_not_deepest(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for k in T.thread_binding(0, 128, thread="threadIdx.x"):
        for i in T.serial(0, 128):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def reduction_loop_bound_to_blockidx(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="blockIdx.x"):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def different_access_indices(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128, 128], dtype="float32")
    B = T.match_buffer(b, [128, 128], dtype="float32")
    for i, j in T.grid(128, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads([B[vi, vj], A[vi, vj, vk]])
                T.writes(
                    [
                        B[
                            T.min(vj, vi) : T.min(vj, vi) + (T.max(vj, vi) + 1 - T.min(vj, vi)),
                            T.min(vi, vj) : T.min(vi, vj) + (T.max(vi, vj) + 1 - T.min(vi, vj)),
                        ]
                    ]
                )
                with T.init():
                    B[vj, vi] = T.float32(0)
                B[vi, vj] = B[vi, vj] + A[vi, vj, vk]


@T.prim_func
def invalid_reducer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([B[vi], A[vi, vk]])
                T.writes([B[vi]])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] - A[vi, vk]


@T.prim_func
def softmax(var_A: T.handle, var_T_softmax_norm: T.handle) -> None:
    A = T.match_buffer(var_A, [256, 256], dtype="float32")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, [256, 256], dtype="float32")
    T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    for i0 in T.thread_binding(0, 256, thread="blockIdx.x"):
        for ax0_0 in T.serial(0, 8):
            for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("T_softmax_maxelem"):
                    i0_1 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax0_0 * 32 + ax0_1)
                    T.reads([T_softmax_maxelem_shared[i0_1], A[i0_1, k]])
                    T.writes([T_softmax_maxelem_shared[i0_1]])
                    with T.init():
                        T_softmax_maxelem_shared[i0_1] = T.min_value("float32")
                    T_softmax_maxelem_shared[i0_1] = T.max(
                        T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                    )
        for ax0_0 in T.serial(0, 8):
            for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("T_softmax_expsum"):
                    i0_2 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax0_0 * 32 + ax0_1)
                    T.reads(
                        [
                            T_softmax_expsum_shared[i0_2],
                            A[i0_2, k],
                            T_softmax_maxelem_shared[i0_2],
                        ]
                    )
                    T.writes([T_softmax_expsum_shared[i0_2]])
                    with T.init():
                        T_softmax_expsum_shared[i0_2] = T.float32(0)
                    T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                        A[i0_2, k] - T_softmax_maxelem_shared[i0_2], dtype="float32"
                    )
        for i1_0 in T.serial(0, 8):
            for i1_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("T_softmax_norm"):
                    i0_3 = T.axis.spatial(256, i0)
                    i1 = T.axis.spatial(256, i1_0 * 32 + i1_1)
                    T.reads(
                        [
                            A[i0_3, i1],
                            T_softmax_maxelem_shared[i0_3],
                            T_softmax_expsum_shared[i0_3],
                        ]
                    )
                    T.writes([T_softmax_norm[i0_3, i1]])
                    T.block_attr({"axis": 1})
                    T_softmax_norm[i0_3, i1] = (
                        T.exp(
                            A[i0_3, i1] - T_softmax_maxelem_shared[i0_3],
                            dtype="float32",
                        )
                        / T_softmax_expsum_shared[i0_3]
                    )


@T.prim_func
def lowered_softmax(var_A: T.handle, var_T_softmax_norm: T.handle) -> None:
    A = T.match_buffer(var_A, [256, 256], dtype="float32")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, [256, 256], dtype="float32")
    T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    normal_reduce_temp0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    reduce_temp1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    normal_reduce_temp1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i0 in T.thread_binding(0, 256, thread="blockIdx.x"):
        for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("T_softmax_maxelem_normal_reduction_init"):
                T.reads([])
                T.writes([normal_reduce_temp0[0]])
                normal_reduce_temp0[0] = T.min_value("float32")
            for ax0_0 in T.serial(0, 8):
                with T.block("T_softmax_maxelem_normal_reduction"):
                    i0_1 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax0_0 * 32 + ax0_1)
                    T.reads([A[i0_1, k], normal_reduce_temp0[0]])
                    T.writes([normal_reduce_temp0[0]])
                    normal_reduce_temp0[0] = T.max(normal_reduce_temp0[0], A[i0_1, k])
            with T.block("T_softmax_maxelem_cross_thread_reduction"):
                T.reads([normal_reduce_temp0[0]])
                T.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: T.max(x, y), [T.min_value("float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp0[0],
                        True,
                        reduce_temp0[0],
                        ax0_1,
                        dtype="handle",
                    )
                )
            with T.block("T_softmax_maxelem_write_back"):
                i0_2 = T.axis.spatial(256, i0)
                T.reads([reduce_temp0[0]])
                T.writes([T_softmax_maxelem_shared[i0_2]])
                T_softmax_maxelem_shared[i0_2] = reduce_temp0[0]
        for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
            with T.block("T_softmax_expsum_normal_reduction_init"):
                T.reads([])
                T.writes([normal_reduce_temp1[0]])
                normal_reduce_temp1[0] = T.float32(0)
            for ax0_0 in T.serial(0, 8):
                with T.block("T_softmax_expsum_normal_reduction"):
                    i0_3 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, ax0_0 * 32 + ax0_1)
                    T.reads(
                        [
                            A[i0_3, k],
                            T_softmax_maxelem_shared[i0_3],
                            normal_reduce_temp1[0],
                        ]
                    )
                    T.writes([normal_reduce_temp1[0]])
                    normal_reduce_temp1[0] = normal_reduce_temp1[0] + T.exp(
                        A[i0_3, k] - T_softmax_maxelem_shared[i0_3], dtype="float32"
                    )
            with T.block("T_softmax_expsum_cross_thread_reduction"):
                T.reads([normal_reduce_temp1[0]])
                T.writes([reduce_temp1[0]])
                T.attr(
                    T.comm_reducer(lambda x_1, y_1: x_1 + y_1, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp1[0],
                        True,
                        reduce_temp1[0],
                        ax0_1,
                        dtype="handle",
                    )
                )
            with T.block("T_softmax_expsum_write_back"):
                i0_4 = T.axis.spatial(256, i0)
                T.reads([reduce_temp1[0]])
                T.writes([T_softmax_expsum_shared[i0_4]])
                T_softmax_expsum_shared[i0_4] = reduce_temp1[0]
        for i1_0 in T.serial(0, 8):
            for i1_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("T_softmax_norm"):
                    i0_5 = T.axis.spatial(256, i0)
                    i1 = T.axis.spatial(256, i1_0 * 32 + i1_1)
                    T.reads(
                        [
                            A[i0_5, i1],
                            T_softmax_maxelem_shared[i0_5],
                            T_softmax_expsum_shared[i0_5],
                        ]
                    )
                    T.writes([T_softmax_norm[i0_5, i1]])
                    T.block_attr({"axis": 1})
                    T_softmax_norm[i0_5, i1] = (
                        T.exp(
                            A[i0_5, i1] - T_softmax_maxelem_shared[i0_5],
                            dtype="float32",
                        )
                        / T_softmax_expsum_shared[i0_5]
                    )


def test_loop_split():
    _check(loop_split, lowered_loop_split)


def test_no_normal_reduction():
    _check(no_normal_reduction, lowered_no_normal_reduction)


def test_two_bound_loops():
    _check(two_bound_loops, lowered_two_bound_loops)


def test_multiple_blocks_under_reduction_loop():
    _check(multiple_blocks_under_reduction_loop, lowered_multiple_blocks_under_reduction_loop)


def test_with_block_predicate():
    _check(with_block_predicate, lowered_with_block_predicate)


def test_single_reduction_loop_with_block_predicate():
    _check(
        single_reduction_loop_with_block_predicate,
        lowered_single_reduction_loop_with_block_predicate,
    )


def test_reducer_max():
    _check(reducer_max, lowered_reducer_max)


def test_zero_rank_buffer():
    _check(zero_rank_buffer, lowered_zero_rank_buffer)


def test_multiple_bufferstore():
    _check_fail(multiple_bufferstore)


def test_reduction_block_not_deepest():
    _check_fail(reduction_loop_not_deepest)


def test_reduction_loop_bound_to_blockidx():
    _check_fail(reduction_loop_bound_to_blockidx)


def test_different_access_indices():
    _check_fail(different_access_indices)


def test_invalid_reducer():
    _check_fail(invalid_reducer)


def test_softmax():
    _check(softmax, lowered_softmax)


def test_lower_te():
    a = te.placeholder((32, 2, 2))
    k1 = te.reduce_axis((0, 2), "k1")
    k2 = te.reduce_axis((0, 2), "k2")
    b = te.compute((32,), lambda i: te.sum(a[i, k1, k2], axis=[k1, k2]))
    s = te.create_schedule(b.op)
    s[b].bind(k1, te.thread_axis("threadIdx.x"))
    s[b].bind(k2, te.thread_axis("threadIdx.y"))
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [a, b])
    mod = tvm.tir.transform.LowerCrossThreadReduction()(orig_mod)
    tvm.ir.assert_structural_equal(
        mod, orig_mod
    )  # LowerCrossThreadReduction should do nothing on TE


if __name__ == "__main__":
    tvm.testing.main()
