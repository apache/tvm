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
from tvm import te
from tvm.script import tir as T

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LowerCrossThreadReduction()(mod)
    tvm.ir.assert_structural_equal(
        mod["main"], transformed.with_attr("global_symbol", "main"), True
    )


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
                T.reads([A[vi, vk]])
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
                    T.reads([A[vi, vk]])
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
                T.where(ki == 0)
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
                T.reads([A[vi, vk]])
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
                T.where(k == 0)
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
                    T.reads([A[vi, vk]])
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
                    T.where(ko == 0 and ki == 0)
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
                    T.reads([A[vi, vk0, vk1]])
                    T.writes([B_rf_local[vk0, vi]])
                    with T.init():
                        B_rf_local[vk0, vi] = T.float32(0)
                    B_rf_local[vk0, vi] = B_rf_local[vk0, vi] + A[vi, vk0, vk1]
            for k0i1 in T.serial(0, 4):
                with T.block("B"):
                    vk0 = T.axis.reduce(16, k0o * 4 + k0i1)
                    vi = T.axis.spatial(16, i)
                    T.reads([B_rf_local[vk0, vi]])
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
                    T.reads([A[vi, vk0, vk1]])
                    T.writes([B_rf_local[vk0, vi]])
                    with T.init():
                        B_rf_local[vk0, vi] = T.float32(0)
                    B_rf_local[vk0, vi] = B_rf_local[vk0, vi] + A[vi, vk0, vk1]
            for k0i1 in T.serial(0, 4):
                with T.block("B_normal_reduction"):
                    vk0 = T.axis.reduce(16, k0o * 4 + k0i1)
                    vi = T.axis.spatial(16, i)
                    T.reads([B_rf_local[vk0, vi]])
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
                T.where(k0o == 0)
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
                T.reads([A[vi, vk]])
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
                    T.reads([A[vi, vk]])
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
                T.where(ki == 0)
                T.reads([reduce_temp0[0]])
                T.writes([B[vi]])
                B[vi] = reduce_temp0[0]


@T.prim_func
def single_reduction_loop_with_block_predicate(
    A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
) -> None:
    T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    for i0 in T.serial(256):
        for ax0, ax1_0 in T.grid(1, 1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_maxelem"):
                    i0_1 = T.axis.spatial(256, i0 + ax0)
                    k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                    T.where(ax1_0 * 512 + ax1_1 < 256)
                    T.reads(A[i0_1, k])
                    T.writes(T_softmax_maxelem_shared[i0_1])
                    with T.init():
                        T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem_shared[i0_1] = T.max(
                        T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                    )
        for ax0, ax1_0 in T.grid(1, 1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_expsum"):
                    i0_2 = T.axis.spatial(256, i0 + ax0)
                    k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                    T.where(ax1_0 * 512 + ax1_1 < 256)
                    T.reads(A[i0_2, k], T_softmax_maxelem_shared[i0_2])
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
                    i1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
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
    A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
) -> None:
    T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
    cross_thread_0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    cross_thread_1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i0 in T.serial(256):
        for ax0 in T.serial(1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_maxelem_in_thread_init"):
                    T.reads()
                    T.writes(in_thread_0[0])
                    in_thread_0[0] = T.float32(-3.4028234663852886e38)
                for ax1_0 in T.serial(1):
                    with T.block("T_softmax_maxelem_in_thread"):
                        T.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_1 = T.axis.spatial(256, i0 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        T.reads(A[i0_1, k])
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
                    i0_2 = T.axis.spatial(256, i0 + ax0)
                    T.where(ax1_1 == 0)
                    T.reads(cross_thread_0[0])
                    T.writes(T_softmax_maxelem_shared[i0_2])
                    T_softmax_maxelem_shared[i0_2] = cross_thread_0[0]
        for ax0 in T.serial(1):
            for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_expsum_in_thread_init"):
                    T.reads()
                    T.writes(in_thread_1[0])
                    in_thread_1[0] = T.float32(0)
                for ax1_0 in T.serial(1):
                    with T.block("T_softmax_expsum_in_thread"):
                        T.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_3 = T.axis.spatial(256, i0 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        T.reads(A[i0_3, k], T_softmax_maxelem_shared[i0_3])
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
                    i0_4 = T.axis.spatial(256, i0 + ax0)
                    T.where(ax1_1 == 0)
                    T.reads(cross_thread_1[0])
                    T.writes(T_softmax_expsum_shared[i0_4])
                    T_softmax_expsum_shared[i0_4] = cross_thread_1[0]
        for i1_0 in T.serial(1):
            for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                with T.block("T_softmax_norm"):
                    i0_5 = T.axis.spatial(256, i0)
                    i1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
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
def spatial_reduction_with_shared_prefetch(
    A: T.Buffer((128, 150528), "float32"),
    B: T.Buffer((128, 150528), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    C_local = T.alloc_buffer((128, 128), scope="local")
    A_shared = T.alloc_buffer((128, 150528), scope="shared")
    B_shared = T.alloc_buffer((128, 150528), scope="shared")
    for ax0_0_ax1_0_fused in T.thread_binding(256, thread="blockIdx.x"):
        for ax0_1_ax1_1_fused in T.thread_binding(64, thread="threadIdx.y"):
            for ax2_1_1_fused in T.thread_binding(2, thread="threadIdx.x"):
                for ax2_0 in range(392):
                    for ax0_ax1_fused_0 in range(6):
                        for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(2, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.serial(4):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(
                                            128,
                                            ax0_0_ax1_0_fused // 16 * 8
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            // 384,
                                        )
                                        v1 = T.axis.spatial(
                                            150528,
                                            ax2_0 * 384
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            % 384,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                    for ax0_ax1_fused_0 in range(6):
                        for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(2, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.serial(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(
                                            128,
                                            ax0_0_ax1_0_fused % 16 * 8
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            // 384,
                                        )
                                        v1 = T.axis.spatial(
                                            150528,
                                            ax2_0 * 384
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            % 384,
                                        )
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                    for ax2_1_0 in range(192):
                        with T.block("B"):
                            v0 = T.axis.spatial(
                                128, ax0_0_ax1_0_fused // 16 * 8 + ax0_1_ax1_1_fused // 8
                            )
                            v1 = T.axis.spatial(
                                128, ax0_0_ax1_0_fused % 16 * 8 + ax0_1_ax1_1_fused % 8
                            )
                            v2 = T.axis.reduce(150528, ax2_0 * 384 + ax2_1_0 * 2 + ax2_1_1_fused)
                            T.reads(A_shared[v0, v2], B_shared[v1, v2])
                            T.writes(C_local[v0, v1])
                            with T.init():
                                C_local[v0, v1] = T.float32(0)
                            C_local[v0, v1] = C_local[v0, v1] + A_shared[v0, v2] * B_shared[v1, v2]
            with T.block("C_local"):
                v0 = T.axis.spatial(128, ax0_0_ax1_0_fused // 16 * 8 + ax0_1_ax1_1_fused // 8)
                v1 = T.axis.spatial(128, ax0_0_ax1_0_fused % 16 * 8 + ax0_1_ax1_1_fused % 8)
                T.reads(C_local[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_local[v0, v1]


@T.prim_func
def lowered_spatial_reduction_with_shared_prefetch(
    A: T.Buffer((128, 150528), "float32"),
    B: T.Buffer((128, 150528), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    C_local = T.alloc_buffer((128, 128), scope="local")
    A_shared = T.alloc_buffer((128, 150528), scope="shared")
    B_shared = T.alloc_buffer((128, 150528), scope="shared")
    cross_thread_C_local = T.alloc_buffer((1,), strides=(1,), scope="local")
    in_thread_C_local = T.alloc_buffer((1,), strides=(1,), scope="local")
    for ax0_0_ax1_0_fused in T.thread_binding(256, thread="blockIdx.x"):
        for ax0_1_ax1_1_fused in T.thread_binding(64, thread="threadIdx.y"):
            for ax2_1_1_fused in T.thread_binding(2, thread="threadIdx.x"):
                with T.block("B_in_thread_init"):
                    T.reads()
                    T.writes(in_thread_C_local[0])
                    in_thread_C_local[0] = T.float32(0)
                for ax2_0 in range(392):
                    for ax0_ax1_fused_0 in range(6):
                        for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(2, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in range(4):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(
                                            128,
                                            ax0_0_ax1_0_fused // 16 * 8
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            // 384,
                                        )
                                        v1 = T.axis.spatial(
                                            150528,
                                            ax2_0 * 384
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            % 384,
                                        )
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                    for ax0_ax1_fused_0 in range(6):
                        for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(2, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in range(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(
                                            128,
                                            ax0_0_ax1_0_fused % 16 * 8
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            // 384,
                                        )
                                        v1 = T.axis.spatial(
                                            150528,
                                            ax2_0 * 384
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 8
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            % 384,
                                        )
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                    for ax2_1_0 in range(192):
                        with T.block("B_in_thread"):
                            v0 = T.axis.spatial(
                                128, ax0_0_ax1_0_fused // 16 * 8 + ax0_1_ax1_1_fused // 8
                            )
                            v1 = T.axis.spatial(
                                128, ax0_0_ax1_0_fused % 16 * 8 + ax0_1_ax1_1_fused % 8
                            )
                            v2 = T.axis.reduce(150528, ax2_0 * 384 + ax2_1_0 * 2 + ax2_1_1_fused)
                            T.reads(A_shared[v0, v2], B_shared[v1, v2])
                            T.writes(in_thread_C_local[0])
                            in_thread_C_local[0] = (
                                in_thread_C_local[0] + A_shared[v0, v2] * B_shared[v1, v2]
                            )
                with T.block("B_cross_thread"):
                    T.reads(in_thread_C_local[0])
                    T.writes(cross_thread_C_local[0])
                    T.attr(
                        T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                        "reduce_scope",
                        T.reinterpret("handle", T.uint64(0)),
                    )
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        in_thread_C_local[0],
                        T.bool(True),
                        cross_thread_C_local[0],
                        ax2_1_1_fused,
                    )
                with T.block("B_write_back"):
                    v0 = T.axis.spatial(128, ax0_0_ax1_0_fused // 16 * 8 + ax0_1_ax1_1_fused // 8)
                    v1 = T.axis.spatial(128, ax0_0_ax1_0_fused % 16 * 8 + ax0_1_ax1_1_fused % 8)
                    T.reads(cross_thread_C_local[0])
                    T.writes(C_local[v0, v1])
                    C_local[v0, v1] = cross_thread_C_local[0]
            for tx in T.thread_binding(2, thread="threadIdx.x"):
                with T.block("C_local"):
                    v0 = T.axis.spatial(128, ax0_0_ax1_0_fused // 16 * 8 + ax0_1_ax1_1_fused // 8)
                    v1 = T.axis.spatial(128, ax0_0_ax1_0_fused % 16 * 8 + ax0_1_ax1_1_fused % 8)
                    T.where(tx == 0)
                    T.reads(C_local[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_local[v0, v1]


@T.prim_func
def spatial_reduction_loop_predicate(A: T.Buffer((2, 32), "float32"), B: T.Buffer((2,), "float32")):
    for i_0 in range(1):
        for i_1 in T.thread_binding(16, thread="threadIdx.y"):
            for k_0 in range(1):
                for k_1 in T.thread_binding(64, thread="threadIdx.x"):
                    with T.block("block"):
                        vi = T.axis.spatial(2, i_0 * 16 + i_1)
                        vk = T.axis.reduce(32, k_0 * 64 + k_1)
                        T.where(i_0 * 16 + i_1 < 2 and k_0 * 64 + k_1 < 32)
                        T.reads(A[vi, vk])
                        T.writes(B[vi])
                        with T.init():
                            B[vi] = T.float32(0)
                        B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def lowered_reduction_spatial_loop_predicate(
    A: T.Buffer((2, 32), "float32"), B: T.Buffer((2,), "float32")
):
    cross_thread_B = T.alloc_buffer((1,), strides=(1,), scope="local")
    in_thread_B = T.alloc_buffer((1,), strides=(1,), scope="local")
    for i_0 in range(1):
        for i_1 in T.thread_binding(16, thread="threadIdx.y"):
            for k_1 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("block_in_thread_init"):
                    T.reads()
                    T.writes(in_thread_B[0])
                    in_thread_B[0] = T.float32(0)
                for k_0 in range(1):
                    with T.block("block_in_thread"):
                        vi = T.axis.spatial(2, i_0 * 16 + i_1)
                        vk = T.axis.reduce(32, k_0 * 64 + k_1)
                        T.where(i_0 * 16 + i_1 < 2 and k_0 * 64 + k_1 < 32)
                        T.reads(A[vi, vk])
                        T.writes(in_thread_B[0])
                        in_thread_B[0] = in_thread_B[0] + A[vi, vk]
                with T.block("block_cross_thread"):
                    T.reads(in_thread_B[0])
                    T.writes(cross_thread_B[0])
                    T.attr(
                        T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                        "reduce_scope",
                        T.reinterpret("handle", T.uint64(0)),
                    )
                    T.tvm_thread_allreduce(
                        T.uint32(1), in_thread_B[0], T.bool(True), cross_thread_B[0], k_1
                    )
                k_0 = T.int32()
                with T.block("block_write_back"):
                    vi = T.axis.spatial(2, i_0 * 16 + i_1)
                    T.where(i_0 * 16 + i_1 < 2 and k_1 == 0)
                    T.reads(cross_thread_B[0])
                    T.writes(B[vi])
                    B[vi] = cross_thread_B[0]


@T.prim_func
def single_reduction_loop_with_tensorize(
    input_A: T.Buffer((1, 64, 7, 7, 32), "uint8"),
    input_B: T.Buffer((16, 64, 1, 1, 8, 32, 4), "int8"),
    output: T.Buffer((1, 16, 7, 7, 32), "int32"),
) -> None:
    # body
    # with T.block("root")
    for i1, i2, i3, i4, i5 in T.grid(16, 4, 98, 2, 32):
        with T.block("compute_o"):
            n = T.axis.spatial(1, 0)
            oc_chunk = T.axis.spatial(16, i1)
            oh = T.axis.spatial(7, (i2 * 6272 + i3 * 64 + i4 * 32 + i5) // 3584)
            ow = T.axis.spatial(7, (i2 * 6272 + i3 * 64 + i4 * 32 + i5) % 3584 // 512)
            kh = T.axis.reduce(1, 0)
            kw = T.axis.reduce(1, 0)
            ic_outer = T.axis.reduce(64, (i2 * 6272 + i3 * 64 + i4 * 32 + i5) % 512 // 8)
            ic_f_inner = T.axis.reduce(8, (i2 * 6272 + i3 * 64 + i4 * 32 + i5) % 8)
            T.reads(
                input_A[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4],
                input_B[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:32, 0:4],
            )
            T.writes(output[n, oc_chunk, oh, ow, 0:32])
            with T.init():
                for x in T.serial(32):
                    with T.block("compute_init"):
                        oc_block_i_init = T.axis.spatial(32, x)
                        T.reads()
                        T.writes(output[n, oc_chunk, oh, ow, oc_block_i_init])
                        output[n, oc_chunk, oh, ow, oc_block_i_init] = 0
            with T.block("compute_o"):
                T.reads(
                    output[n, oc_chunk, oh, ow, 0:32],
                    input_A[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4],
                    input_B[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:32, 0:4],
                )
                T.writes(output[n, oc_chunk, oh, ow, 0:32])
                A = T.match_buffer(
                    input_A[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4],
                    [4],
                    dtype="uint8",
                    offset_factor=1,
                )
                B = T.match_buffer(
                    input_B[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:32, 0:4],
                    [32, 4],
                    dtype="int8",
                    offset_factor=1,
                )
                C = T.match_buffer(
                    output[n, oc_chunk, oh, ow, 0:32], [32], dtype="int32", offset_factor=1
                )
                A_u8x4: T.uint8x4 = A[0:4]
                A_i32: T.int32 = T.reinterpret(A_u8x4, dtype="int32")
                B_i8x128 = B[0, 0:128]
                B_i32x32: T.int32x32 = T.reinterpret(B_i8x128, dtype="int32x32")
                C[0:32] = T.call_llvm_pure_intrin(
                    4217, T.uint32(3), C[0:32], T.broadcast(A_i32, 32), B_i32x32, dtype="int32x32"
                )


@T.prim_func
def nested_reduction_loop_with_inner_match_buffers(
    in0: T.Buffer((4, 16), "int8"),
    in1: T.Buffer((4, 16), "int8"),
    out: T.Buffer((4, 4), "int32"),
) -> None:
    # body
    # with T.block("root")
    for y in T.serial(4):
        with T.block("C"):
            yi = T.axis.spatial(4, y)
            T.reads(in0[yi, 0:16], in1[yi, 0:16])
            T.writes(out[yi, 0:4])
            for x in T.serial(4):
                with T.block("C"):
                    xr = T.axis.reduce(4, x)
                    with T.init():
                        for i in T.serial(4):
                            with T.block("C_init"):
                                ii = T.axis.spatial(4, i)
                                T.reads()
                                T.writes(out[yi, ii])
                                out[yi, ii] = 0
                    with T.block("C"):
                        T.reads(
                            out[yi, xr],
                            in0[yi, yi * 4 + xr : yi * 4 + xr + 4],
                            in1[yi, yi * 4 + xr : yi * 4 + xr + 4],
                        )
                        T.writes(out[yi, xr])
                        A = T.match_buffer(
                            in0[yi, yi * 4 + xr : yi * 4 + xr + 4],
                            [4],
                            dtype="int8",
                            offset_factor=1,
                        )
                        B = T.match_buffer(
                            in1[yi, yi * 4 + xr : yi * 4 + xr + 4],
                            [4],
                            dtype="int8",
                            offset_factor=1,
                        )
                        C = T.match_buffer(out[yi, xr], [1], dtype="int32", offset_factor=1)
                        A_i8x4: T.int8x4 = A[0:4]
                        A_i32: T.int32 = T.reinterpret(A_i8x4, dtype="int32")
                        B_i8x4: T.int8x4 = B[0:4]
                        B_i32: T.int32 = T.reinterpret(B_i8x4, dtype="int32")
                        C[0] = A_i32 + B_i32 + C[0]


@T.prim_func
def reducer_max(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i in T.serial(0, 128):
        for k in T.thread_binding(0, 128, thread="threadIdx.x"):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads([A[vi, vk]])
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
                T.where(k == 0)
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
            T.reads([A[vk]])
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
            T.where(k == 0)
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
                T.reads([A[vi, vk]])
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
                T.reads([A[vi, vk]])
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
                T.reads([A[vi, vj, vk]])
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
                T.reads([A[vi, vk]])
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
                    T.reads([A[i0_1, k]])
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
                    T.reads([A[i0_1, k]])
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
                T.where(ax0_1 == 0)
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
                T.where(ax0_1 == 0)
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


@T.prim_func
def argmax_split(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0 in T.grid(128, 4):
        for i1_1 in T.thread_binding(32, thread="threadIdx.x"):
            with T.block("argmax"):
                i = T.axis.spatial(128, i0)
                k = T.axis.reduce(128, i1_0 * 32 + i1_1)
                T.reads(idx[i, k], val[i, k])
                T.writes(argmax_v0[i], argmax_v1[i])
                with T.init():
                    argmax_v0[i] = -1
                    argmax_v1[i] = T.float32(-3.4028234663852886e38)
                v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
                v_argmax_v1: T.float32 = T.Select(
                    argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                )
                argmax_v0[i] = v_argmax_v0
                argmax_v1[i] = v_argmax_v1


@T.prim_func
def lowered_argmax_split(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    cross_thread_argmax_v0 = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
    cross_thread_argmax_v1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_argmax_v0 = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
    in_thread_argmax_v1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i0 in T.serial(128):
        for i1_1 in T.thread_binding(32, thread="threadIdx.x"):
            with T.block("argmax_in_thread_init"):
                T.reads()
                T.writes(in_thread_argmax_v0[0], in_thread_argmax_v1[0])
                in_thread_argmax_v0[0] = -1
                in_thread_argmax_v1[0] = T.float32(-3.4028234663852886e38)
            for i1_0 in T.serial(4):
                with T.block("argmax_in_thread"):
                    i = T.axis.spatial(128, i0)
                    k = T.axis.reduce(128, i1_0 * 32 + i1_1)
                    T.reads(idx[i, k], val[i, k])
                    T.writes(in_thread_argmax_v0[0], in_thread_argmax_v1[0])
                    v_argmax_v0: T.int32 = T.Select(
                        in_thread_argmax_v1[0] >= val[i, k], in_thread_argmax_v0[0], idx[i, k]
                    )
                    v_argmax_v1: T.float32 = T.Select(
                        in_thread_argmax_v1[0] >= val[i, k], in_thread_argmax_v1[0], val[i, k]
                    )
                    in_thread_argmax_v0[0] = v_argmax_v0
                    in_thread_argmax_v1[0] = v_argmax_v1
            with T.block("argmax_cross_thread"):
                T.reads(in_thread_argmax_v0[0], in_thread_argmax_v1[0])
                T.writes(cross_thread_argmax_v0[0], cross_thread_argmax_v1[0])
                T.attr(
                    T.comm_reducer(
                        lambda x0, x1, y0, y1: (
                            T.Select(x1 >= y1, x0, y0),
                            T.Select(x1 >= y1, x1, y1),
                        ),
                        [-1, T.float32(-3.4028234663852886e38)],
                    ),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(2),
                        in_thread_argmax_v0[0],
                        in_thread_argmax_v1[0],
                        True,
                        cross_thread_argmax_v0[0],
                        cross_thread_argmax_v1[0],
                        i1_1,
                        dtype="handle",
                    )
                )
            with T.block("argmax_write_back"):
                i = T.axis.spatial(128, i0)
                T.where(i1_1 == 0)
                T.reads(cross_thread_argmax_v0[0], cross_thread_argmax_v1[0])
                T.writes(argmax_v0[i], argmax_v1[i])
                argmax_v0[i] = cross_thread_argmax_v0[0]
                argmax_v1[i] = cross_thread_argmax_v1[0]


@T.prim_func
def argmin_split_init_update_reordered(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmin_v0: T.Buffer((128,), "int32"),
    argmin_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0 in T.grid(128, 4):
        for i1_1 in T.thread_binding(32, thread="threadIdx.x"):
            with T.block("argmin"):
                i = T.axis.spatial(128, i0)
                k = T.axis.reduce(128, i1_0 * 32 + i1_1)
                T.reads(idx[i, k], val[i, k])
                T.writes(argmin_v0[i], argmin_v1[i])
                with T.init():
                    argmin_v1[i] = T.float32(3.4028234663852886e38)
                    argmin_v0[i] = -1
                v_argmin_v0: T.int32 = T.Select(argmin_v1[i] <= val[i, k], argmin_v0[i], idx[i, k])
                v_argmin_v1: T.float32 = T.Select(
                    argmin_v1[i] <= val[i, k], argmin_v1[i], val[i, k]
                )
                argmin_v1[i] = v_argmin_v1
                argmin_v0[i] = v_argmin_v0


@T.prim_func
def lowered_argmin_split_init_update_reordered(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmin_v0: T.Buffer((128,), "int32"),
    argmin_v1: T.Buffer((128,), "float32"),
) -> None:
    cross_thread_argmin_v0 = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
    cross_thread_argmin_v1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_argmin_v0 = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
    in_thread_argmin_v1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i0 in T.serial(128):
        for i1_1 in T.thread_binding(32, thread="threadIdx.x"):
            with T.block("argmin_in_thread_init"):
                T.reads()
                T.writes(in_thread_argmin_v0[0], in_thread_argmin_v1[0])
                in_thread_argmin_v0[0] = -1
                in_thread_argmin_v1[0] = T.float32(3.4028234663852886e38)
            for i1_0 in T.serial(4):
                with T.block("argmin_in_thread"):
                    i = T.axis.spatial(128, i0)
                    k = T.axis.reduce(128, i1_0 * 32 + i1_1)
                    T.reads(idx[i, k], val[i, k])
                    T.writes(in_thread_argmin_v0[0], in_thread_argmin_v1[0])
                    v_argmin_v0: T.int32 = T.Select(
                        in_thread_argmin_v1[0] <= val[i, k], in_thread_argmin_v0[0], idx[i, k]
                    )
                    v_argmin_v1: T.float32 = T.Select(
                        in_thread_argmin_v1[0] <= val[i, k], in_thread_argmin_v1[0], val[i, k]
                    )
                    in_thread_argmin_v1[0] = v_argmin_v1
                    in_thread_argmin_v0[0] = v_argmin_v0
            with T.block("argmin_cross_thread"):
                T.reads(in_thread_argmin_v0[0], in_thread_argmin_v1[0])
                T.writes(cross_thread_argmin_v0[0], cross_thread_argmin_v1[0])
                T.attr(
                    T.comm_reducer(
                        lambda x0, x1, y0, y1: (
                            T.Select(x1 <= y1, x0, y0),
                            T.Select(x1 <= y1, x1, y1),
                        ),
                        [-1, T.float32(3.4028234663852886e38)],
                    ),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(2),
                        in_thread_argmin_v0[0],
                        in_thread_argmin_v1[0],
                        True,
                        cross_thread_argmin_v0[0],
                        cross_thread_argmin_v1[0],
                        i1_1,
                        dtype="handle",
                    )
                )
            with T.block("argmin_write_back"):
                i = T.axis.spatial(128, i0)
                T.where(i1_1 == 0)
                T.reads(cross_thread_argmin_v0[0], cross_thread_argmin_v1[0])
                T.writes(argmin_v0[i], argmin_v1[i])
                argmin_v0[i] = cross_thread_argmin_v0[0]
                argmin_v1[i] = cross_thread_argmin_v1[0]


@T.prim_func
def layer_norm_tuple_sum(
    data: T.Buffer((128, 768), "float32"),
    gamma: T.Buffer(768, "float32"),
    bias: T.Buffer(768, "float32"),
    T_layer_norm: T.Buffer((128, 768), "float32"),
) -> None:
    data_red_temp_v0 = T.alloc_buffer([128], dtype="float32")
    data_red_temp_v1 = T.alloc_buffer([128], dtype="float32")
    for i0_fused in T.thread_binding(128, thread="blockIdx.x"):
        for i1_0 in T.serial(24):
            for i1_1 in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("data_red_temp"):
                    ax0 = T.axis.spatial(128, i0_fused)
                    k1 = T.axis.reduce(768, i1_0 * 32 + i1_1)
                    T.reads(data[ax0, k1])
                    T.writes(data_red_temp_v0[ax0], data_red_temp_v1[ax0])
                    with T.init():
                        data_red_temp_v0[ax0] = T.float32(0)
                        data_red_temp_v1[ax0] = T.float32(0)
                    v_data_red_temp_v0: T.float32 = data_red_temp_v0[ax0] + data[ax0, k1]
                    v_data_red_temp_v1: T.float32 = (
                        data_red_temp_v1[ax0] + data[ax0, k1] * data[ax0, k1]
                    )
                    data_red_temp_v0[ax0] = v_data_red_temp_v0
                    data_red_temp_v1[ax0] = v_data_red_temp_v1
    for i0_i1_fused_0 in T.thread_binding(384, thread="blockIdx.x"):
        for i0_i1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("T_layer_norm"):
                ax0 = T.axis.spatial(128, (i0_i1_fused_0 * 256 + i0_i1_fused_1) // 768)
                ax1 = T.axis.spatial(768, (i0_i1_fused_0 * 256 + i0_i1_fused_1) % 768)
                T.reads(
                    data[ax0, ax1],
                    data_red_temp_v0[ax0],
                    data_red_temp_v1[ax0],
                    gamma[ax1],
                    bias[ax1],
                )
                T.writes(T_layer_norm[ax0, ax1])
                T_layer_norm[ax0, ax1] = (
                    data[ax0, ax1] - data_red_temp_v0[ax0] * T.float32(0.0013020833333333333)
                ) * T.rsqrt(
                    data_red_temp_v1[ax0] * T.float32(0.0013020833333333333)
                    - data_red_temp_v0[ax0]
                    * T.float32(0.0013020833333333333)
                    * (data_red_temp_v0[ax0] * T.float32(0.0013020833333333333))
                    + T.float32(1.0000000000000001e-05),
                    dtype="float32",
                ) * gamma[
                    ax1
                ] + bias[
                    ax1
                ]


@T.prim_func
def lowered_layer_norm_tuple_sum(
    data: T.Buffer((128, 768), "float32"),
    gamma: T.Buffer(768, "float32"),
    bias: T.Buffer(768, "float32"),
    T_layer_norm: T.Buffer((128, 768), "float32"),
) -> None:
    # with T.block("root")
    data_red_temp_v0 = T.alloc_buffer([128], dtype="float32")
    data_red_temp_v1 = T.alloc_buffer([128], dtype="float32")
    cross_thread_data_red_temp_v0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    cross_thread_data_red_temp_v1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_data_red_temp_v0 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    in_thread_data_red_temp_v1 = T.alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i0_fused in T.thread_binding(128, thread="blockIdx.x"):
        for i1_1 in T.thread_binding(32, thread="threadIdx.x"):
            with T.block("data_red_temp_in_thread_init"):
                T.reads()
                T.writes(in_thread_data_red_temp_v0[0], in_thread_data_red_temp_v1[0])
                in_thread_data_red_temp_v0[0] = T.float32(0)
                in_thread_data_red_temp_v1[0] = T.float32(0)
            for i1_0 in T.serial(24):
                with T.block("data_red_temp_in_thread"):
                    ax0 = T.axis.spatial(128, i0_fused)
                    k1 = T.axis.reduce(768, i1_0 * 32 + i1_1)
                    T.reads(data[ax0, k1])
                    T.writes(in_thread_data_red_temp_v0[0], in_thread_data_red_temp_v1[0])
                    v_data_red_temp_v0: T.float32 = in_thread_data_red_temp_v0[0] + data[ax0, k1]
                    v_data_red_temp_v1: T.float32 = (
                        in_thread_data_red_temp_v1[0] + data[ax0, k1] * data[ax0, k1]
                    )
                    in_thread_data_red_temp_v0[0] = v_data_red_temp_v0
                    in_thread_data_red_temp_v1[0] = v_data_red_temp_v1
            with T.block("data_red_temp_cross_thread"):
                T.reads(in_thread_data_red_temp_v0[0], in_thread_data_red_temp_v1[0])
                T.writes(cross_thread_data_red_temp_v0[0], cross_thread_data_red_temp_v1[0])
                T.attr(
                    T.comm_reducer(
                        lambda x0, x1, y0, y1: (x0 + y0, x1 + y1), [T.float32(0), T.float32(0)]
                    ),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(2),
                        in_thread_data_red_temp_v0[0],
                        in_thread_data_red_temp_v1[0],
                        True,
                        cross_thread_data_red_temp_v0[0],
                        cross_thread_data_red_temp_v1[0],
                        i1_1,
                        dtype="handle",
                    )
                )
            with T.block("data_red_temp_write_back"):
                ax0 = T.axis.spatial(128, i0_fused)
                T.where(i1_1 == 0)
                T.reads(cross_thread_data_red_temp_v0[0], cross_thread_data_red_temp_v1[0])
                T.writes(data_red_temp_v0[ax0], data_red_temp_v1[ax0])
                data_red_temp_v0[ax0] = cross_thread_data_red_temp_v0[0]
                data_red_temp_v1[ax0] = cross_thread_data_red_temp_v1[0]
    for i0_i1_fused_0 in T.thread_binding(384, thread="blockIdx.x"):
        for i0_i1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("T_layer_norm"):
                ax0 = T.axis.spatial(128, (i0_i1_fused_0 * 256 + i0_i1_fused_1) // 768)
                ax1 = T.axis.spatial(768, (i0_i1_fused_0 * 256 + i0_i1_fused_1) % 768)
                T.reads(
                    data[ax0, ax1],
                    data_red_temp_v0[ax0],
                    data_red_temp_v1[ax0],
                    gamma[ax1],
                    bias[ax1],
                )
                T.writes(T_layer_norm[ax0, ax1])
                T_layer_norm[ax0, ax1] = (
                    data[ax0, ax1] - data_red_temp_v0[ax0] * T.float32(0.0013020833333333333)
                ) * T.rsqrt(
                    data_red_temp_v1[ax0] * T.float32(0.0013020833333333333)
                    - data_red_temp_v0[ax0]
                    * T.float32(0.0013020833333333333)
                    * (data_red_temp_v0[ax0] * T.float32(0.0013020833333333333))
                    + T.float32(1.0000000000000001e-05),
                    dtype="float32",
                ) * gamma[
                    ax1
                ] + bias[
                    ax1
                ]


@T.prim_func
def thread_broadcast_1(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256,), "float32")):
    temp_local = T.alloc_buffer((256,), scope="local")
    for i in T.thread_binding(256, thread="blockIdx.x"):
        for k in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("sum"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(A[vi, vk])
                T.writes(temp_local[vi])
                with T.init():
                    temp_local[vi] = T.float32(0)
                temp_local[vi] = temp_local[vi] + A[vi, vk]
        with T.block("add"):
            vi = T.axis.spatial(256, i)
            T.reads(temp_local[vi])
            T.writes(B[vi])
            B[vi] = temp_local[vi] + T.float32(1)


@T.prim_func
def lowered_thread_broadcast_1(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256,), "float32")):
    temp_local = T.alloc_buffer((256,), scope="local")
    cross_thread_temp_local = T.alloc_buffer((1,), strides=(1,), scope="local")
    for i in T.thread_binding(256, thread="blockIdx.x"):
        for k in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("sum_cross_thread"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(A[vi, vk])
                T.writes(cross_thread_temp_local[0])
                T.attr(
                    T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret("handle", T.uint64(0)),
                )
                T.tvm_thread_allreduce(
                    T.uint32(1), A[vi, vk], T.bool(True), cross_thread_temp_local[0], k
                )
            with T.block("sum_write_back"):
                vi = T.axis.spatial(256, i)
                T.reads(cross_thread_temp_local[0])
                T.writes(temp_local[vi])
                temp_local[vi] = cross_thread_temp_local[0]
        for tx in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("add"):
                vi = T.axis.spatial(256, i)
                T.where(tx == 0)
                T.reads(temp_local[vi])
                T.writes(B[vi])
                B[vi] = temp_local[vi] + T.float32(1)


# fmt: off
@T.prim_func
def thread_broadcast_2(lv1605: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16"), p_lv1606: T.handle, p_lv1582: T.handle, p_output0: T.handle):
    n = T.int64()
    lv1606 = T.match_buffer(p_lv1606, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    lv1582 = T.match_buffer(p_lv1582, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
    var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16", scope="local")
    var_NT_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(256), T.int64(1), T.int64(32), T.int64(1), n), "float16", scope="local")
    for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
        for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            with T.block("NT_matmul_rf_init"):
                vax2_fused_1 = T.axis.spatial(T.int64(256), ax2_fused_1)
                v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                T.reads()
                T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
            for ax2_fused_0 in range(T.int64(1)):
                with T.block("NT_matmul_rf_update"):
                    vax2_fused_1 = T.axis.spatial(T.int64(256), ax2_fused_1)
                    v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                    v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                    vax2_fused_0 = T.axis.reduce(T.int64(1), ax2_fused_0)
                    T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < T.int64(128))
                    T.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1], lv1605[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(256) + vax2_fused_1], lv1606[T.int64(0), v0, v1, vax2_fused_0 * T.int64(256) + vax2_fused_1])
                    T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                    var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] + lv1605[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(256) + vax2_fused_1] * lv1606[T.int64(0), v0, v1, vax2_fused_0 * T.int64(256) + vax2_fused_1]
        for ax1_ax2_fused in range(T.int64(1)):
            for ax0_fused in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("NT_matmul"):
                    vax2_fused_1 = T.axis.reduce(T.int64(256), ax0_fused)
                    v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                    v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                    T.where(T.int64(0) <= ax0_ax1_fused // n and ax0_ax1_fused // n < T.int64(32) and T.int64(0) <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
                    T.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                    T.writes(var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1])
                    with T.init():
                        var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                    var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] = var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] + var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1]
        with T.block("compute"):
            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
            v1 = T.axis.spatial(n, ax0_ax1_fused % n)
            T.where(T.int64(0) <= ax0_ax1_fused // n and ax0_ax1_fused // n < T.int64(32) and T.int64(0) <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
            T.reads(var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1], lv1582[T.int64(0), T.int64(0), T.int64(0), v1])
            T.writes(var_compute_intermediate[T.int64(0), v0, T.int64(0), v1])
            var_compute_intermediate[T.int64(0), v0, T.int64(0), v1] = T.Cast("float32", T.min(T.max(var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] * T.float16(0.088397790055248615), T.float16(-65504)), lv1582[T.int64(0), T.int64(0), T.int64(0), v1]))


@T.prim_func
def lowered_thread_broadcast_2(lv1605: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16"), p_lv1606: T.handle, p_lv1582: T.handle, p_output0: T.handle):
    n = T.int64()
    lv1606 = T.match_buffer(p_lv1606, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    lv1582 = T.match_buffer(p_lv1582, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
    var_NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16", scope="local")
    var_NT_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(256), T.int64(1), T.int64(32), T.int64(1), n), "float16", scope="local")
    cross_thread_var_NT_matmul_intermediate_local = T.alloc_buffer((1,), "float16", strides=(1,), scope="local")
    in_thread_var_NT_matmul_intermediate_local = T.alloc_buffer((1,), "float16", strides=(1,), scope="local")
    for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
        for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            with T.block("NT_matmul_rf_init"):
                vax2_fused_1 = T.axis.spatial(T.int64(256), ax2_fused_1)
                v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                T.reads()
                T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
            for ax2_fused_0 in range(T.int64(1)):
                with T.block("NT_matmul_rf_update"):
                    vax2_fused_1 = T.axis.spatial(T.int64(256), ax2_fused_1)
                    v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                    v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                    vax2_fused_0 = T.axis.reduce(T.int64(1), ax2_fused_0)
                    T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < T.int64(128))
                    T.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1], lv1605[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(256) + vax2_fused_1], lv1606[T.int64(0), v0, v1, vax2_fused_0 * T.int64(256) + vax2_fused_1])
                    T.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                    var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] + lv1605[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(256) + vax2_fused_1] * lv1606[T.int64(0), v0, v1, vax2_fused_0 * T.int64(256) + vax2_fused_1]
        for ax1_ax2_fused in range(T.int64(1)):
            for ax0_fused in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("NT_matmul_in_thread_init"):
                    T.reads()
                    T.writes(in_thread_var_NT_matmul_intermediate_local[0])
                    in_thread_var_NT_matmul_intermediate_local[0] = T.float16(0)
                with T.block("NT_matmul_in_thread"):
                    vax2_fused_1 = T.axis.reduce(T.int64(256), ax0_fused)
                    v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                    v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                    T.where(T.int64(0) <= ax0_ax1_fused // n and ax0_ax1_fused // n < T.int64(32) and T.int64(0) <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
                    T.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                    T.writes(in_thread_var_NT_matmul_intermediate_local[0])
                    in_thread_var_NT_matmul_intermediate_local[0] = in_thread_var_NT_matmul_intermediate_local[0] + var_NT_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1]
                with T.block("NT_matmul_cross_thread"):
                    T.reads(in_thread_var_NT_matmul_intermediate_local[0])
                    T.writes(cross_thread_var_NT_matmul_intermediate_local[0])
                    T.attr(T.comm_reducer(lambda x0, y0: x0 + y0, [T.float16(0)]), "reduce_scope", T.reinterpret("handle", T.uint64(0)))
                    T.tvm_thread_allreduce(T.uint32(1), in_thread_var_NT_matmul_intermediate_local[0], T.bool(True), cross_thread_var_NT_matmul_intermediate_local[0], ax0_fused)
                with T.block("NT_matmul_write_back"):
                    v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                    v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                    T.where(T.int64(0) <= ax0_ax1_fused // n and ax0_ax1_fused // n < T.int64(32) and T.int64(0) <= ax0_ax1_fused % n and ax0_ax1_fused % n < n)
                    T.reads(cross_thread_var_NT_matmul_intermediate_local[0])
                    T.writes(var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1])
                    var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] = cross_thread_var_NT_matmul_intermediate_local[0]
        for tx in T.thread_binding(T.int64(256), thread="threadIdx.x"):
            with T.block("compute"):
                v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                T.where(tx == T.int64(0) and (T.int64(0) <= ax0_ax1_fused // n and ax0_ax1_fused // n < T.int64(32) and T.int64(0) <= ax0_ax1_fused % n and ax0_ax1_fused % n < n))
                T.reads(var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1], lv1582[T.int64(0), T.int64(0), T.int64(0), v1])
                T.writes(var_compute_intermediate[T.int64(0), v0, T.int64(0), v1])
                var_compute_intermediate[T.int64(0), v0, T.int64(0), v1] = T.Cast("float32", T.min(T.max(var_NT_matmul_intermediate_local[T.int64(0), v0, T.int64(0), v1] * T.float16(0.088397790055248615), T.float16(-65504)), lv1582[T.int64(0), T.int64(0), T.int64(0), v1]))
# fmt: on


@T.prim_func
def no_thread_broadcast(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256, 256), "float32")):
    temp_1_local = T.alloc_buffer((256,), scope="local")
    temp_2_local = T.alloc_buffer((1,), scope="local")
    for i in T.thread_binding(256, thread="blockIdx.x"):
        for k in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("sum"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(A[vi, vk])
                T.writes(temp_1_local[vi])
                with T.init():
                    temp_1_local[vi] = T.float32(0)
                temp_1_local[vi] = temp_1_local[vi] + A[vi, vk]
        with T.block("add"):
            vi = T.axis.spatial(256, i)
            T.reads(temp_1_local[vi])
            T.writes(temp_2_local[0])
            temp_2_local[0] = temp_1_local[vi] + T.float32(1)
        for j in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("sum"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(temp_2_local[0])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] + temp_2_local[0]


@T.prim_func
def lowered_no_thread_broadcast(
    A: T.Buffer((256, 256), "float32"), B: T.Buffer((256, 256), "float32")
):
    temp_1_local = T.alloc_buffer((256,), scope="local")
    temp_2_local = T.alloc_buffer((1,), scope="local")
    cross_thread_temp_1_local = T.alloc_buffer((1,), strides=(1,), scope="local")
    for i in T.thread_binding(256, thread="blockIdx.x"):
        for k in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("sum_cross_thread"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(A[vi, vk])
                T.writes(cross_thread_temp_1_local[0])
                T.attr(
                    T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret("handle", T.uint64(0)),
                )
                T.tvm_thread_allreduce(
                    T.uint32(1), A[vi, vk], T.bool(True), cross_thread_temp_1_local[0], k
                )
            with T.block("sum_write_back"):
                vi = T.axis.spatial(256, i)
                T.reads(cross_thread_temp_1_local[0])
                T.writes(temp_1_local[vi])
                temp_1_local[vi] = cross_thread_temp_1_local[0]
        with T.block("add"):
            vi = T.axis.spatial(256, i)
            T.reads(temp_1_local[vi])
            T.writes(temp_2_local[0])
            temp_2_local[0] = temp_1_local[vi] + T.float32(1)
        for j in T.thread_binding(256, thread="threadIdx.x"):
            with T.block("sum"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(temp_2_local[0])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] + temp_2_local[0]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


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


def test_single_reduction_loop_with_shared_memory_prefetch():
    _check(
        spatial_reduction_with_shared_prefetch,
        lowered_spatial_reduction_with_shared_prefetch,
    )


def test_single_reduction_loop_with_block_predicate():
    _check(
        single_reduction_loop_with_block_predicate,
        lowered_single_reduction_loop_with_block_predicate,
    )


def test_spatial_reduction_loop_predicate():
    _check(spatial_reduction_loop_predicate, lowered_reduction_spatial_loop_predicate)


def test_single_reduction_loop_with_tensorize():
    _check(
        single_reduction_loop_with_tensorize,
        single_reduction_loop_with_tensorize,
    )


def test_nested_reduction_loop_with_inner_match_buffers():
    _check(
        nested_reduction_loop_with_inner_match_buffers,
        nested_reduction_loop_with_inner_match_buffers,
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


def test_argmax_split():
    _check(argmax_split, lowered_argmax_split)


def test_argmin_split_init_update_reordered():
    _check(argmin_split_init_update_reordered, lowered_argmin_split_init_update_reordered)


def test_thread_broadcast_rewrite_1():
    _check(thread_broadcast_1, lowered_thread_broadcast_1)


def test_thread_broadcast_rewrite_2():
    _check(thread_broadcast_2, lowered_thread_broadcast_2)


def test_no_thread_broadcast_rewrite():
    _check(no_thread_broadcast, lowered_no_thread_broadcast)


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


def test_layer_norm_tuple_sum():
    _check(layer_norm_tuple_sum, lowered_layer_norm_tuple_sum)


if __name__ == "__main__":
    tvm.testing.main()
