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
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@T.prim_func
def two_elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    with T.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i in range(0, 128):
        for ax0, ax1 in T.grid(1, 128):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i + ax0)
                T.bind(vj, ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 128):
            with T.block([128, 128], "B") as [vi, vj]:
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_1(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    with T.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with T.block([8, 8], "C_outer") as [vi_o, vj_o]:
        T.reads([B[
            vi_o * 16 : vi_o * 16 + 16,
            vj_o * 16 : vj_o * 16 + 16,
        ]])
        T.writes([C[
            vi_o * 16 : vi_o * 16 + 16,
            vj_o * 16 : vj_o * 16 + 16
        ]])
        for i_i, j_i in T.grid(16, 16):
            with T.block([128, 128], "C_inner") as [vi, vj]:
                T.bind(vi, vi_o * 16 + i_i)
                T.bind(vj, vj_o * 16 + j_i)
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i0_0, i1_0 in T.grid(8, 8):
        for ax0, ax1 in T.grid(16, 16):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i0_0 * 16 + ax0)
                T.bind(vj, i1_0 * 16 + ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        with T.block([8, 8], "C_outer") as [vi_o, vj_o]:
            T.bind(vi_o, i0_0)
            T.bind(vj_o, i1_0)
            T.reads([B[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16,
            ]])
            T.writes([C[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16
            ]])
            for i0_1, i1_1 in T.grid(16, 16):
                with T.block([128, 128], "C_inner") as [vi, vj]:
                    T.bind(vi, vi_o * 16 + i0_1)
                    T.bind(vj, vj_o * 16 + i1_1)
                    C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_2(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in T.grid(8, 8):
        with T.block([8, 8], "B_outer") as [vio, vjo]:
            T.bind(vio, i_o)
            T.bind(vjo, j_o)
            T.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            T.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with T.block([128, 128], "B_inner") as [vi, vj]:
                    T.bind(vi, vio * 16 + i_i)
                    T.bind(vj, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
    for i_o, j_o, i_i, j_i in T.grid(4, 4, 32, 32):
        with T.block([128, 128], "C") as [vi, vj]:
            T.bind(vi, i_o * 32 + i_i)
            T.bind(vj, j_o * 32 + j_i)
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_2_after_reverse_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in T.grid(8, 8):
        with T.block([8, 8], "B_outer") as [vio, vjo]:
            T.bind(vio, i_o)
            T.bind(vjo, j_o)
            T.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            T.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with T.block([128, 128], "B_inner") as [vi, vj]:
                    T.bind(vi, vio * 16 + i_i)
                    T.bind(vj, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
        for ax0, ax1 in T.grid(16, 16):
            with T.block([128, 128], "C") as [vi, vj]:
                T.bind(vi, i_o * 16 + ax0)
                T.bind(vj, j_o * 16 + ax1)
                T.reads([B[vi, vj]])
                T.writes([C[vi, vj]])
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_2_after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in T.grid(4, 4):
        for ax0, ax1 in T.grid(2, 2):
            with T.block([8, 8], "blockized_B") as [vio, vjo]:
                T.bind(vio, i_o * 2 + ax0)
                T.bind(vjo, j_o * 2 + ax1)
                T.reads([A[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                T.writes([B[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                for i_i, j_i in T.grid(16, 16):
                    with T.block([128, 128], "B") as [vi, vj]:
                        T.bind(vi, vio * 16 + i_i)
                        T.bind(vj, vjo * 16 + j_i)
                        B[vi, vj] = A[vi, vj] * 2.0
        for i_i, j_i in T.grid(32, 32):
            with T.block([128, 128], "C") as [vi, vj]:
                T.bind(vi, i_o * 32 + i_i)
                T.bind(vj, j_o * 32 + j_i)
                C[vi, vj] = B[vi, vj] + 1.0

@T.prim_func
def cuda_matmul_0(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    A_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    with T.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with T.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with T.block([2048, 2048], "A_shared_local") as [v0, v1]:
        A_shared_local[v0, v1] = A_shared[v0, v1]
    with T.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
        with T.init():
            C_local[vi, vj] = 0.0
        C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for i, j in T.grid(4, 4):
                                with T.block([2048, 2048], "C_local") as [v0_4, v1_4]:
                                    T.bind(v0_4, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(v1_4, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0_4, v1_4] = C_local[v0_4, v1_4]


@T.prim_func
def cuda_matmul_0_after_compute_at(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    A_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    with T.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with T.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with T.block([2048, 2048], "A_shared_local") as [v0, v1]:
        A_shared_local[v0, v1] = A_shared[v0, v1]
    with T.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for i, j, k in T.grid(4, 4, 2048):
                                with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                    T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                    T.bind(vk, k)
                                    with T.init():
                                        C_local[vi, vj] = 0.0
                                    C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block([2048, 2048], "C_local") as [vi, vj]:
                                    T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[vi, vj] = C_local[vi, vj]


@T.prim_func
def cuda_matmul_1(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    A_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    with T.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with T.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with T.block([2048, 2048], "A_shared_local") as [v0, v1]:
        A_shared_local[v0, v1] = A_shared[v0, v1]
    with T.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k_0 in T.serial(0, 256):
                                for k_1 in T.unroll(0, 8):
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k_0 * 8 + k_1)
                                            with T.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block([2048, 2048], "C_local") as [vi, vj]:
                                    T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[vi, vj] = C_local[vi, vj]


@T.prim_func
def cuda_matmul_2(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    A_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    with T.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with T.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with T.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k_0 in T.serial(0, 256):
                                for k_1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with T.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            T.bind(v0, k_0 * 8 + k_1 + i)
                                            T.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k_0 * 8 + k_1)
                                            with T.init():
                                                C_local[vi, vj] = T.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block([2048, 2048], "C_local") as [v0, v1]:
                                    T.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@T.prim_func
def cuda_matmul_3(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    A_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    with T.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with T.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with T.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            T.bind(v0, k0 * 8 + k1 + i)
                                            T.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with T.block([2048, 2048], "B_shared_local") as [v0, v1]:
                                            T.bind(v0, k0 * 8 + k1 + i)
                                            T.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k0 * 8 + k1)
                                            with T.init():
                                                C_local[vi, vj] = T.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block([2048, 2048], "C_local") as [v0, v1]:
                                    T.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@T.prim_func
def cuda_matmul_4(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    A_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    with T.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                for i, j in T.grid(8, 64):
                                    with T.block([2048, 2048], "A_shared") as [v0, v1]:
                                        T.bind(v0, k0 * 8 + i)
                                        T.bind(v1, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with T.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            T.bind(v0, k0 * 8 + k1 + i)
                                            T.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with T.block([2048, 2048], "B_shared_local") as [v0, v1]:
                                            T.bind(v0, k0 * 8 + k1 + i)
                                            T.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k0 * 8 + k1)
                                            with T.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block([2048, 2048], "C_local") as [v0, v1]:
                                    T.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@T.prim_func
def cuda_matmul_5(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    A_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = T.alloc_buffer([2048, 2048], "float32", scope="local")
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                for i, j in T.grid(8, 64):
                                    with T.block([2048, 2048], "A_shared") as [v0, v1]:
                                        T.bind(v0, k0 * 8 + i)
                                        T.bind(v1, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for i, j in T.grid(8, 64):
                                    with T.block([2048, 2048], "B_shared") as [v0, v1]:
                                        T.bind(v0, k0 * 8 + i)
                                        T.bind(v1, bx * 64 + j)
                                        B_shared[v0, v1] = B[v0, v1]
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with T.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            T.bind(v0, k0 * 8 + k1 + i)
                                            T.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with T.block([2048, 2048], "B_shared_local") as [v0, v1]:
                                            T.bind(v0, k0 * 8 + k1 + i)
                                            T.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k0 * 8 + k1)
                                            with T.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block([2048, 2048], "C_local") as [v0, v1]:
                                    T.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    T.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@T.prim_func
def tiled(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
        with T.block([128, 128], "B") as [vi, vj]:
            T.bind(vi, i_0 * 16 + i_1)
            T.bind(vj, j_0 * 16 + j_1)
            B[vi, vj] = A[vi, vj] * 2.0
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def tiled_after_reverse_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_0, j_0, i_1 in T.grid(8, 8, 16):
        for j_1 in T.serial(0, 16):
            with T.block([128, 128], "B") as [vi, vj]:
                T.bind(vi, i_0 * 16 + i_1)
                T.bind(vj, j_0 * 16 + j_1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j_1 in T.serial(0, 16):
            with T.block([128, 128], "C") as [vi, vj]:
                T.bind(vi, i_0 * 16 + i_1)
                T.bind(vj, j_0 * 16 + j_1)
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def factorized(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16, 16], "float32")
    B = T.match_buffer(b, [16], "float32")
    B_rf_local = T.alloc_buffer([16, 16], "float32", scope="local")
    for j in T.thread_binding(0, 16, thread = "blockIdx.x"):
        for i_o in T.thread_binding(0, 4, thread = "threadIdx.x"):
            for i_i, k in T.grid(4, 16):
                with T.block([16, 16, T.reduce_axis(0, 16)], "B_rf") as [vi, vj, vk]:
                    T.bind(vi, i_o * 4 + i_i)
                    T.bind(vj, j)
                    T.bind(vk, k)
                    with T.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
    for i, k in T.grid(16, 16):
        with T.block([16, T.reduce_axis(0, 16)], "B") as [vi, vk]:
            T.bind(vi, i)
            T.bind(vk, k)
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + B_rf_local[vk, vi]


@T.prim_func
def factorized_after_reverse_compute_at(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16, 16], "float32")
    B = T.match_buffer(b, [16], "float32")
    B_rf_local = T.alloc_buffer([16, 16], "float32", scope="local")
    for j in T.thread_binding(0, 16, thread = "blockIdx.x"):
        for i_o in T.thread_binding(0, 4, thread = "threadIdx.x"):
            for i_i, k in T.grid(4, 16):
                with T.block([16, 16, T.reduce_axis(0, 16)], "B_rf") as [vi, vj, vk]:
                    T.bind(vi, i_o * 4 + i_i)
                    T.bind(vj, j)
                    T.bind(vk, k)
                    with T.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
            for k in T.serial(0, 4):
                with T.block([16, T.reduce_axis(0, 16)], "B") as [vi, vk]:
                    T.bind(vi, j)
                    T.bind(vk, i_o * 4 + k)
                    with T.init():
                        B[vi] = 0.0
                    B[vi] = B[vi] + B_rf_local[vk, vi]


@T.prim_func
def fail_subtree_compact_dataflow(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i in range(0, 128):
        for j in range(0, 64):
            with T.block([128, 128], "B_0") as [vi, vj]:
                T.bind(vi, i)
                T.bind(vj, j)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 64):
            with T.block([128, 128], "B_1") as [vi, vj]:
                T.bind(vi, i)
                T.bind(vj, j + 64)
                B[vi, vj] = A[vi, vj] * 2.0
    with T.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def fail_all_consumers_under_loop(a: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    D = T.match_buffer(d, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "D") as [vi, vj]:
            D[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def fail_all_producers_under_loop(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.alloc_buffer((128, 128), "float32")
    D = T.match_buffer(d, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = A[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "D") as [vi, vj]:
            D[vi, vj] = B[vi, vj] + C[vi, vj]


@T.prim_func
def read_out_of_bound(a: T.handle, c:T.handle) -> None:
    A = T.match_buffer(a, [16], "float32")
    B = T.alloc_buffer([16], "float32")
    C = T.match_buffer(c, [16], "float32")
    for i in T.serial(0, 16):
        with T.block([16], "B") as [v]:
            B[v] = A[v]
    for j in T.serial(0, 16):
        with T.block([16], "C") as [v]:
            T.reads(B[v : v + 2])
            C[v] = T.if_then_else(v < 15, T.max(B[v], B[v + 1]), B[v], dtype="float32")


@T.prim_func
def read_out_of_bound_after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16], "float32")
    B = T.alloc_buffer([16], "float32")
    C = T.match_buffer(c, [16], "float32")
    for j in T.serial(0, 16):
        for i in T.serial(0, T.min(1, 15 - j) + 1):
            with T.block([16], "B") as [v]:
                T.bind(v, j + i)
                B[v] = A[v]
        with T.block([16], "C") as [v]:
            T.bind(v, j)
            T.reads([B[v : v + 2]])
            C[v] = T.if_then_else(v < 15, T.max(B[v], B[v + 1]), B[v], dtype="float32")


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on


def test_compute_at_two_elementwise():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("B")
    loop, _ = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(two_elementwise_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_compute_at_blockized_1():
    sch = tir.Schedule(blockized_1, debug_mask="all")
    block = sch.get_block("B")
    _, loop = sch.get_loops(sch.get_block("C_outer"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(blockized_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=blockized_1)


def test_compute_at_blockized_2():
    sch = tir.Schedule(blockized_2, debug_mask="all")
    block = sch.get_block("B_outer")
    _, loop, _, _ = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(blockized_2_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=blockized_2)


def test_compute_at_cuda_matmul_0():
    sch = tir.Schedule(cuda_matmul_0, debug_mask="all")
    block = sch.get_block("C")
    _, _, _, _, _, loop, _, _ = sch.get_loops(sch.get_block("C_local"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(cuda_matmul_0_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_0)


def test_compute_at_cuda_matmul_1():
    sch = tir.Schedule(cuda_matmul_1, debug_mask="all")
    block = sch.get_block("A_shared_local")
    _, _, _, _, _, _, _, loop, _, _, _ = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(cuda_matmul_2, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_1)


def test_compute_at_cuda_matmul_2():
    sch = tir.Schedule(cuda_matmul_2, debug_mask="all")
    block = sch.get_block("B_shared_local")
    _, _, _, _, _, _, _, loop, _, _, _ = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(cuda_matmul_3, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_2)


def test_compute_at_cuda_matmul_3():
    sch = tir.Schedule(cuda_matmul_3, debug_mask="all")
    block = sch.get_block("A_shared")
    _, _, _, _, _, _, loop, _, _, _, _ = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(cuda_matmul_4, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_3)


def test_compute_at_cuda_matmul_4():
    sch = tir.Schedule(cuda_matmul_4, debug_mask="all")
    block = sch.get_block("B_shared")
    _, _, _, _, _, _, loop, _, _, _, _ = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(cuda_matmul_5, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_4)


def test_reverse_compute_at_tiled():
    sch = tir.Schedule(tiled, debug_mask="all")
    block = sch.get_block("C")
    _, _, loop, _ = sch.get_loops(sch.get_block("B"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
    tvm.ir.assert_structural_equal(tiled_after_reverse_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=tiled)


def test_reverse_compute_at_blockized_2():
    sch = tir.Schedule(blockized_2, debug_mask="all")
    block = sch.get_block("C")
    _, loop = sch.get_loops(sch.get_block("B_outer"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(blockized_2_after_reverse_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=blockized_2)


def test_reverse_compute_at_factorized():
    sch = tir.Schedule(factorized, debug_mask="all")
    block = sch.get_block("B")
    _, loop, _, _ = sch.get_loops(sch.get_block("B_rf"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
    tvm.ir.assert_structural_equal(factorized_after_reverse_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=factorized)


def test_read_out_of_bound():
    sch = tir.Schedule(read_out_of_bound, debug_mask="all")
    block = sch.get_block("B")
    (loop,) = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop)
    tvm.ir.assert_structural_equal(read_out_of_bound_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=read_out_of_bound)


def test_fail_subtree_compact_dataflow():
    sch = tir.Schedule(fail_subtree_compact_dataflow, debug_mask="all")
    block = sch.get_block("B_0")
    loop, _ = sch.get_loops(sch.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError, match="compact dataflow"):
        sch.compute_at(block, loop)


def test_fail_not_in_same_scope():
    sch = tir.Schedule(blockized_1, debug_mask="all")
    block = sch.get_block("B")
    loop, _ = sch.get_loops(sch.get_block("C_inner"))
    with pytest.raises(tvm.tir.ScheduleError, match="same block scope"):
        sch.compute_at(block, loop)


def test_fail_loop_is_ancestor_of_block():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("B")
    loop, _ = sch.get_loops(sch.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError, match="ancestor of block"):
        sch.compute_at(block, loop)


def test_fail_output_block():
    sch = tir.Schedule(tiled, debug_mask="all")
    block = sch.get_block("C")
    loop, _, _, _ = sch.get_loops(sch.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError, match="output block"):
        sch.compute_at(block, loop)


def test_fail_all_consumers_under_loop():
    sch = tir.Schedule(fail_all_consumers_under_loop, debug_mask="all")
    block = sch.get_block("B")
    loop, _ = sch.get_loops(sch.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError, match="requires all the consumer"):
        sch.compute_at(block, loop)


def test_fail_all_producers_under_loop():
    sch = tir.Schedule(fail_all_producers_under_loop, debug_mask="all")
    block = sch.get_block("D")
    loop, _ = sch.get_loops(sch.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError, match="requires all the producer"):
        sch.reverse_compute_at(block, loop)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
