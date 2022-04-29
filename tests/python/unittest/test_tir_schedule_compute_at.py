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
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i in range(0, 128):
        for ax0, ax1 in T.grid(1, 128):
            with T.block("B"):
                vi = T.axis.S(128, i + ax0)
                vj = T.axis.S(128, ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_1(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(8, 8):
        with T.block("C_outer"):
            vi_o, vj_o = T.axis.remap("SS", [i, j])
            T.reads([B[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16,
            ]])
            T.writes([C[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with T.block("C_inner"):
                    vi = T.axis.S(128, vi_o * 16 + i_i)
                    vj = T.axis.S(128, vj_o * 16 + j_i)
                    C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i0_0, i1_0 in T.grid(8, 8):
        for ax0, ax1 in T.grid(16, 16):
            with T.block("B"):
                vi = T.axis.S(128, i0_0 * 16 + ax0)
                vj = T.axis.S(128, i1_0 * 16 + ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        with T.block("C_outer"):
            vi_o, vj_o = T.axis.remap("SS", [i0_0, i1_0])
            T.reads([B[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16,
            ]])
            T.writes([C[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16
            ]])
            for i0_1, i1_1 in T.grid(16, 16):
                with T.block("C_inner"):
                    vi = T.axis.S(128, vi_o * 16 + i0_1)
                    vj = T.axis.S(128, vj_o * 16 + i1_1)
                    C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_2(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in T.grid(8, 8):
        with T.block("B_outer"):
            vio, vjo = T.axis.remap("SS", [i_o, j_o])
            T.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            T.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with T.block("B_inner"):
                    vi = T.axis.S(128, vio * 16 + i_i)
                    vj = T.axis.S(128, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
    for i_o, j_o, i_i, j_i in T.grid(4, 4, 32, 32):
        with T.block("C"):
            vi = T.axis.S(128, i_o * 32 + i_i)
            vj = T.axis.S(128, j_o * 32 + j_i)
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockized_2_after_reverse_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in T.grid(8, 8):
        with T.block("B_outer"):
            vio, vjo = T.axis.remap("SS", [i_o, j_o])
            T.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            T.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with T.block("B_inner"):
                    vi = T.axis.S(128, vio * 16 + i_i)
                    vj = T.axis.S(128, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
        for ax0, ax1 in T.grid(16, 16):
            with T.block("C"):
                vi = T.axis.S(128, i_o * 16 + ax0)
                vj = T.axis.S(128, j_o * 16 + ax1)
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
            with T.block("blockized_B"):
                vio = T.axis.S(8, i_o * 2 + ax0)
                vjo = T.axis.S(8, j_o * 2 + ax1)
                T.reads([A[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                T.writes([B[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                for i_i, j_i in T.grid(16, 16):
                    with T.block("B"):
                        vi = T.axis.S(128, vio * 16 + i_i)
                        vj = T.axis.S(128, vjo * 16 + j_i)
                        B[vi, vj] = A[vi, vj] * 2.0
        for i_i, j_i in T.grid(32, 32):
            with T.block("C"):
                vi = T.axis.S(128, i_o * 32 + i_i)
                vj = T.axis.S(128, j_o * 32 + j_i)
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
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared_local"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared_local[v0, v1] = A_shared[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared_local"):
            v0, v1 = T.axis.remap("SS", [i, j])
            B_shared_local[v0, v1] = B_shared[v0, v1]
    for i, j, k in T.grid(2048, 2048, 2048):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
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
                                with T.block("C_local"):
                                    v0_4 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1_4 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
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
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared_local"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared_local[v0, v1] = A_shared[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared_local"):
            v0, v1 = T.axis.remap("SS", [i, j])
            B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for i, j, k in T.grid(4, 4, 2048):
                                with T.block("C"):
                                    vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    vk = T.axis.R(2048, k)
                                    with T.init():
                                        C_local[vi, vj] = 0.0
                                    C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block("C_local"):
                                    vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
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
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared_local"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared_local[v0, v1] = A_shared[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared_local"):
            v0, v1 = T.axis.remap("SS", [i, j])
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
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k_0 * 8 + k_1)
                                            with T.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block("C_local"):
                                    vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
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
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared_local"):
            v0, v1 = T.axis.remap("SS", [i, j])
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
                                        with T.block("A_shared_local"):
                                            v0 = T.axis.S(2048, k_0 * 8 + k_1 + i)
                                            v1 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k_0 * 8 + k_1)
                                            with T.init():
                                                C_local[vi, vj] = T.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block("C_local"):
                                    v0 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
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
    for i, j in T.grid(2048, 2048):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
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
                                        with T.block("A_shared_local"):
                                            v0 = T.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with T.block("B_shared_local"):
                                            v0 = T.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k0 * 8 + k1)
                                            with T.init():
                                                C_local[vi, vj] = T.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block("C_local"):
                                    v0 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
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
    for i, j in T.grid(2048, 2048):
        with T.block("B_shared"):
            v0, v1 = T.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                for i, j in T.grid(8, 64):
                                    with T.block("A_shared"):
                                        v0 = T.axis.S(2048, k0 * 8 + i)
                                        v1 = T.axis.S(2048, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with T.block("A_shared_local"):
                                            v0 = T.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with T.block("B_shared_local"):
                                            v0 = T.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k0 * 8 + k1)
                                            with T.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block("C_local"):
                                    v0 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
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
                                    with T.block("A_shared"):
                                        v0 = T.axis.S(2048, k0 * 8 + i)
                                        v1 = T.axis.S(2048, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for i, j in T.grid(8, 64):
                                    with T.block("B_shared"):
                                        v0 = T.axis.S(2048, k0 * 8 + i)
                                        v1 = T.axis.S(2048, bx * 64 + j)
                                        B_shared[v0, v1] = B[v0, v1]
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with T.block("A_shared_local"):
                                            v0 = T.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with T.block("B_shared_local"):
                                            v0 = T.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k0 * 8 + k1)
                                            with T.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with T.block("C_local"):
                                    v0 = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@T.prim_func
def tiled(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
        with T.block("B"):
            vi = T.axis.S(128, i_0 * 16 + i_1)
            vj = T.axis.S(128, j_0 * 16 + j_1)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def tiled_after_reverse_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], "float32")
    B = T.alloc_buffer([128, 128], "float32")
    C = T.match_buffer(c, [128, 128], "float32")
    for i_0, j_0, i_1 in T.grid(8, 8, 16):
        for j_1 in T.serial(0, 16):
            with T.block("B"):
                vi = T.axis.S(128, i_0 * 16 + i_1)
                vj = T.axis.S(128, j_0 * 16 + j_1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j_1 in T.serial(0, 16):
            with T.block("C"):
                vi = T.axis.S(128, i_0 * 16 + i_1)
                vj = T.axis.S(128, j_0 * 16 + j_1)
                C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def factorized(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [16, 16, 16], "float32")
    B = T.match_buffer(b, [16], "float32")
    B_rf_local = T.alloc_buffer([16, 16], "float32", scope="local")
    for j in T.thread_binding(0, 16, thread = "blockIdx.x"):
        for i_o in T.thread_binding(0, 4, thread = "threadIdx.x"):
            for i_i, k in T.grid(4, 16):
                with T.block("B_rf"):
                    vi = T.axis.S(16, i_o * 4 + i_i)
                    vj, vk = T.axis.remap("SR", [j, k])
                    with T.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
    for i, k in T.grid(16, 16):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
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
                with T.block("B_rf"):
                    vi = T.axis.S(16, i_o * 4 + i_i)
                    vj = T.axis.S(16, j)
                    vk = T.axis.R(16, k)
                    with T.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
            for k in T.serial(0, 4):
                with T.block("B"):
                    vi = T.axis.S(16, j)
                    vk = T.axis.R(16, i_o * 4 + k)
                    with T.init():
                        B[vi] = 0.0
                    B[vi] = B[vi] + B_rf_local[vk, vi]


@T.prim_func
def not_all_compact_data_flow(a: T.handle, c: T.handle):
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 64):
        with T.block("C_1"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj * 2] = B[vi, vj * 2] + 1.0
        with T.block("C_2"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj * 2 + 1] = B[vi, vj * 2 + 1] * 2.0


@T.prim_func
def not_all_compact_data_flow_after_compute_at(a: T.handle, c: T.handle):
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 64):
        for t in range(2):
            with T.block("B"):
                vi = T.axis.S(128, i)
                vj = T.axis.S(128, j * 2 + t)
                B[vi, vj] = A[vi, vj]
        with T.block("C_1"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj * 2] = B[vi, vj * 2] + 1.0
        with T.block("C_2"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj * 2 + 1] = B[vi, vj * 2 + 1] * 2.0


@T.prim_func
def fail_subtree_compact_dataflow(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i in range(0, 128):
        for j in range(0, 64):
            with T.block("B_0"):
                vi = T.axis.S(128, i)
                vj = T.axis.S(128, j)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 64):
            with T.block("B_1"):
                vi = T.axis.S(128, i)
                vj = T.axis.S(128, j + 64)
                B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def fail_all_consumers_under_loop(a: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    D = T.match_buffer(d, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block("D"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def fail_all_producers_under_loop(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.alloc_buffer((128, 128), "float32")
    D = T.match_buffer(d, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with T.block("D"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + C[vi, vj]


@T.prim_func
def read_out_of_bound(a: T.handle, c:T.handle) -> None:
    A = T.match_buffer(a, [16], "float32")
    B = T.alloc_buffer([16], "float32")
    C = T.match_buffer(c, [16], "float32")
    for i in T.serial(0, 16):
        with T.block("B"):
            v = T.axis.S(16, i)
            B[v] = A[v]
    for j in T.serial(0, 16):
        with T.block("C"):
            v = T.axis.S(16, j)
            T.reads(B[v : v + 2])
            C[v] = T.if_then_else(v < 15, T.max(B[v], B[v + 1]), B[v], dtype="float32")


@T.prim_func
def read_out_of_bound_after_compute_at(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16], "float32")
    B = T.alloc_buffer([16], "float32")
    C = T.match_buffer(c, [16], "float32")
    for j in T.serial(0, 16):
        for i in T.serial(0, 2):
            with T.block("B"):
                v = T.axis.S(16, j + i)
                T.where(j + i < 16)
                B[v] = A[v]
        with T.block("C"):
            v = T.axis.S(16, j)
            T.reads([B[v : v + 2]])
            C[v] = T.if_then_else(v < 15, T.max(B[v], B[v + 1]), B[v], dtype="float32")


@T.prim_func
def multi_reduction(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(), "float32"]):
    B = T.alloc_buffer((16, ), dtype="float32")
    for i, k in T.grid(16, 16):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi] = 0.0
            B[vi] += A[vi, vk]
    for k in T.grid(16):
        with T.block("C"):
            vk = T.axis.remap("R", [k])
            with T.init():
                C[()] = 0.0
            C[()] += B[vk]


@T.prim_func
def multi_reduction_after_compute_at(
    A: T.Buffer[(16, 16), "float32"],
    C:T.Buffer[(), "float32"],
):
    B = T.alloc_buffer((16, ), dtype="float32")
    for k in T.grid(16):
        for kk in T.grid(16):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [k, kk])
                with T.init():
                    B[vi] = 0.0
                B[vi] += A[vi, vk]
        with T.block("C"):
            vk = T.axis.remap("R", [k])
            with T.init():
                C[()] = 0.0
            C[()] += B[vk]


@T.prim_func
def tiled_pooling_read_cache(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [224, 224], dtype="float32")
    Y = T.match_buffer(b, [224, 224], dtype="float32")
    cache = T.alloc_buffer([224, 224], dtype="float32")
    for hh, ww in T.grid(224, 224):
        with T.block("cache"):
            h, w = T.axis.remap("SS", [hh, ww])
            cache[h, w] = X[h, w]
    for hh_0, ww_0, hh_1, ww_1, khh, kww in T.grid(28, 28, 8, 8, 3, 3):
        with T.block("compute"):
            h = T.axis.spatial(224, hh_0 * 8 + hh_1)
            w = T.axis.spatial(224, ww_0 * 8 + ww_1)
            kh, kw = T.axis.remap("RR", [khh, kww])
            with T.init():
                Y[h, w] = 0.0
            Y[h, w] = T.max(Y[h, w], T.if_then_else(
                T.likely(1 <= h + kh, dtype="bool") and \
                T.likely(h + kh < 225, dtype="bool") and \
                T.likely(1 <= w + kw, dtype="bool") and \
                T.likely(w + kw < 225, dtype="bool"),
                cache[h + kh - 1, w + kw - 1], 0.0, dtype="float32"))

@T.prim_func
def tiled_pooling_read_cache_after_compute_at(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [224, 224], dtype="float32")
    Y = T.match_buffer(b, [224, 224], dtype="float32")
    cache = T.alloc_buffer([224, 224], dtype="float32")
    for hh_0, ww_0 in T.grid(28, 28):
        for ax0, ax1 in T.grid(10, 10):
            with T.block("cache"):
                h = T.axis.spatial(224, hh_0 * 8 - 1 + ax0)
                w = T.axis.spatial(224, ww_0 * 8 - 1 + ax1)
                T.where(1 <= hh_0 * 8 + ax0 and hh_0 * 8 + ax0 < 225 and 1 <= ww_0 * 8 + ax1 and ww_0 * 8 + ax1 < 225)
                cache[h, w] = X[h, w]
        for hh_1, ww_1, khh, kww in T.grid(8, 8, 3, 3):
            with T.block("compute"):
                h = T.axis.spatial(224, hh_0 * 8 + hh_1)
                w = T.axis.spatial(224, ww_0 * 8 + ww_1)
                kh, kw = T.axis.remap("RR", [khh, kww])
                with T.init():
                    Y[h, w] = 0.0
                Y[h, w] = T.max(Y[h, w], T.if_then_else(
                    T.likely(1 <= h + kh, dtype="bool") and \
                    T.likely(h + kh < 225, dtype="bool") and \
                    T.likely(1 <= w + kw, dtype="bool") and \
                    T.likely(w + kw < 225, dtype="bool"),
                    cache[h + kh - 1, w + kw - 1], 0.0, dtype="float32"))

@T.prim_func
def non_uniform_tiled_conv(x: T.Buffer[(1, 3, 100, 100), "float32"],
                           w: T.Buffer[(16, 3, 3, 3), "float32"],
                           y: T.Buffer[(1, 16, 98, 98), "float32"]) -> None:
    x_global = T.alloc_buffer([1, 3, 100, 100], dtype="float32")
    for ax0, ax1, ax2, ax3 in T.grid(1, 3, 100, 100):
        with T.block("cache"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
    for h_o, w_o, n, c_o, h_i, w_i, c_i, kh, kw in T.grid(7, 7, 1, 16, 15, 15, 3, 3, 3):
        with T.block("compute"):
            nn = T.axis.spatial(1, 0)
            cc = T.axis.spatial(16, c_o)
            hh = T.axis.spatial(98, h_o * 15 + h_i)
            ww = T.axis.spatial(98, w_o * 15 + w_i)
            rc, rh, rw = T.axis.remap("RRR", [c_i, kh, kw])
            T.where(h_o * 15 + h_i < 98 and w_o * 15 + w_i < 98)
            with T.init():
                y[nn, cc, hh, ww] = T.float32(0)
            y[nn, cc, hh, ww] = y[nn, cc, hh, ww] + \
                x_global[nn, cc // 16 * 3 + rc, hh + rh, ww + rw] * w[cc, rc, rh, rw]

@T.prim_func
def non_uniform_tiled_conv_after_compute_at(x: T.Buffer[(1, 3, 100, 100), "float32"],
                                            w: T.Buffer[(16, 3, 3, 3), "float32"],
                                            y: T.Buffer[(1, 16, 98, 98), "float32"]) -> None:
    x_global = T.alloc_buffer([1, 3, 100, 100], dtype="float32")
    for h_o, w_o in T.grid(7, 7):
        for ax0, ax1, ax2 in T.grid(3, 17, 17):
            with T.block("cache"):
                v0 = T.axis.spatial(1, 0)
                v1 = T.axis.spatial(3, ax0)
                v2 = T.axis.spatial(100, h_o * 15 + ax1)
                v3 = T.axis.spatial(100, w_o * 15 + ax2)
                T.where(h_o * 15 + ax1 < 100 and w_o * 15 + ax2 < 100)
                x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
        for n, c_o, h_i, w_i, c_i, kh, kw in T.grid(1, 16, 15, 15, 3, 3, 3):
            with T.block("compute"):
                nn = T.axis.spatial(1, 0)
                cc = T.axis.spatial(16, c_o)
                hh = T.axis.spatial(98, h_o * 15 + h_i)
                ww = T.axis.spatial(98, w_o * 15 + w_i)
                rc, rh, rw = T.axis.remap("RRR", [c_i, kh, kw])
                T.where(h_o * 15 + h_i < 98 and w_o * 15 + w_i < 98)
                with T.init():
                    y[nn, cc, hh, ww] = T.float32(0)
                y[nn, cc, hh, ww] = y[nn, cc, hh, ww] + \
                    x_global[nn, cc // 16 * 3 + rc, hh + rh, ww + rw] * w[cc, rc, rh, rw]

@T.prim_func
def concat_two_elemwise(x: T.Buffer[(16,), "float32"],
                        y: T.Buffer[(8,), "float32"],
                        T_concat: T.Buffer[(24,), "float32"]) -> None:
    T_add_1 = T.alloc_buffer([16], dtype="float32")
    T_add_2 = T.alloc_buffer([8], dtype="float32")
    for i in T.serial(16):
        with T.block("T_add_1"):
            ax = T.axis.spatial(16, i)
            T_add_1[ax] = x[ax] + T.float32(1)
    for i in T.serial(8):
        with T.block("T_add_2"):
            ax = T.axis.spatial(8, i)
            T_add_2[ax] = y[ax] + T.float32(2)
    for i in T.serial(24):
        with T.block("T_concat"):
            ax = T.axis.spatial(24, i)
            T_concat[ax] = T.if_then_else(16 <= ax, T_add_2[ax - 16], T_add_1[ax], dtype="float32")

@T.prim_func
def concat_two_elemwise_after_compute_at(x: T.Buffer[(16,), "float32"],
                                         y: T.Buffer[(8,), "float32"],
                                         T_concat: T.Buffer[(24,), "float32"]) -> None:
    T_add_1 = T.alloc_buffer([16], dtype="float32")
    T_add_2 = T.alloc_buffer([8], dtype="float32")
    for i in T.serial(24):
        with T.block("T_add_1"):
            ax = T.axis.spatial(16, i)
            T.where(i < 16)
            T_add_1[ax] = x[ax] + T.float32(1)
        with T.block("T_add_2"):
            ax = T.axis.spatial(8, i - 16)
            T.where(16 <= i)
            T_add_2[ax] = y[ax] + T.float32(2)
        with T.block("T_concat"):
            ax = T.axis.spatial(24, i)
            T_concat[ax] = T.if_then_else(16 <= ax, T_add_2[ax - 16], T_add_1[ax], dtype="float32")

@T.prim_func
def floordiv_and_floormod_indices(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [16, 16])
    Y = T.match_buffer(b, [256])
    temp = T.alloc_buffer([16, 16])
    for i, j in T.grid(16, 16):
        with T.block("A"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            temp[v_i, v_j] = X[v_j, v_i] + 1.0
    for i in T.serial(0, 256):
        with T.block("B"):
            v_i = T.axis.remap("S", [i])
            Y[v_i] = temp[v_i // 16, v_i % 16]

@T.prim_func
def floordiv_and_floormod_indices_after_reverse_compute_at(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [16, 16], dtype="float32")
    Y = T.match_buffer(b, [256], dtype="float32")
    temp = T.alloc_buffer([16, 16], dtype="float32")
    for i in T.serial(0, 16):
        for j in T.serial(0, 16):
            with T.block("A"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                temp[v_i, v_j] = X[v_j, v_i] + T.float32(1)
        for ax0 in T.serial(0, 16):
            with T.block("B"):
                v_i = T.axis.spatial(256, i * 16 + ax0)
                Y[v_i] = temp[v_i // 16, v_i % 16]


@T.prim_func
def tiled_repeat_op(x: T.Buffer[(4,), "float32"], T_repeat: T.Buffer[(64,), "float32"]) -> None:
    T_add = T.alloc_buffer([4], dtype="float32")
    for i0 in T.serial(4):
        with T.block("T_add"):
            ax0 = T.axis.spatial(4, i0)
            T_add[ax0] = x[ax0] + 1.0
    for i0_0, i0_1 in T.grid(8, 8):
        with T.block("T_repeat"):
            ax0 = T.axis.spatial(64, i0_0 * 8 + i0_1)
            T_repeat[ax0] = T_add[ax0 // 16]

@T.prim_func
def tiled_repeat_op_after_compute_at(x: T.Buffer[(4,), "float32"], T_repeat: T.Buffer[(64,), "float32"]) -> None:
    T_add = T.alloc_buffer([4], dtype="float32")
    for i0_0 in T.serial(8):
        with T.block("T_add"):
            ax0 = T.axis.spatial(4, i0_0 // 2)
            T_add[ax0] = x[ax0] + T.float32(1)
        for i0_1 in T.serial(8):
            with T.block("T_repeat"):
                ax0 = T.axis.spatial(64, i0_0 * 8 + i0_1)
                T_repeat[ax0] = T_add[ax0 // 16]

@T.prim_func
def static_bound(A: T.Buffer[(32, 1), "float32"], C: T.Buffer[(32, 1), "float32"]) -> None:
    B = T.alloc_buffer((32, 1), "float32")
    for i, j in T.grid(32, 1):
        with T.block("B"):
            vi = T.axis.spatial(32, i)
            vj = T.axis.spatial(1, j)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(32, 32):
        with T.block("C"):
            vi = T.axis.spatial(32, i)
            vj = T.axis.spatial(1, j)
            T.where(j < 1)
            C[vi, vj] = B[vi, vj] + 1.0

@T.prim_func
def static_bound_after_compute_at(A: T.Buffer[(32, 1), "float32"], C: T.Buffer[(32, 1), "float32"]) -> None:
    B = T.alloc_buffer((32, 1), "float32")
    for i in range(32):
        for ax0, ax1 in T.grid(1, 1):
            with T.block("B"):
                vi = T.axis.spatial(32, i + ax0)
                vj = T.axis.spatial(1, ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(32):
            with T.block("C"):
                vi = T.axis.spatial(32, i)
                vj = T.axis.spatial(1, j)
                T.where(j < 1)
                C[vi, vj] = B[vi, vj] + 1.0
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


def test_compute_at_reduction_block():
    sch = tir.Schedule(multi_reduction, debug_mask="all")
    block = sch.get_block("B")
    (loop,) = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=False)
    tvm.ir.assert_structural_equal(multi_reduction_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=multi_reduction)


def test_compute_at_tiled_pooling_read_cache():
    sch = tir.Schedule(tiled_pooling_read_cache, debug_mask="all")
    compute = sch.get_block("compute")
    _, w_o, _, _, _, _ = sch.get_loops(compute)
    cache = sch.get_block("cache")
    sch.compute_at(cache, w_o)
    tvm.ir.assert_structural_equal(tiled_pooling_read_cache_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=tiled_pooling_read_cache)


def test_compute_at_non_uniform_tiled_conv():
    sch = tir.Schedule(non_uniform_tiled_conv, debug_mask="all")
    compute = sch.get_block("compute")
    sch.compute_at(sch.get_block("cache"), sch.get_loops(compute)[1])
    tvm.ir.assert_structural_equal(non_uniform_tiled_conv_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=non_uniform_tiled_conv)


def test_compute_at_concat():
    sch = tir.Schedule(concat_two_elemwise, debug_mask="all")
    concat = sch.get_block("T_concat")
    add1 = sch.get_block("T_add_1")
    add2 = sch.get_block("T_add_2")
    axis = sch.get_loops(concat)[0]
    sch.compute_at(add1, axis)
    sch.compute_at(add2, axis)
    tvm.ir.assert_structural_equal(concat_two_elemwise_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=concat_two_elemwise)


def test_compute_at_tiled_repeat_op():
    sch = tir.Schedule(tiled_repeat_op, debug_mask="all")
    outer_ax, _ = sch.get_loops(sch.get_block("T_repeat"))
    sch.compute_at(sch.get_block("T_add"), outer_ax)
    tvm.ir.assert_structural_equal(tiled_repeat_op_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=tiled_repeat_op)


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


def test_reverse_compute_at_floordiv_and_floormod_indices():
    sch = tir.Schedule(floordiv_and_floormod_indices, debug_mask="all")
    A = sch.get_block("A")
    B = sch.get_block("B")
    sch.reverse_compute_at(B, sch.get_loops(A)[0])
    tvm.ir.assert_structural_equal(
        floordiv_and_floormod_indices_after_reverse_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=floordiv_and_floormod_indices)


def test_read_out_of_bound():
    sch = tir.Schedule(read_out_of_bound, debug_mask="all")
    block = sch.get_block("B")
    (loop,) = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop)
    tvm.ir.assert_structural_equal(read_out_of_bound_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=read_out_of_bound)


def test_compact_dataflow():
    sch = tir.Schedule(not_all_compact_data_flow, debug_mask="all")
    block = sch.get_block("B")
    _, loop = sch.get_loops(sch.get_block("C_1"))
    sch.compute_at(block, loop)
    tvm.ir.assert_structural_equal(not_all_compact_data_flow_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=not_all_compact_data_flow)


def test_compute_at_simplify_static_bound():
    sch = tir.Schedule(static_bound, debug_mask="all")
    block = sch.get_block("B")
    loop, _ = sch.get_loops(sch.get_block("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    tvm.ir.assert_structural_equal(static_bound_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=static_bound)


def test_fail_subtree_complete_block():
    sch = tir.Schedule(fail_subtree_compact_dataflow, debug_mask="all")
    block = sch.get_block("B_0")
    loop, _ = sch.get_loops(sch.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError, match="complete block"):
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
