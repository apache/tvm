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
from tvm.script import ty
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@tvm.script.tir
def two_elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def two_elementwise_after_compute_at(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    for i in range(0, 128):
        for ax0, ax1 in tir.grid(1, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i + ax0)
                tir.bind(vj, ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def blockized_1(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], "float32")
    B = tir.alloc_buffer([128, 128], "float32")
    C = tir.match_buffer(c, [128, 128], "float32")
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([8, 8], "C_outer") as [vi_o, vj_o]:
        tir.reads([B[
            vi_o * 16 : vi_o * 16 + 16,
            vj_o * 16 : vj_o * 16 + 16,
        ]])
        tir.writes([C[
            vi_o * 16 : vi_o * 16 + 16,
            vj_o * 16 : vj_o * 16 + 16
        ]])
        for i_i, j_i in tir.grid(16, 16):
            with tir.block([128, 128], "C_inner") as [vi, vj]:
                tir.bind(vi, vi_o * 16 + i_i)
                tir.bind(vj, vj_o * 16 + j_i)
                C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def blockized_after_compute_at(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], "float32")
    B = tir.alloc_buffer([128, 128], "float32")
    C = tir.match_buffer(c, [128, 128], "float32")
    for i0_0, i1_0 in tir.grid(8, 8):
        for ax0, ax1 in tir.grid(16, 16):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i0_0 * 16 + ax0)
                tir.bind(vj, i1_0 * 16 + ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        with tir.block([8, 8], "C_outer") as [vi_o, vj_o]:
            tir.bind(vi_o, i0_0)
            tir.bind(vj_o, i1_0)
            tir.reads([B[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16,
            ]])
            tir.writes([C[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16
            ]])
            for i0_1, i1_1 in tir.grid(16, 16):
                with tir.block([128, 128], "C_inner") as [vi, vj]:
                    tir.bind(vi, vi_o * 16 + i0_1)
                    tir.bind(vj, vj_o * 16 + i1_1)
                    C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def blockized_2(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], "float32")
    B = tir.alloc_buffer([128, 128], "float32")
    C = tir.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in tir.grid(8, 8):
        with tir.block([8, 8], "B_outer") as [vio, vjo]:
            tir.bind(vio, i_o)
            tir.bind(vjo, j_o)
            tir.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            tir.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in tir.grid(16, 16):
                with tir.block([128, 128], "B_inner") as [vi, vj]:
                    tir.bind(vi, vio * 16 + i_i)
                    tir.bind(vj, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
    for i_o, j_o, i_i, j_i in tir.grid(4, 4, 32, 32):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.bind(vi, i_o * 32 + i_i)
            tir.bind(vj, j_o * 32 + j_i)
            C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def blockized_2_after_reverse_compute_at(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], "float32")
    B = tir.alloc_buffer([128, 128], "float32")
    C = tir.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in tir.grid(8, 8):
        with tir.block([8, 8], "B_outer") as [vio, vjo]:
            tir.bind(vio, i_o)
            tir.bind(vjo, j_o)
            tir.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            tir.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in tir.grid(16, 16):
                with tir.block([128, 128], "B_inner") as [vi, vj]:
                    tir.bind(vi, vio * 16 + i_i)
                    tir.bind(vj, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
        for ax0, ax1 in tir.grid(16, 16):
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i_o * 16 + ax0)
                tir.bind(vj, j_o * 16 + ax1)
                tir.reads([B[vi, vj]])
                tir.writes([C[vi, vj]])
                C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def blockized_2_after_compute_at(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], "float32")
    B = tir.alloc_buffer([128, 128], "float32")
    C = tir.match_buffer(c, [128, 128], "float32")
    for i_o, j_o in tir.grid(4, 4):
        for ax0, ax1 in tir.grid(2, 2):
            with tir.block([8, 8], "blockized_B") as [vio, vjo]:
                tir.bind(vio, i_o * 2 + ax0)
                tir.bind(vjo, j_o * 2 + ax1)
                tir.reads([A[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                tir.writes([B[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                for i_i, j_i in tir.grid(16, 16):
                    with tir.block([128, 128], "B") as [vi, vj]:
                        tir.bind(vi, vio * 16 + i_i)
                        tir.bind(vj, vjo * 16 + j_i)
                        B[vi, vj] = A[vi, vj] * 2.0
        for i_i, j_i in tir.grid(32, 32):
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i_o * 32 + i_i)
                tir.bind(vj, j_o * 32 + j_i)
                C[vi, vj] = B[vi, vj] + 1.0

@tvm.script.tir
def cuda_matmul_0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = tir.match_buffer(a, [2048, 2048], "float32")
    B = tir.match_buffer(b, [2048, 2048], "float32")
    C = tir.match_buffer(c, [2048, 2048], "float32")
    A_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    with tir.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with tir.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with tir.block([2048, 2048], "A_shared_local") as [v0, v1]:
        A_shared_local[v0, v1] = A_shared[v0, v1]
    with tir.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
        with tir.init():
            C_local[vi, vj] = 0.0
        C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
    for by in tir.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in tir.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in tir.thread_binding(0, 2, thread = "vthread.y"):
                for vx in tir.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in tir.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in tir.thread_binding(0, 8, thread = "threadIdx.x"):
                            for i, j in tir.grid(4, 4):
                                with tir.block([2048, 2048], "C_local") as [v0_4, v1_4]:
                                    tir.bind(v0_4, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(v1_4, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0_4, v1_4] = C_local[v0_4, v1_4]


@tvm.script.tir
def cuda_matmul_0_after_compute_at(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = tir.match_buffer(a, [2048, 2048], "float32")
    B = tir.match_buffer(b, [2048, 2048], "float32")
    C = tir.match_buffer(c, [2048, 2048], "float32")
    A_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    with tir.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with tir.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with tir.block([2048, 2048], "A_shared_local") as [v0, v1]:
        A_shared_local[v0, v1] = A_shared[v0, v1]
    with tir.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in tir.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in tir.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in tir.thread_binding(0, 2, thread = "vthread.y"):
                for vx in tir.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in tir.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in tir.thread_binding(0, 8, thread = "threadIdx.x"):
                            for i, j, k in tir.grid(4, 4, 2048):
                                with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                    tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                    tir.bind(vk, k)
                                    with tir.init():
                                        C_local[vi, vj] = 0.0
                                    C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in tir.grid(4, 4):
                                with tir.block([2048, 2048], "C_local") as [vi, vj]:
                                    tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[vi, vj] = C_local[vi, vj]


@tvm.script.tir
def cuda_matmul_1(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = tir.match_buffer(a, [2048, 2048], "float32")
    B = tir.match_buffer(b, [2048, 2048], "float32")
    C = tir.match_buffer(c, [2048, 2048], "float32")
    A_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    with tir.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with tir.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with tir.block([2048, 2048], "A_shared_local") as [v0, v1]:
        A_shared_local[v0, v1] = A_shared[v0, v1]
    with tir.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in tir.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in tir.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in tir.thread_binding(0, 2, thread = "vthread.y"):
                for vx in tir.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in tir.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in tir.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k_0 in tir.serial(0, 256):
                                for k_1 in tir.unroll(0, 8):
                                    for _, i, j in tir.grid(1, 4, 4):
                                        with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            tir.bind(vk, k_0 * 8 + k_1)
                                            with tir.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in tir.grid(4, 4):
                                with tir.block([2048, 2048], "C_local") as [vi, vj]:
                                    tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[vi, vj] = C_local[vi, vj]


@tvm.script.tir
def cuda_matmul_2(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = tir.match_buffer(a, [2048, 2048], "float32")
    B = tir.match_buffer(b, [2048, 2048], "float32")
    C = tir.match_buffer(c, [2048, 2048], "float32")
    A_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    with tir.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with tir.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    with tir.block([2048, 2048], "B_shared_local") as [v0, v1]:
        B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in tir.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in tir.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in tir.thread_binding(0, 2, thread = "vthread.y"):
                for vx in tir.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in tir.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in tir.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k_0 in tir.serial(0, 256):
                                for k_1 in tir.unroll(0, 8):
                                    for i, j in tir.grid(1, 4):
                                        with tir.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            tir.bind(v0, k_0 * 8 + k_1 + i)
                                            tir.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for _, i, j in tir.grid(1, 4, 4):
                                        with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            tir.bind(vk, k_0 * 8 + k_1)
                                            with tir.init():
                                                C_local[vi, vj] = tir.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in tir.grid(4, 4):
                                with tir.block([2048, 2048], "C_local") as [v0, v1]:
                                    tir.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def cuda_matmul_3(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = tir.match_buffer(a, [2048, 2048], "float32")
    B = tir.match_buffer(b, [2048, 2048], "float32")
    C = tir.match_buffer(c, [2048, 2048], "float32")
    A_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    with tir.block([2048, 2048], "A_shared") as [v0, v1]:
        A_shared[v0, v1] = A[v0, v1]
    with tir.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    for by in tir.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in tir.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in tir.thread_binding(0, 2, thread = "vthread.y"):
                for vx in tir.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in tir.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in tir.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in tir.serial(0, 256):
                                for k1 in tir.unroll(0, 8):
                                    for i, j in tir.grid(1, 4):
                                        with tir.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            tir.bind(v0, k0 * 8 + k1 + i)
                                            tir.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in tir.grid(1, 4):
                                        with tir.block([2048, 2048], "B_shared_local") as [v0, v1]:
                                            tir.bind(v0, k0 * 8 + k1 + i)
                                            tir.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in tir.grid(1, 4, 4):
                                        with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            tir.bind(vk, k0 * 8 + k1)
                                            with tir.init():
                                                C_local[vi, vj] = tir.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in tir.grid(4, 4):
                                with tir.block([2048, 2048], "C_local") as [v0, v1]:
                                    tir.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def cuda_matmul_4(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = tir.match_buffer(a, [2048, 2048], "float32")
    B = tir.match_buffer(b, [2048, 2048], "float32")
    C = tir.match_buffer(c, [2048, 2048], "float32")
    A_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    with tir.block([2048, 2048], "B_shared") as [v0, v1]:
        B_shared[v0, v1] = B[v0, v1]
    for by in tir.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in tir.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in tir.thread_binding(0, 2, thread = "vthread.y"):
                for vx in tir.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in tir.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in tir.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in tir.serial(0, 256):
                                for i, j in tir.grid(8, 64):
                                    with tir.block([2048, 2048], "A_shared") as [v0, v1]:
                                        tir.bind(v0, k0 * 8 + i)
                                        tir.bind(v1, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for k1 in tir.unroll(0, 8):
                                    for i, j in tir.grid(1, 4):
                                        with tir.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            tir.bind(v0, k0 * 8 + k1 + i)
                                            tir.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in tir.grid(1, 4):
                                        with tir.block([2048, 2048], "B_shared_local") as [v0, v1]:
                                            tir.bind(v0, k0 * 8 + k1 + i)
                                            tir.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in tir.grid(1, 4, 4):
                                        with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            tir.bind(vk, k0 * 8 + k1)
                                            with tir.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in tir.grid(4, 4):
                                with tir.block([2048, 2048], "C_local") as [v0, v1]:
                                    tir.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def cuda_matmul_5(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = tir.match_buffer(a, [2048, 2048], "float32")
    B = tir.match_buffer(b, [2048, 2048], "float32")
    C = tir.match_buffer(c, [2048, 2048], "float32")
    A_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    B_shared = tir.alloc_buffer([2048, 2048], "float32", scope="shared")
    A_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    B_shared_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    C_local = tir.alloc_buffer([2048, 2048], "float32", scope="local")
    for by in tir.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in tir.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in tir.thread_binding(0, 2, thread = "vthread.y"):
                for vx in tir.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in tir.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in tir.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in tir.serial(0, 256):
                                for i, j in tir.grid(8, 64):
                                    with tir.block([2048, 2048], "A_shared") as [v0, v1]:
                                        tir.bind(v0, k0 * 8 + i)
                                        tir.bind(v1, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for i, j in tir.grid(8, 64):
                                    with tir.block([2048, 2048], "B_shared") as [v0, v1]:
                                        tir.bind(v0, k0 * 8 + i)
                                        tir.bind(v1, bx * 64 + j)
                                        B_shared[v0, v1] = B[v0, v1]
                                for k1 in tir.unroll(0, 8):
                                    for i, j in tir.grid(1, 4):
                                        with tir.block([2048, 2048], "A_shared_local") as [v0, v1]:
                                            tir.bind(v0, k0 * 8 + k1 + i)
                                            tir.bind(v1, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in tir.grid(1, 4):
                                        with tir.block([2048, 2048], "B_shared_local") as [v0, v1]:
                                            tir.bind(v0, k0 * 8 + k1 + i)
                                            tir.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in tir.grid(1, 4, 4):
                                        with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            tir.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            tir.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            tir.bind(vk, k0 * 8 + k1)
                                            with tir.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in tir.grid(4, 4):
                                with tir.block([2048, 2048], "C_local") as [v0, v1]:
                                    tir.bind(v0, by * 64 + vy * 32 + ty * 4 + i)
                                    tir.bind(v1, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def tiled(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], "float32")
    B = tir.alloc_buffer([128, 128], "float32")
    C = tir.match_buffer(c, [128, 128], "float32")
    for i_0, j_0, i_1, j_1 in tir.grid(8, 8, 16, 16):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, i_0 * 16 + i_1)
            tir.bind(vj, j_0 * 16 + j_1)
            B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def tiled_after_reverse_compute_at(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128], "float32")
    B = tir.alloc_buffer([128, 128], "float32")
    C = tir.match_buffer(c, [128, 128], "float32")
    for i_0, j_0, i_1 in tir.grid(8, 8, 16):
        for j_1 in tir.serial(0, 16):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i_0 * 16 + i_1)
                tir.bind(vj, j_0 * 16 + j_1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j_1 in tir.serial(0, 16):
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i_0 * 16 + i_1)
                tir.bind(vj, j_0 * 16 + j_1)
                C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def factorized(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16, 16], "float32")
    B = tir.match_buffer(b, [16], "float32")
    B_rf_local = tir.alloc_buffer([16, 16], "float32", scope="local")
    for j in tir.thread_binding(0, 16, thread = "blockIdx.x"):
        for i_o in tir.thread_binding(0, 4, thread = "threadIdx.x"):
            for i_i, k in tir.grid(4, 16):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "B_rf") as [vi, vj, vk]:
                    tir.bind(vi, i_o * 4 + i_i)
                    tir.bind(vj, j)
                    tir.bind(vk, k)
                    with tir.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
    for i, k in tir.grid(16, 16):
        with tir.block([16, tir.reduce_axis(0, 16)], "B") as [vi, vk]:
            tir.bind(vi, i)
            tir.bind(vk, k)
            with tir.init():
                B[vi] = 0.0
            B[vi] = B[vi] + B_rf_local[vk, vi]


@tvm.script.tir
def factorized_after_reverse_compute_at(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16, 16], "float32")
    B = tir.match_buffer(b, [16], "float32")
    B_rf_local = tir.alloc_buffer([16, 16], "float32", scope="local")
    for j in tir.thread_binding(0, 16, thread = "blockIdx.x"):
        for i_o in tir.thread_binding(0, 4, thread = "threadIdx.x"):
            for i_i, k in tir.grid(4, 16):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "B_rf") as [vi, vj, vk]:
                    tir.bind(vi, i_o * 4 + i_i)
                    tir.bind(vj, j)
                    tir.bind(vk, k)
                    with tir.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
            for k in tir.serial(0, 4):
                with tir.block([16, tir.reduce_axis(0, 16)], "B") as [vi, vk]:
                    tir.bind(vi, j)
                    tir.bind(vk, i_o * 4 + k)
                    with tir.init():
                        B[vi] = 0.0
                    B[vi] = B[vi] + B_rf_local[vk, vi]


@tvm.script.tir
def fail_subtree_compact_dataflow(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    for i in range(0, 128):
        for j in range(0, 64):
            with tir.block([128, 128], "B_0") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 64):
            with tir.block([128, 128], "B_1") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j + 64)
                B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def fail_all_consumers_under_loop(a: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    D = tir.match_buffer(d, (128, 128), "float32")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = B[vi, vj] + 1.0
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "D") as [vi, vj]:
            D[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def fail_all_producers_under_loop(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    C = tir.alloc_buffer((128, 128), "float32")
    D = tir.match_buffer(d, (128, 128), "float32")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = A[vi, vj] + 1.0
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "D") as [vi, vj]:
            D[vi, vj] = B[vi, vj] + C[vi, vj]


@tvm.script.tir
def read_out_of_bound(a: ty.handle, c:ty.handle) -> None:
    A = tir.match_buffer(a, [16], "float32")
    B = tir.alloc_buffer([16], "float32")
    C = tir.match_buffer(c, [16], "float32")
    for i in tir.serial(0, 16):
        with tir.block([16], "B") as [v]:
            B[v] = A[v]
    for j in tir.serial(0, 16):
        with tir.block([16], "C") as [v]:
            tir.reads(B[v : v + 2])
            C[v] = tir.if_then_else(v < 15, tir.max(B[v], B[v + 1]), B[v], dtype="float32")


@tvm.script.tir
def read_out_of_bound_after_compute_at(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16], "float32")
    B = tir.alloc_buffer([16], "float32")
    C = tir.match_buffer(c, [16], "float32")
    for j in tir.serial(0, 16):
        for i in tir.serial(0, tir.min(1, 15 - j) + 1):
            with tir.block([16], "B") as [v]:
                tir.bind(v, j + i)
                B[v] = A[v]
        with tir.block([16], "C") as [v]:
            tir.bind(v, j)
            tir.reads([B[v : v + 2]])
            C[v] = tir.if_then_else(v < 15, tir.max(B[v], B[v + 1]), B[v], dtype="float32")


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
