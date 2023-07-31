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
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks,not-callable

@T.prim_func
def cuda_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                for k1 in T.unroll(0, 8):
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k0 * 8 + k1)
                                            T.reads([C[vi, vj], A[vi, vk], B[vk, vj]])
                                            T.writes([C[vi, vj]])
                                            with T.init():
                                                C[vi, vj] = 0.0
                                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def cuda_matmul_read_at_a(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 2048], dtype="float32")
    B = T.match_buffer(b, [2048, 2048], dtype="float32")
    C = T.match_buffer(c, [2048, 2048], dtype="float32")
    A_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    for by in T.thread_binding(0, 32, thread="blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread="blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread="vthread.y"):
                for vx in T.thread_binding(0, 2, thread="vthread.x"):
                    for ty in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread="threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                with T.block("A_shared"):
                                    v0 = T.axis.S(32, by)
                                    v1 = T.axis.S(256, k0)
                                    T.reads([A[v0 * 64 : v0 * 64 + 64, v1 * 8 : v1 * 8 + 8]])
                                    T.writes([A_shared[v0 * 64 : v0 * 64 + 64, v1 * 8 : v1 * 8 + 8]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(64, 8):
                                        A_shared[v0 * 64 + ax0, v1 * 8 + ax1] = A[v0 * 64 + ax0, v1 * 8 + ax1]
                                for k1 in T.unroll(0, 8):
                                    for v_, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k0 * 8 + k1)
                                            T.reads([C[vi, vj], A_shared[vi, vk], B[vk, vj]])
                                            T.writes([C[vi, vj]])
                                            with T.init():
                                                C[vi, vj] = T.float32(0)
                                            C[vi, vj] = C[vi, vj] + A_shared[vi, vk] * B[vk, vj]


@T.prim_func
def cuda_matmul_read_at_ab(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 2048], dtype="float32")
    B = T.match_buffer(b, [2048, 2048], dtype="float32")
    C = T.match_buffer(c, [2048, 2048], dtype="float32")
    A_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    for by in T.thread_binding(0, 32, thread="blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread="blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread="vthread.y"):
                for vx in T.thread_binding(0, 2, thread="vthread.x"):
                    for ty in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread="threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                with T.block("A_shared"):
                                    v0 = T.axis.S(32, by)
                                    v1 = T.axis.S(256, k0)
                                    T.reads([A[v0 * 64 : v0 * 64 + 64, v1 * 8 : v1 * 8 + 8]])
                                    T.writes([A_shared[v0 * 64 : v0 * 64 + 64, v1 * 8 : v1 * 8 + 8]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(64, 8):
                                        A_shared[v0 * 64 + ax0, v1 * 8 + ax1] = A[v0 * 64 + ax0, v1 * 8 + ax1]
                                with T.block("B_shared"):
                                    v0 = T.axis.S(256, k0)
                                    v1 = T.axis.S(32, bx)
                                    T.reads([B[v0 * 8 : v0 * 8 + 8, v1 * 64 : v1 * 64 + 64]])
                                    T.writes([B_shared[v0 * 8 : v0 * 8 + 8, v1 * 64 : v1 * 64 + 64]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(8, 64):
                                        B_shared[v0 * 8 + ax0, v1 * 64 + ax1] = B[v0 * 8 + ax0, v1 * 64 + ax1]
                                for k1 in T.unroll(0, 8):
                                    for v_, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k0 * 8 + k1)
                                            T.reads([C[vi, vj], A_shared[vi, vk], B_shared[vk, vj]])
                                            T.writes([C[vi, vj]])
                                            with T.init():
                                                C[vi, vj] = T.float32(0)
                                            C[vi, vj] = C[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]

@T.prim_func
def cuda_matmul_write_at_c(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 2048], dtype="float32")
    B = T.match_buffer(b, [2048, 2048], dtype="float32")
    C = T.match_buffer(c, [2048, 2048], dtype="float32")
    A_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    C_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    for by in T.thread_binding(0, 32, thread="blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread="blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread="vthread.y"):
                for vx in T.thread_binding(0, 2, thread="vthread.x"):
                    for ty in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread="threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                with T.block("A_shared"):
                                    v0 = T.axis.S(32, by)
                                    v1 = T.axis.S(256, k0)
                                    T.reads([A[v0 * 64 : v0 * 64 + 64, v1 * 8 : v1 * 8 + 8]])
                                    T.writes([A_shared[v0 * 64 : v0 * 64 + 64, v1 * 8 : v1 * 8 + 8]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(64, 8):
                                        A_shared[v0 * 64 + ax0, v1 * 8 + ax1] = A[v0 * 64 + ax0, v1 * 8 + ax1]
                                with T.block("B_shared"):
                                    v0 = T.axis.S(256, k0)
                                    v1 = T.axis.S(32, bx)
                                    T.reads([B[v0 * 8 : v0 * 8 + 8, v1 * 64 : v1 * 64 + 64]])
                                    T.writes([B_shared[v0 * 8 : v0 * 8 + 8, v1 * 64 : v1 * 64 + 64]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(8, 64):
                                        B_shared[v0 * 8 + ax0, v1 * 64 + ax1] = B[v0 * 8 + ax0, v1 * 64 + ax1]
                                for k1 in T.unroll(0, 8):
                                    for v_, i, j in T.grid(1, 4, 4):
                                        with T.block("C"):
                                            vi = T.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = T.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = T.axis.R(2048, k0 * 8 + k1)
                                            T.reads([C_shared[vi, vj], A_shared[vi, vk], B_shared[vk, vj]])
                                            T.writes([C_shared[vi, vj]])
                                            with T.init():
                                                C_shared[vi, vj] = T.float32(0)
                                            C_shared[vi, vj] = C_shared[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                            with T.block("C_shared"):
                                v0 = T.axis.S(32, by)
                                v1 = T.axis.S(32, bx)
                                T.reads([C_shared[v0 * 64 : v0 * 64 + 64, v1 * 64 : v1 * 64 + 64]])
                                T.writes([C[v0 * 64 : v0 * 64 + 64, v1 * 64 : v1 * 64 + 64]])
                                T.block_attr({"auto_copy":1})
                                for ax0, ax1 in T.grid(64, 64):
                                    C[v0 * 64 + ax0, v1 * 64 + ax1] = C_shared[v0 * 64 + ax0, v1 * 64 + ax1]


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks,not-callable
# fmt: on


def test_read_at_global_to_shared_a():
    sch = tir.Schedule(cuda_matmul, debug_mask="all")
    block = sch.get_block("C")
    # pylint: disable=invalid-name
    _by, _bx, _vy, _vx, _ty, _tx, k0, _k1, _, _i, _j = sch.get_loops(block)
    # pylint: enable=invalid-name
    sch.read_at(k0, block, 1, "shared")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], cuda_matmul_read_at_a)
    verify_trace_roundtrip(sch, cuda_matmul)


def test_read_at_global_to_shared_ab():
    sch = tir.Schedule(cuda_matmul_read_at_a, debug_mask="all")
    block = sch.get_block("C")
    # pylint: disable=invalid-name
    _by, _bx, _vy, _vx, _ty, _tx, k0, _k1, _, _i, _j = sch.get_loops(block)
    # pylint: enable=invalid-name
    sch.read_at(k0, block, 2, "shared")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], cuda_matmul_read_at_ab)
    verify_trace_roundtrip(sch, cuda_matmul_read_at_a)


def test_read_at_local_to_shared_c():
    sch = tir.Schedule(cuda_matmul_read_at_ab, debug_mask="all")
    block = sch.get_block("C")
    # pylint: disable=invalid-name
    _by, _bx, _vy, _vx, _ty, tx, _k0, _k1, _, _i, _j = sch.get_loops(block)
    # pylint: enable=invalid-name
    sch.write_at(tx, block, 0, "shared")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], cuda_matmul_write_at_c)
    verify_trace_roundtrip(sch, cuda_matmul_read_at_ab)


if __name__ == "__main__":
    tvm.testing.main()
