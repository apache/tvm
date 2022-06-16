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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.postproc import VerifyGPUCode
from tvm.script import tir as T
from tvm.target import Target


def _target() -> Target:
    return Target("nvidia/geforce-rtx-3080")


def _create_context(mod, target) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        postprocs=[
            VerifyGPUCode(),
        ],
        task_name="test",
    )
    return ctx


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant
# fmt: off

@tvm.script.ir_module
class Conv2dCuda0:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14*14*256*256], dtype="float32")
        B = T.match_buffer(b, [14*14*512*256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([64], "float32", "local")
        Apad_shared = T.allocate([512], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),
                    dtype="float32x4",
                )
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner] # fmt: on


@tvm.script.ir_module
class Conv2dCuda1:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14*14*256*256], dtype="float32")
        B = T.match_buffer(b, [14*14*512*256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([6400000], "float32", "local")
        Apad_shared = T.allocate([512], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
            # Access of the last element of B_local prevents buffer
            # compacting from reducing the amount of shared memory
            # used.
            B_local[6400000-1 + ff_c_init*8] = 0.0
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),
                    dtype="float32x4",
                )
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner]# fmt: on


@tvm.script.ir_module
class Conv2dCuda2:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14*14*256*256], dtype="float32")
        B = T.match_buffer(b, [14*14*512*256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([64], "float32", "local")
        Apad_shared = T.allocate([512000], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),
                    dtype="float32x4",
                )
                # Access of the last element of Apad_shared prevents
                # buffer compacting from reducing the amount of shared
                # memory used.
                Apad_shared[512000-1] = 0.0
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner]# fmt: on


@tvm.script.ir_module
class Conv2dCuda3:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14*14*256*256], dtype="float32")
        B = T.match_buffer(b, [14*14*512*256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([64], "float32", "local")
        Apad_shared = T.allocate([512], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 800000)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),
                    dtype="float32x4",
                )
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner]# fmt: on

@T.prim_func
def GmmCuda0(X: T.Buffer[(1, 128, 128), "float32"], Y: T.Buffer[(1, 128, 128), "float32"], Z: T.Buffer[(1, 128, 128), "float32"]) -> None:
    Z_local = T.alloc_buffer([1, 128, 128], dtype="float32", scope="local")
    X_shared = T.alloc_buffer([1, 128, 128], dtype="float32", scope="shared")
    Y_shared = T.alloc_buffer([1, 128, 128], dtype="float32", scope="shared")
    for i0_0_i1_0_i2_0_fused in T.thread_binding(16, thread="blockIdx.x"):
        for i0_1_i1_1_i2_1_fused in T.thread_binding(1, thread="vthread.x"):
            for i0_2_i1_2_i2_2_fused in T.thread_binding(128, thread="threadIdx.x"):
                for i1_3_init, i2_4_init in T.grid(4, 2):
                    with T.block("Z_init"):
                        b = T.axis.spatial(1, 0)
                        i = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3_init)
                        j = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4_init)
                        T.reads()
                        T.writes(Z_local[b, i, j])
                        Z_local[b, i, j] = T.float32(0)
                for i3_0 in T.serial(4):
                    for ax0_ax1_ax2_fused_0 in T.serial(4):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                            for ax0_ax1_ax2_fused_2 in T.vectorized(2):
                                with T.block("X_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) // 32)
                                    v2 = T.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) % 32)
                                    T.reads(X[v0, v1, v2])
                                    T.writes(X_shared[v0, v1, v2])
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                    for ax0_ax1_ax2_fused_0 in T.serial(8):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                            with T.block("Y_shared"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) // 32)
                                v2 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) % 32)
                                T.reads(Y[v0, v1, v2])
                                T.writes(Y_shared[v0, v1, v2])
                                Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                    for i3_1, i0_3, i1_3, i2_3, i3_2, i0_4, i1_4, i2_4 in T.grid(1, 1, 4, 1, 32, 1, 1, 2):
                        with T.block("Z_update"):
                            b = T.axis.spatial(1, 0)
                            i = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3)
                            j = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4)
                            k = T.axis.reduce(128, i3_0 * 32 + i3_2)
                            T.reads(Z_local[b, i, j], X_shared[b, i, k], Y_shared[b, k, j])
                            T.writes(Z_local[b, i, j])
                            Z_local[b, i, j] = Z_local[b, i, j] + X_shared[b, i, k] * Y_shared[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 4, 2):
                    with T.block("Z_local"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + ax1)
                        v2 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + ax2)
                        T.reads(Z_local[v0, v1, v2])
                        T.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_local[v0, v1, v2]

@T.prim_func
def GmmCuda1(X: T.Buffer[(1, 128, 128), "float32"], Y: T.Buffer[(1, 128, 128), "float32"], Z: T.Buffer[(1, 128, 128), "float32"]) -> None:
    Z_local = T.alloc_buffer([1, 128, 128], dtype="float32", scope="local")
    X_shared = T.alloc_buffer([1, 128, 128], dtype="float32", scope="shared")
    Y_shared = T.alloc_buffer([1, 128, 128], dtype="float32", scope="shared")
    for i0_0_i1_0_i2_0_fused in T.thread_binding(16, thread="blockIdx.x"):
        for i0_1_i1_1_i2_1_fused in T.thread_binding(1, thread="vthread.x"):
            for i0_2_i1_2_i2_2_fused in T.thread_binding(128, thread="threadIdx.x"):
                for i1_3_init, i2_4_init in T.grid(4, 2):
                    with T.block("Z_init"):
                        b = T.axis.spatial(1, 0)
                        i = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3_init)
                        j = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4_init)
                        T.reads()
                        T.writes(Z_local[b, i, j])
                        Z_local[b, i, j] = T.float32(0)
                for i3_0 in T.serial(4):
                    for ax0_ax1_ax2_fused_0 in T.serial(4):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                            for ax0_ax1_ax2_fused_2 in T.vectorized(2):
                                with T.block("X_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) // 32)
                                    v2 = T.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) % 32)
                                    T.reads(X[v0, v1, v2])
                                    T.writes(X_shared[v0, v1, v2])
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                    for ax0_ax1_ax2_fused_0 in T.serial(8):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                            with T.block("Y_shared"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) // 32)
                                v2 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) % 32)
                                T.reads(Y[v0, v1, v2])
                                T.writes(Y_shared[v0, v1, v2])
                                Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                    for i3_1, i0_3, i1_3, i2_3, i3_2, i0_4, i1_4, i2_4 in T.grid(1, 1, 4, 1, 32, 1, 1, 2):
                        with T.block("Z_update"):
                            b = T.axis.spatial(1, 0)
                            i = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3)
                            j = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4)
                            k = T.axis.reduce(128, i3_0 * 32 + i3_2)
                            T.block_attr({
                                "meta_schedule.thread_extent_low_inclusive": 0,
                                "meta_schedule.thread_extent_high_inclusive": 32,
                            })
                            T.reads(Z_local[b, i, j], X_shared[b, i, k], Y_shared[b, k, j])
                            T.writes(Z_local[b, i, j])
                            Z_local[b, i, j] = Z_local[b, i, j] + X_shared[b, i, k] * Y_shared[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 4, 2):
                    with T.block("Z_local"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + ax1)
                        v2 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + ax2)
                        T.reads(Z_local[v0, v1, v2])
                        T.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_local[v0, v1, v2]


@T.prim_func
def GmmCuda2(X: T.Buffer[(1, 128, 128), "float32"], Y: T.Buffer[(1, 128, 128), "float32"], Z: T.Buffer[(1, 128, 128), "float32"]) -> None:
    Z_local = T.alloc_buffer([1, 128, 128], dtype="float32", scope="local")
    X_shared = T.alloc_buffer([1, 128, 128], dtype="float32", scope="shared")
    Y_shared = T.alloc_buffer([1, 128, 128], dtype="float32", scope="shared")
    for i0_0_i1_0_i2_0_fused in T.thread_binding(16, thread="blockIdx.x"):
        for i0_1_i1_1_i2_1_fused in T.thread_binding(1, thread="vthread.x"):
            for i0_2_i1_2_i2_2_fused in T.thread_binding(128, thread="threadIdx.x"):
                for i1_3_init, i2_4_init in T.grid(4, 2):
                    with T.block("Z_init"):
                        b = T.axis.spatial(1, 0)
                        i = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3_init)
                        j = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4_init)
                        T.reads()
                        T.writes(Z_local[b, i, j])
                        Z_local[b, i, j] = T.float32(0)
                for i3_0 in T.serial(4):
                    for ax0_ax1_ax2_fused_0 in T.serial(4):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                            for ax0_ax1_ax2_fused_2 in T.vectorized(2):
                                with T.block("X_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) // 32)
                                    v2 = T.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) % 32)
                                    T.reads(X[v0, v1, v2])
                                    T.writes(X_shared[v0, v1, v2])
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                    for ax0_ax1_ax2_fused_0 in T.serial(8):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(128, thread="threadIdx.x"):
                            with T.block("Y_shared"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) // 32)
                                v2 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) % 32)
                                T.reads(Y[v0, v1, v2])
                                T.writes(Y_shared[v0, v1, v2])
                                Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                    for i3_1, i0_3, i1_3, i2_3, i3_2, i0_4, i1_4, i2_4 in T.grid(1, 1, 4, 1, 32, 1, 1, 2):
                        with T.block("Z_update"):
                            b = T.axis.spatial(1, 0)
                            i = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3)
                            j = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4)
                            k = T.axis.reduce(128, i3_0 * 32 + i3_2)
                            T.block_attr({
                                "meta_schedule.thread_extent_low_inclusive": 1024,
                                "meta_schedule.thread_extent_high_inclusive": 1024,
                            })
                            T.reads(Z_local[b, i, j], X_shared[b, i, k], Y_shared[b, k, j])
                            T.writes(Z_local[b, i, j])
                            Z_local[b, i, j] = Z_local[b, i, j] + X_shared[b, i, k] * Y_shared[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 4, 2):
                    with T.block("Z_local"):
                        v0 = T.axis.spatial(1, ax0)
                        v1 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + ax1)
                        v2 = T.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + ax2)
                        T.reads(Z_local[v0, v1, v2])
                        T.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_local[v0, v1, v2]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant


def test_postproc_verify_gpu_0():
    mod = Conv2dCuda0
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_1():
    mod = Conv2dCuda1
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_2():
    mod = Conv2dCuda2
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    # Should fail due to too much local memory per block (large
    # Apad_shared allocation).
    assert not ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_3():
    mod = Conv2dCuda3
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    # Should fail due to too many threads per block (large
    # threadIdx.x extent).
    assert not ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_4():
    mod = GmmCuda0
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_5():
    mod = GmmCuda1
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert not ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_6():
    mod = GmmCuda2
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert not ctx.postprocs[0].apply(sch)


if __name__ == "__main__":
    tvm.testing.main()
