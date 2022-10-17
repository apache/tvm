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
import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.target import Target


def _target() -> Target:
    return Target("nvidia/geforce-rtx-3080")


def _create_context(mod, target) -> ms.TuneContext:
    return ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[ms.postproc.VerifyGPUCode()],
            mutator_probs={},
        ),
        task_name="test",
    )


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
        B_local = T.decl_buffer([64], "float32", scope="local")
        Apad_shared = T.decl_buffer([512], "float32", scope="shared")
        Apad_shared_local = T.decl_buffer([8], "float32", scope="local")
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
        B_local = T.decl_buffer([6400000], "float32", scope="local")
        Apad_shared = T.decl_buffer([512], "float32", scope="shared")
        Apad_shared_local = T.decl_buffer([8], "float32", scope="local")
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
        B_local = T.decl_buffer([64], "float32", scope="local")
        Apad_shared = T.decl_buffer([512000], "float32", scope="shared")
        Apad_shared_local = T.decl_buffer([8], "float32", scope="local")
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
        B_local = T.decl_buffer([64], "float32", scope="local")
        Apad_shared = T.decl_buffer([512], "float32", scope="shared")
        Apad_shared_local = T.decl_buffer([8], "float32", scope="local")
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


@T.prim_func
def GMMCUDATensorCore(
    X: T.Buffer[(1024, 1024), "float16"],
    Y: T.Buffer[(1024, 1024), "float16"],
    Z: T.Buffer[(1024, 1024), "float32"],
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    s0 = T.var("int32")
    s0_1 = T.var("int32")
    s0_2 = T.var("int32")
    s1 = T.var("int32")
    s1_1 = T.var("int32")
    s1_2 = T.var("int32")
    # body
    # with T.block("root")
    Z_wmma_accumulator = T.alloc_buffer([1024, 1024], dtype="float32", scope="wmma.accumulator")
    X_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
    Y_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
    X_shared_wmma_matrix_a = T.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_a")
    Y_shared_wmma_matrix_b = T.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_b")
    for ax0_0_ax1_0_0_ax2_0_0_fused in T.thread_binding(64, thread="blockIdx.x"):
        for ax0_1_ax1_0_1_ax2_0_1_fused in T.thread_binding(2, thread="blockIdx.y"):
            for ax0_2_ax1_0_2_ax2_0_2_fused in T.thread_binding(2, thread="threadIdx.y"):
                for ax1_0_3_init, ax2_0_3_init, ax1_0_4_init, ax2_0_4_init in T.grid(2, 1, 2, 4):
                    with T.block("Z_o_init"):
                        v0 = T.axis.spatial(1, 0)
                        v1_o = T.axis.spatial(
                            64,
                            ax0_0_ax1_0_0_ax2_0_0_fused % 64 // 16 * 16
                            + ax0_1_ax1_0_1_ax2_0_1_fused % 2 * 8
                            + ax0_2_ax1_0_2_ax2_0_2_fused % 2 * 4
                            + ax1_0_3_init * 2
                            + ax1_0_4_init,
                        )
                        v2_o = T.axis.spatial(
                            64,
                            (ax0_0_ax1_0_0_ax2_0_0_fused % 16 + 0 + 0 + ax2_0_3_init) * 4
                            + ax2_0_4_init,
                        )
                        T.reads()
                        T.writes(
                            Z_wmma_accumulator[
                                v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                            ]
                        )
                        T.block_attr(
                            {
                                "meta_schedule.thread_extent_high_inclusive": 1024,
                                "meta_schedule.thread_extent_low_inclusive": 32,
                                "warp_execution": 1,
                            }
                        )
                        C = T.match_buffer(
                            Z_wmma_accumulator[
                                v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                            ],
                            [16, 16],
                            dtype="float32",
                            scope="wmma.accumulator",
                            offset_factor=16,
                        )
                        T.evaluate(
                            T.tvm_fill_fragment(
                                C.data,
                                16,
                                16,
                                16,
                                C.elem_offset // 256 + C.elem_offset % 256 // 16,
                                T.float32(0),
                                dtype="handle",
                            )
                        )
                for ax3_0_0 in T.serial(32):
                    for ax0_ax1_fused_0 in T.serial(16):
                        for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.vectorized(4):
                                    with T.block("X_shared"):
                                        v0 = T.axis.spatial(
                                            1024,
                                            ax0_0_ax1_0_0_ax2_0_0_fused // 16 * 256
                                            + ax0_1_ax1_0_1_ax2_0_1_fused * 128
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 128
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            // 32,
                                        )
                                        v1 = T.axis.spatial(
                                            1024,
                                            ax3_0_0 * 32
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 128
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            % 32,
                                        )
                                        T.reads(X[v0, v1])
                                        T.writes(X_shared[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        X_shared[v0, v1] = X[v0, v1]
                    for ax0_ax1_fused_0 in T.serial(8):
                        for ax0_ax1_fused_1 in T.thread_binding(2, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.vectorized(4):
                                    with T.block("Y_shared"):
                                        v0 = T.axis.spatial(
                                            1024,
                                            ax3_0_0 * 32
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 128
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            // 64,
                                        )
                                        v1 = T.axis.spatial(
                                            1024,
                                            ax0_0_ax1_0_0_ax2_0_0_fused % 16 * 64
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 128
                                                + ax0_ax1_fused_2 * 4
                                                + ax0_ax1_fused_3
                                            )
                                            % 64,
                                        )
                                        T.reads(Y[v0, v1])
                                        T.writes(Y_shared[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 8]]})
                                        Y_shared[v0, v1] = Y[v0, v1]
                    for ax3_0_1 in T.serial(2):
                        for ax0_0, ax1_0 in T.grid(4, 1):
                            with T.block("X_shared_wmma.matrix_a_o"):
                                v0_o = T.axis.spatial(
                                    64,
                                    ax0_0_ax1_0_0_ax2_0_0_fused // 16 * 16
                                    + ax0_1_ax1_0_1_ax2_0_1_fused * 8
                                    + ax0_2_ax1_0_2_ax2_0_2_fused * 4
                                    + ax0_0,
                                )
                                v1_o = T.axis.spatial(64, ax3_0_0 * 2 + ax3_0_1)
                                T.reads(
                                    X_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16]
                                )
                                T.writes(
                                    X_shared_wmma_matrix_a[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ]
                                )
                                A = T.match_buffer(
                                    X_shared[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ],
                                    [16, 16],
                                    dtype="float16",
                                    strides=[s1, s0],
                                    scope="shared",
                                    offset_factor=16,
                                )
                                C_1 = T.match_buffer(
                                    X_shared_wmma_matrix_a[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ],
                                    [16, 16],
                                    dtype="float16",
                                    scope="wmma.matrix_a",
                                    offset_factor=16,
                                )
                                T.evaluate(
                                    T.tvm_load_matrix_sync(
                                        C_1.data,
                                        16,
                                        16,
                                        16,
                                        C_1.elem_offset // 256 + C_1.elem_offset % 256 // 16,
                                        T.tvm_access_ptr(
                                            T.type_annotation(dtype="float16"),
                                            A.data,
                                            A.elem_offset,
                                            s1 * 16,
                                            1,
                                            dtype="handle",
                                        ),
                                        s1,
                                        "row_major",
                                        dtype="handle",
                                    )
                                )
                        for ax0_0, ax1_0 in T.grid(1, 4):
                            with T.block("Y_shared_wmma.matrix_b_o"):
                                v0_o = T.axis.spatial(64, ax3_0_0 * 2 + ax3_0_1)
                                v1_o = T.axis.spatial(
                                    64, ax0_0_ax1_0_0_ax2_0_0_fused % 16 * 4 + ax1_0
                                )
                                T.reads(
                                    Y_shared[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16]
                                )
                                T.writes(
                                    Y_shared_wmma_matrix_b[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ]
                                )
                                A_1 = T.match_buffer(
                                    Y_shared[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ],
                                    [16, 16],
                                    dtype="float16",
                                    strides=[s1_1, s0_1],
                                    scope="shared",
                                    offset_factor=16,
                                )
                                C_2 = T.match_buffer(
                                    Y_shared_wmma_matrix_b[
                                        v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                                    ],
                                    [16, 16],
                                    dtype="float16",
                                    scope="wmma.matrix_b",
                                    offset_factor=16,
                                )
                                T.evaluate(
                                    T.tvm_load_matrix_sync(
                                        C_2.data,
                                        16,
                                        16,
                                        16,
                                        C_2.elem_offset // 256 + C_2.elem_offset % 256 // 16,
                                        T.tvm_access_ptr(
                                            T.type_annotation(dtype="float16"),
                                            A_1.data,
                                            A_1.elem_offset,
                                            s1_1 * 16,
                                            1,
                                            dtype="handle",
                                        ),
                                        s1_1,
                                        "row_major",
                                        dtype="handle",
                                    )
                                )
                        for ax0_3, ax1_0_3, ax2_0_3, ax3_0_2, ax0_4, ax1_0_4, ax2_0_4 in T.grid(
                            1, 2, 1, 1, 1, 2, 4
                        ):
                            with T.block("Z_o_update"):
                                v0 = T.axis.spatial(1, 0)
                                v1_o = T.axis.spatial(
                                    64,
                                    ax0_0_ax1_0_0_ax2_0_0_fused % 64 // 16 * 16
                                    + ax0_1_ax1_0_1_ax2_0_1_fused % 2 * 8
                                    + ax0_2_ax1_0_2_ax2_0_2_fused % 2 * 4
                                    + ax1_0_3 * 2
                                    + ax1_0_4,
                                )
                                v2_o = T.axis.spatial(
                                    64,
                                    (ax0_0_ax1_0_0_ax2_0_0_fused % 16 + 0 + 0 + ax2_0_3) * 4
                                    + ax2_0_4,
                                )
                                v3_o = T.axis.reduce(64, ax3_0_0 * 2 + ax3_0_1 + ax3_0_2)
                                T.reads(
                                    Z_wmma_accumulator[
                                        v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                                    ],
                                    X_shared_wmma_matrix_a[
                                        v1_o * 16 : v1_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16
                                    ],
                                    Y_shared_wmma_matrix_b[
                                        v3_o * 16 : v3_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                                    ],
                                )
                                T.writes(
                                    Z_wmma_accumulator[
                                        v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                                    ]
                                )
                                T.block_attr(
                                    {
                                        "meta_schedule.thread_extent_high_inclusive": 1024,
                                        "meta_schedule.thread_extent_low_inclusive": 32,
                                        "warp_execution": 1,
                                    }
                                )
                                A_2 = T.match_buffer(
                                    X_shared_wmma_matrix_a[
                                        v1_o * 16 : v1_o * 16 + 16, v3_o * 16 : v3_o * 16 + 16
                                    ],
                                    [16, 16],
                                    dtype="float16",
                                    scope="wmma.matrix_a",
                                    offset_factor=16,
                                )
                                B = T.match_buffer(
                                    Y_shared_wmma_matrix_b[
                                        v3_o * 16 : v3_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                                    ],
                                    [16, 16],
                                    dtype="float16",
                                    scope="wmma.matrix_b",
                                    offset_factor=16,
                                )
                                C_3 = T.match_buffer(
                                    Z_wmma_accumulator[
                                        v1_o * 16 : v1_o * 16 + 16, v2_o * 16 : v2_o * 16 + 16
                                    ],
                                    [16, 16],
                                    dtype="float32",
                                    scope="wmma.accumulator",
                                    offset_factor=16,
                                )
                                T.evaluate(
                                    T.tvm_mma_sync(
                                        C_3.data,
                                        C_3.elem_offset // 256 + C_3.elem_offset % 256 // 16,
                                        A_2.data,
                                        A_2.elem_offset // 256,
                                        B.data,
                                        B.elem_offset // 256,
                                        C_3.data,
                                        C_3.elem_offset // 256 + C_3.elem_offset % 256 // 16,
                                        dtype="handle",
                                    )
                                )
                for ax0_0, ax1_0 in T.grid(4, 4):
                    with T.block("Z_wmma.accumulator_o"):
                        v0_o = T.axis.spatial(
                            64,
                            ax0_0_ax1_0_0_ax2_0_0_fused // 16 * 16
                            + ax0_1_ax1_0_1_ax2_0_1_fused * 8
                            + ax0_2_ax1_0_2_ax2_0_2_fused * 4
                            + ax0_0,
                        )
                        v1_o = T.axis.spatial(64, ax0_0_ax1_0_0_ax2_0_0_fused % 16 * 4 + ax1_0)
                        T.reads(
                            Z_wmma_accumulator[
                                v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                            ]
                        )
                        T.writes(Z[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16])
                        A_3 = T.match_buffer(
                            Z_wmma_accumulator[
                                v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16
                            ],
                            [16, 16],
                            dtype="float32",
                            scope="wmma.accumulator",
                            offset_factor=16,
                        )
                        C_4 = T.match_buffer(
                            Z[v0_o * 16 : v0_o * 16 + 16, v1_o * 16 : v1_o * 16 + 16],
                            [16, 16],
                            dtype="float32",
                            strides=[s1_2, s0_2],
                            offset_factor=16,
                        )
                        T.evaluate(
                            T.tvm_store_matrix_sync(
                                A_3.data,
                                16,
                                16,
                                16,
                                A_3.elem_offset // 256 + A_3.elem_offset % 256 // 16,
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype="float32"),
                                    C_4.data,
                                    C_4.elem_offset,
                                    s1_2 * 16,
                                    2,
                                    dtype="handle",
                                ),
                                s1_2,
                                "row_major",
                                dtype="handle",
                            )
                        )


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant


@pytest.mark.parametrize("mod", [Conv2dCuda0, Conv2dCuda1, GmmCuda0, GMMCUDATensorCore])
def test_postproc_check_pass(mod):
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert ctx.space_generator.postprocs[0].apply(sch)


@pytest.mark.parametrize(
    "mod",
    [
        Conv2dCuda2,  # Should fail due to too much local memory per block (large Apad_shared allocation)
        Conv2dCuda3,  # Should fail due to too many threads per block (large threadIdx.x extent)
        GmmCuda1,
        GmmCuda2,
    ],
)
def test_postproc_check_fail(mod):
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert not ctx.space_generator.postprocs[0].apply(sch)


if __name__ == "__main__":
    tvm.testing.main()
