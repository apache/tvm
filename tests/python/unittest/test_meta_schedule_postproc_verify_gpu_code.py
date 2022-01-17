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
    for rule in ctx.postprocs:
        rule.initialize_with_tune_context(ctx)
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
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
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
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


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
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
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
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


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
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
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
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


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
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
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
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


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
    assert not ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_2():
    mod = Conv2dCuda2
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert not ctx.postprocs[0].apply(sch)


def test_postproc_verify_gpu_3():
    mod = Conv2dCuda3
    ctx = _create_context(mod, target=_target())
    sch = tir.Schedule(mod, debug_mask="all")
    assert not ctx.postprocs[0].apply(sch)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
