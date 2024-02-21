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
# pylint: disable=missing-docstring
import tvm.testing

from tvm.ir import IRModule, assert_structural_equal
from tvm import dlight as dl
from tvm.script import ir as I
from tvm.target import Target
from tvm.script import tir as T


def _check(mod_before: IRModule, mod_after: IRModule):
    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.RMSNorm(),
        )(mod_before)
    assert_structural_equal(mod, mod_after)


def test_rms_norm_with_casting():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(var_data: T.handle, weight: T.Buffer((4096,), "float16"), var_T_cast: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int32()
            data = T.match_buffer(var_data, (1, n, 4096), "float16")
            T_cast = T.match_buffer(var_T_cast, (1, n, 4096), "float16")
            # with T.block("root"):
            T_cast_1 = T.alloc_buffer((1, n, 4096))
            T_multiply = T.alloc_buffer((1, n, 4096))
            T_multiply_red = T.alloc_buffer((1, n))
            rsqrt = T.alloc_buffer((1, n))
            T_cast_2 = T.alloc_buffer((4096,))
            T_rms_norm = T.alloc_buffer((1, n, 4096))
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with T.block("T_cast"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(data[v_ax0, v_ax1, v_ax2])
                    T.writes(T_cast_1[v_ax0, v_ax1, v_ax2])
                    T_cast_1[v_ax0, v_ax1, v_ax2] = T.Cast("float32", data[v_ax0, v_ax1, v_ax2])
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_cast_1[v_ax0, v_ax1, v_ax2])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_1[v_ax0, v_ax1, v_ax2]
            for ax0, ax1, k2 in T.grid(1, n, 4096):
                with T.block("T_multiply_red"):
                    v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                    T.reads(T_multiply[v_ax0, v_ax1, v_k2])
                    T.writes(T_multiply_red[v_ax0, v_ax1])
                    with T.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2]
            for ax0, ax1 in T.grid(1, n):
                with T.block("rsqrt"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_red[v_ax0, v_ax1])
                    T.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
            for ax0 in range(4096):
                with T.block("T_cast_1"):
                    v_ax0 = T.axis.spatial(4096, ax0)
                    T.reads(weight[v_ax0])
                    T.writes(T_cast_2[v_ax0])
                    T_cast_2[v_ax0] = T.Cast("float32", weight[v_ax0])
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with T.block("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2], T_cast_2[v_ax2])
                    T.writes(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T_rms_norm[v_ax0, v_ax1, v_ax2] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_2[v_ax2]
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with T.block("T_cast_2"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T.writes(T_cast[v_ax0, v_ax1, v_ax2])
                    T_cast[v_ax0, v_ax1, v_ax2] = T.Cast("float16", T_rms_norm[v_ax0, v_ax1, v_ax2])

    @I.ir_module
    class After:
        @T.prim_func
        def main(var_data: T.handle, weight: T.Buffer((4096,), "float16"), var_T_cast: T.handle):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            n = T.int32()
            data = T.match_buffer(var_data, (1, n, 4096), "float16")
            T_cast = T.match_buffer(var_T_cast, (1, n, 4096), "float16")
            # with T.block("root"):
            T_multiply_local = T.alloc_buffer((1, n, 4096), scope="local")
            T_multiply_red_local = T.alloc_buffer((1, n), scope="local")
            rsqrt_shared = T.alloc_buffer((1, n), scope="shared")
            T_rms_norm_local = T.alloc_buffer((1, n, 4096), scope="local")
            data_local = T.alloc_buffer((1, n, 4096), "float16", scope="local")
            for ax0_ax1_fused in T.thread_binding(n, thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(512, thread="threadIdx.x"):
                    for ax2_1 in range(1):
                        for ax2_2 in T.vectorized(8):
                            with T.block("data_local"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(n, ax0_ax1_fused)
                                v2 = T.axis.spatial(4096, ax2_0 * 8 + ax2_1 * 8 + ax2_2)
                                T.reads(data[v0, v1, v2])
                                T.writes(data_local[v0, v1, v2])
                                data_local[v0, v1, v2] = data[v0, v1, v2]
                    for ax0 in range(8):
                        with T.block("T_multiply"):
                            v_ax0 = T.axis.spatial(1, 0)
                            v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = T.axis.spatial(4096, ax2_0 * 8 + ax0)
                            T.reads(data_local[v_ax0, v_ax1, v_ax2])
                            T.writes(T_multiply_local[v_ax0, v_ax1, v_ax2])
                            T_multiply_local[v_ax0, v_ax1, v_ax2] = T.Cast("float32", data_local[v_ax0, v_ax1, v_ax2]) * T.Cast("float32", data_local[v_ax0, v_ax1, v_ax2])
                    for ax0 in range(8):
                        with T.block("T_multiply_red"):
                            v_ax0 = T.axis.spatial(1, 0)
                            v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                            v_k2 = T.axis.reduce(4096, ax2_0 * 8 + ax0)
                            T.reads(T_multiply_local[v_ax0, v_ax1, v_k2])
                            T.writes(T_multiply_red_local[v_ax0, v_ax1])
                            with T.init():
                                T_multiply_red_local[v_ax0, v_ax1] = T.float32(0)
                            T_multiply_red_local[v_ax0, v_ax1] = T_multiply_red_local[v_ax0, v_ax1] + T_multiply_local[v_ax0, v_ax1, v_k2]
                with T.block("rsqrt"):
                    v_ax0 = T.axis.spatial(1, 0)
                    v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                    T.reads(T_multiply_red_local[v_ax0, v_ax1])
                    T.writes(rsqrt_shared[v_ax0, v_ax1])
                    rsqrt_shared[v_ax0, v_ax1] = T.rsqrt(T_multiply_red_local[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
                for ax0_0 in T.thread_binding(512, thread="threadIdx.x"):
                    for ax0_1, ax0_2 in T.grid(1, 8):
                        with T.block("T_rms_norm"):
                            v_ax0 = T.axis.spatial(1, 0)
                            v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = T.axis.spatial(4096, ax0_0 * 8 + ax0_1 * 8 + ax0_2)
                            T.reads(rsqrt_shared[v_ax0, v_ax1], data_local[v_ax0, v_ax1, v_ax2], weight[v_ax2])
                            T.writes(T_rms_norm_local[v_ax0, v_ax1, v_ax2])
                            T_rms_norm_local[v_ax0, v_ax1, v_ax2] = rsqrt_shared[v_ax0, v_ax1] * T.Cast("float32", data_local[v_ax0, v_ax1, v_ax2]) * T.Cast("float32", weight[v_ax2])
                    for ax0 in T.vectorized(8):
                        with T.block("T_cast_local"):
                            v0 = T.axis.spatial(1, 0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused)
                            v2 = T.axis.spatial(4096, ax0_0 * 8 + ax0)
                            T.reads(T_rms_norm_local[v0, v1, v2])
                            T.writes(T_cast[v0, v1, v2])
                            T_cast[v0, v1, v2] = T.Cast("float16", T_rms_norm_local[v0, v1, v2])
    # fmt: on
    _check(Before, After)


def test_rms_norm_without_casting():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(var_data: T.handle, weight: T.Buffer((4096,), "float32"), var_T_cast: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int32()
            data = T.match_buffer(var_data, (1, n, 4096))
            T_cast = T.match_buffer(var_T_cast, (1, n, 4096))
            # with T.block("root"):
            T_multiply = T.alloc_buffer((1, n, 4096))
            T_multiply_red = T.alloc_buffer((1, n))
            rsqrt = T.alloc_buffer((1, n))
            T_rms_norm = T.alloc_buffer((1, n, 4096))
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(data[v_ax0, v_ax1, v_ax2])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = data[v_ax0, v_ax1, v_ax2] * data[v_ax0, v_ax1, v_ax2]
            for ax0, ax1, k2 in T.grid(1, n, 4096):
                with T.block("T_multiply_red"):
                    v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                    T.reads(T_multiply[v_ax0, v_ax1, v_k2])
                    T.writes(T_multiply_red[v_ax0, v_ax1])
                    with T.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2]
            for ax0, ax1 in T.grid(1, n):
                with T.block("rsqrt"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_red[v_ax0, v_ax1])
                    T.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with T.block("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rsqrt[v_ax0, v_ax1], data[v_ax0, v_ax1, v_ax2], weight[v_ax2])
                    T.writes(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T_rms_norm[v_ax0, v_ax1, v_ax2] = rsqrt[v_ax0, v_ax1] * data[v_ax0, v_ax1, v_ax2] * weight[v_ax2]
            for ax0, ax1, ax2 in T.grid(1, n, 4096):
                with T.block("T_cast_2"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T.writes(T_cast[v_ax0, v_ax1, v_ax2])
                    T_cast[v_ax0, v_ax1, v_ax2] = T_rms_norm[v_ax0, v_ax1, v_ax2]

    @I.ir_module
    class After:
        @T.prim_func
        def main(var_data: T.handle, weight: T.Buffer((4096,), "float32"), var_T_cast: T.handle):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            n = T.int32()
            data = T.match_buffer(var_data, (1, n, 4096))
            T_cast = T.match_buffer(var_T_cast, (1, n, 4096))
            # with T.block("root"):
            T_multiply_local = T.alloc_buffer((1, n, 4096), scope="local")
            T_multiply_red_local = T.alloc_buffer((1, n), scope="local")
            rsqrt_shared = T.alloc_buffer((1, n), scope="shared")
            T_rms_norm_local = T.alloc_buffer((1, n, 4096), scope="local")
            data_local = T.alloc_buffer((1, n, 4096), scope="local")
            for ax0_ax1_fused in T.thread_binding(n, thread="blockIdx.x"):
                for ax2_0 in T.thread_binding(512, thread="threadIdx.x"):
                    for ax2_1 in range(1):
                        for ax2_2 in T.vectorized(8):
                            with T.block("data_local"):
                                v0 = T.axis.spatial(1, 0)
                                v1 = T.axis.spatial(n, ax0_ax1_fused)
                                v2 = T.axis.spatial(4096, ax2_0 * 8 + ax2_1 * 8 + ax2_2)
                                T.reads(data[v0, v1, v2])
                                T.writes(data_local[v0, v1, v2])
                                data_local[v0, v1, v2] = data[v0, v1, v2]
                    for ax0 in range(8):
                        with T.block("T_multiply"):
                            v_ax0 = T.axis.spatial(1, 0)
                            v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = T.axis.spatial(4096, ax2_0 * 8 + ax0)
                            T.reads(data_local[v_ax0, v_ax1, v_ax2])
                            T.writes(T_multiply_local[v_ax0, v_ax1, v_ax2])
                            T_multiply_local[v_ax0, v_ax1, v_ax2] = data_local[v_ax0, v_ax1, v_ax2] * data_local[v_ax0, v_ax1, v_ax2]
                    for ax0 in range(8):
                        with T.block("T_multiply_red"):
                            v_ax0 = T.axis.spatial(1, 0)
                            v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                            v_k2 = T.axis.reduce(4096, ax2_0 * 8 + ax0)
                            T.reads(T_multiply_local[v_ax0, v_ax1, v_k2])
                            T.writes(T_multiply_red_local[v_ax0, v_ax1])
                            with T.init():
                                T_multiply_red_local[v_ax0, v_ax1] = T.float32(0)
                            T_multiply_red_local[v_ax0, v_ax1] = T_multiply_red_local[v_ax0, v_ax1] + T_multiply_local[v_ax0, v_ax1, v_k2]
                with T.block("rsqrt"):
                    v_ax0 = T.axis.spatial(1, 0)
                    v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                    T.reads(T_multiply_red_local[v_ax0, v_ax1])
                    T.writes(rsqrt_shared[v_ax0, v_ax1])
                    rsqrt_shared[v_ax0, v_ax1] = T.rsqrt(T_multiply_red_local[v_ax0, v_ax1] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))
                for ax0_0 in T.thread_binding(512, thread="threadIdx.x"):
                    for ax0_1, ax0_2 in T.grid(1, 8):
                        with T.block("T_rms_norm"):
                            v_ax0 = T.axis.spatial(1, 0)
                            v_ax1 = T.axis.spatial(n, ax0_ax1_fused)
                            v_ax2 = T.axis.spatial(4096, ax0_0 * 8 + ax0_1 * 8 + ax0_2)
                            T.reads(rsqrt_shared[v_ax0, v_ax1], data_local[v_ax0, v_ax1, v_ax2], weight[v_ax2])
                            T.writes(T_rms_norm_local[v_ax0, v_ax1, v_ax2])
                            T_rms_norm_local[v_ax0, v_ax1, v_ax2] = rsqrt_shared[v_ax0, v_ax1] * data_local[v_ax0, v_ax1, v_ax2] * weight[v_ax2]
                    for ax0 in T.vectorized(8):
                        with T.block("T_cast_local"):
                            v0 = T.axis.spatial(1, 0)
                            v1 = T.axis.spatial(n, ax0_ax1_fused)
                            v2 = T.axis.spatial(4096, ax0_0 * 8 + ax0)
                            T.reads(T_rms_norm_local[v0, v1, v2])
                            T.writes(T_cast[v0, v1, v2])
                            T_cast[v0, v1, v2] = T_rms_norm_local[v0, v1, v2]
    # fmt: on
    _check(Before, After)


if __name__ == "__main__":
    tvm.testing.main()
