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
import tvm
import tvm.testing
from tvm import dlight as dl
from tvm.ir import IRModule, assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


def _check(mod_before: IRModule, mod_after: IRModule):
    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.GeneralReduction(),
        )(mod_before)
    assert_structural_equal(mod, mod_after)


def test_softmax_1():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(p_lv44: T.handle, p_output0: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n, m = T.int64(), T.int64()
            lv44 = T.match_buffer(p_lv44, (T.int64(1), T.int64(32), n, m))
            var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
            # with T.block("root"):
            T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n))
            T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
            T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n))
            var_T_softmax_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m))
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
                with T.block("T_softmax_maxelem"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(lv44[v_i0, v_i1, v_i2, v_k])
                    T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                    with T.init():
                        T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], lv44[v_i0, v_i1, v_i2, v_k])
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
                with T.block("T_softmax_exp"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(lv44[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                    T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv44[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
                with T.block("T_softmax_expsum"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                    T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                    with T.init():
                        T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
                with T.block("T_softmax_norm"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                    T.writes(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.block_attr({"axis": 3})
                    var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
                with T.block("compute"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])

    @I.ir_module
    class After:
        @T.prim_func
        def main(p_lv44: T.handle, p_output0: T.handle):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            n, m = T.int64(), T.int64()
            lv44 = T.match_buffer(p_lv44, (T.int64(1), T.int64(32), n, m))
            var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
            # with T.block("root"):
            T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
            T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(32), n), scope="shared")
            for ax0_ax1_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            with T.block("T_softmax_maxelem"):
                                v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                                v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                                v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                                T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                                T.reads(lv44[T.int64(0), v0, v1, v2])
                                T.writes(T_softmax_maxelem_shared[T.int64(0), v0, v1])
                                with T.init():
                                    T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.float32(-3.4028234663852886e+38)
                                T_softmax_maxelem_shared[T.int64(0), v0, v1] = T.max(T_softmax_maxelem_shared[T.int64(0), v0, v1], lv44[T.int64(0), v0, v1, v2])
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax2_fused_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            with T.block("T_softmax_expsum"):
                                v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n + ax0)
                                v1 = T.axis.spatial(n, ax0_ax1_fused % n + ax1)
                                v2 = T.axis.reduce(m, ax2_fused_0 * T.int64(256) + ax2_fused_1)
                                T.where(ax2_fused_0 * T.int64(256) + ax2_fused_1 < m)
                                T.reads(lv44[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1])
                                T.writes(T_softmax_expsum_shared[T.int64(0), v0, v1])
                                with T.init():
                                    T_softmax_expsum_shared[T.int64(0), v0, v1] = T.float32(0)
                                T_softmax_expsum_shared[T.int64(0), v0, v1] = T_softmax_expsum_shared[T.int64(0), v0, v1] + T.exp(lv44[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1])
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_0 in T.serial((m + T.int64(255)) // T.int64(256), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(T.int64(32), ax0_ax1_fused // n)
                            v1 = T.axis.spatial(n, ax0_ax1_fused % n)
                            v2 = T.axis.spatial(m, ax2_0 * T.int64(256) + ax2_1)
                            T.where(ax2_0 * T.int64(256) + ax2_1 < m)
                            T.reads(lv44[T.int64(0), v0, v1, v2], T_softmax_maxelem_shared[T.int64(0), v0, v1], T_softmax_expsum_shared[T.int64(0), v0, v1])
                            T.writes(var_compute_intermediate[T.int64(0), v0, v1, v2])
                            var_compute_intermediate[T.int64(0), v0, v1, v2] = T.Cast("float16", T.exp(lv44[T.int64(0), v0, v1, v2] - T_softmax_maxelem_shared[T.int64(0), v0, v1]) / T_softmax_expsum_shared[T.int64(0), v0, v1])
    # fmt: on
    _check(Before, After)


def test_softmax_2():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
            # with T.block("root"):
            T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(1)))
            T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)))
            T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(1)))
            for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with T.block("T_softmax_maxelem"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(A[v_i0, v_i1, v_k])
                    T.writes(T_softmax_maxelem[v_i0, v_i1])
                    with T.init():
                        T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with T.block("T_softmax_exp"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                    T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                    T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
            for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with T.block("T_softmax_expsum"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                    T.writes(T_softmax_expsum[v_i0, v_i1])
                    with T.init():
                        T_softmax_expsum[v_i0, v_i1] = T.float32(0)
                    T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
                with T.block("T_softmax_norm"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                    T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                    T.block_attr({"axis": 2})
                    T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]


    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
            T.func_attr({"tir.is_scheduled": 1})
            # with T.block("root"):
            T_softmax_maxelem_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
            T_softmax_expsum_shared = T.alloc_buffer((T.int64(1), T.int64(1)), scope="shared")
            for ax0_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax0 in range(T.int64(1)):
                    for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_fused_0 in T.serial(T.int64(125), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            with T.block("T_softmax_maxelem"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.reduce(T.int64(32000), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                T.reads(A[T.int64(0), T.int64(0), v1])
                                T.writes(T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                                with T.init():
                                    T_softmax_maxelem_shared[T.int64(0), T.int64(0)] = T.float32(-3.4028234663852886e+38)
                                T_softmax_maxelem_shared[T.int64(0), T.int64(0)] = T.max(T_softmax_maxelem_shared[T.int64(0), T.int64(0)], A[T.int64(0), T.int64(0), v1])
                for ax0 in range(T.int64(1)):
                    for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_fused_0 in T.serial(T.int64(125), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            with T.block("T_softmax_expsum"):
                                v0 = T.axis.spatial(T.int64(1), ax0)
                                v1 = T.axis.reduce(T.int64(32000), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                T.reads(A[T.int64(0), T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                                T.writes(T_softmax_expsum_shared[T.int64(0), T.int64(0)])
                                with T.init():
                                    T_softmax_expsum_shared[T.int64(0), T.int64(0)] = T.float32(0)
                                T_softmax_expsum_shared[T.int64(0), T.int64(0)] = T_softmax_expsum_shared[T.int64(0), T.int64(0)] + T.exp(A[T.int64(0), T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), T.int64(0)])
                for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_0 in T.serial(T.int64(125), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_softmax_norm"):
                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v1 = T.axis.spatial(T.int64(32000), ax1_0 * T.int64(256) + ax1_1)
                            T.reads(A[T.int64(0), T.int64(0), v1], T_softmax_maxelem_shared[T.int64(0), T.int64(0)], T_softmax_expsum_shared[T.int64(0), T.int64(0)])
                            T.writes(T_softmax_norm[T.int64(0), T.int64(0), v1])
                            T.block_attr({"axis": 2})
                            T_softmax_norm[T.int64(0), T.int64(0), v1] = T.exp(A[T.int64(0), T.int64(0), v1] - T_softmax_maxelem_shared[T.int64(0), T.int64(0)]) / T_softmax_expsum_shared[T.int64(0), T.int64(0)]

    # fmt: on
    _check(Before, After)


def test_layer_norm():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(p_lv6: T.handle, weight1: T.Buffer((T.int64(2560),), "float32"), bias: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
            var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
            # with T.block("root"):
            A_red_temp_v0 = T.alloc_buffer((T.int64(1), n))
            A_red_temp_v1 = T.alloc_buffer((T.int64(1), n))
            var_T_layer_norm_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(2560)))
            for ax0, ax1, k2 in T.grid(T.int64(1), n, T.int64(2560)):
                with T.block("A_red_temp"):
                    v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                    T.reads(lv6[v_ax0, v_ax1, v_k2])
                    T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                    with T.init():
                        A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2]
                    v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2] * lv6[v_ax0, v_ax1, v_k2]
                    A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                    A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
            for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(2560)):
                with T.block("T_layer_norm"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(lv6[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], weight1[v_ax2], bias[v_ax2])
                    T.writes(var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2])
                    var_T_layer_norm_intermediate[v_ax0, v_ax1, v_ax2] = (lv6[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00039062500000000002) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v_ax2] + bias[v_ax2]
            for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(2560)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])
                    T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                    var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_layer_norm_intermediate[v_i0, v_i1, v_i2])

    @I.ir_module
    class After:
        @T.prim_func
        def main(p_lv6: T.handle, weight1: T.Buffer((T.int64(2560),), "float32"), bias: T.Buffer((T.int64(2560),), "float32"), p_output0: T.handle):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            n = T.int64()
            lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(2560)))
            var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(2560)), "float16")
            # with T.block("root"):
            A_red_temp_v0_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
            A_red_temp_v1_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
            for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
                for ax0 in range(T.int64(1)):
                    for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_fused_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            with T.block("A_red_temp"):
                                v0 = T.axis.spatial(n, ax0_fused + ax0)
                                v1 = T.axis.reduce(T.int64(2560), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                T.reads(lv6[T.int64(0), v0, v1])
                                T.writes(A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0])
                                with T.init():
                                    A_red_temp_v0_shared[T.int64(0), v0] = T.float32(0)
                                    A_red_temp_v1_shared[T.int64(0), v0] = T.float32(0)
                                v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1]
                                v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[T.int64(0), v0] + lv6[T.int64(0), v0, v1] * lv6[T.int64(0), v0, v1]
                                A_red_temp_v0_shared[T.int64(0), v0] = v_A_red_temp_v0
                                A_red_temp_v1_shared[T.int64(0), v0] = v_A_red_temp_v1
                for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_0 in T.serial(T.int64(10), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("compute"):
                            v0 = T.axis.spatial(n, ax0_fused)
                            v1 = T.axis.spatial(T.int64(2560), ax1_0 * T.int64(256) + ax1_1)
                            T.reads(lv6[T.int64(0), v0, v1], A_red_temp_v0_shared[T.int64(0), v0], A_red_temp_v1_shared[T.int64(0), v0], weight1[v1], bias[v1])
                            T.writes(var_compute_intermediate[T.int64(0), v0, v1])
                            var_compute_intermediate[T.int64(0), v0, v1] = T.Cast("float16", (lv6[T.int64(0), v0, v1] - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v1] + bias[v1])
    # fmt: on
    _check(Before, After)


def test_rms_norm():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
            T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
            rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
            # with T.block("root"):
            Ared_temp = T.alloc_buffer((T.int64(1), n))
            for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
                with T.block("Ared_temp"):
                    v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                    T.reads(A[v_bsz, v_i, v_k])
                    T.writes(Ared_temp[v_bsz, v_i])
                    with T.init():
                        Ared_temp[v_bsz, v_i] = T.float32(0)
                    Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
                with T.block("rms_norm"):
                    v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                    T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                    T.writes(rms_norm_1[v_bsz, v_i, v_k])
                    rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

    @I.ir_module
    class After:
        @T.prim_func
        def main(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
            T.func_attr({"op_pattern": 4, "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
            rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
            # with T.block("root"):
            Ared_temp_shared = T.alloc_buffer((T.int64(1), n), scope="shared")
            for ax0_fused in T.thread_binding(n, thread="blockIdx.x"):
                for ax0 in range(T.int64(1)):
                    for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax1_fused_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            with T.block("Ared_temp"):
                                v0 = T.axis.spatial(n, ax0_fused + ax0)
                                v1 = T.axis.reduce(T.int64(4096), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                                T.reads(A[T.int64(0), v0, v1])
                                T.writes(Ared_temp_shared[T.int64(0), v0])
                                with T.init():
                                    Ared_temp_shared[T.int64(0), v0] = T.float32(0)
                                Ared_temp_shared[T.int64(0), v0] = Ared_temp_shared[T.int64(0), v0] + T.Cast("float32", A[T.int64(0), v0, v1]) * T.Cast("float32", A[T.int64(0), v0, v1])
                for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_0 in T.serial(T.int64(16), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("rms_norm"):
                            v0 = T.axis.spatial(n, ax0_fused)
                            v1 = T.axis.spatial(T.int64(4096), ax1_0 * T.int64(256) + ax1_1)
                            T.reads(B[v1], A[T.int64(0), v0, v1], Ared_temp_shared[T.int64(0), v0])
                            T.writes(rms_norm_1[T.int64(0), v0, v1])
                            rms_norm_1[T.int64(0), v0, v1] = T.Cast("float16", T.Cast("float32", B[v1]) * (T.Cast("float32", A[T.int64(0), v0, v1]) / T.sqrt(Ared_temp_shared[T.int64(0), v0] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
    # fmt: on
    _check(Before, After)


def test_group_norm():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((1, 2048), "float32"), B: T.Buffer((2048,), "float32"), C: T.Buffer((2048,), "float32"), T_reshape: T.Buffer((1, 2048), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            T_reshape_1 = T.alloc_buffer((1, 32, 64))
            A_red_temp_v0 = T.alloc_buffer((1, 32))
            A_red_temp_v1 = T.alloc_buffer((1, 32))
            T_reshape_2 = T.alloc_buffer((32, 64))
            T_reshape_3 = T.alloc_buffer((32, 64))
            T_group_norm = T.alloc_buffer((1, 32, 64))
            for ax0, ax1, ax2 in T.grid(1, 32, 64):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(A[0, (v_ax1 * 64 + v_ax2) % 2048])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2])
                    T_reshape_1[v_ax0, v_ax1, v_ax2] = A[0, (v_ax1 * 64 + v_ax2) % 2048]
            for ax0, ax1, k2 in T.grid(1, 32, 64):
                with T.block("A_red_temp"):
                    v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_k2])
                    T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
                    with T.init():
                        A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2]
                    v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2] * T_reshape_1[v_ax0, v_ax1, v_k2]
                    A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
                    A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
            for ax0, ax1 in T.grid(32, 64):
                with T.block("T_reshape_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(B[(v_ax0 * 64 + v_ax1) % 2048])
                    T.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = B[(v_ax0 * 64 + v_ax1) % 2048]
            for ax0, ax1 in T.grid(32, 64):
                with T.block("T_reshape_2"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(C[(v_ax0 * 64 + v_ax1) % 2048])
                    T.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = C[(v_ax0 * 64 + v_ax1) % 2048]
            for ax0, ax1, ax2 in T.grid(1, 32, 64):
                with T.block("T_group_norm"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    T.writes(T_group_norm[v_ax0, v_ax1, v_ax2])
                    T_group_norm[v_ax0, v_ax1, v_ax2] = (T_reshape_1[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.015625)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.015625) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.015625) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.015625)) + T.float32(1.0000000000000001e-05)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1 in T.grid(1, 2048):
                with T.block("T_reshape_3"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_group_norm[0, v_ax1 % 2048 // 64, v_ax1 % 64])
                    T.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = T_group_norm[0, v_ax1 % 2048 // 64, v_ax1 % 64]

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((1, 2048), "float32"), B: T.Buffer((2048,), "float32"), C: T.Buffer((2048,), "float32"), T_reshape: T.Buffer((1, 2048), "float32")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            A_red_temp_v0_shared = T.alloc_buffer((1, 32), scope="shared")
            A_red_temp_v1_shared = T.alloc_buffer((1, 32), scope="shared")
            for ax0_fused in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax0 in range(32):
                    for ax1_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                        for ax1_fused_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                            with T.block("A_red_temp"):
                                v0 = T.axis.spatial(32, ax0)
                                v1 = T.axis.reduce(64, ax1_fused_0 * 256 + ax1_fused_1)
                                T.where(ax1_fused_0 * 256 + ax1_fused_1 < 64)
                                T.reads(A[0, v0 * 64 + v1])
                                T.writes(A_red_temp_v0_shared[0, v0], A_red_temp_v1_shared[0, v0])
                                with T.init():
                                    A_red_temp_v0_shared[0, v0] = T.float32(0)
                                    A_red_temp_v1_shared[0, v0] = T.float32(0)
                                v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[0, v0] + A[0, v0 * 64 + v1]
                                v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[0, v0] + A[0, v0 * 64 + v1] * A[0, v0 * 64 + v1]
                                A_red_temp_v0_shared[0, v0] = v_A_red_temp_v0
                                A_red_temp_v1_shared[0, v0] = v_A_red_temp_v1
                for ax1_1 in T.thread_binding(256, thread="threadIdx.x"):
                    for ax1_0 in T.serial(8, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("T_reshape_3"):
                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v1 = T.axis.spatial(2048, ax1_0 * 256 + ax1_1)
                            T.reads(A[0, v1], A_red_temp_v0_shared[0, v1 // 64], A_red_temp_v1_shared[0, v1 // 64], B[v1], C[v1])
                            T.writes(T_reshape[0, v1])
                            T_reshape[0, v1] = (A[0, v1] - A_red_temp_v0_shared[0, v1 // 64] * T.float32(0.015625)) * T.rsqrt(A_red_temp_v1_shared[0, v1 // 64] * T.float32(0.015625) - A_red_temp_v0_shared[0, v1 // 64] * T.float32(0.015625) * (A_red_temp_v0_shared[0, v1 // 64] * T.float32(0.015625)) + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]    # fmt: on
    _check(Before, After)


def test_logsumexp():
    @I.ir_module
    class Before:
        @T.prim_func
        def compute_lse(var_A: T.handle, var_blocked_lse: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            batch_size = T.int64(is_size_var=True)
            vocab_size = T.int64(is_size_var=True)
            num_chunks = T.int64(is_size_var=True)
            A = T.match_buffer(var_A, (batch_size, vocab_size), dtype="float32")
            blocked_lse = T.match_buffer(var_blocked_lse, (batch_size, num_chunks), dtype="float32")
            A_pad = T.alloc_buffer((batch_size, num_chunks, T.int64(4096)), dtype="float32")
            temp_max = T.alloc_buffer((batch_size, num_chunks), dtype="float32")
            temp_sum = T.alloc_buffer((batch_size, num_chunks), dtype="float32")

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(4096)):
                with T.block("pad"):
                    v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                    A_pad[v0, v1, v2] = T.if_then_else(
                        v1 * T.int64(4096) + v2 < vocab_size,
                        A[v0, v1 * T.int64(4096) + v2],
                        T.min_value("float32"),
                    )

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(4096)):
                with T.block("max"):
                    v0, v1, v2 = T.axis.remap("SSR", [l0, l1, l2])
                    with T.init():
                        temp_max[v0, v1] = T.min_value("float32")
                    temp_max[v0, v1] = T.max(temp_max[v0, v1], A_pad[v0, v1, v2])

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(4096)):
                with T.block("sum_exp"):
                    v0, v1, v2 = T.axis.remap("SSR", [l0, l1, l2])
                    with T.init():
                        temp_sum[v0, v1] = T.float32(0)
                    temp_sum[v0, v1] += T.if_then_else(
                        v1 * T.int64(4096) + v2 < vocab_size,
                        T.exp(A_pad[v0, v1, v2] - temp_max[v0, v1]),
                        T.float32(0),
                    )

            for l0, l1, l2 in T.grid(batch_size, num_chunks, T.int64(1)):
                with T.block("log"):
                    v0, v1, v2 = T.axis.remap("SSS", [l0, l1, l2])
                    blocked_lse[v0, v1] = T.log(temp_sum[v0, v1]) + temp_max[v0, v1]

    @I.ir_module
    class After:
        @T.prim_func
        def compute_lse(var_A: T.handle, var_blocked_lse: T.handle):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            batch_size, vocab_size = T.int64(is_size_var=True), T.int64(is_size_var=True)
            A = T.match_buffer(var_A, (batch_size, vocab_size))
            num_chunks = T.int64(is_size_var=True)
            blocked_lse = T.match_buffer(var_blocked_lse, (batch_size, num_chunks))
            temp_max_shared = T.alloc_buffer((batch_size, num_chunks), scope="shared")
            temp_sum_shared = T.alloc_buffer((batch_size, num_chunks), scope="shared")
            for ax0_ax1_fused in T.thread_binding(batch_size * num_chunks, thread="blockIdx.x"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax2_fused_0 in T.serial(
                            T.int64(16),
                            annotations={
                                "pragma_auto_unroll_max_step": 256,
                                "pragma_unroll_explicit": 1,
                            },
                        ):
                            with T.block("max"):
                                v0 = T.axis.spatial(
                                    batch_size,
                                    ax0_ax1_fused % (num_chunks * batch_size) // num_chunks + ax0,
                                )
                                v1 = T.axis.spatial(num_chunks, ax0_ax1_fused % num_chunks + ax1)
                                v2 = T.axis.reduce(
                                    T.int64(4096), ax2_fused_0 * T.int64(256) + ax2_fused_1
                                )
                                T.reads(A[v0, v1 * T.int64(4096) + v2])
                                T.writes(temp_max_shared[v0, v1])
                                with T.init():
                                    temp_max_shared[v0, v1] = T.min_value("float32")
                                temp_max_shared[v0, v1] = T.max(
                                    temp_max_shared[v0, v1],
                                    T.if_then_else(
                                        v1 * T.int64(4096) + v2 < vocab_size,
                                        A[v0, v1 * T.int64(4096) + v2],
                                        T.min_value("float32"),
                                    ),
                                )
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        for ax2_fused_0 in T.serial(
                            T.int64(16),
                            annotations={
                                "pragma_auto_unroll_max_step": 256,
                                "pragma_unroll_explicit": 1,
                            },
                        ):
                            with T.block("sum_exp"):
                                v0 = T.axis.spatial(
                                    batch_size,
                                    ax0_ax1_fused % (num_chunks * batch_size) // num_chunks + ax0,
                                )
                                v1 = T.axis.spatial(num_chunks, ax0_ax1_fused % num_chunks + ax1)
                                v2 = T.axis.reduce(
                                    T.int64(4096), ax2_fused_0 * T.int64(256) + ax2_fused_1
                                )
                                T.reads(A[v0, v1 * T.int64(4096) + v2], temp_max_shared[v0, v1])
                                T.writes(temp_sum_shared[v0, v1])
                                with T.init():
                                    temp_sum_shared[v0, v1] = T.float32(0)
                                temp_sum_shared[v0, v1] = temp_sum_shared[v0, v1] + T.if_then_else(
                                    v1 * T.int64(4096) + v2 < vocab_size,
                                    T.exp(
                                        (
                                            T.if_then_else(
                                                v1 * T.int64(4096) + v2 < vocab_size,
                                                A[v0, v1 * T.int64(4096) + v2],
                                                T.min_value("float32"),
                                            )
                                            - temp_max_shared[v0, v1]
                                        )
                                    ),
                                    T.float32(0),
                                )
                for ax2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax2_0 in T.serial(
                        T.int64(1),
                        annotations={
                            "pragma_auto_unroll_max_step": 256,
                            "pragma_unroll_explicit": 1,
                        },
                    ):
                        with T.block("log"):
                            v0 = T.axis.spatial(
                                batch_size, ax0_ax1_fused % (num_chunks * batch_size) // num_chunks
                            )
                            v1 = T.axis.spatial(num_chunks, ax0_ax1_fused % num_chunks)
                            v2 = T.axis.spatial(T.int64(1), ax2_0 * T.int64(256) + ax2_1)
                            T.where(ax2_0 * T.int64(256) + ax2_1 < T.int64(1))
                            T.reads(temp_sum_shared[v0, v1], temp_max_shared[v0, v1])
                            T.writes(blocked_lse[v0, v1])
                            blocked_lse[v0, v1] = (
                                T.log(temp_sum_shared[v0, v1]) + temp_max_shared[v0, v1]
                            )

    _check(Before, After)


if __name__ == "__main__":
    tvm.testing.main()
