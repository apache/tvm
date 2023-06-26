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
from tvm import dlight as dl
from tvm.ir import IRModule, assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


def _check(mod_before: IRModule, mod_after: IRModule):
    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.Reduction(),
        )(mod_before)
    assert_structural_equal(mod, mod_after)


def test_softmax():
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
            for i0_i1_i2_fused in T.thread_binding(n * T.int64(32), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0, ax1, ax2, ax3_fused_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), (m + T.int64(255)) // T.int64(256)):
                    for ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("T_softmax_maxelem"):
                            v_i0 = T.axis.spatial(T.int64(1), ax0)
                            v_i1 = T.axis.spatial(T.int64(32), i0_i1_i2_fused // n + ax1)
                            v_i2 = T.axis.spatial(n, i0_i1_i2_fused % n + ax2)
                            v_k = T.axis.reduce(m, ax3_fused_0 * T.int64(256) + ax3_fused_1)
                            T.where(T.int64(0) <= i0_i1_i2_fused // n and i0_i1_i2_fused // n < T.int64(32) and T.int64(0) <= i0_i1_i2_fused % n and i0_i1_i2_fused % n < n and ax3_fused_0 * T.int64(256) + ax3_fused_1 < m)
                            T.reads(lv44[v_i0, v_i1, v_i2, v_k])
                            T.writes(T_softmax_maxelem_shared[v_i0, v_i1, v_i2])
                            with T.init():
                                T_softmax_maxelem_shared[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem_shared[v_i0, v_i1, v_i2], lv44[v_i0, v_i1, v_i2, v_k])
                for ax0, ax1, ax2, ax3_fused_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), (m + T.int64(255)) // T.int64(256)):
                    for ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("T_softmax_expsum"):
                            v_i0 = T.axis.spatial(T.int64(1), ax0)
                            v_i1 = T.axis.spatial(T.int64(32), i0_i1_i2_fused // n + ax1)
                            v_i2 = T.axis.spatial(n, i0_i1_i2_fused % n + ax2)
                            v_k = T.axis.reduce(m, ax3_fused_0 * T.int64(256) + ax3_fused_1)
                            T.where(T.int64(0) <= i0_i1_i2_fused // n and i0_i1_i2_fused // n < T.int64(32) and T.int64(0) <= i0_i1_i2_fused % n and i0_i1_i2_fused % n < n and ax3_fused_0 * T.int64(256) + ax3_fused_1 < m)
                            T.reads(lv44[v_i0, v_i1, v_i2, v_k], T_softmax_maxelem_shared[v_i0, v_i1, v_i2])
                            T.writes(T_softmax_expsum_shared[v_i0, v_i1, v_i2])
                            with T.init():
                                T_softmax_expsum_shared[v_i0, v_i1, v_i2] = T.float32(0)
                            T_softmax_expsum_shared[v_i0, v_i1, v_i2] = T_softmax_expsum_shared[v_i0, v_i1, v_i2] + T.exp(lv44[v_i0, v_i1, v_i2, v_k] - T_softmax_maxelem_shared[v_i0, v_i1, v_i2])
                for i3_0 in range((m + T.int64(255)) // T.int64(256)):
                    for i3_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("compute"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(T.int64(32), i0_i1_i2_fused // n)
                            v_i2 = T.axis.spatial(n, i0_i1_i2_fused % n)
                            v_i3 = T.axis.spatial(m, i3_0 * T.int64(256) + i3_1)
                            T.where(i3_0 * T.int64(256) + i3_1 < m)
                            T.reads(lv44[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem_shared[v_i0, v_i1, v_i2], T_softmax_expsum_shared[v_i0, v_i1, v_i2])
                            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.exp(lv44[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem_shared[v_i0, v_i1, v_i2]) / T_softmax_expsum_shared[v_i0, v_i1, v_i2])
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
            for i0_i1_fused in T.thread_binding(n, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0, ax1, ax2_fused_0 in T.grid(T.int64(1), T.int64(1), T.int64(10)):
                    for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("A_red_temp"):
                            v_ax0 = T.axis.spatial(T.int64(1), ax0)
                            v_ax1 = T.axis.spatial(n, i0_i1_fused + ax1)
                            v_k2 = T.axis.reduce(T.int64(2560), ax2_fused_0 * T.int64(256) + ax2_fused_1)
                            T.reads(lv6[v_ax0, v_ax1, v_k2])
                            T.writes(A_red_temp_v0_shared[v_ax0, v_ax1], A_red_temp_v1_shared[v_ax0, v_ax1])
                            with T.init():
                                A_red_temp_v0_shared[v_ax0, v_ax1] = T.float32(0)
                                A_red_temp_v1_shared[v_ax0, v_ax1] = T.float32(0)
                            v_A_red_temp_v0: T.float32 = A_red_temp_v0_shared[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2]
                            v_A_red_temp_v1: T.float32 = A_red_temp_v1_shared[v_ax0, v_ax1] + lv6[v_ax0, v_ax1, v_k2] * lv6[v_ax0, v_ax1, v_k2]
                            A_red_temp_v0_shared[v_ax0, v_ax1] = v_A_red_temp_v0
                            A_red_temp_v1_shared[v_ax0, v_ax1] = v_A_red_temp_v1
                for i2_0 in range(T.int64(10)):
                    for i2_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("compute"):
                            v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i1 = T.axis.spatial(n, i0_i1_fused)
                            v_i2 = T.axis.spatial(T.int64(2560), i2_0 * T.int64(256) + i2_1)
                            T.reads(lv6[v_i0, v_i1, v_i2], A_red_temp_v0_shared[v_i0, v_i1], A_red_temp_v1_shared[v_i0, v_i1], weight1[v_i2], bias[v_i2])
                            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
                            var_compute_intermediate[v_i0, v_i1, v_i2] = T.Cast("float16", (lv6[v_i0, v_i1, v_i2] - A_red_temp_v0_shared[v_i0, v_i1] * T.float32(0.00039062500000000002)) * T.rsqrt(A_red_temp_v1_shared[v_i0, v_i1] * T.float32(0.00039062500000000002) - A_red_temp_v0_shared[v_i0, v_i1] * T.float32(0.00039062500000000002) * (A_red_temp_v0_shared[v_i0, v_i1] * T.float32(0.00039062500000000002)) + T.float32(1.0000000000000001e-05)) * weight1[v_i2] + bias[v_i2])
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
            for bsz_i_fused in T.thread_binding(n, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax0, ax1, ax2_fused_0 in T.grid(T.int64(1), T.int64(1), T.int64(16)):
                    for ax2_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("Ared_temp"):
                            v_bsz = T.axis.spatial(T.int64(1), ax0)
                            v_i = T.axis.spatial(n, bsz_i_fused + ax1)
                            v_k = T.axis.reduce(T.int64(4096), ax2_fused_0 * T.int64(256) + ax2_fused_1)
                            T.reads(A[v_bsz, v_i, v_k])
                            T.writes(Ared_temp_shared[v_bsz, v_i])
                            with T.init():
                                Ared_temp_shared[v_bsz, v_i] = T.float32(0)
                            Ared_temp_shared[v_bsz, v_i] = Ared_temp_shared[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
                for k_0 in range(T.int64(16)):
                    for k_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                        with T.block("rms_norm"):
                            v_bsz = T.axis.spatial(T.int64(1), T.int64(0))
                            v_i = T.axis.spatial(n, bsz_i_fused)
                            v_k = T.axis.spatial(T.int64(4096), k_0 * T.int64(256) + k_1)
                            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp_shared[v_bsz, v_i])
                            T.writes(rms_norm_1[v_bsz, v_i, v_k])
                            rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp_shared[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
    # fmt: on
    _check(Before, After)


if __name__ == "__main__":
    test_softmax()
    test_layer_norm()
    test_rms_norm()
