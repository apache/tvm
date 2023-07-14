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
from tvm import dlight as dl
from tvm.ir import IRModule, assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


def _check(mod_before: IRModule, mod_after: IRModule):
    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            dl.gpu.Transpose(),
        )(mod_before)
    assert_structural_equal(mod, mod_after)


def test_transpose():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(rxplaceholder: T.Buffer((T.int64(512), T.int64(4096)), "float32"), T_transpose: T.Buffer((T.int64(4096), T.int64(512)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(512)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T_transpose[v_ax0, v_ax1] = rxplaceholder[v_ax1, v_ax0]

    @I.ir_module
    class After:
        @T.prim_func
        def main(rxplaceholder: T.Buffer((T.int64(512), T.int64(4096)), "float32"), T_transpose: T.Buffer((T.int64(4096), T.int64(512)), "float32")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            rxplaceholder_shared = T.alloc_buffer((T.int64(512), T.int64(4096)), scope="shared")
            for ax0_0_0 in T.thread_binding(T.int64(512), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax1_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
                    for ax0_ax1_fused_0 in range(T.int64(1)):
                        for ax0_ax1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.unroll(T.int64(1)):
                                    with T.block("rxplaceholder_shared"):
                                        v0 = T.axis.spatial(T.int64(512), ax1_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(16) + ax0_ax1_fused_2 + ax0_ax1_fused_3) // T.int64(8))
                                        v1 = T.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(8) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(16) + ax0_ax1_fused_2 + ax0_ax1_fused_3) % T.int64(8))
                                        T.reads(rxplaceholder[v0, v1])
                                        T.writes(rxplaceholder_shared[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 1]]})
                                        rxplaceholder_shared[v0, v1] = rxplaceholder[v0, v1]
                    for ax0_0_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax1_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                            for ax0_1_0 in range(T.int64(1)):
                                for ax0_1_1 in range(T.int64(1)):
                                    with T.block("T_transpose"):
                                        v0 = T.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(8) + ax0_0_1 + ax0_1_0 + ax0_1_1)
                                        v1 = T.axis.spatial(T.int64(512), ax1_0 * T.int64(16) + ax1_1)
                                        T.reads(rxplaceholder_shared[v1, v0])
                                        T.writes(T_transpose[v0, v1])
                                        T_transpose[v0, v1] = rxplaceholder_shared[v1, v0]
    # fmt: on
    _check(Before, After)


def test_decode_transpose():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(rxplaceholder: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            decode = T.alloc_buffer((T.int64(4096), T.int64(4096)))
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(rxplaceholder[v_i // T.int64(8), v_j], rxplaceholder_1[v_i // T.int64(32), v_j])
                    T.writes(decode[v_i, v_j])
                    decode[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(rxplaceholder[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(rxplaceholder_1[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(decode[v_ax1, v_ax0])
                    T.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = decode[v_ax1, v_ax0]

    @I.ir_module
    class After:
        @T.prim_func
        def main(rxplaceholder: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(4096)), "uint32"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float32")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            decode_shared = T.alloc_buffer((T.int64(4096), T.int64(4096)), scope="shared")
            for ax0_0_0 in T.thread_binding(T.int64(64), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax1_0 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                    for ax0_ax1_fused_0 in range(T.int64(1)):
                        for ax0_ax1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.unroll(T.int64(8)):
                                    with T.block("decode_shared"):
                                        v0 = T.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(64))
                                        v1 = T.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(64))
                                        T.reads(rxplaceholder[v0 // T.int64(8), v1], rxplaceholder_1[v0 // T.int64(32), v1])
                                        T.writes(decode_shared[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 1]]})
                                        decode_shared[v0, v1] = T.Cast("float32", T.bitwise_and(T.shift_right(rxplaceholder[v0 // T.int64(8), v1], T.Cast("uint32", v0 % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret("float32", T.shift_left(T.bitwise_and(rxplaceholder_1[v0 // T.int64(32), v1], T.uint32(65535)), T.uint32(16))) + T.reinterpret("float32", T.shift_left(T.bitwise_and(T.shift_right(rxplaceholder_1[v0 // T.int64(32), v1], T.uint32(16)), T.uint32(65535)), T.uint32(16)))
                    for ax0_0_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax1_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                            for ax0_1_0 in range(T.int64(2)):
                                for ax0_1_1 in T.vectorized(T.int64(4)):
                                    with T.block("T_transpose"):
                                        v0 = T.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(64) + ax0_0_1 * T.int64(8) + ax0_1_0 * T.int64(4) + ax0_1_1)
                                        v1 = T.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + ax1_1)
                                        T.reads(decode_shared[v1, v0])
                                        T.writes(T_transpose[v0, v1])
                                        T_transpose[v0, v1] = decode_shared[v1, v0]
    # fmt: on
    _check(Before, After)


def test_decode_int3_transpose():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((T.int64(412), T.int64(4096)), "uint32"), B: T.Buffer((T.int64(103), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float16")):
            T.func_attr({"tir.noalias": T.bool(True)})
            decode_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i // T.int64(10), v_j], B[v_i // T.int64(40), v_j])
                    T.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v_i // T.int64(10), v_j], T.Cast("uint32", v_i % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v_i // T.int64(40), v_j]
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(decode_1[v_ax1, v_ax0])
                    T.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = decode_1[v_ax1, v_ax0]

    @I.ir_module
    class After:
        @T.prim_func
        def main(A: T.Buffer((T.int64(412), T.int64(4096)), "uint32"), B: T.Buffer((T.int64(103), T.int64(4096)), "float16"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float16")):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            decode_1_shared = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16", scope="shared")
            for ax0_0_0 in T.thread_binding(T.int64(52), thread="blockIdx.y", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                for ax1_0 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                    for ax0_ax1_fused_0 in range(T.int64(2)):
                        for ax0_ax1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.unroll(T.int64(10)):
                                    with T.block("decode_1_shared"):
                                        v0 = T.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + (ax0_ax1_fused_0 * T.int64(1280) + ax0_ax1_fused_1 * T.int64(160) + ax0_ax1_fused_2 * T.int64(10) + ax0_ax1_fused_3) // T.int64(82))
                                        v1 = T.axis.spatial(T.int64(4096), ax0_0_0 * T.int64(80) + (ax0_ax1_fused_0 * T.int64(1280) + ax0_ax1_fused_1 * T.int64(160) + ax0_ax1_fused_2 * T.int64(10) + ax0_ax1_fused_3) % T.int64(82))
                                        T.where(ax0_0_0 * T.int64(80) + (((ax0_ax1_fused_0 * T.int64(8) + ax0_ax1_fused_1) * T.int64(16) + ax0_ax1_fused_2) * T.int64(10) + ax0_ax1_fused_3) % T.int64(82) < T.int64(4096) and ((ax0_ax1_fused_0 * T.int64(8) + ax0_ax1_fused_1) * T.int64(16) + ax0_ax1_fused_2) * T.int64(10) + ax0_ax1_fused_3 < T.int64(1312))
                                        T.reads(A[v0 // T.int64(10), v1], B[v0 // T.int64(40), v1])
                                        T.writes(decode_1_shared[v0, v1])
                                        T.block_attr({"buffer_dim_align": [[0, 0, 32, 1]]})
                                        decode_1_shared[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(A[v0 // T.int64(10), v1], T.Cast("uint32", v0 % T.int64(10)) * T.uint32(3)), T.uint32(7))) - T.float16(3)) * B[v0 // T.int64(40), v1]
                    for ax0_0_1 in T.thread_binding(T.int64(8), thread="threadIdx.y"):
                        for ax1_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                            for ax0_1_0 in range(T.int64(3)):
                                for ax0_1_1 in T.vectorized(T.int64(4)):
                                    with T.block("T_transpose"):
                                        v0 = T.axis.spatial(T.int64(4096), (ax0_0_0 * T.int64(8) + ax0_0_1) * T.int64(10) + (ax0_1_0 * T.int64(4) + ax0_1_1))
                                        v1 = T.axis.spatial(T.int64(4096), ax1_0 * T.int64(16) + ax1_1)
                                        T.where((ax0_0_0 * T.int64(8) + ax0_0_1) * T.int64(10) + (ax0_1_0 * T.int64(4) + ax0_1_1) < T.int64(4096) and ax0_0_0 * T.int64(8) + ax0_0_1 < T.int64(410) and ax0_1_0 * T.int64(4) + ax0_1_1 < T.int64(10))
                                        T.reads(decode_1_shared[v1, v0])
                                        T.writes(T_transpose[v0, v1])
                                        T_transpose[v0, v1] = decode_1_shared[v1, v0]
    # fmt: on
    _check(Before, After)
