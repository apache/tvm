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
# pylint: disable=invalid-name,,missing-function-docstring
from typing import List, Tuple
import numpy as np

import tvm
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tvm.tir.tensor_intrin.x86 import *  # pylint: disable=wildcard-import,unused-wildcard-import
from tvm.meta_schedule.testing.tune_utils import generate_input_data
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.relax.transform import FewShotTuning
import tvm.testing

import pytest

# pylint: disable=no-self-argument,missing-class-docstring,line-too-long
# fmt: off
@tvm.script.ir_module
class MatMul:
    @T.prim_func
    def matmul(
        A: T.Buffer((32, 32), "float16"),
        B: T.Buffer((32, 32), "float16"),
        C: T.Buffer((32, 32), "float16"),
    ):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i, j, k in T.grid(32, 32, 32):
            with T.block("C"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(C[v_i, v_j])
                with T.init():
                    C[v_i, v_j] = T.float16(0)
                C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

@tvm.script.ir_module
class Softmax:
    @T.prim_func
    def softmax(rxplaceholder: T.Buffer((T.int64(8), T.int64(3456), T.int64(3456)), "float32"), T_softmax_norm: T.Buffer((T.int64(8), T.int64(3456), T.int64(3456)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": True})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(8), T.int64(3456)), "float32")
        T_softmax_exp = T.alloc_buffer((T.int64(8), T.int64(3456), T.int64(3456)), "float32")
        T_softmax_expsum = T.alloc_buffer((T.int64(8), T.int64(3456)), "float32")
        for i0, i1, k in T.grid(T.int64(8), T.int64(3456), T.int64(3456)):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rxplaceholder[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float16(-65504)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], rxplaceholder[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(T.int64(8), T.int64(3456), T.int64(3456)):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(rxplaceholder[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(rxplaceholder[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(T.int64(8), T.int64(3456), T.int64(3456)):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1] = T.float16(0)
                T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
        for i0, i1, i2 in T.grid(T.int64(8), T.int64(3456), T.int64(3456)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T.block_attr({"axis": 2})
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

@tvm.script.ir_module
class Fused_Variance_Cast1:
    @T.prim_func
    def main(lv3: T.Buffer((T.int64(1), T.int64(32), T.int64(34560)), "float32"), compute: T.Buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        rxplaceholder_red = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(34560)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(34560)))
        T_multiply_red = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_divide_1 = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        for ax0, ax1, ax2, k2 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(34560)):
            with T.block("rxplaceholder_red"):
                v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                T.reads(lv3[v_ax0, v_ax1, v_k2])
                T.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2])
                with T.init():
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2] = T.float32(0)
                rxplaceholder_red[v_ax0, v_ax1, v_ax2] = rxplaceholder_red[v_ax0, v_ax1, v_ax2] + lv3[v_ax0, v_ax1, v_k2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(1)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(rxplaceholder_red[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = rxplaceholder_red[v_ax0, v_ax1, v_ax2] * T.float32(2.8935185185185186e-05)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(34560)):
            with T.block("T_subtract"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv3[v_ax0, v_ax1, v_ax2], T_divide[v_ax0, v_ax1, T.int64(0)])
                T.writes(T_subtract[v_ax0, v_ax1, v_ax2])
                T_subtract[v_ax0, v_ax1, v_ax2] = lv3[v_ax0, v_ax1, v_ax2] - T_divide[v_ax0, v_ax1, T.int64(0)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(34560)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_subtract[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = T_subtract[v_ax0, v_ax1, v_ax2] * T_subtract[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2, k2 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(34560)):
            with T.block("T_multiply_red"):
                v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                T.reads(T_multiply[v_ax0, v_ax1, v_k2])
                T.writes(T_multiply_red[v_ax0, v_ax1, v_ax2])
                with T.init():
                    T_multiply_red[v_ax0, v_ax1, v_ax2] = T.float32(0)
                T_multiply_red[v_ax0, v_ax1, v_ax2] = T_multiply_red[v_ax0, v_ax1, v_ax2] + T_multiply[v_ax0, v_ax1, v_k2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(1)):
            with T.block("T_divide_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_red[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide_1[v_ax0, v_ax1, v_ax2])
                T_divide_1[v_ax0, v_ax1, v_ax2] = T_multiply_red[v_ax0, v_ax1, v_ax2] * T.float32(2.8935185185185186e-05)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(32), T.int64(1)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_divide_1[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", T_divide_1[v_i0, v_i1, v_i2])

@tvm.script.ir_module
class Fuse_Mean_Cast1:
    @T.prim_func
    def main(lv: T.Buffer((T.int64(1), T.int64(32), T.int64(34560)), "float32"), compute: T.Buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        rxplaceholder_red = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
        for ax0, ax1, ax2, k2 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(34560)):
            with T.block("rxplaceholder_red"):
                v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                T.reads(lv[v_ax0, v_ax1, v_k2])
                T.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2])
                with T.init():
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2] = T.float32(0)
                rxplaceholder_red[v_ax0, v_ax1, v_ax2] = rxplaceholder_red[v_ax0, v_ax1, v_ax2] + lv[v_ax0, v_ax1, v_k2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(1)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(rxplaceholder_red[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = rxplaceholder_red[v_ax0, v_ax1, v_ax2] * T.float32(2.8935185185185186e-05)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(32), T.int64(1)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_divide[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float16", T_divide[v_i0, v_i1, v_i2])

@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(lv26: T.Buffer((T.int64(1), T.int64(3456), T.int64(2560)), "float16"), T_multiply: T.Buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        T_strided_slice_with_axes = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_divide = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_2 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        compute = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)))
        compute_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)))
        compute_2 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_3 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_add = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_5 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_divide_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_add_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_add_2 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_6 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_strided_slice_with_axes_1 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        T_multiply_7 = T.alloc_buffer((T.int64(1), T.int64(3456), T.int64(1280)), "float16")
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv26[v_ax0, v_ax1, v_ax2 + T.int64(1280)])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] = lv26[v_ax0, v_ax1, v_ax2 + T.int64(1280)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] * T.float16(0.70718232044198892)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_divide[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T_multiply_1[v_ax0, v_ax1, v_ax2] = T_divide[v_ax0, v_ax1, v_ax2] * T.float16(1.4140625)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_2[v_ax0, v_ax1, v_ax2])
                T_multiply_2[v_ax0, v_ax1, v_ax2] = T_multiply_1[v_ax0, v_ax1, v_ax2] * T.float16(0.70710678118654757)
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_multiply_2[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.Cast("float32", T_multiply_2[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("compute_1"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute[v_i0, v_i1, v_i2])
                T.writes(compute_1[v_i0, v_i1, v_i2])
                compute_1[v_i0, v_i1, v_i2] = T.erf(compute[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(compute_1[v_i0, v_i1, v_i2])
                T.writes(compute_2[v_i0, v_i1, v_i2])
                compute_2[v_i0, v_i1, v_i2] = T.Cast("float16", compute_1[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_1_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(compute_2[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_3[v_ax0, v_ax1, v_ax2])
                T_multiply_3[v_ax0, v_ax1, v_ax2] = compute_2[v_ax0, v_ax1, v_ax2] * T.float16(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_3[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = T.float16(0.5) + T_multiply_3[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2])
                T_multiply_4[v_ax0, v_ax1, v_ax2] = T_multiply_1[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_3"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_4[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_5[v_ax0, v_ax1, v_ax2])
                T_multiply_5[v_ax0, v_ax1, v_ax2] = T_multiply_4[v_ax0, v_ax1, v_ax2] * T.float16(1.4140625)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_divide_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_5[v_ax0, v_ax1, v_ax2], T_divide[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide_1[v_ax0, v_ax1, v_ax2])
                T_divide_1[v_ax0, v_ax1, v_ax2] = T_multiply_5[v_ax0, v_ax1, v_ax2] / T_divide[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_divide_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_1[v_ax0, v_ax1, v_ax2])
                T_add_1[v_ax0, v_ax1, v_ax2] = T_divide_1[v_ax0, v_ax1, v_ax2] + T.float16(-1)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_add_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_2[v_ax0, v_ax1, v_ax2])
                T_add_2[v_ax0, v_ax1, v_ax2] = T_add_1[v_ax0, v_ax1, v_ax2] + T.float16(1)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_4"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2], T_add_2[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_6[v_ax0, v_ax1, v_ax2])
                T_multiply_6[v_ax0, v_ax1, v_ax2] = T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] * T_add_2[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_strided_slice_with_axes_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv26[v_ax0, v_ax1, v_ax2])
                T.writes(T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2])
                T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2] = lv26[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_5"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_multiply_6[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply_7[v_ax0, v_ax1, v_ax2])
                T_multiply_7[v_ax0, v_ax1, v_ax2] = T_multiply_6[v_ax0, v_ax1, v_ax2] * T.float16(0.5)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(3456), T.int64(1280)):
            with T.block("T_multiply_6"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2], T_multiply_7[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2] * T_multiply_7[v_ax0, v_ax1, v_ax2]
# fmt: on
# pylint: enable=no-self-argument,missing-class-docstring,line-too-long


def _target() -> tvm.target.Target:
    return tvm.target.Target("llvm -num-cores=4")
    # for local testing only
    # return tvm.target.Target("nvidia/geforce-rtx-3070")


def _acc() -> float:
    return 1e-2 if _target().kind.name == "cuda" else 1e-7


def _get_single_prim_func(mod: tvm.ir.IRModule) -> tvm.tir.PrimFunc:
    funcs = [func for func in mod.functions.values()]
    assert len(funcs) == 1, "Only one function is supported."
    return funcs[0]


def _get_input_output_info(func: tvm.tir.PrimFunc) -> Tuple[List[np.ndarray], Tuple, str]:
    args = ArgInfo.from_prim_func(func)
    inputs = [generate_input_data(x.shape, x.dtype) for x in args[:-1]]
    output_shape = args[-1].shape
    output_dtype = args[-1].dtype
    return inputs, output_shape, output_dtype


def _expected_results(
    mod: tvm.ir.IRModule, inputs: List[np.ndarray], output_shape: Tuple, output_dtype: str
) -> np.ndarray:
    func = _get_single_prim_func(mod)
    func = func.with_attr("global_symbol", "main")
    rt_mod = tvm.build(func, target="llvm")
    data = [
        tvm.nd.array(x)
        for x in [
            *inputs,
            np.zeros(output_shape, dtype=output_dtype),
        ]
    ]
    rt_mod(*data)
    return data[-1].numpy()


def _actual_results(
    actual: tvm.ir.IRModule, inputs: List[np.ndarray], output_shape: Tuple, output_dtype: str
):
    target = _target()
    actual_rt_mod = tvm.build(actual, target=target)
    actual_data = [
        tvm.nd.array(x, device=tvm.cuda() if target.kind.name == "cuda" else tvm.cpu())
        for x in [
            *inputs,
            np.zeros(output_shape, dtype=output_dtype),
        ]
    ]
    actual_rt_mod(*actual_data)
    return actual_data[-1].numpy()


def _assert_allclose(mod: tvm.ir.IRModule, actual: tvm.ir.IRModule) -> None:
    inputs, output_shape, output_dtype = _get_input_output_info(_get_single_prim_func(mod))
    expected_output = _expected_results(mod, inputs, output_shape, output_dtype)
    actual_output = _actual_results(actual, inputs, output_shape, output_dtype)
    tvm.testing.assert_allclose(expected_output, actual_output, rtol=_acc(), atol=_acc())


# Fused_Variance_Cast1 not added due to https://github.com/apache/tvm/issues/14791
@pytest.mark.parametrize("mod", [Softmax, MatMul, Fuse_Mean_Cast1, Module])
@pytest.mark.parametrize("benchmark", [False, True])
def test_funcs(mod: tvm.ir.IRModule, benchmark: bool) -> None:
    valid_count = 10 if benchmark else 1
    with _target(), tvm.transform.PassContext(opt_level=3):
        actual = FewShotTuning(valid_count=valid_count)(mod)
    assert _get_single_prim_func(actual).attrs["tir.is_scheduled"], "Schedule is not applied."
    _assert_allclose(mod, actual)


if __name__ == "__main__":
    tvm.testing.main()
