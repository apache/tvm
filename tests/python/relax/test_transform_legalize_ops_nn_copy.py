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

# TODO remove
import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build

import pytest

import tvm
import tvm.testing
from tvm.relax.transform import LegalizeOps
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

##################### Neural network #####################

def test_batch_norm():
    # fmt: off
    @tvm.script.ir_module
    class BatchNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), gamma: R.Tensor((3,), "float32"), beta: R.Tensor((3,), "float32"), moving_mean: R.Tensor((3,), "float32"), moving_var: R.Tensor((3,), "float32")) -> R.Tuple(R.Tensor((2, 3, 28, 28), "float32"), R.Tensor((3,), "float32"), R.Tensor((3,), "float32")):
            gv: R.Tuple(R.Tensor((2, 3, 28, 28), "float32"), R.Tensor((3,), "float32"), R.Tensor((3,), "float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def batch_norm(var_x: T.handle, var_gamma: T.handle, var_beta: T.handle, var_moving_mean: T.handle, var_moving_var: T.handle, var_T_add: T.handle, var_T_add_1: T.handle, var_T_add_2: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            x = T.match_buffer(var_x, (T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            gamma = T.match_buffer(var_gamma, (T.int64(3),))
            beta = T.match_buffer(var_beta, (T.int64(3),))
            moving_mean = T.match_buffer(var_moving_mean, (T.int64(3),))
            moving_var = T.match_buffer(var_moving_var, (T.int64(3),))
            T_add = T.match_buffer(var_T_add, (T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_add_1 = T.match_buffer(var_T_add_1, (T.int64(3),))
            T_add_2 = T.match_buffer(var_T_add_2, (T.int64(3),))
            with T.block("root"):
                T.reads()
                T.writes()
                T_reshape = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_subtract = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_add_3 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                compute = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_divide = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_multiply = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_multiply_1 = T.alloc_buffer((T.int64(3),))
                x_red = T.alloc_buffer((T.int64(3),))
                T_divide_1 = T.alloc_buffer((T.int64(3),))
                T_multiply_2 = T.alloc_buffer((T.int64(3),))
                T_multiply_3 = T.alloc_buffer((T.int64(3),))
                T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
                T_subtract_1 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_subtract_2 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_multiply_4 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
                T_multiply_red = T.alloc_buffer((T.int64(3),))
                T_divide_2 = T.alloc_buffer((T.int64(3),))
                T_multiply_5 = T.alloc_buffer((T.int64(3),))
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with T.block("T_reshape"):
                                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(1), ax3)
                                    T.reads(moving_mean[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = moving_mean[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with T.block("T_subtract"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(28), ax3)
                                    T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with T.block("T_reshape_1"):
                                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(1), ax3)
                                    T.reads(moving_var[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = moving_var[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with T.block("T_add"):
                                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(1), ax3)
                                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
                for i0 in range(T.int64(1)):
                    for i1 in range(T.int64(3)):
                        for i2 in range(T.int64(1)):
                            for i3 in range(T.int64(1)):
                                with T.block("compute"):
                                    v_i0 = T.axis.spatial(T.int64(1), i0)
                                    v_i1 = T.axis.spatial(T.int64(3), i1)
                                    v_i2 = T.axis.spatial(T.int64(1), i2)
                                    v_i3 = T.axis.spatial(T.int64(1), i3)
                                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with T.block("T_divide"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(28), ax3)
                                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with T.block("T_reshape_2"):
                                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(1), ax3)
                                    T.reads(gamma[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = gamma[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with T.block("T_multiply"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(28), ax3)
                                    T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with T.block("T_reshape_3"):
                                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(1), ax3)
                                    T.reads(beta[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = beta[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with T.block("T_add_1"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(28), ax3)
                                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(3)):
                    with T.block("T_multiply_1"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(moving_mean[v_ax0])
                        T.writes(T_multiply_1[v_ax0])
                        T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * moving_mean[v_ax0]
                for ax0 in range(T.int64(3)):
                    for k0 in range(T.int64(2)):
                        for k2 in range(T.int64(28)):
                            for k3 in range(T.int64(28)):
                                with T.block("x_red"):
                                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                                    v_k0 = T.axis.reduce(T.int64(2), k0)
                                    v_k2 = T.axis.reduce(T.int64(28), k2)
                                    v_k3 = T.axis.reduce(T.int64(28), k3)
                                    T.reads(x[v_k0, v_ax0, v_k2, v_k3])
                                    T.writes(x_red[v_ax0])
                                    with T.init():
                                        x_red[v_ax0] = T.float32(0.0)
                                    x_red[v_ax0] = x_red[v_ax0] + x[v_k0, v_ax0, v_k2, v_k3]
                for ax0 in range(T.int64(3)):
                    with T.block("T_divide_1"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(x_red[v_ax0])
                        T.writes(T_divide_1[v_ax0])
                        T_divide_1[v_ax0] = x_red[v_ax0] * T.float32(0.00063775510204081628)
                for ax0 in range(T.int64(3)):
                    with T.block("T_multiply_2"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(T_divide_1[v_ax0])
                        T.writes(T_multiply_2[v_ax0])
                        T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
                for ax0 in range(T.int64(3)):
                    with T.block("T_add_2"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                        T.writes(T_add_1[v_ax0])
                        T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
                for ax0 in range(T.int64(3)):
                    with T.block("T_multiply_3"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(moving_var[v_ax0])
                        T.writes(T_multiply_3[v_ax0])
                        T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * moving_var[v_ax0]
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(1)):
                            for ax3 in range(T.int64(1)):
                                with T.block("T_reshape_4"):
                                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(1), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(1), ax3)
                                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with T.block("T_subtract_1"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(28), ax3)
                                    T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with T.block("T_subtract_2"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(28), ax3)
                                    T.reads(x[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = x[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(28)):
                            for ax3 in range(T.int64(28)):
                                with T.block("T_multiply_4"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(28), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(28), ax3)
                                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
                for ax0 in range(T.int64(3)):
                    for k0 in range(T.int64(2)):
                        for k2 in range(T.int64(28)):
                            for k3 in range(T.int64(28)):
                                with T.block("T_multiply_red"):
                                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                                    v_k0 = T.axis.reduce(T.int64(2), k0)
                                    v_k2 = T.axis.reduce(T.int64(28), k2)
                                    v_k3 = T.axis.reduce(T.int64(28), k3)
                                    T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                                    T.writes(T_multiply_red[v_ax0])
                                    with T.init():
                                        T_multiply_red[v_ax0] = T.float32(0.0)
                                    T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
                for ax0 in range(T.int64(3)):
                    with T.block("T_divide_2"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(T_multiply_red[v_ax0])
                        T.writes(T_divide_2[v_ax0])
                        T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.00063775510204081628)
                for ax0 in range(T.int64(3)):
                    with T.block("T_multiply_5"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(T_divide_2[v_ax0])
                        T.writes(T_multiply_5[v_ax0])
                        T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
                for ax0 in range(T.int64(3)):
                    with T.block("T_add_3"):
                        v_ax0 = T.axis.spatial(T.int64(3), ax0)
                        T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                        T.writes(T_add_2[v_ax0])
                        T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]
    
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), gamma: R.Tensor((3,), dtype="float32"), beta: R.Tensor((3,), dtype="float32"), moving_mean: R.Tensor((3,), dtype="float32"), moving_var: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")):
            cls = Expected
            gv = R.call_tir(cls.batch_norm, (x, gamma, beta, moving_mean, moving_var), out_sinfo=[R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(BatchNorm)
    tvm.ir.assert_structural_equal(mod, Expected)



# def test_batch_norm_symbolic():
#     # fmt: off
#     @tvm.script.ir_module
#     class BatchNorm:
#         @R.function
#         def main(x: R.Tensor(("n", "h", "w", "c"), "float32"), gamma: R.Tensor(("c",), "float32"), beta: R.Tensor(("c",), "float32"), moving_mean: R.Tensor(("c",), "float32"), moving_var: R.Tensor(("c",), "float32")) -> R.Tuple(R.Tensor(("n", "h", "w", "c"), "float32"), R.Tensor(("c",), "float32"), R.Tensor(("c",), "float32")):
#             n = T.int64()
#             h = T.int64()
#             w = T.int64()
#             c = T.int64()
#             gv: R.Tuple(R.Tensor((n, h, w, c), "float32"), R.Tensor((c,), "float32"), R.Tensor((c,), "float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
#             return gv

#     @tvm.script.ir_module
#     class Expected:
#         @T.prim_func(private=True)
#         def batch_norm(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_rxplaceholder_3: T.handle, var_rxplaceholder_4: T.handle, var_T_add: T.handle, var_T_add_1: T.handle, var_T_add_2: T.handle):
#             T.func_attr({"tir.noalias": True})
#             n = T.int64()
#             h = T.int64()
#             w = T.int64()
#             c = T.int64()
#             rxplaceholder = T.match_buffer(var_rxplaceholder, (n, h, w, c))
#             rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (c,))
#             rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, (c,))
#             rxplaceholder_3 = T.match_buffer(var_rxplaceholder_3, (c,))
#             rxplaceholder_4 = T.match_buffer(var_rxplaceholder_4, (c,))
#             T_add = T.match_buffer(var_T_add, (n, h, w, c))
#             T_add_1 = T.match_buffer(var_T_add_1, (T.max(c, h),))
#             T_add_2 = T.match_buffer(var_T_add_2, (T.max(c, h),))
#             # with T.block("root"):
#             rxplaceholder_red = T.alloc_buffer((h,))
#             T_divide = T.alloc_buffer((h,))
#             T_reshape = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
#             T_subtract = T.alloc_buffer((n, h, w, c))
#             T_subtract_1 = T.alloc_buffer((n, h, w, c))
#             T_subtract_2 = T.alloc_buffer((n, h, w, c))
#             T_multiply = T.alloc_buffer((n, h, w, c))
#             T_multiply_red = T.alloc_buffer((h,))
#             T_divide_1 = T.alloc_buffer((h,))
#             T_reshape_1 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
#             T_add_3 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
#             compute = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
#             T_divide_2 = T.alloc_buffer((n, h, w, c))
#             T_reshape_2 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
#             T_multiply_1 = T.alloc_buffer((n, h, w, c))
#             T_reshape_3 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
#             T_multiply_2 = T.alloc_buffer((c,))
#             T_multiply_3 = T.alloc_buffer((h,))
#             T_multiply_4 = T.alloc_buffer((c,))
#             T_subtract_3 = T.alloc_buffer((n, h, w, c))
#             T_subtract_4 = T.alloc_buffer((n, h, w, c))
#             T_multiply_5 = T.alloc_buffer((n, h, w, c))
#             T_multiply_red_1 = T.alloc_buffer((h,))
#             T_divide_3 = T.alloc_buffer((h,))
#             T_multiply_6 = T.alloc_buffer((h,))
#             for ax0, k0, k2, k3 in T.grid(h, n, w, c):
#                 with T.block("rxplaceholder_red"):
#                     v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
#                     T.reads(rxplaceholder[v_k0, v_ax0, v_k2, v_k3])
#                     T.writes(rxplaceholder_red[v_ax0])
#                     with T.init():
#                         rxplaceholder_red[v_ax0] = T.float32(0)
#                     rxplaceholder_red[v_ax0] = rxplaceholder_red[v_ax0] + rxplaceholder[v_k0, v_ax0, v_k2, v_k3]
#             for ax0 in range(h):
#                 with T.block("T_divide"):
#                     v_ax0 = T.axis.spatial(h, ax0)
#                     T.reads(rxplaceholder_red[v_ax0])
#                     T.writes(T_divide[v_ax0])
#                     T_divide[v_ax0] = rxplaceholder_red[v_ax0] / T.Cast("float32", n * w * c)
#             for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
#                 with T.block("T_reshape"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_divide[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h])
#                     T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_subtract"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_subtract_1"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_subtract_2"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_multiply"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
#             for ax0, k0, k2, k3 in T.grid(h, n, w, c):
#                 with T.block("T_multiply_red"):
#                     v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
#                     T.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
#                     T.writes(T_multiply_red[v_ax0])
#                     with T.init():
#                         T_multiply_red[v_ax0] = T.float32(0)
#                     T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
#             for ax0 in range(h):
#                 with T.block("T_divide_1"):
#                     v_ax0 = T.axis.spatial(h, ax0)
#                     T.reads(T_multiply_red[v_ax0])
#                     T.writes(T_divide_1[v_ax0])
#                     T_divide_1[v_ax0] = T_multiply_red[v_ax0] / T.Cast("float32", n * w * c)
#             for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
#                 with T.block("T_reshape_1"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_divide_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h])
#                     T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h]
#             for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
#                 with T.block("T_add"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
#             for i0, i1, i2, i3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
#                 with T.block("compute"):
#                     v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
#                     T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
#                     T.writes(compute[v_i0, v_i1, v_i2, v_i3])
#                     compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_divide_2"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
#                 with T.block("T_reshape_2"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(rxplaceholder_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c])
#                     T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_multiply_1"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
#                 with T.block("T_reshape_3"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(rxplaceholder_2[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c])
#                     T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_2[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_add_1"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0 in range(c):
#                 with T.block("T_multiply_2"):
#                     v_ax0 = T.axis.spatial(c, ax0)
#                     T.reads(rxplaceholder_3[v_ax0])
#                     T.writes(T_multiply_2[v_ax0])
#                     T_multiply_2[v_ax0] = T.float32(0.90000000000000002) * rxplaceholder_3[v_ax0]
#             for ax0 in range(h):
#                 with T.block("T_multiply_3"):
#                     v_ax0 = T.axis.spatial(h, ax0)
#                     T.reads(T_divide[v_ax0])
#                     T.writes(T_multiply_3[v_ax0])
#                     T_multiply_3[v_ax0] = T.float32(0.10000000000000001) * T_divide[v_ax0]
#             for ax0 in range(T.max(c, h)):
#                 with T.block("T_add_2"):
#                     v_ax0 = T.axis.spatial(T.max(c, h), ax0)
#                     T.reads(T_multiply_2[v_ax0], T_multiply_3[v_ax0])
#                     T.writes(T_add_1[v_ax0])
#                     T_add_1[v_ax0] = T_multiply_2[v_ax0] + T_multiply_3[v_ax0]
#             for ax0 in range(c):
#                 with T.block("T_multiply_4"):
#                     v_ax0 = T.axis.spatial(c, ax0)
#                     T.reads(rxplaceholder_4[v_ax0])
#                     T.writes(T_multiply_4[v_ax0])
#                     T_multiply_4[v_ax0] = T.float32(0.90000000000000002) * rxplaceholder_4[v_ax0]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_subtract_3"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_subtract_4"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
#                     T.writes(T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
#             for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
#                 with T.block("T_multiply_5"):
#                     v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
#                     T.reads(T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T.writes(T_multiply_5[v_ax0, v_ax1, v_ax2, v_ax3])
#                     T_multiply_5[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3]
#             for ax0, k0, k2, k3 in T.grid(h, n, w, c):
#                 with T.block("T_multiply_red_1"):
#                     v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
#                     T.reads(T_multiply_5[v_k0, v_ax0, v_k2, v_k3])
#                     T.writes(T_multiply_red_1[v_ax0])
#                     with T.init():
#                         T_multiply_red_1[v_ax0] = T.float32(0)
#                     T_multiply_red_1[v_ax0] = T_multiply_red_1[v_ax0] + T_multiply_5[v_k0, v_ax0, v_k2, v_k3]
#             for ax0 in range(h):
#                 with T.block("T_divide_3"):
#                     v_ax0 = T.axis.spatial(h, ax0)
#                     T.reads(T_multiply_red_1[v_ax0])
#                     T.writes(T_divide_3[v_ax0])
#                     T_divide_3[v_ax0] = T_multiply_red_1[v_ax0] / T.Cast("float32", n * w * c)
#             for ax0 in range(h):
#                 with T.block("T_multiply_6"):
#                     v_ax0 = T.axis.spatial(h, ax0)
#                     T.reads(T_divide_3[v_ax0])
#                     T.writes(T_multiply_6[v_ax0])
#                     T_multiply_6[v_ax0] = T.float32(0.10000000000000001) * T_divide_3[v_ax0]
#             for ax0 in range(T.max(c, h)):
#                 with T.block("T_add_3"):
#                     v_ax0 = T.axis.spatial(T.max(c, h), ax0)
#                     T.reads(T_multiply_4[v_ax0], T_multiply_6[v_ax0])
#                     T.writes(T_add_2[v_ax0])
#                     T_add_2[v_ax0] = T_multiply_4[v_ax0] + T_multiply_6[v_ax0]

#         @R.function
#         def main(x: R.Tensor(("n", "h", "w", "c"), dtype="float32"), gamma: R.Tensor(("c",), dtype="float32"), beta: R.Tensor(("c",), dtype="float32"), moving_mean: R.Tensor(("c",), dtype="float32"), moving_var: R.Tensor(("c",), dtype="float32")) -> R.Tuple(R.Tensor(("n", "h", "w", "c"), dtype="float32"), R.Tensor(("T.max(c,h)",), dtype="float32"), R.Tensor(("T.max(c,h)",), dtype="float32")):
#             n = T.int64()
#             h = T.int64()
#             w = T.int64()
#             c = T.int64()
#             gv = R.call_tir(Expected.batch_norm, (x, gamma, beta, moving_mean, moving_var), out_sinfo=[R.Tensor((n, h, w, c), dtype="float32"), R.Tensor((T.max(c, h),), dtype="float32"), R.Tensor((T.max(c, h),), dtype="float32")])
#             return gv
#     # fmt: on

#     mod = LegalizeOps()(BatchNorm)
#     tvm.ir.assert_structural_equal(mod, Expected)

if __name__ == "__main__":
    tvm.testing.main()
