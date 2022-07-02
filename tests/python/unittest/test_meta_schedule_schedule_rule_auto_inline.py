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
import tvm
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.schedule_rule import auto_inline
from tvm.meta_schedule.tune_context import TuneContext
from tvm.script import tir as T
from tvm.target import Target

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class Conv2DBiasBnReLU:
    @T.prim_func
    def main(var_X: T.handle, var_W: T.handle, var_B: T.handle, var_bn_scale: T.handle, var_bn_offset: T.handle, var_compute: T.handle) -> None:
        X = T.match_buffer(var_X, [1, 512, 56, 56], dtype="float32")
        W = T.match_buffer(var_W, [512, 512, 3, 3], dtype="float32")
        B = T.match_buffer(var_B, [512, 1, 1], dtype="float32")
        bn_scale = T.match_buffer(var_bn_scale, [512, 1, 1], dtype="float32")
        bn_offset = T.match_buffer(var_bn_offset, [512, 1, 1], dtype="float32")
        compute = T.match_buffer(var_compute, [1, 512, 56, 56], dtype="float32")
        pad_temp = T.alloc_buffer([1, 512, 58, 58], dtype="float32")
        compute_1 = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        bias_add = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        bn_mul = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        bn_add = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 512, 56, 56, 512, 3, 3):
            with T.block("compute"):
                nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                with T.init():
                    compute_1[nn, ff, yy, xx] = T.float32(0)
                compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * W[ff, rc, ry, rx]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("bias_add"):
                i, j, k, l = T.axis.remap("SSSS", [i0, i1, i2, i3])
                bias_add[i, j, k, l] = compute_1[i, j, k, l] + B[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("bn_mul"):
                i, j, k, l = T.axis.remap("SSSS", [i0, i1, i2, i3])
                bn_mul[i, j, k, l] = bias_add[i, j, k, l] * bn_scale[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("bn_add"):
                i, j, k, l = T.axis.remap("SSSS", [i0, i1, i2, i3])
                bn_add[i, j, k, l] = bn_mul[i, j, k, l] + bn_offset[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max(bn_add[i0_2, i1_2, i2_2, i3_2], T.float32(0))


@tvm.script.ir_module
class Conv2DBiasBnReLUInlined:
    @T.prim_func
    def main(var_X: T.handle, var_W: T.handle, var_B: T.handle, var_bn_scale: T.handle, var_bn_offset: T.handle, var_compute: T.handle) -> None:
        X = T.match_buffer(var_X, [1, 512, 56, 56], dtype="float32")
        W = T.match_buffer(var_W, [512, 512, 3, 3], dtype="float32")
        B = T.match_buffer(var_B, [512, 1, 1], dtype="float32")
        bn_scale = T.match_buffer(var_bn_scale, [512, 1, 1], dtype="float32")
        bn_offset = T.match_buffer(var_bn_offset, [512, 1, 1], dtype="float32")
        compute = T.match_buffer(var_compute, [1, 512, 56, 56], dtype="float32")
        pad_temp = T.alloc_buffer([1, 512, 58, 58], dtype="float32")
        compute_1 = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 512, 56, 56, 512, 3, 3):
            with T.block("compute"):
                nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                with T.init():
                    compute_1[nn, ff, yy, xx] = T.float32(0)
                compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * W[ff, rc, ry, rx]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max((compute_1[i0_2, i1_2, i2_2, i3_2] + B[i1_2, 0, 0]) * bn_scale[i1_2, 0, 0] + bn_offset[i1_2, 0, 0], T.float32(0))


@tvm.script.ir_module
class MultiLevelTiledConv2D:
    @T.prim_func
    def main(var_X: T.handle, var_W: T.handle, var_B: T.handle, var_bn_scale: T.handle, var_bn_offset: T.handle, var_compute: T.handle) -> None:
        X = T.match_buffer(var_X, [1, 512, 56, 56], dtype="float32")
        W = T.match_buffer(var_W, [512, 512, 3, 3], dtype="float32")
        B = T.match_buffer(var_B, [512, 1, 1], dtype="float32")
        bn_scale = T.match_buffer(var_bn_scale, [512, 1, 1], dtype="float32")
        bn_offset = T.match_buffer(var_bn_offset, [512, 1, 1], dtype="float32")
        compute = T.match_buffer(var_compute, [1, 512, 56, 56], dtype="float32")
        pad_temp = T.alloc_buffer([1, 512, 58, 58], dtype="float32")
        compute_1 = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        compute_local = T.alloc_buffer([1, 512, 56, 56], dtype="float32", scope="local")
        pad_temp_shared = T.alloc_buffer([1, 512, 58, 58], dtype="float32", scope="shared")
        W_shared = T.alloc_buffer([512, 512, 3, 3], dtype="float32", scope="shared")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(0, 224, thread="blockIdx.x"):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(0, 2, thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i4_0, i5_0, i6_0 in T.grid(1, 3, 1):
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 40960, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 3):
                                with T.block("pad_temp_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 // 8 % 512)
                                    v2 = T.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i5_0 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 % 8)
                                    v3 = T.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) % 30)
                                    pad_temp_shared[v0, v1, v2, v3] = pad_temp[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 12288, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 4):
                                with T.block("W_shared"):
                                    v0 = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 1536)
                                    v1 = T.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 3 % 512)
                                    v2 = T.axis.spatial(3, i5_0)
                                    v3 = T.axis.spatial(3, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) % 3)
                                    W_shared[v0, v1, v2, v3] = W[v0, v1, v2, v3]
                        for i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(32, 1, 1, 1, 1, 1, 1, 16, 1, 3, 1, 8, 2, 28):
                            with T.block("compute"):
                                nn = T.axis.spatial(1, 0)
                                ff = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + i1_4)
                                yy = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused // 2 % 7 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + i2_4)
                                xx = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + i3_4)
                                rc = T.axis.reduce(512, i4_1 * 16 + i4_2)
                                ry, rx = T.axis.remap("RR", [i5_0, i6_2])
                                with T.init():
                                    compute_local[nn, ff, yy, xx] = T.float32(0)
                                compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + pad_temp_shared[nn, rc, yy + ry, xx + rx] * W_shared[ff, rc, ry, rx]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 8, 2, 28):
                        with T.block("compute_local"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + ax1)
                            v2 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + ax2)
                            v3 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + ax3)
                            compute_1[v0, v1, v2, v3] = compute_local[v0, v1, v2, v3]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max((compute_1[i0_2, i1_2, i2_2, i3_2] + B[i1_2, 0, 0]) * bn_scale[i1_2, 0, 0] + bn_offset[i1_2, 0, 0], T.float32(0))


@tvm.script.ir_module
class MultiLevelTiledConv2DAfterInline:
    @T.prim_func
    def main(X: T.Buffer[(1, 512, 56, 56), "float32"], W: T.Buffer[(512, 512, 3, 3), "float32"], B: T.Buffer[(512, 1, 1), "float32"], bn_scale: T.Buffer[(512, 1, 1), "float32"], bn_offset: T.Buffer[(512, 1, 1), "float32"], compute: T.Buffer[(1, 512, 56, 56), "float32"]) -> None:
        compute_local = T.alloc_buffer([1, 512, 56, 56], dtype="float32", scope="local")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(224, thread="blockIdx.x"):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(8, thread="threadIdx.x"):
                    for i4_0, i5_0, i6_0, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(1, 3, 1, 32, 1, 1, 1, 1, 1, 1, 16, 1, 3, 1, 8, 2, 28):
                        with T.block("compute"):
                            nn = T.axis.spatial(1, 0)
                            ff = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + i1_4)
                            yy = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused // 2 % 7 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + i2_4)
                            xx = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + i3_4)
                            rc = T.axis.reduce(512, i4_1 * 16 + i4_2)
                            ry, rx = T.axis.remap("RR", [i5_0, i6_2])
                            with T.init():
                                compute_local[nn, ff, yy, xx] = T.float32(0)
                            compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + T.if_then_else(yy + ry >= 1 and yy + ry < 57 and xx + rx >= 1 and xx + rx < 57, X[nn, rc, yy + ry - 1, xx + rx - 1], T.float32(0), dtype="float32") * W[ff, rc, ry, rx]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 8, 2, 28):
                        with T.block("compute_local"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + ax1)
                            v2 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + ax2)
                            v3 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + ax3)
                            compute[v0, v1, v2, v3] = T.max((compute_local[v0, v1, v2, v3] + B[v1, 0, 0]) * bn_scale[v1, 0, 0] + bn_offset[v1, 0, 0], T.float32(0))


@tvm.script.ir_module
class SoftmaxBeforeInline:
    @T.prim_func
    def main(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.min_value("float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_exp"):
                i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
                T_softmax_exp[i0_2, i1_1] = T.exp(A[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32")
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_4, k = T.axis.remap("SR", [i0_3, i1])
                with T.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]


@tvm.script.ir_module
class SoftmaxAfterInline:
    @T.prim_func
    def main(A: T.Buffer[(256, 256), "float32"], T_softmax_norm: T.Buffer[(256, 256), "float32"]) -> None:
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.min_value("float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_2, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(A[i0_2, k] - T_softmax_maxelem[i0_2], dtype="float32")
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_4, i1_1 = T.axis.remap("SS", [i0_3, i1])
                T_softmax_norm[i0_4, i1_1] = T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4], dtype="float32") / T_softmax_expsum[i0_4]


@tvm.script.ir_module
class BeforePureSpatial:
    @T.prim_func
    def main(
        placeholder: T.Buffer[(1, 384), "int64"],
        placeholder_1: T.Buffer[(30522, 768), "float32"],
        placeholder_2: T.Buffer[(1, 384, 768), "float32"],
        T_add: T.Buffer[(1, 384, 768), "float32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        compile_engine_const = T.alloc_buffer([], dtype="int64")
        T_less = T.alloc_buffer([1, 384], dtype="bool")
        compile_engine_const_1 = T.alloc_buffer([], dtype="int64")
        T_add_1 = T.alloc_buffer([1, 384], dtype="int64")
        T_where = T.alloc_buffer([1, 384], dtype="int64")
        T_take = T.alloc_buffer([1, 384, 768], dtype="float32")
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = T.int64(0)
        for i0, i1 in T.grid(1, 384):
            with T.block("T_less"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(placeholder[ax0, ax1], compile_engine_const[()])
                T.writes(T_less[ax0, ax1])
                T_less[ax0, ax1] = placeholder[ax0, ax1] < compile_engine_const[()]
        with T.block("compile_engine_const_1"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const_1[()])
            compile_engine_const_1[()] = T.int64(30522)
        for i0, i1 in T.grid(1, 384):
            with T.block("T_add"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(placeholder[ax0, ax1], compile_engine_const_1[()])
                T.writes(T_add_1[ax0, ax1])
                T_add_1[ax0, ax1] = placeholder[ax0, ax1] + compile_engine_const_1[()]
        for i0, i1 in T.grid(1, 384):
            with T.block("T_where"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_less[ax0, ax1], T_add_1[ax0, ax1], placeholder[ax0, ax1])
                T.writes(T_where[ax0, ax1])
                T_where[ax0, ax1] = T.Select(
                    T.cast(T_less[ax0, ax1], "int32") != 0, T_add_1[ax0, ax1], placeholder[ax0, ax1]
                )
        for i0, i1, i2 in T.grid(1, 384, 768):
            with T.block("T_take"):
                ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(
                    placeholder_1[T.min(T.max(T.int64(0), T_where[ax0, ax1]), T.int64(30521)), ax2],
                    T_where[ax0, ax1],
                )
                T.writes(T_take[ax0, ax1, ax2])
                T_take[ax0, ax1, ax2] = placeholder_1[
                    T.min(T.max(T.int64(0), T_where[ax0, ax1]), T.int64(30521)), ax2
                ]
        for i0, i1, i2 in T.grid(1, 384, 768):
            with T.block("T_add_1"):
                ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_take[ax0, ax1, ax2], placeholder_2[ax0, ax1, ax2])
                T.writes(T_add[ax0, ax1, ax2])
                T_add[ax0, ax1, ax2] = T_take[ax0, ax1, ax2] + placeholder_2[ax0, ax1, ax2]


@tvm.script.ir_module
class AfterPureSpatial:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 384), "int64"], placeholder_1: T.Buffer[(30522, 768), "float32"], placeholder_2: T.Buffer[(1, 384, 768), "float32"], T_add: T.Buffer[(1, 384, 768), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0, i1, i2 in T.grid(1, 384, 768):
            with T.block("T_add_1"):
                ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(placeholder[ax0, ax1], placeholder_1[T.min(T.max(T.int64(0), placeholder[ax0, ax1]), T.int64(30521)) : T.min(T.max(T.int64(0), placeholder[ax0, ax1] + T.int64(30522)), T.int64(30521)) + T.int64(1), ax2], placeholder_2[ax0, ax1, ax2])
                T.writes(T_add[ax0, ax1, ax2])
                T_add[ax0, ax1, ax2] = placeholder_1[T.min(T.max(T.int64(0), T.Select(T.cast(placeholder[ax0, ax1] < T.int64(0), "int32") != 0, placeholder[ax0, ax1] + T.int64(30522), placeholder[ax0, ax1])), T.int64(30521)), ax2] + placeholder_2[ax0, ax1, ax2]

# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def _create_context(mod, target, rule):
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    return ctx


def test_inline_consumer_chain():
    mod = Conv2DBiasBnReLU
    target = Target("llvm")
    ctx = _create_context(
        mod=mod,
        target=target,
        rule=auto_inline(target=target),
    )
    (space,) = ctx.space_generator.generate_design_space(mod=mod)
    tvm.ir.assert_structural_equal(lhs=space.mod, rhs=Conv2DBiasBnReLUInlined)


def test_inline_into_cache():
    mod = MultiLevelTiledConv2D
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        mod=mod,
        target=target,
        rule=auto_inline(target=target),
    )
    (space,) = ctx.space_generator.generate_design_space(mod=mod)
    tvm.ir.assert_structural_equal(lhs=space.mod, rhs=MultiLevelTiledConv2DAfterInline)


def test_inline_into_multiple_consumers():
    mod = SoftmaxBeforeInline
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        mod=mod,
        target=target,
        rule=auto_inline(target=target),
    )
    (space,) = ctx.space_generator.generate_design_space(mod=mod)
    tvm.ir.assert_structural_equal(lhs=space.mod, rhs=SoftmaxAfterInline)


def test_inline_pure_spatial():
    mod = BeforePureSpatial
    target = Target("llvm")
    ctx = _create_context(
        mod=mod,
        target=target,
        rule=auto_inline(target=target),
    )
    (space,) = ctx.space_generator.generate_design_space(mod=mod)
    tvm.ir.assert_structural_equal(lhs=space.mod, rhs=AfterPureSpatial)


if __name__ == "__main__":
    test_inline_consumer_chain()
    test_inline_into_cache()
    test_inline_into_multiple_consumers()
    test_inline_pure_spatial()
