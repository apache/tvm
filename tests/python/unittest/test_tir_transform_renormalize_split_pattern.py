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

import tvm
from tvm.script import tir as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg

@tvm.script.ir_module
class Before:
    @T.prim_func
    def main(inputs: T.Buffer[(8192,), "float32"], weight: T.Buffer[(2097152,), "float32"], conv2d_transpose_nhwc: T.Buffer[(16384,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        T.preflattened_buffer(inputs, [1, 4, 4, 512], dtype="float32", data=inputs.data)
        T.preflattened_buffer(weight, [4, 4, 512, 256], dtype="float32", data=weight.data)
        T.preflattened_buffer(conv2d_transpose_nhwc, [1, 8, 8, 256], dtype="float32", data=conv2d_transpose_nhwc.data)
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        # body
        T.launch_thread(blockIdx_x, 64)
        conv2d_transpose_nhwc_local = T.allocate([8], "float32", "local")
        PadInput_shared = T.allocate([768], "float32", "shared")
        weight_shared = T.allocate([4096], "float32", "shared")
        T.launch_thread(threadIdx_x, 32)
        for i2_3_init, i1_4_init, i2_4_init in T.grid(2, 2, 2):
            conv2d_transpose_nhwc_local[i1_4_init * 4 + i2_3_init * 2 + i2_4_init] = T.float32(0)
        for i6_0 in T.serial(16):
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(24):
                PadInput_shared[ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x] = T.if_then_else(128 <= ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x and ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x < 640 and 1 <= blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x) % 128 // 32 and blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x) % 128 // 32 < 5, inputs[blockIdx_x // 32 * 1024 + ax0_ax1_ax2_ax3_fused_0 * 512 + i6_0 * 32 + threadIdx_x - 2560], T.float32(0), dtype="float32")
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(32):
                weight_shared[T.ramp(ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4, 1, 4)] = weight[T.ramp((ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4) // 256 * 131072 + i6_0 * 8192 + (ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4) % 256 // 8 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 2 * 4, 1, 4)]
            for i6_1, i2_3, i4_2, i5_2, i6_2, i1_4, i2_4 in T.grid(4, 2, 4, 4, 8, 2, 2):
                conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] = conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] + T.if_then_else((i1_4 + i4_2) % 2 == 0 and (i2_4 + i5_2) % 2 == 0, PadInput_shared[threadIdx_x // 8 * 128 + (i1_4 + i4_2) // 2 * 128 + (i2_4 + i5_2) // 2 * 32 + i2_3 * 32 + i6_1 * 8 + i6_2], T.float32(0), dtype="float32") * weight_shared[i6_1 * 64 + i6_2 * 8 + threadIdx_x % 8 + 3840 - i5_2 * 256 - i4_2 * 1024]
        for ax1, ax2 in T.grid(2, 4):
            conv2d_transpose_nhwc[threadIdx_x // 8 * 4096 + ax1 * 2048 + blockIdx_x // 32 * 1024 + ax2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 8] = conv2d_transpose_nhwc_local[ax1 * 4 + ax2]


@tvm.script.ir_module
class After:
    @T.prim_func
    def main(inputs: T.Buffer[(8192,), "float32"], weight: T.Buffer[(2097152,), "float32"], conv2d_transpose_nhwc: T.Buffer[(16384,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        T.preflattened_buffer(inputs, [1, 4, 4, 512], dtype="float32", data=inputs.data)
        T.preflattened_buffer(weight, [4, 4, 512, 256], dtype="float32", data=weight.data)
        T.preflattened_buffer(conv2d_transpose_nhwc, [1, 8, 8, 256], dtype="float32", data=conv2d_transpose_nhwc.data)
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        # body
        T.launch_thread(blockIdx_x, 64)
        conv2d_transpose_nhwc_local = T.allocate([8], "float32", "local")
        PadInput_shared = T.allocate([768], "float32", "shared")
        weight_shared = T.allocate([4096], "float32", "shared")
        T.launch_thread(threadIdx_x, 32)
        for i2_3_init, i1_4_init, i2_4_init in T.grid(2, 2, 2):
            conv2d_transpose_nhwc_local[i1_4_init * 4 + i2_3_init * 2 + i2_4_init] = T.float32(0)
        for i6_0 in T.serial(16):
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(24):
                PadInput_shared[ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x] = T.if_then_else(1 <= (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) // 4 and (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) // 20 < 1 and 1 <= blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) % 4 and (blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) % 4) // 5 < 1, inputs[blockIdx_x // 32 * 1024 + ax0_ax1_ax2_ax3_fused_0 * 512 + i6_0 * 32 + threadIdx_x - 2560], T.float32(0), dtype="float32")
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(32):
                weight_shared[T.ramp(ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4, 1, 4)] = weight[T.ramp((ax0_ax1_ax2_ax3_fused_0 + threadIdx_x * 4 // 128) // 2 * 131072 + i6_0 * 8192 + (ax0_ax1_ax2_ax3_fused_0 * 16 + threadIdx_x * 4 // 8) % 32 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 2 * 4, 1, 4)]
            for i6_1, i2_3, i4_2, i5_2, i6_2, i1_4, i2_4 in T.grid(4, 2, 4, 4, 8, 2, 2):
                conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] = conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] + T.if_then_else((i1_4 + i4_2) % 2 == 0 and (i2_4 + i5_2) % 2 == 0, PadInput_shared[threadIdx_x // 8 * 128 + (i1_4 + i4_2) // 2 * 128 + (i2_4 + i5_2) // 2 * 32 + i2_3 * 32 + i6_1 * 8 + i6_2], T.float32(0), dtype="float32") * weight_shared[i6_1 * 64 + i6_2 * 8 + threadIdx_x % 8 + 3840 - i5_2 * 256 - i4_2 * 1024]
        for ax1, ax2 in T.grid(2, 4):
            conv2d_transpose_nhwc[threadIdx_x // 8 * 4096 + ax1 * 2048 + blockIdx_x // 32 * 1024 + ax2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 8] = conv2d_transpose_nhwc_local[ax1 * 4 + ax2]


@tvm.script.ir_module
class After_simplified:
    @T.prim_func
    def main(inputs: T.Buffer[(8192,), "float32"], weight: T.Buffer[(2097152,), "float32"], conv2d_transpose_nhwc: T.Buffer[(16384,), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        T.preflattened_buffer(inputs, [1, 4, 4, 512], dtype="float32", data=inputs.data)
        T.preflattened_buffer(weight, [4, 4, 512, 256], dtype="float32", data=weight.data)
        T.preflattened_buffer(conv2d_transpose_nhwc, [1, 8, 8, 256], dtype="float32", data=conv2d_transpose_nhwc.data)
        # body
        T.launch_thread(blockIdx_x, 64)
        conv2d_transpose_nhwc_local = T.allocate([8], "float32", "local")
        PadInput_shared = T.allocate([768], "float32", "shared")
        weight_shared = T.allocate([4096], "float32", "shared")
        T.launch_thread(threadIdx_x, 32)
        for i2_3_init, i1_4_init, i2_4_init in T.grid(2, 2, 2):
            conv2d_transpose_nhwc_local[i1_4_init * 4 + i2_3_init * 2 + i2_4_init] = T.float32(0)
        for i6_0 in T.serial(16):
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(24):
                PadInput_shared[ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x] = T.if_then_else(4 <= ax0_ax1_ax2_ax3_fused_0 and ax0_ax1_ax2_ax3_fused_0 < 20 and 1 <= blockIdx_x // 32 * 2 + ax0_ax1_ax2_ax3_fused_0 % 4 and blockIdx_x // 32 * 2 + ax0_ax1_ax2_ax3_fused_0 % 4 < 5, inputs[blockIdx_x // 32 * 1024 + ax0_ax1_ax2_ax3_fused_0 * 512 + i6_0 * 32 + threadIdx_x - 2560], T.float32(0), dtype="float32")
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(32):
                weight_shared[T.ramp(ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4, 1, 4)] = weight[T.ramp(ax0_ax1_ax2_ax3_fused_0 // 2 * 131072 + i6_0 * 8192 + ax0_ax1_ax2_ax3_fused_0 % 2 * 4096 + threadIdx_x // 2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 2 * 4, 1, 4)]
            for i6_1, i2_3, i4_2, i5_2, i6_2, i1_4, i2_4 in T.grid(4, 2, 4, 4, 8, 2, 2):
                conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] = conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] + T.if_then_else((i1_4 + i4_2) % 2 == 0 and (i2_4 + i5_2) % 2 == 0, PadInput_shared[threadIdx_x // 8 * 128 + (i1_4 + i4_2) // 2 * 128 + (i2_4 + i5_2) // 2 * 32 + i2_3 * 32 + i6_1 * 8 + i6_2], T.float32(0), dtype="float32") * weight_shared[i6_1 * 64 + i6_2 * 8 + threadIdx_x % 8 + 3840 - i5_2 * 256 - i4_2 * 1024]
        for ax1, ax2 in T.grid(2, 4):
            conv2d_transpose_nhwc[threadIdx_x // 8 * 4096 + ax1 * 2048 + blockIdx_x // 32 * 1024 + ax2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 8] = conv2d_transpose_nhwc_local[ax1 * 4 + ax2]

# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg
# fmt: on


def test_renormalize_split_pattern():
    after = tvm.tir.transform.RenormalizeSplitPattern()(Before)
    tvm.ir.assert_structural_equal(after, After)
    after = tvm.tir.transform.Simplify()(after)
    tvm.ir.assert_structural_equal(after, After_simplified)


if __name__ == "__main__":
    tesd_renormalize_split_pattern()
