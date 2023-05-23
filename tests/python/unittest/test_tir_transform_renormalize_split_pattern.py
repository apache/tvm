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
import tvm.testing
from tvm.script import tir as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg

@tvm.script.ir_module
class Before:
    @T.prim_func
    def main(inputs: T.Buffer((1, 4, 4, 512), "float32"), weight: T.Buffer((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Buffer((1, 8, 8, 256), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        inputs_flat = T.Buffer([8192], dtype="float32", data=inputs.data)
        weight_flat = T.Buffer([2097152], dtype="float32", data=weight.data)
        conv2d_transpose_nhwc_flat = T.Buffer([16384], dtype="float32", data=conv2d_transpose_nhwc.data)
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        # body
        T.launch_thread(blockIdx_x, 64)
        conv2d_transpose_nhwc_local = T.decl_buffer([8], "float32", scope="local")
        PadInput_shared = T.decl_buffer([768], "float32", scope="shared")
        weight_shared = T.decl_buffer([4096], "float32", scope="shared")
        T.launch_thread(threadIdx_x, 32)
        for i2_3_init, i1_4_init, i2_4_init in T.grid(2, 2, 2):
            conv2d_transpose_nhwc_local[i1_4_init * 4 + i2_3_init * 2 + i2_4_init] = T.float32(0)
        for i6_0 in T.serial(16):
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(24):
                PadInput_shared[ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x] = T.if_then_else(128 <= ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x and ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x < 640 and 1 <= blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x) % 128 // 32 and blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x) % 128 // 32 < 5, inputs_flat[blockIdx_x // 32 * 1024 + ax0_ax1_ax2_ax3_fused_0 * 512 + i6_0 * 32 + threadIdx_x - 2560], T.float32(0), dtype="float32")
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(32):
                weight_shared[T.ramp(ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4, 1, 4)] = weight_flat[T.ramp((ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4) // 256 * 131072 + i6_0 * 8192 + (ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4) % 256 // 8 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 2 * 4, 1, 4)]
            for i6_1, i2_3, i4_2, i5_2, i6_2, i1_4, i2_4 in T.grid(4, 2, 4, 4, 8, 2, 2):
                conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] = conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] + T.if_then_else((i1_4 + i4_2) % 2 == 0 and (i2_4 + i5_2) % 2 == 0, PadInput_shared[threadIdx_x // 8 * 128 + (i1_4 + i4_2) // 2 * 128 + (i2_4 + i5_2) // 2 * 32 + i2_3 * 32 + i6_1 * 8 + i6_2], T.float32(0), dtype="float32") * weight_shared[i6_1 * 64 + i6_2 * 8 + threadIdx_x % 8 + 3840 - i5_2 * 256 - i4_2 * 1024]
        for ax1, ax2 in T.grid(2, 4):
            conv2d_transpose_nhwc_flat[threadIdx_x // 8 * 4096 + ax1 * 2048 + blockIdx_x // 32 * 1024 + ax2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 8] = conv2d_transpose_nhwc_local[ax1 * 4 + ax2]


@tvm.script.ir_module
class After:
    @T.prim_func
    def main(inputs: T.Buffer((1, 4, 4, 512), "float32"), weight: T.Buffer((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Buffer((1, 8, 8, 256), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        inputs_flat = T.Buffer([8192], dtype="float32", data=inputs.data)
        weight_flat = T.Buffer([2097152], dtype="float32", data=weight.data)
        conv2d_transpose_nhwc_flat = T.Buffer([16384], dtype="float32", data=conv2d_transpose_nhwc.data)
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        # body
        T.launch_thread(blockIdx_x, 64)
        conv2d_transpose_nhwc_local = T.decl_buffer([8], "float32", scope="local")
        PadInput_shared = T.decl_buffer([768], "float32", scope="shared")
        weight_shared = T.decl_buffer([4096], "float32", scope="shared")
        T.launch_thread(threadIdx_x, 32)
        for i2_3_init, i1_4_init, i2_4_init in T.grid(2, 2, 2):
            conv2d_transpose_nhwc_local[i1_4_init * 4 + i2_3_init * 2 + i2_4_init] = T.float32(0)
        for i6_0 in T.serial(16):
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(24):
                PadInput_shared[ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x] = T.if_then_else(1 <= (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) // 4 and (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) // 20 < 1 and 1 <= blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) % 4 and (blockIdx_x // 32 * 2 + (ax0_ax1_ax2_ax3_fused_0 + threadIdx_x // 32) % 4) // 5 < 1, inputs_flat[blockIdx_x // 32 * 1024 + ax0_ax1_ax2_ax3_fused_0 * 512 + i6_0 * 32 + threadIdx_x - 2560], T.float32(0), dtype="float32")
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(32):
                weight_shared[T.ramp(ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4, 1, 4)] = weight_flat[T.ramp((ax0_ax1_ax2_ax3_fused_0 + threadIdx_x * 4 // 128) // 2 * 131072 + i6_0 * 8192 + (ax0_ax1_ax2_ax3_fused_0 * 16 + threadIdx_x * 4 // 8) % 32 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 2 * 4, 1, 4)]
            for i6_1, i2_3, i4_2, i5_2, i6_2, i1_4, i2_4 in T.grid(4, 2, 4, 4, 8, 2, 2):
                conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] = conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] + T.if_then_else((i1_4 + i4_2) % 2 == 0 and (i2_4 + i5_2) % 2 == 0, PadInput_shared[threadIdx_x // 8 * 128 + (i1_4 + i4_2) // 2 * 128 + (i2_4 + i5_2) // 2 * 32 + i2_3 * 32 + i6_1 * 8 + i6_2], T.float32(0), dtype="float32") * weight_shared[i6_1 * 64 + i6_2 * 8 + threadIdx_x % 8 + 3840 - i5_2 * 256 - i4_2 * 1024]
        for ax1, ax2 in T.grid(2, 4):
            conv2d_transpose_nhwc_flat[threadIdx_x // 8 * 4096 + ax1 * 2048 + blockIdx_x // 32 * 1024 + ax2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 8] = conv2d_transpose_nhwc_local[ax1 * 4 + ax2]


@tvm.script.ir_module
class After_simplified:
    @T.prim_func
    def main(inputs: T.Buffer((1, 4, 4, 512), "float32"), weight: T.Buffer((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Buffer((1, 8, 8, 256), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        inputs_flat = T.Buffer([8192], dtype="float32", data=inputs.data)
        weight_flat = T.Buffer([2097152], dtype="float32", data=weight.data)
        conv2d_transpose_nhwc_flat = T.Buffer([16384], dtype="float32", data=conv2d_transpose_nhwc.data)
        # body
        T.launch_thread(blockIdx_x, 64)
        conv2d_transpose_nhwc_local = T.decl_buffer([8], "float32", scope="local")
        PadInput_shared = T.decl_buffer([768], "float32", scope="shared")
        weight_shared = T.decl_buffer([4096], "float32", scope="shared")
        T.launch_thread(threadIdx_x, 32)
        for i2_3_init, i1_4_init, i2_4_init in T.grid(2, 2, 2):
            conv2d_transpose_nhwc_local[i1_4_init * 4 + i2_3_init * 2 + i2_4_init] = T.float32(0)
        for i6_0 in T.serial(16):
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(24):
                PadInput_shared[ax0_ax1_ax2_ax3_fused_0 * 32 + threadIdx_x] = T.if_then_else(4 <= ax0_ax1_ax2_ax3_fused_0 and ax0_ax1_ax2_ax3_fused_0 < 20 and 1 <= blockIdx_x // 32 * 2 + ax0_ax1_ax2_ax3_fused_0 % 4 and blockIdx_x // 32 * 2 + ax0_ax1_ax2_ax3_fused_0 % 4 < 5, inputs_flat[blockIdx_x // 32 * 1024 + ax0_ax1_ax2_ax3_fused_0 * 512 + i6_0 * 32 + threadIdx_x - 2560], T.float32(0), dtype="float32")
            for ax0_ax1_ax2_ax3_fused_0 in T.serial(32):
                weight_shared[T.ramp(ax0_ax1_ax2_ax3_fused_0 * 128 + threadIdx_x * 4, 1, 4)] = weight_flat[T.ramp(ax0_ax1_ax2_ax3_fused_0 // 2 * 131072 + i6_0 * 8192 + ax0_ax1_ax2_ax3_fused_0 % 2 * 4096 + threadIdx_x // 2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 2 * 4, 1, 4)]
            for i6_1, i2_3, i4_2, i5_2, i6_2, i1_4, i2_4 in T.grid(4, 2, 4, 4, 8, 2, 2):
                conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] = conv2d_transpose_nhwc_local[i1_4 * 4 + i2_3 * 2 + i2_4] + T.if_then_else((i1_4 + i4_2) % 2 == 0 and (i2_4 + i5_2) % 2 == 0, PadInput_shared[threadIdx_x // 8 * 128 + (i1_4 + i4_2) // 2 * 128 + (i2_4 + i5_2) // 2 * 32 + i2_3 * 32 + i6_1 * 8 + i6_2], T.float32(0), dtype="float32") * weight_shared[i6_1 * 64 + i6_2 * 8 + threadIdx_x % 8 + 3840 - i5_2 * 256 - i4_2 * 1024]
        for ax1, ax2 in T.grid(2, 4):
            conv2d_transpose_nhwc_flat[threadIdx_x // 8 * 4096 + ax1 * 2048 + blockIdx_x // 32 * 1024 + ax2 * 256 + blockIdx_x % 32 * 8 + threadIdx_x % 8] = conv2d_transpose_nhwc_local[ax1 * 4 + ax2]

# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg
# fmt: on


def test_renormalize_split_pattern():
    after = tvm.tir.transform.RenormalizeSplitPattern()(Before)
    tvm.ir.assert_structural_equal(after, After)
    after = tvm.tir.transform.Simplify()(after)
    tvm.ir.assert_structural_equal(after, After_simplified)


@T.prim_func
def impossible_equality(n: T.int32):
    # Prior to bugfix, this conditional defined the expression "2" as
    # equal to zero within the then_case. [min_value=2, max_value=0]
    if 2 == 0:
        # Then this expression evaluates n/2, using the min/max values
        # of "2", which is caught as a divide by zero error.
        if n // 2 >= 16:
            T.evaluate(0)


@T.prim_func
def impossible_inequality(n: T.int32):
    # Prior to bugfix, this conditional set up a range of possible
    # values for the expression "-2" as [0, kPosInf].
    if -1 < -2:
        if n // (-2) >= 16:
            T.evaluate(0)


integer_condition = tvm.testing.parameter(
    impossible_equality,
    impossible_inequality,
)


def test_analyze_inside_integer_conditional(integer_condition):
    """Avoid crash occurring in ConstIntBoundAnalyzer.

    Crash occurred when simplifying some expressions with provably
    false integer expressions.  If the expressions were renormalized
    before calling Simplify, conditional statements could assign a
    range of possible values to integers, as if they were variables.
    This would result in divide by zero throwing an exception,
    followed by a second exception during stack unwinding causing the
    program to crash.
    """

    # Similar issue would occur in most transformations that subclass
    # IRMutatorWithAnalyzer.  tir.transform.Simplify() is an
    # exception, as it rewrites the integer conditionals first.  These
    # tests are written using RenormalizeSplitPattern as it is the
    # first case identified.
    transform = tvm.tir.transform.RenormalizeSplitPattern()

    # Issue would result in an error through while applying the transformation.
    mod = tvm.IRModule.from_expr(integer_condition)
    transform(mod)


if __name__ == "__main__":
    tvm.testing.main()
