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
import tvm.testing
from tvm.meta_schedule.postproc import RewriteParallelVectorizeUnroll
from tvm.script import tir as T
from tvm.tir.schedule import Schedule
from tvm.tir.schedule.testing import assert_structural_equal_ignore_global_symbol

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant
# fmt: off

@tvm.script.ir_module
class Move_PUV:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, [1024, 1024, 1024], dtype="float32")
        B = T.match_buffer(b, [1024, 1024, 1024], dtype="float32")
        # body
        with T.block("root"):
            T.block_attr({"meta_schedule.parallel":128, "meta_schedule.vectorize":32})
            for i0, j0, i1, j1, k0, i2, j2, k1 in T.grid(128, 64, 4, 4, 64, 4, 8, 32):
                with T.block("move"):
                    vi = T.axis.spatial(1024, i0 * 16 + i1 * 4 + i2)
                    vj = T.axis.spatial(1024, j0 * 32 + j1 * 8 + j2)
                    vk = T.axis.spatial(1024, k0 * 32 + k1)
                    T.where((i0 * 4 + i1) * 4 + i2 < 1024 and (j0 * 4 + j1) * 8 + j2 < 1024 and k0 * 32 + k1 < 1024)
                    T.reads([A[vi, vj, vk]])
                    T.writes([B[vi, vj, vk]])
                    B[vi, vj, vk] = A[vi, vj, vk]


@T.prim_func
def Move_PUV0(a: T.handle, b: T.handle) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main"})
    A = T.match_buffer(a, [1024, 1024, 1024], dtype="float32")
    B = T.match_buffer(b, [1024, 1024, 1024], dtype="float32")
    # body
    with T.block("root"):
        for i0_j0_fused in T.parallel(0, 8192):
            for i1, j1, k0, i2, j2 in T.grid(4, 4, 64, 4, 8):
                for k1_fused in T.vectorized(0, 32):
                    with T.block("move"):
                        vi = T.axis.spatial(1024, i0_j0_fused // 64 * 16 + i1 * 4 + i2)
                        vj = T.axis.spatial(1024, i0_j0_fused % 64 * 32 + j1 * 8 + j2)
                        vk = T.axis.spatial(1024, k0 * 32 + k1_fused)
                        T.where(
                            i0_j0_fused < 4064
                            and i0_j0_fused % 64 < 32
                            and k0 < 32
                        )
                        T.reads([A[vi, vj, vk]])
                        T.writes([B[vi, vj, vk]])
                        B[vi, vj, vk] = A[vi, vj, vk]


@tvm.script.ir_module
class Fused_NN_Dense:
    @T.prim_func
    def main(placeholder: T.Buffer((64, 768), "float32"), placeholder_1: T.Buffer((768, 768), "float32"), T_matmul_NT: T.Buffer((64, 768), "float32")) -> None:
        for i0, i1, i2 in T.grid(64, 768, 768):
            with T.block("T_matmul_NT"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(placeholder[i, k], placeholder_1[j, k])
                T.writes(T_matmul_NT[i, j])
                with T.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + placeholder[i, k] * placeholder_1[j, k]

@T.prim_func
def before_matmul_vectorize(
    placeholder: T.Buffer((64, 768), "float32"),
    placeholder_1: T.Buffer((768, 768), "float32"),
    T_matmul_NT: T.Buffer((64, 768), "float32"),
) -> None:
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"meta_schedule.vectorize":64})
        T_matmul_NT_global = T.alloc_buffer([64, 768], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1 in T.grid(1, 16, 1, 3):
            for i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(48, 8, 1, 16, 8, 16):
                with T.block("T_matmul_NT"):
                    i = T.axis.spatial(64, i0_2 * 8 + i0_3)
                    j = T.axis.spatial(768, i1_0 * 48 + i1_1 * 16 + i1_3)
                    k = T.axis.reduce(768, i2_0 * 16 + i2_1)
                    T.reads(placeholder[i, k], placeholder_1[j, k])
                    T.writes(T_matmul_NT_global[i, j])
                    with T.init():
                        T_matmul_NT_global[i, j] = T.float32(0)
                    T_matmul_NT_global[i, j] = T_matmul_NT_global[i, j] + placeholder[i, k] * placeholder_1[j, k]
            for ax0, ax1 in T.grid(64, 16):
                with T.block("T_matmul_NT_global"):
                    v0 = T.axis.spatial(64, ax0)
                    v1 = T.axis.spatial(768, i1_0 * 48 + i1_1 * 16 + ax1)
                    T.reads(T_matmul_NT_global[v0, v1])
                    T.writes(T_matmul_NT[v0, v1])
                    T_matmul_NT[v0, v1] = T_matmul_NT_global[v0, v1]

@T.prim_func
def after_matmul_vectorize(
    placeholder: T.Buffer((64, 768), "float32"),
    placeholder_1: T.Buffer((768, 768), "float32"),
    T_matmul_NT: T.Buffer((64, 768), "float32"),
) -> None:
    T_matmul_NT_global = T.alloc_buffer([64, 768], dtype="float32")
    for i0_0, i1_0, i0_1, i1_1 in T.grid(1, 16, 1, 3):
        for i2_0, i0_2, i1_2, i2_1, i0_3 in T.grid(48, 8, 1, 16, 8):
            for i1_3_fused in T.vectorized(16):
                with T.block("T_matmul_NT"):
                    i = T.axis.spatial(64, i0_2 * 8 + i0_3)
                    j = T.axis.spatial(768, i1_0 * 48 + i1_1 * 16 + i1_3_fused)
                    k = T.axis.reduce(768, i2_0 * 16 + i2_1)
                    T.reads(placeholder[i, k], placeholder_1[j, k])
                    T.writes(T_matmul_NT_global[i, j])
                    with T.init():
                        T_matmul_NT_global[i, j] = T.float32(0)
                    T_matmul_NT_global[i, j] = T_matmul_NT_global[i, j] + placeholder[i, k] * placeholder_1[j, k]
        for ax0 in T.serial(64):
            for ax1_fused in T.vectorized(16):
                with T.block("T_matmul_NT_global"):
                    v0 = T.axis.spatial(64, ax0)
                    v1 = T.axis.spatial(768, i1_0 * 48 + i1_1 * 16 + ax1_fused)
                    T.reads(T_matmul_NT_global[v0, v1])
                    T.writes(T_matmul_NT[v0, v1])
                    T_matmul_NT[v0, v1] = T_matmul_NT_global[v0, v1]


@T.prim_func
def before_postproc_add(
    lhs: T.Buffer((1, 8, 56, 56, 32), "uint8"),
    rhs: T.Buffer((1, 8, 56, 56, 32), "uint8"),
    add_compute: T.Buffer((1, 8, 56, 56, 32), "uint8"),
) -> None:
    with T.block("root"):
        T.block_attr({"meta_schedule.parallel":64, "meta_schedule.vectorize":128})
        for n, c0, h, w, c1 in T.grid(1, 8, 56, 56, 32):
            with T.block("add_compute"):
                v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [n, c0, h, w, c1])
                T.reads(lhs[v0, v1, v2, v3, v4], rhs[v0, v1, v2, v3, v4])
                T.writes(add_compute[v0, v1, v2, v3, v4])
                add_compute[v0, v1, v2, v3, v4] = lhs[v0, v1, v2, v3, v4] + rhs[v0, v1, v2, v3, v4]


@T.prim_func
def after_postproc_add(
    lhs: T.Buffer((1, 8, 56, 56, 32), "uint8"),
    rhs: T.Buffer((1, 8, 56, 56, 32), "uint8"),
    add_compute: T.Buffer((1, 8, 56, 56, 32), "uint8"),
) -> None:
    with T.block("root"):
        for n_c0_h_w_c1_fused_0 in T.parallel(0, 6272):
            for n_c0_h_w_c1_fused_1 in T.vectorized(0, 128):
                with T.block("add_compute"):
                    v0 = T.axis.spatial(1, 0)
                    v1 = T.axis.spatial(8, (n_c0_h_w_c1_fused_0 * 128 + n_c0_h_w_c1_fused_1) // 100352)
                    v2 = T.axis.spatial(56, (n_c0_h_w_c1_fused_0 * 128 + n_c0_h_w_c1_fused_1) % 100352 // 1792)
                    v3 = T.axis.spatial(56, (n_c0_h_w_c1_fused_0 * 128 + n_c0_h_w_c1_fused_1) % 1792 // 32)
                    v4 = T.axis.spatial(32, (n_c0_h_w_c1_fused_0 * 128 + n_c0_h_w_c1_fused_1) % 32)
                    T.reads(lhs[v0, v1, v2, v3, v4], rhs[v0, v1, v2, v3, v4])
                    T.writes(add_compute[v0, v1, v2, v3, v4])
                    add_compute[v0, v1, v2, v3, v4] = lhs[v0, v1, v2, v3, v4] + rhs[v0, v1, v2, v3, v4]


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable


def test_meta_schedule_postproc_rewrite_parallel_unroll_vectorize():
    postproc = RewriteParallelVectorizeUnroll()
    sch = Schedule(Move_PUV)
    assert postproc.apply(sch)
    mod = tvm.tir.transform.Simplify()(sch.mod)
    tvm.ir.assert_structural_equal(mod["main"], Move_PUV0)


def test_vectorize_inner_loop():
    sch = Schedule(before_matmul_vectorize)
    rule = RewriteParallelVectorizeUnroll()
    assert rule.apply(sch)
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], after_matmul_vectorize)


def test_parallel_vectorize_add():
    sch = Schedule(before_postproc_add)
    rule = RewriteParallelVectorizeUnroll()
    assert rule.apply(sch)
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], after_postproc_add)


def test_no_unroll_for_spatial_block():
    # fmt: off
    @T.prim_func
    def layer_norm(A: T.Buffer((1, 4, 4, 32), "float32"), B: T.Buffer((4, 4, 32), "float32"), C: T.Buffer((4, 4, 32), "float32"), T_layer_norm: T.Buffer((1, 4, 4, 32), "float32")):
        with T.block("root"):
            T.block_attr({"meta_schedule.unroll_explicit": 512})
            A_red_temp_v0 = T.alloc_buffer((1,))
            A_red_temp_v1 = T.alloc_buffer((1,))
            for ax0, k1, k2, k3 in T.grid(1, 4, 4, 32):
                with T.block("A_red_temp"):
                    v_ax0, v_k1, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k1, k2, k3])
                    T.reads(A[v_ax0, v_k1, v_k2, v_k3])
                    T.writes(A_red_temp_v0[v_ax0], A_red_temp_v1[v_ax0])
                    with T.init():
                        A_red_temp_v0[v_ax0] = T.float32(0)
                        A_red_temp_v1[v_ax0] = T.float32(0)
                    v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0] + A[v_ax0, v_k1, v_k2, v_k3]
                    v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0] + A[v_ax0, v_k1, v_k2, v_k3] * A[v_ax0, v_k1, v_k2, v_k3]
                    A_red_temp_v0[v_ax0] = v_A_red_temp_v0
                    A_red_temp_v1[v_ax0] = v_A_red_temp_v1
            for ax0, ax1, ax2, ax3 in T.grid(1, 4, 4, 32):
                with T.block("T_layer_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], A_red_temp_v0[v_ax0], A_red_temp_v1[v_ax0], B[v_ax1, v_ax2, v_ax3], C[v_ax1, v_ax2, v_ax3])
                    T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3] = (A[v_ax0, v_ax1, v_ax2, v_ax3] - A_red_temp_v0[v_ax0] * T.float32(0.001953125)) * T.rsqrt(A_red_temp_v1[v_ax0] * T.float32(0.001953125) - A_red_temp_v0[v_ax0] * T.float32(0.001953125) * (A_red_temp_v0[v_ax0] * T.float32(0.001953125)) + T.float32(1.0000000000000001e-05)) * B[v_ax1, v_ax2, v_ax3] + C[v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def expected(A: T.Buffer((1, 4, 4, 32), "float32"), B: T.Buffer((4, 4, 32), "float32"), C: T.Buffer((4, 4, 32), "float32"), T_layer_norm: T.Buffer((1, 4, 4, 32), "float32")):
        with T.block("root"):
            A_red_temp_v0 = T.alloc_buffer((1,))
            A_red_temp_v1 = T.alloc_buffer((1,))
            for ax0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
                for k1, k2, k3 in T.grid(4, 4, 32):
                    with T.block("A_red_temp"):
                        v_ax0 = T.axis.spatial(1, 0)
                        v_k1, v_k2, v_k3 = T.axis.remap("RRR", [k1, k2, k3])
                        T.reads(A[0, v_k1, v_k2, v_k3])
                        T.writes(A_red_temp_v0[0], A_red_temp_v1[0])
                        with T.init():
                            A_red_temp_v0[0] = T.float32(0)
                            A_red_temp_v1[0] = T.float32(0)
                        v_A_red_temp_v0: T.float32 = A_red_temp_v0[0] + A[0, v_k1, v_k2, v_k3]
                        v_A_red_temp_v1: T.float32 = A_red_temp_v1[0] + A[0, v_k1, v_k2, v_k3] * A[0, v_k1, v_k2, v_k3]
                        A_red_temp_v0[0] = v_A_red_temp_v0
                        A_red_temp_v1[0] = v_A_red_temp_v1
            for ax0, ax1, ax2, ax3 in T.grid(1, 4, 4, 32):
                with T.block("T_layer_norm"):
                    v_ax0 = T.axis.spatial(1, 0)
                    v_ax1, v_ax2, v_ax3 = T.axis.remap("SSS", [ax1, ax2, ax3])
                    T.reads(A[0, v_ax1, v_ax2, v_ax3], A_red_temp_v0[0], A_red_temp_v1[0], B[v_ax1, v_ax2, v_ax3], C[v_ax1, v_ax2, v_ax3])
                    T.writes(T_layer_norm[0, v_ax1, v_ax2, v_ax3])
                    T_layer_norm[0, v_ax1, v_ax2, v_ax3] = (A[0, v_ax1, v_ax2, v_ax3] - A_red_temp_v0[0] * T.float32(0.001953125)) * T.rsqrt(A_red_temp_v1[0] * T.float32(0.001953125) - A_red_temp_v0[0] * T.float32(0.001953125) * (A_red_temp_v0[0] * T.float32(0.001953125)) + T.float32(1.0000000000000001e-05)) * B[v_ax1, v_ax2, v_ax3] + C[v_ax1, v_ax2, v_ax3]
    # fmt: on

    postproc = RewriteParallelVectorizeUnroll()
    sch = Schedule(layer_norm)
    assert postproc.apply(sch)
    mod = tvm.tir.transform.Simplify()(sch.mod)
    assert_structural_equal_ignore_global_symbol(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
