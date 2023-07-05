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
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.tir.tensor_intrin import arm_cpu, cuda, rocm, x86


@tvm.script.ir_module
class Conv2dNCHWcVNNIModuleTiled:
    @T.prim_func
    def main(
        placeholder: T.Buffer((1, 4, 56, 56, 16), "uint8"),
        placeholder_1: T.Buffer((16, 4, 1, 1, 4, 16, 4), "int8"),
        conv2d_NCHWc_int8: T.Buffer((1, 16, 56, 56, 16), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for (
            i0_0,
            i1_0,
            i2_0,
            i3_0,
            i4_0_0,
            i0_1,
            i1_1,
            i2_1,
            i3_1,
            i4_0_1,
            i5_0,
            i6_0,
            i7_0,
            i8_0,
            i9_0_0,
            i0_2,
            i1_2,
            i2_2,
            i3_2,
            i4_0_2,
            i5_1,
            i6_1,
            i7_1,
            i8_1,
            i9_0_1,
            i0_3,
            i1_3,
            i2_3,
            i3_3,
            i4_0_3,
        ) in T.grid(
            1,
            1,
            2,
            1,
            1,
            1,
            4,
            1,
            14,
            1,
            1,
            1,
            4,
            1,
            1,
            1,
            4,
            7,
            1,
            1,
            1,
            1,
            1,
            4,
            1,
            1,
            1,
            4,
            4,
            1,
        ):
            with T.block("conv2d_NCHWc_int8_o"):
                n = T.axis.spatial(1, 0)
                oc_chunk = T.axis.spatial(16, i1_1 * 4 + i1_2)
                oh = T.axis.spatial(56, i2_0 * 28 + i2_2 * 4 + i2_3)
                ow = T.axis.spatial(56, i3_1 * 4 + i3_3)
                oc_block_o = T.axis.spatial(1, 0)
                kh = T.axis.reduce(1, 0)
                kw = T.axis.reduce(1, 0)
                ic_outer, ic_f_inner = T.axis.remap("RR", [i7_0, i8_1])
                ic_s_inner_o = T.axis.reduce(1, 0)
                T.reads(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4],
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4],
                )
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16])
                T.block_attr({"meta_schedule.auto_tensorize": "dot_16x4_vnni"})
                with T.init():
                    for i4_1 in T.serial(16):
                        with T.block("conv2d_NCHWc_int8_init"):
                            oc_block_init = T.axis.spatial(16, i4_1)
                            T.reads()
                            T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init])
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init] = 0
                for i4_1, i9_1 in T.grid(16, 4):
                    with T.block("conv2d_NCHWc_int8"):
                        oc_block, ic_s_inner = T.axis.remap("SR", [i4_1, i9_1])
                        T.reads(
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block],
                            placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                            placeholder_1[
                                oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner
                            ],
                        )
                        T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[
                            n, oc_chunk, oh, ow, oc_block
                        ] + T.cast(
                            placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                            "int32",
                        ) * T.cast(
                            placeholder_1[
                                oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner
                            ],
                            "int32",
                        )


@tvm.script.ir_module
class Conv2dNCHWcVNNIModuleTensorized:
    @T.prim_func
    def main(
        placeholder: T.Buffer((1, 4, 56, 56, 16), "uint8"),
        placeholder_1: T.Buffer((16, 4, 1, 1, 4, 16, 4), "int8"),
        conv2d_NCHWc_int8: T.Buffer((1, 16, 56, 56, 16), "int32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_0, i1_0, i2_0, i3_0, i4_0_0, i0_1, i1_1, i2_1, i3_1, i4_0_1, i5_0, i6_0 in T.grid(
            1, 1, 2, 1, 1, 1, 4, 1, 14, 1, 1, 1
        ):
            for i1_2_init, i2_2_init, i2_3_init, i3_3_init in T.grid(4, 7, 4, 4):
                with T.block("conv2d_NCHWc_int8_o_init"):
                    n = T.axis.spatial(1, 0)
                    oc_chunk = T.axis.spatial(16, i1_1 * 4 + i1_2_init)
                    oh = T.axis.spatial(56, i2_0 * 28 + i2_2_init * 4 + i2_3_init)
                    ow = T.axis.spatial(56, i3_1 * 4 + i3_3_init)
                    oc_block_o = T.axis.spatial(1, 0)
                    T.reads()
                    T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16])
                    for i4_1 in T.vectorized(16):
                        with T.block("conv2d_NCHWc_int8_init"):
                            oc_block_init = T.axis.spatial(16, i4_1)
                            T.reads()
                            T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init])
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init] = 0
            for (
                i7_0,
                i8_0,
                i9_0_0,
                i0_2,
                i1_2,
                i2_2,
                i3_2,
                i4_0_2,
                i5_1,
                i6_1,
                i7_1,
                i8_1,
                i9_0_1,
                i0_3,
                i1_3,
                i2_3,
                i3_3,
                i4_0_3,
            ) in T.grid(4, 1, 1, 1, 4, 7, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 4, 1):
                with T.block("conv2d_NCHWc_int8_o_update"):
                    n = T.axis.spatial(1, 0)
                    oc_chunk = T.axis.spatial(16, i1_1 * 4 + i1_2)
                    oh = T.axis.spatial(56, i2_0 * 28 + i2_2 * 4 + i2_3)
                    ow = T.axis.spatial(56, i3_1 * 4 + i3_3)
                    oc_block_o = T.axis.spatial(1, 0)
                    kh = T.axis.reduce(1, 0)
                    kw = T.axis.reduce(1, 0)
                    ic_outer, ic_f_inner = T.axis.remap("RR", [i7_0, i8_1])
                    ic_s_inner_o = T.axis.reduce(1, 0)
                    T.reads(
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16],
                        placeholder[
                            n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4
                        ],
                        placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4],
                    )
                    T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16])
                    A = T.match_buffer(
                        placeholder[
                            n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4
                        ],
                        [4],
                        dtype="uint8",
                        offset_factor=1,
                    )
                    B = T.match_buffer(
                        placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4],
                        [16, 4],
                        dtype="int8",
                        offset_factor=1,
                    )
                    C = T.match_buffer(
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16],
                        [16],
                        dtype="int32",
                        offset_factor=1,
                    )
                    A_u8x4 = A.vload([0], "uint8x4")
                    A_i32 = T.reinterpret(A_u8x4, dtype="int32")
                    B_i8x64 = B.vload([0, 0], dtype="int8x64")
                    B_i32x16 = T.reinterpret(B_i8x64, dtype="int32x16")
                    C_i32x16 = C.vload([0], dtype="int32x16")
                    C[T.ramp(0, 1, 16)] = T.call_llvm_pure_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512"),
                        T.uint32(3),
                        C_i32x16,
                        T.broadcast(A_i32, 16),
                        B_i32x16,
                        dtype="int32x16",
                    )


@tvm.script.ir_module
class DenseDP4ATiled:
    @T.prim_func
    def main(
        X: T.Buffer((128, 128), "int8"),
        W: T.Buffer((128, 128), "int8"),
        compute: T.Buffer((128, 128), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        compute_local = T.alloc_buffer([128, 128], dtype="int32", scope="local")
        X_shared = T.alloc_buffer([128, 128], dtype="int8", scope="shared")
        W_shared = T.alloc_buffer([128, 128], dtype="int8", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(2, thread="threadIdx.x"):
                    for i2_0_0 in T.serial(2):
                        for ax0_ax1_fused in T.serial(1024):
                            with T.block("X_shared"):
                                v0 = T.axis.spatial(
                                    128, i0_0_i1_0_fused // 2 * 16 + ax0_ax1_fused // 64
                                )
                                v1 = T.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(X[v0, v1])
                                T.writes(X_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                X_shared[v0, v1] = X[v0, v1]
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("W_shared"):
                                v0 = T.axis.spatial(
                                    128, i0_0_i1_0_fused % 2 * 64 + ax0_ax1_fused // 64
                                )
                                v1 = T.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(W[v0, v1])
                                T.writes(W_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                W_shared[v0, v1] = W[v0, v1]
                        for i2_0_1, i0_3, i1_3, i2_0_2, i0_4, i1_4 in T.grid(2, 4, 16, 8, 4, 1):
                            with T.block("compute_o"):
                                i = T.axis.spatial(128, i0_0_i1_0_fused // 2 * 16 + i0_3 * 4 + i0_4)
                                j = T.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused % 2 * 64
                                    + i0_1_i1_1_fused * 32
                                    + i0_2_i1_2_fused * 16
                                    + i1_3,
                                )
                                k_o = T.axis.reduce(32, i2_0_0 * 16 + i2_0_1 * 8 + i2_0_2)
                                T.reads(
                                    X_shared[i, k_o * 4 : k_o * 4 + 4],
                                    W_shared[j, k_o * 4 : k_o * 4 + 4],
                                )
                                T.writes(compute_local[i, j])
                                T.block_attr({"meta_schedule.auto_tensorize": "dp4a"})
                                with T.init():
                                    with T.block("compute_init"):
                                        T.reads()
                                        T.writes(compute_local[i, j])
                                        compute_local[i, j] = 0
                                for i2_1 in T.serial(4):
                                    with T.block("compute"):
                                        k = T.axis.reduce(4, i2_1)
                                        T.reads(
                                            compute_local[i, j],
                                            X_shared[i, k_o * 4 + k],
                                            W_shared[j, k_o * 4 + k],
                                        )
                                        T.writes(compute_local[i, j])
                                        T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                        compute_local[i, j] = compute_local[i, j] + T.cast(
                                            X_shared[i, k_o * 4 + k], "int32"
                                        ) * T.cast(W_shared[j, k_o * 4 + k], "int32")
                    for ax0, ax1 in T.grid(16, 16):
                        with T.block("compute_local"):
                            v0 = T.axis.spatial(128, i0_0_i1_0_fused // 2 * 16 + ax0)
                            v1 = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 2 * 64
                                + i0_1_i1_1_fused * 32
                                + i0_2_i1_2_fused * 16
                                + ax1,
                            )
                            T.reads(compute_local[v0, v1])
                            T.writes(compute[v0, v1])
                            compute[v0, v1] = compute_local[v0, v1]


@tvm.script.ir_module
class DenseDP4ATensorized:
    @T.prim_func
    def main(
        X: T.Buffer((128, 128), "int8"),
        W: T.Buffer((128, 128), "int8"),
        compute: T.Buffer((128, 128), "int32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compute_local = T.alloc_buffer([128, 128], dtype="int32", scope="local")
        X_shared = T.alloc_buffer([128, 128], dtype="int8", scope="shared")
        W_shared = T.alloc_buffer([128, 128], dtype="int8", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(2, thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i0_4_init in T.grid(4, 16, 4):
                        with T.block("compute_o_init"):
                            i = T.axis.spatial(
                                128, i0_0_i1_0_fused // 2 * 16 + i0_3_init * 4 + i0_4_init
                            )
                            j = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 2 * 64
                                + i0_1_i1_1_fused * 32
                                + i0_2_i1_2_fused * 16
                                + i1_3_init,
                            )
                            T.reads()
                            T.writes(compute_local[i, j])
                            T.block_attr({"meta_schedule.auto_tensorize": ""})
                            with T.block("compute_init"):
                                T.reads()
                                T.writes(compute_local[i, j])
                                compute_local[i, j] = 0
                    for i2_0_0 in T.serial(2):
                        for ax0_ax1_fused in T.serial(1024):
                            with T.block("X_shared"):
                                v0 = T.axis.spatial(
                                    128, i0_0_i1_0_fused // 2 * 16 + ax0_ax1_fused // 64
                                )
                                v1 = T.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(X[v0, v1])
                                T.writes(X_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                X_shared[v0, v1] = X[v0, v1]
                        for ax0_ax1_fused in T.serial(4096):
                            with T.block("W_shared"):
                                v0 = T.axis.spatial(
                                    128, i0_0_i1_0_fused % 2 * 64 + ax0_ax1_fused // 64
                                )
                                v1 = T.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                T.reads(W[v0, v1])
                                T.writes(W_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                W_shared[v0, v1] = W[v0, v1]
                        for i2_0_1, i0_3, i1_3, i2_0_2, i0_4, i1_4 in T.grid(2, 4, 16, 8, 4, 1):
                            with T.block("compute_o_update"):
                                i = T.axis.spatial(128, i0_0_i1_0_fused // 2 * 16 + i0_3 * 4 + i0_4)
                                j = T.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused % 2 * 64
                                    + i0_1_i1_1_fused * 32
                                    + i0_2_i1_2_fused * 16
                                    + i1_3,
                                )
                                k_o = T.axis.reduce(32, i2_0_0 * 16 + i2_0_1 * 8 + i2_0_2)
                                T.reads(
                                    compute_local[i, j],
                                    X_shared[i, k_o * 4 : k_o * 4 + 4],
                                    W_shared[j, k_o * 4 : k_o * 4 + 4],
                                )
                                T.writes(compute_local[i, j])
                                A = T.match_buffer(
                                    X_shared[i, k_o * 4 : k_o * 4 + 4],
                                    [4],
                                    dtype="int8",
                                    scope="shared",
                                    align=4,
                                    offset_factor=1,
                                )
                                B = T.match_buffer(
                                    W_shared[j, k_o * 4 : k_o * 4 + 4],
                                    [4],
                                    dtype="int8",
                                    scope="shared",
                                    align=4,
                                    offset_factor=1,
                                )
                                C = T.match_buffer(
                                    compute_local[i, j],
                                    [1],
                                    dtype="int32",
                                    scope="local",
                                    align=4,
                                    offset_factor=1,
                                )
                                C[0] = C[0] + T.call_pure_extern(
                                    "__dp4a",
                                    A[T.ramp(0, 1, 4)],
                                    B[T.ramp(0, 1, 4)],
                                    0,
                                    dtype="int32",
                                )
                    for ax0, ax1 in T.grid(16, 16):
                        with T.block("compute_local"):
                            v0 = T.axis.spatial(128, i0_0_i1_0_fused // 2 * 16 + ax0)
                            v1 = T.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 2 * 64
                                + i0_1_i1_1_fused * 32
                                + i0_2_i1_2_fused * 16
                                + ax1,
                            )
                            T.reads(compute_local[v0, v1])
                            T.writes(compute[v0, v1])
                            compute[v0, v1] = compute_local[v0, v1]


def _create_context(mod, target, postprocs) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=postprocs,
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


def test_rewrite_tensorize_conv2d_nchwc_vnni():
    mod = Conv2dNCHWcVNNIModuleTiled
    target = tvm.target.Target("llvm -mcpu=cascadelake -num-cores 4")
    ctx = _create_context(
        mod,
        target,
        [
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(True),
        ],
    )
    sch = tvm.tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()

    for proc in ctx.space_generator.postprocs:
        proc.apply(sch)

    tvm.ir.assert_structural_equal(sch.mod, Conv2dNCHWcVNNIModuleTensorized)


def test_rewrite_tensorize_dense_dp4a():
    mod = DenseDP4ATiled
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    ctx = _create_context(
        mod,
        target,
        [
            ms.postproc.RewriteCooperativeFetch(),
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(),
        ],
    )
    sch = tvm.tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()

    for proc in ctx.space_generator.postprocs:
        proc.apply(sch)

    tvm.ir.assert_structural_equal(sch.mod, DenseDP4ATensorized)


if __name__ == "__main__":
    test_rewrite_tensorize_conv2d_nchwc_vnni()
    test_rewrite_tensorize_dense_dp4a()
