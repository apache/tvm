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
from tvm import meta_schedule as ms
from tvm import te
from tvm.ir import assert_structural_equal
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.tensor_intrin.arm_cpu import DP4A_INTRIN
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN


def test_vnni_conv2d_nchwc():
    @T.prim_func
    def conv2d_nchwc(
        placeholder: T.Buffer[(1, 4, 56, 56, 16), "uint8"],
        placeholder_1: T.Buffer[(16, 4, 1, 1, 4, 16, 4), "int8"],
        conv2d_NCHWc_int8: T.Buffer[(1, 16, 56, 56, 16), "int32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 16, 56, 56, 16, 1, 1, 4, 4, 4):
            with T.block("conv2d_NCHWc_int8"):
                (
                    n,
                    oc_chunk,
                    oh,
                    ow,
                    oc_block,
                    kh,
                    kw,
                    ic_outer,
                    ic_f_inner,
                    ic_s_inner,
                ) = T.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                T.reads(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                )
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                with T.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[
                    n, oc_chunk, oh, ow, oc_block
                ] + T.cast(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], "int32"
                ) * T.cast(
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                    "int32",
                )

    # fmt: off
    @T.prim_func
    def vnni_conv2d_nchwc_0(placeholder: T.Buffer[(1, 4, 56, 56, 16), "uint8"], placeholder_1: T.Buffer[(16, 4, 1, 1, 4, 16, 4), "int8"], conv2d_NCHWc_int8: T.Buffer[(1, 16, 56, 56, 16), "int32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        conv2d_NCHWc_int8_global = T.alloc_buffer([1, 16, 56, 56, 16], dtype="int32")
        for i0_0, i1_0, i2_0, i3_0, i4_0_0, i0_1, i1_1, i2_1, i3_1, i4_0_1 in T.grid(1, 8, 28, 56, 1, 1, 2, 1, 1, 1):
            for i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1):
                with T.block("conv2d_NCHWc_int8_o"):
                    n = T.axis.spatial(1, 0)
                    oc_chunk = T.axis.spatial(16, i1_0 * 2 + i1_1 + i1_2 + i1_3)
                    oh = T.axis.spatial(56, i2_0 * 2 + i2_1 * 2 + i2_2 + i2_3)
                    ow = T.axis.spatial(56, i3_3 + i3_0 + i3_1 + i3_2)
                    oc_block_o = T.axis.spatial(1, 0)
                    kh = T.axis.reduce(1, 0)
                    kw = T.axis.reduce(1, 0)
                    ic_outer = T.axis.reduce(4, i7_0 * 4 + i7_1)
                    ic_f_inner = T.axis.reduce(4, i8_0 + i8_1)
                    ic_s_inner_o = T.axis.reduce(1, 0)
                    T.reads(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0 : 16, 0 : 4])
                    T.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, 0 : 16])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_16x4_vnni"})
                    with T.init():
                        for i4_1 in T.serial(16):
                            with T.block("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = T.axis.spatial(16, i4_1)
                                T.reads()
                                T.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init])
                                conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                    for i4_1, i9_1 in T.grid(16, 4):
                        with T.block("conv2d_NCHWc_int8"):
                            oc_block_i, ic_s_inner_i = T.axis.remap("SR", [i4_1, i9_1])
                            T.reads(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i], placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
                            T.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] = conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] + T.cast(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], "int32") * T.cast(placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i], "int32")
            for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 1, 2, 1, 16):
                with T.block("conv2d_NCHWc_int8_global"):
                    v0 = T.axis.spatial(1, ax0)
                    v1 = T.axis.spatial(16, i1_0 * 2 + i1_1 + ax1)
                    v2 = T.axis.spatial(56, i2_0 * 2 + ax2)
                    v3 = T.axis.spatial(56, i3_0 + ax3)
                    v4 = T.axis.spatial(16, ax4)
                    T.reads(conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4])
                    T.writes(conv2d_NCHWc_int8[v0, v1, v2, v3, v4])
                    conv2d_NCHWc_int8[v0, v1, v2, v3, v4] = conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4]

    @T.prim_func
    def vnni_conv2d_nchwc_1(placeholder: T.Buffer[(1, 4, 56, 56, 16), "uint8"], placeholder_1: T.Buffer[(16, 4, 1, 1, 4, 16, 4), "int8"], conv2d_NCHWc_int8: T.Buffer[(1, 16, 56, 56, 16), "int32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        conv2d_NCHWc_int8_global = T.alloc_buffer([1, 16, 56, 56, 16], dtype="int32")
        for i0_0, i1_0, i2_0, i3_0, i4_0_0 in T.grid(1, 8, 28, 56, 1):
            for i0_1, i1_1, i2_1, i3_1, i4_0_1, i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1):
                with T.block("conv2d_NCHWc_int8_o"):
                    n = T.axis.spatial(1, 0)
                    oc_chunk = T.axis.spatial(16, i1_0 * 2 + i1_1 + i1_2 + i1_3)
                    oh = T.axis.spatial(56, i2_0 * 2 + i2_1 * 2 + i2_2 + i2_3)
                    ow = T.axis.spatial(56, i3_3 + i3_0 + i3_1 + i3_2)
                    oc_block_o = T.axis.spatial(1, 0)
                    kh = T.axis.reduce(1, 0)
                    kw = T.axis.reduce(1, 0)
                    ic_outer = T.axis.reduce(4, i7_0 * 4 + i7_1)
                    ic_f_inner = T.axis.reduce(4, i8_0 + i8_1)
                    ic_s_inner_o = T.axis.reduce(1, 0)
                    T.reads(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0 : 16, 0 : 4])
                    T.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, 0 : 16])
                    T.block_attr({"meta_schedule.auto_tensorize":"dot_16x4_vnni"})
                    with T.init():
                        for i4_1 in T.serial(16):
                            with T.block("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = T.axis.spatial(16, i4_1)
                                T.reads()
                                T.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init])
                                conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                    for i4_1, i9_1 in T.grid(16, 4):
                        with T.block("conv2d_NCHWc_int8"):
                            oc_block_i, ic_s_inner_i = T.axis.remap("SR", [i4_1, i9_1])
                            T.reads(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i], placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
                            T.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                            conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] = conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] + T.cast(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], "int32") * T.cast(placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i], "int32")
            for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 2, 2, 1, 16):
                with T.block("conv2d_NCHWc_int8_global"):
                    v0 = T.axis.spatial(1, ax0)
                    v1 = T.axis.spatial(16, i1_0 * 2 + ax1)
                    v2 = T.axis.spatial(56, i2_0 * 2 + ax2)
                    v3 = T.axis.spatial(56, i3_0 + ax3)
                    v4 = T.axis.spatial(16, ax4)
                    T.reads(conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4])
                    T.writes(conv2d_NCHWc_int8[v0, v1, v2, v3, v4])
                    conv2d_NCHWc_int8[v0, v1, v2, v3, v4] = conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4]

    @T.prim_func
    def vnni_conv2d_nchwc_2(placeholder: T.Buffer[(1, 4, 56, 56, 16), "uint8"], placeholder_1: T.Buffer[(16, 4, 1, 1, 4, 16, 4), "int8"], conv2d_NCHWc_int8: T.Buffer[(1, 16, 56, 56, 16), "int32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0_0, i1_0, i2_0, i3_0, i4_0_0, i0_1, i1_1, i2_1, i3_1, i4_0_1, i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 8, 28, 56, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1):
            with T.block("conv2d_NCHWc_int8_o"):
                n = T.axis.spatial(1, 0)
                oc_chunk = T.axis.spatial(16, i1_0 * 2 + i1_1 + i1_2 + i1_3)
                oh = T.axis.spatial(56, i2_0 * 2 + i2_1 * 2 + i2_2 + i2_3)
                ow = T.axis.spatial(56, i3_3 + i3_0 + i3_1 + i3_2)
                oc_block_o = T.axis.spatial(1, 0)
                kh = T.axis.reduce(1, 0)
                kw = T.axis.reduce(1, 0)
                ic_outer = T.axis.reduce(4, i7_0 * 4 + i7_1)
                ic_f_inner = T.axis.reduce(4, i8_0 + i8_1)
                ic_s_inner_o = T.axis.reduce(1, 0)
                T.reads(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0 : 16, 0 : 4])
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16])
                T.block_attr({"meta_schedule.auto_tensorize":"dot_16x4_vnni"})
                with T.init():
                    for i4_1 in T.serial(16):
                        with T.block("conv2d_NCHWc_int8_init"):
                            oc_block_i_init = T.axis.spatial(16, i4_1)
                            T.reads()
                            T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init])
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                for i4_1, i9_1 in T.grid(16, 4):
                    with T.block("conv2d_NCHWc_int8"):
                        oc_block_i, ic_s_inner_i = T.axis.remap("SR", [i4_1, i9_1])
                        T.reads(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i], placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
                        T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i] = conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i] + T.cast(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], "int32") * T.cast(placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i], "int32")
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [8, 2, 1, 1]),
        ("SamplePerfectTile", [28, 1, 2, 1]),
        ("SamplePerfectTile", [56, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 4]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 1]),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [8, 2, 1, 1]),
        ("SamplePerfectTile", [28, 1, 2, 1]),
        ("SamplePerfectTile", [56, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 4]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 1]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [8, 2, 1, 1]),
        ("SamplePerfectTile", [28, 1, 2, 1]),
        ("SamplePerfectTile", [56, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 4]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 1]),
    ]

    mod = conv2d_nchwc
    target = Target("llvm -mcpu=cascadelake -num-cores=4")
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target(target),
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                VNNI_INTRIN,
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(req="may", levels=[1, 2], scope="global"),
            ),
        ],
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[vnni_conv2d_nchwc_0, vnni_conv2d_nchwc_1, vnni_conv2d_nchwc_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def _check_dp4a_dense(m, n, k, in_dtype, out_dtype, expected_mods, expected_decisions):
    def _dense(m, n, k, in_dtype, out_dtype):
        X = te.placeholder((m, k), name="X", dtype=in_dtype)
        W = te.placeholder((n, k), name="W", dtype=in_dtype)
        ak = te.reduce_axis((0, k), name="k")
        matmul = te.compute(
            (m, n),
            lambda i, j: te.sum(
                X[i, ak].astype(out_dtype) * W[j, ak].astype(out_dtype),
                axis=ak,
            ),
            name="compute",
        )
        return te.create_prim_func([X, W, matmul])

    mod = _dense(m, n, k, in_dtype, out_dtype)
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("cuda"),
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                DP4A_INTRIN,
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 3, 4],
                reuse_read=ms.schedule_rule.ReuseType(req="must", levels=[4], scope="shared"),
                reuse_write=ms.schedule_rule.ReuseType(req="must", levels=[3], scope="local"),
            )
        ],
    )
    if expected_mods is None:
        assert expected_decisions is None
        assert len(actual) == 1
        assert_structural_equal(mod, actual[0].mod["main"])
    else:
        check_sketches(mod, actual, expected_mods, expected_decisions)


def test_dp4a_dense():
    @T.prim_func
    def dp4a_dense_0(
        X: T.Buffer[(128, 128), "int8"],
        W: T.Buffer[(128, 128), "int8"],
        compute: T.Buffer[(128, 128), "int32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compute_local = T.alloc_buffer([128, 128], dtype="int32", scope="local")
        X_shared = T.alloc_buffer([128, 128], dtype="int8", scope="shared")
        W_shared = T.alloc_buffer([128, 128], dtype="int8", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(1, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(512, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(2, thread="threadIdx.x"):
                    for i2_0_0 in T.serial(1):
                        for ax0_ax1_fused in T.serial(16384):
                            with T.block("X_shared"):
                                v0 = T.axis.spatial(128, ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(X[v0, v1])
                                T.writes(X_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                X_shared[v0, v1] = X[v0, v1]
                        for ax0_ax1_fused in T.serial(16384):
                            with T.block("W_shared"):
                                v0 = T.axis.spatial(128, ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(128, ax0_ax1_fused % 128)
                                T.reads(W[v0, v1])
                                T.writes(W_shared[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                W_shared[v0, v1] = W[v0, v1]
                        for i2_0_1, i0_3, i1_3, i2_0_2, i0_4, i1_4 in T.grid(1, 2, 4, 32, 2, 1):
                            with T.block("compute_o"):
                                i = T.axis.spatial(
                                    128,
                                    i0_1_i1_1_fused // 32 * 8
                                    + i0_2_i1_2_fused * 4
                                    + i0_3 * 2
                                    + i0_4,
                                )
                                j = T.axis.spatial(128, i1_4 + i0_1_i1_1_fused % 32 * 4 + i1_3)
                                k_o = T.axis.reduce(32, i2_0_0 * 32 + i2_0_1 * 32 + i2_0_2)
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
                                        k_i = T.axis.reduce(4, i2_1)
                                        T.reads(
                                            compute_local[i, j],
                                            X_shared[i, k_o * 4 + k_i],
                                            W_shared[j, k_o * 4 + k_i],
                                        )
                                        T.writes(compute_local[i, j])
                                        T.block_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                        compute_local[i, j] = compute_local[i, j] + T.cast(
                                            X_shared[i, k_o * 4 + k_i], "int32"
                                        ) * T.cast(W_shared[j, k_o * 4 + k_i], "int32")
                    for ax0, ax1 in T.grid(4, 4):
                        with T.block("compute_local"):
                            v0 = T.axis.spatial(
                                128, i0_1_i1_1_fused // 32 * 8 + i0_2_i1_2_fused * 4 + ax0
                            )
                            v1 = T.axis.spatial(128, i0_1_i1_1_fused % 32 * 4 + ax1)
                            T.reads(compute_local[v0, v1])
                            T.writes(compute[v0, v1])
                            compute[v0, v1] = compute_local[v0, v1]

    decision_0 = [
        ("SamplePerfectTile", [1, 16, 2, 2, 2]),
        ("SamplePerfectTile", [1, 32, 1, 4, 1]),
        ("SamplePerfectTile", [1, 1, 32]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 0),
    ]
    _check_dp4a_dense(
        m=128,
        n=128,
        k=128,
        in_dtype="int8",
        out_dtype="int32",
        expected_mods=[dp4a_dense_0],
        expected_decisions=[decision_0],
    )


def test_dp4a_dense_no_tensorize_1():
    _check_dp4a_dense(
        m=128,
        n=128,
        k=128,
        in_dtype="float32",
        out_dtype="float32",
        expected_mods=None,
        expected_decisions=None,
    )


def test_dp4a_dense_no_tensorize_2():
    _check_dp4a_dense(
        m=127,
        n=127,
        k=127,
        in_dtype="int8",
        out_dtype="int32",
        expected_mods=None,
        expected_decisions=None,
    )


if __name__ == "__main__":
    test_vnni_conv2d_nchwc()
    test_dp4a_dense()
    test_dp4a_dense_no_tensorize_1()
    test_dp4a_dense_no_tensorize_2()
