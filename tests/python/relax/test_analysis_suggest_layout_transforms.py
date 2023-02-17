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

import pytest
import tvm.testing

from typing import Callable, Any
from tvm import relay, topi
from tvm import relax, tir
from tvm.script import relax as R, tir as T, ir as I


def apply_transformations(func, suggested_transfoms):
    sch = tir.Schedule(func)
    visited = set()
    for block, per_block_transformations in suggested_transfoms.items():
        blockrv = sch.get_block(block.name_hint)
        for obj, index_map in per_block_transformations.items():
            if obj in visited:
                continue
            visited.add(obj)
            if isinstance(obj, tir.Block):
                block_name = obj.name_hint
                print("Block transformation: ", block_name, " :: ", index_map)
                sch.transform_block_layout(block_name, index_map)
            elif isinstance(obj, tir.Buffer):
                buffer = obj
                print("Buffer transformation: ", buffer, " :: ", index_map)
                sch.transform_layout(blockrv, buffer, index_map)
            else:
                print("Unknown object: ", obj)
                return
    return sch.mod["main"]


def test_op_elemwise():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        relu: T.Buffer((32, 64, 224, 224), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 224, 224):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(arg[v_i0, v_i1, v_i2, v_i3])
                T.writes(relu[v_i0, v_i1, v_i2, v_i3])
                relu[v_i0, v_i1, v_i2, v_i3] = T.max(arg[v_i0, v_i1, v_i2, v_i3], T.float32(0))

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 224, 224, 64), "float32"),
        relu: T.Buffer((32, 224, 224, 64), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 64):
            with T.block("compute"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v0, v1, v2, v3])
                T.writes(relu[v0, v1, v2, v3])
                relu[v0, v1, v2, v3] = T.max(arg[v0, v1, v2, v3], T.float32(0))

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_pool_nchw_nhwc():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        pool_max: T.Buffer((32, 64, 111, 223), "float32"),
    ):
        for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(32, 64, 111, 223, 2, 2):
            with T.block("pool_max"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap(
                    "SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1]
                )
                T.reads(
                    arg[
                        v_ax0,
                        v_ax1,
                        v_ax2 * 2 + v_rv0 * 2,
                        v_ax3 + v_rv1,
                    ]
                )
                T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                with T.init():
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-3.4028234663852886e38)
                pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3],
                    arg[
                        v_ax0,
                        v_ax1,
                        v_ax2 * 2 + v_rv0 * 2,
                        v_ax3 + v_rv1,
                    ],
                )

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 224, 224, 64), "float32"),
        pool_max: T.Buffer((32, 111, 223, 64), "float32"),
    ):
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(32, 111, 223, 64, 2, 2):
            with T.block("pool_max"):
                v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, ax4, ax5])
                T.reads(arg[v0, v1 * 2 + v4 * 2, v2 + v5, v3])
                T.writes(pool_max[v0, v1, v2, v3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                with T.init():
                    pool_max[v0, v1, v2, v3] = T.float32(-3.4028234663852886e38)
                pool_max[v0, v1, v2, v3] = T.max(
                    pool_max[v0, v1, v2, v3],
                    arg[v0, v1 * 2 + v4 * 2, v2 + v5, v3],
                )

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c)],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_pool_nchw16c_nhwc():
    @T.prim_func
    def before(
        arg: T.Buffer(
            (32, 4, 224, 224, 16),
            "float32",
        ),
        pool_max: T.Buffer(
            (32, 4, 110, 220, 16),
            "float32",
        ),
    ):
        for ax0, ax1, ax2, ax3, ax4, rv0, rv1 in T.grid(32, 4, 110, 220, 16, 5, 5):
            with T.block("pool_max"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_rv0, v_rv1 = T.axis.remap(
                    "SSSSSRR", [ax0, ax1, ax2, ax3, ax4, rv0, rv1]
                )
                T.reads(arg[v_ax0, v_ax1, v_ax2 * 2 + v_rv0, v_ax3 + v_rv1, v_ax4])
                T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                with T.init():
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.float32(-3.4028234663852886e38)
                pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.max(
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4],
                    arg[v_ax0, v_ax1, v_ax2 * 2 + v_rv0, v_ax3 + v_rv1, v_ax4],
                )

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 224, 224, 64), "float32"),
        pool_max: T.Buffer((32, 110, 220, 64), "float32"),
    ):
        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(32, 110, 220, 64, 5, 5):
            with T.block("pool_max"):
                v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, ax4, ax5])
                T.reads(arg[v0, v1 * 2 + v4, v2 + v5, v3])
                T.writes(pool_max[v0, v1, v2, v3])
                T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                with T.init():
                    pool_max[v0, v1, v2, v3] = T.float32(-3.4028234663852886e38)
                pool_max[v0, v1, v2, v3] = T.max(
                    pool_max[v0, v1, v2, v3],
                    arg[v0, v1 * 2 + v4, v2 + v5, v3],
                )

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_buffer_transforms=[lambda n, C, h, w, c: (n, h, w, C * 16 + c)],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_reduce():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        sum: T.Buffer((32, 64), "float32"),
    ):
        for ax0, ax1, k2, k3 in T.grid(32, 64, 224, 224):
            with T.block("rxplaceholder_red"):
                v_ax0, v_ax1, v_k2, v_k3 = T.axis.remap("SSRR", [ax0, ax1, k2, k3])
                T.reads(arg[v_ax0, v_ax1, v_k2, v_k3])
                T.writes(sum[v_ax0, v_ax1])
                with T.init():
                    sum[v_ax0, v_ax1] = T.float32(0)
                sum[v_ax0, v_ax1] = sum[v_ax0, v_ax1] + arg[v_ax0, v_ax1, v_k2, v_k3]

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 4, 224, 224, 16), "float32"),
        sum: T.Buffer((32, 4, 16), "float32"),
    ):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 4, 224, 224, 16):
            with T.block("rxplaceholder_red"):
                v0, v1, v2, v3, v4 = T.axis.remap("SSRRS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(arg[v0, v1, v2, v3, v4])
                T.writes(sum[v0, v1, v4])
                with T.init():
                    sum[v0, v1, v4] = T.float32(0)
                sum[v0, v1, v4] = sum[v0, v1, v4] + arg[v0, v1, v2, v3, v4]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_buffer_transforms=[lambda n, c: (n, c // 16, c % 16)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_upsampling():
    # relay materializes the layout if H, W or D dimensions are moved or tiled.
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        resize: T.Buffer((32, 64, 202, 246), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 202, 246):
            with T.block("resize"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(arg[v_i0, v_i1, 0:224, 0:224])
                T.writes(resize[v_i0, v_i1, v_i2, v_i3])
                resize[v_i0, v_i1, v_i2, v_i3] = arg[
                    v_i0,
                    v_i1,
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(1.1089109182357788) * T.Cast("float32", v_i2)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            223,
                        ),
                        0,
                    ),
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(0.91056913137435913) * T.Cast("float32", v_i3)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            223,
                        ),
                        0,
                    ),
                ]

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        resize: T.Buffer((32, 202, 246, 64), "float32"),
    ):
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(32, 202, 246, 64):
            with T.block("resize"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v0, v3, 0:224, 0:224])
                T.writes(resize[v0, v1, v2, v3])
                resize[v0, v1, v2, v3] = arg[
                    v0,
                    v3,
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(1.1089109182357788) * T.Cast("float32", v1)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            T.int64(223),
                        ),
                        T.int64(0),
                    ),
                    T.max(
                        T.min(
                            T.Cast(
                                "int64",
                                T.floor(
                                    T.float32(0.91056913137435913) * T.Cast("float32", v2)
                                    + T.float32(1.0000000000000001e-05)
                                ),
                            ),
                            T.int64(223),
                        ),
                        T.int64(0),
                    ),
                ]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_strided_slice():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        T_strided_slice_with_axes: T.Buffer((32, 64, 10, 8), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 64, 10, 8):
            with T.block("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(
                    arg[
                        v_ax0,
                        v_ax1,
                        v_ax2 * 5 + 2,
                        v_ax3 * 7 + 4,
                    ]
                )
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = arg[
                    v_ax0,
                    v_ax1,
                    v_ax2 * 5 + 2,
                    v_ax3 * 7 + 4,
                ]

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 224, 224, 16, 4), "float32"),
        T_strided_slice_with_axes: T.Buffer((32, 10, 8, 16, 4), "float32"),
    ):
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 10, 8, 16, 4):
            with T.block("T_strided_slice_with_axes"):
                v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(arg[v0, v1 * 5 + 2, v2 * 7 + 4, v3, v4])
                T.writes(T_strided_slice_with_axes[v0, v1, v2, v3, v4])
                T_strided_slice_with_axes[v0, v1, v2, v3, v4] = arg[
                    v0, v1 * 5 + 2, v2 * 7 + 4, v3, v4
                ]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c // 4, c % 4)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_binary_broadcast():
    @T.prim_func
    def before(
        arg0: T.Buffer((32, 64, 224, 224), "float32"),
        arg1: T.Buffer((64, 224, 224), "float32"),
        T_add: T.Buffer((32, 64, 224, 224), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(32, 64, 224, 224):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(
                    arg0[v_ax0, v_ax1, v_ax2, v_ax3],
                    arg1[v_ax1, v_ax2, v_ax3],
                )
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = (
                    arg0[v_ax0, v_ax1, v_ax2, v_ax3] + arg1[v_ax1, v_ax2, v_ax3]
                )

    @T.prim_func
    def expected(
        arg0: T.Buffer((32, 224, 224, 16, 4), "float32"),
        arg1: T.Buffer((224, 224, 16, 4), "float32"),
        T_add: T.Buffer((32, 224, 224, 16, 4), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 224, 224, 16, 4):
            with T.block("T_add"):
                v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(arg0[v0, v1, v2, v3, v4], arg1[v1, v2, v3, v4])
                T.writes(T_add[v0, v1, v2, v3, v4])
                T_add[v0, v1, v2, v3, v4] = arg0[v0, v1, v2, v3, v4] + arg1[v1, v2, v3, v4]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c // 4, c % 4)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_transpose():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        T_transpose: T.Buffer((32, 224, 224, 64), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 64):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v_ax0, v_ax3, v_ax1, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax3, v_ax1, v_ax2]

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        T_transpose: T.Buffer((32, 224, 64, 224), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 64, 224):
            with T.block("T_transpose"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v0, v2, v3, v1])
                T.writes(T_transpose[v0, v1, v2, v3])
                T_transpose[v0, v1, v2, v3] = arg[v0, v2, v3, v1]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_pad():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        PadInput: T.Buffer((32, 64, 230, 230), "float32"),
    ):
        for i0, i1, i2, i3 in T.grid(32, 64, 230, 230):
            with T.block("PadInput"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(arg[v_i0, v_i1, v_i2 - 2, v_i3 - 2])
                T.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                    2 <= v_i2 and v_i2 < 226 and 2 <= v_i3 and v_i3 < 226,
                    arg[v_i0, v_i1, v_i2 - 2, v_i3 - 2],
                    T.float32(2),
                )

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 224, 224, 16, 4), "float32"),
        PadInput: T.Buffer((32, 230, 230, 16, 4), "float32"),
    ):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 230, 230, 16, 4):
            with T.block("PadInput"):
                v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(arg[v0, v1 - 2, v2 - 2, v3, v4])
                T.writes(PadInput[v0, v1, v2, v3, v4])
                PadInput[v0, v1, v2, v3, v4] = T.if_then_else(
                    2 <= v1 and v1 < 226 and 2 <= v2 and v2 < 226,
                    arg[v0, v1 - 2, v2 - 2, v3, v4],
                    T.float32(2),
                )

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before, write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c // 4, c % 4)]
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_split():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        split0: T.Buffer((32, 32, 224, 224), "float32"),
        split1: T.Buffer((32, 32, 224, 224), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with T.block("T_split_sections"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(split0[v_ax0, v_ax1, v_ax2, v_ax3])
                split0[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3])
                T.writes(split1[v_ax0, v_ax1, v_ax2, v_ax3])
                split1[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3]

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 224, 224, 64), "float32"),
        split0: T.Buffer((32, 224, 224, 32), "float32"),
        split1: T.Buffer((32, 224, 224, 32), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 32):
            with T.block("T_split_sections"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v0, v1, v2, v3])
                T.writes(split0[v0, v1, v2, v3])
                split0[v0, v1, v2, v3] = arg[v0, v1, v2, v3]
        for ax0, ax1, ax2, ax3 in T.grid(32, 224, 224, 32):
            with T.block("T_split_sections_1"):
                v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v0, v1, v2, v3 + 32])
                T.writes(split1[v0, v1, v2, v3])
                split1[v0, v1, v2, v3] = arg[v0, v1, v2, v3 + 32]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_buffer_transforms=[lambda n, c, h, w: (n, h, w, c), lambda n, c, h, w: (n, h, w, c)],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_op_split_tiling_split_dim():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        split0: T.Buffer((32, 32, 224, 224), "float32"),
        split1: T.Buffer((32, 32, 224, 224), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with T.block("T_split_sections"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(split0[v_ax0, v_ax1, v_ax2, v_ax3])
                split0[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3])
                T.writes(split1[v_ax0, v_ax1, v_ax2, v_ax3])
                split1[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3]

    @T.prim_func
    def expected(
        arg: T.Buffer((32, 224, 224, 16, 4), "float32"),
        split0: T.Buffer((32, 224, 224, 8, 4), "float32"),
        split1: T.Buffer((32, 224, 224, 8, 4), "float32"),
    ):
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 224, 224, 8, 4):
            with T.block("T_split_sections"):
                v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(arg[v0, v1, v2, v3, v4])
                T.writes(split0[v0, v1, v2, v3, v4])
                split0[v0, v1, v2, v3, v4] = arg[v0, v1, v2, v3, v4]
        for ax0, ax1, ax2, ax3, ax4 in T.grid(32, 224, 224, 8, 4):
            with T.block("T_split_sections_1"):
                v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(arg[v0, v1, v2, v3 + 8, v4])
                T.writes(split1[v0, v1, v2, v3, v4])
                split1[v0, v1, v2, v3, v4] = arg[v0, v1, v2, v3 + 8, v4]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_buffer_transforms=[
            lambda n, c, h, w: (n, h, w, c // 4, c % 4),
            lambda n, c, h, w: (n, h, w, c // 4, c % 4),
        ],
    )
    after = apply_transformations(before, suggested_transforms)
    tvm.ir.assert_structural_equal(after, expected)


def test_invalid_non_bijective():
    @T.prim_func
    def before(
        arg: T.Buffer((32, 64, 224, 224), "float32"),
        split0: T.Buffer((32, 32, 224, 224), "float32"),
        split1: T.Buffer((32, 32, 224, 224), "float32"),
    ):
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with T.block("T_split_sections"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(split0[v_ax0, v_ax1, v_ax2, v_ax3])
                split0[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0, ax1, ax2, ax3 in T.grid(32, 32, 224, 224):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3])
                T.writes(split1[v_ax0, v_ax1, v_ax2, v_ax3])
                split1[v_ax0, v_ax1, v_ax2, v_ax3] = arg[v_ax0, v_ax1 + 32, v_ax2, v_ax3]

    suggested_transforms = relax.analysis.suggest_layout_transforms(
        func=before,
        write_buffer_transforms=[
            lambda n, c, h, w: (n, h, w, c // 5, c % 5),
            lambda n, c, h, w: (n, h, w, c // 5, c % 5),
        ],
    )
    after = apply_transformations(before, suggested_transforms)
    # no transformations suggested
    tvm.ir.assert_structural_equal(after, before)
