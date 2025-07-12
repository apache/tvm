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
from tvm import relax
import tvm.testing
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.ir.module import IRModule


def verify(input, expected):
    mod = tvm.relax.backend.adreno.transform.FoldVDeviceScopeChange()(input)
    tvm.ir.assert_structural_equal(mod, expected)


def test_maxpool2d_scope_folding():
    @I.ir_module
    class Input:
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice({"device": "adreno", "kind": "opencl"}, 0, "global.texture-weight"),
                    I.vdevice({"device": "adreno", "kind": "opencl"}, 0, "global"),
                ]
            }
        )

        @T.prim_func(private=True)
        def max_pool2d_opencl(
            gv: T.Buffer((T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"),
            pool_max: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4)), "float32"
            ),
        ):
            # with T.block("root"):
            for ax0, ax1, ax2, ax3, ax4, rv0, rv1 in T.grid(
                T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4), T.int64(2), T.int64(2)
            ):
                with T.block("pool_max"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_rv0, v_rv1 = T.axis.remap(
                        "SSSSSRR", [ax0, ax1, ax2, ax3, ax4, rv0, rv1]
                    )
                    T.reads(
                        gv[
                            v_ax0,
                            v_ax1,
                            v_ax2 * T.int64(2) + v_rv0,
                            v_ax3 * T.int64(2) + v_rv1,
                            v_ax4,
                        ]
                    )
                    T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                    with T.init():
                        pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.float32(
                            -340282346638528859811704183484516925440.0
                        )
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.max(
                        pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4],
                        gv[
                            v_ax0,
                            v_ax1,
                            v_ax2 * T.int64(2) + v_rv0,
                            v_ax3 * T.int64(2) + v_rv1,
                            v_ax4,
                        ],
                    )

        @T.prim_func(private=True)
        def te_layout_transform(
            x: T.Buffer((T.int64(2), T.int64(4), T.int64(26), T.int64(26)), "float32"),
            te_layout_transform: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"
            ),
        ):
            # with T.block("root"):
            for self, i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(26), T.int64(26)):
                with T.block("te_layout_transform"):
                    v_self, v_i0, v_i1, v_i2 = T.axis.remap("SSSS", [self, i0, i1, i2])
                    T.reads(x[v_self, v_i0, v_i1, v_i2])
                    T.writes(
                        te_layout_transform[
                            v_self, v_i0 // T.int64(4), v_i1, v_i2, v_i0 % T.int64(4)
                        ]
                    )
                    te_layout_transform[
                        v_self, v_i0 // T.int64(4), v_i1, v_i2, v_i0 % T.int64(4)
                    ] = x[v_self, v_i0, v_i1, v_i2]

        @T.prim_func(private=True)
        def te_layout_transform2(
            lv2: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4)), "float32"
            ),
            te_layout_transform: T.Buffer(
                (T.int64(2), T.int64(4), T.int64(13), T.int64(13)), "float32"
            ),
        ):
            # with T.block("root"):
            for self, i0, i1, i2, i3 in T.grid(
                T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4)
            ):
                with T.block("te_layout_transform"):
                    v_self, v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSSS", [self, i0, i1, i2, i3])
                    T.reads(lv2[v_self, v_i0, v_i1, v_i2, v_i3])
                    T.writes(te_layout_transform[v_self, v_i3, v_i1, v_i2])
                    te_layout_transform[v_self, v_i3, v_i1, v_i2] = lv2[
                        v_self, v_i0, v_i1, v_i2, v_i3
                    ]

        @R.function
        def main(
            x: R.Tensor((2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"),  # noqa: F722
        ) -> R.Tensor((2, 4, 13, 13), dtype="float32", vdevice="opencl:1:global"):  # noqa: F722
            cls = Input
            with R.dataflow():
                lv = R.call_tir(
                    cls.te_layout_transform,
                    (x,),
                    out_sinfo=R.Tensor(
                        (2, 1, 26, 26, 4), dtype="float32", vdevice="opencl:0:global.texture-weight"
                    ),
                )
                lv2 = R.call_tir(
                    cls.max_pool2d_opencl,
                    (lv,),
                    out_sinfo=R.Tensor(
                        (2, 1, 13, 13, 4), dtype="float32", vdevice="opencl:0:global.texture-weight"
                    ),
                )
                lv5: R.Tensor(
                    (2, 1, 13, 13, 4), dtype="float32", vdevice="opencl:1:global"  # noqa: F722
                ) = R.to_vdevice(lv2, dst_vdevice="opencl:1:global")
                gv2 = R.call_tir(
                    cls.te_layout_transform2,
                    (lv5,),
                    out_sinfo=R.Tensor((2, 4, 13, 13), dtype="float32", vdevice="opencl:1:global"),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice({"device": "adreno", "kind": "opencl"}, 0, "global.texture-weight"),
                    I.vdevice({"device": "adreno", "kind": "opencl"}, 0, "global"),
                ]
            }
        )

        @T.prim_func(private=True)
        def max_pool2d_opencl(
            gv: T.Buffer((T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"),
            pool_max: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4)), "float32"
            ),
        ):
            # with T.block("root"):
            for ax0, ax1, ax2, ax3, ax4, rv0, rv1 in T.grid(
                T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4), T.int64(2), T.int64(2)
            ):
                with T.block("pool_max"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_rv0, v_rv1 = T.axis.remap(
                        "SSSSSRR", [ax0, ax1, ax2, ax3, ax4, rv0, rv1]
                    )
                    T.reads(
                        gv[
                            v_ax0,
                            v_ax1,
                            v_ax2 * T.int64(2) + v_rv0,
                            v_ax3 * T.int64(2) + v_rv1,
                            v_ax4,
                        ]
                    )
                    T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                    with T.init():
                        pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.float32(
                            -340282346638528859811704183484516925440.0
                        )
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.max(
                        pool_max[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4],
                        gv[
                            v_ax0,
                            v_ax1,
                            v_ax2 * T.int64(2) + v_rv0,
                            v_ax3 * T.int64(2) + v_rv1,
                            v_ax4,
                        ],
                    )

        @T.prim_func(private=True)
        def te_layout_transform(
            x: T.Buffer((T.int64(2), T.int64(4), T.int64(26), T.int64(26)), "float32"),
            te_layout_transform: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"
            ),
        ):
            # with T.block("root"):
            for self, i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(26), T.int64(26)):
                with T.block("te_layout_transform"):
                    v_self, v_i0, v_i1, v_i2 = T.axis.remap("SSSS", [self, i0, i1, i2])
                    T.reads(x[v_self, v_i0, v_i1, v_i2])
                    T.writes(
                        te_layout_transform[
                            v_self, v_i0 // T.int64(4), v_i1, v_i2, v_i0 % T.int64(4)
                        ]
                    )
                    te_layout_transform[
                        v_self, v_i0 // T.int64(4), v_i1, v_i2, v_i0 % T.int64(4)
                    ] = x[v_self, v_i0, v_i1, v_i2]

        @T.prim_func(private=True)
        def te_layout_transform2(
            lv2: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4)), "float32"
            ),
            te_layout_transform: T.Buffer(
                (T.int64(2), T.int64(4), T.int64(13), T.int64(13)), "float32"
            ),
        ):
            # with T.block("root"):
            for self, i0, i1, i2, i3 in T.grid(
                T.int64(2), T.int64(1), T.int64(13), T.int64(13), T.int64(4)
            ):
                with T.block("te_layout_transform"):
                    v_self, v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSSS", [self, i0, i1, i2, i3])
                    T.reads(lv2[v_self, v_i0, v_i1, v_i2, v_i3])
                    T.writes(te_layout_transform[v_self, v_i3, v_i1, v_i2])
                    te_layout_transform[v_self, v_i3, v_i1, v_i2] = lv2[
                        v_self, v_i0, v_i1, v_i2, v_i3
                    ]

        @R.function
        def main(
            x: R.Tensor((2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"),  # noqa: F722
        ) -> R.Tensor((2, 4, 13, 13), dtype="float32", vdevice="opencl:1:global"):  # noqa: F722
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(
                    cls.te_layout_transform,
                    (x,),
                    out_sinfo=R.Tensor(
                        (2, 1, 26, 26, 4), dtype="float32", vdevice="opencl:0:global.texture-weight"
                    ),
                )
                lv5 = R.call_tir(
                    cls.max_pool2d_opencl,
                    (lv,),
                    out_sinfo=R.Tensor(
                        (2, 1, 13, 13, 4), dtype="float32", vdevice="opencl:1:global"
                    ),
                )
                gv2 = R.call_tir(
                    cls.te_layout_transform2,
                    (lv5,),
                    out_sinfo=R.Tensor((2, 4, 13, 13), dtype="float32", vdevice="opencl:1:global"),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


if __name__ == "__main__":
    tvm.testing.main()
