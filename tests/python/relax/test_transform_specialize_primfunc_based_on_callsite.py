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
from tvm.relax.transform.legalize_ops import adreno as legalize_adreno
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor


@visitor
class ValidateBufferScopes(PyExprVisitor):  # pylint: disable=abstract-method
    def __init__(self, is_matched: bool) -> None:
        self.is_matched = is_matched

    def visit(self, mod: IRModule) -> None:
        """Entry point"""
        self.mod = mod
        for key, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.visit_expr(func)

    def visit_call_(self, call: relax.Call) -> None:  # pylint: disable=arguments-renamed
        if call.op.name == "relax.call_tir":
            pfunc = self.mod[call.args[0]]
            if not self.is_matched:
                # All scopes should be global in before pass
                for _, buf in pfunc.buffer_map.items():
                    assert (
                        "global" == buf.data.type_annotation.storage_scope
                    ), f"expected to be global scoped, but got {val.data.type_annotation.storage_scope}"
            else:
                for idx, arg in enumerate(call.args[1]):
                    arg_sinfo = arg.struct_info
                    assert isinstance(
                        arg_sinfo, relax.TensorStructInfo
                    ), f"Expected TensorStructInfo but git {type(arg_sinfo)}"
                    buf = pfunc.buffer_map[pfunc.params[idx]]
                    assert (
                        arg_sinfo.vdevice.memory_scope == buf.data.type_annotation.storage_scope
                    ), f"scope mismatched after specialization {arg_sinfo.vdevice.memory_scope} vs {buf.data.type_annotation.storage_scope}"
                if isinstance(call.sinfo_args[0], relax.TensorStructInfo):
                    buf = pfunc.buffer_map[pfunc.params[-1]]
                    assert (
                        call.sinfo_args[0].vdevice.memory_scope
                        == buf.data.type_annotation.storage_scope
                    ), f"scope mismatched after specialization {call.sinfo_args[0].vdevice.memory_scope} vs {buf.data.type_annotation.storage_scope}"
                else:
                    assert isinstance(
                        call.sinfo_args[0], relax.TupleStructInfo
                    ), f"Expected TupleStructInfo but git {type(call.sinfo_args[0])}"
                    for idx, sinfo in enumerate(call.sinfo_args[0].fields):
                        buf = pfunc.buffer_map[pfunc.params[len(call.args[1]) + idx]]
                        assert (
                            sinfo.vdevice.memory_scope == buf.data.type_annotation.storage_scope
                        ), f"scope mismatched after specialization {sinfo.vdevice.memory_scope} vs {buf.data.type_annotation.storage_scope}"


def verify(input):
    ValidateBufferScopes(False).visit(input)
    mod = tvm.relax.transform.SpecializePrimFuncBasedOnCallSite()(input)
    ValidateBufferScopes(True).visit(mod)


def test_single_arg_return():
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

    verify(Input)


def test_multi_arg_return():
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
        def conv2d_NCHWc_OIHWo_opencl(
            lv: T.Buffer((T.int64(2), T.int64(4), T.int64(28), T.int64(28), T.int64(4)), "float32"),
            lv1: T.Buffer((T.int64(1), T.int64(16), T.int64(3), T.int64(3), T.int64(4)), "float32"),
            conv2d_NCHWc_OIHWo: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"
            ),
        ):
            conv2d_NCHWc_OIHWo[0, 0, 0, 0, 0] = T.float32(0.0)

        @T.prim_func(private=True)
        def fused_relu_concatenate_split(
            gv: T.Buffer((T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"),
            T_split_sections_intermediate: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"
            ),
            T_split_sections_intermediate_1: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"
            ),
        ):
            T_split_sections_intermediate[0, 0, 0, 0, 0] = T.float32(0.0)
            T_split_sections_intermediate_1[0, 0, 0, 0, 0] = T.float32(0.0)

        @T.prim_func(private=True)
        def te_layout_transform(
            x: T.Buffer((T.int64(2), T.int64(16), T.int64(28), T.int64(28)), "float32"),
            te_layout_transform: T.Buffer(
                (T.int64(2), T.int64(4), T.int64(28), T.int64(28), T.int64(4)), "float32"
            ),
        ):
            te_layout_transform[0, 0, 0, 0, 0] = T.float32(0.0)

        @T.prim_func(private=True)
        def te_layout_transform1(
            w: T.Buffer((T.int64(4), T.int64(16), T.int64(3), T.int64(3)), "float32"),
            te_layout_transform: T.Buffer(
                (T.int64(1), T.int64(16), T.int64(3), T.int64(3), T.int64(4)), "float32"
            ),
        ):
            te_layout_transform[0, 0, 0, 0, 0] = T.float32(0.0)

        @T.prim_func(private=True)
        def te_layout_transform2(
            lv3: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(26), T.int64(26), T.int64(4)), "float32"
            ),
            te_layout_transform: T.Buffer(
                (T.int64(2), T.int64(4), T.int64(26), T.int64(26)), "float32"
            ),
        ):
            te_layout_transform[0, 0, 0, 0] = T.float32(0.0)

        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32", vdevice="opencl:1:global"),  # noqa: F722
            w: R.Tensor((4, 16, 3, 3), dtype="float32", vdevice="opencl:1:global"),  # noqa: F722
        ) -> R.Tuple(
            R.Tensor((2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"),  # noqa: F722
            R.Tensor((2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"),  # noqa: F722
        ):
            cls = Input
            with R.dataflow():
                lv = R.call_tir(
                    cls.te_layout_transform,
                    (x,),
                    out_sinfo=R.Tensor(
                        (2, 4, 28, 28, 4), dtype="float32", vdevice="opencl:0:global.texture-weight"
                    ),
                )
                lv1 = R.call_tir(
                    cls.te_layout_transform1,
                    (w,),
                    out_sinfo=R.Tensor(
                        (1, 16, 3, 3, 4), dtype="float32", vdevice="opencl:0:global.texture-weight"
                    ),
                )
                gv = R.call_tir(
                    cls.conv2d_NCHWc_OIHWo_opencl,
                    (lv, lv1),
                    out_sinfo=R.Tensor(
                        (2, 1, 26, 26, 4), dtype="float32", vdevice="opencl:0:global.texture-weight"
                    ),
                )
                lv_1 = R.call_tir(
                    cls.fused_relu_concatenate_split,
                    (gv,),
                    out_sinfo=[
                        R.Tensor((2, 1, 26, 26, 4), dtype="float32", vdevice="opencl:1:global"),
                        R.Tensor((2, 1, 26, 26, 4), dtype="float32", vdevice="opencl:1:global"),
                    ],
                )
                lv3: R.Tensor(
                    (2, 1, 26, 26, 4), dtype="float32", vdevice="opencl:1:global"  # noqa: F722
                ) = lv_1[0]
                lv4 = R.call_tir(
                    cls.te_layout_transform2,
                    (lv3,),
                    out_sinfo=R.Tensor((2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"),
                )
                lv5: R.Tensor(
                    (2, 1, 26, 26, 4), dtype="float32", vdevice="opencl:1:global"  # noqa: F722
                ) = lv_1[1]
                lv6 = R.call_tir(
                    cls.te_layout_transform2,
                    (lv5,),
                    out_sinfo=R.Tensor((2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"),
                )
                gv4: R.Tuple(
                    R.Tensor(
                        (2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"  # noqa: F722
                    ),
                    R.Tensor(
                        (2, 4, 26, 26), dtype="float32", vdevice="opencl:1:global"  # noqa: F722
                    ),
                ) = (lv4, lv6)
                R.output(gv4)
            return gv4

    verify(Input)


if __name__ == "__main__":
    tvm.testing.main()
