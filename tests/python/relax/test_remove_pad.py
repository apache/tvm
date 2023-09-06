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
import pytest
import tvm.testing
import tvm.topi.testing
import numpy as np
from tvm import relax
from tvm.relax.transform import LegalizeOps
from tvm.script import tir as T, ir as I, relax as R


def get_original_mod():
    @I.ir_module
    class Before:
        @T.prim_func
        def add(
            arg0: T.Buffer((1, 17, 147, 147), "float16"),
            arg1: T.Buffer((1, 17, 147, 147), "float16"),
            output: T.Buffer((1, 17, 147, 147), "float16"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            for ax0, ax1, ax2, ax3 in T.grid(1, 17, 147, 147):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(arg0[v_ax0, v_ax1, v_ax2, v_ax3], arg1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(output[v_ax0, v_ax1, v_ax2, v_ax3])
                    output[v_ax0, v_ax1, v_ax2, v_ax3] = (
                        arg0[v_ax0, v_ax1, v_ax2, v_ax3] + arg1[v_ax0, v_ax1, v_ax2, v_ax3]
                    )

        @R.function
        def main(
            x: R.Tensor((1, 17, 147, 147), dtype="float16"),
            y: R.Tensor((1, 17, 147, 147), dtype="float16"),
        ) -> R.Tensor((1, 17, 147, 147), dtype="float16"):
            with R.dataflow():
                lv = R.call_tir(
                    Before.add, (x, y), out_sinfo=R.Tensor((1, 17, 147, 147), dtype="float16")
                )
                gv: R.Tensor((1, 17, 147, 147), dtype="float16") = lv
                R.output(gv)
            return gv

    mod = LegalizeOps()(Before)
    return mod


def get_modified_mod():
    @I.ir_module
    class Module:
        @T.prim_func
        def relax_add_replacement(
            arg0: T.Buffer((1, 3, 37, 5, 8, 2, 32, 2), "float16"),
            arg1: T.Buffer((1, 3, 37, 5, 8, 2, 32, 2), "float16"),
            output: T.Buffer((1, 3, 37, 5, 8, 2, 32, 2), "float16"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with T.block("root"):
            for axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7 in T.grid(
                1, 3, 37, 5, 8, 2, 32, 2
            ):
                with T.block("buffer_arg1_assumptions"):
                    (
                        v_axis0,
                        v_axis1,
                        v_axis2,
                        v_axis3,
                        v_axis4,
                        v_axis5,
                        v_axis6,
                        v_axis7,
                    ) = T.axis.remap(
                        "SSSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7]
                    )
                    T.reads(
                        arg1[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7]
                    )
                    T.writes()
                    T.assume(
                        not (
                            v_axis1 == 2
                            and 1 <= v_axis4
                            or v_axis2 == 36
                            and v_axis5 * 2 + v_axis7 == 3
                            or v_axis3 == 4
                            and 19 <= v_axis6
                        )
                        or arg1[
                            v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7
                        ]
                        == T.float16(0)
                    )
            for axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7 in T.grid(
                1, 3, 37, 5, 8, 2, 32, 2
            ):
                with T.block("buffer_arg0_assumptions"):
                    (
                        v_axis0,
                        v_axis1,
                        v_axis2,
                        v_axis3,
                        v_axis4,
                        v_axis5,
                        v_axis6,
                        v_axis7,
                    ) = T.axis.remap(
                        "SSSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7]
                    )
                    T.reads(
                        arg0[v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7]
                    )
                    T.writes()
                    T.assume(
                        not (
                            v_axis1 == 2
                            and 1 <= v_axis4
                            or v_axis2 == 36
                            and v_axis5 * 2 + v_axis7 == 3
                            or v_axis3 == 4
                            and 19 <= v_axis6
                        )
                        or arg0[
                            v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7
                        ]
                        == T.float16(0)
                    )
            for axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7 in T.grid(
                1, 3, 37, 5, 8, 2, 32, 2
            ):
                with T.block("T_add"):
                    (
                        v_axis0,
                        v_axis1,
                        v_axis2,
                        v_axis3,
                        v_axis4,
                        v_axis5,
                        v_axis6,
                        v_axis7,
                    ) = T.axis.remap(
                        "SSSSSSSS", [axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7]
                    )
                    T.reads(
                        arg0[
                            v_axis0,
                            (v_axis1 * 8 + v_axis4) // 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 4,
                            (v_axis3 * 32 + v_axis6) // 32,
                            (v_axis1 * 8 + v_axis4) % 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 4 // 2,
                            (v_axis3 * 32 + v_axis6) % 32,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 2,
                        ],
                        arg1[
                            v_axis0,
                            (v_axis1 * 8 + v_axis4) // 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 4,
                            (v_axis3 * 32 + v_axis6) // 32,
                            (v_axis1 * 8 + v_axis4) % 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 4 // 2,
                            (v_axis3 * 32 + v_axis6) % 32,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 2,
                        ],
                    )
                    T.writes(
                        output[
                            v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7
                        ]
                    )
                    output[
                        v_axis0, v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7
                    ] = T.if_then_else(
                        v_axis1 == 2
                        and 1 <= v_axis4
                        or v_axis2 == 36
                        and v_axis5 * 2 + v_axis7 == 3
                        or v_axis3 == 4
                        and 19 <= v_axis6,
                        T.float16(0),
                        arg0[
                            v_axis0,
                            (v_axis1 * 8 + v_axis4) // 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 4,
                            (v_axis3 * 32 + v_axis6) // 32,
                            (v_axis1 * 8 + v_axis4) % 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 4 // 2,
                            (v_axis3 * 32 + v_axis6) % 32,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 2,
                        ]
                        + arg1[
                            v_axis0,
                            (v_axis1 * 8 + v_axis4) // 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 4,
                            (v_axis3 * 32 + v_axis6) // 32,
                            (v_axis1 * 8 + v_axis4) % 8,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 4 // 2,
                            (v_axis3 * 32 + v_axis6) % 32,
                            (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 2,
                        ],
                    )

        @R.function
        def main(
            x: R.Tensor((1, 17, 147, 147), dtype="float16"),
            y: R.Tensor((1, 17, 147, 147), dtype="float16"),
        ) -> R.Tensor((1, 17, 147, 147), dtype="float16"):
            cls = Module
            with R.dataflow():
                lv: R.Tensor((1, 3, 37, 5, 8, 2, 32, 2), dtype="float16") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda n, h, w, c: (
                            n,
                            h // 8,
                            w // 4,
                            c // 32,
                            h % 8,
                            w % 4 // 2,
                            c % 32,
                            w % 2,
                        )
                    ),
                    pad_value=0.0,
                )
                lv1: R.Tensor((1, 3, 37, 5, 8, 2, 32, 2), dtype="float16") = R.layout_transform(
                    y,
                    index_map=T.index_map(
                        lambda n, h, w, c: (
                            n,
                            h // 8,
                            w // 4,
                            c // 32,
                            h % 8,
                            w % 4 // 2,
                            c % 32,
                            w % 2,
                        )
                    ),
                    pad_value=0.0,
                )
                lv2 = R.call_tir(
                    cls.relax_add_replacement,
                    (lv, lv1),
                    out_sinfo=R.Tensor((1, 3, 37, 5, 8, 2, 32, 2), dtype="float16"),
                )
                lv3: R.Tensor((1, 24, 148, 160), dtype="float16") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda axis0, axis1, axis2, axis3, axis4, axis5, axis6, axis7: (
                            axis0,
                            axis1 * 8 + axis4,
                            axis2 * 4 + axis5 * 2 + axis7,
                            axis3 * 32 + axis6,
                        )
                    ),
                    pad_value=0.0,
                )
                lv_1: R.Tensor((1, 17, 147, 147), dtype="float16") = R.remove_pad(
                    lv3, orig_shape=[1, 17, 147, 147]
                )
                gv: R.Tensor((1, 17, 147, 147), dtype="float16") = lv_1
                R.output(gv)
            return gv

    mod = LegalizeOps()(Module)
    return mod


def test_remove_pad():
    target = "llvm"
    dev = tvm.device(target, 0)

    inp1 = np.random.rand(1, 17, 147, 147).astype(np.float16)
    inp2 = np.random.rand(1, 17, 147, 147).astype(np.float16)
    inputs = [tvm.nd.array(inp, dev) for inp in [inp1, inp2]]

    ex1 = relax.build(mod=get_original_mod(), target=target, exec_mode="compiled")
    vm_rt1 = relax.VirtualMachine(ex1, dev)

    ex2 = relax.build(mod=get_modified_mod(), target=target, exec_mode="compiled")
    vm_rt2 = relax.VirtualMachine(ex2, dev)

    res1 = vm_rt1["main"](*inputs).numpy()
    res2 = vm_rt2["main"](*inputs).numpy()

    assert (res1 == res2).all()


if __name__ == "__main__":
    tvm.testing.main()
