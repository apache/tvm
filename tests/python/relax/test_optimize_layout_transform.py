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
"""Tests to validate relax optimize layout tranform pass."""

import numpy as np
import pytest
import tvm.testing
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.relax.transform import DeadCodeElimination, FuseTIR, OptimizeLayoutTransform
from tvm.script import ir as I, tir as T, relax as R


def _run_pass_compare_output(Before, Expected):
    fused_mod = OptimizeLayoutTransform()(Before)
    if not relax.analysis.well_formed(fused_mod):
        print("IRModule is not well-formed")

    fused_mode = DeadCodeElimination()(fused_mod)
    if not relax.analysis.well_formed(fused_mod):
        print("IRModule is not well-formed")

    fused_mod = FuseTIR()(fused_mod)
    if not relax.analysis.well_formed(fused_mod):
        print("IRModule is not well-formed")

    tvm.ir.assert_structural_equal(Expected, fused_mod)


def test_optimize_transform_layout_pass_one_arg():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def relax_add_replacement(
            arg0: T.Buffer((4, 4), "float32"),
            arg1: T.Buffer((4, 4), "float32"),
            output: T.Buffer((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with T.block("root"):
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv, lv1),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv0: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv2, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                lv3: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    lv0, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv4: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv5 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv4, lv3),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv2_1: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                gv: R.Tensor((16,), dtype="float32") = lv2_1
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_add_replacement(
            arg0: T.Buffer((4, 4), "float32"),
            arg1: T.Buffer((4, 4), "float32"),
            output: T.Buffer((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with T.block("root"):
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv, lv1),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv4: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv5 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv4, lv2),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv2_1: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                gv: R.Tensor((16,), dtype="float32") = lv2_1
                R.output(gv)
            return gv

    _run_pass_compare_output(Before, Expected)


def test_optimize_transform_layout_pass_two_args():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def relax_add_replacement(
            arg0: T.Buffer((4, 4), "float32"),
            arg1: T.Buffer((4, 4), "float32"),
            output: T.Buffer((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with T.block("root"):
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"),
            y: R.Tensor((16,), dtype="float32"),
            z: R.Tensor((16,), dtype="float32"),
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    z, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv3 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv, lv1),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv4 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv, lv2),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv5: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv3, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                lv6: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv4, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                lv7: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv8: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    lv6, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv9 = R.call_tir(
                    Before.relax_add_replacement,
                    (lv7, lv8),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv10: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv9, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                gv: R.Tensor((16,), dtype="float32") = lv10
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_add_replacement(
            arg0: T.Buffer((4, 4), "float32"),
            arg1: T.Buffer((4, 4), "float32"),
            output: T.Buffer((4, 4), "float32"),
        ):
            T.func_attr({"operator_name": "relax.add"})
            # with T.block("root"):
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"),
            y: R.Tensor((16,), dtype="float32"),
            z: R.Tensor((16,), dtype="float32"),
        ) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    y, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv2: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    z, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                lv3 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv, lv1),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv4 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv, lv2),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv5 = R.call_tir(
                    Expected.relax_add_replacement,
                    (lv3, lv4),
                    out_sinfo=R.Tensor((4, 4), dtype="float32"),
                )
                lv6: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None
                )
                gv: R.Tensor((16,), dtype="float32") = lv6
                R.output(gv)
            return gv

    _run_pass_compare_output(Before, Expected)


if __name__ == "__main__":
    tvm.testing.main()
