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

from tvm import relax
from tvm.script import tir as T, ir as I, relax as R
from tvm.tir import IndexMap

kOperatorName = "operator_name"


def _check(
    before, expected, operator_name, replacement_primfunc, layout_changes, axis_separator=None
):
    after = relax.transform.AlterOpImpl(
        {operator_name: replacement_primfunc},
        {operator_name: layout_changes},
        {operator_name: axis_separator},
    )(before)
    after = relax.transform.DeadCodeElimination()(after)
    tvm.ir.assert_structural_equal(after, expected)


def test_single_output():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def add(arg0: T.Buffer((16,), "float32"), arg1: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
            T.func_attr({"operator_name": "relax.add"})
            for ax0 in range(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(16, ax0)
                    T.reads(arg0[v_ax0], arg1[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = arg0[v_ax0] + arg1[v_ax0]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv = R.call_tir(Before.add, (x, y), out_sinfo=R.Tensor((16,), dtype="float32"))
                gv: R.Tensor((16,), dtype="float32") = lv
                R.output(gv)
            return gv
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_add_replacement(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output: T.Buffer((4, 4), "float32")):
            T.func_attr({"operator_name": "relax.add"})
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(x, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(y, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv2 = R.call_tir(Expected.relax_add_replacement, (lv, lv1), out_sinfo=R.Tensor((4, 4), dtype="float32"))
                lv_1: R.Tensor((16,), dtype="float32") = R.layout_transform(lv2, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None)
                gv: R.Tensor((16,), dtype="float32") = lv_1
                R.output(gv)
            return gv

    @T.prim_func(private=True)
    def add_2d(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output: T.Buffer((4, 4), "float32")):
        for ax0, ax1 in T.grid(4, 4):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                T.writes(output[v_ax0, v_ax1])
                output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
    # fmt: on
    index_map = lambda i: (i // 4, i % 4)
    _check(
        Before,
        Expected,
        operator_name="relax.add",
        replacement_primfunc=add_2d,
        layout_changes=[index_map, index_map, index_map],
    )


def test_empty_layout_changes():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def mul_by_2(arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
            T.func_attr({"operator_name": "relax.mul_by_2"})
            for ax0 in range(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(16, ax0)
                    T.reads(arg0[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = arg0[v_ax0] * T.float32(2)

        @R.function
        def main(x: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv = R.call_tir(Before.mul_by_2, (x,), out_sinfo=R.Tensor((16,), dtype="float32"))
                gv: R.Tensor((16,), dtype="float32") = lv
                R.output(gv)
            return gv
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_mul_by_2_replacement(arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
            T.func_attr({"operator_name": "relax.mul_by_2"})
            for ax0 in range(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(16, ax0)
                    T.reads(arg0[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = arg0[v_ax0] + arg0[v_ax0]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv = R.call_tir(Expected.relax_mul_by_2_replacement, (x,), out_sinfo=R.Tensor((16,), dtype="float32"))
                gv: R.Tensor((16,), dtype="float32") = lv
                R.output(gv)
            return gv

    @T.prim_func(private=True)
    def add_x_x(arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
        T.func_attr({"operator_name": "relax.mul_by_2"})
        for ax0 in range(16):
            with T.block("T_add"):
                v_ax0 = T.axis.spatial(16, ax0)
                T.reads(arg0[v_ax0])
                T.writes(output[v_ax0])
                output[v_ax0] = arg0[v_ax0] + arg0[v_ax0]
    # fmt: on
    _check(
        Before,
        Expected,
        operator_name="relax.mul_by_2",
        replacement_primfunc=add_x_x,
        layout_changes=[],
    )


def test_multiple_outputs():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def some_op(arg0: T.Buffer((16,), "float32"), arg1: T.Buffer((16,), "float32"), output0: T.Buffer((16,), "float32"), output1: T.Buffer((16,), "float32")):
            T.func_attr({"operator_name": "relax.some_op"})
            for ax0 in range(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(16, ax0)
                    T.reads(arg0[v_ax0], arg1[v_ax0])
                    T.writes(output0[v_ax0], output1[v_ax0])
                    output0[v_ax0] = arg0[v_ax0] + arg1[v_ax0]
                    output1[v_ax0] = arg0[v_ax0] - arg1[v_ax0]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")):
            with R.dataflow():
                gv = R.call_tir(Before.some_op, (x, y), out_sinfo=[R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")])
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_some_op_replacement(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output0: T.Buffer((4, 4), "float32"), output1: T.Buffer((4, 4), "float32")):
            T.func_attr({"operator_name": "relax.some_op"})
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output0[v_ax0, v_ax1], output1[v_ax0, v_ax1])
                    output0[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
                    output1[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] - arg1[v_ax0, v_ax1]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(x, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(y, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv2 = R.call_tir(Expected.relax_some_op_replacement, (lv, lv1), out_sinfo=[R.Tensor((4, 4), dtype="float32"), R.Tensor((4, 4), dtype="float32")])
                lv3: R.Tensor((4, 4), dtype="float32") = lv2[0]
                lv4: R.Tensor((16,), dtype="float32") = R.layout_transform(lv3, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None)
                lv5: R.Tensor((4, 4), dtype="float32") = lv2[1]
                lv6: R.Tensor((16,), dtype="float32") = R.layout_transform(lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None)
                gv: R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")) = (lv4, lv6)
                R.output(gv)
            return gv

    @T.prim_func(private=True)
    def some_op_2d(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output0: T.Buffer((4, 4), "float32"), output1: T.Buffer((4, 4), "float32")):
        for ax0, ax1 in T.grid(4, 4):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                T.writes(output0[v_ax0, v_ax1], output1[v_ax0, v_ax1])
                output0[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
                output1[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] - arg1[v_ax0, v_ax1]
    # fmt: on

    index_map = lambda i: (i // 4, i % 4)
    _check(
        Before,
        Expected,
        operator_name="relax.some_op",
        replacement_primfunc=some_op_2d,
        layout_changes=[index_map, index_map, index_map, index_map],
    )


def test_multiple_outputs_with_axis_sep():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def some_op(arg0: T.Buffer((16,), "float32"), arg1: T.Buffer((16,), "float32"), output0: T.Buffer((16,), "float32"), output1: T.Buffer((16,), "float32")):
            T.func_attr({"operator_name": "relax.some_op"})
            for ax0 in range(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(16, ax0)
                    T.reads(arg0[v_ax0], arg1[v_ax0])
                    T.writes(output0[v_ax0], output1[v_ax0])
                    output0[v_ax0] = arg0[v_ax0] + arg1[v_ax0]
                    output1[v_ax0] = arg0[v_ax0] - arg1[v_ax0]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")):
            with R.dataflow():
                gv = R.call_tir(Before.some_op, (x, y), out_sinfo=[R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")])
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_some_op_replacement(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output0: T.Buffer((4, 4), "float32"), output1: T.Buffer((4, 4), "float32")):
            T.func_attr({"operator_name": "relax.some_op"})
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output0[v_ax0, v_ax1], output1[v_ax0, v_ax1])
                    output0[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
                    output1[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] - arg1[v_ax0, v_ax1]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(x, index_map=lambda i: (i // 4, i % 4), pad_value=None, axis_separators=[1])
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(y, index_map=lambda i: (i // 4, i % 4), pad_value=None, axis_separators=[1])
                lv2 = R.call_tir(Expected.relax_some_op_replacement, (lv, lv1), out_sinfo=[R.Tensor((4, 4), dtype="float32"), R.Tensor((4, 4), dtype="float32")])
                lv3: R.Tensor((4, 4), dtype="float32") = lv2[0]
                lv4: R.Tensor((16,), dtype="float32") = R.layout_transform(lv3, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None, axis_separators=[1])
                lv5: R.Tensor((4, 4), dtype="float32") = lv2[1]
                lv6: R.Tensor((16,), dtype="float32") = R.layout_transform(lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None, axis_separators=[1])
                gv: R.Tuple(R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")) = (lv4, lv6)
                R.output(gv)
            return gv

    @T.prim_func(private=True)
    def some_op_2d(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output0: T.Buffer((4, 4), "float32"), output1: T.Buffer((4, 4), "float32")):
        for ax0, ax1 in T.grid(4, 4):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                T.writes(output0[v_ax0, v_ax1], output1[v_ax0, v_ax1])
                output0[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
                output1[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] - arg1[v_ax0, v_ax1]
    # fmt: on

    index_map, axis_sep = IndexMap.from_func_with_separators(
        lambda i: (i // 4, IndexMap.AXIS_SEPARATOR, i % 4)
    )
    _check(
        Before,
        Expected,
        operator_name="relax.some_op",
        replacement_primfunc=some_op_2d,
        layout_changes=[index_map, index_map, index_map, index_map],
        axis_separator=[axis_sep, axis_sep, axis_sep, axis_sep],
    )


def test_supported_implicit_padding():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((14,), dtype="float32")) -> R.Tensor((14,), dtype="float32"):
            with R.dataflow():
                lv = R.call_tir(Before.relu, (x,), out_sinfo=R.Tensor((14,), dtype="float32"))
                gv: R.Tensor((14,), dtype="float32") = lv
                R.output(gv)
            return gv

        @T.prim_func(private=True)
        def relu(arg0: T.Buffer((14,), "float32"), output: T.Buffer((14,), "float32")):
            T.func_attr({"operator_name": "relax.relu"})
            for ax0 in T.grid(14):
                with T.block("T_add"):
                    v_ax0 = T.axis.remap("S", [ax0])
                    T.reads(arg0[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((14,), dtype="float32")) -> R.Tensor((14,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(lambda i: (i % 16,)),
                    pad_value=None,
                    axis_separators=[],
                )
                lv1 = R.call_tir(
                    Expected.relax_relu_replacement,
                    (lv,),
                    out_sinfo=R.Tensor((16,), dtype="float32"),
                )
                lv2: R.Tensor((16,), dtype="float32") = R.layout_transform(
                    lv1,
                    index_map=T.index_map(lambda axis0: (axis0,)),
                    pad_value=None,
                    axis_separators=[],
                )
                lv_1 = R.call_tir(
                    Expected.remove_pad, (lv2,), out_sinfo=R.Tensor((14,), dtype="float32")
                )
                gv: R.Tensor((14,), dtype="float32") = lv_1
                R.output(gv)
            return gv

        @T.prim_func(private=True)
        def relax_relu_replacement(
            arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")
        ):
            T.func_attr({"operator_name": "relax.relu"})
            # with T.block("root"):
            for ax0 in range(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(16, ax0)
                    T.reads(arg0[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

        @T.prim_func(private=True)
        def remove_pad(var_input: T.handle, var_output: T.handle):
            T.func_attr({"operator_name": "remove_pad", "tir.noalias": T.bool(True)})
            p0 = T.int64()
            input = T.match_buffer(var_input, (p0,))
            i0 = T.int64()
            output = T.match_buffer(var_output, (i0,))
            # with T.block("root"):
            for ax0 in range(i0):
                with T.block("output"):
                    v_ax0 = T.axis.spatial(i0, ax0)
                    T.reads(input[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = input[v_ax0]

    @T.prim_func(private=True)
    def relu_pad(arg0: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
        for ax0 in T.grid(16):
            with T.block("T_add"):
                v_ax0 = T.axis.remap("S", [ax0])
                T.reads(arg0[v_ax0])
                T.writes(output[v_ax0])
                output[v_ax0] = T.max(arg0[v_ax0], T.float32(0))

    # introduces implicit padding for shape (14,)
    index_map = lambda i: (i % 16)
    operator_name = "relax.relu"
    _check(
        Before,
        Expected,
        operator_name="relax.relu",
        replacement_primfunc=relu_pad,
        layout_changes=[index_map, index_map],
    )


def test_multiple_call_sites():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def add(arg0: T.Buffer((16,), "float32"), arg1: T.Buffer((16,), "float32"), output: T.Buffer((16,), "float32")):
            T.func_attr({"operator_name": "relax.add"})
            for ax0 in range(16):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(16, ax0)
                    T.reads(arg0[v_ax0], arg1[v_ax0])
                    T.writes(output[v_ax0])
                    output[v_ax0] = arg0[v_ax0] + arg1[v_ax0]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv0 = R.call_tir(Before.add, (x, y), out_sinfo=R.Tensor((16,), dtype="float32"))
                lv1 = R.nn.relu(lv0)
                lv2 = R.call_tir(Before.add, (lv0, lv1), out_sinfo=R.Tensor((16,), dtype="float32"))
                gv: R.Tensor((16,), dtype="float32") = lv2
                R.output(gv)
            return gv
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_add_replacement(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output: T.Buffer((4, 4), "float32")):
            T.func_attr({"operator_name": "relax.add"})
            # with T.block("root"):
            for ax0, ax1 in T.grid(4, 4):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

        @R.function
        def main(x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(x, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv1: R.Tensor((4, 4), dtype="float32") = R.layout_transform(y, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv2 = R.call_tir(Expected.relax_add_replacement, (lv, lv1), out_sinfo=R.Tensor((4, 4), dtype="float32"))
                lv0: R.Tensor((16,), dtype="float32") = R.layout_transform(lv2, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None)
                lv1_1: R.Tensor((16,), dtype="float32") = R.nn.relu(lv0)
                lv3: R.Tensor((4, 4), dtype="float32") = R.layout_transform(lv0, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv4: R.Tensor((4, 4), dtype="float32") = R.layout_transform(lv1_1, index_map=lambda i: (i // 4, i % 4), pad_value=None)
                lv5 = R.call_tir(Expected.relax_add_replacement, (lv3, lv4), out_sinfo=R.Tensor((4, 4), dtype="float32"))
                lv2_1: R.Tensor((16,), dtype="float32") = R.layout_transform(lv5, index_map=lambda axis0, axis1: (axis0 * 4 + axis1,), pad_value=None)
                gv: R.Tensor((16,), dtype="float32") = lv2_1
                R.output(gv)
            return gv
    @T.prim_func(private=True)
    def add_2d(arg0: T.Buffer((4, 4), "float32"), arg1: T.Buffer((4, 4), "float32"), output: T.Buffer((4, 4), "float32")):
        for ax0, ax1 in T.grid(4, 4):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                T.writes(output[v_ax0, v_ax1])
                output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]
    # fmt: on
    index_map = lambda i: (i // 4, i % 4)
    _check(
        Before,
        Expected,
        operator_name="relax.add",
        replacement_primfunc=add_2d,
        layout_changes=[index_map, index_map, index_map],
    )


def test_reshape():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def reshape(
            A: T.Buffer((T.int64(850), T.int64(2048)), "float16"),
            T_reshape: T.Buffer((T.int64(850), T.int64(1), T.int64(2048)), "float16"),
        ):
            T.func_attr({"operator_name": "relax.reshape"})
            for ax0, ax1, ax2 in T.grid(T.int64(850), T.int64(1), T.int64(2048)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        A[
                            (v_ax2 // T.int64(2048) + v_ax0 + v_ax1) % T.int64(850),
                            v_ax2 % T.int64(2048),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = A[
                        (v_ax2 // T.int64(2048) + v_ax0 + v_ax1) % T.int64(850),
                        v_ax2 % T.int64(2048),
                    ]

        @R.function
        def main(
            x: R.Tensor((850, 2048), dtype="float16")
        ) -> R.Tensor((850, 1, 2048), dtype="float16"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir(
                    cls.reshape, (x,), out_sinfo=R.Tensor((850, 1, 2048), dtype="float16")
                )
                gv: R.Tensor((850, 1, 2048), dtype="float16") = lv
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def relax_reshape_replacement(
            A: T.Buffer((T.int64(850), T.int64(2), T.int64(1024)), "float16"),
            T_reshape: T.Buffer((T.int64(850), T.int64(1), T.int64(2048)), "float16"),
        ):
            T.func_attr({"operator_name": "relax.reshape"})
            for ax0, ax1, ax2 in T.grid(T.int64(850), T.int64(1), T.int64(2048)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(A[v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = A[
                        v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)
                    ]

        @R.function
        def main(
            x: R.Tensor((850, 2048), dtype="float16")
        ) -> R.Tensor((850, 1, 2048), dtype="float16"):
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((850, 2, 1024), dtype="float16") = R.layout_transform(
                    x,
                    index_map=T.index_map(lambda i, j: (i, j // 1024, j % 1024)),
                    pad_value=None,
                    axis_separators=[],
                )
                lv_1 = R.call_tir(
                    cls.relax_reshape_replacement,
                    (lv,),
                    out_sinfo=R.Tensor((850, 1, 2048), dtype="float16"),
                )
                gv: R.Tensor((850, 1, 2048), dtype="float16") = lv_1
                R.output(gv)
            return gv

    @T.prim_func(private=True)
    def reshape_new(
        A: T.Buffer((T.int64(850), T.int64(2), T.int64(1024)), "float16"),
        T_reshape: T.Buffer((T.int64(850), T.int64(1), T.int64(2048)), "float16"),
    ):
        for ax0, ax1, ax2 in T.grid(T.int64(850), T.int64(1), T.int64(2048)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[
                    v_ax0, v_ax2 // T.int64(1024), v_ax2 % T.int64(1024)
                ]

    # fmt: on
    index_map = lambda i, j: (i, j // 1024, j % 1024)
    _check(
        Before,
        Expected,
        operator_name="relax.reshape",
        replacement_primfunc=reshape_new,
        layout_changes=[index_map, None],
    )


if __name__ == "__main__":
    tvm.testing.main()
