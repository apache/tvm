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
from tvm.relax.transform import LegalizeOps
from tvm.relax.transform.legalize_ops.common import register_legalize
from tvm.script import relax as R, tir as T
import tvm.testing


def test_customize_legalize():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.add(x, y)
            return gv


    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            cls = Expected
            gv = R.call_tir(cls.add, (y, x), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def add(rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] + rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call):
        from tvm import topi  # pylint: disable=import-outside-toplevel

        return bb.call_te(topi.add, call.args[1], call.args[0])

    mod = LegalizeOps({"relax.add": customize_legalize_add})(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_legalize_multiple_types_of_call():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @R.function
        def mul2(x: R.Tensor((3, 3), "float32")):
            gv = R.multiply(x, R.const(2.0, "float32"))
            return gv

        @T.prim_func(private=True)
        def identity(rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "float32"), T_id: T.Buffer((T.int64(3), T.int64(3)), "float32")):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_id[v_ax0, v_ax1])
                    T_id[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1]

        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            cls = Before
            gv: R.Tensor((3, 3), "float32") = cls.mul2(x)
            gv1 = R.call_tir(cls.identity, gv, R.Tensor((3, 3), dtype="float32"))
            gv2 = R.multiply(gv1, R.const(2.0, "float32"))
            return gv2

    @tvm.script.ir_module
    class Expected:
        @R.function
        def mul2(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.multiply, (x,), R.Tensor((3, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def identity(rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "float32"), T_id: T.Buffer((T.int64(3), T.int64(3)), "float32")):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_id[v_ax0, v_ax1])
                    T_id[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1]

        @T.prim_func(private=True)
        def multiply(rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(3), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * T.float32(2)

        @R.function
        def main(x1: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
            cls = Expected
            gv1: R.Tensor((3, 3), dtype="float32") = cls.mul2(x1)
            gv11 = R.call_tir(cls.identity, gv1, R.Tensor((3, 3), dtype="float32"))
            gv2 = R.call_tir(cls.multiply, (gv11,), R.Tensor((3, 3), dtype="float32"))
            return gv2
    # fmt: on

    After = LegalizeOps()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_can_not_legalize():
    # case 1: does't have legalization
    add_legalize = tvm.ir.Op.get("relax.add").get_attr("FLegalize")
    # reset it for test
    tvm.ir.Op.get("relax.add").reset_attr("FLegalize")

    # fmt: off
    @tvm.script.ir_module
    class Before0:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            gv: R.Tensor((3, 3), "float32") = R.add(x, x)
            return gv
    # fmt: on
    After0 = LegalizeOps()(Before0)
    tvm.ir.assert_structural_equal(After0, Before0)

    register_legalize("relax.add", add_legalize)

    # case 2: don't know all shape
    s = relax.Var("s", relax.ShapeStructInfo((3, 3)))
    x = relax.Var("x", relax.TensorStructInfo((3, 3), "float32"))
    y = relax.Var("y", relax.TensorStructInfo(s, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            gv = bb.emit_output(R.add(x, y))
        bb.emit_func_output(gv)
    Before1 = bb.get()
    After1 = LegalizeOps()(Before1)
    tvm.ir.assert_structural_equal(After1, Before1)


def test_legalize_scalar_data_type_preserve():
    # fmt: off
    @tvm.script.ir_module
    class Before0:
        @R.function
        def main(x: R.Tensor((3, 3), "float16")):
            gv: R.Tensor((3, 3), "float16") = R.multiply(x, R.const(1.14514, "float16"))
            return gv

    @tvm.script.ir_module
    class Before1:
        @R.function
        def main(x: R.Tensor((3, 3), "uint8")):
            gv: R.Tensor((3, 3), "uint8") = R.multiply(x, R.const(2, "uint8"))
            return gv

    @tvm.script.ir_module
    class Before2:
        @R.function
        def main(x: R.Tensor((3, 3), "bool")):
            gv: R.Tensor((3, 3), "bool") = R.equal(x, R.const(True, "bool"))
            return gv

    @tvm.script.ir_module
    class Expected0:
        @T.prim_func(private=True)
        def multiply(
            rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "float16"),
            T_multiply: T.Buffer((T.int64(3), T.int64(3)), "float16"),
        ):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * T.float16(
                        1.1455078125
                    )

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float16")) -> R.Tensor((3, 3), dtype="float16"):
            cls = Expected0
            gv = R.call_tir(cls.multiply, (x,), out_sinfo=R.Tensor((3, 3), dtype="float16"))
            return gv

    @tvm.script.ir_module
    class Expected1:
        @T.prim_func(private=True)
        def multiply(
            rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "uint8"),
            T_multiply: T.Buffer((T.int64(3), T.int64(3)), "uint8"),
        ):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * T.uint8(2)

        @R.function
        def main(x: R.Tensor((3, 3), dtype="uint8")) -> R.Tensor((3, 3), dtype="uint8"):
            cls = Expected1
            gv = R.call_tir(cls.multiply, (x,), out_sinfo=R.Tensor((3, 3), dtype="uint8"))
            return gv

    @tvm.script.ir_module
    class Expected2:
        @T.prim_func(private=True)
        def equal(
            rxplaceholder: T.Buffer((T.int64(3), T.int64(3)), "bool"),
            T_equal: T.Buffer((T.int64(3), T.int64(3)), "bool"),
        ):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_equal"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_equal[v_ax0, v_ax1])
                    T_equal[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] == tvm.tir.const(True, "bool")

        @R.function
        def main(x: R.Tensor((3, 3), dtype="bool")) -> R.Tensor((3, 3), dtype="bool"):
            cls = Expected2
            gv = R.call_tir(cls.equal, (x,), out_sinfo=R.Tensor((3, 3), dtype="bool"))
            return gv
    # fmt: on

    After0 = LegalizeOps()(Before0)
    tvm.ir.assert_structural_equal(After0, Expected0)
    After1 = LegalizeOps()(Before1)
    tvm.ir.assert_structural_equal(After1, Expected1)
    After2 = LegalizeOps()(Before2)
    tvm.ir.assert_structural_equal(After2, Expected2)


if __name__ == "__main__":
    tvm.testing.main()
