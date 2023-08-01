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
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T
import tvm.testing


##################### Binary arithmetic #####################


def test_add():
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
            gv = R.call_tir(Expected.add, (x, y), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def add(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_add_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.add(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.add, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def add(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = rxplaceholder[ax0, ax1] + T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_add_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.add, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def add(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_add: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = T.float32(1) + rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_add_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.add(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.add, (x, y), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def add(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_add: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_add = T.match_buffer(var_T_add, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_divide():
    # fmt: off
    @tvm.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.divide(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.divide, (x, y), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def divide(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_divide_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.divide(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.divide, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def divide(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_divide: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder[ax0, ax1] / T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_divide_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.divide(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.divide, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def divide(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_divide: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = T.float32(1) / rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_divide_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.divide(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.divide, (x, y), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def divide(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_divide: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_divide = T.match_buffer(var_T_divide, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_floor_divide():
    # fmt: off
    @tvm.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.floor_divide(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.floor_divide, (x, y), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def floor_divide(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_floor_divide: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_floor_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_floor_divide[ax0, ax1, ax2, ax3])
                    T_floor_divide[ax0, ax1, ax2, ax3] = T.floor(rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_floor_divide_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.floor_divide(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.floor_divide, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def floor_divide(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_floor_divide: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_floor_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_floor_divide[ax0, ax1])
                    T_floor_divide[ax0, ax1] = T.floor(rxplaceholder[ax0, ax1] / T.float32(1))
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_floor_divide_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.floor_divide(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.floor_divide, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def floor_divide(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_floor_divide: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_floor_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_floor_divide[ax0, ax1])
                    T_floor_divide[ax0, ax1] = T.floor(T.float32(1) / rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_floor_divide_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.floor_divide(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.floor_divide, (x, y), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def floor_divide(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_floor_divide: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_floor_divide = T.match_buffer(var_T_floor_divide, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_floor_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_floor_divide[ax0, ax1, ax2, ax3])
                    T_floor_divide[ax0, ax1, ax2, ax3] = T.floor(rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_multiply():
    # fmt: off
    @tvm.script.ir_module
    class Multiply:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.multiply(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.multiply, (x, y), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def multiply(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] * rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Multiply)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_multiply_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Multiply:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.multiply(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.multiply, (x, y), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def multiply(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_multiply: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_multiply = T.match_buffer(var_T_multiply, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] * rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Multiply)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_power():
    # fmt: off
    @tvm.script.ir_module
    class Power:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.power(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def power(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_power: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_power"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])
                    T.writes(T_power[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_power[v_ax0, v_ax1, v_ax2, v_ax3] = T.pow(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])

        @R.function
        def main(x: R.Tensor((1, 2, 3), dtype="float32"), y: R.Tensor((4, 3, 2, 1), dtype="float32")) -> R.Tensor((4, 3, 2, 3), dtype="float32"):
            gv = R.call_tir(Expected.power, (x, y), out_sinfo=R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

    # fmt: on

    mod = LegalizeOps()(Power)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_power_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Power:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.power(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def power(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_power: T.handle):
            T.func_attr({"tir.noalias": True})
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(1), c, d))
            a = T.int64()
            b = T.int64()
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (a, b, c, T.int64(1)))
            T_power = T.match_buffer(var_T_power, (a, b, c, d))
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(a, b, c, d):
                with T.block("T_power"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])
                    T.writes(T_power[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_power[v_ax0, v_ax1, v_ax2, v_ax3] = T.pow(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])

        @R.function
        def main(x: R.Tensor((1, "c", "d"), dtype="float32"), y: R.Tensor(("a", "b", "c", 1), dtype="float32")) -> R.Tensor(("a", "b", "c", "d"), dtype="float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.power, (x, y), out_sinfo=R.Tensor((a, b, c, d), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(Expected)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_subtract():
    # fmt: off
    @tvm.script.ir_module
    class Subtract:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.subtract(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.subtract, (x, y), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def subtract(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_subtract: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] - rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Subtract)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_subtract_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Subtract:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.subtract(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.subtract, (x, y), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def subtract(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_subtract: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_subtract = T.match_buffer(var_T_subtract, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] - rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Subtract)
    tvm.ir.assert_structural_equal(mod, Expected)


##################### Binary comparison #####################


def test_equal():
    # fmt: off
    @tvm.script.ir_module
    class Equal:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.equal, (x, y), R.Tensor((4, 3, 2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def equal(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_equal: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_equal[ax0, ax1, ax2, ax3])
                    T_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] == rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Equal)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_equal_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.equal(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.equal, (x,), R.Tensor((2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def equal(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_equal: T.Buffer((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_equal"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_equal[ax0, ax1])
                    T_equal[ax0, ax1] = rxplaceholder[ax0, ax1] == T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_equal_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.equal(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.equal, (x,), R.Tensor((2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def equal(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_equal: T.Buffer((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_equal"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_equal[ax0, ax1])
                    T_equal[ax0, ax1] = T.float32(1) == rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_equal_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Equal:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "bool") = R.equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.equal, (x, y), R.Tensor((a, b, c, d), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def equal(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_equal: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_equal = T.match_buffer(var_T_equal, [a, b, c, d], dtype="bool")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_equal[ax0, ax1, ax2, ax3])
                    T_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] == rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Equal)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_greater():
    # fmt: off
    @tvm.script.ir_module
    class Greater:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.greater(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.greater, (x, y), R.Tensor((4, 3, 2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def greater(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_greater: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_greater"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    T.writes(T_greater[ax0, ax1, ax2, ax3])
                    T_greater[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] < rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(Greater)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_greater_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.greater(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.greater, (x,), R.Tensor((2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def greater(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_greater: T.Buffer((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_greater"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_greater[ax0, ax1])
                    T_greater[ax0, ax1] = T.float32(1) < rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_greater_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.greater(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.greater, (x,), R.Tensor((2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def greater(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_greater: T.Buffer((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_greater"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_greater[ax0, ax1])
                    T_greater[ax0, ax1] = rxplaceholder[ax0, ax1] < T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_greater_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Greater:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "bool") = R.greater(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.greater, (x, y), R.Tensor((a, b, c, d), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def greater(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_greater: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_greater = T.match_buffer(var_T_greater, [a, b, c, d], dtype="bool")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_greater"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    T.writes(T_greater[ax0, ax1, ax2, ax3])
                    T_greater[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] < rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(Greater)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_greater_equal():
    # fmt: off
    @tvm.script.ir_module
    class GreaterEqual:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.greater_equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.greater_equal, (x, y), R.Tensor((4, 3, 2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def greater_equal(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_greater_equal: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_greater_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    T.writes(T_greater_equal[ax0, ax1, ax2, ax3])
                    T_greater_equal[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] <= rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(GreaterEqual)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_greater_equal_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class GreaterEqual:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "bool") = R.greater_equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.greater_equal, (x, y), R.Tensor((a, b, c, d), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def greater_equal(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_greater_equal: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_greater_equal = T.match_buffer(var_T_greater_equal, [a, b, c, d], dtype="bool")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_greater_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    T.writes(T_greater_equal[ax0, ax1, ax2, ax3])
                    T_greater_equal[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] <= rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(GreaterEqual)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_less():
    # fmt: off
    @tvm.script.ir_module
    class Less:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.less(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.less, (x, y), R.Tensor((4, 3, 2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def less(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_less: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_less"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_less[ax0, ax1, ax2, ax3])
                    T_less[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] < rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Less)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_less_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Less:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "bool") = R.less(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.less, (x, y), R.Tensor((a, b, c, d), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def less(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_less: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_less = T.match_buffer(var_T_less, [a, b, c, d], dtype="bool")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_less"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_less[ax0, ax1, ax2, ax3])
                    T_less[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] < rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Less)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_less_equal():
    # fmt: off
    @tvm.script.ir_module
    class LessEqual:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.less_equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.less_equal, (x, y), R.Tensor((4, 3, 2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def less_equal(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_less_equal: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_less_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_less_equal[ax0, ax1, ax2, ax3])
                    T_less_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] <= rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(LessEqual)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_less_equal_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.less_equal(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.less_equal, (x,), R.Tensor((2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def less_equal(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_less_equal: T.Buffer((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_less_equal"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_less_equal[ax0, ax1])
                    T_less_equal[ax0, ax1] = rxplaceholder[ax0, ax1] <= T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_less_equal_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.less_equal(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.less_equal, (x,), R.Tensor((2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def less_equal(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_less_equal: T.Buffer((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_less_equal"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_less_equal[ax0, ax1])
                    T_less_equal[ax0, ax1] = T.float32(1) <= rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_less_equal_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class LessEqual:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "bool") = R.less_equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.less_equal, (x, y), R.Tensor((a, b, c, d), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def less_equal(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_less_equal: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_less_equal = T.match_buffer(var_T_less_equal, [a, b, c, d], dtype="bool")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_less_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_less_equal[ax0, ax1, ax2, ax3])
                    T_less_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] <= rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(LessEqual)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_not_equal():
    # fmt: off
    @tvm.script.ir_module
    class NotEqual:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.not_equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.not_equal, (x, y), R.Tensor((4, 3, 2, 3), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def not_equal(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_not_equal: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_not_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_not_equal[ax0, ax1, ax2, ax3])
                    T_not_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] != rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(NotEqual)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_not_equal_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class NotEqual:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "bool") = R.not_equal(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "bool"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.not_equal, (x, y), R.Tensor((a, b, c, d), dtype="bool"))
            return gv

        @T.prim_func(private=True)
        def not_equal(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_not_equal: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_not_equal = T.match_buffer(var_T_not_equal, [a, b, c, d], dtype="bool")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_not_equal"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_not_equal[ax0, ax1, ax2, ax3])
                    T_not_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] != rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(NotEqual)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_maximum():
    # fmt: off
    @tvm.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.maximum(x, y)
            return gv


    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.maximum, (x, y), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def maximum(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_maximum: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_maximum"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_maximum[ax0, ax1, ax2, ax3])
                    T_maximum[ax0, ax1, ax2, ax3] = T.max(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_maximum_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.maximum(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.maximum, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def maximum(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_maximum: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_maximum"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_maximum[ax0, ax1])
                    T_maximum[ax0, ax1] = T.max(rxplaceholder[ax0, ax1], T.float32(1))
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_maximum_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.maximum(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.maximum, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def maximum(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_maximum: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_maximum"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_maximum[ax0, ax1])
                    T_maximum[ax0, ax1] = T.max(T.float32(1), rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_maximum_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.maximum(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.maximum, (x, y), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def maximum(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_maximum: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_maximum = T.match_buffer(var_T_maximum, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_maximum"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_maximum[ax0, ax1, ax2, ax3])
                    T_maximum[ax0, ax1, ax2, ax3] = T.max(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_minimum():
    # fmt: off
    @tvm.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.minimum(x, y)
            return gv


    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.minimum, (x, y), R.Tensor((4, 3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def minimum(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_minimum: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_minimum"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_minimum[ax0, ax1, ax2, ax3])
                    T_minimum[ax0, ax1, ax2, ax3] = T.min(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_minimum_with_arg0_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.minimum(x, R.const(1, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.minimum, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def minimum(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_minimum: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_minimum"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_minimum[ax0, ax1])
                    T_minimum[ax0, ax1] = T.min(rxplaceholder[ax0, ax1], T.float32(1))
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_minimum_with_arg1_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.minimum(R.const(1, "float32"), x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.minimum, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def minimum(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_minimum: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_minimum"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_minimum[ax0, ax1])
                    T_minimum[ax0, ax1] = T.min(T.float32(1), rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_minimum_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.minimum(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, "c", "d"), "float32"), y: R.Tensor(("a", "b", "c", 1), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.minimum, (x, y), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def minimum(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_minimum: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(1), c, d], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b, c, T.int64(1)], dtype="float32")
            T_minimum = T.match_buffer(var_T_minimum, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_minimum"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_minimum[ax0, ax1, ax2, ax3])
                    T_minimum[ax0, ax1, ax2, ax3] = T.min(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
