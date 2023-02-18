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
import tvm.testing
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R
from tvm.script import tir as T


def test_abs():
    # fmt: off
    @tvm.script.ir_module
    class Abs:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.abs(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            gv = R.call_tir(tir_abs, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_abs(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32"),):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.fabs(rxplaceholder[v_i0, v_i1], dtype="float32")
    # fmt: on

    mod = LegalizeOps()(Abs)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_abs_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Abs:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.abs(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_abs, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_abs(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.fabs(rxplaceholder[v_i0, v_i1], dtype="float32")
    # fmt: on

    mod = LegalizeOps()(Abs)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cos():
    # fmt: off
    @tvm.script.ir_module
    class Cos:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.cos(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(tir_cos, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_cos(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.cos(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Cos)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cos_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Cos:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.cos(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_cos, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_cos(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.cos(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Cos)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_exp():
    # fmt: off
    @tvm.script.ir_module
    class Exp:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.exp(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            gv = R.call_tir(tir_exp, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32"),):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.exp(rxplaceholder[v_i0, v_i1], dtype="float32")
    # fmt: on

    mod = LegalizeOps()(Exp)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_exp_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Exp:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.exp(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_exp, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.exp(rxplaceholder[v_i0, v_i1], dtype="float32")
    # fmt: on

    mod = LegalizeOps()(Exp)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_log():
    # fmt: off
    @tvm.script.ir_module
    class Log:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.log(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(tir_log, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_log(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.log(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Log)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_log_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Log:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.log(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_log, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_log(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.log(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Log)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_negative():
    # fmt: off
    @tvm.script.ir_module
    class Negative:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.negative(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(tir_negative, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_negative(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = rxplaceholder[i0_1, i1_1] * T.float32(-1)
    # fmt: on

    mod = LegalizeOps()(Negative)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_negative_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Negative:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.negative(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_negative, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_negative(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = rxplaceholder[i0_1, i1_1] * T.float32(-1)
    # fmt: on

    mod = LegalizeOps()(Negative)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sigmoid():
    # fmt: off
    @tvm.script.ir_module
    class Sigmoid:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.sigmoid(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(tir_sigmoid, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_sigmoid(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Sigmoid)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sigmoid_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Sigmoid:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.sigmoid(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_sigmoid, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_sigmoid(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Sigmoid)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sin():
    # fmt: off
    @tvm.script.ir_module
    class Sin:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.sin(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(tir_sin, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_sin(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sin(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Sin)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sin_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Sin:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.sin(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_sin, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_sin(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sin(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Sin)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sqrt():
    # fmt: off
    @tvm.script.ir_module
    class Sqrt:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.sqrt(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(tir_sqrt, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_sqrt(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sqrt(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Sqrt)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sqrt_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Sqrt:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.sqrt(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_sqrt, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_sqrt(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sqrt(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Sqrt)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tanh():
    # fmt: off
    @tvm.script.ir_module
    class Tanh:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.tanh(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(tir_tanh, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def tir_tanh(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.tanh(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Tanh)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tanh_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Tanh:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.tanh(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_tanh, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_tanh(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.tanh(rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Tanh)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_clip_symbolic():
    @tvm.script.ir_module
    class Clip:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.clip(x, 5, 8)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(tir_clip, (x,), out_sinfo=R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def tir_clip(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    compute[v_i0, v_i1] = T.max(
                        T.min(rxplaceholder[v_i0, v_i1], T.float32(8)), T.float32(5)
                    )

    mod = LegalizeOps()(Clip)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
