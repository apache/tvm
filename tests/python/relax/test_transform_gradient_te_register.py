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
"""Unit tests for registering tir gradient functions in the gradient pass."""
import pytest

import tvm
import tvm.testing
from tvm import relax, tir
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import relax as R, tir as T, ir as I

from tvm.relax.training.utils import register_te_gradient
from tvm.relax.transform import Gradient


# Only run once in the whole test session
@pytest.fixture(scope="module")
def register_te_grads():
    # register the gradient function
    @register_te_gradient("f_mul_grad")
    def f_mul_grad(output_grad, src1, src2):
        def mul_grad_1(*idx):
            return src2[idx] * output_grad[idx]

        def mul_grad_2(*idx):
            return src1[idx] * output_grad[idx]

        return [
            tvm.te.compute(src1.shape, mul_grad_1, name="f_mul_grad_1"),
            tvm.te.compute(src2.shape, mul_grad_2, name="f_mul_grad_2"),
        ]

    # register the gradient function
    @register_te_gradient("f_mulk_grad")
    def f_mulk_grad(output_grad, src1, k):
        def mulk_grad(*idx):
            return output_grad[idx] * k

        return [
            tvm.te.compute(src1.shape, mulk_grad, name="f_mulk_grad"),
        ]


def get_expected_1():
    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def f_mul(A: T.Buffer((T.int64(5), T.int64(5)), "float32"), B: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mul_1: T.Buffer((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i0, v_i1])
                    T.writes(f_mul_1[v_i0, v_i1])
                    f_mul_1[v_i0, v_i1] = A[v_i0, v_i1] * B[v_i0, v_i1]

        @T.prim_func(private=True)
        def f_mul_grad(A: T.Buffer((T.int64(5), T.int64(5)), "float32"), B: T.Buffer((T.int64(5), T.int64(5)), "float32"), C: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mul_grad_1: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mul_grad_2: T.Buffer((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul_grad_1"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(C[v_i0, v_i1], A[v_i0, v_i1])
                    T.writes(f_mul_grad_1[v_i0, v_i1])
                    f_mul_grad_1[v_i0, v_i1] = C[v_i0, v_i1] * A[v_i0, v_i1]
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul_grad_2"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(B[v_i0, v_i1], A[v_i0, v_i1])
                    T.writes(f_mul_grad_2[v_i0, v_i1])
                    f_mul_grad_2[v_i0, v_i1] = B[v_i0, v_i1] * A[v_i0, v_i1]

        @R.function
        def main_adjoint(a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((5, 5), dtype="float32"), R.Tensor((5, 5), dtype="float32"))):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a, b), out_sinfo=R.Tensor((5, 5), dtype="float32"))
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([5, 5]))
                lv_1 = R.call_tir(cls.f_mul_grad, (lv_adjoint, a, b), out_sinfo=[R.Tensor((5, 5), dtype="float32"), R.Tensor((5, 5), dtype="float32")])
                a_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1[0]
                b_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1[1]
                a_adjoint_out: R.Tensor((5, 5), dtype="float32") = a_adjoint
                b_adjoint_out: R.Tensor((5, 5), dtype="float32") = b_adjoint
                R.output(gv, a_adjoint_out, b_adjoint_out)
            return (gv, (a_adjoint_out, b_adjoint_out))

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a, b), out_sinfo=R.Tensor((5, 5), dtype="float32"), te_grad_name="f_mul_grad")
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on
    return Expected


def test_emit_te(register_te_grads):
    # Build the target module using emit_te
    def f_mul(src1, src2):
        def mul(*idx):
            return src1[idx] * src2[idx]

        return tvm.te.compute(src1.shape, mul, name="f_mul")

    a = relax.Var("a", relax.TensorStructInfo([5, 5], "float32"))
    b = relax.Var("b", relax.TensorStructInfo([5, 5], "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [a, b]):
        with bb.dataflow():
            d = bb.emit(
                bb.call_te_with_grad(
                    f_mul, a, b, primfunc_name_hint="f_mul", te_grad_name="f_mul_grad"
                )
            )
            out = bb.emit_output(R.sum(d))
        bb.emit_func_output(out)

    Before = bb.get()
    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_1())


def test_call_tir(register_te_grads):
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def f_mul(A: T.Buffer((T.int64(5), T.int64(5)), "float32"), B: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mul_1: T.Buffer((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i0, v_i1])
                    T.writes(f_mul_1[v_i0, v_i1])
                    f_mul_1[v_i0, v_i1] = A[v_i0, v_i1] * B[v_i0, v_i1]

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a, b), out_sinfo=R.Tensor((5, 5), dtype="float32"), te_grad_name="f_mul_grad")
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: off

    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_1())


def get_expected_2():
    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def f_mul(A: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mul2: T.Buffer((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul2"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1])
                    T.writes(f_mul2[v_i0, v_i1])
                    f_mul2[v_i0, v_i1] = A[v_i0, v_i1] * T.float32(2)

        @T.prim_func(private=True)
        def f_mulk_grad(A: T.Buffer((T.int64(5), T.int64(5)), "float32"), B: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mulk_grad_1: T.Buffer((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mulk_grad"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1])
                    T.writes(f_mulk_grad_1[v_i0, v_i1])
                    f_mulk_grad_1[v_i0, v_i1] = A[v_i0, v_i1] * T.float32(2)

        @R.function
        def main_adjoint(a: R.Tensor((5, 5), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((5, 5), dtype="float32"))):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a,), out_sinfo=R.Tensor((5, 5), dtype="float32"))
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([5, 5]))
                lv_1 = R.call_tir(cls.f_mulk_grad, (lv_adjoint, a), out_sinfo=R.Tensor((5, 5), dtype="float32"))
                a_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1
                a_adjoint_out: R.Tensor((5, 5), dtype="float32") = a_adjoint
                R.output(gv, a_adjoint_out)
            return (gv, (a_adjoint_out,))

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a,), out_sinfo=R.Tensor((5, 5), dtype="float32"), te_grad_name="f_mulk_grad", te_grad_kwargs={"k": T.float32(2)})
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on
    return Expected


def test_emit_te_kwargs(register_te_grads):
    # Build the target module using emit_te
    def f_mul2(src):
        return tvm.te.compute(src.shape, lambda *idx: src[idx] * T.float32(2), name="f_mul2")

    a = relax.Var("a", relax.TensorStructInfo([5, 5], "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [a]):
        with bb.dataflow():
            d = bb.emit(
                bb.call_te_with_grad(
                    f_mul2,
                    a,
                    primfunc_name_hint="f_mul",
                    te_grad_name="f_mulk_grad",
                    te_grad_kwargs={"k": T.float32(2)},
                )
            )
            out = bb.emit_output(R.sum(d))
        bb.emit_func_output(out)

    Before = bb.get()
    After = Gradient("main")(Before)

    assert_structural_equal(After, get_expected_2())


def test_call_tir_kwargs(register_te_grads):
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def f_mul(A: T.Buffer((T.int64(5), T.int64(5)), "float32"), f_mul2: T.Buffer((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul2"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1])
                    T.writes(f_mul2[v_i0, v_i1])
                    f_mul2[v_i0, v_i1] = A[v_i0, v_i1] * T.float32(2)

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a,), out_sinfo=R.Tensor((5, 5), dtype="float32"), te_grad_name="f_mulk_grad", te_grad_kwargs={"k": T.float32(2)})
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_2())


def get_expected_3():
    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def f_mul(var_A: T.handle, var_B: T.handle, var_f_mul: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (n, n))
            B = T.match_buffer(var_B, (n, n))
            f_mul_1 = T.match_buffer(var_f_mul, (n, n))
            # with T.block("root"):
            for i0, i1 in T.grid(n, n):
                with T.block("f_mul"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i0, v_i1])
                    T.writes(f_mul_1[v_i0, v_i1])
                    f_mul_1[v_i0, v_i1] = A[v_i0, v_i1] * B[v_i0, v_i1]

        @T.prim_func(private=True)
        def f_mul_grad(var_A: T.handle, var_B: T.handle, var_C: T.handle, var_f_mul_grad_1: T.handle, var_f_mul_grad_2: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (n, n))
            B = T.match_buffer(var_B, (n, n))
            C = T.match_buffer(var_C, (n, n))
            f_mul_grad_1 = T.match_buffer(var_f_mul_grad_1, (n, n))
            f_mul_grad_2 = T.match_buffer(var_f_mul_grad_2, (n, n))
            # with T.block("root"):
            for i0, i1 in T.grid(n, n):
                with T.block("f_mul_grad_1"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(C[v_i0, v_i1], A[v_i0, v_i1])
                    T.writes(f_mul_grad_1[v_i0, v_i1])
                    f_mul_grad_1[v_i0, v_i1] = C[v_i0, v_i1] * A[v_i0, v_i1]
            for i0, i1 in T.grid(n, n):
                with T.block("f_mul_grad_2"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(B[v_i0, v_i1], A[v_i0, v_i1])
                    T.writes(f_mul_grad_2[v_i0, v_i1])
                    f_mul_grad_2[v_i0, v_i1] = B[v_i0, v_i1] * A[v_i0, v_i1]

        @R.function
        def main_adjoint(a: R.Tensor(("n", "n"), dtype="float32"), b: R.Tensor(("n", "n"), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor(("n", "n"), dtype="float32"), R.Tensor(("n", "n"), dtype="float32"))):
            n = T.int64()
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a, b), out_sinfo=R.Tensor((n, n), dtype="float32"))
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((n, n), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([n, n]))
                lv_1 = R.call_tir(cls.f_mul_grad, (lv_adjoint, a, b), out_sinfo=[R.Tensor((n, n), dtype="float32"), R.Tensor((n, n), dtype="float32")])
                a_adjoint: R.Tensor((n, n), dtype="float32") = lv_1[0]
                b_adjoint: R.Tensor((n, n), dtype="float32") = lv_1[1]
                a_adjoint_out: R.Tensor((n, n), dtype="float32") = a_adjoint
                b_adjoint_out: R.Tensor((n, n), dtype="float32") = b_adjoint
                R.output(gv, a_adjoint_out, b_adjoint_out)
            return (gv, (a_adjoint_out, b_adjoint_out))

        @R.function
        def main(a: R.Tensor(("n", "n"), dtype="float32"), b: R.Tensor(("n", "n"), dtype="float32")) -> R.Tensor((), dtype="float32"):
            n = T.int64()
            cls = Expected
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a, b), out_sinfo=R.Tensor((n, n), dtype="float32"), te_grad_name="f_mul_grad")
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on
    return Expected


def test_tir_var(register_te_grads):
    def f_mul(src1, src2):
        def mul(*idx):
            return src1[idx] * src2[idx]

        return tvm.te.compute(src1.shape, mul, name="f_mul")

    n = tir.Var("n", "int64")
    a = relax.Var("a", relax.TensorStructInfo([n, n], "float32"))
    b = relax.Var("b", relax.TensorStructInfo([n, n], "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [a, b]):
        with bb.dataflow():
            d = bb.emit(
                bb.call_te_with_grad(
                    f_mul, a, b, primfunc_name_hint="f_mul", te_grad_name="f_mul_grad"
                )
            )
            out = bb.emit_output(R.sum(d))
        bb.emit_func_output(out)

    Before = bb.get()
    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_3())
    assert relax.analysis.well_formed(After)


if __name__ == "__main__":
    tvm.testing.main()
