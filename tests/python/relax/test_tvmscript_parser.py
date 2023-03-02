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

from typing import Optional, Union

import pytest
import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax, tir, topi
from tvm.ir import DummyGlobalInfo
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]] = None,
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.from_source(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_simple_func():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
        R.func_attr({"Primitive": 1})
        gv0 = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
        return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), attrs={"Primitive": 1}):
        out = bb.emit(relax.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32")))
        bb.emit_func_output(out)

    _check(foo, bb.get()["foo"])


def test_error_report():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            # error: a = b = c is not allowed.
            gv0 = gv1 = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
            return gv0


def test_mismatch_cast_dims_and_ndim():
    with pytest.raises(Exception):

        @R.function
        def f(
            x: R.Tensor((2, 3), "float32", ndim=3)
        ):  # error: ndim and the shape dims are mismatch
            return x


def test_unexpected_num_kw_args():
    with pytest.raises(Exception):

        @R.function
        def f(x: R.Tensor(dtype="float32", ndim=1, foo=2)):  # error: unexpected kw args foo
            return x


def test_unexpected_ndim():
    with pytest.raises(Exception):

        @R.function
        # error: dim is expected to be non-negative int or -1 for unknown
        def f(x: R.Tensor(dtype="float32", ndim=-2)):
            return x


def test_unexpected_ndim_type():
    with pytest.raises(Exception):

        @R.function
        def f(x: R.Tensor(dtype="float32", ndim="1")):  # error: dim is expected to be int
            return x


def test_unexpected_tir_cast_args():

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(("m",), "float32")):
            m = T.int64()
            # tir.cast expects 2 arguments, but got 3
            return R.call_tir("foo", (x,), R.Tensor((T.cast("int32", m, 1),), dtype="float32"))


def test_unexpected_tir_max_args():

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(("m", "n"), "float32")):
            m = T.int64()
            # tir.max expects 2 arguments, but got 1
            return relax.call_tir("foo", (x,), R.Tensor((T.max(m),), dtype="float32"))


def test_func_type_annotation_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x, y):  # error: the parameter type annotation is missing
            z = R.add(x, y)
            y = z
            return y


def test_if_mismatch_var_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                z = R.add(w, w)  # error: The binding var is expected to `y`
            return z


def test_unassigned_call_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor):
            R.add(x, x)
            return x


def test_simple_module():
    @I.ir_module
    class TestModule:
        @T.prim_func
        def tir_func(
            x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
            y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            # TODO(Siyuan): Need to change to `TestModule.tir_func`
            gv0 = R.call_tir(tir_func, x, R.Tensor((128, 128), dtype="float32"))
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)

    _check(TestModule, bb.get())


def test_module_with_attr_and_global_info():
    @I.ir_module
    class TestModule:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "dummy": [
                    I.dummy_global_info(),  # dummy[0]
                    I.dummy_global_info(),  # dummy[1]
                ]
            }
        )

        @T.prim_func
        def tir_func(
            x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
            y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            # TODO(Siyuan): Need to change to `TestModule.tir_func`
            gv0 = R.call_tir(tir_func, x, R.Tensor((128, 128), dtype="float32"))
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)
    mod = bb.get()
    mod.update_global_info("dummy", [DummyGlobalInfo(), DummyGlobalInfo()])
    mod = mod.with_attr("attr", tvm.tir.IntImm("int32", 10))
    _check(TestModule, mod)


def test_relax_tensor_op():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")) -> R.Tensor((4, 4), "float32"):
        y = R.add(x, x)
        z = R.multiply(x, y)
        return z

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.op.add(x, x))
        z = bb.emit(relax.op.multiply(x, y))
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_relax_base_op():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        alloc = R.builtin.alloc_tensor(R.shape([4, 4]), runtime_device_index=0, dtype="float32")
        shape = R.shape_of(alloc)
        return shape

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        alloc = bb.emit(relax.op.builtin.alloc_tensor(relax.ShapeExpr((4, 4)), "float32", 0))
        shape = bb.emit(relax.op.shape_of(alloc))
        bb.emit_func_output(shape)

    _check(foo, bb.get()["foo"])


def test_symbolic_shape():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
        m = T.int64()
        n = T.int64()
        gv0 = R.call_tir("extern_func", x, R.Tensor((m, n), dtype="float32"))
        return gv0

    @R.function
    def bar(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
        m = T.int64()
        n = T.int64()
        gv0 = R.call_tir("extern_func", x, R.Tensor((m, n), dtype="float32"))
        return gv0

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def mismatch_dtype(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
            m = T.int64()
            n = T.int32()  # The shape dtype should be int64
            gv0 = R.call_tir("extern_func", x, R.Tensor((m, n), dtype="float32"))
            return gv0

    def _expected(name: str):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = relax.Var("x", R.Tensor([m, n], "float32"))
        bb = relax.BlockBuilder()
        with bb.function(name, (x,)):
            out = bb.emit(relax.call_tir("extern_func", x, R.Tensor((m, n), dtype="float32")))
            bb.emit_func_output(out)
        return bb.get()[name]

    _check(foo, _expected("foo"))
    _check(bar, _expected("bar"))


def test_shadowing():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        y = R.add(x, x)
        z = R.multiply(x, y)
        y = R.add(x, y)
        y = z
        y = R.multiply(y, x)
        z = y
        return z

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.op.add(x, x))
        z = bb.emit(relax.op.multiply(x, y))
        y = bb.emit(relax.op.add(x, y))
        y = bb.emit(z)
        y = bb.emit(relax.op.multiply(y, x))
        z = bb.emit(y)
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_match_cast():
    @R.function
    def foo(x: R.Tensor("float32"), y: R.Tensor("float32")):
        m = T.int64()
        n = T.int64()
        x0 = R.match_cast(x, R.Tensor([m], "float32"))
        with R.dataflow():
            y0 = R.match_cast(y, R.Tensor([n], "float32"))
            gv = y0
            R.output(gv)
        return (x0, R.shape([m, n * 2]))

    x = relax.Var("x", R.Tensor("float32"))
    y = relax.Var("y", R.Tensor("float32"))
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    y2 = relax.Var("y", R.Tensor([n], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        x0 = bb.match_cast(x, R.Tensor([m], "float32"))
        with bb.dataflow():
            y0 = bb.match_cast(y, R.Tensor([n], "float32"))
            bb.emit_output(y0)
        bb.emit_func_output(relax.Tuple([x0, relax.ShapeExpr([m, n * 2])]))

    _check(foo, bb.get()["foo"])


def test_tuple_return():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        gv0 = R.call_tir("extern_func_0", x, R.Tensor((4, 4), dtype="float32"))
        gv1 = R.call_tir("extern_func_1", x, R.Tensor((4, 4), dtype="float32"))
        return (gv0, gv1)

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        gv0 = bb.emit(relax.call_tir("extern_func_0", x, R.Tensor((4, 4), dtype="float32")))
        gv1 = bb.emit(relax.call_tir("extern_func_1", x, R.Tensor((4, 4), dtype="float32")))
        bb.emit_func_output(relax.Tuple((gv0, gv1)))

    _check(foo, bb.get()["foo"])


def test_tuple_return_2():
    @R.function
    def foo(x: R.Tensor("float32", ndim=2)):
        n, m = T.int64(), T.int64()
        x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
        return (x0, R.shape([n + 1, m, 1]))

    x = relax.Var("x", R.Tensor("float32", ndim=2))
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        x0 = bb.match_cast(x, R.Tensor((n, m), "float32"))
        bb.emit_func_output(relax.Tuple([x0, relax.ShapeExpr([n + 1, m, 1])]))

    _check(foo, bb.get()["foo"])


def test_tuple_binding():
    @R.function
    def foo(x: R.Tensor("float32", ndim=2)):
        n, m = T.int64(), T.int64()
        x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
        t0 = (x, x0)
        t1 = (x, R.shape([n, m]), t0)
        return t1

    x = relax.Var("x", R.Tensor("float32", ndim=2))
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        x0 = bb.match_cast(x, R.Tensor((n, m), "float32"))
        t0 = bb.emit(relax.Tuple([x, x0]))
        t1 = bb.emit(relax.Tuple([x, relax.ShapeExpr([n, m]), t0]))
        bb.emit_func_output(t1)

    _check(foo, bb.get()["foo"])


def test_tuple_get_item():
    @R.function
    def foo(x: R.Tensor, y: R.Tensor):
        t1 = R.tuple(x, y)
        t2 = (x, y)
        a = t1[0]
        b = R.TupleGetItem(t2, 1)
        c = R.add(a, b)
        return c

    x = relax.Var("x", R.Tensor())
    y = relax.Var("y", R.Tensor())
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        t1 = bb.emit(relax.Tuple([x, y]))
        t2 = bb.emit(relax.Tuple([x, y]))
        a = bb.emit(relax.TupleGetItem(t1, 0))
        b = bb.emit(relax.TupleGetItem(t2, 1))
        c = bb.emit(relax.op.add(a, b))
        bb.emit_func_output(c)

    _check(foo, bb.get()["foo"])


def test_dataflow_block():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        with R.dataflow():
            lv0 = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
            lv1 = R.call_tir("extern_func", lv0, R.Tensor((128, 128), dtype="float32"))
            gv = lv1
            R.output(gv)
        return gv

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        with bb.dataflow():
            lv0 = bb.emit(relax.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32")))
            lv1 = bb.emit(relax.call_tir("extern_func", lv0, R.Tensor((128, 128), dtype="float32")))
            gv = bb.emit_output(lv1)
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_dataflow_block_advanced():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
        gv1 = R.call_tir("extern_func", gv0, R.Tensor((128, 128), dtype="float32"))
        with R.dataflow():
            m = T.int64()
            n = T.int64()
            lv0 = R.call_tir("extern_func", gv1, R.Tensor((128, 128), dtype="float32"))
            lv1 = R.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv2 = R.call_tir("extern_func", lv0, R.Tensor((128, 128), dtype="float32"))
            gv2 = R.call_tir("extern_func", gv2, R.Tensor((128, 128), dtype="float32"))
            gv3 = R.match_cast(gv2, R.Tensor((m, n), "float32"))
            gv3 = R.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv4 = gv3
            gv5 = gv2
            R.output(gv5, gv4)
        gv6 = R.call_tir("extern_func", gv5, R.Tensor((128, 128), dtype="float32"))
        gv7 = R.call_tir("extern_func", gv6, R.Tensor((128, 128), dtype="float32"))
        return gv7

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    with bb.function("foo", (x,)):
        gv0 = bb.emit(relax.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32")))
        gv1 = bb.emit(relax.call_tir("extern_func", gv0, R.Tensor((128, 128), dtype="float32")))
        with bb.dataflow():
            lv0 = bb.emit(relax.call_tir("extern_func", gv1, R.Tensor((128, 128), dtype="float32")))
            lv1 = bb.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv2 = bb.emit(relax.call_tir("extern_func", lv0, R.Tensor((128, 128), dtype="float32")))
            gv21 = bb.emit(
                relax.call_tir("extern_func", gv2, R.Tensor((128, 128), dtype="float32"))
            )
            gv3 = bb.match_cast(gv21, R.Tensor((m, n), "float32"))
            gv31 = bb.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv32 = bb.emit_output(gv31)
            gv22 = bb.emit_output(gv21)
        gv4 = bb.emit(relax.call_tir("extern_func", gv22, R.Tensor((128, 128), dtype="float32")))
        gv5 = bb.emit(relax.call_tir("extern_func", gv4, R.Tensor((128, 128), dtype="float32")))
        bb.emit_func_output(gv5)

    _check(foo, bb.get()["foo"])


def test_dataflow_binding_after_output():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
                R.output(gv)
                lv = R.call_tir("extern_func", gv, R.Tensor((128, 128), dtype="float32"))
            return gv


def test_dataflow_output_global_var():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
            with R.dataflow():
                gv1 = R.call_tir("extern_func", gv0, R.Tensor((128, 128), dtype="float32"))
                R.output(gv0, gv1)
            return gv1


def test_dataflow_multiple_output():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
                R.output(gv)
                R.output(gv)
            return gv


def test_dataflow_output_outside_dataflow_block():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
            R.output(gv)
            return gv


def test_dataflow_scope_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(ndim=2)):
            with R.dataflow():
                y = R.add(x, x)
                z = R.multiply(y, x)
                w = R.add(z, x)
                R.output(y, w)
            t = R.multiply(y, z)  # z is not in the outer scope
            return t


def test_return_without_binding():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")):
        return x

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        bb.emit_func_output(x)

    _check(foo, bb.get()["foo"])


def test_multiple_return():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            return x
            return x


def test_function_without_return():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            gv0 = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))


def test_tensor_type_without_args():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        v = R.call_tir("tir_relu", x, R.Tensor((32, 32), dtype="float32"))
        return v

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        v = bb.emit(relax.call_tir("tir_relu", x, R.Tensor((32, 32), dtype="float32")))
        bb.emit_func_output(v)

    _check(foo, bb.get()["foo"])


def test_direct_return():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
        return x

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        bb.emit_func_output(x)

    _check(foo, bb.get()["foo"])


def test_call_packed():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        z = R.call_packed("vm.builtin.copy", x, sinfo_args=R.Tensor((32, 32), "float32"))
        return z

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        z = bb.emit(
            relax.Call(
                relax.ExternFunc("vm.builtin.copy"),
                (x,),
                None,
                sinfo_args=[R.Tensor((32, 32), "float32")],
            )
        )
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_annotation():
    @R.function
    def foo(
        x: R.Tensor((32, "m"), "float32"),
        y: R.Tensor(("m",), "float32"),
        r: R.Tensor(dtype="int64"),
    ) -> R.Object:
        m = T.int64()
        z: R.Tensor((32, m), "float32") = R.multiply(x, y)
        w: R.Tensor = R.multiply(z, z)
        q: R.Tensor(ndim=2) = R.add(w, w)
        t = R.add(w, z)
        sh: R.Shape = R.call_packed("shape_of", x, sinfo_args=R.Shape)
        o: R.Object = R.call_packed("contrib.tensor_array_stack", x, y, sinfo_args=R.Object)
        return o

    def _check_struct_info(binding, expected_sinfo):
        tvm.ir.assert_structural_equal(binding.var.struct_info, expected_sinfo)
        tvm.ir.assert_structural_equal(binding.value.struct_info, expected_sinfo)

    # Cannot use block builder here because we need to check the annotated type,
    # which may be inconsistent with deduced type.
    assert isinstance(foo.ret_struct_info, relax.ObjectStructInfo)
    m = relax.get_shape_of(foo.params[0])[1]
    bindings = foo.body.blocks[0].bindings

    _check_struct_info(bindings[0], relax.TensorStructInfo([32, m], "float32"))
    _check_struct_info(bindings[1], relax.TensorStructInfo(dtype="", ndim=-1))
    _check_struct_info(bindings[2], relax.TensorStructInfo(dtype="", ndim=2))
    _check_struct_info(bindings[3], relax.TensorStructInfo(dtype="", ndim=-1))
    _check_struct_info(bindings[4], relax.ShapeStructInfo(ndim=-1))
    _check_struct_info(bindings[5], relax.ObjectStructInfo())


def test_annotate_override():
    @R.function
    def foo(x: R.Tensor):
        y = x
        # z will be treated as object type even though it's a tensor
        z: R.Object = R.add(x, y)
        return z

    assert isinstance(foo.ret_struct_info, relax.ObjectStructInfo)
    y_bind, z_bind = foo.body.blocks[0].bindings
    assert isinstance(y_bind.var.struct_info, relax.TensorStructInfo)
    assert isinstance(z_bind.var.struct_info, relax.ObjectStructInfo)

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def test(x: R.Tensor):
            # Error: x is of Tensor StructInfo, which can not annotate to R.Shape.
            z: R.Shape = x
            return z

    @R.function
    def bar(x: R.Tensor):
        # x is of Tensor StructInfo, the annotation of `z` is ignored.
        z: R.Object = x
        return z

    assert isinstance(bar.ret_struct_info, relax.TensorStructInfo)
    (z_bind,) = bar.body.blocks[0].bindings
    assert isinstance(z_bind.var.struct_info, relax.TensorStructInfo)


def test_call_tir_empty_shape():
    @R.function
    def foo(x: R.Tensor((), "float32")):
        z = R.call_tir("scalar_add", x, R.Tensor((), dtype="float32"))
        return z

    (z_bind,) = foo.body.blocks[0].bindings
    shape_expr = z_bind.value.sinfo_args[0].shape

    assert isinstance(shape_expr, relax.ShapeExpr)
    assert len(shape_expr.values) == 0


def test_call_tir_empty_tuple_arg():
    bb = relax.BlockBuilder()
    dummy_param = relax.Var("dummy_param", R.Tensor(()))
    with bb.function("foo", [dummy_param]):
        output = bb.emit_te(topi.full, shape=(16, 32), dtype="float32", fill_value=1.0)
        bb.emit_func_output(output)

    _check(bb.get())


def test_call_tir_with_tir_var():
    @I.ir_module
    class Module:
        @R.function
        def main(
            dumb_param: R.Tensor(("n",), "float32"), x: R.Tensor(("n * 2", "float32"))
        ) -> R.Tensor(("n * 2",), "float32"):
            n = T.int64()
            y = R.call_tir(copy, (x,), R.Tensor(((n * 2,)), dtype="float32"), tir_vars=(n,))
            return y

        @T.prim_func
        def copy(var_x: T.handle, var_y: T.handle, n: T.int64):
            X = T.match_buffer(var_x, (n * 2,), dtype="float32")
            Y = T.match_buffer(var_y, (n * 2,), dtype="float32")
            for i in T.grid(n * 2):
                with T.block("block"):
                    vi = T.axis.remap("S", [i])
                    Y[vi] = X[vi]

    _check(Module)


def test_local_function():
    @R.function
    def main(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        @R.function
        def outer_func(
            c1: R.Tensor((2, 3), "float32")
        ) -> R.Callable((R.Tensor(None, "float32", ndim=2),), R.Tensor(None, "float32", ndim=2)):
            @R.function
            def inner_func(x1: R.Tensor((2, 3), "float32")):
                s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                return s

            return inner_func

        in_call = outer_func(x)
        res = in_call(y)
        return res

    main_bindings = main.body.blocks[0].bindings
    assert len(main_bindings) == 3
    outer_func = main_bindings[0].value
    assert isinstance(outer_func, relax.Function)

    outer_func_bindings = outer_func.body.blocks[0].bindings
    assert len(outer_func_bindings) == 1
    inner_func = outer_func_bindings[0].value
    assert isinstance(inner_func, relax.Function)


def test_inline_prim_func():
    with pytest.raises(tvm.error.DiagnosticError):

        @I.ir_module
        class TestModule:
            @R.function
            def f(x: R.Tensor((128, 128), "float32"), y: R.Tensor((128, 128), "float32")):
                @T.prim_func
                def my_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
                    A = T.match_buffer(a, (128, 128))
                    B = T.match_buffer(b, (128, 128))
                    C = T.match_buffer(c, (128, 128))

                    for i, j, k in T.grid(128, 128, 128):
                        with T.block():
                            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                            with T.init():
                                C[vi, vj] = 0.0
                            C[vi, vj] += A[vi, vk] * B[vj, vk]

                z = relax.call_tir(my_matmul, (x, y), R.Tensor((128, 128), dtype="float32"))
                return z


def test_cross_function_call():
    @I.ir_module
    class Mod0:
        @R.function
        def foo(x: R.Tensor((10, 5), "float32")):
            s = R.add(x, x)
            return s

        @R.function
        def main(x: R.Tensor((10, 5), "float32")):
            inner = foo
            gv1 = inner(x)
            gv2 = foo(x)
            return (inner, gv1, gv2)

    @I.ir_module
    class Mod1:
        @R.function
        def main(x: R.Tensor((10, 5), "float32")):
            inner = foo
            gv1 = inner(x)
            gv2 = foo(x)
            return (inner, gv1, gv2)

        @R.function
        def foo(x: R.Tensor((10, 5), "float32")) -> R.Tensor((10, 5), "float32"):
            s = R.add(x, x)
            return s


def test_if_branch():
    @R.function
    def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")) -> R.Tensor((1,), "float32"):
        if cond:
            w = R.add(x, x)
            y = R.multiply(w, w)
        else:
            w = R.multiply(x, x)
            y = R.add(w, w)
        return y

    cond, x = foo.params
    y_bind = foo.body.blocks[0].bindings[0]
    y, ite = y_bind.var, y_bind.value

    assert isinstance(y, relax.Var)
    assert y.name_hint == "y"

    assert isinstance(ite, relax.If)
    assert isinstance(ite.true_branch, relax.SeqExpr)
    assert isinstance(ite.false_branch, relax.SeqExpr)

    def check_call(call, op, args):
        assert isinstance(call, relax.Call)
        if isinstance(op, str):
            assert call.op.name == op
        else:
            assert call.op == op
        tvm.ir.assert_structural_equal(call.args, args)

    w_bind = ite.true_branch.blocks[0].bindings[0]
    # the seq exprts in the branches are normalized to bind any call
    # in the seq expr "body" to a var
    y_bind = ite.true_branch.blocks[-1].bindings[-1]
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "relax.add", [x, x])
    check_call(y_bind.value, "relax.multiply", [w_bind.var, w_bind.var])

    w_bind = ite.false_branch.blocks[0].bindings[0]
    y_bind = ite.false_branch.blocks[-1].bindings[-1]
    assert w_bind.var.name_hint == "w"
    check_call(w_bind.value, "relax.multiply", [x, x])
    check_call(y_bind.value, "relax.add", [w_bind.var, w_bind.var])


def test_if_inside_dataflow():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            with R.dataflow():
                if cond:
                    w = R.add(x, x)
                    y = R.multiply(w, w)
                else:
                    w = R.multiply(x, x)
                    y = R.add(w, w)
                R.output(y)
            return y


def test_var_if_scoping_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                y = R.add(w, w)
            return w  # error: The w is not defined in the outer scope


def test_if_branch_var_scope():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                y = R.add(w, w)
            return w


def test_erase_to_well_defined():
    @R.function
    def foo(x: R.Tensor):
        q = x
        m, n = T.int64(), T.int64()
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    tvm.ir.assert_structural_equal(foo.ret_struct_info, R.Tensor(ndim=2))
    _check(foo)


def test_empty_tuple():
    @R.function
    def foo(x: R.Tuple()):
        y: R.Tuple() = R.tuple()
        return y

    x = relax.Var("x", relax.TupleStructInfo([]))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.Tuple([]))
        bb.emit_func_output(y)

    _check(foo, bb.get()["foo"])


def test_symbolic_shape_computing():
    # Tensor Case 1
    @R.function
    def foo(x: R.Tensor(("m + 1",), "float32"), y: R.Tensor(("m", 1), "float32")):
        z = R.add(x, y)
        return z

    m = tir.Var("m", "int64")
    x = relax.Var("x", relax.TensorStructInfo([m + 1], "float32"))
    y = relax.Var("y", relax.TensorStructInfo([m, 1], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        z = bb.emit(relax.op.add(x, y))
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])

    # Tensor Case 2
    @R.function
    def bar(
        x: R.Tensor(("m",), "float32"), y: R.Tensor(("T.max(m, 20)",), "float32")
    ) -> R.Tensor(("T.max(m, 20) + 1",), "float32"):
        m = T.int64()
        z = R.call_tir("test_intrin", (x, y), R.Tensor((T.max(m, 20) + 1,), dtype="float32"))
        return z

    m = tir.Var("m", "int64")
    x = relax.Var("x", relax.TensorStructInfo([m], "float32"))
    y = relax.Var("y", relax.TensorStructInfo([tir.max(m, 20)], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("bar", (x, y)):
        z = bb.emit(
            relax.call_tir("test_intrin", (x, y), R.Tensor((tir.max(m, 20) + 1,), dtype="float32"))
        )
        bb.emit_func_output(z)

    _check(bar, bb.get()["bar"])

    # Shape Case
    @R.function
    def baz(x: R.Shape(("m",)), y: R.Tensor(("m * 2",), "float32")):
        m = T.int64()
        z = R.call_tir("test_intrin", y, R.Tensor((m * 2,), dtype="float32"))
        return z

    m = tir.Var("m", "int64")
    x = relax.Var("x", relax.ShapeStructInfo([m]))
    y = relax.Var("y", relax.TensorStructInfo([m * 2], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("baz", (x, y)):
        z = bb.emit(relax.call_tir("test_intrin", (y), R.Tensor((m * 2,), dtype="float32")))
        bb.emit_func_output(z)

    _check(baz, bb.get()["baz"])

    # Error Case
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor(("m + 1", "m * 2"), "float32")):  # name 'm' is not defined
            z = R.add(x, x)
            return z


def test_arith_operators():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32"), y: R.Tensor(("m", "n"), "float32")):
        a0 = -x
        a1 = x + y
        a2 = x - y
        a3 = x * y
        a4 = x / y
        a5 = x // y
        a6 = x**y

        c0 = x > y
        c1 = x < y
        c2 = x >= y
        c3 = x <= y

        tuple_expr = ((x, x), y)
        t0 = tuple_expr[0]
        t1 = tuple_expr[1]
        t2 = tuple_expr[0][0]  # <= Will normalize to two bindings
        return a0, a1, a2, a3, a4, a5, a6, c0, c1, c2, c3, t0, t1, t2

    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = relax.Var("x", relax.TensorStructInfo([m, n], "float32"))
    y = relax.Var("y", relax.TensorStructInfo([m, n], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        a0 = bb.emit(relax.op.negative(x))
        a1 = bb.emit(relax.op.add(x, y))
        a2 = bb.emit(relax.op.subtract(x, y))
        a3 = bb.emit(relax.op.multiply(x, y))
        a4 = bb.emit(relax.op.divide(x, y))
        a5 = bb.emit(relax.op.floor_divide(x, y))
        a6 = bb.emit(relax.op.power(x, y))

        c0 = bb.emit(relax.op.greater(x, y))
        c1 = bb.emit(relax.op.less(x, y))
        c2 = bb.emit(relax.op.greater_equal(x, y))
        c3 = bb.emit(relax.op.less_equal(x, y))

        tuple_expr = bb.emit(relax.Tuple((relax.Tuple((x, x)), y)))
        t0 = bb.emit(relax.TupleGetItem(tuple_expr, 0))
        t1 = bb.emit(relax.TupleGetItem(tuple_expr, 1))
        tmp = bb.emit(relax.TupleGetItem(tuple_expr, 0))
        t2 = bb.emit(relax.TupleGetItem(tmp, 0))
        bb.emit_func_output(relax.Tuple((a0, a1, a2, a3, a4, a5, a6, c0, c1, c2, c3, t0, t1, t2)))

    _check(foo, bb.get()["foo"])


# TODO(relax-team): enable this when vm ops are ready
@pytest.mark.xfail
def test_vm_ops():
    @R.function
    def foo(x: R.Tensor(("m", "n"), dtype="float32")):
        m = T.int64()
        n = T.int64()
        storage = R.vm.alloc_storage(R.shape([4 * m * n]), dtype="float32", runtime_device_index=0)
        alloc = R.vm.alloc_tensor(storage, shape=R.shape([m, n]), offset=0, dtype="float32")
        tensor = R.builtin.alloc_tensor(R.shape([m, n]), dtype="float32", runtime_device_index=0)
        _ = R.vm.call_tir_dyn("te_func", (x, tensor, (m, n)))
        gv = tensor
        return alloc, gv


def test_prim_value():
    @R.function
    def foo():
        gv = R.call_packed("test", 1, sinfo_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_string_imm():
    @R.function
    def foo():
        gv = R.call_packed("test", "hello", sinfo_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_datatype_imm():
    @R.function
    def foo():
        gv = R.call_packed("test", R.dtype("float32"), sinfo_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_function_void_return_type():
    @tvm.script.ir_module
    class Foo:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")):
            res = mul(x)
            return res

        @R.function
        def mul(x: R.Tensor((3, 3), dtype="float32")):
            res = R.multiply(x, x)
            return res

    _check(Foo)
    # Since the return type of function `mul` is not annotated,
    # the function `main` regards it as a generic return type.
    assert isinstance(Foo["main"].ret_struct_info, relax.ObjectStructInfo)
    assert isinstance(Foo["mul"].ret_struct_info, relax.TensorStructInfo)

    @tvm.script.ir_module
    class Bar:
        @R.function
        def main(x1: R.Tensor((3, 3), dtype="float32")):
            res1 = mul(x1)
            return res1

        @R.function
        def mul(x: R.Tensor((3, 3), dtype="float32")) -> None:
            res = R.multiply(x, x)
            return res

    # Since the return type of function `mul` is not annotated,
    # the function `main` regards it as a generic return type.
    _check(Bar)
    tvm.ir.assert_structural_equal(Bar["main"].ret_struct_info, relax.TupleStructInfo([]))
    tvm.ir.assert_structural_equal(Bar["mul"].ret_struct_info, relax.TupleStructInfo([]))


def test_class_normalize():
    @tvm.script.ir_module
    class InputModule:
        @R.function
        def mul_add(x: R.Tensor) -> R.Tensor:
            return R.multiply(R.add(x, x), R.add(x, x))

    # The parser automatically normalizes the input AST to the following ANF form
    @tvm.script.ir_module
    class OutputModule:
        @R.function
        def mul_add(x: R.Tensor) -> R.Tensor:
            gv = R.add(x, x)
            gv1 = R.add(x, x)
            return R.multiply(gv, gv1)

    _check(InputModule, OutputModule)


if __name__ == "__main__":
    tvm.testing.main()
