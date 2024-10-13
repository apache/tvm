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
import sys
from typing import Optional, Union

import pytest
import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax, tir, topi
from tvm.ir import VDevice, DummyGlobalInfo
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
    if isinstance(parsed, IRModule) and isinstance(roundtrip_mod, IRModule):
        assert relax.analysis.well_formed(parsed)
        assert relax.analysis.well_formed(roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_simple_func():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
        R.func_attr({"Primitive": 1})
        gv0 = R.call_dps_packed("extern_func", x, R.Tensor((128, 128), dtype="float32"))
        gv1 = R.call_dps_packed("extern_dps_func", gv0, R.Tensor((128, 128), dtype="float32"))
        return gv1

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), attrs={"Primitive": 1}):
        y = bb.emit(relax.call_dps_packed("extern_func", x, R.Tensor((128, 128), dtype="float32")))
        out = bb.emit(
            relax.call_dps_packed("extern_dps_func", y, R.Tensor((128, 128), dtype="float32"))
        )
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
            x: R.Tensor((2, 3), "float32", ndim=3),
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


def test_unexpected_tir_args():
    with pytest.raises(tvm.error.DiagnosticError):

        @tvm.script.ir_module
        class TestWellCallTIR:
            @T.prim_func
            def tir_addone(A: T.Buffer((16, 16), "int32"), B: T.Buffer((16, 16), "int32")) -> None:
                T.func_attr(({"global_symbol": "tir_addone"}))
                for i, j in T.grid(16, 16):
                    with T.block("tir_addone"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] + T.int32(1)

            @R.function
            def foo(x: R.Tensor(("m", "m"), "float32")):
                m = T.int64()
                # tir.max expects 2 arguments, but got 1
                gv = R.call_tir(tir_addone, (x,), R.Tensor((T.max(16),), dtype="float32"))
                return gv

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor(("m", "n"), "float32")):
            m = T.int64()
            # call_tir expected a tir prim_func
            return relax.call_tir("extern_func", (x,), R.Tensor((T.max(m),), dtype="float32"))


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


def test_incorrect_tensor_shape():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def f(x: R.Tensor([16])):
            y: R.Tensor(16) = R.add(x, x)
            return y


def test_simple_module():
    @I.ir_module
    class TestModule:
        @T.prim_func(private=True)
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
            cls = TestModule
            gv0 = R.call_tir(cls.tir_func, x, R.Tensor((128, 128), dtype="float32"))
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)

    _check(TestModule, bb.get())


def test_emit_te_primfunc_attrs():
    @I.ir_module
    class TestModule:
        @T.prim_func(private=True)
        def plus_one(
            x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
            y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"some_attr": "foo", "another_attr": True, "tir.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            cls = TestModule
            gv0 = R.call_tir(cls.plus_one, x, R.Tensor((128, 128), dtype="float32"))
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        out = bb.emit_te(
            lambda x: x + 1,
            x,
            primfunc_name_hint="plus_one",
            primfunc_attrs={"some_attr": "foo", "another_attr": True},
        )
        bb.emit_func_output(out)
    _check(TestModule, bb.get())


def test_emit_te():
    @I.ir_module
    class EmitTE:
        @R.function
        def main(x: R.Tensor((10, 20), "float32")) -> R.Tensor((10, 20), dtype="float32"):
            lv1 = R.emit_te(topi.add, x, x)
            out = R.emit_te(topi.multiply, lv1, lv1)
            return out

    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo([10, 20], "float32"))
    with bb.function("main", [x], {"global_symbol": "main"}):
        lv1 = bb.emit_te(topi.add, x, x)
        out = bb.emit_te(topi.multiply, lv1, lv1)
        bb.emit_func_output(out)

    _check(EmitTE, bb.get())


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

        @T.prim_func(private=True)
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
            cls = TestModule
            gv0 = R.call_tir(cls.tir_func, x, R.Tensor((128, 128), dtype="float32"))
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)
    mod = bb.get()
    mod.update_global_info("dummy", [DummyGlobalInfo(), DummyGlobalInfo()])
    mod = mod.with_attr("attr", tvm.tir.IntImm("int32", 10))
    _check(TestModule, mod)


def test_global_info_vdevice():
    vdevices = [
        VDevice("llvm"),
        VDevice("cuda", 0),
        VDevice("cuda -arch=sm_80", 0),
        VDevice("metal", 0, "global"),
    ]

    @I.ir_module
    class TestModule:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("cuda", 0),
                    I.vdevice("cuda -arch=sm_80", 0),
                    I.vdevice("metal", 0, "global"),
                ]
            }
        )

        @T.prim_func(private=True)
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
            cls = TestModule
            gv0 = R.call_tir(cls.tir_func, x, R.Tensor((128, 128), dtype="float32"))
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)
    mod = bb.get()
    mod.update_global_info("vdevice", vdevices)
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


def test_relax_shape_to_tensor():
    @R.function
    def foo(x: R.Shape((4, 4))):
        tensor = R.shape_to_tensor(x)
        return tensor

    x = relax.Var("x", R.Shape((4, 4)))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        tensor = bb.emit(relax.op.shape_to_tensor(x))
        bb.emit_func_output(tensor)

    _check(foo, bb.get()["foo"])


def test_symbolic_shape():
    @R.function
    def foo(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
        m = T.int64()
        n = T.int64()
        gv0 = R.call_dps_packed("extern_func", x, R.Tensor((m, n), dtype="float32"))
        return gv0

    @R.function
    def bar(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
        m = T.int64()
        n = T.int64()
        gv0 = R.call_dps_packed("extern_func", x, R.Tensor((m, n), dtype="float32"))
        return gv0

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def mismatch_dtype(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
            m = T.int64()
            n = T.int32()  # The shape dtype should be int64
            gv0 = R.call_dps_packed("extern_func", x, R.Tensor((m, n), dtype="float32"))
            return gv0

    def _expected(name: str):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = relax.Var("x", R.Tensor([m, n], "float32"))
        bb = relax.BlockBuilder()
        with bb.function(name, (x,)):
            out = bb.emit(
                relax.call_dps_packed("extern_func", x, R.Tensor((m, n), dtype="float32"))
            )
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
        gv0 = R.call_dps_packed("extern_func_0", x, R.Tensor((4, 4), dtype="float32"))
        gv1 = R.call_dps_packed("extern_func_1", x, R.Tensor((4, 4), dtype="float32"))
        return (gv0, gv1)

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        gv0 = bb.emit(relax.call_dps_packed("extern_func_0", x, R.Tensor((4, 4), dtype="float32")))
        gv1 = bb.emit(relax.call_dps_packed("extern_func_1", x, R.Tensor((4, 4), dtype="float32")))
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


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
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


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
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
            lv0 = R.call_dps_packed("extern_func", x, R.Tensor((128, 128), dtype="float32"))
            lv1 = R.call_dps_packed("extern_func", lv0, R.Tensor((128, 128), dtype="float32"))
            gv = lv1
            R.output(gv)
        return gv

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_dps_packed("extern_func", x, R.Tensor((128, 128), dtype="float32"))
            )
            lv1 = bb.emit(
                relax.call_dps_packed("extern_func", lv0, R.Tensor((128, 128), dtype="float32"))
            )
            gv = bb.emit_output(lv1)
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_dataflow_block_advanced():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_dps_packed("extern_func", x, R.Tensor((128, 128), dtype="float32"))
        gv1 = R.call_dps_packed("extern_func", gv0, R.Tensor((128, 128), dtype="float32"))
        with R.dataflow():
            m = T.int64()
            n = T.int64()
            lv0 = R.call_dps_packed("extern_func", gv1, R.Tensor((128, 128), dtype="float32"))
            lv1 = R.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv2 = R.call_dps_packed("extern_func", lv0, R.Tensor((128, 128), dtype="float32"))
            gv2 = R.call_dps_packed("extern_func", gv2, R.Tensor((128, 128), dtype="float32"))
            gv3 = R.match_cast(gv2, R.Tensor((m, n), "float32"))
            gv3 = R.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv4 = gv3
            gv5 = gv2
            R.output(gv5, gv4)
        gv6 = R.call_dps_packed("extern_func", gv5, R.Tensor((128, 128), dtype="float32"))
        gv7 = R.call_dps_packed("extern_func", gv6, R.Tensor((128, 128), dtype="float32"))
        return gv7

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    with bb.function("foo", (x,)):
        gv0 = bb.emit(
            relax.call_dps_packed("extern_func", x, R.Tensor((128, 128), dtype="float32"))
        )
        gv1 = bb.emit(
            relax.call_dps_packed("extern_func", gv0, R.Tensor((128, 128), dtype="float32"))
        )
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_dps_packed("extern_func", gv1, R.Tensor((128, 128), dtype="float32"))
            )
            lv1 = bb.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv2 = bb.emit(
                relax.call_dps_packed("extern_func", lv0, R.Tensor((128, 128), dtype="float32"))
            )
            gv21 = bb.emit(
                relax.call_dps_packed("extern_func", gv2, R.Tensor((128, 128), dtype="float32"))
            )
            gv3 = bb.match_cast(gv21, R.Tensor((m, n), "float32"))
            gv31 = bb.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv32 = bb.emit_output(gv31)
            gv22 = bb.emit_output(gv21)
        gv4 = bb.emit(
            relax.call_dps_packed("extern_func", gv22, R.Tensor((128, 128), dtype="float32"))
        )
        gv5 = bb.emit(
            relax.call_dps_packed("extern_func", gv4, R.Tensor((128, 128), dtype="float32"))
        )
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
        v = R.call_dps_packed("extern_relu", x, R.Tensor((32, 32), dtype="float32"))
        return v

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        v = bb.emit(relax.call_dps_packed("extern_relu", x, R.Tensor((32, 32), dtype="float32")))
        bb.emit_func_output(v)

    _check(foo, bb.get()["foo"])


def test_tensor_with_vdevice():
    vdevices = [
        VDevice("llvm"),
        VDevice("cuda", 0),
        VDevice("metal", 0, "global"),
        VDevice("cuda -arch=sm_80", 0),
    ]

    @I.ir_module
    class TestModule:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("cuda", 0),
                    I.vdevice("metal", 0, "global"),
                    I.vdevice("cuda -arch=sm_80", 0),
                ]
            }
        )

        @R.function
        def foo(
            a: R.Tensor((128, 128), "float32", "cuda:1"),  # noqa: F722
            b: R.Tensor((128, 128), "float32", "llvm"),
            c: R.Tensor((128, 128), "float32", "vdevice:3"),  # noqa: F722
        ) -> R.Tensor((128, 128), "float32", "cuda:1"):  # noqa: F722
            s = R.add(a, c)
            return s

    a = relax.Var("a", R.Tensor((128, 128), "float32", vdevices[3]))
    b = relax.Var("b", R.Tensor((128, 128), "float32", vdevices[0]))
    c = relax.Var("c", R.Tensor((128, 128), "float32", vdevices[3]))
    bb = relax.BlockBuilder()
    with bb.function("foo", (a, b, c)):
        out = bb.emit(relax.op.add(a, c))
        bb.emit_func_output(out)
    mod = bb.get()
    mod = mod.with_attr("attr", tvm.tir.IntImm("int32", 10))
    mod.update_global_info("vdevice", vdevices)

    _check(TestModule, mod)


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
    @R.function(pure=False)
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        z = R.call_packed("vm.builtin.copy", x, sinfo_args=R.Tensor((32, 32), "float32"))
        return z

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x), pure=False):
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


def test_call_packed_without_sinfo_args():
    @R.function(pure=False)
    def foo(x: R.Object) -> R.Object:
        z = R.call_packed("test", x)
        return z

    x = relax.Var("x", R.Object())
    bb = relax.BlockBuilder()
    with bb.function("foo", (x), pure=False):
        z = bb.emit(
            relax.Call(
                relax.ExternFunc("test"),
                (x,),
                None,
                sinfo_args=[],
            )
        )
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_annotation():
    @R.function(pure=False)
    def foo(
        x: R.Tensor((32, "m"), "float32"),
        y: R.Tensor(("m",), "float32"),
        r: R.Tensor(dtype="int64"),
    ) -> R.Object:
        m = T.int64()
        z: R.Tensor((32, m), "float32") = R.multiply(x, y)
        w: R.Tensor(ndim=2) = R.multiply(z, z)
        q: R.Tensor = R.add(w, w)
        t = R.add(w, z)
        sh: R.Shape = R.call_packed("shape_of", x, sinfo_args=R.Shape)
        lv: R.Tensor(sh, dtype="float32") = R.reshape(x, sh)
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
    sh = bindings[4].var

    _check_struct_info(bindings[0], relax.TensorStructInfo([32, m], "float32"))
    _check_struct_info(bindings[1], relax.TensorStructInfo(dtype="", ndim=2))
    _check_struct_info(bindings[2], relax.TensorStructInfo(dtype="", ndim=-1))
    _check_struct_info(bindings[3], relax.TensorStructInfo(dtype="", ndim=2))
    _check_struct_info(bindings[4], relax.ShapeStructInfo(ndim=-1))
    _check_struct_info(bindings[5], relax.TensorStructInfo(sh))
    _check_struct_info(bindings[6], relax.ObjectStructInfo())


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


def test_call_dps_packed_empty_shape():
    @R.function
    def foo(x: R.Tensor((), "float32")):
        z = R.call_dps_packed("scalar_add", x, R.Tensor((), dtype="float32"))
        return z

    (z_bind,) = foo.body.blocks[0].bindings
    shape_expr = z_bind.value.sinfo_args[0].shape

    assert isinstance(shape_expr, relax.ShapeExpr)
    assert len(shape_expr.values) == 0


def test_call_tir_empty_tuple_arg():
    bb = relax.BlockBuilder()
    dummy_param = relax.Var("dummy_param", R.Tensor(()))
    with bb.function("foo", [dummy_param], {"global_symbol": "foo"}):
        output = bb.emit_te(topi.full, shape=(16, 32), dtype="float32", fill_value=1.0)
        bb.emit_func_output(output)

    _check(bb.get())


def test_call_tir_with_tir_var():
    @I.ir_module
    class Module:
        @R.function
        def main(
            dumb_param: R.Tensor(("n",), "float32"), x: R.Tensor(("n * 2",), "float32")
        ) -> R.Tensor(("n * 2",), "float32"):
            n = T.int64()
            cls = Module
            y = R.call_tir(cls.copy, x, R.Tensor((n * 2,), dtype="float32"), tir_vars=(n,))
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


def test_call_tir_with_grad():
    @I.ir_module
    class Module:
        @T.prim_func
        def identity_tir(a: T.handle, b: T.handle) -> None:
            A = T.match_buffer(a, [54, 96])
            B = T.match_buffer(b, [54, 96])

            for i, j in T.grid(54, 96):
                with T.block("compute"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def main(v0: R.Tensor([54, 96], "float32")):
            cls = Module
            out = R.call_tir_with_grad(
                cls.identity_tir,
                (v0,),
                R.Tensor((54, 96), "float32"),
                te_grad_name="identity_k_grad",
                te_grad_kwargs={"k": 1.0},
            )
            return out

    _check(Module)


def test_call_tir_inplace():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def copy(
            A: T.Buffer((2, 3), "int32"),
            B: T.Buffer((2, 3), "int32"),
            out1: T.Buffer((2, 3), "int32"),
        ):
            # copies the contents of B into A and out1
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_zeros"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(B[ax0, ax1])
                    T.writes(A[ax0, ax1], out1[ax0, ax1])
                    A[ax0, ax1] = B[ax0, ax1]
                    out1[ax0, ax1] = B[ax0, ax1]

        @R.function
        def main(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")
        ):
            res = R.call_tir_inplace(
                Module.copy,
                (x, y),
                [0, -1],
                [R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")],
            )
            return res

    _check(Module)


def test_call_tir_inplace_with_tuple_var_raises_error():
    with pytest.raises(tvm.error.DiagnosticError):

        @tvm.script.ir_module
        class Module:
            @R.function
            def main(x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")):
                cls = Module
                args = (x, y)
                res = R.call_tir_inplace(
                    cls.copy,
                    # The `args` tuple must be an in-line tuple, not a
                    # reference to a tuple.  This error should be
                    # caught and raised during parsing.
                    args,
                    inplace_indices=[0, -1],
                    out_sinfo=[R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")],
                )
                return res

            @T.prim_func
            def copy(
                A: T.Buffer((2, 3), "int32"),
                B: T.Buffer((2, 3), "int32"),
                out1: T.Buffer((2, 3), "int32"),
            ):
                # copies the contents of B into A and out1
                T.func_attr({"tir.noalias": True})
                for iters in T.grid(T.int64(2), T.int64(3)):
                    with T.block("T_zeros"):
                        i, j = T.axis.remap("SS", iters)
                        A[i, j] = B[i, j]
                        out1[i, j] = B[i, j]


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
            cls = Mod0
            inner = cls.foo
            gv1 = inner(x)
            gv2 = Mod0.foo(x)
            return (inner, gv1, gv2)

    @I.ir_module
    class Mod1:
        @R.function
        def main(x: R.Tensor((10, 5), "float32")):
            cls = Mod1
            inner = cls.foo
            gv1 = inner(x)
            gv2 = Mod1.foo(x)
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


def test_if_branch_with_match_cast():
    """The last branch of a relax::If node may be a MatchCast

    This is a regression test.  In previous implementations, using
    R.match_cast as the last binding would cause a segfault while
    parsing.
    """

    @R.function
    def func(A: R.Tensor([16, 16]), is_bfloat16: R.Prim("bool")):
        if is_bfloat16:
            A = R.match_cast(A, R.Tensor([16, 16], "bfloat16"))
            B = A.astype("float16")
        else:
            B = R.match_cast(A, R.Tensor([16, 16], "float16"))
        return B

    A, is_bfloat16 = func.params
    (block,) = func.body.blocks
    (B_binding,) = block.bindings

    B_var = B_binding.var
    assert isinstance(B_var, relax.Var)
    assert B_var.name_hint == "B"

    if_then_else = B_binding.value
    assert isinstance(if_then_else, relax.If)
    assert isinstance(if_then_else.true_branch, relax.SeqExpr)
    assert isinstance(if_then_else.false_branch, relax.SeqExpr)

    else_branch = if_then_else.false_branch
    (else_block,) = else_branch.blocks

    assert isinstance(else_block.bindings[-1], relax.MatchCast)

    # If the `R.match_cast` were removed, the function would infer the
    # return value as `R.Tensor([16,16])`, with an unknown dtype.
    # With the `R.match_cast` retained, the output dtype is known.
    tvm.ir.assert_structural_equal(func.ret_struct_info, R.Tensor([16, 16], "float16"))


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


def test_scalar_tensor_as_branch_condition():
    """Branch condition can be 0-d tensor"""

    @R.function
    def func(cond: R.Tensor([], "bool"), x: R.Tensor((1,), "float32")):
        if cond:
            out = R.add(x, x)
        else:
            out = R.multiply(x, x)
        return out

    if_else = func.body.blocks[0].bindings[0].value
    assert isinstance(if_else.cond, relax.Var)
    tvm.ir.assert_structural_equal(if_else.cond.struct_info, R.Tensor([], "bool"))


def test_prim_value_as_branch_condition():
    """In addition to scalar tensor, can use R.Prim condition"""

    @R.function
    def func(cond: R.Prim("bool"), x: R.Tensor((1,), "float32")):
        if cond:
            out = R.add(x, x)
        else:
            out = R.multiply(x, x)
        return out

    if_else = func.body.blocks[0].bindings[0].value
    assert isinstance(if_else.cond, relax.Var)
    tvm.ir.assert_structural_equal(if_else.cond.struct_info, R.Prim("bool"))


def test_computed_prim_value_as_branch_condition():
    """The R.Prim condition may be computed within the function"""

    @R.function
    def func(x: R.Tensor(["N"], "float32")):
        N = T.int64()
        if R.prim_value(N % 16 == 0):
            out = R.call_pure_packed("fast_vectorized_impl", x, sinfo_args=[x.struct_info])
        else:
            out = R.call_pure_packed("slow_non_vectorized_impl", x, sinfo_args=[x.struct_info])
        return out

    N = func.params[0].struct_info.shape[0]
    if_else = func.body.blocks[0].bindings[0].value
    assert isinstance(if_else.cond, relax.PrimValue)
    tvm.ir.assert_structural_equal(N % 16 == 0, if_else.cond.value)
    tvm.ir.assert_structural_equal(if_else.cond.struct_info, R.Prim(value=N % 16 == 0))


def test_tir_expr_as_branch_condition():
    """Syntactic sugar, wrap PrimExpr as PrimValue"""

    @R.function(private=True)
    def sugared(x: R.Tensor(["N"], "float32")):
        N = T.int64()
        if N % 16 == 0:
            out = R.call_pure_packed("fast_vectorized_impl", x, sinfo_args=[x.struct_info])
        else:
            out = R.call_pure_packed("slow_non_vectorized_impl", x, sinfo_args=[x.struct_info])
        return out

    @R.function(private=True)
    def unsugared(x: R.Tensor(["N"], "float32")):
        N = T.int64()
        if R.prim_value(N % 16 == 0):
            out = R.call_pure_packed("fast_vectorized_impl", x, sinfo_args=[x.struct_info])
        else:
            out = R.call_pure_packed("slow_non_vectorized_impl", x, sinfo_args=[x.struct_info])
        return out

    tvm.ir.assert_structural_equal(unsugared, sugared)


def test_scalar_tensor_as_assert_condition():
    """Branch condition can be 0-d tensor"""

    @R.function(pure=False)
    def func(cond: R.Tensor([], "bool"), x: R.Tensor((1,), "float32")):
        _ = R.assert_op(cond)
        out = R.add(x, x)
        return out

    assert_op = func.body.blocks[0].bindings[0].value
    condition = assert_op.args[0]
    assert isinstance(condition, relax.Var)
    tvm.ir.assert_structural_equal(condition.struct_info, R.Tensor([], "bool"))


def test_prim_value_as_assert_condition():
    """In addition to scalar tensor, can use R.Prim condition"""

    @R.function(pure=False)
    def func(cond: R.Prim("bool"), x: R.Tensor((1,), "float32")):
        _ = R.assert_op(cond)
        out = R.add(x, x)
        return out

    assert_op = func.body.blocks[0].bindings[0].value
    condition = assert_op.args[0]
    assert isinstance(condition, relax.Var)
    tvm.ir.assert_structural_equal(condition.struct_info, R.Prim("bool"))


def test_computed_prim_value_as_assert_condition():
    """The R.Prim condition may be computed within the function"""

    @R.function(pure=False)
    def func(x: R.Tensor(["N"], "float32")):
        N = T.int64()
        _ = R.assert_op(R.prim_value(N % 16 == 0))
        out = R.call_packed("fast_vectorized_impl", x, sinfo_args=[x.struct_info])
        return out

    N = func.params[0].struct_info.shape[0]
    assert_op = func.body.blocks[0].bindings[0].value
    condition = assert_op.args[0]
    assert isinstance(condition, relax.PrimValue)
    tvm.ir.assert_structural_equal(N % 16 == 0, condition.value)
    tvm.ir.assert_structural_equal(condition.struct_info, R.Prim(value=N % 16 == 0))


def test_tir_expr_as_assert_condition():
    """Syntactic sugar, wrap PrimExpr as PrimValue"""

    @R.function(pure=False, private=True)
    def sugared(x: R.Tensor(["N"], "float32")):
        N = T.int64()
        _ = R.assert_op(N % 16 == 0)
        out = R.call_packed("fast_vectorized_impl", x, sinfo_args=[x.struct_info])
        return out

    @R.function(pure=False, private=True)
    def unsugared(x: R.Tensor(["N"], "float32")):
        N = T.int64()
        _ = R.assert_op(R.prim_value(N % 16 == 0))
        out = R.call_packed("fast_vectorized_impl", x, sinfo_args=[x.struct_info])
        return out

    tvm.ir.assert_structural_equal(unsugared, sugared)


def test_erase_to_well_defined_removes_internal_vars():
    @R.function
    def foo(x: R.Tensor):
        q = x
        m, n = T.int64(), T.int64()
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    tvm.ir.assert_structural_equal(foo.ret_struct_info, R.Tensor(ndim=2))
    assert foo.ret_struct_info.shape is None
    _check(foo)


def test_erase_to_well_defined_keeps_variables_exposed_by_tensor_shape():
    @R.function
    def foo(x: R.Tensor(["m", "n"])):
        q = x
        m, n = T.int64(), T.int64()
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    assert foo.ret_struct_info.shape is not None
    _check(foo)


def test_erase_to_well_defined_keeps_variants_exposed_by_shape_expr():
    @R.function
    def foo(x: R.Tensor, _: R.Shape(["m", "n"])):
        q = x
        m, n = T.int64(), T.int64()
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    assert foo.ret_struct_info.shape is not None
    _check(foo)


def test_erase_to_well_defined_keeps_variants_exposed_by_prim_value():
    @R.function
    def foo(x: R.Tensor, _m: R.Prim(value="m"), _n: R.Prim(value="n")):
        q = x
        m, n = T.int64(), T.int64()
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    assert foo.ret_struct_info.shape is not None
    _check(foo)


def test_erase_to_well_defined_infers_from_shape_expr():
    @I.ir_module
    class Module:
        # The subroutine's symbolic variables are only in-scope for the subroutine.
        @R.function
        def subroutine(x: R.Tensor, _: R.Shape(["m", "n"])) -> R.Tensor(["m", "n"]):
            q = x
            m, n = T.int64(), T.int64()
            z = R.match_cast(q, R.Tensor((m, n)))
            w = z
            return w

        # However, struct inference can make the symbolic variables in
        # the main function to the symbolic variables in the
        # subroutine.  Therefore, the shape of the tensor returned
        # from main can have a well-defined shape.
        @R.function
        def main(x: R.Tensor, shape: R.Shape(["m", "n"])):
            output = Module.subroutine(x, shape)
            return output

    assert Module["main"].ret_struct_info.shape is not None
    _check(Module)


def test_erase_to_well_defined_infers_from_prim_value():
    @I.ir_module
    class Module:
        # The subroutine's symbolic variables are only in-scope for the subroutine.
        @R.function
        def subroutine(
            x: R.Tensor, _m: R.Prim(value="m"), _n: R.Prim(value="n")
        ) -> R.Tensor(["m", "n"]):
            q = x
            m, n = T.int64(), T.int64()
            z = R.match_cast(q, R.Tensor((m, n)))
            w = z
            return w

        # However, struct inference can make the symbolic variables in
        # the main function to the symbolic variables in the
        # subroutine.  Therefore, the shape of the tensor returned
        # from main can have a well-defined shape.
        @R.function
        def main(x: R.Tensor, relax_m: R.Prim(value="m"), relax_n: R.Prim(value="n")):
            output = Module.subroutine(x, relax_m, relax_n)
            return output

    assert Module["main"].ret_struct_info.shape is not None
    _check(Module)


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


def test_symbolic_vars_in_tensor_shape_with_usage_first():
    """First param may use symbolic variable defined in second param"""

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


def test_symbolic_vars_in_tensor_shape_with_definition_first():
    """Second param may use symbolic variable defined in first param"""

    @R.function
    def bar(
        x: R.Tensor(("m",), "float32"), y: R.Tensor(("T.max(m, 20)",), "float32")
    ) -> R.Tensor(("T.max(m, 20) + 1",), "float32"):
        m = T.int64()
        z = R.call_dps_packed("test_intrin", (x, y), R.Tensor((T.max(m, 20) + 1,), dtype="float32"))
        return z

    m = tir.Var("m", "int64")
    x = relax.Var("x", relax.TensorStructInfo([m], "float32"))
    y = relax.Var("y", relax.TensorStructInfo([tir.max(m, 20)], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("bar", (x, y)):
        z = bb.emit(
            relax.call_dps_packed(
                "test_intrin", (x, y), R.Tensor((tir.max(m, 20) + 1,), dtype="float32")
            )
        )
        bb.emit_func_output(z)

    _check(bar, bb.get()["bar"])


def test_symbolic_vars_in_shape():
    """Symbolic variable may be defined in R.Shape"""

    @R.function
    def baz(x: R.Shape(("m",)), y: R.Tensor(("m * 2",), "float32")):
        m = T.int64()
        z = R.call_dps_packed("test_intrin", y, R.Tensor((m * 2,), dtype="float32"))
        return z

    m = tir.Var("m", "int64")
    x = relax.Var("x", relax.ShapeStructInfo([m]))
    y = relax.Var("y", relax.TensorStructInfo([m * 2], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("baz", (x, y)):
        z = bb.emit(relax.call_dps_packed("test_intrin", (y), R.Tensor((m * 2,), dtype="float32")))
        bb.emit_func_output(z)

    _check(baz, bb.get()["baz"])


def test_symbolic_vars_in_prim_value():
    """Symbolic variable may be defined in R.Prim"""

    @R.function
    def baz(x: R.Prim(value="m"), y: R.Tensor(("m * 2",), "float32")):
        m = T.int64()
        z = R.call_dps_packed("test_intrin", y, R.Tensor((m * 2,), dtype="float32"))
        return z

    m = tir.Var("m", "int64")
    x = relax.Var("x", relax.PrimStructInfo(value=m))
    y = relax.Var("y", relax.TensorStructInfo([m * 2], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("baz", (x, y)):
        z = bb.emit(relax.call_dps_packed("test_intrin", (y), R.Tensor((m * 2,), dtype="float32")))
        bb.emit_func_output(z)

    _check(baz, bb.get()["baz"])


def test_undefined_symbolic_var_raises_error():
    """An undefined symbolic variable in an error

    A symbolic variables is defined at the first site where it appears
    as a shape parameter without any modification.  TVMScript does not
    support solving for a symbolic variable in terms of the argument
    shape.  That is, this test case raises an error, and will not
    attempt to define `m` as either `x.shape[0]-1` or `x.shape[1]//2`.
    """
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function
        def foo(x: R.Tensor(("m + 1", "m * 2"), "float32")):  # name 'm' is not defined
            z = R.add(x, x)
            return z


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
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
        return (a0, a1, a2, a3, a4, a5, a6, c0, c1, c2, c3, t0, t1, t2)

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


def test_memory_ops():
    @R.function
    def foo(x: R.Tensor(("m", "n"), dtype="float32")):
        m = T.int64()
        n = T.int64()
        storage = R.memory.alloc_storage(
            R.shape([4 * m * n]), virtual_device_index=0, storage_scope="global", dtype="float32"
        )
        alloc = R.memory.alloc_tensor(storage, offset=0, shape=R.shape([m, n]), dtype="float32")
        tensor = R.builtin.alloc_tensor(R.shape([m, n]), dtype="float32", runtime_device_index=0)
        gv = tensor
        return alloc, gv

    _check(foo)


def test_vm_ops():
    @R.function(pure=False)
    def foo(x: R.Tensor(("m", "n"), dtype="float32")):
        m = T.int64()
        n = T.int64()
        storage = R.vm.alloc_storage(R.shape([4 * m * n]), runtime_device_index=0, dtype="uint8")
        alloc = R.vm.alloc_tensor(storage, offset=0, shape=R.shape([m, n]), dtype="float32")
        tensor = R.builtin.alloc_tensor(R.shape([m, n]), dtype="float32", runtime_device_index=0)
        tir_dym = R.vm.call_tir_dyn("te_func", (x, tensor, R.ShapeExpr((m, n))))
        return alloc, tir_dym

    _check(foo)


def test_builtin_ops():
    @R.function
    def foo(x: R.Tensor(("m", "n"), dtype="float32")):
        tensor = R.builtin.stop_lift_params(x)
        gv = tensor
        return gv

    _check(foo)


def test_prim_value():
    @R.function(pure=False)
    def foo():
        gv = R.call_packed("test", 1, sinfo_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_string_imm():
    @R.function(pure=False)
    def foo():
        gv = R.call_packed("test", "hello", sinfo_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_datatype_imm():
    @R.function(pure=False)
    def foo():
        gv = R.call_packed("test", R.dtype("float32"), sinfo_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_function_void_return_type():
    @tvm.script.ir_module
    class Foo:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")):
            res = Foo.mul(x)
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
            res1 = Bar.mul(x1)
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


def test_context_aware_parsing(monkeypatch):
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add(
            X: T.Buffer([T.int64(2), T.int64(4)], "float32"),
            Y: T.Buffer((), "float32"),
            Z: T.Buffer([T.int64(2), T.int64(4)], "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": 1})
            cls = Module
            alloc = R.builtin.alloc_tensor(R.shape([2, 4]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = cls.add(x, R.const(1, "float32"), alloc)
            return alloc

    _check(Module)

    # Break the env settings, but context-aware parsing can still handle it
    def _break_env(self, *args):
        raise RuntimeError("Fail to pass context-aware parsing")

    monkeypatch.setattr(tvm.ir.GlobalVar, "__call__", _break_env)

    _check(Module)


def test_unit_tuple_on_rhs_of_assign():
    @I.ir_module
    class Module:
        @R.function
        def main(input: R.Tensor((5, 5))) -> R.Tuple(R.Tensor((5, 5))):
            gv = (input,)
            return gv

    _check(Module)


def test_empty_tuple_on_rhs_of_assign():
    @I.ir_module
    class Module:
        @R.function
        def main(input: R.Tensor((5, 5))) -> R.Tuple():
            gv = ()
            return gv

    _check(Module)


def test_global_var_sinfo():
    @I.ir_module
    class Module:
        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            gv0 = R.emit_te(topi.add, x, x)
            return gv0

    target_sinfo = R.Callable(
        (R.Tensor((128, 128), dtype="float32"),), R.Tensor((128, 128), dtype="float32")
    )
    gv = Module.get_global_var("foo")
    tvm.ir.assert_structural_equal(gv.struct_info, target_sinfo)
    tvm.ir.assert_structural_equal(Module["foo"].struct_info, target_sinfo)
    _check(Module)


def test_assert_op():
    @I.ir_module
    class AssertOp:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.assert_op(R.const(False, dtype="bool"), x, format="x: {}")
            return x

    _check(AssertOp)


def test_assert_outside_of_class():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        y = R.assert_op(R.const(False, dtype="bool"), x, format="x: {}")
        return x

    # this just makes sure that the machinery regarding the pure attribute parses
    # in the case where the function is outside of a class too
    _check(func)


def test_impure_inner_function():
    @R.function
    def f(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        # we will not actually call it
        @R.function(pure=False)
        def g(y: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            z = R.assert_op(R.const(False, dtype="bool"), y, format="y: {}")
            return y

        return x

    assert f.is_pure
    # definition of g
    assert not f.body.blocks[0].bindings[0].value.is_pure

    # make sure we are not incorrectly passing state for inner functions
    _check(f)


def test_impure_inner_function_in_class():
    @I.ir_module
    class ImpureInner:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            # we will not actually call it
            @R.function(pure=False)
            def g(y: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                z = R.assert_op(R.const(False, dtype="bool"), y, format="y: {}")
                return y

            return x

    assert ImpureInner["main"].is_pure
    # definition of g
    assert not ImpureInner["main"].body.blocks[0].bindings[0].value.is_pure

    # make sure we are not incorrectly passing state for inner functions
    _check(ImpureInner)


def test_print():
    @I.ir_module
    class Print:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.print(x, format="x: {}")
            return x

    _check(Print)


def test_parse_multiple_pure_and_impure_funcs():
    @I.ir_module
    class Mixture:
        @R.function(pure=False)
        def print(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.print(x, format="x: {}")
            return x

        @R.function(pure=False)
        def assert_func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.assert_op(R.const(False, dtype="bool"), x, format="x: {}")
            return x

        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            return x

    assert not Mixture["print"].is_pure
    assert not Mixture["assert_func"].is_pure
    assert Mixture["main"].is_pure
    _check(Mixture)


def test_function_with_void_return_type_may_be_used_as_statements():
    """Void return of calls do not need to be assigned"""

    @I.ir_module
    class Unsugared:
        @R.function(pure=False)
        def print(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.print(x, format="x: {}")
            return x

        @R.function(pure=False)
        def assert_func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.assert_op(R.const(False, dtype="bool"), x, format="x: {}")
            return x

    @I.ir_module
    class Sugared:
        @R.function(pure=False)
        def print(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.print(x, format="x: {}")
            return x

        @R.function(pure=False)
        def assert_func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.assert_op(R.const(False, dtype="bool"), x, format="x: {}")
            return x

    tvm.ir.assert_structural_equal(Unsugared, Sugared)


def test_function_with_non_void_return_type_must_be_assigned():
    """Non-void results must be assigned to a variable"""

    with pytest.raises(tvm.error.DiagnosticError):

        @R.function(pure=False)
        def func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.add(x, x)
            return x


def test_function_with_void_return_type_in_if_else():
    """Last statement in if/else may be a void return"""

    @I.ir_module
    class Unsugared:
        @R.function(pure=False)
        def conditional(
            x: R.Tensor((), "int32"), condition: R.Tensor((), "bool")
        ) -> R.Tensor((), "int32"):
            if condition:
                y = R.print(x, format="True condition: {}")
            else:
                y = R.print(x, format="False condition: {}")
            return x

    @I.ir_module
    class Sugared:
        @R.function(pure=False)
        def conditional(
            x: R.Tensor((), "int32"), condition: R.Tensor((), "bool")
        ) -> R.Tensor((), "int32"):
            if condition:
                R.print(x, format="True condition: {}")
            else:
                R.print(x, format="False condition: {}")
            return x

    _check(Sugared, Unsugared)


def test_call_pure_packed():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        z = R.call_pure_packed("vm.builtin.copy", x, sinfo_args=R.Tensor((32, 32), "float32"))
        return z

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        z = bb.emit(
            R.call_pure_packed("vm.builtin.copy", x, sinfo_args=[R.Tensor((32, 32), "float32")])
        )
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_call_pure_packed_returning_object():
    @R.function
    def foo() -> R.Object:
        z = R.call_pure_packed("dummy_func", sinfo_args=R.Object)
        return z

    bb = relax.BlockBuilder()
    with bb.function("foo", params=[]):
        z = bb.emit(R.call_pure_packed("dummy_func", sinfo_args=[relax.ObjectStructInfo()]))
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_private_function():
    @I.ir_module
    class Addition:
        @R.function(private=True)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            return y

    x = relax.Var("x", R.Tensor((), "int32"))
    bb = relax.BlockBuilder()
    with bb.function("main", (x), private=True):
        y = bb.emit(R.add(x, x))
        bb.emit_func_output(y)

    _check(Addition, bb.get())


def test_private_function_with_global_symbol_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @I.ir_module
        class Addition:
            @R.function(private=True)
            def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                # it is an error to simultaneously mark a function private
                # and give it a global symbol manually
                R.func_attr({"global_symbol": "main"})
                y = R.add(x, x)
                return y

        # should not execute
        _check(Addition)


def test_private_function_with_global_symbol_no_module_fail():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.function(private=True)
        def func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"global_symbol": "main"})
            y = R.add(x, x)
            return y

        # should not execute
        _check(func)


def test_macro_hygienic():
    x = R.prim_value(2)

    @R.macro(hygienic=True)
    def alloc_and_shape(dtype: str):
        alloc = R.builtin.alloc_tensor(R.shape([4, 4]), runtime_device_index=x, dtype=dtype)
        shape = R.shape_of(alloc)
        return shape

    x = R.prim_value(1)

    @R.function(private=True)
    def func(z: R.Tensor((4, 4), "float32")):
        shape = alloc_and_shape(dtype="float32")
        return shape

    @R.function(private=True)
    def expect(z: R.Tensor((4, 4), dtype="float32")) -> R.Shape([4, 4]):
        alloc: R.Tensor((4, 4), dtype="float32") = R.builtin.alloc_tensor(
            R.shape([4, 4]),
            R.dtype("float32"),
            R.prim_value(2),  # Make sure prim_value is 2
        )
        shape: R.Shape([4, 4]) = R.shape_of(alloc)
        shape_1: R.Shape([4, 4]) = shape
        return shape_1

    _check(func, expect)


def test_macro_non_hygienic():
    global global_x_var  # Lookup doesn't find this variable if it's not global

    global_x_var = R.prim_value(2)

    @R.macro(hygienic=False)
    def alloc_and_shape(dtype: str):
        alloc = R.builtin.alloc_tensor(
            R.shape([4, 4]), runtime_device_index=global_x_var, dtype=dtype
        )
        shape = R.shape_of(alloc)
        return shape

    global_x_var = R.prim_value(1)

    @R.function(private=True)
    def func(z: R.Tensor((4, 4), "float32")):
        shape = alloc_and_shape(dtype="float32")
        return shape

    @R.function(private=True)
    def expect(z: R.Tensor((4, 4), dtype="float32")) -> R.Shape([4, 4]):
        alloc: R.Tensor((4, 4), dtype="float32") = R.builtin.alloc_tensor(
            R.shape([4, 4]),
            R.dtype("float32"),
            R.prim_value(1),  # Make sure prim_value is 1
        )
        shape: R.Shape([4, 4]) = R.shape_of(alloc)
        shape_1: R.Shape([4, 4]) = shape
        return shape_1

    _check(func, expect)


def test_macro_no_variable_leak():
    with pytest.raises(tvm.error.DiagnosticError):

        @R.macro(hygienic=True)
        def add_two(value):
            x = value + R.const(1)  # `x` defined in macro
            y = x + R.const(1)
            return y

        @R.function(private=True)
        def func(t: R.Tensor((), "int32")):
            u = add_two(t)
            return x  # Should be undefined here


def test_reused_extern_func():
    """ExternFunc lookups can become bindings in EliminateCommonSubexpr"""

    @R.function(private=True)
    def parsed(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
        func = R.ExternFunc("extern_func")
        gv0 = R.call_dps_packed(func, x, R.Tensor((128, 128), dtype="float32"))
        gv1 = R.call_dps_packed(func, gv0, R.Tensor((128, 128), dtype="float32"))
        return gv1

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x], private=True):
        func = bb.emit(relax.ExternFunc("extern_func"))
        y = bb.emit(relax.call_dps_packed(func, x, out_sinfo=R.Tensor((128, 128), "float32")))
        z = bb.emit(relax.call_dps_packed(func, y, out_sinfo=R.Tensor((128, 128), "float32")))
        bb.emit_func_output(z)

    expected = bb.get()["main"]

    _check(parsed, expected)


def test_extern_func_in_module():
    """Module-level parsing may produce function bindings"""

    @I.ir_module
    class parsed_module:
        my_ext = R.ExternFunc("my_ext")

        @R.function
        def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
            return a

    @R.function
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
        return a

    expected = tvm.IRModule({"my_ext": relax.ExternFunc("my_ext"), "func": func})

    _check(parsed_module, expected)


def test_define_relax_function_using_global_var():
    """A @R.function may call a GlobalVar

    When parsing a @R.function, the function's body may reference
    GlobalVar instances available in the calling python scope.  The
    resulting function should pass TVMScript's well-formed check, as
    the GlobalVar may be available in the IRModule for which the
    function is being defined.
    """

    @I.ir_module
    class DefinedAllAtOnce:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return DefinedAllAtOnce.subroutine(A, B)

        @R.function(private=True)
        def subroutine(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return R.matmul(A, B)

    @I.ir_module
    class MainDefinedLater:
        @R.function(private=True)
        def subroutine(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return R.matmul(A, B)

    subroutine_gvar = MainDefinedLater.get_global_var("subroutine")

    @R.function
    def main(A: R.Tensor, B: R.Tensor):
        return subroutine_gvar(A, B)

    MainDefinedLater["main"] = main

    tvm.ir.assert_structural_equal(DefinedAllAtOnce, MainDefinedLater)


def test_function_attributes_are_defined():
    """func.attrs defaults to an empty DictAttrs"""

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor, shape: R.Shape(["m", "n"])):
            output = Module.subroutine(x, shape)
            return output

        @R.function
        def subroutine(x: R.Tensor, _: R.Shape(["m", "n"])) -> R.Tensor(["m", "n"]):
            q = x
            m, n = T.int64(), T.int64()
            z = R.match_cast(q, R.Tensor((m, n)))
            w = z
            return w

    for gvar, func in Module.functions.items():
        assert func.attrs is not None


@pytest.mark.xfail(reason="Bug: Implicit bounds not provided when parsing")
def test_function_symbolic_variables_are_annotated():
    """Symbolic variables must be exposed for struct inference

    Because Relax struct inference is performed while the function is
    being built, all constraints on symbolic variables that are used
    for simplifications must be provided to the analyzer.
    """

    @R.function(private=True)
    def inferred_sinfo(A: R.Tensor(["extent"])):
        extent = T.int64()
        output = R.strided_slice(A, [0], [0], [extent - 1])
        return output

    @R.function(private=True)
    def expected(A: R.Tensor(["extent"])) -> R.Tensor(["extent-1"]):
        extent = T.int64()
        output: R.Tensor([extent - 1]) = R.strided_slice(A, [0], [0], [extent - 1])
        return output

    tvm.ir.assert_structural_equal(inferred_sinfo, expected)


def test_conditional_may_use_symbolic_variables_from_function_scope():
    """Symbolic variables from function scope may be used in branch

    This is a regression test.  In earlier implementations, the
    branches of `relax::If` were normalized with
    `EraseToWellDefinedInScope`, using a fresh variable scope.  While
    this had the intended behavior of preventing variables defined in
    a single branch from being usable outside of the conditional, it
    also caused the conditional's branches to treat function-scope
    symbolic variables as if they were undefined.

    """

    @R.function(private=True)
    def explicit_sinfo(
        A: R.Tensor(["N"], "float32"),
        B: R.Tensor(["N"], "float32"),
        cond: R.Prim("bool"),
    ) -> R.Tensor(["N"], "float32"):
        N = T.int64()

        if cond:
            out: R.Tensor([N], "float32") = A + B
        else:
            out: R.Tensor([N], "float32") = A * B

        return out

    @R.function(private=True)
    def inferred_sinfo(
        A: R.Tensor(["N"], "float32"),
        B: R.Tensor(["N"], "float32"),
        cond: R.Prim("bool"),
    ):
        N = T.int64()
        if cond:
            out = A + B
        else:
            out = A * B

        return out

    tvm.ir.assert_structural_equal(explicit_sinfo, inferred_sinfo)


def test_return_from_dataflow_block():
    """Return statements imply

    The `R.output` statement in a `R.dataflow()` block marks a
    variable that should be a `relax.Var` instead of a
    `relax.DataflowVar`, allowing it to be used outside of the
    `DataflowBlock` that defined it.  A relax function's output is not
    part of any binding, and must not contain any `DataflowVar`, so
    these are exposed implicitly.

    """

    @R.function(private=True)
    def output_then_return(A: R.Tensor([16], "float16")):
        with R.dataflow():
            B = R.add(A, A)
            C = R.multiply(B, B)
            R.output(C)

        return C

    @R.function(private=True)
    def return_inside_dataflow(A: R.Tensor([16], "float16")):
        with R.dataflow():
            B = R.add(A, A)
            C = R.multiply(B, B)
            return C

    tvm.ir.assert_structural_equal(output_then_return, return_inside_dataflow)


if __name__ == "__main__":
    tvm.testing.main()
