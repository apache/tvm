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
from tvm import relax, tir
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder.base import IRBuilder


def test_function_simple():
    """
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        out = R.call_tir("extern_func", x, R.Tensor((128, 128), dtype="float32"))
        return out
    """
    # create with Script IRBuilder
    with IRBuilder() as ir_builder:
        with R.function():
            R.func_name("foo")
            R.func_attr({"Primitive": 1})
            x = R.arg("x", relax.TensorStructInfo((128, 128), "float32"))
            R.func_ret_struct_info(relax.TensorStructInfo(dtype="float32", ndim=2))
            out = R.emit(
                R.call_tir("extern_func", x, relax.TensorStructInfo((128, 128), dtype="float32"))
            )
            IRBuilder.name("out", out)
            R.func_ret_value(out)
    func = ir_builder.get()
    # create with BlockBuilder
    x = relax.Var("x", relax.TensorStructInfo((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), attrs={"Primitive": 1}):
        out = bb.emit(
            relax.call_tir("extern_func", x, relax.TensorStructInfo((128, 128), dtype="float32"))
        )
        bb.emit_func_output(out)
    mod = bb.get()

    tvm.ir.assert_structural_equal(func, mod["foo"])
    # check names
    assert func.params[0].name_hint == "x"
    assert func.body.body.name_hint == "out"


def test_match_cast():
    """
    @R.function
    def foo(x: R.Tensor(None, "float32"), y: R.Tensor(None, "float32")):
        m = T.var("int64")
        n = T.var("int64")
        _ = R.match_cast(x, R.Tensor((m,), "float32"))
        y1 = R.match_cast(x, R.Tensor((n,), "float32"))
        return (m, n * 2)
    """
    # create with Script IRBuilder
    with IRBuilder() as ir_builder:
        with R.function():
            R.func_name("foo")
            x = R.arg("x", relax.TensorStructInfo(ndim=-1, dtype="float32"))
            y = R.arg("y", relax.TensorStructInfo(ndim=-1, dtype="float32"))
            m = tir.Var("m", dtype="int64")
            n = tir.Var("n", dtype="int64")
            _ = R.emit_match_cast(x, relax.TensorStructInfo((m,), "float32"))
            y1 = R.emit_match_cast(y, relax.TensorStructInfo((n,), "float32"))
            IRBuilder.name("y1", y1)
            R.func_ret_value(relax.ShapeExpr([m, n * 2]))
    func = ir_builder.get()

    # create with BlockBuilder
    x = relax.Var("x", relax.TensorStructInfo(dtype="float32", ndim=-1))
    y = relax.Var("y", relax.TensorStructInfo(dtype="float32", ndim=-1))
    m = tir.Var("m", dtype="int64")
    n = tir.Var("n", dtype="int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        _ = bb.match_cast(x, relax.TensorStructInfo((m,), "float32"))
        y1 = bb.match_cast(y, relax.TensorStructInfo((n,), "float32"))
        bb.emit_func_output(relax.ShapeExpr([m, n * 2]))
    mod = bb.get()

    tvm.ir.assert_structural_equal(func, mod["foo"])


def test_dataflow_block():
    """
    @R.function
    def foo(x: Tensor((128, 128), "float32")) -> Tensor(None, "float32", ndim = 2):
        # block 0
        with R.dataflow():
            lv0 = R.call_tir("extern_func", (x,), R.Tensor((128, 128), dtype="float32"))
            gv: Tensor((128, 128), "float32") = lv0
            R.output(gv)
        return gv
    """
    # create with Script IRBuilder
    with IRBuilder() as ir_builder:
        with R.function():
            R.func_name("foo")
            x = R.arg("x", relax.TensorStructInfo((128, 128), "float32"))
            with R.dataflow() as df:
                lv0 = R.emit(
                    R.call_tir(
                        "extern_func", x, relax.TensorStructInfo((128, 128), dtype="float32")
                    )
                )
                IRBuilder.name("lv0", lv0)
                gv = R.emit(lv0)
                IRBuilder.name("gv", gv)
                R.output(gv)
            (gv,) = df.output_vars
            R.func_ret_value(gv)
    func = ir_builder.get()

    # create with BlockBuilder
    x = relax.Var("x", relax.TensorStructInfo((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_tir(
                    "extern_func", x, relax.TensorStructInfo((128, 128), dtype="float32")
                )
            )
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(func, bb.get()["foo"])


def test_regression_py_print():
    # Test that the py_print directs to python builtin print
    from tvm.script.ir_builder.relax.ir import py_print  # pylint: disable=import-outside-toplevel

    assert py_print == print


if __name__ == "__main__":
    tvm.testing.main()
