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

import tvm
from tvm import tirx
from tvm.ir import prim
from tvm.relax.script import ir_builder as R
from tvm.s_tir.script import ir_builder as S
from tvm.script import ir as I
from tvm.script import relax as Rs
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import ir_builder as B


@pytest.mark.parametrize("builder", [B, S, R])
@pytest.mark.parametrize("previous", [True, False])
def test_function_frame_lifetime(builder, previous):
    with prim.OpConstFoldScope(previous):
        with IRBuilder():
            frame = builder.function_(decl=True)
            with frame as declaration:
                assert declaration.same_as(frame)
                assert not prim.op_const_fold_enabled()
                builder.func_name("main")
            assert prim.op_const_fold_enabled() == previous
            with frame as body:
                assert body.same_as(frame)
                assert not prim.op_const_fold_enabled()
                if builder is R:
                    R.func_ret_value(prim.const(1))
                else:
                    builder.evaluate(prim.const(1) + prim.const(2))
            assert prim.op_const_fold_enabled() == previous
    assert prim.op_const_fold_enabled()


@pytest.mark.parametrize("builder", [B, S, R])
def test_function_frame_exceptions(builder):
    with IRBuilder():
        frame = builder.function_()
    with pytest.raises(ValueError, match="No builder"):
        frame.__enter__()
    assert prim.op_const_fold_enabled()
    with pytest.raises(RuntimeError, match="body failure"):
        with IRBuilder():
            with frame:
                assert not prim.op_const_fold_enabled()
                raise RuntimeError("body failure")
    assert prim.op_const_fold_enabled()

    def fail_exit():
        raise RuntimeError("exit failure")

    with pytest.raises(RuntimeError, match="exit failure"):
        with IRBuilder():
            with builder.function_() as frame:
                frame.add_callback(fail_exit)
                if builder is R:
                    R.func_ret_value(prim.const(1))
                else:
                    builder.evaluate(0)
    assert prim.op_const_fold_enabled()


def test_nested_function_frames():
    with IRBuilder():
        with B.prim_func():
            assert not prim.op_const_fold_enabled()
            with prim.OpConstFoldScope(True):
                with IRBuilder():
                    with R.function():
                        assert not prim.op_const_fold_enabled()
                        R.func_ret_value(prim.const(1))
                    assert prim.op_const_fold_enabled()
            assert not prim.op_const_fold_enabled()
            B.evaluate(0)
    assert prim.op_const_fold_enabled()


@pytest.mark.parametrize("dialect", [T, Ts])
def test_parser_preserves_ir_arithmetic(dialect):
    @dialect.prim_func
    def main():
        T.evaluate(T.int32(1) + T.int32(2))
        T.evaluate(1 + 2)

    body = main.body
    if isinstance(body, tvm.s_tir.SBlockRealize):
        body = body.block.body
    assert isinstance(body.seq[0].value, prim.Add)
    assert body.seq[1].value.value == 3
    assert prim.op_const_fold_enabled()
    tvm.ir.assert_structural_equal(main, tvm.script.from_source(main.script()))


def test_nested_relax_parser_frames():
    @I.ir_module
    class Module:
        @Rs.function
        def outer(x: Rs.Tensor((T.int64(1) + T.int64(2),), "float32")):
            @Rs.function
            def inner(y: Rs.Tensor((T.int64(1) + T.int64(2),), "float32")):
                return y

            return inner(x)

    assert prim.op_const_fold_enabled()
    assert isinstance(Module["outer"].params[0].ty.shape[0], prim.Add)
    tvm.ir.assert_structural_equal(Module, tvm.script.from_source(Module.script()))


@pytest.mark.parametrize("dtype", ["int32", "int64", "float16", "float32", "float64"])
@pytest.mark.parametrize("node_type", [prim.Add, prim.Sub, prim.Mul, prim.LT, prim.EQ])
def test_binary_literal_roundtrip(dtype, node_type):
    value = node_type(prim.const(1, dtype), prim.const(2, dtype))
    function = tirx.PrimFunc([], tirx.Evaluate(value)).with_attr("global_symbol", "main")
    script = function.script()
    assert f"T.{dtype}(1" in script
    assert f"T.{dtype}(2" in script
    actual = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(function, actual)


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
@pytest.mark.parametrize("node_type", [prim.Add, prim.Sub, prim.Mul])
def test_identity_roundtrip(node_type, dtype):
    var = tirx.Var("x", dtype)
    value = node_type(var, prim.const(0 if node_type in (prim.Add, prim.Sub) else 1, dtype))
    function = tirx.PrimFunc([var], tirx.Evaluate(value)).with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(function, tvm.script.from_source(function.script()))


def test_explicit_call_and_division_roundtrip():
    for dtype in ("int32", "float32", "float64"):
        value = prim.Div(prim.const(6, dtype), prim.const(2, dtype))
        function = tirx.PrimFunc([], tirx.Evaluate(value)).with_attr("global_symbol", "main")
        script = function.script()
        if dtype == "int32":
            assert "T.Div(6, 2)" in script
        else:
            assert " / " in script
        tvm.ir.assert_structural_equal(function, tvm.script.from_source(script))
    value = prim.Min(prim.const(1), prim.const(2))
    function = tirx.PrimFunc([], tirx.Evaluate(value)).with_attr("global_symbol", "main")
    script = function.script()
    assert "T.min(1, 2)" in script
    tvm.ir.assert_structural_equal(function, tvm.script.from_source(script))
