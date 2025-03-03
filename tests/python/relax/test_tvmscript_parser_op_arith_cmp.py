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

from typing import Optional, Union, Callable

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.script import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.from_source(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


(unary_arith_op,) = tvm.testing.parameters(
    (relax.op.abs,),
    (relax.op.acos,),
    (relax.op.acosh,),
    (relax.op.asin,),
    (relax.op.asinh,),
    (relax.op.atan,),
    (relax.op.atanh,),
    (relax.op.ceil,),
    (relax.op.cos,),
    (relax.op.cosh,),
    (relax.op.exp,),
    (relax.op.floor,),
    (relax.op.log,),
    (relax.op.negative,),
    (relax.op.round,),
    (relax.op.rsqrt,),
    (relax.op.sigmoid,),
    (relax.op.sign,),
    (relax.op.sin,),
    (relax.op.sinh,),
    (relax.op.square,),
    (relax.op.sqrt,),
    (relax.op.tan,),
    (relax.op.tanh,),
)


def test_unary_arith(unary_arith_op: Callable):
    @R.function
    def foo(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = unary_arith_op(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(unary_arith_op(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


(unary_check_op,) = tvm.testing.parameters(
    (relax.op.isfinite,),
    (relax.op.isinf,),
    (relax.op.isnan,),
)


def test_unary_check(unary_check_op: Callable):
    @R.function
    def foo(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
        gv: R.Tensor((2, 3), "bool") = unary_check_op(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(unary_check_op(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


(binary_arith_op,) = tvm.testing.parameters(
    (relax.op.add,),
    (relax.op.divide,),
    (relax.op.floor_divide,),
    (relax.op.multiply,),
    (relax.op.power,),
    (relax.op.subtract,),
    (relax.op.maximum,),
    (relax.op.minimum,),
)


def test_binary_arith(binary_arith_op: Callable):
    @R.function
    def foo(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = binary_arith_op(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, y]):
        gv = bb.emit(binary_arith_op(x, y))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


(binary_cmp_op,) = tvm.testing.parameters(
    (relax.op.equal,),
    (relax.op.greater,),
    (relax.op.greater_equal,),
    (relax.op.less,),
    (relax.op.less_equal,),
    (relax.op.not_equal,),
)


def test_binary_cmp(binary_cmp_op: Callable):
    @R.function
    def foo(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "bool"):
        gv: R.Tensor((2, 3), "bool") = binary_cmp_op(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, y]):
        gv = bb.emit(binary_cmp_op(x, y))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_relax_ewise_fma():
    @R.function
    def foo(
        x: R.Tensor((2, 3, 4), dtype="float32"),
        y: R.Tensor((2, 3, 4), dtype="float32"),
        z: R.Tensor((2, 3, 4), dtype="float32"),
    ) -> R.Tensor((2, 3, 4), dtype="float32"):
        gv: R.Tensor((2, 3, 4), dtype="float32") = R.ewise_fma(x, y, z)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))
    z = relax.Var("z", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, y, z]):
        gv = bb.emit(relax.op.ewise_fma(x, y, z))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
