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


def test_sum():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1, 3), "float32"):
        gv: R.Tensor((1, 3), "float32") = R.sum(x, axis=[1, 3])
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.sum(x, axis=[1, 3]))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_sum_without_specified_axis():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((), "float32"):
        gv: R.Tensor((), "float32") = R.sum(x)
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.sum(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_sum_keep_dims():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1, 1, 3, 1), "float32"):
        gv: R.Tensor((1, 1, 3, 1), "float32") = R.sum(x, axis=[1, 3], keepdims=True)
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.sum(x, axis=[1, 3], keepdims=True))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_mean():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1, 3), "float32"):
        gv: R.Tensor((1, 3), "float32") = R.mean(x, axis=[1, 3])
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.mean(x, axis=[1, 3]))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_variance():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1,), "float32"):
        gv: R.Tensor((1,), "float32") = R.variance(x, axis=[-1, -2, -3])
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.variance(x, axis=[-1, -2, -3]))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_max():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
        gv: R.Tensor((1, 1, 1, 1), "float32") = R.variance(x, axis=[-1, -2, -3], keepdims=True)
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.variance(x, axis=[-1, -2, -3], keepdims=True))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_min():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1, 3, 4), "float32"):
        gv: R.Tensor((1, 3, 4), "float32") = R.min(x, axis=1)
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.min(x, axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_prod():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1, 3, 4), "float32"):
        gv: R.Tensor((1, 3, 4), "float32") = R.prod(x, axis=1)
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.prod(x, axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_std():
    @R.function
    def foo(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((1, 3, 4), "float32"):
        gv: R.Tensor((1, 3, 4), "float32") = R.std(x, axis=1)
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.std(x, axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_scan():
    @R.function
    def foo(x: R.Tensor((2, 3, 4), "float32")):
        lv = R.cumsum(x, axis=1, dtype="int32")
        gv = R.cumprod(lv, axis=1, dtype="int32")
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        lv = bb.emit(relax.op.cumsum(x, axis=1, dtype="int32"))
        gv = bb.emit(relax.op.cumprod(lv, axis=1, dtype="int32"))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
