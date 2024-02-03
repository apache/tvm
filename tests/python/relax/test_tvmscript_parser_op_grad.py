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
# "AS IS" BASIS, WITHOUT WA`RRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import Optional, Union

import tvm
import tvm.testing
from tvm import IRModule, relax
from tvm.script.parser import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.from_source(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_nll_loss_backward():
    @R.function
    def foo(
        output_grad: R.Tensor((3, 10, 10), dtype="float32"),
        predictions: R.Tensor((3, 5, 10, 10), dtype="float32"),
        targets: R.Tensor((3, 10, 10), dtype="int64"),
        weights: R.Tensor((5,), dtype="float32"),
    ) -> R.Tensor((3, 5, 10, 10), dtype="float32"):
        gv: R.Tensor((3, 5, 10, 10), dtype="float32") = R.grad.nll_loss_backward(
            output_grad, predictions, targets, weights, "mean", -1
        )
        return gv

    output_grad = relax.Var("output_grad", R.Tensor((3, 10, 10), "float32"))
    predictions = relax.Var("predictions", R.Tensor((3, 5, 10, 10), "float32"))
    targets = relax.Var("targets", R.Tensor((3, 10, 10), "int64"))
    weights = relax.Var("weights", R.Tensor((5,), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [output_grad, predictions, targets, weights]):
        gv = bb.emit(
            relax.op.grad.nll_loss_backward(
                output_grad, predictions, targets, weights, reduction="mean", ignore_index=-1
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_nll_loss_backward_no_weights():
    @R.function
    def foo(
        output_grad: R.Tensor((3, 10, 10), dtype="float32"),
        predictions: R.Tensor((3, 5, 10, 10), dtype="float32"),
        targets: R.Tensor((3, 10, 10), dtype="int64"),
    ) -> R.Tensor((3, 5, 10, 10), dtype="float32"):
        gv: R.Tensor((3, 5, 10, 10), dtype="float32") = R.grad.nll_loss_backward(
            output_grad, predictions, targets, reduction="mean", ignore_index=-1
        )
        return gv

    output_grad = relax.Var("output_grad", R.Tensor((3, 10, 10), "float32"))
    predictions = relax.Var("predictions", R.Tensor((3, 5, 10, 10), "float32"))
    targets = relax.Var("targets", R.Tensor((3, 10, 10), "int64"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [output_grad, predictions, targets]):
        gv = bb.emit(
            relax.op.grad.nll_loss_backward(
                output_grad, predictions, targets, reduction="mean", ignore_index=-1
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_max_pool2d_backward():
    @R.function
    def foo(
        output_grad: R.Tensor((3, 2, 6, 5), "float32"), data: R.Tensor((3, 2, 10, 10), "float32")
    ):
        gv = R.grad.max_pool2d_backward(
            output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True
        )
        return gv

    output_grad = relax.Var("output_grad", R.Tensor((3, 2, 6, 5), "float32"))
    data = relax.Var("data", R.Tensor((3, 2, 10, 10), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [output_grad, data]):
        gv = bb.emit(
            relax.op.grad.max_pool2d_backward(
                output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_avg_pool2d_backward():
    @R.function
    def foo(
        output_grad: R.Tensor((3, 2, 6, 5), "float32"), data: R.Tensor((3, 2, 10, 10), "float32")
    ):
        gv = R.grad.avg_pool2d_backward(
            output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True
        )
        return gv

    output_grad = relax.Var("output_grad", R.Tensor((3, 2, 6, 5), "float32"))
    data = relax.Var("data", R.Tensor((3, 2, 10, 10), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [output_grad, data]):
        gv = bb.emit(
            relax.op.grad.avg_pool2d_backward(
                output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
