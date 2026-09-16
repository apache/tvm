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
# ruff: noqa: F841
import operator

import numpy as np
import pytest
import torch
from test_frontend_from_fx import (
    UnaryModule,
    activation_cases,
    constants,
    make_expected,
    verify_numerically,
)
from torch import nn
from torch.export import export
from torch.nn import Module

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.testing import env


def verify_model(
    torch_model,
    example_args,
    binding,
    expected,
    dynamic_shapes=None,
    run_ep_decomposition=True,
    keep_params_as_input=False,
    unwrap_unit_return_tuple=False,
    no_bind_return_tuple=False,
    map_free_vars=False,
    custom_convert_map=None,
):
    exported_program = export(torch_model, args=example_args, dynamic_shapes=dynamic_shapes)
    mod = from_exported_program(
        exported_program,
        run_ep_decomposition=run_ep_decomposition,
        keep_params_as_input=keep_params_as_input,
        unwrap_unit_return_tuple=unwrap_unit_return_tuple,
        no_bind_return_tuple=no_bind_return_tuple,
        custom_convert_map=custom_convert_map,
    )

    binding = {k: tvm.runtime.tensor(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm.ir.assert_structural_equal(mod, expected, map_free_vars=map_free_vars)


def verify_model_numerically(
    torch_model,
    example_args,
    rtol=1e-7,
    atol=1e-7,
    *,
    dynamic_shapes=None,
    input_sets=None,
    run_ep_decomposition=True,
):
    exported_program = export(torch_model, args=example_args, dynamic_shapes=dynamic_shapes)
    mod = from_exported_program(exported_program, run_ep_decomposition=run_ep_decomposition)
    verify_numerically(mod, torch_model, example_args, input_sets=input_sets, rtol=rtol, atol=atol)
    return mod


operator_basic_unary = [
    (torch.abs, R.abs),
    (torch.acos, R.acos),
    (torch.acosh, R.acosh),
    (torch.asin, R.asin),
    (torch.asinh, R.asinh),
    (torch.atan, R.atan),
    (torch.atanh, R.atanh),
    (torch.bitwise_not, R.bitwise_not),
    (torch.ceil, R.ceil),
    (torch.cos, R.cos),
    (torch.cosh, R.cosh),
    (torch.erf, R.erf),
    (torch.exp, R.exp),
    (torch.floor, R.floor),
    (torch.ops.aten.gelu, R.nn.gelu),
    (torch.log, R.log),
    (torch.neg, R.negative),
    (torch.relu, R.nn.relu),
    (torch.round, R.round),
    (torch.rsqrt, R.rsqrt),
    (torch.sigmoid, R.sigmoid),
    (torch.sin, R.sin),
    (torch.sinh, R.sinh),
    (torch.sign, R.sign),
    (torch.sqrt, R.sqrt),
    (torch.tan, R.tan),
    (torch.tanh, R.tanh),
    (torch.trunc, R.trunc),
]


@pytest.mark.parametrize("pytorch_op, relax_op", operator_basic_unary)
def test_basic_unary_ops(pytorch_op, relax_op):
    dtype = "int32" if pytorch_op is torch.bitwise_not else "float32"
    example_args = (torch.ones(1, 3, 10, 10, dtype=getattr(torch, dtype)),)

    class UnaryOp(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype=dtype)) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype=dtype)
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype=dtype) = relax_op(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype=dtype)) = (lv,)
                R.output(gv)
            return gv

    verify_model(UnaryOp(), example_args, {}, expected)


@pytest.mark.parametrize(
    "dtype, decimals",
    [
        pytest.param(torch.float32, (0, 1, -1), id="ties-to-even"),
        pytest.param(torch.float16, (4, 5, -5), id="float16-scaling"),
    ],
)
def test_round_decimals(dtype, decimals):
    class RoundDecimals(Module):
        def forward(self, x):
            return tuple(torch.round(x, decimals=d) for d in decimals)

    # Negative decimals catch reciprocal-scaling errors; float16 catches overflow
    # both in the scaled input (4) and in the scale itself (5 and -5).
    x = torch.tensor([0.5, 1.5, 2.5, -0.5, -2.5, 25.0, 125.0, 165.0, 2.25], dtype=dtype)
    verify_model_numerically(RoundDecimals(), (x,), rtol=1e-6, atol=1e-6)


def test_round_decimals_large():
    """An overflowing scale must remain importable in either direction."""

    class RoundDecimals(Module):
        def forward(self, x):
            return torch.round(x, decimals=309), torch.round(x, decimals=-309)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2,), "float32")):
            with R.dataflow():
                scaled_up = R.multiply(x, R.const(float("inf"), "float32"))
                rounded_up = R.round(scaled_up)
                positive = R.divide(rounded_up, R.const(float("inf"), "float32"))
                scaled_down = R.divide(x, R.const(float("inf"), "float32"))
                rounded_down = R.round(scaled_down)
                negative = R.multiply(rounded_down, R.const(float("inf"), "float32"))
                result = (positive, negative)
                R.output(result)
            return result

    verify_model(RoundDecimals(), (torch.ones(2),), {}, Expected)


operator_bool_unary = [
    (torch.isinf, R.isinf),
    (torch.isnan, R.isnan),
]


@pytest.mark.parametrize("pytorch_op, relax_op", operator_bool_unary)
def test_bool_unary_ops(pytorch_op, relax_op):
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class UnaryOp(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="bool")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    verify_model(UnaryOp(), example_args, {}, expected)


def test_sqrt_integer_input():
    """Test that sqrt operation works with integer tensors by auto-converting to float."""
    example_args = (torch.tensor([[4, 9, 16, 25]], dtype=torch.int64),)

    class SqrtIntModel(Module):
        def forward(self, input):
            return torch.sqrt(input)

    @tvm.script.ir_module
    class expected_int64:
        @R.function
        def main(input_1: R.Tensor((1, 4), dtype="int64")) -> R.Tuple(
            R.Tensor((1, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.astype(input_1, dtype="float32")
                lv1: R.Tensor((1, 4), dtype="float32") = R.sqrt(lv)
                gv: R.Tuple(R.Tensor((1, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(SqrtIntModel(), example_args, {}, expected_int64)


@pytest.mark.parametrize("torch_op,expected", activation_cases(exported=True))
def test_extended_unary_ops(torch_op, expected):
    shape = (2, 3)
    expected = make_expected([(shape, "float32")], expected, exported=True)
    verify_model(
        UnaryModule(torch_op), (torch.randn(shape),), {}, expected, run_ep_decomposition=False
    )


def test_clamp():
    example_args = (torch.randn(1, 3, 10, 10),)

    # clamp
    class Clamp(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.1, max=0.5)

    @tvm.script.ir_module
    class expected_clamp:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    input,
                    R.prim_value(T.float64(0.10000000000000001)),
                    R.prim_value(T.float64(0.5)),
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Clamp(), example_args, {}, expected_clamp)

    class ClampMinOnly(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.5, max=None)

    @tvm.script.ir_module
    class expected_clamp_min_only:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    input, R.prim_value(T.float64(0.5)), R.prim_value(T.float64("inf"))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(ClampMinOnly(), example_args, {}, expected_clamp_min_only)

    class ClampTensors(Module):
        def forward(self, input):
            return torch.clamp(input, min=input, max=input)

    @tvm.script.ir_module
    class expected_clamp_tensors:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.broadcast_to(
                    input, R.shape([1, 3, 10, 10])
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.maximum(input, lv)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.broadcast_to(
                    input, R.shape([1, 3, 10, 10])
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.minimum(lv1, lv2)
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    lv3, R.prim_value(T.float64("-inf")), R.prim_value(T.float64("inf"))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    verify_model(ClampTensors(), example_args, {}, expected_clamp_tensors)


def test_softplus():
    class Softplus(Module):
        def forward(self, input):
            return torch.nn.functional.softplus(input, 1.0, 20.0)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    x, R.const(1.0, "float32")
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(lv)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv1, R.const(1.0, "float32"))
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(lv2)
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv3, R.const(1.0, "float32")
                )
                lv5: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(
                    lv, R.const(20.0, "float32")
                )
                lv6: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv5, x, lv4)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv6,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softplus(), example_args, {}, expected)


def test_leakyrelu():
    model = UnaryModule(lambda x: torch.nn.functional.leaky_relu(x, 0.02))
    expected = make_expected(
        [((2, 3), "float32")], lambda x: relax.op.nn.leakyrelu(x, alpha=0.02), exported=True
    )
    verify_model(model, (torch.randn(2, 3),), {}, expected)


def test_logaddexp():
    class LogAddExp(Module):
        def forward(self, x, y):
            return torch.logaddexp(x, y)

    def expected(x, y):
        emit = relax.BlockBuilder.current().emit
        op = relax.op
        choose_x = emit(op.greater_equal(x, y))
        high, low = emit(op.where(choose_x, x, y)), emit(op.where(choose_x, y, x))
        finite = emit(op.not_equal(op.abs(x), relax.const(float("inf"), "float32")))
        not_nan = emit(op.equal(x, x))
        exceptional = emit(op.logical_not(op.multiply(not_nan, finite)))
        same_exception = emit(op.logical_and(exceptional, op.equal(x, y)))
        result = emit(
            op.add(
                high, op.log(op.add(op.exp(op.subtract(low, high)), relax.const(1.0, "float32")))
            )
        )
        return op.where(same_exception, x, result)

    info = [((2, 3), "float32")] * 2
    verify_model(
        LogAddExp(),
        (torch.randn(2, 3), torch.randn(2, 3)),
        {},
        make_expected(info, expected, exported=True),
    )


def test_atan2():
    class Atan2(Module):
        def forward(self, lhs, rhs):
            return torch.atan2(lhs, rhs)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.atan2(lhs, rhs)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(Atan2(), example_args, {}, expected)


@pytest.mark.parametrize(
    "torch_op, relax_op",
    [
        (torch.logical_and, R.logical_and),
        (torch.logical_or, R.logical_or),
        (torch.logical_xor, R.logical_xor),
    ],
)
def test_logical_binary(torch_op, relax_op):
    class LogicalBinary(Module):
        def forward(self, lhs, rhs):
            return torch_op(lhs, rhs)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(lhs, dtype="bool")
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(rhs, dtype="bool")
                lv2: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(lv, lv1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(LogicalBinary(), example_args, {}, expected)


def test_logical_not():
    class LogicalNot(Module):
        def forward(self, input):
            return torch.logical_not(input)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="bool")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(input, dtype="bool")
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.logical_not(lv)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(LogicalNot(), example_args, {}, expected)


def test_pow_integer():
    class Pow(Module):
        def forward(self, input):
            return input.pow(4)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(input: R.Tensor((4,), dtype="int64")) -> R.Tuple(R.Tensor((4,), dtype="int64")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int64") = R.multiply(input, input)
                lv1: R.Tensor((4,), dtype="int64") = R.multiply(lv, input)
                lv2: R.Tensor((4,), dtype="int64") = R.multiply(lv1, input)
                gv: R.Tuple(R.Tensor((4,), dtype="int64")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.tensor([-1, 1, 2, 3], dtype=torch.int64),)
    verify_model(Pow(), example_args, {}, expected)


def test_logsoftmax():
    class LogSoftmax(Module):
        def forward(self, input):
            return torch.nn.functional.log_softmax(input, dim=1)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.log_softmax(input_1, axis=1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(LogSoftmax(), example_args, {}, expected1)


def test_prelu():
    class Prelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.tensor([0.25]))

        def forward(self, x):
            return torch.nn.functional.prelu(x, self.alpha)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.reshape(
                    R.const([0.25], dtype="float32"), R.shape([1, 1, 1, 1])
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(x, R.const(0.0, "float32"))
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(lv, x)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv1, x, lv2)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Prelu(), example_args, {}, expected)


def test_softmax():
    class Softmax(Module):
        def forward(self, input):
            return torch.nn.functional.softmax(input, dim=1)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.softmax(input_1, axis=1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softmax(), example_args, {}, expected1)


def test_softsign():
    class Softsign(Module):
        def forward(self, input):
            return torch.nn.functional.softsign(input)

    @tvm.script.ir_module
    class expected_softsign:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                abs_val = R.abs(input)
                denom = R.add(abs_val, R.const(1.0, "float32"))
                result = R.divide(input, denom)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (result,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softsign(), example_args, {}, expected_softsign)


def test_softshrink():
    class Softshrink(Module):
        def forward(self, input):
            return torch.nn.functional.softshrink(input, lambd=0.5)

    @tvm.script.ir_module
    class expected_softshrink:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.abs(input)
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sign(input)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    lv2, R.const(0.5, "float32")
                )
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(input, lv3)
                lv5: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    input, R.const(0.0, "float32")
                )
                lv6: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv1, lv4, lv5)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv6,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softshrink(), example_args, {}, expected_softshrink)


@pytest.mark.parametrize(
    "torch_op,compare", [(torch.tril, relax.op.less_equal), (torch.triu, relax.op.greater_equal)]
)
def test_tril_triu(torch_op, compare):
    def expected(x):
        emit = relax.BlockBuilder.current().emit
        cols = emit(relax.op.expand_dims(relax.op.arange(0, 4, 1, dtype="int64"), axis=[-2]))
        rows = emit(relax.op.expand_dims(relax.op.arange(0, 3, 1, dtype="int64"), axis=[-1]))
        mask = emit(compare(relax.op.subtract(cols, rows), relax.const(1, "int64")))
        zero = emit(relax.const(0.0, "float32"))
        return relax.op.where(mask, x, zero)

    model = UnaryModule(lambda x: torch_op(x, 1))
    verify_model(
        model,
        (torch.randn(3, 4),),
        {},
        make_expected([((3, 4), "float32")], expected, exported=True),
    )


operator_binary_1 = [
    (operator.add, R.add),
    (operator.sub, R.subtract),
    (operator.mul, R.multiply),
    (operator.truediv, R.divide),
    (operator.floordiv, R.floor_divide),
    (torch.ops.aten.fmod, R.mod),
    (operator.pow, R.power),
    (operator.mod, R.floor_mod),
    (operator.and_, R.bitwise_and),
    (operator.or_, R.bitwise_or),
    (operator.xor, R.bitwise_xor),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_1)
def test_binary1(op, relax_op):
    dtype = (
        "int32"
        if "bitwise" in str(op) or op in (operator.and_, operator.or_, operator.xor)
        else "float32"
    )
    example_args1 = (
        torch.ones(10, 10, dtype=getattr(torch, dtype)),
        torch.ones(10, 10, dtype=getattr(torch, dtype)),
    )
    example_args2 = (torch.ones(10, 10, dtype=getattr(torch, dtype)),)

    class Binary1(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs, rhs):
            return self.op(lhs, rhs)

    @tvm.script.ir_module
    class expected_binary1:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype=dtype),
            rhs: R.Tensor((10, 10), dtype=dtype),
        ) -> R.Tuple(R.Tensor((10, 10), dtype=dtype)):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype=dtype) = relax_op(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype=dtype)) = (lv,)
                R.output(gv)
            return gv

    class Binary2(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs):
            return self.op(lhs, 1)

    @tvm.script.ir_module
    class expected_binary2:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype=dtype),
        ) -> R.Tuple(R.Tensor((10, 10), dtype=dtype)):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype=dtype) = relax_op(lhs, R.const(1, dtype))
                gv: R.Tuple(R.Tensor((10, 10), dtype=dtype)) = (lv,)
                R.output(gv)
            return gv

    verify_model(Binary1(op), example_args1, {}, expected_binary1)
    if op is operator.sub:
        verify_model(Binary2(op), example_args2, {}, expected_binary2)


operator_binary_scalar = [
    (torch.ops.aten.add.Scalar, R.add),
    (torch.ops.aten.bitwise_and.Scalar, R.bitwise_and),
    (torch.ops.aten.bitwise_or.Scalar, R.bitwise_or),
    (torch.ops.aten.bitwise_xor.Scalar, R.bitwise_xor),
    (torch.ops.aten.div.Scalar, R.divide),
    (torch.ops.aten.sub.Scalar, R.subtract),
    (torch.ops.aten.mul.Scalar, R.multiply),
    (torch.ops.aten.remainder.Scalar, R.floor_mod),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_scalar)
def test_binary_scalar(op, relax_op):
    dtype = (
        "int32"
        if "bitwise" in str(op) or op in (operator.and_, operator.or_, operator.xor)
        else "float32"
    )
    example_args = (torch.ones(1, 3, 10, 10, dtype=getattr(torch, dtype)),)

    class BinaryScalar(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs):
            return self.op(lhs, 1)

    @tvm.script.ir_module
    class expected_binary_scalar:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype=dtype),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype=dtype)):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype=dtype) = relax_op(lhs, R.const(1, dtype))
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype=dtype)) = (lv,)
                R.output(gv)
            return gv

    verify_model(BinaryScalar(op), example_args, {}, expected_binary_scalar)


operator_binary_promote = [(operator.sub, R.subtract)]


@pytest.mark.parametrize("op, relax_op", operator_binary_promote)
def test_binary_dtype_promotion(op, relax_op):
    """Ensure binary ops promote differing dtypes following PyTorch rules."""

    class BinaryPromoteLHS(Module):
        def forward(self, x):
            arange_val = torch.arange(x.shape[1])  # int64 by default
            return op(x, arange_val)

    @tvm.script.ir_module
    class expected_promote_lhs:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(3), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((3,), dtype="float32") = R.astype(lv, dtype="float32")
                lv2: R.Tensor((2, 3), dtype="float32") = relax_op(x, lv1)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    class BinaryPromoteRHS(Module):
        def forward(self, x):
            arange_val = torch.arange(x.shape[1])  # int64 by default
            return op(arange_val, x)

    @tvm.script.ir_module
    class expected_promote_rhs:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(3), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((3,), dtype="float32") = R.astype(lv, dtype="float32")
                lv2: R.Tensor((2, 3), dtype="float32") = relax_op(lv1, x)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32),)
    verify_model(BinaryPromoteLHS(), example_args, {}, expected_promote_lhs)
    verify_model(BinaryPromoteRHS(), example_args, {}, expected_promote_rhs)


operator_binary_2 = [
    (operator.eq, R.equal),
    (operator.ne, R.not_equal),
    (operator.lt, R.less),
    (operator.le, R.less_equal),
    (operator.gt, R.greater),
    (operator.ge, R.greater_equal),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_2)
def test_binary2(op, relax_op):
    example_args1 = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )
    example_args2 = (torch.randn(10, 10, dtype=torch.float32),)

    class Binary1(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs, rhs):
            return self.op(lhs, rhs)

    @tvm.script.ir_module
    class expected_binary1:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype="float32"),
            rhs: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = relax_op(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    class Binary2(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs):
            return self.op(lhs, 1.0)

    @tvm.script.ir_module
    class expected_binary2:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = relax_op(lhs, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Binary1(op), example_args1, {}, expected_binary1)
    if op is operator.lt:
        verify_model(Binary2(op), example_args2, {}, expected_binary2)


def test_binary3():
    example_args1 = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )
    example_args2 = (torch.randn(10, 10, dtype=torch.float32),)

    # Max
    class Max1(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    @I.ir_module
    class expected_max1:
        @R.function
        def main(
            inp_0: R.Tensor((10, 10), dtype="float32"),
            inp_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.maximum(inp_0, inp_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Max1(), example_args1, {}, expected_max1)

    # Min
    class Min1(Module):
        def forward(self, x, y):
            return torch.min(x, y)

    @I.ir_module
    class expected_min1:
        @R.function
        def main(
            inp_0: R.Tensor((10, 10), dtype="float32"),
            inp_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.minimum(inp_0, inp_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Min1(), example_args1, {}, expected_min1)

    # RSub
    class RSub1(Module):
        def forward(self, x, y):
            return torch.rsub(x, y)

    class RSub2(Module):
        def forward(self, x):
            return torch.rsub(x, 5.0)

    @tvm.script.ir_module
    class expected_rsub1:
        @R.function
        def main(
            x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(y, x)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_rsub2:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(R.const(5.0, "float32"), x)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(RSub1(), example_args1, {}, expected_rsub1)
    verify_model(RSub2(), example_args2, {}, expected_rsub2)


# IsIn


def test_isin():
    class IsInModel(torch.nn.Module):
        def forward(self, x, test_elements):
            return torch.isin(x, test_elements)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            x: R.Tensor((10, 10), dtype="float32"), test_elements: R.Tensor((8,), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((10, 10, 1), dtype="float32") = R.reshape(x, R.shape([10, 10, 1]))
                lv1: R.Tensor((10, 10, 8), dtype="bool") = R.equal(lv, test_elements)
                lv2: R.Tensor((10, 10, 8), dtype="int8") = R.astype(lv1, dtype="int8")
                lv3: R.Tensor((10, 10), dtype="int8") = R.max(lv2, axis=[-1], keepdims=False)
                lv4: R.Tensor((10, 10), dtype="bool") = R.astype(lv3, dtype="bool")
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv4,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(8, dtype=torch.float32),
    )
    verify_model(IsInModel(), example_args, {}, expected)


def test_div_mode():
    class DivTrunc(Module):
        def forward(self, x, y):
            return torch.div(x, y, rounding_mode="trunc")

    info = [((2, 3), "float32"), ((3,), "float32")]
    expected = make_expected(
        info, lambda x, y: relax.op.trunc(relax.op.divide(x, y)), exported=True
    )
    verify_model(DivTrunc(), (torch.randn(2, 3), torch.randn(3)), {}, expected)


def test_batchnorm2d():
    class BatchNorm2dCustom(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3, eps=0.001, momentum=0.01)

        def forward(self, input):
            return self.bn(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
            w3: R.Tensor((3,), dtype="float32"),
            w4: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = R.nn.batch_norm(
                    input_1,
                    w1,
                    w2,
                    w3,
                    w4,
                    axis=1,
                    epsilon=0.001,
                    center=True,
                    scale=True,
                    momentum=0.01,
                    training=False,
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[0]
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model_2 = BatchNorm2dCustom().eval()
    binding_2 = {
        "w1": model_2.bn.weight.detach().numpy(),
        "w2": model_2.bn.bias.detach().numpy(),
        "w3": model_2.bn.running_mean.detach().numpy(),
        "w4": model_2.bn.running_var.detach().numpy(),
    }
    verify_model(model_2, example_args, binding_2, expected2)

    class BatchNorm2dTraining(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3, track_running_stats=True)

        def forward(self, input):
            return self.bn(input)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((2, 3, 4, 4), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
            w3: R.Tensor((3,), dtype="float32"),
            w4: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 4, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="int64") = R.add(R.const(0, "int64"), R.const(1, "int64"))
                lv1: R.Tuple(
                    R.Tensor((2, 3, 4, 4), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = R.nn.batch_norm(
                    input_1,
                    w1,
                    w2,
                    w3,
                    w4,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    momentum=0.1,
                    training=True,
                )
                lv2: R.Tensor((2, 3, 4, 4), dtype="float32") = lv1[0]
                lv3: R.Tensor((3,), dtype="float32") = lv1[1]
                lv4: R.Tensor((3,), dtype="float32") = R.zeros(R.shape([3]), dtype="float32")
                lv5: R.Tuple(
                    R.Tensor((2, 3, 4, 4), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = (lv2, lv3, lv4, lv4, lv4)
                lv6: R.Tensor((2, 3, 4, 4), dtype="float32") = lv5[0]
                lv7: R.Tensor((3,), dtype="float32") = lv5[3]
                lv8: R.Tensor((3,), dtype="float32") = lv5[4]
                gv: R.Tuple(R.Tensor((2, 3, 4, 4), dtype="float32")) = (lv6,)
                R.output(gv)
            return gv

    example_args_train = (torch.randn(2, 3, 4, 4, dtype=torch.float32),)

    model_3 = BatchNorm2dTraining()
    model_3.train()  # Set to training mode
    binding_3 = {
        "w1": model_3.bn.weight.detach().numpy(),
        "w2": model_3.bn.bias.detach().numpy(),
        "w3": model_3.bn.running_mean.detach().numpy(),
        "w4": model_3.bn.running_var.detach().numpy(),
    }
    verify_model(model_3, example_args_train, binding_3, expected3)


def test_adaptive_avgpool1d():
    class AdaptiveAvgPool1d(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool1d(input, output_size=5)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 10), dtype="float32") = R.expand_dims(input_1, axis=[-2])
                lv1: R.Tensor((1, 3, 1, 5), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    lv, output_size=[1, 5], layout="NCHW"
                )
                lv2: R.Tensor((1, 3, 5), dtype="float32") = R.squeeze(lv1, axis=[-2])
                gv: R.Tuple(R.Tensor((1, 3, 5), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool1d(), example_args, {}, expected1)


def test_adaptive_avgpool2d():
    class AdaptiveAvgPool2d(Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool2d(input, [10, 10])

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    input_1, output_size=[10, 10], layout="NCHW", out_layout="NCHW"
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool2d(), example_args, {}, expected1)


def test_adaptive_avgpool3d():
    class AdaptiveAvgPool3d(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool3d(input, [4, 4, 4])

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 4, 4, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.nn.adaptive_avg_pool3d(
                    input_1, output_size=[4, 4, 4], layout="NCDHW", out_layout="NCDHW"
                )
                gv: R.Tuple(R.Tensor((1, 3, 4, 4, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool3d(), example_args, {}, expected1)


def test_addmm():
    class Addmm1(Module):
        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3)

    class Addmm2(Module):
        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3, beta=0.8, alpha=0.5)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.add(x1, lv)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.multiply(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((10, 10), dtype="float32") = R.multiply(x1, R.const(0.8, "float32"))
                lv3: R.Tensor((10, 10), dtype="float32") = R.add(lv2, lv1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )

    verify_model(Addmm1(), example_args, {}, expected1)
    verify_model(Addmm2(), example_args, {}, expected2)


def test_sparse_addmm():
    class SparseAddmm1(Module):
        def forward(self, x1, x2, x3):
            return torch.sparse.addmm(x1, x2, x3)

    class SparseAddmm2(Module):
        def forward(self, x1, x2, x3):
            return torch.sparse.addmm(x1, x2, x3, beta=0.8, alpha=0.5)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.add(x1, lv)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.multiply(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((10, 10), dtype="float32") = R.multiply(x1, R.const(0.8, "float32"))
                lv3: R.Tensor((10, 10), dtype="float32") = R.add(lv2, lv1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )

    verify_model(SparseAddmm1(), example_args, {}, expected1)
    verify_model(SparseAddmm2(), example_args, {}, expected2)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("kind", ["max", "avg"])
def test_pool(rank, kind):
    shape = (1, 2) + (9,) * rank
    layout = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[rank]
    as_module = True
    kwargs = (
        dict(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        if as_module
        else dict(kernel_size=2)
    )
    if as_module:
        kwargs.update(dilation=2) if kind == "max" else kwargs.update(count_include_pad=False)
    op_name = f"{kind}_pool{rank}d"
    module_name = f"{kind.title()}Pool{rank}d"
    op = (
        getattr(torch.nn, module_name)(**kwargs)
        if as_module
        else lambda x: getattr(torch.nn.functional, op_name)(x, **kwargs)
    )
    attrs = dict(
        pool_size=[3 if as_module else 2] * rank,
        strides=[2] * rank,
        padding=[1 if as_module else 0] * (2 * rank),
        layout=layout,
        ceil_mode=as_module,
    )
    if kind == "max":
        attrs["dilation"] = [2 if as_module else 1] * rank
    else:
        attrs["count_include_pad"] = not as_module
    info = [(shape, "float32")]
    expected = make_expected(
        info, lambda x: getattr(relax.op.nn, op_name)(x, **attrs), exported=True
    )
    verify_model(UnaryModule(op), (torch.randn(shape),), {}, expected, run_ep_decomposition=False)


def test_baddbmm():
    class BAddBMM1(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3, 5), dtype="float32"),
            inp_1: R.Tensor((2, 3, 4), dtype="float32"),
            inp_2: R.Tensor((2, 4, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 5), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(
                    inp_1, inp_2, out_dtype="float32"
                )
                lv1: R.Tensor((2, 3, 5), dtype="float32") = R.add(inp_0, lv)
                gv: R.Tuple(R.Tensor((2, 3, 5), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    class BAddBMM2(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3, 5), dtype="float32"),
            inp_1: R.Tensor((2, 3, 4), dtype="float32"),
            inp_2: R.Tensor((2, 4, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 5), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(
                    inp_1, inp_2, out_dtype="float32"
                )
                lv1: R.Tensor((2, 3, 5), dtype="float32") = R.multiply(lv, R.const(2, "float32"))
                gv: R.Tuple(R.Tensor((2, 3, 5), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    class BAddBMM3(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=3)

    @tvm.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3, 5), dtype="float32"),
            inp_1: R.Tensor((2, 3, 4), dtype="float32"),
            inp_2: R.Tensor((2, 4, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 5), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(
                    inp_1, inp_2, out_dtype="float32"
                )
                lv1: R.Tensor((2, 3, 5), dtype="float32") = R.multiply(lv, R.const(2, "float32"))
                lv2: R.Tensor((2, 3, 5), dtype="float32") = R.multiply(inp_0, R.const(3, "float32"))
                lv3: R.Tensor((2, 3, 5), dtype="float32") = R.add(lv2, lv1)
                gv: R.Tuple(R.Tensor((2, 3, 5), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(2, 3, 5, dtype=torch.float32),
        torch.randn(2, 3, 4, dtype=torch.float32),
        torch.randn(2, 4, 5, dtype=torch.float32),
    )
    verify_model(
        BAddBMM1(),
        example_args,
        {},
        Expected1,
        run_ep_decomposition=True,
    )

    verify_model(
        BAddBMM2(),
        example_args,
        {},
        Expected2,
        run_ep_decomposition=True,
    )

    verify_model(
        BAddBMM3(),
        example_args,
        {},
        Expected3,
        run_ep_decomposition=True,
    )


def test_bmm():
    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input_1: R.Tensor((2, 3, 4), dtype="float32"),
            input_2: R.Tensor((2, 4, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 5), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 3, 5), dtype="float32") = R.matmul(
                    input_1, input_2, out_dtype="float32"
                )
                gv: R.Tuple(R.Tensor((2, 3, 5), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(2, 3, 4, dtype=torch.float32),
        torch.randn(2, 4, 5, dtype=torch.float32),
    )
    verify_model(
        BMM(),
        example_args,
        {},
        Expected,
        run_ep_decomposition=True,
    )


def test_conv_transpose1d():
    class ConvTranspose1d1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(6, 6, 3, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 6, 4), dtype="float32"),
            w1: R.Tensor((6, 6, 3), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 6), dtype="float32") = R.nn.conv1d_transpose(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    output_padding=[0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="IOW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1)) = R.reshape(w2, [1, 6, 1])
                lv3: R.Tensor((1, 6, 6), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 6), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class ConvTranspose1d2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(6, 6, 3, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 6, 4), dtype="float32"),
            w1: R.Tensor((6, 6, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 6), dtype="float32") = R.nn.conv1d_transpose(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    output_padding=[0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="IOW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 6), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 6, 4, dtype=torch.float32),)

    model = ConvTranspose1d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = ConvTranspose1d2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv_transpose2d():
    class ConvTranspose2d1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(3, 3, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3, 3, 7, 7), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 3, 16, 16), dtype="float32") = R.nn.conv2d_transpose(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 3, 1, 1)) = R.reshape(w2, [1, 3, 1, 1])
                lv3: R.Tensor((1, 3, 16, 16), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class ConvTranspose2d2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(3, 3, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3, 3, 7, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 3, 16, 16), dtype="float32") = R.nn.conv2d_transpose(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = ConvTranspose2d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = ConvTranspose2d2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv1d():
    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            w1: R.Tensor((6, 3, 7), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
            input_1: R.Tensor((1, 3, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4), dtype="float32") = R.nn.conv1d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="OIW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1), dtype="float32") = R.reshape(w2, [1, 6, 1])
                lv3: R.Tensor((1, 6, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            w1: R.Tensor((6, 3, 7), dtype="float32"),
            input_1: R.Tensor((1, 3, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4), dtype="float32") = R.nn.conv1d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="OIW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)

    model = Conv1D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv1D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv2d():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1, 1)) = R.reshape(w2, [1, 6, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = Conv2D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv2D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv3d():
    class Conv3D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7, 7), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.nn.conv3d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0, 0],
                    dilation=[1],
                    data_layout="NCDHW",
                    kernel_layout="OIDHW",
                    out_layout="NCDHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1, 1, 1)) = R.reshape(w2, [1, 6, 1, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class Conv3D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.nn.conv3d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0, 0],
                    dilation=[1],
                    data_layout="NCDHW",
                    kernel_layout="OIDHW",
                    out_layout="NCDHW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, 10, dtype=torch.float32),)

    model = Conv3D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv3D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


@pytest.mark.parametrize(
    "mode,pad_mode", [("constant", "constant"), ("reflect", "reflect"), ("replicate", "edge")]
)
def test_pad(mode, pad_mode):
    shape = (1, 2, 4, 5)
    model = UnaryModule(lambda x: torch.nn.functional.pad(x, (1, 1, 2, 2), mode=mode))

    def expected(x):
        if mode == "constant":
            return relax.op.nn.pad(x, [0, 0, 0, 0, 2, 2, 1, 1])
        for axis, size, padding in ((2, 4, 2), (3, 5, 1)):
            indices = relax.op.arange(-padding, size + padding, 1, dtype="int64")
            if mode == "reflect":
                last = relax.const(size - 1, "int64")
                indices = relax.op.subtract(
                    last, relax.op.abs(relax.op.subtract(last, relax.op.abs(indices)))
                )
            else:
                indices = relax.op.clip(indices, 0, size - 1)
            x = relax.op.take(x, indices, axis=axis, mode="fast")
        return x

    expected = make_expected([(shape, "float32")], expected, exported=True)
    verify_model(model, (torch.randn(shape),), {}, expected)


def test_decomposed_image_operations():
    class ImageOps(Module):
        def forward(self, x):
            return (
                torch.nn.functional.interpolate(x, (4, 6), mode="bicubic"),
                torch.nn.functional.pad(x, (1, 1, 1, 1), mode="circular"),
            )

    verify_model_numerically(
        ImageOps(), (torch.arange(6.0).reshape(1, 1, 2, 3),), rtol=1e-5, atol=1e-5
    )


def test_pixel_shuffle():
    class PixelShuffle(torch.nn.Module):
        def __init__(self, upscale_factor=2):
            super().__init__()
            self.upscale_factor = upscale_factor

        def forward(self, x):
            return torch.nn.functional.pixel_shuffle(x, self.upscale_factor)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 8, 10, 15), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 20, 30), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 10, 15), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 10, 15])
                )
                lv1: R.Tensor((1, 2, 10, 2, 15, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 1, 4, 2, 5, 3]
                )
                lv2: R.Tensor((1, 2, 20, 30), dtype="float32") = R.reshape(
                    lv1, R.shape([1, 2, 20, 30])
                )
                gv: R.Tuple(R.Tensor((1, 2, 20, 30), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 8, 10, 15, dtype=torch.float32),)
    verify_model(PixelShuffle(upscale_factor=2), example_args, {}, expected)


def test_einsum():
    class Einsum1(Module):
        def forward(self, x):
            return torch.einsum("ii", x)

    class Einsum2(Module):
        def forward(self, x, y):
            return torch.einsum("i,j->ij", x, y)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((4, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.einsum((inp_0,), subscripts="ii")
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"), inp_1: R.Tensor((4,), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 4), dtype="float32") = R.einsum(
                    (inp_0, inp_1), subscripts="i,j->ij"
                )
                gv: R.Tuple(R.Tensor((5, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(4, 4, dtype=torch.float32),)
    verify_model(Einsum1(), example_args, {}, Expected1, run_ep_decomposition=False)

    example_args = (torch.randn(5, dtype=torch.float32), torch.randn(4, dtype=torch.float32))
    verify_model(Einsum2(), example_args, {}, Expected2, run_ep_decomposition=False)


def test_einsum_repeated_subscript():
    """Decomposed diagonal extraction must lower to a single einsum."""

    class EinsumDiag(Module):
        def forward(self, x):
            return torch.einsum("ii->i", x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((3,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.einsum((x,), subscripts="zz->z")
                lv1: R.Tensor((3,), dtype="float32") = R.permute_dims(lv, axes=[0])
                lv2: R.Tensor((3,), dtype="float32") = R.permute_dims(lv1, axes=[0])
                gv: R.Tuple(R.Tensor((3,), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, 3, dtype=torch.float32),)
    verify_model(EinsumDiag(), example_args, {}, Expected)

    class BatchedDiagonal(Module):
        def forward(self, x):
            return torch.einsum("...ii->...i", x), torch.einsum("...ii->...", x)

    verify_model_numerically(BatchedDiagonal(), (torch.arange(18.0).reshape(2, 3, 3),))


def test_diagonal_offsets():
    class Diagonal(Module):
        def forward(self, x):
            # Non-square input, both offset signs, and one empty result per sign.
            return tuple(torch.diagonal(x, offset, 0, 1) for offset in (1, -1, 4, -3))

    verify_model_numerically(Diagonal(), (torch.arange(12.0).reshape(3, 4),))


def test_outer():
    class Outer(torch.nn.Module):
        def forward(self, x, y):
            return torch.outer(x, y)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32"), y: R.Tensor((4,), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 1), dtype="float32") = R.reshape(x, R.shape([3, 1]))
                lv1: R.Tensor((3, 4), dtype="float32") = R.multiply(lv, y)
                gv: R.Tuple(R.Tensor((3, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(3, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
    )
    verify_model(Outer(), example_args, {}, expected)


def test_embedding():
    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, input):
            return self.embedding(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((4,), dtype="int64"), w1: R.Tensor((10, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((4, 3), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int32") = R.astype(input_1, dtype="int32")
                lv1: R.Tensor((4, 3), dtype="float32") = R.take(w1, lv, axis=0)
                gv: R.Tuple(R.Tensor((4, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randint(low=0, high=10, size=(4,), dtype=torch.int64),)

    model = Embedding()
    binding = {"w1": model.embedding.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected1)


def test_groupnorm():
    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.GroupNorm(3, 3)

        def forward(self, input):
            return self.gn(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.group_norm(
                    input_1,
                    w1,
                    w2,
                    num_groups=3,
                    channel_axis=1,
                    axes=[2, 3],
                    epsilon=1.0000000000000001e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = GroupNorm()
    binding = {
        "w1": model.gn.weight.detach().numpy(),
        "w2": model.gn.bias.detach().numpy(),
    }
    verify_model(model, example_args, binding, expected1)


def test_instancenorm2d():
    class InstanceNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.InstanceNorm2d(3)

        def forward(self, input):
            return self.gn(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.instance_norm(
                    input_1,
                    w1,
                    w2,
                    channel_axis=1,
                    axes=[0, 2, 3],
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = InstanceNorm2d()
    binding = {
        "w1": torch.ones(3).detach().numpy(),
        "w2": torch.zeros(3).detach().numpy(),
    }
    verify_model(model, example_args, binding, expected1)


def test_layernorm():
    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm((10, 10))

        def forward(self, input):
            return self.ln(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((10, 10), dtype="float32"),
            w2: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.layer_norm(
                    input_1,
                    w1,
                    w2,
                    axes=[-2, -1],
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = LayerNorm()
    binding = {
        "w1": model.ln.weight.detach().numpy(),
        "w2": model.ln.bias.detach().numpy(),
    }
    verify_model(LayerNorm(), example_args, binding, expected1)


def test_linear():
    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, input):
            return self.linear(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            w1: R.Tensor((7, 10), dtype="float32"),
            w2: R.Tensor((7,), dtype="float32"),
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((30, 10), dtype="float32") = R.reshape(input_1, R.shape([30, 10]))
                lv1: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                lv2: R.Tensor((30, 7), dtype="float32") = R.matmul(lv, lv1, out_dtype="float32")
                lv3: R.Tensor((30, 7), dtype="float32") = R.add(w2, lv2)
                lv4: R.Tensor((1, 3, 10, 7), dtype="float32") = R.reshape(
                    lv3, R.shape([1, 3, 10, 7])
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, input):
            return self.linear(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            w1: R.Tensor((7, 10), dtype="float32"),
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                lv1: R.Tensor((30, 10), dtype="float32") = R.reshape(input_1, R.shape([30, 10]))
                lv2: R.Tensor((30, 7), dtype="float32") = R.matmul(lv1, lv, out_dtype="float32")
                lv3: R.Tensor((1, 3, 10, 7), dtype="float32") = R.reshape(
                    lv2, R.shape([1, 3, 10, 7])
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = Dense1()
    binding = {"w1": model.linear.weight.detach().numpy(), "w2": model.linear.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Dense2()
    binding = {"w1": model.linear.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_scaled_dot_product_attention():
    class Attention1(Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            q: R.Tensor((2, 2, 4, 8), dtype="float32"),
            k: R.Tensor((2, 2, 4, 8), dtype="float32"),
            v: R.Tensor((2, 2, 4, 8), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2, 4, 8), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 4, 2, 8), dtype="float32") = R.permute_dims(q, axes=[0, 2, 1, 3])
                lv1: R.Tensor((2, 4, 2, 8), dtype="float32") = R.permute_dims(k, axes=[0, 2, 1, 3])
                lv2: R.Tensor((2, 4, 2, 8), dtype="float32") = R.permute_dims(v, axes=[0, 2, 1, 3])
                lv3: R.Tensor((2, 4, 2, 8), dtype="float32") = R.nn.attention(
                    lv, lv1, lv2, scale=None, causal_mask=None, window_size=None
                )
                lv4: R.Tensor((2, 2, 4, 8), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 2, 1, 3]
                )
                gv: R.Tuple(R.Tensor((2, 2, 4, 8), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    class Attention2(Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            q: R.Tensor((2, 2, 4, 8), dtype="float32"),
            k: R.Tensor((2, 2, 4, 8), dtype="float32"),
            v: R.Tensor((2, 2, 4, 8), dtype="float32"),
            mask: R.Tensor((2, 2, 4, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2, 4, 8), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 4, 2, 8), dtype="float32") = R.permute_dims(q, axes=[0, 2, 1, 3])
                lv1: R.Tensor((2, 4, 2, 8), dtype="float32") = R.permute_dims(k, axes=[0, 2, 1, 3])
                lv2: R.Tensor((2, 4, 2, 8), dtype="float32") = R.permute_dims(v, axes=[0, 2, 1, 3])
                lv3: R.Tensor((2, 4, 2, 8), dtype="float32") = R.nn.attention_bias(
                    lv, lv1, lv2, mask, scale=None, causal_mask=None, window_size=None
                )
                lv4: R.Tensor((2, 2, 4, 8), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 2, 1, 3]
                )
                gv: R.Tuple(R.Tensor((2, 2, 4, 8), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    verify_model(
        Attention1(),
        (
            torch.randn(2, 2, 4, 8, dtype=torch.float32),
            torch.randn(2, 2, 4, 8, dtype=torch.float32),
            torch.randn(2, 2, 4, 8, dtype=torch.float32),
        ),
        {},
        Expected1,
        run_ep_decomposition=False,
    )

    verify_model(
        Attention2(),
        (
            torch.randn(2, 2, 4, 8, dtype=torch.float32),
            torch.randn(2, 2, 4, 8, dtype=torch.float32),
            torch.randn(2, 2, 4, 8, dtype=torch.float32),
            torch.randn(2, 2, 4, 4, dtype=torch.float32),
        ),
        {},
        Expected2,
        run_ep_decomposition=False,
    )

    # Test 2D input (seq_len, head_dim) - bug fix for #18441
    class Attention2D(Module):
        def forward(self, x):
            return torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=False)

    @I.ir_module
    class Expected2D:
        @R.function
        def main(
            x: R.Tensor((8, 32), dtype="float32"),
        ) -> R.Tuple(R.Tensor((8, 32), dtype="float32")):
            with R.dataflow():
                # Expand to add batch dimension for query, key, value separately
                # (8, 32) -> (1, 8, 32)
                lv: R.Tensor((1, 8, 32), dtype="float32") = R.expand_dims(x, axis=[0])
                lv1: R.Tensor((1, 8, 32), dtype="float32") = R.expand_dims(x, axis=[0])
                lv2: R.Tensor((1, 8, 32), dtype="float32") = R.expand_dims(x, axis=[0])
                # Expand to add num_heads dimension: (1, 8, 32) -> (1, 1, 8, 32)
                lv3: R.Tensor((1, 1, 8, 32), dtype="float32") = R.expand_dims(lv, axis=[1])
                lv4: R.Tensor((1, 1, 8, 32), dtype="float32") = R.expand_dims(lv1, axis=[1])
                lv5: R.Tensor((1, 1, 8, 32), dtype="float32") = R.expand_dims(lv2, axis=[1])
                # Attention operation: (1, 1, 8, 32) -> (1, 1, 8, 32)
                lv6: R.Tensor((1, 1, 8, 32), dtype="float32") = R.nn.attention(
                    lv3, lv4, lv5, scale=None, causal_mask=None, window_size=None
                )
                # Squeeze batch and num_heads dimensions: (1, 1, 8, 32) -> (8, 32)
                lv7: R.Tensor((8, 32), dtype="float32") = R.squeeze(lv6, axis=[0, 1])
                gv: R.Tuple(R.Tensor((8, 32), dtype="float32")) = (lv7,)
                R.output(gv)
            return gv

    verify_model(
        Attention2D(),
        (torch.randn(8, 32, dtype=torch.float32),),
        {},
        Expected2D,
        run_ep_decomposition=False,
    )


def test_unbind():
    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(data: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 4, 5), dtype="float32"),
            R.Tensor((2, 4, 5), dtype="float32"),
            R.Tensor((2, 4, 5), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 1, 4, 5), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv1: R.Tensor((2, 1, 4, 5), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv2: R.Tensor((2, 1, 4, 5), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv3: R.Tensor((2, 4, 5), dtype="float32") = R.squeeze(lv, axis=[1])
                lv4: R.Tensor((2, 4, 5), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv5: R.Tensor((2, 4, 5), dtype="float32") = R.squeeze(lv2, axis=[1])
                gv: R.Tuple(
                    R.Tensor((2, 4, 5), dtype="float32"),
                    R.Tensor((2, 4, 5), dtype="float32"),
                    R.Tensor((2, 4, 5), dtype="float32"),
                ) = (lv3, lv4, lv5)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(data: R.Tensor((3, 1, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 1, 3), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv1: R.Tensor((3, 3), dtype="float32") = R.squeeze(lv, axis=[1])
                gv: R.Tuple(R.Tensor((3, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, 4, 5, dtype=torch.float32),)
    verify_model(Unbind2(), example_args, {}, expected2)
    single_dim_args = (torch.randn(3, 1, 3, dtype=torch.float32),)
    verify_model(Unbind2(), single_dim_args, {}, expected3)


@pytest.mark.parametrize(
    "mode,method", [("bilinear", "linear"), ("nearest", "nearest_neighbor"), ("bicubic", "cubic")]
)
def test_interpolate(mode, method):
    shape = (1, 2, 4, 5)
    model = UnaryModule(lambda x: torch.nn.functional.interpolate(x, (8, 10), mode=mode))
    expected = make_expected(
        [(shape, "float32")],
        lambda x: relax.op.image.resize2d(
            x,
            (8, 10),
            layout="NCHW",
            method=method,
            coordinate_transformation_mode="half_pixel",
            rounding_method="round",
            cubic_alpha=-0.75,
        ),
        exported=True,
    )
    verify_model(model, (torch.randn(shape),), {}, expected, run_ep_decomposition=False)


def test_interpolate_antialiased():
    args = (torch.arange(12.0).reshape(1, 1, 3, 4),)
    model = UnaryModule(
        lambda x: torch.nn.functional.interpolate(x, (6, 8), mode="bilinear", antialias=True)
    )
    verify_model_numerically(model, args, rtol=1e-5, atol=1e-5)
    downsample = UnaryModule(
        lambda x: torch.nn.functional.interpolate(x, (2, 2), mode="bilinear", antialias=True)
    )
    with pytest.raises(NotImplementedError, match="Antialiased"):
        from_exported_program(export(downsample, args))


@pytest.mark.parametrize("dim", [None, 1])
def test_mean(dim):
    shape = (2, 3)
    model = UnaryModule(lambda x: torch.mean(x, dim=dim, keepdim=dim is not None))
    info = [(shape, "float32")]
    expected = make_expected(
        info, lambda x: relax.op.mean(x, axis=dim, keepdims=dim is not None), exported=True
    )
    verify_model(model, (torch.randn(shape),), {}, expected)


def test_median():
    class MedianKeepDim(Module):
        def forward(self, input):
            return input.median(-1, keepdim=True)

    class MedianWithoutDim(Module):
        def forward(self, input):
            return input.median()

    @I.ir_module
    class Expected2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((256, 1), dtype="float32"), R.Tensor((256, 1), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((256, 1), dtype="float32"), R.Tensor((256, 1), dtype="int64")
                ) = R.median(inp_0, axis=[-1], keepdims=True)
                lv1: R.Tensor((256, 1), dtype="float32") = lv[0]
                lv2: R.Tensor((256, 1), dtype="int64") = lv[1]
                gv: R.Tuple(
                    R.Tensor((256, 1), dtype="float32"), R.Tensor((256, 1), dtype="int64")
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.median(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    verify_model(MedianKeepDim(), example_args, {}, Expected2)
    verify_model(MedianWithoutDim(), example_args, {}, Expected3)


def test_sum():
    class SumKeepDim(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1), keepdim=True)

    class SumWithoutDim(Module):
        def forward(self, x):
            return torch.sum(x)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 1, 1, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1, 1, 4), dtype="float32") = R.sum(
                    inp_0, axis=[2, 1], keepdims=True
                )
                gv: R.Tuple(R.Tensor((1, 1, 1, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.sum(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(SumKeepDim(), example_args, {}, expected2)
    verify_model(SumWithoutDim(), example_args, {}, expected3)


@pytest.mark.parametrize("op,dim,keepdim", [(torch.argmax, 1, True), (torch.argmin, None, False)])
def test_arg_reduce(op, dim, keepdim):
    shape = (2, 3)
    model = UnaryModule(lambda x: op(x, dim=dim, keepdim=keepdim))
    info = [(shape, "float32")]
    relax_op = relax.op.argmax if op is torch.argmax else relax.op.argmin
    expected = make_expected(info, lambda x: relax_op(x, axis=dim, keepdims=keepdim), exported=True)
    verify_model(model, (torch.randn(shape),), {}, expected)


def test_cat():
    class Cat1(Module):
        def forward(self, x, y):
            return torch.cat((x, y), dim=1)

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 6), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 6), dtype="float32") = R.concat((inp_0, inp_1), axis=1)
                gv: R.Tuple(R.Tensor((2, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32), torch.randn(2, 3, dtype=torch.float32))
    verify_model(Cat1(), example_args, {}, Expected2)


def test_cumsum():
    class Cumsum(Module):
        def forward(self, input):
            return torch.cumsum(input, dim=1, dtype=torch.int32)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 3, 4), dtype="int32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="int32") = R.cumsum(input_1, axis=1, dtype="int32")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="int32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Cumsum(), example_args, {}, expected1)


def test_expand():
    class Expand1(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    class Expand2(Module):
        def forward(self, x):
            return x.expand(4, -1, -1, 4)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 2, 3, 4), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 2, 3, 4), dtype="float32") = R.broadcast_to(x, (4, 2, 3, 4))
                gv: R.Tuple(R.Tensor((4, 2, 3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Expand1(), example_args, {}, expected1)
    verify_model(Expand2(), example_args, {}, expected1)


def test_flatten():
    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, input):
            return self.f(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 100), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 100), dtype="float32") = R.reshape(input_1, (1, 3, 100))
                gv: R.Tuple(R.Tensor((1, 3, 100), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Flatten(), example_args, {}, expected1)


def test_zero_sized_shapes():
    class EmptyShapes(Module):
        def forward(self, x, y):
            return (
                x.flatten(),
                x.flatten(1, 2),
                x.reshape(0, x.shape[0]),
                x.reshape(x.shape[0], 0, 4),
                x.reshape(0, 0),
                y.reshape(0, 0, 4),
                y.unflatten(0, (0, 2)),
            )

    batch = torch.export.Dim("batch", min=1, max=8)
    inputs = [(torch.empty(n, 2, 0, 4), torch.empty(0, 3)) for n in (3, 5)]
    verify_model_numerically(
        EmptyShapes(), inputs[0], input_sets=inputs, dynamic_shapes={"x": {0: batch}, "y": {}}
    )


def test_meshgrid():
    class Meshgrid1(Module):
        def forward(self, input1, input2):
            return torch.meshgrid((input1, input2), indexing="ij")

    class Meshgrid2(Module):
        def forward(self, input1, input2):
            return torch.meshgrid((input1, input2), indexing="xy")

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input1: R.Tensor((3,), dtype="float32"), input2: R.Tensor((3,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 1), dtype="float32") = R.reshape(input1, R.shape([3, 1]))
                lv1: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv, R.shape([3, 3]))
                lv2: R.Tensor((1, 3), dtype="float32") = R.reshape(input2, R.shape([1, 3]))
                lv3: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv2, R.shape([3, 3]))
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv1, lv3)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input1: R.Tensor((3,), dtype="float32"), input2: R.Tensor((3,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 1), dtype="float32") = R.reshape(input2, R.shape([3, 1]))
                lv1: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv, R.shape([3, 3]))
                lv2: R.Tensor((1, 3), dtype="float32") = R.reshape(input1, R.shape([1, 3]))
                lv3: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv2, R.shape([3, 3]))
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv3, lv1)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(3, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
    )
    verify_model(Meshgrid1(), example_args, {}, expected1)
    verify_model(Meshgrid2(), example_args, {}, expected2)


def test_permute():
    class Permute1(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 4, 3, 2), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Permute1(), example_args, {}, expected1)


def test_repeat():
    model = UnaryModule(lambda x: x.repeat(4, 2))
    expected = make_expected([((3,), "float32")], lambda x: relax.op.tile(x, [4, 2]), exported=True)
    verify_model(model, (torch.randn(3),), {}, expected)


def test_reshape():
    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 12), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tuple(R.Tensor((2, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Reshape(), example_args, {}, expected1)


@pytest.mark.parametrize("multi_axis", [False, True])
def test_roll(multi_axis):
    shape = (4, 3)
    model = UnaryModule(
        lambda x: torch.roll(x, (-1, 1), (0, 1)) if multi_axis else torch.roll(x, 1)
    )

    def expected(x):
        def take_roll(x, size, offset, axis):
            indices = relax.op.arange(0, size, 1, dtype="int64")
            indices = relax.op.mod(
                relax.op.add(indices, relax.const(offset, "int64")), relax.const(size, "int64")
            )
            return relax.op.take(x, indices, axis=axis, mode="fast")

        if multi_axis:
            return take_roll(take_roll(x, 4, 1, 0), 3, 2, 1)
        return relax.op.reshape(take_roll(relax.op.reshape(x, (12,)), 12, 11, 0), shape)

    info = [(shape, "float32")]
    expected = make_expected(info, expected, exported=True)
    verify_model(model, (torch.randn(shape),), {}, expected)


def test_select_slice():
    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 10, 3), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((3, 10, 10), dtype="float32") = R.take(
                    x, R.const(0, "int64"), axis=0, mode="fast"
                )
                lv1: R.Tensor((1, 10, 10), dtype="float32") = R.strided_slice(
                    lv,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(9223372036854775807),),
                    (R.prim_value(2),),
                    assume_inbound=False,
                )
                lv2: R.Tensor((1, 10, 3), dtype="float32") = R.strided_slice(
                    lv1,
                    (R.prim_value(2),),
                    (R.prim_value(0),),
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                gv: R.Tuple(R.Tensor((1, 10, 3), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    @I.ir_module
    class expected2:
        @R.function
        def main(x: R.Tensor((8, 16), dtype="float32")) -> R.Tuple(
            R.Tensor((8, 1, 1, 16, 1), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((8, 1, 16), dtype="float32") = R.expand_dims(x, axis=[1])
                lv1: R.Tensor((8, 1, 1, 16), dtype="float32") = R.expand_dims(lv, axis=[2])
                lv2: R.Tensor((8, 1, 1, 16, 1), dtype="float32") = R.expand_dims(lv1, axis=[4])
                gv: R.Tuple(R.Tensor((8, 1, 1, 16, 1), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Slice1(), example_args, {}, expected1)

    example_args = (torch.randn(8, 16, dtype=torch.float32),)
    verify_model(Slice2(), example_args, {}, expected2)


@pytest.mark.parametrize("start,end,step", [(1, 7, 2), (0, -2, 1)])
def test_slice_scatter(start, end, step):
    class Scatter(Module):
        def forward(self, x, src):
            return torch.slice_scatter(x, src, dim=1, start=start, end=end, step=step)

    stop = end if end > 0 else 8 + end
    info = [((2, 8), "float32"), ((2, len(range(start, stop, step))), "float32")]
    expected = make_expected(
        info,
        lambda x, src: relax.op.slice_scatter(x, src, start, stop, step, axis=1),
        exported=True,
    )
    verify_model(Scatter(), tuple(torch.randn(shape) for shape, _ in info), {}, expected)


def test_slice_with_symbolic_end():
    """_slice correctly handles symbolic end values from dynamic shapes."""

    class SliceIdentityModel(torch.nn.Module):
        def forward(self, x):
            # x[:, :x.size(1)] is an identity slice that torch.export emits
            # as slice(x, 1, 0, sym_size_int(x, 1), 1) with dynamic shapes.
            seq_len = x.size(1)
            return x[:, :seq_len] + 0.0  # +0.0 to ensure output is a new tensor

    # The identity slice is elided; only x + 0.0 remains.
    @I.ir_module
    class ExpectedIdentity:
        @R.function
        def main(x: R.Tensor(("s0", "s1", 4), dtype="float32")) -> R.Tuple(
            R.Tensor(("s0", "s1", 4), dtype="float32")
        ):
            s0 = T.int64()
            s1 = T.int64()
            R.func_attr({"tir_var_lower_bound": {"s27": 2, "s77": 2}})
            with R.dataflow():
                lv: R.Tensor((s0, s1, 4), dtype="float32") = R.add(x, R.const(0.0, "float32"))
                gv: R.Tuple(R.Tensor((s0, s1, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 8, 4, dtype=torch.float32),)
    batch = torch.export.Dim("batch", min=2)
    seq = torch.export.Dim("seq", min=2)
    dynamic_shapes = {"x": {0: batch, 1: seq}}

    verify_model(
        SliceIdentityModel(),
        example_args,
        {},
        ExpectedIdentity,
        dynamic_shapes=dynamic_shapes,
        map_free_vars=True,
    )


def test_derived_input_dimension_without_exported_program_decomposition():
    class IdentityPair(torch.nn.Module):
        def forward(self, x, y):
            return x, y

    frames = torch.export.Dim("frames", min=1, max=8)
    exported_program = export(
        IdentityPair(),
        args=(torch.randn(1, 4, 3), torch.randn(1, 8, 3)),
        dynamic_shapes=({1: frames}, {1: 2 * frames}),
    )
    mod = from_exported_program(
        exported_program,
        keep_params_as_input=True,
        run_ep_decomposition=False,
    )

    x_shape = mod["main"].params[0].ty.shape.values
    y_shape = mod["main"].params[1].ty.shape.values
    assert tvm.arith.Analyzer().can_prove_equal(y_shape[1], x_shape[1] * 2)


def test_expand_with_new_leading_dimension():
    class ExpandLeading(torch.nn.Module):
        def forward(self, x):
            return x.expand(2, -1, -1)

    tokens = torch.export.Dim("tokens", min=1, max=8)
    exported_program = export(
        ExpandLeading(),
        args=(torch.randn(4, 3),),
        dynamic_shapes={"x": {0: tokens}},
    )
    mod = from_exported_program(exported_program)

    input_shape = mod["main"].params[0].ty.shape.values
    output_shape = mod["main"].ret_ty.fields[0].shape.values
    assert tvm.arith.Analyzer().can_prove_equal(output_shape[0], 2)
    assert tvm.arith.Analyzer().can_prove_equal(output_shape[1], input_shape[0])
    assert tvm.arith.Analyzer().can_prove_equal(output_shape[2], input_shape[1])


def test_dynamic_scalar_item_in_shape_operations():
    class DynamicShapeOps(torch.nn.Module):
        def forward(self, x):
            lengths = torch.full(
                (x.shape[0],),
                x.shape[1],
                device=x.device,
                dtype=torch.int64,
            )
            max_len = lengths.max().item()
            positions = torch.arange(max_len, device=x.device)
            mask = positions.unsqueeze(0).expand(x.shape[0], -1) == lengths.unsqueeze(1)
            filled = torch.full(
                (x.shape[0], x.shape[1]),
                x.shape[1],
                device=x.device,
                dtype=torch.int64,
            )
            shifted = torch.arange(x.shape[1] + 1, device=x.device)
            shortened = torch.full_like(lengths, x.shape[1] - 1)
            return mask, filled, shifted, shortened

    example_args = (torch.randn(1, 4, 3, dtype=torch.float32),)
    tokens = torch.export.Dim("tokens", min=1, max=8)
    mod = verify_model_numerically(
        DynamicShapeOps(),
        example_args,
        dynamic_shapes={"x": {1: tokens}},
        input_sets=[(torch.randn(1, token_count, 3),) for token_count in (4, 6)],
        rtol=0,
        atol=0,
    )
    script = mod.script()
    assert "R.tensor_to_shape" in script
    assert "R.shape_to_tensor" in script


def test_dynamic_scalar_item_integer_dtypes():
    class Items(Module):
        def forward(self, x):
            outputs = []
            for dtype, shape in ((torch.int32, ()), (torch.uint8, (1,)), (torch.int64, (1, 1))):
                value = (
                    torch.full((x.shape[0],), x.shape[1], dtype=dtype).max().reshape(shape).item()
                )
                outputs.extend((torch.arange(value), torch.full_like(x, value, dtype=torch.int64)))
            return tuple(outputs)

    shapes = {
        "x": {0: torch.export.Dim("rows", min=1, max=8), 1: torch.export.Dim("cols", min=1, max=8)}
    }
    inputs = [(torch.randn(shape),) for shape in ((3, 4), (5, 2))]
    verify_model_numerically(
        Items(),
        inputs[0],
        input_sets=inputs,
        dynamic_shapes=shapes,
        rtol=0,
        atol=0,
        run_ep_decomposition=False,
    )


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_runtime_scalar_item_integer_value_ranges():
    class RuntimeItems(torch.nn.Module):
        def forward(self, x, signed, unsigned, wide):
            return (
                torch.full_like(x, signed.item(), dtype=torch.int64),
                torch.full_like(x, unsigned.item(), dtype=torch.int64),
                torch.full_like(x, wide.item(), dtype=torch.int64),
            )

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.tensor([[-3]], dtype=torch.int8),
        torch.tensor([200], dtype=torch.uint8),
        torch.tensor(1 << 40, dtype=torch.int64),
    )
    verify_model_numerically(
        RuntimeItems(), example_args, rtol=0, atol=0, run_ep_decomposition=False
    )


def test_dynamic_scalar_operations():
    class ScalarOps(Module):
        def forward(self, x):
            rows, cols = x.shape
            equal = rows == cols
            mask = x > 0
            value = torch.full((rows,), cols, dtype=torch.int64).max().item()
            return (
                torch.full((rows,), cols),
                torch.full_like(x, rows, dtype=torch.float64),
                torch.fill(x, cols),
                x.clone().fill_(rows),
                torch.full((rows, cols), equal, dtype=torch.bool),
                torch.full_like(x, equal, dtype=torch.bool),
                torch.fill(mask, equal),
                mask.clone().fill_(equal),
                x.masked_fill(mask, rows),
                x.clone().masked_fill_(mask, cols),
                torch.full_like(x, -value, dtype=torch.int64),
                torch.arange(value // 2),
                torch.arange(value % 3),
            )

    shapes = {
        "x": {0: torch.export.Dim("rows", min=1, max=8), 1: torch.export.Dim("cols", min=1, max=8)}
    }
    inputs = [(torch.randn(shape),) for shape in ((3, 4), (3, 3), (5, 2))]
    verify_model_numerically(
        ScalarOps(),
        inputs[0],
        input_sets=inputs,
        dynamic_shapes=shapes,
        rtol=0,
        atol=0,
        run_ep_decomposition=False,
    )


@pytest.mark.parametrize("value,dtype", [(1 << 40, torch.int64), (1.0 + 2**-40, torch.float64)])
def test_fill_precision(value, dtype):
    class Fills(Module):
        def forward(self, x, y, mask):
            return (
                torch.full(x.shape, value, dtype=dtype),
                torch.full_like(x, value, dtype=dtype),
                torch.fill(y, value),
                y.clone().fill_(value),
                y.masked_fill(mask, value),
                y.clone().masked_fill_(mask, value),
            )

    args = (
        torch.ones(2, 3),
        torch.arange(6, dtype=dtype).reshape(2, 3),
        torch.tensor([[True, False, True], [False, True, False]]),
    )
    verify_model_numerically(Fills(), args, rtol=0, atol=0, run_ep_decomposition=False)


def test_split():
    class Chunk(Module):
        def forward(self, input):
            return torch.chunk(input, 3, dim=1)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 1, 10, 10), dtype="float32"),
            R.Tensor((1, 1, 10, 10), dtype="float32"),
            R.Tensor((1, 1, 10, 10), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                ) = R.split(input, indices_or_sections=[1, 2], axis=1)
                lv1: R.Tensor((1, 1, 10, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 1, 10, 10), dtype="float32") = lv[1]
                lv3: R.Tensor((1, 1, 10, 10), dtype="float32") = lv[2]
                gv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                ) = (lv1, lv2, lv3)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Chunk(), example_args, {}, Expected)


def test_split_int_split_size():
    """A non-divisible split size denotes chunk length, not section count."""

    class Split6(Module):
        def forward(self, input):
            return input.split(6, dim=0)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10,), dtype="float32")) -> R.Tuple(
            R.Tensor((6,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((6,), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                ) = R.split(input, indices_or_sections=[6], axis=0)
                lv1: R.Tensor((6,), dtype="float32") = lv[0]
                lv2: R.Tensor((4,), dtype="float32") = lv[1]
                gv: R.Tuple(
                    R.Tensor((6,), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    example_args = (torch.arange(10, dtype=torch.float32) + 1,)
    verify_model(Split6(), example_args, {}, Expected)


def test_squeeze():
    class Squeeze1(Module):
        def forward(self, input):
            return input.squeeze(1)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 4, 1), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 4, 1), dtype="float32") = R.squeeze(inp_0, axis=[1])
                gv: R.Tuple(R.Tensor((3, 4, 1), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Squeeze2(Module):
        def forward(self, input):
            return input.squeeze()

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(input: R.Tensor((3, 1, 4, 1), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.squeeze(input, axis=[0, 1, 2, 3])
                gv: R.Tuple(R.Tensor((3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, 1, 4, 1, dtype=torch.float32),)

    verify_model(Squeeze1(), example_args, {}, Expected1)
    verify_model(Squeeze2(), example_args, {}, Expected2)


def test_stack():
    class Stack1(Module):
        def forward(self, x, y):
            return torch.stack((x, y), dim=1)

    class Stack3(Module):
        def forward(self, x, y):
            return torch.stack((x, y), dim=-1)  # negative dim

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 6), dtype="float32") = R.concat((x, y), axis=1)
                lv1: R.Tensor((2, 2, 3), dtype="float32") = R.reshape(lv, R.shape([2, 2, 3]))
                gv: R.Tuple(R.Tensor((2, 2, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 1), dtype="float32") = R.expand_dims(x, axis=[2])
                lv1: R.Tensor((2, 3, 1), dtype="float32") = R.expand_dims(y, axis=[2])
                lv2: R.Tensor((2, 3, 2), dtype="float32") = R.concat((lv, lv1), axis=-1)
                gv: R.Tuple(R.Tensor((2, 3, 2), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32), torch.randn(2, 3, dtype=torch.float32))

    verify_model(Stack1(), example_args, {}, Expected1)
    verify_model(Stack3(), example_args, {}, Expected3)


def test_transpose():
    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 4, 3, 2), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Transpose(), example_args, {}, expected1)


def test_unsqueeze():
    shape = (2, 3)
    model = UnaryModule(lambda x: x.unsqueeze(-1))
    info = [(shape, "float32")]
    expected = make_expected(info, lambda x: relax.op.expand_dims(x, axis=-1), exported=True)
    verify_model(model, (torch.randn(shape),), {}, expected)


def test_as_strided():
    class AsStrided(Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, (3, 2, 2), (4, 2, 1))

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 2, 2), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 2, 2), dtype="float32") = R.reshape(x, (3, 2, 2))
                gv: R.Tuple(R.Tensor((3, 2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class AsStridedNonContiguous(Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, (2, 2, 2), (6, 3, 1))

    class AsStridedWithStorageOffset(Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, (2, 2), (2, 1), 1)

    example_args = (torch.randn(2, 2, 3, dtype=torch.float32),)
    verify_model(AsStrided(), example_args, {}, Expected)

    exported = export(AsStridedNonContiguous(), args=example_args)
    with pytest.raises(AssertionError, match="non-contiguous stride"):
        from_exported_program(exported)

    example_args = (torch.randn(2, 3, dtype=torch.float32),)
    exported = export(AsStridedWithStorageOffset(), args=example_args)
    with pytest.raises(AssertionError, match="storage_offset"):
        from_exported_program(exported)


def test_arange():
    class Arange(Module):
        def forward(self, input):
            return torch.arange(0, 20, dtype=torch.int32)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((20,), dtype="int32")
        ):
            with R.dataflow():
                lv: R.Tensor((20,), dtype="int32") = R.arange(0, 20, 1, dtype="int32")
                gv: R.Tuple(R.Tensor((20,), dtype="int32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Arange(), example_args, {}, Expected)


def test_hamming_window():
    class HammingWindow(Module):
        def forward(self, input):
            return torch.hamming_window(20, True, dtype=torch.float32)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((20,), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((20,), dtype="float32") = R.hamming_window(
                    R.prim_value(20),
                    R.prim_value(True),
                    R.prim_value(T.float64(0.54000000000000004)),
                    R.prim_value(T.float64(0.46000000000000002)),
                    dtype="float32",
                )
                gv: R.Tuple(R.Tensor((20,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(HammingWindow(), example_args, {}, Expected)


def test_clone():
    class Clone(Module):
        def forward(self, input):
            return torch.clone(input)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            with R.dataflow():
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (input,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Clone(), example_args, {}, Expected)


def test_empty_without_dtype():
    class EmptyWithoutDtype(Module):
        def forward(self, input):
            return torch.empty((5, 5))

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 5), dtype="float32") = R.zeros(R.shape([5, 5]), dtype="float32")
                gv: R.Tuple(R.Tensor((5, 5), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(EmptyWithoutDtype(), example_args, {}, Expected)


def test_fill():
    class Fill(Module):
        def forward(self, input: torch.Tensor):
            return torch.fill(input, 1.5)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.full_like(
                    input, R.const(1.5, "float32")
                )
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Fill(), example_args, {}, Expected)


def test_masked_fill():
    class Masked_Fill(Module):
        def forward(self, input: torch.Tensor, mask: torch.Tensor):
            return torch.masked_fill(input, mask, 0)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((128, 128), dtype="float32"), mask: R.Tensor((128, 128), dtype="bool")
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.const(0.0, "float32")
                lv1: R.Tensor((128, 128), dtype="float32") = R.where(mask, lv, input)
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(128, 128, dtype=torch.float32),
        torch.testing.make_tensor((128, 128), dtype=torch.bool, device="cpu"),
    )
    verify_model(Masked_Fill(), example_args, {}, Expected)


def test_masked_select():
    class MaskedSelect(Module):
        def forward(self, data: torch.Tensor, mask: torch.Tensor):
            return torch.masked_select(data, mask)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((2, 3), dtype="float32"), mask: R.Tensor((2, 3), dtype="bool")
        ) -> R.Tuple(R.Tensor(dtype="float32", ndim=1)):
            R.func_attr({"tir_var_lower_bound": {"u0": 0}, "tir_var_upper_bound": {"u0": 6}})
            u0 = T.int64()
            with R.dataflow():
                lv: R.Tensor((6,), dtype="float32") = R.reshape(data, R.shape([6]))
                lv1: R.Tensor((6,), dtype="bool") = R.reshape(mask, R.shape([6]))
                lv2: R.Tensor(dtype="int64", ndim=2) = R.nonzero(lv1)
                lv3: R.Tensor((1, u0), dtype="int64") = R.match_cast(
                    lv2, R.Tensor((1, u0), dtype="int64")
                )
                lv4: R.Tensor((u0,), dtype="int64") = R.squeeze(lv3, axis=[0])
                lv5: R.Tensor((u0,), dtype="float32") = R.take(lv, lv4, axis=0, mode="fast")
                lv6: T.bool = u0 >= 0
                lv7: T.bool = u0 <= 6
                gv: R.Tuple(R.Tensor((u0,), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.tensor([[True, False, True], [False, True, False]]),
    )
    verify_model(MaskedSelect(), example_args, {}, Expected)


@pytest.mark.skipif(not tvm.testing.device_enabled("llvm"), reason="llvm not enabled")
def test_masked_select_numerically():
    class MaskedSelect(Module):
        def forward(self, data: torch.Tensor, mask: torch.Tensor):
            return torch.masked_select(data, mask)

    example_args = (
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
        torch.tensor([[True, False, True], [False, True, False]]),
    )
    verify_model_numerically(MaskedSelect(), example_args)


def test_new_ones():
    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 3), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3), dtype="float32") = R.full(
                    (1, 2, 3), R.const(1, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((1, 2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, dtype=torch.float32),)
    verify_model(NewOnes(), example_args, {}, expected1)


def test_new_zeros():
    class NewZeros(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros(1, 128, 128)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 128, 128), dtype="float32") = R.full(
                    R.shape([1, 128, 128]), R.const(0, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((1, 128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 128, 128, dtype=torch.float32),)
    verify_model(NewZeros(), example_args, {}, expected1)


def test_copy():
    class CopyBroadcast(Module):
        def forward(self, x, src):
            x.copy_(src)
            return x

    @tvm.script.ir_module
    class expected_copy:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), src: R.Tensor((), dtype="int64")) -> R.Tuple(
            R.Tensor((2, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.astype(src, dtype="float32")
                lv1: R.Tensor((2, 3), dtype="float32") = R.broadcast_to(lv, (2, 3))
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.zeros(2, 3, dtype=torch.float32), torch.tensor(1, dtype=torch.int64))
    verify_model(CopyBroadcast(), example_args, {}, expected_copy)


def test_to_copy():
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 3, 4), dtype="float16")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float16") = R.astype(x, dtype="float16")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float16")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(ToHalf(), example_args, {}, Expected)


def test_keep_params():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            conv_weight: R.Tensor((6, 3, 7, 7), dtype="float32"),
            conv_bias: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")):
            R.func_attr({"num_input": 1})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    conv_weight,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(conv_bias, [1, 6, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    from tvm.relax.frontend import detach_params

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    model = Conv2D1()

    exported_program = torch.export.export(model, example_args)
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    mod, params = detach_params(mod)
    tvm.ir.assert_structural_equal(mod, expected1)
    func = mod["main"]
    params = params["main"]

    assert len(params) == len(func.params) - 1
    for param_var, param_tensor in zip(func.params[1:], params):
        assert tuple(x.value for x in param_var.ty.shape.values) == param_tensor.shape
        assert param_var.ty.dtype == param_tensor.dtype

    tvm.testing.assert_allclose(params[0].numpy(), model.conv.weight.detach().detach().numpy())
    tvm.testing.assert_allclose(params[1].numpy(), model.conv.bias.detach().detach().numpy())


def test_unwrap_unit_return_tuple():
    class Identity(Module):
        def forward(self, x):
            return (x,)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor(
            (256, 256), dtype="float32"
        ):
            with R.dataflow():
                gv: R.Tensor((256, 256), dtype="float32") = inp_0
                R.output(gv)
            return gv

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    verify_model(Identity(), example_args, {}, Expected, unwrap_unit_return_tuple=True)


def test_no_bind_return_tuple():
    class Identity(Module):
        def forward(self, x, y):
            return (x, y)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32"),
            inp_1: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32"), R.Tensor((256, 256), dtype="float32")):
            with R.dataflow():
                gv: R.Tensor((256, 256), dtype="float32") = inp_0
                gv1: R.Tensor((256, 256), dtype="float32") = inp_1
                R.output(gv, gv1)
            return (gv, gv1)

    example_args = (
        torch.randn(256, 256, dtype=torch.float32),
        torch.randn(256, 256, dtype=torch.float32),
    )
    verify_model(Identity(), example_args, {}, Expected, no_bind_return_tuple=True)


def test_register_buffer():
    class ModelWithBuffer(Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset", torch.arange(6.0).reshape(2, 3), persistent=False)

        def forward(self, x):
            return x + self.offset

    model = ModelWithBuffer()
    info = [((2, 3), "float32")]
    expected = make_expected(
        info, lambda x: relax.op.add(x, relax.const(model.offset.numpy())), exported=True
    )
    verify_model(model, (torch.ones(2, 3),), {}, expected)


def test_custom_op():
    class AddOp(Module):
        def forward(self, x, y):
            return torch.ops.aten.add.Tensor(x, y)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5,), dtype="float32"),
            y: R.Tensor((5,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.subtract(x, y)
                gv: R.Tuple(R.Tensor((5,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    from tvm.relax.frontend.torch.exported_program_translator import (
        ExportedProgramImporter,
    )

    def custom_add_converter(node: torch.fx.Node, self: ExportedProgramImporter) -> relax.Var:
        x = self.env[node.args[0]]
        y = self.env[node.args[1]]

        return self.block_builder.emit(R.subtract(x, y))

    example_args = (torch.randn(5, dtype=torch.float32), torch.randn(5, dtype=torch.float32))
    verify_model(
        AddOp(), example_args, {}, Expected, custom_convert_map={"add.Tensor": custom_add_converter}
    )


def test_empty_like():
    class EmptyLike(Module):
        def forward(self, data):
            return torch.empty_like(data)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((5,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.zeros(R.shape([5]), dtype="float32")
                gv: R.Tuple(R.Tensor((5,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, dtype=torch.float32),)

    verify_model(EmptyLike(), example_args, {}, Expected)


def test_one_hot():
    model = UnaryModule(lambda x: torch.nn.functional.one_hot(x, num_classes=3))
    args = (torch.tensor([0, 2], dtype=torch.int64),)
    mod = from_exported_program(export(model, args))
    vm = verify_numerically(mod, model, args)
    # Newer PyTorch versions include runtime index guards in the decomposition.
    if "relax.assert_op" in mod.script() or "R.assert_op" in mod.script():
        with pytest.raises(AssertionError, match="class|value|index"):
            vm["main"](tvm.runtime.tensor(np.array([0, 3], dtype="int64")))


def test_one_hot_invalid_num_classes():
    class OneHot(Module):
        def forward(self, indices):
            return torch.nn.functional.one_hot(indices, num_classes=0)

    example_args = (torch.randint(0, 5, (5,), dtype=torch.int64),)
    exported_program = export(OneHot(), args=example_args)

    # With the default decomposition, one_hot is rewritten to arange/equal/astype and never
    # reaches this converter. Without it, the non-positive num_classes must be rejected by
    # the frontend instead of failing an internal `depth > 0` check in relax.op.one_hot.
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        from_exported_program(exported_program, run_ep_decomposition=False)


def test_ones_like():
    class OnesLike(Module):
        def forward(self, input):
            return torch.ones_like(input)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(1.0, "float32")
                )
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)

    verify_model(OnesLike(), example_args, {}, Expected)


def test_zero_inplace():
    class ZeroInplace(Module):
        def forward(self, input):
            return input.zero_()

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(0.0, "float32")
                )
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)

    verify_model(ZeroInplace(), example_args, {}, Expected)


def test_zeros():
    class Zeros(Module):
        def forward(self, input):
            return torch.zeros(5, 2)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 2), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 2), dtype="float32") = R.full(
                    R.shape([5, 2]), R.const(0.0, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((5, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)

    verify_model(Zeros(), example_args, {}, Expected)


def test_zeros_like():
    class ZerosLike(Module):
        def forward(self, input):
            return torch.zeros_like(input)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(0.0, "float32")
                )
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)
    verify_model(ZerosLike(), example_args, {}, Expected)


@pytest.mark.parametrize("like", [False, True])
def test_random_constant(like):
    model = UnaryModule(lambda x: x + (torch.randn_like(x) if like else torch.randn(2, 3)))
    exported = export(model, (torch.zeros(2, 3),))
    state = np.random.get_state()
    try:
        np.random.seed(0)
        reference = np.random.randn(2, 3).astype("float32")
        np.random.seed(0)
        mod = from_exported_program(exported)
    finally:
        np.random.set_state(state)
    values = constants(mod)
    assert len(values) == 1
    np.testing.assert_array_equal(values[0], reference)


def test_type_as():
    class TypeAs(Module):
        def forward(self, input, other):
            return input.type_as(other)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((128, 128), dtype="float32"),
            other: R.Tensor((128, 128), dtype="float16"),
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float16")):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float16") = R.astype(input, dtype="float16")
                gv: R.Tuple(R.Tensor((128, 128), dtype="float16")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.rand(128, 128, dtype=torch.float32),
        torch.rand(128, 128, dtype=torch.float16),
    )

    verify_model(TypeAs(), example_args, {}, Expected)


def test_unflatten():
    class Unflatten1(Module):
        def forward(self, input):
            return torch.ops.aten.unflatten(input, -2, (3, 5))

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((2, 15, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 5, 7), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 5, 7), dtype="float32") = R.reshape(inp_0, [2, 3, 5, 7])
                gv: R.Tuple(R.Tensor((2, 3, 5, 7), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 15, 7, dtype=torch.float32),)

    verify_model(Unflatten1(), example_args, {}, Expected)


def test_gather():
    class Gather0(Module):
        def forward(self, data, indices):
            return torch.gather(data, 0, indices)

    class Gather2(Module):
        def forward(self, data, indices):
            return torch.gather(data, -1, indices)

    @tvm.script.ir_module
    class Expected0:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="int64"),
        ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=0)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="int64"),
        ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=-1)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randint(0, 2, (2, 3), dtype=torch.int64),
    )

    verify_model(Gather0(), example_args, {}, Expected0)
    verify_model(Gather2(), example_args, {}, Expected2)


def test_index_put():
    # Test case 1: 1D input

    # Test case 2: 2D input
    class IndexPut2D(Module):
        def forward(self, data, indices_0, indices_1, values):
            indices_tuple = (indices_0, indices_1)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_2d = (
        torch.randn(32, 64, dtype=torch.float32),
        torch.randint(0, 32, (128,), dtype=torch.int64),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected2D:
        @R.function
        def main(
            data: R.Tensor((32, 64), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            indices_1: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32, 64), dtype="float32") = R.index_put(
                    data, (indices_0, indices_1), values, accumulate=False
                )
                gv: R.Tuple(R.Tensor((32, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # Test case 6: 2D input with multi-dimensional index (broadcasting)
    # This tests the multi-dimensional index support with broadcasting
    class IndexPutBroadcast1D(Module):
        def forward(self, data, indices_1):
            indices_0 = torch.arange(data.shape[0]).unsqueeze(1)
            values = torch.ones(data.shape[0], len(indices_1), dtype=data.dtype)
            return data.index_put_((indices_0, indices_1), values, accumulate=False)

    example_args_broadcast1 = (
        torch.randn(32, 64, dtype=torch.float32),
        torch.randint(0, 64, (10,), dtype=torch.int64),
    )

    @I.ir_module
    class ExpectedBroadcast1D:
        @R.function
        def main(
            data: R.Tensor((32, 64), dtype="float32"),
            indices_1: R.Tensor((10,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(32), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((32, 1), dtype="int64") = R.expand_dims(lv, axis=[1])
                lv2: R.Tensor((32, 10), dtype="float32") = R.full(
                    R.shape([32, 10]), R.const(1.0, "float32"), dtype="float32"
                )
                lv3: R.Tensor((32, 64), dtype="float32") = R.index_put(
                    data, (lv1, indices_1), lv2, accumulate=False
                )
                gv: R.Tuple(R.Tensor((32, 64), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    # Test case 7: 2D input with multi-dimensional index (second position)

    # Test case 8: 3D input with mixed 1D and 2D indices

    # Test case 9: batched indexing with slice (e.g., M[:, rows, cols] = x)
    class IndexPutBatchedWithNone(Module):
        def forward(self, x):
            B = x.size(0)
            M = torch.zeros(B, 11, 11)
            rows = torch.arange(10)
            cols = rows + 1
            M[:, rows, cols] = x  # Batched index assignment
            return M

    example_args_batched_none = (torch.randn(2, 10, dtype=torch.float32),)

    @I.ir_module
    class ExpectedBatchedWithNone:
        @R.function
        def main(x: R.Tensor((2, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 11, 11), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 11, 11), dtype="float32") = R.full(
                    R.shape([2, 11, 11]), R.const(0.0, "float32"), dtype="float32"
                )
                lv1: R.Tensor((10,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(10), R.prim_value(1), dtype="int64"
                )
                lv2: R.Tensor((10,), dtype="int64") = R.add(lv1, R.const(1, "int64"))
                lv3: R.Tensor((2,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(2), R.prim_value(1), dtype="int64"
                )
                lv4: R.Tensor((2, 1), dtype="int64") = R.reshape(lv3, R.shape([2, 1]))
                lv5: R.Tensor((2, 11, 11), dtype="float32") = R.index_put(
                    lv, (lv4, lv1, lv2), x, accumulate=False
                )
                gv: R.Tuple(R.Tensor((2, 11, 11), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    # Run verification for each case
    verify_model(IndexPut2D(), example_args_2d, {}, Expected2D)
    verify_model(IndexPutBroadcast1D(), example_args_broadcast1, {}, ExpectedBroadcast1D)
    verify_model(IndexPutBatchedWithNone(), example_args_batched_none, {}, ExpectedBatchedWithNone)


def test_index_put_mutation_through_alias_regression():
    class IndexPutAlias(Module):
        def forward(self, x, idx, values):
            y = torch.ops.aten.alias.default(x)
            y[idx] = values
            return x, y

    example_args = (
        torch.zeros(5, dtype=torch.float32),
        torch.tensor([1, 3], dtype=torch.int64),
        torch.tensor([2.0, 4.0], dtype=torch.float32),
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5,), dtype="float32"),
            idx: R.Tensor((2,), dtype="int64"),
            values: R.Tensor((2,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((5,), dtype="float32"),
            R.Tensor((5,), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.index_put(
                    x, (idx,), values, accumulate=False
                )
                # Mutation outputs introduced by functionalization are dropped;
                # only the user outputs (x, y) remain.
                gv: R.Tuple(
                    R.Tensor((5,), dtype="float32"),
                    R.Tensor((5,), dtype="float32"),
                ) = (
                    lv,
                    lv,
                )
                R.output(gv)
            return gv

    verify_model(IndexPutAlias(), example_args, {}, Expected)


def test_flip():
    shape = (2, 3, 4)
    model = UnaryModule(lambda x: torch.flip(x, dims=(0, -1)))
    info = [(shape, "float32")]
    expected = make_expected(
        info, lambda x: relax.op.flip(relax.op.flip(x, axis=0), axis=-1), exported=True
    )
    verify_model(model, (torch.randn(shape),), {}, expected)


def test_take():
    class Take(Module):
        def forward(self, data, indices):
            return torch.take(data, indices)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((5,), dtype="float32"),
            indices: R.Tensor((3,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((3,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.reshape(data, R.shape([5]))
                lv1: R.Tensor((3,), dtype="float32") = R.take(lv, indices, axis=0, mode="fast")
                gv: R.Tuple(R.Tensor((3,), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(5, dtype=torch.float32),
        torch.randint(0, 5, (3,), dtype=torch.int64),
    )

    verify_model(Take(), example_args, {}, Expected)


def test_any():
    class AnyAten(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.any(x, dim=1)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="bool"),
        ) -> R.Tuple(R.Tensor((2,), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="int8") = relax.op.astype(x, dtype="int8")
                lv2: R.Tensor((2,), dtype="int8") = relax.op.max(lv, axis=1, keepdims=False)
                lv3: R.Tensor((2,), dtype="bool") = relax.op.astype(lv2, dtype="bool")
                gv: R.Tuple(R.Tensor((2,), dtype="bool")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.bool),)
    verify_model(AnyAten(), example_args, {}, Expected)


def test_std():
    # torch.std(x) defaults to correction=1 (Bessel); decomposes to var.correction + sqrt.
    class Std(Module):
        def forward(self, x):
            return torch.std(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.variance(x, axis=None, keepdims=False)
                lv1: R.Tensor((), dtype="float32") = R.multiply(lv, R.const(15.0 / 14.0, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sqrt(lv1)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Std(), example_args, {}, Expected)


def test_var():
    # torch.var(x) defaults to correction=1 (Bessel).
    class Var(Module):
        def forward(self, x):
            return torch.var(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.variance(x, axis=None, keepdims=False)
                lv1: R.Tensor((), dtype="float32") = R.multiply(lv, R.const(15.0 / 14.0, "float32"))
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Var(), example_args, {}, Expected)


def test_var_correction():
    class VarCorrection2(Module):
        def forward(self, x):
            return torch.var(x, dim=-1, correction=2)

    class VarCorrection0(Module):
        def forward(self, x):
            return torch.var(x, dim=1, correction=0)

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = R.variance(x, axis=[-1], keepdims=False)
                lv1: R.Tensor((2,), dtype="float32") = R.multiply(lv, R.const(5.0 / 3.0, "float32"))
                gv: R.Tuple(R.Tensor((2,), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected0:
        @R.function
        def main(
            x: R.Tensor((2, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = R.variance(x, axis=[1], keepdims=False)
                gv: R.Tuple(R.Tensor((2,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 5, dtype=torch.float32),)
    verify_model(VarCorrection2(), example_args, {}, Expected2)
    verify_model(VarCorrection0(), example_args, {}, Expected0)


@pytest.mark.parametrize(
    "torch_dtype,relax_dtype",
    [(torch.float32, "float32"), (torch.bool, "bool")],
)
def test_prod(torch_dtype, relax_dtype):
    class Prod(Module):
        def forward(self, x):
            return torch.prod(x, dtype=torch_dtype)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype=relax_dtype),
        ) -> R.Tuple(R.Tensor((), dtype=relax_dtype)):
            with R.dataflow():
                lv: R.Tensor((), dtype=relax_dtype) = R.prod(x, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype=relax_dtype)) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.ones(5, 3, dtype=torch_dtype),)
    verify_model(Prod(), example_args, {}, Expected)


def test_cumprod():
    class Cumprod(Module):
        def forward(self, x):
            return torch.cumprod(x, 0)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.cumprod(inp_0, axis=0, exclusive=False)
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_input = torch.randn(5, 3, dtype=torch.float32)
    verify_model(Cumprod(), (example_input,), {}, Expected)


def test_where():
    class Where(Module):
        def forward(self, condition, x, y):
            return torch.where(condition, x, y)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="bool"),
            inp_1: R.Tensor((5, 3), dtype="float32"),
            inp_2: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.where(inp_0, inp_1, inp_2)
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    condition = torch.testing.make_tensor((5, 3), dtype=torch.bool, device="cpu")
    x = torch.randn(5, 3, dtype=torch.float32)
    y = torch.randn(5, 3, dtype=torch.float32)

    verify_model(Where(), (condition, x, y), {}, Expected)


def test_bucketize():
    class Bucketize(Module):
        def forward(self, input_tensor, boundaries):
            return torch.bucketize(input_tensor, boundaries)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((20,), dtype="int64"), boundaries: R.Tensor((10,), dtype="int64")
        ) -> R.Tuple(R.Tensor((20,), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((20,), dtype="int64") = R.bucketize(
                    input, boundaries, out_int32=False, right=False
                )
                gv: R.Tuple(R.Tensor((20,), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    input_tensor = torch.arange(0, 20)
    boundaries = torch.arange(0, 20, 2)

    verify_model(Bucketize(), (input_tensor, boundaries), {}, Expected)


def test_bucketize_numerically():
    class Bucketize(Module):
        def forward(self, x, boundaries):
            return (
                torch.bucketize(x, boundaries, right=False, out_int32=False),
                torch.bucketize(x, boundaries, right=True, out_int32=True),
            )

    verify_model_numerically(
        Bucketize(), (torch.tensor([-1.0, 0.0, 1.0, 3.0]), torch.tensor([0.0, 1.0, 2.0]))
    )


def test_sort():
    class Sort(Module):
        def forward(self, x):
            return torch.sort(x, dim=1, descending=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int64") = R.argsort(
                    x, axis=1, descending=True, dtype="int64"
                )
                lv1: R.Tensor((5, 3), dtype="float32") = R.gather_elements(x, lv, axis=1)
                lv2: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")) = (
                    lv1,
                    lv,
                )
                lv3: R.Tensor((5, 3), dtype="float32") = lv2[0]
                lv4: R.Tensor((5, 3), dtype="int64") = lv2[1]
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")) = (
                    lv3,
                    lv4,
                )
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Sort(), example_args, {}, Expected)


def test_topk():
    class Topk(Module):
        def forward(self, x):
            return torch.topk(x, k=2, dim=1, largest=True, sorted=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")) = (
                    R.topk(x, k=2, axis=1, ret_type="both", largest=True, dtype="int64")
                )
                lv1: R.Tensor((5, 2), dtype="float32") = lv[0]
                lv2: R.Tensor((5, 2), dtype="int64") = lv[1]
                gv: R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")) = (
                    lv1,
                    lv2,
                )
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Topk(), example_args, {}, Expected)


def test_dynamic_shape():
    class DynamicModel(torch.nn.Module):
        def forward(self, x1, x2):
            return torch.ops.aten.add.Tensor(x1, x2)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            lhs: R.Tensor(("s0", 4), dtype="float32"),
            rhs: R.Tensor(("s0", 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor(("s0", 4), dtype="float32")):
            s0 = T.int64()
            R.func_attr({"tir_var_lower_bound": {"s24": 0}})
            with R.dataflow():
                lv: R.Tensor((s0, 4), dtype="float32") = R.add(lhs, rhs)
                gv: R.Tuple(R.Tensor((s0, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 4), torch.randn(2, 4))
    batch = torch.export.Dim("batch")
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    verify_model(
        DynamicModel(),
        example_args,
        {},
        Expected,
        dynamic_shapes=dynamic_shapes,
        run_ep_decomposition=True,
        map_free_vars=True,
    )


def test_item():
    class Item(Module):
        def forward(self, x):
            return x.item()

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((1,), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.take(input, R.const(0, "int64"), axis=0)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, dtype=torch.float32),)
    verify_model(Item(), example_args, {}, Expected)


@pytest.mark.parametrize("p", [float("inf"), float("-inf"), 0.5])
def test_norm(p):
    shape = (2, 3, 4)
    axis = 1

    def expected(x):
        if p == float("inf"):
            return relax.op.max(relax.op.abs(x), axis=axis, keepdims=True)
        if p == float("-inf"):
            return relax.op.min(relax.op.abs(x), axis=axis, keepdims=True)
        if p == "fro":
            return relax.op.sqrt(relax.op.sum(relax.op.multiply(x, x), axis=axis, keepdims=True))
        return relax.op.power(
            relax.op.sum(
                relax.op.power(relax.op.abs(x), relax.const(p, "float32")), axis=axis, keepdims=True
            ),
            relax.const(1 / p, "float32"),
        )

    model = UnaryModule(lambda x: torch.norm(x, p=p, dim=axis, keepdim=True))
    info = [(shape, "float32")]
    expected = make_expected(info, expected, exported=True)
    verify_model(model, (torch.randn(shape),), {}, expected)


def test_eye():
    class Eye1(Module):
        def forward(self, input):
            return torch.eye(3, 5, dtype=torch.float32)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(input: R.Tensor((3, 5), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="uint8") = R.arange(
                    R.prim_value(0), R.prim_value(3), R.prim_value(1), dtype="uint8"
                )
                lv1: R.Tensor((5,), dtype="uint8") = R.arange(
                    R.prim_value(0), R.prim_value(5), R.prim_value(1), dtype="uint8"
                )
                lv2: R.Tensor((3, 1), dtype="uint8") = R.expand_dims(lv, axis=[-1])
                lv3: R.Tensor((3, 5), dtype="bool") = R.equal(lv2, lv1)
                lv4: R.Tensor((3, 5), dtype="float32") = R.astype(lv3, dtype="float32")
                gv: R.Tuple(R.Tensor((3, 5), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    example_args1 = (torch.randn(3, 5, dtype=torch.float32),)
    verify_model(Eye1(), example_args1, {}, Expected1)


def test_cross_entropy():
    class CrossEntropyModule(Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()
            self.target = torch.tensor([0, 1, 2, 1])

        def forward(self, x):
            return self.criterion(x, self.target)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(x: R.Tensor((4, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 3), dtype="float32") = R.nn.log_softmax(x, axis=1)
                lv1: R.Tensor((4,), dtype="bool") = R.not_equal(
                    R.const([0, 1, 2, 1], dtype="int64"), R.const(-100, "int64")
                )
                lv2: R.Tensor((), dtype="int64") = R.const(0, "int64")
                lv3: R.Tensor((4,), dtype="int64") = R.where(
                    lv1, R.const([0, 1, 2, 1], dtype="int64"), lv2
                )
                lv4: R.Tensor((4, 1), dtype="int64") = R.expand_dims(lv3, axis=[1])
                lv5: R.Tensor((4, 1), dtype="float32") = R.gather_elements(lv, lv4, axis=1)
                lv6: R.Tensor((4,), dtype="float32") = R.squeeze(lv5, axis=[1])
                lv7: R.Tensor((4,), dtype="float32") = R.negative(lv6)
                lv8: R.Tensor((4,), dtype="bool") = R.not_equal(
                    R.const([0, 1, 2, 1], dtype="int64"), R.const(-100, "int64")
                )
                lv9: R.Tensor((), dtype="float32") = R.const(0.0, "float32")
                lv10: R.Tensor((4,), dtype="float32") = R.where(lv8, lv7, lv9)
                lv11: R.Tensor((4,), dtype="bool") = R.not_equal(
                    R.const([0, 1, 2, 1], dtype="int64"), R.const(-100, "int64")
                )
                lv12: R.Tensor((4,), dtype="int64") = R.astype(lv11, dtype="int64")
                lv13: R.Tensor((), dtype="int64") = R.sum(lv12, axis=None, keepdims=False)
                lv14: R.Tensor((), dtype="float32") = R.astype(lv13, dtype="float32")
                lv15: R.Tensor((), dtype="float32") = R.sum(lv10, axis=None, keepdims=False)
                lv16: R.Tensor((), dtype="float32") = R.divide(lv15, lv14)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv16,)
                R.output(gv)
            return gv

    example_args1 = (torch.randn(4, 3, dtype=torch.float32),)
    verify_model(CrossEntropyModule(), example_args1, {}, Expected1)


def test_linspace():
    class Linspace(Module):
        def forward(self, input):
            return torch.linspace(0, 1, steps=9, dtype=torch.float32)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((9, 9), dtype="float32")) -> R.Tuple(
            R.Tensor((9,), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((9,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(9), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((9,), dtype="bool") = R.less(lv, R.const(4, "int64"))
                lv2: R.Tensor((9,), dtype="float32") = R.astype(lv, dtype="float32")
                lv3: R.Tensor((9,), dtype="float32") = R.multiply(lv2, R.const(0.125, "float32"))
                lv4: R.Tensor((9,), dtype="float32") = R.add(lv3, R.const(0.0, "float32"))
                lv5: R.Tensor((9,), dtype="int64") = R.subtract(R.const(8, "int64"), lv)
                lv6: R.Tensor((9,), dtype="float32") = R.astype(lv5, dtype="float32")
                lv7: R.Tensor((9,), dtype="float32") = R.multiply(lv6, R.const(0.125, "float32"))
                lv8: R.Tensor((9,), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv7)
                lv9: R.Tensor((9,), dtype="float32") = R.where(lv1, lv4, lv8)
                gv: R.Tuple(R.Tensor((9,), dtype="float32")) = (lv9,)
                R.output(gv)
            return gv

    example_args = (torch.randn(9, 9, dtype=torch.float32),)
    verify_model(Linspace(), example_args, {}, Expected)


@pytest.mark.parametrize(
    "torch_dtype, relax_dtype",
    [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.int64, "int64"),
        (torch.int32, "int32"),
        (torch.bool, "bool"),
    ],
)
def test_dtypes(torch_dtype, relax_dtype):
    example_args = (
        torch.testing.make_tensor((10, 10), dtype=torch_dtype, device="cpu", low=0, high=10),
        torch.testing.make_tensor((10, 10), dtype=torch_dtype, device="cpu", low=0, high=10),
    )

    class Model(Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor):
            return torch.ops.aten.add(lhs, rhs)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype=relax_dtype),
            rhs: R.Tensor((10, 10), dtype=relax_dtype),
        ) -> R.Tuple(R.Tensor((10, 10), dtype=relax_dtype)):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype=relax_dtype) = relax.op.add(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype=relax_dtype)) = (lv,)
                R.output(gv)
            return gv

    verify_model(Model(), example_args, {}, Expected)


def test_mm():
    class MatrixMultiply(Module):
        def forward(self, a, b):
            return torch.mm(a, b)

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
    )

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((2, 3), dtype="float32"),
            b: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 4), dtype="float32") = R.matmul(a, b, out_dtype="float32")
                gv: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(MatrixMultiply(), example_args, {}, Expected)


def test_sparse_mm():
    class SparseMatrixMultiply(Module):
        def forward(self, sparse_input, dense_input):
            return torch.sparse.mm(sparse_input, dense_input)

    indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    sparse_input = torch.sparse_coo_tensor(indices, values, size=(3, 100))
    dense_input = torch.randn(100, 50, dtype=torch.float32)

    example_args = (sparse_input, dense_input)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            sparse_input: R.Tensor((3, 100), dtype="float32"),
            dense_input: R.Tensor((100, 50), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 50), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 50), dtype="float32") = R.full(
                    R.shape([3, 50]), R.const(0.0, "float32"), dtype="float32"
                )
                lv1: R.Tensor((3, 50), dtype="float32") = R.matmul(
                    sparse_input, dense_input, out_dtype="float32"
                )
                gv: R.Tuple(R.Tensor((3, 50), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(SparseMatrixMultiply(), example_args, {}, Expected)


@pytest.mark.parametrize("rnn_type", [nn.LSTM, nn.GRU, nn.RNN], ids=["lstm", "gru", "rnn-tanh"])
@pytest.mark.parametrize(
    "batch_first, bidirectional",
    [(True, False), (False, True)],
    ids=["batch-first", "bidirectional"],
)
def test_recurrent(rnn_type, batch_first, bidirectional):
    class Recurrent(Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_type(3, 4, batch_first=batch_first, bidirectional=bidirectional)

        def forward(self, x):
            return self.rnn(x)

    # Exercise both layouts and direction counts without repeating their product.
    # Compare sequence and all hidden/cell states for every cell type.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        x = torch.randn(2, 3, 3) if batch_first else torch.randn(3, 2, 3)
        verify_model_numerically(
            Recurrent(),
            (x,),
            rtol=1e-4,
            atol=1e-5,
            run_ep_decomposition=True,
        )


def test_tensor_none_tuple():
    example_args = (torch.tensor([1.0, 2.0, 3.0]),)

    class TensorNoneModel(Module):
        def forward(self, x):
            return x + 1, None

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32")) -> R.Tuple(
            R.Tensor((3,), dtype="float32"), R.Any
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.add(x, R.const(1.0, "float32"))
                gv: R.Tuple(R.Tensor((3,), dtype="float32"), R.Any) = (lv, R.null_value())
                R.output(gv)
            return gv

    verify_model(TensorNoneModel(), example_args, {}, Expected)


@pytest.mark.parametrize("bounded", [True, False])
def test_dynamic_shape_constraints(bounded):
    class Model(Module):
        def forward(self, x, added, subtracted, multiplied):
            return x, added, subtracted, multiplied

    dim = torch.export.Dim("batch", min=3, **({"max": 8} if bounded else {}))
    args = tuple(torch.randn(n, 2) for n in ((4, 5, 3, 8) if bounded else (4, 4, 4, 4)))
    dynamic = ({0: dim}, {0: dim + 1}, {0: dim - 1}, {0: 2 * dim}) if bounded else ({0: dim},) * 4
    mod = from_exported_program(export(Model(), args, dynamic_shapes=dynamic))
    func = mod["main"]
    base, added, subtracted, multiplied = [p.ty.shape[0] for p in func.params]
    analyzer = tvm.arith.Analyzer()
    for actual, expected in zip(
        (added, subtracted, multiplied), (base + 1, base - 1, base * 2) if bounded else (base,) * 3
    ):
        assert analyzer.can_prove_equal(actual, expected)
    variables = tvm.tirx.analysis.undefined_vars(base)
    lower = func.attrs["tir_var_lower_bound"]
    upper = func.attrs.get("tir_var_upper_bound", {})
    for var in variables:
        analyzer.update(
            var,
            tvm.arith.ConstIntBound(
                int(lower[var.name]),
                int(upper[var.name]) if var.name in upper else tvm.arith.ConstIntBound.POS_INF,
            ),
        )
    bound = analyzer.const_int_bound(base)
    assert bound.min_value == 3
    assert bound.max_value == (8 if bounded else tvm.arith.ConstIntBound.POS_INF)


def test_sym_size_int():
    class SymSizeInt(Module):
        def forward(self, x):
            shape_dim = torch.ops.aten.sym_size.int(x, 0)
            return x.reshape(shape_dim, -1)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("s0", 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor(("s0", 12), dtype="float32")
        ):
            s0 = T.int64()
            R.func_attr({"tir_var_lower_bound": {"s77": 0}})
            with R.dataflow():
                lv: R.Tensor((s0, 12), dtype="float32") = R.reshape(x, R.shape([s0, 12]))
                gv: R.Tuple(R.Tensor((s0, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, 4),)
    dynamic_shapes = {"x": {0: torch.export.Dim("dim")}}
    verify_model(
        SymSizeInt(),
        example_args,
        {},
        Expected,
        dynamic_shapes=dynamic_shapes,
        map_free_vars=True,
    )


def test_exponential_unsupported():
    model = UnaryModule(lambda x: x.exponential_())
    with pytest.raises(NotImplementedError, match="exponential sampling"):
        from_exported_program(export(model, (torch.ones(2, 3),)))


def test_max_dim():
    class MaxDim(Module):
        def forward(self, x):
            return torch.max(x, dim=1), torch.max(x, dim=1, keepdim=True)

    def expected(x):
        emit = relax.BlockBuilder.current().emit
        outputs = []
        for keepdim in (False, True):
            top = emit(relax.op.topk(x, k=1, axis=1, ret_type="both", largest=True, dtype="int64"))
            fields = [
                top[i] if keepdim else emit(relax.op.squeeze(top[i], axis=[1])) for i in range(2)
            ]
            result = emit(relax.Tuple(fields))
            outputs.extend([emit(result[0]), emit(result[1])])
        return outputs

    verify_model(
        MaxDim(),
        (torch.randn(2, 3, 4),),
        {},
        make_expected([((2, 3, 4), "float32")], expected, exported=True),
    )


def test_scatter_value():
    class ScatterValue(Module):
        def forward(self, x, index):
            return x.scatter(1, index, 0.5)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4, 8), dtype="float32"),
            index: R.Tensor((4, 2), dtype="int64"),
        ) -> R.Tuple(R.Tensor((4, 8), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 2), dtype="float32") = R.broadcast_to(
                    R.const(0.5, "float32"), R.shape([4, 2])
                )
                lv1: R.Tensor((4, 8), dtype="float32") = R.scatter_elements(x, index, lv, axis=1)
                gv: R.Tuple(R.Tensor((4, 8), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 8, dtype=torch.float32),
        torch.randint(0, 8, (4, 2), dtype=torch.int64),
    )
    verify_model(ScatterValue(), example_args, {}, Expected)


def test_scatter_src():
    class ScatterSrc(Module):
        def forward(self, x, index, src):
            return x.scatter(1, index, src)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4, 8), dtype="float32"),
            index: R.Tensor((4, 2), dtype="int64"),
            src: R.Tensor((4, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 8), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 8), dtype="float32") = R.scatter_elements(x, index, src, axis=1)
                gv: R.Tuple(R.Tensor((4, 8), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 8, dtype=torch.float32),
        torch.randint(0, 8, (4, 2), dtype=torch.int64),
        torch.randn(4, 2, dtype=torch.float32),
    )
    verify_model(ScatterSrc(), example_args, {}, Expected)


def test_grid_sample():
    class GridSample(Module):
        def forward(self, input, grid):
            return torch.nn.functional.grid_sample(
                input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 2, 2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 2, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 2, 2), dtype="float32") = R.image.grid_sample(
                    input_1,
                    grid,
                    method="bilinear",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 4, 4, dtype=torch.float32),
        torch.randn(1, 2, 2, 2, dtype=torch.float32),
    )
    verify_model(GridSample(), example_args, {}, expected)


def test_torchvision_roi_align():
    torchvision = pytest.importorskip("torchvision")

    class ROIAlign(Module):
        def forward(self, input, rois):
            return torchvision.ops.roi_align(
                input,
                rois,
                output_size=(3, 3),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=False,
            )

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 8, 8), dtype="float32"),
            rois: R.Tensor((2, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 3, 3), dtype="float32") = R.vision.roi_align(
                    input_1,
                    rois,
                    pooled_size=(3, 3),
                    spatial_scale=1.0,
                    sample_ratio=2,
                    layout="NCHW",
                    mode="avg",
                )
                gv: R.Tuple(R.Tensor((2, 3, 3, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 8, 8, dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 1.0, 6.0, 6.0], [0.0, 0.5, 0.5, 7.0, 7.0]], dtype=torch.float32),
    )
    verify_model(ROIAlign(), example_args, {}, expected)


def test_torchvision_roi_align_aligned():
    torchvision = pytest.importorskip("torchvision")

    class ROIAlign(Module):
        def forward(self, input, rois):
            return torchvision.ops.roi_align(
                input,
                rois,
                output_size=(1, 1),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )

    example_args = (
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        torch.tensor([[0.0, 1.0, 1.0, 1.2, 1.2]], dtype=torch.float32),
    )
    verify_model_numerically(ROIAlign(), example_args, rtol=1e-5, atol=1e-5)


def test_upsample_nearest2d():
    class UpsampleNearest2dScale(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, scale_factor=2.0, mode="nearest")

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    @tvm.script.ir_module
    class expected_scale:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 20, 20), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 20, 20), dtype="float32") = R.image.resize2d(
                    input_1,
                    size=(20, 20),
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode="half_pixel",
                )
                gv: R.Tuple(R.Tensor((1, 3, 20, 20), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(UpsampleNearest2dScale(), example_args, {}, expected_scale)


def test_from_exported_program_sparse_csr_buffer():
    class SparseCsrBufferModule(nn.Module):
        def __init__(self):
            super().__init__()
            crow_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
            col_indices = torch.tensor([0, 1], dtype=torch.int64)
            values = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
            csr_tensor = torch.sparse_csr_tensor(
                crow_indices, col_indices, values, dtype=torch.float32
            )
            self.register_buffer("csr_tensor", csr_tensor)
            self.csr_tensor.requires_grad_(True)

        def forward(self, x):
            csr2 = self.csr_tensor.to_sparse(layout=torch.sparse_csr)
            y = torch.matmul(csr2, x)
            return y.sum()

    model = SparseCsrBufferModule().eval()
    x = torch.ones((2, 1), dtype=torch.float32)
    exported_program = export(model, (x,))
    mod = from_exported_program(exported_program)
    weights = [value for value in constants(mod) if value.shape == (2, 2)]
    assert len(weights) == 1
    np.testing.assert_array_equal(weights[0], model.csr_tensor.detach().to_dense().numpy())
    assert tuple(mod["main"].ret_ty.fields[0].shape) == ()


def test_cond_basic():
    """Basic data-dependent cond with runtime predicate."""

    class CondModel(Module):
        def forward(self, x):
            def true_fn(x):
                return x.cos()

            def false_fn(x):
                return x.sin()

            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    @tvm.script.ir_module
    class expected:
        @R.function
        def cond_true_branch_0(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            gv: R.Tensor((3, 4), dtype="float32") = R.cos(x)
            gv1: R.Tensor((3, 4), dtype="float32") = gv
            return gv1

        @R.function
        def cond_false_branch_1(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            gv: R.Tensor((3, 4), dtype="float32") = R.sin(x)
            gv1: R.Tensor((3, 4), dtype="float32") = gv
            return gv1

        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32")):
            cls = expected
            gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((), dtype="bool") = R.greater(gv, R.const(0.0, "float32"))
            if gv1:
                gv2: R.Tensor((3, 4), dtype="float32") = cls.cond_true_branch_0(x)
                cond_result: R.Tensor((3, 4), dtype="float32") = gv2
            else:
                gv3: R.Tensor((3, 4), dtype="float32") = cls.cond_false_branch_1(x)
                cond_result: R.Tensor((3, 4), dtype="float32") = gv3
            return (cond_result,)

    verify_model(CondModel(), (torch.randn(3, 4),), {}, expected, map_free_vars=True)


def test_cond_shape_predicate():
    """Cond with a shape-derived predicate and dynamic shapes."""

    class CondShapeModel(Module):
        def forward(self, x):
            def true_fn(x):
                return x + 1.0

            def false_fn(x):
                return x - 1.0

            return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    @tvm.script.ir_module
    class expected:
        @R.function
        def cond_true_branch_0(
            x: R.Tensor(("s77", 4), dtype="float32"),
        ) -> R.Tensor(("s77", 4), dtype="float32"):
            s77 = T.int64()
            gv: R.Tensor((s77, 4), dtype="float32") = R.add(x, R.const(1.0, "float32"))
            gv1: R.Tensor((s77, 4), dtype="float32") = gv
            return gv1

        @R.function
        def cond_false_branch_1(
            x: R.Tensor(("s77", 4), dtype="float32"),
        ) -> R.Tensor(("s77", 4), dtype="float32"):
            s77 = T.int64()
            gv: R.Tensor((s77, 4), dtype="float32") = R.subtract(x, R.const(1.0, "float32"))
            gv1: R.Tensor((s77, 4), dtype="float32") = gv
            return gv1

        @R.function
        def main(
            x: R.Tensor(("s77", 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor(("s77", 4), dtype="float32")):
            s77 = T.int64()
            R.func_attr({"tir_var_lower_bound": {"s77": 1}})
            cls = expected
            gv: T.bool = s77 > 4
            if gv:
                gv1: R.Tensor((s77, 4), dtype="float32") = cls.cond_true_branch_0(x)
                cond_result: R.Tensor((s77, 4), dtype="float32") = gv1
            else:
                gv2: R.Tensor((s77, 4), dtype="float32") = cls.cond_false_branch_1(x)
                cond_result: R.Tensor((s77, 4), dtype="float32") = gv2
            return (cond_result,)

    batch = torch.export.Dim("batch", min=1)
    verify_model(
        CondShapeModel(),
        (torch.randn(3, 4),),
        {},
        expected,
        dynamic_shapes={"x": {0: batch}},
        map_free_vars=True,
    )


def test_cond_shape_comparison():
    class CondShapeModel(Module):
        def forward(self, x):
            def true_fn(x):
                return x + 1.0

            def false_fn(x):
                return x - 1.0

            return (
                torch.cond(x.shape[0] == x.shape[1], true_fn, false_fn, (x,)),
                torch.cond(x.shape[0] != x.shape[1], true_fn, false_fn, (x,)),
            )

    rows = torch.export.Dim("rows", min=1, max=8)
    columns = torch.export.Dim("columns", min=1, max=8)
    verify_model_numerically(
        CondShapeModel(),
        (torch.zeros(2, 3),),
        dynamic_shapes={"x": {0: rows, 1: columns}},
        input_sets=[(torch.zeros(shape),) for shape in ((2, 3), (3, 3))],
        rtol=0,
        atol=0,
    )


def test_cond_tuple_output():
    """Cond where both branches return a tuple."""

    class CondTupleModel(Module):
        def forward(self, x):
            def true_fn(x):
                return (x.cos(), x.sin())

            def false_fn(x):
                return (x.sin(), x.cos())

            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    @tvm.script.ir_module
    class expected:
        @R.function
        def cond_true_branch_0(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")):
            gv: R.Tensor((3, 4), dtype="float32") = R.cos(x)
            gv1: R.Tensor((3, 4), dtype="float32") = R.sin(x)
            gv2: R.Tensor((3, 4), dtype="float32") = gv
            gv3: R.Tensor((3, 4), dtype="float32") = gv1
            return (gv2, gv3)

        @R.function
        def cond_false_branch_1(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")):
            gv: R.Tensor((3, 4), dtype="float32") = R.sin(x)
            gv1: R.Tensor((3, 4), dtype="float32") = R.cos(x)
            gv2: R.Tensor((3, 4), dtype="float32") = gv
            gv3: R.Tensor((3, 4), dtype="float32") = gv1
            return (gv2, gv3)

        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")):
            cls = expected
            gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((), dtype="bool") = R.greater(gv, R.const(0.0, "float32"))
            if gv1:
                gv2: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = cls.cond_true_branch_0(x)
                cond_result: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = gv2
            else:
                gv3: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = cls.cond_false_branch_1(x)
                cond_result: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = gv3
            gv4: R.Tensor((3, 4), dtype="float32") = cond_result[0]
            gv5: R.Tensor((3, 4), dtype="float32") = cond_result[1]
            return (gv4, gv5)

    verify_model(CondTupleModel(), (torch.randn(3, 4),), {}, expected, map_free_vars=True)


def test_cond_nested():
    """Nested cond: a cond inside one of the branches."""

    class CondNestedModel(Module):
        def forward(self, x):
            def true_fn(x):
                def inner_true(x):
                    torch._assert_async(torch.all(x < 2), "x must be less than 2")
                    return x * 2.0

                def inner_false(x):
                    return x * 3.0

                return torch.cond(x.sum() > 1, inner_true, inner_false, (x,))

            def false_fn(x):
                return x - 1.0

            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    inputs = [(torch.full((2, 3), value),) for value in (-1.0, 0.1, 1.0)]
    verify_model_numerically(CondNestedModel(), inputs[0], input_sets=inputs)


def test_affine_grid():
    class AffineGrid(Module):
        def forward(self, theta):
            return torch.nn.functional.affine_grid(theta, [1, 3, 16, 16], align_corners=True)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            theta: R.Tensor((1, 2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 16, 16, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 2, 16, 16), dtype="float32") = R.image.affine_grid(
                    theta, size=(16, 16)
                )
                lv1: R.Tensor((1, 16, 16, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 2, 3, 1]
                )
                gv: R.Tuple(R.Tensor((1, 16, 16, 2), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, dtype=torch.float32),)
    # Disable decomposition to keep aten.affine_grid_generator as a single op
    verify_model(AffineGrid(), example_args, {}, expected, run_ep_decomposition=False)


def test_affine_grid_numerically():
    """Verify affine_grid numerical correctness: PyTorch vs TVM via our converter."""

    class AffineGrid(Module):
        def forward(self, theta):
            return torch.nn.functional.affine_grid(theta, [2, 3, 8, 12], align_corners=True)

    verify_model_numerically(
        AffineGrid(),
        (torch.randn(2, 2, 3),),
        rtol=1e-5,
        atol=1e-5,
        run_ep_decomposition=False,
    )


@pytest.mark.parametrize("as_module", [False, True])
def test_pool_divisor_override(as_module):
    op = (
        torch.nn.AvgPool2d(2, divisor_override=3)
        if as_module
        else lambda x: torch.nn.functional.avg_pool2d(x, 2, divisor_override=3)
    )
    model = UnaryModule(op)
    with pytest.raises(NotImplementedError, match="divisor_override"):
        from_exported_program(export(model, (torch.ones(1, 1, 4, 4),)))


def test_numeric_semantics():
    class Model(Module):
        def __init__(self):
            super().__init__()
            self.celu = torch.nn.CELU(alpha=2)

        def forward(self, x):
            return (
                self.celu(x),
                torch.nn.functional.celu(x, alpha=2),
                torch.nn.functional.selu(x),
                torch.std(x),
                torch.var(x),
                torch.var(x, correction=0),
                torch.div(x.to(torch.int32), 2.5, rounding_mode="floor"),
                torch.norm(x, p=0.5, dim=1, keepdim=True),
                x[-1, None, :],
                torch.select(x, 0, -1),
            )

    model = Model()
    args = (torch.tensor([[-1.0, 1.0, 3.0], [2.0, -2.0, 4.0]]),)
    verify_model_numerically(model, args, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    tvm.testing.main()
