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
import operator
import pytest
import torch
from torch import nn
from torch.nn import Module
from torch.export import export

import tvm
from tvm import relax
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend.torch import from_exported_program


def verify_model(torch_model, example_args, binding, expected, dynamic_shapes=None):
    exported_program = export(torch_model, args=example_args, dynamic_shapes=dynamic_shapes)
    mod = from_exported_program(exported_program)

    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm.ir.assert_structural_equal(mod, expected)


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
    (torch.relu_, R.nn.relu),
    (torch.round, R.round),
    (torch.rsqrt, R.rsqrt),
    (torch.selu, R.nn.selu),
    (torch.sigmoid, R.sigmoid),
    (torch.ops.aten.silu, R.nn.silu),
    (torch.ops.aten.silu_, R.nn.silu),
    (torch.sin, R.sin),
    (torch.sinh, R.sinh),
    (torch.sign, R.sign),
    (torch.sqrt, R.sqrt),
    (torch.square, R.square),
    (torch.tan, R.tan),
    (torch.tanh, R.tanh),
    (torch.trunc, R.trunc),
]


@pytest.mark.parametrize("pytorch_op, relax_op", operator_basic_unary)
def test_basic_unary_ops(pytorch_op, relax_op):
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class UnaryOp(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = relax_op(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(UnaryOp(), example_args, {}, expected)


operator_bool_unary = [
    (torch.isfinite, R.isfinite),
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
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    verify_model(UnaryOp(), example_args, {}, expected)


def test_extended_unary_ops():
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    # celu
    class Celu1(Module):
        def __init__(self):
            super().__init__()
            self.celu = torch.nn.CELU()

        def forward(self, input):
            return self.celu(input)

    class Celu2(Module):
        def forward(self, input):
            return torch.nn.functional.celu(input)

    # alpha * min(0, exp(x / alpha) - 1) + max(0, x)
    @tvm.script.ir_module
    class expected_celu:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input_1)
                lv_div: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(1.0, "float32")
                )
                lv_sub: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(
                    lv_div, R.const(1.0, "float32")
                )
                lv_min: R.Tensor((1, 3, 10, 10), dtype="float32") = R.minimum(
                    R.const(0.0, "float32"), lv_sub
                )
                lv_scaled: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    R.const(1.0, "float32"), lv_min
                )
                lv_relu_x: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input_1)
                lv_celu: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv_scaled, lv_relu_x)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv_celu,)
                R.output(gv)
            return gv

    verify_model(Celu1(), example_args, {}, expected_celu)
    verify_model(Celu2(), example_args, {}, expected_celu)

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

    # dropout

    class Dropout1(Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, input):
            return self.dropout(input)

    class Dropout2(Module):
        def forward(self, input):
            return torch.dropout(input, 0.5, train=True)

    class Dropout3(Module):
        def forward(self, input):
            return torch.ops.aten.dropout_(input, 0.5, train=True)

    @tvm.script.ir_module
    class expected_dropout:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (input_1,)
                R.output(gv)
            return gv

    verify_model(Dropout1(), example_args, {}, expected_dropout)
    verify_model(Dropout2(), example_args, {}, expected_dropout)
    verify_model(Dropout3(), example_args, {}, expected_dropout)

    # elu
    class Elu(Module):
        def __init__(self):
            super().__init__()
            self.elu = torch.nn.ELU()

        def forward(self, input):
            return self.elu(input)

    class Elu2(Module):
        def forward(self, input):
            return torch.nn.functional.elu(input)

    @tvm.script.ir_module
    class expected_elu:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv_exp: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input_1)
                lv_one_minus_exp: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(
                    R.const(1.0, dtype="float32"), lv_exp
                )
                lv_relu_one_minus_exp: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(
                    lv_one_minus_exp
                )
                lv_scaled: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    R.const(-1.0, dtype="float32"), lv_relu_one_minus_exp
                )
                lv_relu_x: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input_1)
                lv_elu: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv_scaled, lv_relu_x)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv_elu,)
                R.output(gv)
            return gv

    verify_model(Elu(), example_args, {}, expected_elu)
    verify_model(Elu2(), example_args, {}, expected_elu)

    # hardsigmoid
    class Hardsigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hs = torch.nn.Hardsigmoid()

        def forward(self, input):
            return self.hs(input)

    class Hardsigmoid2(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardsigmoid(input)

    @tvm.script.ir_module
    class expected_hardsigmoid:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(inp_0, R.const(3, "float32"))
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(lv, 0, 6)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv1, R.const(6, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    verify_model(Hardsigmoid(), example_args, {}, expected_hardsigmoid)
    verify_model(Hardsigmoid2(), example_args, {}, expected_hardsigmoid)

    # hardwish
    class Hardswish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hs = torch.nn.Hardswish()

        def forward(self, input):
            return self.hs(input)

    class Hardswish2(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardswish(input)

    class Hardswish3(torch.nn.Module):
        def forward(self, input):
            return torch.ops.aten.hardswish_(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(inp_0, R.const(3, "float32"))
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(lv, 0, 6)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv1, R.const(6, "float32")
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(inp_0, lv2)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    verify_model(Hardswish(), example_args, {}, expected1)
    verify_model(Hardswish2(), example_args, {}, expected1)
    verify_model(Hardswish3(), example_args, {}, expected1)

    # log2
    class Log2(Module):
        def forward(self, x):
            return torch.log2(x)

    @tvm.script.ir_module
    class Expected_log2:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(inp_0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(0.69314718246459961, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(Log2(), example_args, {}, Expected_log2)

    # log10
    class Log10(Module):
        def forward(self, x):
            return torch.log10(x)

    @tvm.script.ir_module
    class Expected_log10:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(inp_0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(2.302585092994046, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(Log10(), example_args, {}, Expected_log10)

    # log1p
    class Log1p(Module):
        def forward(self, x):
            return torch.log1p(x)

    @tvm.script.ir_module
    class Expected_log1p:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(
                    R.add(inp_0, R.const(1, "float32"))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Log1p(), example_args, {}, Expected_log1p)

    # reciprocal
    class Reciprocal(Module):
        def forward(self, input):
            return torch.reciprocal(input)

    @tvm.script.ir_module
    class expected_reciprocal:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    R.const(1.0, "float32"), input_1
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Reciprocal(), example_args, {}, expected_reciprocal)

    # Returns the maximum value of all elements in the input tensor.
    class MaxModel(Module):
        def forward(self, input):
            return torch.max(input)

    @tvm.script.ir_module
    class expected_max:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.max(input, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(MaxModel(), example_args, {}, expected_max)

    # Returns the minimum value of all elements in the input tensor.
    class MinModel(Module):
        def forward(self, input):
            return torch.min(input)

    @tvm.script.ir_module
    class expected_min:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.min(input, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(MinModel(), example_args, {}, expected_min)

    # relu6
    class ReLU6_1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, x):
            return self.relu6(x)

    class ReLU6_2(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu6(x)

    class ReLU6_3(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.relu6_(x)

    @tvm.script.ir_module
    class expected_relu6_1:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    x, R.prim_value(T.float64(0.0)), R.prim_value(T.float64(6.0))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_relu6_2:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu6(x)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(ReLU6_1(), example_args, {}, expected_relu6_1)
    verify_model(ReLU6_2(), example_args, {}, expected_relu6_2)
    verify_model(ReLU6_3(), example_args, {}, expected_relu6_2)


def test_hardtanh():
    class Hardtanh(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ht = torch.nn.Hardtanh()

        def forward(self, input):
            return self.ht(input)

    class Hardtanh2(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardtanh(input)

    class Hardtanh3(torch.nn.Module):
        def forward(self, input):
            return torch.ops.aten.hardtanh_(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    inp_0, R.prim_value(T.float64(-1.0)), R.prim_value(T.float64(1.0))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Hardtanh(), example_args, {}, expected1)
    verify_model(Hardtanh2(), example_args, {}, expected1)
    verify_model(Hardtanh3(), example_args, {}, expected1)


def test_softplus():
    import torch
    from torch.nn import Module

    torch.set_grad_enabled(False)

    class Softplus0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.softplus = torch.nn.Softplus(1.0, 20.0)

        def forward(self, x):
            return self.softplus(x)

    class Softplus1(Module):
        def forward(self, input):
            return torch.nn.functional.softplus(input, 1.0, 20.0)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.softplus(
                    x, beta=1.0, threshold=20.0
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softplus0(), example_args, {}, expected)
    verify_model(Softplus1(), example_args, {}, expected)


def test_leakyrelu():
    import torch
    from torch.nn import Module

    torch.set_grad_enabled(False)

    class LeakyReLU0(Module):
        def __init__(self):
            super().__init__()
            self.leakyrelu = torch.nn.LeakyReLU(0.02)

        def forward(self, input):
            return self.leakyrelu(input)

    class LeakyReLU1(Module):
        def forward(self, input):
            return torch.nn.functional.leaky_relu(input, 0.02)

    class LeakyReLU2(Module):
        def forward(self, input):
            return torch.ops.aten.leaky_relu_(input, 0.02)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.leakyrelu(input_1, 0.02)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(LeakyReLU0(), example_args, {}, expected)
    verify_model(LeakyReLU1(), example_args, {}, expected)
    verify_model(LeakyReLU2(), example_args, {}, expected)


def test_logaddexp():
    class LogAddExp(Module):
        def forward(self, input1, input2):
            return torch.logaddexp(input1, input2)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            input_2: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log_add_exp(input_1, input_2)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(LogAddExp(), example_args, {}, expected)


def test_logsoftmax():
    class LogSoftmax(Module):
        def __init__(self):
            super().__init__()
            self.lsm = torch.nn.LogSoftmax(dim=1)

        def forward(self, input):
            return self.lsm(input)

    class LogSoftmax2(Module):
        def forward(self, input):
            return torch.nn.functional.log_softmax(input, dim=1)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.log_softmax(input_1, axis=1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(LogSoftmax(), example_args, {}, expected1)
    verify_model(LogSoftmax2(), example_args, {}, expected1)


def test_prelu():
    class Prelu1(Module):
        def __init__(self, num_parameters=1, alpha=0.25):
            super().__init__()
            self.prelu = torch.nn.PReLU(num_parameters=num_parameters, init=alpha)

        def forward(self, x):
            return self.prelu(x)

    class Prelu2(torch.nn.Module):
        def __init__(self):
            super(Prelu2, self).__init__()
            self.alpha = torch.nn.Parameter(torch.tensor([0.25]))

        def forward(self, x):
            return torch.nn.functional.prelu(x, self.alpha)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.prelu(
                    x, R.const([0.25], dtype="float32"), axis=1
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Prelu1(), example_args, {}, expected)
    verify_model(Prelu2(), example_args, {}, expected)


def test_softmax():
    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.sm = torch.nn.Softmax(dim=1)

        def forward(self, input):
            return self.sm(input)

    class Softmax2(Module):
        def forward(self, input):
            return torch.nn.functional.softmax(input, dim=1)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.softmax(input_1, axis=1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softmax(), example_args, {}, expected1)
    verify_model(Softmax2(), example_args, {}, expected1)


def test_softshrink():
    class Softshrink(Module):
        def __init__(self):
            super().__init__()
            self.softshrink = torch.nn.Softshrink(lambd=0.5)

        def forward(self, input):
            return self.softshrink(input)

    class Softshrink2(Module):
        def forward(self, input):
            return torch.nn.functional.softshrink(input, lambd=0.5)

    @tvm.script.ir_module
    class expected_softshrink:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(
                    input, R.const(0.5, "float32")
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(
                    input, R.const(0.5, "float32")
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.astype(lv1, "float32")
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(lv, lv2)

                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(
                    input, R.const(0.5, "float32")
                )
                lv5: R.Tensor((), dtype="float32") = R.negative(R.const(0.5, "float32"))
                lv6: R.Tensor((1, 3, 10, 10), dtype="bool") = R.less(input, lv5)
                lv7: R.Tensor((1, 3, 10, 10), dtype="float32") = R.astype(lv6, "float32")
                lv8: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(lv4, lv7)

                lv9: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv3, lv8)

                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv9,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softshrink(), example_args, {}, expected_softshrink)
    verify_model(Softshrink2(), example_args, {}, expected_softshrink)


def test_tril_triu():
    example_args = (torch.randn(10, 10, dtype=torch.float32),)

    class Tril(Module):
        def forward(self, input):
            return torch.tril(input, 1)

    @tvm.script.ir_module
    class expected_tril:
        @R.function
        def main(
            input_1: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.tril(input_1, 1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Tril(), example_args, {}, expected_tril)

    class Triu(Module):
        def forward(self, input):
            return torch.triu(input, 1)

    @tvm.script.ir_module
    class expected_triu:
        @R.function
        def main(
            input_1: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.triu(input_1, 1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Triu(), example_args, {}, expected_triu)


operator_binary_1 = [
    (operator.add, R.add),
    (torch.ops.aten.add_, R.add),
    (torch.ops.aten.bitwise_or, R.bitwise_or),
    (torch.ops.aten.bitwise_or_, R.bitwise_or),
    (operator.sub, R.subtract),
    (operator.mul, R.multiply),
    (torch.ops.aten.mul_, R.multiply),
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
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = relax_op(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
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
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = relax_op(lhs, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Binary1(op), example_args1, {}, expected_binary1)
    verify_model(Binary2(op), example_args2, {}, expected_binary2)


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
        def main(
            x: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
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
                lv: R.Tensor((10, 10, 1), dtype="float32") = R.expand_dims(x, axis=[-1])
                lv1: R.Tensor((8,), dtype="float32") = R.reshape(test_elements, R.shape([8]))
                lv2: R.Tensor((10, 10, 8), dtype="bool") = R.equal(lv, lv1)
                lv3: R.Tensor((10, 10), dtype="bool") = R.sum(lv2, axis=[-1], keepdims=False)
                lv4: R.Tensor((10, 10), dtype="bool") = R.greater(lv3, R.const(0.0, "float32"))
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv4,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(8, dtype=torch.float32),
    )
    verify_model(IsInModel(), example_args, {}, expected)


def test_div_mode():
    # Case 1: Basic division (no rounding mode)
    class DivModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.div(a, b)

    @tvm.script.ir_module
    class expected_div:
        @R.function
        def main(
            a: R.Tensor((64, 64), dtype="float32"), b: R.Tensor((64,), dtype="float32")
        ) -> R.Tuple(R.Tensor((64, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.divide(a, b)
                gv: R.Tuple(R.Tensor((64, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(64, 64, dtype=torch.float32),
        torch.randn(64, dtype=torch.float32),
    )
    verify_model(DivModel(), example_args, {}, expected_div)

    # Case 2: Division with trunc rounding
    class DivTruncModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.div(a, b, rounding_mode="trunc")

    @tvm.script.ir_module
    class expected_div_trunc:
        @R.function
        def main(
            a: R.Tensor((64, 64), dtype="float32"), b: R.Tensor((64,), dtype="float32")
        ) -> R.Tuple(R.Tensor((64, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.divide(a, b)
                lv1: R.Tensor((64, 64), dtype="float32") = R.trunc(lv)
                gv: R.Tuple(R.Tensor((64, 64), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(DivTruncModel(), example_args, {}, expected_div_trunc)

    # Case 3: Division with floor rounding
    class DivFloorModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.div(a, b, rounding_mode="floor")

    @tvm.script.ir_module
    class expected_div_floor:
        @R.function
        def main(
            a: R.Tensor((64, 64), dtype="float32"), b: R.Tensor((64,), dtype="float32")
        ) -> R.Tuple(R.Tensor((64, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.floor_divide(a, b)
                gv: R.Tuple(R.Tensor((64, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(DivFloorModel(), example_args, {}, expected_div_floor)


def test_batchnorm2d():
    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, input):
            return self.bn(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
            w3: R.Tensor((3,), dtype="float32"),
            w4: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
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
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[0]
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = BatchNorm2d().eval()
    binding = {
        "w1": model.bn.weight.detach().numpy(),
        "w2": model.bn.bias.detach().numpy(),
        "w3": model.bn.running_mean.detach().numpy(),
        "w4": model.bn.running_var.detach().numpy(),
    }
    verify_model(model, example_args, binding, expected1)


def test_adaptive_avgpool1d():
    class AdaptiveAvgPool1d0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool1d(output_size=5)

        def forward(self, input):
            return self.pool(input)

    class AdaptiveAvgPool1d1(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool1d(input, output_size=5)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 5), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5), dtype="float32") = R.nn.adaptive_avg_pool1d(
                    input_1, output_size=[5], layout="NCW"
                )
                gv: R.Tuple(R.Tensor((1, 3, 5), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool1d0(), example_args, {}, expected1)
    verify_model(AdaptiveAvgPool1d1(), example_args, {}, expected1)


def test_adaptive_avgpool2d():
    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, input):
            return self.pool(input)

    class AdaptiveAvgPool2d1(Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool2d(input, [10, 10])

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    input_1, output_size=[10, 10], layout="NCHW", out_layout="NCHW"
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool2d0(), example_args, {}, expected1)
    verify_model(AdaptiveAvgPool2d1(), example_args, {}, expected1)


def test_adaptive_avgpool3d():
    class AdaptiveAvgPool3d0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool3d([4, 4, 4])

        def forward(self, input):
            return self.pool(input)

    class AdaptiveAvgPool3d1(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool3d(input, [4, 4, 4])

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 4, 4, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.nn.adaptive_avg_pool3d(
                    input_1, output_size=[4, 4, 4], layout="NCDHW", out_layout="NCDHW"
                )
                gv: R.Tuple(R.Tensor((1, 3, 4, 4, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool3d0(), example_args, {}, expected1)
    verify_model(AdaptiveAvgPool3d1(), example_args, {}, expected1)


def test_addmm():
    class Addmm1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3)

    class Addmm2(Module):
        def __init__(self):
            super().__init__()

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


def test_avg_pool1d():
    class AvgPool1d1(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool1d(kernel_size=1)

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10), dtype="float32") = R.nn.avg_pool1d(
                    input_1,
                    pool_size=[1],
                    strides=[1],
                    dilation=[1],
                    padding=[0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCW",
                    out_layout="NCW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class AvgPool1d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        def forward(self, input):
            return self.pool(input)

    class AvgPool1d3(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool1d(
                input, kernel_size=3, stride=2, padding=1, ceil_mode=True
            )

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool1d(
                    input_1,
                    pool_size=[3],
                    strides=[2],
                    dilation=[1],
                    padding=[1, 1],
                    ceil_mode=True,
                    count_include_pad=True,
                    layout="NCW",
                    out_layout="NCW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class AvgPool1d4(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool1d(input, kernel_size=2, stride=2, padding=0)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool1d(
                    input_1,
                    pool_size=[2],
                    strides=[2],
                    dilation=[1],
                    padding=[0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCW",
                    out_layout="NCW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)
    verify_model(AvgPool1d1(), example_args, {}, expected1)
    verify_model(AvgPool1d2(), example_args, {}, expected2)
    verify_model(AvgPool1d3(), example_args, {}, expected2)
    verify_model(AvgPool1d4(), example_args, {}, expected3)


def test_avg_pool2d():
    class AvgPool2d1(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.avg_pool2d(
                    input_1,
                    pool_size=[1, 1],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class AvgPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True)

        def forward(self, input):
            return self.pool(input)

    class AvgPool2d3(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(
                input, kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True
            )

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool2d(
                    input_1,
                    pool_size=[4, 4],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[2, 2, 2, 2],
                    ceil_mode=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class AvgPool2d4(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(input, kernel_size=[2, 1], divisor_override=2)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool2d(
                    input_1,
                    pool_size=[2, 1],
                    strides=[2, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(AvgPool2d1(), example_args, {}, expected1)
    verify_model(AvgPool2d2(), example_args, {}, expected2)
    verify_model(AvgPool2d3(), example_args, {}, expected2)
    verify_model(AvgPool2d4(), example_args, {}, expected3)


def test_avg_pool3d():
    class AvgPool3d1(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool3d(kernel_size=1)

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 8, 8, 8), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 8, 8, 8), dtype="float32") = R.nn.avg_pool3d(
                    input_1,
                    pool_size=[1, 1, 1],
                    strides=[1, 1, 1],
                    dilation=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 8, 8, 8), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class AvgPool3d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        def forward(self, input):
            return self.pool(input)

    class AvgPool3d3(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool3d(
                input, kernel_size=3, stride=2, padding=1, ceil_mode=True
            )

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool3d(
                    input_1,
                    pool_size=[3, 3, 3],
                    strides=[2, 2, 2],
                    dilation=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    ceil_mode=True,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class AvgPool3d4(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool3d(input, kernel_size=[2, 1, 2], stride=[2, 1, 2])

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool3d(
                    input_1,
                    pool_size=[2, 1, 2],
                    strides=[2, 1, 2],
                    dilation=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32),)
    verify_model(AvgPool3d1(), example_args, {}, expected1)
    verify_model(AvgPool3d2(), example_args, {}, expected2)
    verify_model(AvgPool3d3(), example_args, {}, expected2)
    verify_model(AvgPool3d4(), example_args, {}, expected3)


def test_baddbmm():
    class BAddBMM1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(inp_1, inp_2)
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.add(lv, inp_0)
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    class BAddBMM2(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(inp_1, inp_2)
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.multiply(
                    lv, R.const(2, "float32")
                )
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    class BAddBMM3(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=3)

    @tvm.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(inp_1, inp_2)
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.multiply(
                    lv, R.const(2, "float32")
                )
                lv2: R.Tensor((4, 128, 512), dtype="float32") = R.multiply(
                    inp_0, R.const(3, "float32")
                )
                lv3: R.Tensor((4, 128, 512), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 128, 512, dtype=torch.float32),
        torch.randn(4, 128, 256, dtype=torch.float32),
        torch.randn(4, 256, 512, dtype=torch.float32),
    )
    verify_model(
        BAddBMM1(),
        example_args,
        {},
        Expected1,
    )

    verify_model(
        BAddBMM2(),
        example_args,
        {},
        Expected2,
    )

    verify_model(
        BAddBMM3(),
        example_args,
        {},
        Expected3,
    )


def test_bmm():
    class BMM(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.bmm(x, y)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input_1: R.Tensor((4, 128, 256), dtype="float32"),
            input_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(
                    input_1, input_2, out_dtype="float32"
                )
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 128, 256, dtype=torch.float32),
        torch.randn(4, 256, 512, dtype=torch.float32),
    )
    verify_model(
        BMM(),
        example_args,
        {},
        Expected,
    )


def test_conv_transpose1d():
    class ConvTranspose1d1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(6, 6, 3, bias=True)

        def forward(self, input):
            return self.conv(input)

    class ConvTranspose1d1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 6, 3])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv_transpose1d(input, self.weight, self.bias)

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

    model = ConvTranspose1d1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
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

    class ConvTranspose2d1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[3, 3, 7, 7])
            self.bias = torch.randn(size=[3])

        def forward(self, input):
            return torch.nn.functional.conv_transpose2d(input, self.weight, self.bias)

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

    model = ConvTranspose2d1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
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

    class Conv1D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv1d(input, self.weight, self.bias)

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

    model = Conv1D1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
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

    class Conv2D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv2d(input, self.weight, self.bias)

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

    model = Conv2D1Func()
    binding = {"w1": model.weight.numpy(), "w2": model.bias.numpy()}
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

    class Conv3D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7, 7, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv3d(input, self.weight, self.bias)

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

    model = Conv3D1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv3D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_pad():
    class PadModel(torch.nn.Module):
        def __init__(self, pad, mode="constant", value=0.0):
            super().__init__()
            self.pad = pad
            self.mode = mode
            self.value = value

        def forward(self, x):
            if self.mode == "constant":
                return torch.nn.functional.pad(x, self.pad, mode=self.mode, value=self.value)
            else:
                return torch.nn.functional.pad(x, self.pad, mode=self.mode)

    @tvm.script.ir_module
    class expected_constant:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="constant",
                    pad_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_reflect:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="reflect",
                    pad_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_replicate:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="replicate",
                    pad_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_circular:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="circular",
                    pad_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(PadModel(pad=[1, 1, 2, 2]), example_args, {}, expected_constant)
    verify_model(PadModel(pad=[1, 1, 2, 2], mode="reflect"), example_args, {}, expected_reflect)
    verify_model(PadModel(pad=[1, 1, 2, 2], mode="replicate"), example_args, {}, expected_replicate)
    verify_model(PadModel(pad=[1, 1, 2, 2], mode="circular"), example_args, {}, expected_circular)


def test_pixel_shuffle():
    class PixelShuffle1(torch.nn.Module):
        def __init__(self, upscale_factor=2):
            super().__init__()
            self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)

        def forward(self, x):
            return self.pixel_shuffle(x)

    class PixelShuffle2(torch.nn.Module):
        def __init__(self, upscale_factor=2):
            super().__init__()
            self.upscale_factor = upscale_factor

        def forward(self, x):
            return torch.nn.functional.pixel_shuffle(x, self.upscale_factor)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            x: R.Tensor((1, 8, 10, 15), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 20, 30), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 2, 20, 30), dtype="float32") = R.nn.pixel_shuffle(
                    x, upscale_factor=2
                )
                gv: R.Tuple(R.Tensor((1, 2, 20, 30), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 8, 10, 15, dtype=torch.float32),)
    verify_model(PixelShuffle1(upscale_factor=2), example_args, {}, expected)
    verify_model(PixelShuffle2(upscale_factor=2), example_args, {}, expected)


def test_einsum():
    class Einsum1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.einsum("ii", x)

    class Einsum2(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.einsum("i,j->ij", x, y)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((4, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
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
    verify_model(Einsum1(), example_args, {}, Expected1)

    example_args = (torch.randn(5, dtype=torch.float32), torch.randn(4, dtype=torch.float32))
    verify_model(Einsum2(), example_args, {}, Expected2)


def test_outer():
    class Outer(torch.nn.Module):
        def forward(self, x, y):
            return torch.outer(x, y)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            a: R.Tensor((3,), dtype="float32"), b: R.Tensor((4,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.outer(a, b)
                gv: R.Tuple(R.Tensor((3, 4), dtype="float32")) = (lv,)
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

    example_args = (torch.randint(low=-int(1e5), high=int(1e5), size=(4,), dtype=torch.int64),)

    model = Embedding()
    binding = {"w1": model.embedding.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected1)


def test_groupnorm():
    import torch
    from torch.nn import Module

    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

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

    class Dense1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[7, 10])
            self.bias = torch.randn(size=[7])

        def forward(self, input):
            return torch.nn.functional.linear(input, self.weight, self.bias)

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
                lv: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=None)
                lv1: R.Tensor((1, 3, 10, 7), dtype="float32") = R.matmul(
                    input_1, lv, out_dtype="float32"
                )
                lv2: R.Tensor((1, 3, 10, 7), dtype="float32") = R.add(lv1, w2)
                gv: R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")) = (lv2,)
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
                lv: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=None)
                lv1: R.Tensor((1, 3, 10, 7), dtype="float32") = R.matmul(
                    input_1, lv, out_dtype="float32"
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = Dense1()
    binding = {"w1": model.linear.weight.detach().numpy(), "w2": model.linear.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Dense1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Dense2()
    binding = {"w1": model.linear.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_maxpool1d():
    class MaxPool1d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool1d(kernel_size=2)

        def forward(self, input):
            return self.pool(input)

    class MaxPool1d_functional(Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.max_pool1d(input, kernel_size=2)

    class MaxPool1d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=2)

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 8), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 4), dtype="float32")):
            with R.dataflow():
                lv = R.nn.max_pool1d(
                    input_1,
                    pool_size=[2],
                    strides=[2],
                    dilation=[1],
                    padding=[0, 0],
                    layout="NCW",
                    out_layout="NCW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 8), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 4), dtype="float32")):
            with R.dataflow():
                lv = R.nn.max_pool1d(
                    input_1,
                    pool_size=[2],
                    strides=[2],
                    dilation=[1],
                    padding=[0, 0],
                    layout="NCW",
                    out_layout="NCW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 4), dtype="float32")):
            with R.dataflow():
                lv = R.nn.max_pool1d(
                    input_1,
                    pool_size=[3],
                    strides=[2],
                    dilation=[1],
                    padding=[0, 0],
                    layout="NCW",
                    out_layout="NCW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    # Example inputs
    example_args1 = (torch.randn(1, 3, 8, dtype=torch.float32),)
    example_args2 = (torch.randn(1, 3, 8, dtype=torch.float32),)
    example_args3 = (torch.randn(1, 3, 10, dtype=torch.float32),)

    # Verify the models
    verify_model(MaxPool1d(), example_args1, {}, expected1)
    verify_model(MaxPool1d_functional(), example_args2, {}, expected2)
    verify_model(MaxPool1d2(), example_args3, {}, expected3)


def test_maxpool2d():
    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d_functional(Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.max_pool2d(input, kernel_size=[1, 1])

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[1, 1],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    dilation=[2, 3],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 4, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 6, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 6, 6), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[4, 4],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[2, 2, 2, 2],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 6, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(MaxPool2d(), example_args, {}, expected1)
    verify_model(MaxPool2d_functional(), example_args, {}, expected1)
    verify_model(MaxPool2d2(), example_args, {}, expected2)
    verify_model(MaxPool2d3(), example_args, {}, expected3)


def test_maxpool3d():
    class MaxPool3d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=[1, 1, 1])

        def forward(self, input):
            return self.pool(input)

    class MaxPool3d_functional(Module):
        def __init__(self):
            super().__init__()

        def forward(self, input):
            return torch.nn.functional.max_pool3d(input, kernel_size=[1, 1, 1])

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 4, 4, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 4, 4, 4), dtype="float32")):
            with R.dataflow():
                lv = R.nn.max_pool3d(
                    input_1,
                    pool_size=[1, 1, 1],
                    strides=[1, 1, 1],
                    dilation=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class MaxPool3d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=[2, 2, 2], dilation=[2, 2, 2])

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 3, 3, 3), dtype="float32")):
            with R.dataflow():
                lv = R.nn.max_pool3d(
                    input_1,
                    pool_size=[2, 2, 2],
                    strides=[2, 2, 2],
                    dilation=[2, 2, 2],
                    padding=[0, 0, 0, 0, 0, 0],
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class MaxPool3d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=[3, 3, 3], padding=1, stride=2)

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 5, 5, 5), dtype="float32")):
            with R.dataflow():
                lv = R.nn.max_pool3d(
                    input_1,
                    pool_size=[3, 3, 3],
                    strides=[2, 2, 2],
                    dilation=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    # Example input tensors
    example_args1 = (torch.randn(1, 3, 4, 4, 4, dtype=torch.float32),)
    example_args2 = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32),)
    example_args3 = (torch.randn(1, 3, 10, 10, 10, dtype=torch.float32),)

    # Verify the models with expected IR modules
    verify_model(MaxPool3d(), example_args1, {}, expected1)
    verify_model(MaxPool3d_functional(), example_args1, {}, expected1)
    verify_model(MaxPool3d2(), example_args2, {}, expected2)
    verify_model(MaxPool3d3(), example_args3, {}, expected3)


def test_scaled_dot_product_attention():
    class Attention1(Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_1: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_2: R.Tensor((32, 8, 128, 64), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    inp_0, axes=[0, 2, 1, 3]
                )
                lv1: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    inp_1, axes=[0, 2, 1, 3]
                )
                lv2: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    inp_2, axes=[0, 2, 1, 3]
                )
                lv3: R.Tensor((32, 128, 8, 64), dtype="float32") = R.nn.attention(
                    lv, lv1, lv2, scale=None
                )
                lv4: R.Tensor((32, 8, 128, 64), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 2, 1, 3]
                )
                gv: R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    class Attention2(Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_1: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_2: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_3: R.Tensor((32, 8, 128, 128), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    inp_0, axes=[0, 2, 1, 3]
                )
                lv1: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    inp_1, axes=[0, 2, 1, 3]
                )
                lv2: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    inp_2, axes=[0, 2, 1, 3]
                )
                lv3: R.Tensor((32, 128, 8, 64), dtype="float32") = R.nn.attention(
                    lv, lv1, lv2, inp_3, scale=None
                )
                lv4: R.Tensor((32, 8, 128, 64), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 2, 1, 3]
                )
                gv: R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    verify_model(
        Attention1(),
        (
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
        ),
        {},
        Expected1,
    )

    verify_model(
        Attention2(),
        (
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 128, dtype=torch.float32),
        ),
        {},
        Expected2,
    )


def test_unbind():
    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((3, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=3, axis=0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv1, axis=[0])
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[1]
                lv4: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv3, axis=[0])
                lv5: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[2]
                lv6: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv5, axis=[0])
                lv7: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv2, lv4, lv6)
                lv8: R.Tensor((3, 10, 10), dtype="float32") = lv7[0]
                lv9: R.Tensor((3, 10, 10), dtype="float32") = lv7[1]
                lv10: R.Tensor((3, 10, 10), dtype="float32") = lv7[2]
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv8, lv9, lv10)
                R.output(gv)
            return gv

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((3, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((3, 1, 10, 10), dtype="float32"),
                    R.Tensor((3, 1, 10, 10), dtype="float32"),
                    R.Tensor((3, 1, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=3, axis=1)
                lv1: R.Tensor((3, 1, 10, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv3: R.Tensor((3, 1, 10, 10), dtype="float32") = lv[1]
                lv4: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv3, axis=[1])
                lv5: R.Tensor((3, 1, 10, 10), dtype="float32") = lv[2]
                lv6: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv5, axis=[1])
                lv7: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv2, lv4, lv6)
                lv8: R.Tensor((3, 10, 10), dtype="float32") = lv7[0]
                lv9: R.Tensor((3, 10, 10), dtype="float32") = lv7[1]
                lv10: R.Tensor((3, 10, 10), dtype="float32") = lv7[2]
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv8, lv9, lv10)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, 3, 10, 10, dtype=torch.float32),)
    verify_model(Unbind1(), example_args, {}, expected1)
    verify_model(Unbind2(), example_args, {}, expected2)


def test_interpolate():
    class InterpolateBilinear(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (224, 224), mode="bilinear")

    @tvm.script.ir_module
    class expected_bilinear:
        @R.function
        def main(
            input: R.Tensor((1, 3, 112, 112), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 224, 224), dtype="float32") = R.image.resize2d(
                    input,
                    R.shape([224, 224]),
                    roi=[T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0)],
                    layout="NCHW",
                    method="linear",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                    out_dtype="void",
                )
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class InterpolateNearest(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (224, 224), mode="nearest")

    @tvm.script.ir_module
    class expected_nearest:
        @R.function
        def main(
            input: R.Tensor((1, 3, 112, 112), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 224, 224), dtype="float32") = R.image.resize2d(
                    input,
                    R.shape([224, 224]),
                    roi=[T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0)],
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                    out_dtype="void",
                )
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class InterpolateBicubic(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (224, 224), mode="bicubic")

    @tvm.script.ir_module
    class expected_bicubic:
        @R.function
        def main(
            input: R.Tensor((1, 3, 112, 112), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 224, 224), dtype="float32") = R.image.resize2d(
                    input,
                    R.shape([224, 224]),
                    roi=[T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0)],
                    layout="NCHW",
                    method="cubic",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                    out_dtype="void",
                )
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 112, 112, dtype=torch.float32),)
    verify_model(InterpolateBilinear(), example_args, {}, expected_bilinear)
    verify_model(InterpolateNearest(), example_args, {}, expected_nearest)
    verify_model(InterpolateBicubic(), example_args, {}, expected_bicubic)


def test_mean():
    class Mean(Module):
        def forward(self, input):
            return input.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, input: torch.Tensor):
            return input.mean(-1, keepdim=True)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tuple(R.Tensor((256,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((256,), dtype="float32") = R.mean(inp_0, axis=[-1], keepdims=False)
                gv: R.Tuple(R.Tensor((256,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tuple(R.Tensor((256, 1), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((256, 1), dtype="float32") = R.mean(inp_0, axis=[-1], keepdims=True)
                gv: R.Tuple(R.Tensor((256, 1), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    verify_model(Mean(), example_args, {}, Expected1)
    verify_model(MeanKeepDim(), example_args, {}, Expected2)


def test_sum():
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.sum(inp_0, axis=[2, 1], keepdims=False)
                gv: R.Tuple(R.Tensor((1, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Sum(), example_args, {}, expected1)


def test_argmax_argmin():
    example_args = (torch.randn(256, 256, dtype=torch.float32),)

    class Argmax1(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmax(input, dim=-1)

    class Argmax2(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmax(input, dim=-1, keepdim=True)

    @tvm.script.ir_module
    class expected_argmax1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tuple(R.Tensor((256,), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((256,), dtype="int64") = R.argmax(inp_0, axis=-1, keepdims=False)
                gv: R.Tuple(R.Tensor((256,), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_argmax2:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tuple(R.Tensor((256, 1), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((256, 1), dtype="int64") = R.argmax(inp_0, axis=-1, keepdims=True)
                gv: R.Tuple(R.Tensor((256, 1), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Argmax1(), example_args, {}, expected_argmax1)
    verify_model(Argmax2(), example_args, {}, expected_argmax2)

    class Argmin1(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmin(input)

    class Argmin2(Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, input):
            return torch.argmin(input, keepdim=True)

    @tvm.script.ir_module
    class expected_argmin1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tuple(R.Tensor((), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((), dtype="int64") = R.argmin(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_argmin2:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 1), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((1, 1), dtype="int64") = R.argmin(inp_0, axis=None, keepdims=True)
                gv: R.Tuple(R.Tensor((1, 1), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Argmin1(), example_args, {}, expected_argmin1)
    verify_model(Argmin2(), example_args, {}, expected_argmin2)


def test_cat_concat():
    class Cat0(Module):
        def forward(self, x, y):
            return torch.cat((x, y))

    class Cat1(Module):
        def forward(self, x, y):
            return torch.cat((x, y), dim=1)

    class Cat2(Module):
        def forward(self, x, y):
            return torch.cat((x, y), 1)

    class Cat3(Module):
        def forward(self, x, y):
            return torch.concat((x, y), dim=0)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 3), dtype="float32") = R.concat((inp_0, inp_1), axis=0)
                gv: R.Tuple(R.Tensor((4, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

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
    verify_model(Cat0(), example_args, {}, Expected1)
    verify_model(Cat1(), example_args, {}, Expected2)
    verify_model(Cat2(), example_args, {}, Expected2)
    verify_model(Cat3(), example_args, {}, Expected1)


def test_cumsum():
    class Cumsum(Module):
        def forward(self, input):
            return torch.cumsum(input, dim=1, dtype=torch.int32)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 3, 4), dtype="int32")):
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
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((4, 2, 3, 4), dtype="float32")):
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
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 100), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 100), dtype="float32") = R.reshape(input_1, (1, 3, 100))
                gv: R.Tuple(R.Tensor((1, 3, 100), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Flatten(), example_args, {}, expected1)


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
                lv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = R.meshgrid((input1, input2), indexing="ij")
                lv1: R.Tensor((3, 3), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 3), dtype="float32") = lv[1]
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input1: R.Tensor((3,), dtype="float32"), input2: R.Tensor((3,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = R.meshgrid((input1, input2), indexing="xy")
                lv1: R.Tensor((3, 3), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 3), dtype="float32") = lv[1]
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv1, lv2)
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

    class Permute2(Module):
        def forward(self, x):
            return torch.permute(x, (0, 3, 2, 1))

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Permute1(), example_args, {}, expected1)
    verify_model(Permute2(), example_args, {}, expected1)


def test_repeat():
    class Tile1(Module):
        def forward(self, x: torch.Tensor):
            return x.repeat(2)

    class Tile2(Module):
        def forward(self, x: torch.Tensor):
            return x.repeat(4, 2)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor((6,), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((6,), dtype="float32") = R.tile(x, 2)
                gv: R.Tuple(R.Tensor((6,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            x: R.Tensor((1, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((4, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 6), dtype="float32") = R.tile(x, [4, 2])
                gv: R.Tuple(R.Tensor((4, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, dtype=torch.float32),)
    verify_model(Tile1(), example_args, {}, expected1)

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile2(), example_args, {}, expected2)

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile2(), example_args, {}, expected2)


def test_reshape():
    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((2, 12), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tuple(R.Tensor((2, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Reshape(), example_args, {}, expected1)


def test_reshape_as():
    class ReshapeAs(Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor):
            return x.reshape_as(y)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32"),
            y: R.Tensor((2, 12), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 12), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tuple(R.Tensor((2, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 2, 3, 4, dtype=torch.float32),
        torch.randn(2, 12, dtype=torch.float32),
    )
    verify_model(ReshapeAs(), example_args, {}, expected1)


def test_roll():
    class Roll1(Module):
        def forward(self, x):
            return torch.roll(x, 1)

    class Roll2(Module):
        def forward(self, x):
            return torch.roll(x, -1, 0)

    class Roll3(Module):
        def forward(self, x):
            return torch.roll(x, shifts=(2, 1), dims=(0, 1))

    # Test case 1: torch.roll(x, 1)
    @I.ir_module
    class Expected1:
        @R.function
        def main(x: R.Tensor((4, 2), dtype="int64")) -> R.Tuple(R.Tensor((4, 2), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((8,), dtype="int64") = R.reshape(x, R.shape([8]))
                lv1: R.Tensor((7,), dtype="int64") = R.strided_slice(
                    lv,
                    axes=[0],
                    begin=[R.prim_value(0)],
                    end=[R.prim_value(7)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv2: R.Tensor((1,), dtype="int64") = R.strided_slice(
                    lv,
                    axes=[0],
                    begin=[R.prim_value(7)],
                    end=[R.prim_value(8)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv3: R.Tensor((8,), dtype="int64") = R.concat((lv2, lv1), axis=0)
                lv4: R.Tensor((4, 2), dtype="int64") = R.reshape(lv3, R.shape([4, 2]))
                gv: R.Tuple(R.Tensor((4, 2), dtype="int64")) = (lv4,)
                R.output(gv)
            return gv

    # Test case 2: torch.roll(x, -1, 0)
    @I.ir_module
    class Expected2:
        @R.function
        def main(x: R.Tensor((4, 2), dtype="int64")) -> R.Tuple(R.Tensor((4, 2), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((1, 2), dtype="int64") = R.strided_slice(
                    x,
                    axes=[0],
                    begin=[R.prim_value(0)],
                    end=[R.prim_value(1)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv1: R.Tensor((3, 2), dtype="int64") = R.strided_slice(
                    x,
                    axes=[0],
                    begin=[R.prim_value(1)],
                    end=[R.prim_value(4)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv2: R.Tensor((4, 2), dtype="int64") = R.concat((lv1, lv), axis=0)
                gv: R.Tuple(R.Tensor((4, 2), dtype="int64")) = (lv2,)
                R.output(gv)
            return gv

    # Test case 3: torch.roll(x, shifts=(2,1), dims=(0,1))
    @I.ir_module
    class Expected3:
        @R.function
        def main(x: R.Tensor((4, 2), dtype="int64")) -> R.Tuple(R.Tensor((4, 2), dtype="int64")):
            with R.dataflow():
                # First roll along dim=0 with shift=2
                lv: R.Tensor((2, 2), dtype="int64") = R.strided_slice(
                    x,
                    axes=[0],
                    begin=[R.prim_value(0)],
                    end=[R.prim_value(2)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv1: R.Tensor((2, 2), dtype="int64") = R.strided_slice(
                    x,
                    axes=[0],
                    begin=[R.prim_value(2)],
                    end=[R.prim_value(4)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv2: R.Tensor((4, 2), dtype="int64") = R.concat((lv1, lv), axis=0)

                # Second roll along dim=1 with shift=1
                lv3: R.Tensor((4, 1), dtype="int64") = R.strided_slice(
                    lv2,
                    axes=[1],
                    begin=[R.prim_value(0)],
                    end=[R.prim_value(1)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv4: R.Tensor((4, 1), dtype="int64") = R.strided_slice(
                    lv2,
                    axes=[1],
                    begin=[R.prim_value(1)],
                    end=[R.prim_value(2)],
                    strides=[R.prim_value(1)],
                    assume_inbound=False,
                )
                lv5: R.Tensor((4, 2), dtype="int64") = R.concat((lv4, lv3), axis=1)
                gv: R.Tuple(R.Tensor((4, 2), dtype="int64")) = (lv5,)
                R.output(gv)
            return gv

    # Test inputs
    example_input = torch.randint(0, 10, (4, 2), dtype=torch.int64)

    # Run verification for each case
    verify_model(Roll1(), (example_input,), {}, Expected1)
    verify_model(Roll2(), (example_input,), {}, Expected2)
    verify_model(Roll3(), (example_input,), {}, Expected3)


def test_select_slice():
    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 10, 3), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((3, 10, 10), dtype="float32") = R.take(x, R.const(0, "int64"), axis=0)
                lv1: R.Tensor((1, 10, 10), dtype="float32") = R.strided_slice(
                    lv,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(9223372036854775807),),
                    (R.prim_value(2),),
                    assume_inbound=False,
                )
                lv2: R.Tensor((1, 10, 10), dtype="float32") = R.strided_slice(
                    lv1,
                    (R.prim_value(1),),
                    (R.prim_value(0),),
                    (R.prim_value(9223372036854775807),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv3: R.Tensor((1, 10, 3), dtype="float32") = R.strided_slice(
                    lv2,
                    (R.prim_value(2),),
                    (R.prim_value(0),),
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                gv: R.Tuple(R.Tensor((1, 10, 3), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    @I.ir_module
    class expected2:
        @R.function
        def main(
            x: R.Tensor((8, 16), dtype="float32")
        ) -> R.Tuple(R.Tensor((8, 1, 1, 16, 1), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((8, 16), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(0),),
                    (R.prim_value(0),),
                    (R.prim_value(9223372036854775807),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv1: R.Tensor((8, 1, 16), dtype="float32") = R.expand_dims(lv, axis=[1])
                lv2: R.Tensor((8, 1, 1, 16), dtype="float32") = R.expand_dims(lv1, axis=[2])
                lv3: R.Tensor((8, 1, 1, 16), dtype="float32") = R.strided_slice(
                    lv2,
                    (R.prim_value(3),),
                    (R.prim_value(0),),
                    (R.prim_value(9223372036854775807),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv4: R.Tensor((8, 1, 1, 16, 1), dtype="float32") = R.expand_dims(lv3, axis=[4])
                gv: R.Tuple(R.Tensor((8, 1, 1, 16, 1), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Slice1(), example_args, {}, expected1)

    example_args = (torch.randn(8, 16, dtype=torch.float32),)
    verify_model(Slice2(), example_args, {}, expected2)


def test_slice_scatter():
    class SliceScatter1(Module):
        def forward(self, input, src):
            return torch.slice_scatter(input, src, dim=1, start=1, end=7, step=2)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            a: R.Tensor((8, 8, 10, 10), dtype="float32"),
            b: R.Tensor((8, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((8, 8, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((8, 8, 10, 10), dtype="float32") = R.slice_scatter(
                    a, b, R.prim_value(1), R.prim_value(7), R.prim_value(2), axis=1
                )
                gv: R.Tuple(R.Tensor((8, 8, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class SliceScatter2(Module):
        def forward(self, input, src):
            return torch.slice_scatter(input, src, dim=0, start=0, end=6, step=1)

    @I.ir_module
    class expected2:
        @R.function
        def main(
            a: R.Tensor((8, 16), dtype="float32"), b: R.Tensor((6, 16), dtype="float32")
        ) -> R.Tuple(R.Tensor((8, 16), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((8, 16), dtype="float32") = R.slice_scatter(
                    a, b, R.prim_value(0), R.prim_value(6), R.prim_value(1), axis=0
                )
                gv: R.Tuple(R.Tensor((8, 16), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(8, 8, 10, 10, dtype=torch.float32), torch.randn(8, 3, 10, 10))
    verify_model(SliceScatter1(), example_args, {}, expected1)

    example_args = (torch.randn(8, 16, dtype=torch.float32), torch.randn(6, 16))
    verify_model(SliceScatter2(), example_args, {}, expected2)


def test_split():
    class Chunk(Module):
        def forward(self, input):
            return torch.chunk(input, 3, dim=1)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((1, 1, 10, 10), dtype="float32"),
            R.Tensor((1, 1, 10, 10), dtype="float32"),
            R.Tensor((1, 1, 10, 10), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=3, axis=1)
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

    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((3, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=3, axis=0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv1, axis=[0])
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[1]
                lv4: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv3, axis=[0])
                lv5: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[2]
                lv6: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv5, axis=[0])
                lv7: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv2, lv4, lv6)
                lv8: R.Tensor((3, 10, 10), dtype="float32") = lv7[0]
                lv9: R.Tensor((3, 10, 10), dtype="float32") = lv7[1]
                lv10: R.Tensor((3, 10, 10), dtype="float32") = lv7[2]
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv8, lv9, lv10)
                R.output(gv)
            return gv

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((3, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((3, 1, 10, 10), dtype="float32"),
                    R.Tensor((3, 1, 10, 10), dtype="float32"),
                    R.Tensor((3, 1, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=3, axis=1)
                lv1: R.Tensor((3, 1, 10, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv3: R.Tensor((3, 1, 10, 10), dtype="float32") = lv[1]
                lv4: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv3, axis=[1])
                lv5: R.Tensor((3, 1, 10, 10), dtype="float32") = lv[2]
                lv6: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv5, axis=[1])
                lv7: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv2, lv4, lv6)
                lv8: R.Tensor((3, 10, 10), dtype="float32") = lv7[0]
                lv9: R.Tensor((3, 10, 10), dtype="float32") = lv7[1]
                lv10: R.Tensor((3, 10, 10), dtype="float32") = lv7[2]
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv8, lv9, lv10)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Chunk(), example_args, {}, Expected)

    example_args = (torch.randn(3, 3, 10, 10, dtype=torch.float32),)
    verify_model(Unbind1(), example_args, {}, expected1)
    verify_model(Unbind2(), example_args, {}, expected2)


def test_squeeze():
    class Squeeze1(Module):
        def forward(self, input):
            return input.squeeze(1)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 4, 1), dtype="float32")):
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
        def main(
            inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.squeeze(inp_0, axis=None)
                gv: R.Tuple(R.Tensor((3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, 1, 4, 1, dtype=torch.float32),)

    verify_model(Squeeze1(), example_args, {}, Expected1)
    verify_model(Squeeze2(), example_args, {}, Expected2)


def test_stack():
    class Stack0(Module):
        def forward(self, x, y):
            return torch.stack((x, y))  # default dim=0

    class Stack1(Module):
        def forward(self, x, y):
            return torch.stack((x, y), dim=1)

    class Stack2(Module):
        def forward(self, x, y):
            return torch.stack((x, y), 1)  # positional dim

    class Stack3(Module):
        def forward(self, x, y):
            return torch.stack((x, y), dim=-1)  # negative dim

    @I.ir_module
    class Expected0:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 2, 3), dtype="float32") = R.stack((inp_0, inp_1), axis=0)
                gv: R.Tuple(R.Tensor((2, 2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 2, 3), dtype="float32") = R.stack((inp_0, inp_1), axis=1)
                gv: R.Tuple(R.Tensor((2, 2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 2), dtype="float32") = R.stack((inp_0, inp_1), axis=-1)
                gv: R.Tuple(R.Tensor((2, 3, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32), torch.randn(2, 3, dtype=torch.float32))

    verify_model(Stack0(), example_args, {}, Expected0)
    verify_model(Stack1(), example_args, {}, Expected1)
    verify_model(Stack2(), example_args, {}, Expected1)
    verify_model(Stack3(), example_args, {}, Expected3)


def test_tile():
    class Tile1(Module):
        def forward(self, x):
            return x.tile((2,))

    class Tile2(Module):
        def forward(self, x):
            return x.tile(4, 2)

    class Tile3(Module):
        def forward(self, x):
            return torch.tile(x, (4, 2))

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 6), dtype="float32") = R.tile(x, [2])
                gv: R.Tuple(R.Tensor((1, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            x: R.Tensor((1, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((4, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 6), dtype="float32") = R.tile(x, [4, 2])
                gv: R.Tuple(R.Tensor((4, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile1(), example_args, {}, expected1)
    verify_model(Tile2(), example_args, {}, expected2)
    verify_model(Tile3(), example_args, {}, expected2)


def test_transpose():
    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Transpose(), example_args, {}, expected1)


def test_unsqueeze():
    class Unsqueeze1(Module):
        def forward(self, input):
            return input.unsqueeze(1)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 1, 3, 10, 10), dtype="float32") = R.expand_dims(input_1, 1)
                gv: R.Tuple(R.Tensor((1, 1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Unsqueeze2(Module):
        def forward(self, input):
            return input.unsqueeze(-1)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10, 1), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10, 1), dtype="float32") = R.expand_dims(input_1, -1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10, 1), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Unsqueeze1(), example_args, {}, expected1)
    verify_model(Unsqueeze2(), example_args, {}, expected2)


def test_view():
    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((2, 12), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tuple(R.Tensor((2, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(View(), example_args, {}, expected1)


def test_arange():
    class Arange(Module):
        def forward(self, input):
            return torch.arange(0, 20, dtype=torch.int32)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((20,), dtype="int32")):
            with R.dataflow():
                lv: R.Tensor((20,), dtype="int32") = R.arange(0, 20, 1, dtype="int32")
                gv: R.Tuple(R.Tensor((20,), dtype="int32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Arange(), example_args, {}, Expected)


def test_contiguous():
    class Contiguous(Module):
        def forward(self, input):
            return input.contiguous()

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (input,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Contiguous(), example_args, {}, Expected)


def test_clone():
    class Clone(Module):
        def forward(self, input):
            return torch.clone(input)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (input,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Clone(), example_args, {}, Expected)


def test_empty():
    class Empty(Module):
        def forward(self, input):
            return torch.empty((10, 10), dtype=torch.float32)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.zeros(
                    R.shape([10, 10]), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Empty(), example_args, {}, Expected)


def test_fill():
    class Fill(Module):
        def forward(self, input: torch.Tensor):
            return torch.fill(input, 1.5)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.full(
                    R.shape([10, 10]), R.const(1.5, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Fill(), example_args, {}, Expected)


def test_fill_inplace():
    class FillInplace(Module):
        def forward(self, input: torch.Tensor):
            input.fill_(42.0)
            return input

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.full(
                    R.shape([2, 3]), R.const(42.0, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32),)
    verify_model(FillInplace(), example_args, {}, Expected)


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
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(0, "int32"), dtype="void"
                )
                lv1: R.Tensor((128, 128), dtype="float32") = R.where(mask, lv, input)
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(128, 128, dtype=torch.float32), torch.rand(128, 128) < 0.5)
    verify_model(Masked_Fill(), example_args, {}, Expected)


def test_masked_fill_inplace():
    class Masked_Fill_Inplace(Module):
        def forward(self, input: torch.Tensor, mask: torch.Tensor):
            return input.masked_fill_(mask, 1.5)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((128, 128), dtype="float32"), mask: R.Tensor((128, 128), dtype="bool")
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(1.5, "float32"), dtype="void"
                )
                lv1: R.Tensor((128, 128), dtype="float32") = R.where(mask, lv, input)
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(128, 128, dtype=torch.float32), torch.rand(128, 128) < 0.5)
    verify_model(Masked_Fill_Inplace(), example_args, {}, Expected)


def test_new_ones():
    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 3), dtype="float32")):
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
        def main(
            x: R.Tensor((1, 128, 128), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 128, 128), dtype="float32") = R.full(
                    R.shape([1, 128, 128]), R.const(0, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((1, 128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 128, 128, dtype=torch.float32),)
    verify_model(NewZeros(), example_args, {}, expected1)


def test_to_copy():
    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    @tvm.script.ir_module
    class expected_float:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float32") = R.astype(x, dtype="float32")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    @tvm.script.ir_module
    class expected_half:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float16")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float16") = R.astype(x, dtype="float16")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float16")) = (lv,)
                R.output(gv)
            return gv

    # type
    class Type(Module):
        def forward(self, x):
            return x.type(torch.float32)

    @tvm.script.ir_module
    class expected_type:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float32") = R.astype(x, dtype="float32")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class To1(Module):
        def forward(self, input):
            return input.to(torch.float16)

    @I.ir_module
    class expected_to1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float16")):
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float16") = R.astype(inp_0, dtype="float16")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float16")) = (lv,)
                R.output(gv)
            return gv

    class To2(Module):
        def forward(self, input):
            return input.to("cpu")

    @I.ir_module
    class expected_to2:
        @R.function
        def main(
            inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float32") = R.astype(inp_0, dtype="float32")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(ToFloat(), example_args, {}, expected_float)
    verify_model(ToHalf(), example_args, {}, expected_half)
    verify_model(Type(), example_args, {}, expected_type)
    verify_model(To1(), example_args, {}, expected_to1)
    verify_model(To2(), example_args, {}, expected_to2)


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
    for param_var, param_ndarray in zip(func.params[1:], params):
        assert tuple(x.value for x in param_var.struct_info.shape.values) == param_ndarray.shape
        assert param_var.struct_info.dtype == param_ndarray.dtype

    tvm.testing.assert_allclose(params[0].numpy(), model.conv.weight.detach().detach().numpy())
    tvm.testing.assert_allclose(params[1].numpy(), model.conv.bias.detach().detach().numpy())


def test_unwrap_unit_return_tuple():
    class Identity(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return (x,)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((256, 256), dtype="float32") = inp_0
                R.output(gv)
            return gv

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    exported_program = export(Identity(), args=example_args)
    mod = from_exported_program(exported_program, unwrap_unit_return_tuple=True)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_no_bind_return_tuple():
    class Identity(Module):
        def __init__(self):
            super().__init__()

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
    exported_program = export(Identity(), args=example_args)
    mod = from_exported_program(exported_program, no_bind_return_tuple=True)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_empty_like():
    class EmptyLike(Module):
        def forward(self, data):
            return torch.empty_like(data)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.zeros_like(inp_0, dtype="void")
                gv: R.Tuple(R.Tensor((5,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, dtype=torch.float32),)

    verify_model(EmptyLike(), example_args, {}, Expected)


def test_one_hot():
    class OneHot(Module):
        def forward(self, indices):
            return torch.nn.functional.one_hot(indices, num_classes=10)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((5, 10), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((5, 10), dtype="int64") = R.one_hot(
                    inp_0, R.prim_value(1), R.prim_value(0), depth=10, axis=-1
                )
                gv: R.Tuple(R.Tensor((5, 10), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randint(0, 10, (5,), dtype=torch.int64),)

    verify_model(OneHot(), example_args, {}, Expected)


def test_ones_like():
    class OnesLike(Module):
        def forward(self, input):
            return torch.ones_like(input)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((128, 128), dtype="float32")
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.ones_like(input, dtype="void")
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
        def main(
            input: R.Tensor((128, 128), dtype="float32")
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.zeros_like(input, dtype="void")
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
        def main(
            input: R.Tensor((128, 128), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 2), dtype="float32") = R.zeros(R.shape([5, 2]), dtype="float32")
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
        def main(
            input: R.Tensor((128, 128), dtype="float32")
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.zeros_like(input, dtype="void")
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)
    verify_model(ZerosLike(), example_args, {}, Expected)


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


def test_select():
    class Select(Module):
        def forward(self, input):
            return torch.select(input, 0, 1)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.take(inp_0, R.const(1, "int64"), axis=0)
                gv: R.Tuple(R.Tensor((3,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32),)

    verify_model(Select(), example_args, {}, Expected)


def test_unflatten():
    class Unflatten(Module):
        def forward(self, input):
            return torch.ops.aten.unflatten(input, 1, (3, 5))

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

    verify_model(Unflatten(), example_args, {}, Expected)
    verify_model(Unflatten1(), example_args, {}, Expected)


def test_gather():
    class Gather0(Module):
        def forward(self, data, indices):
            return torch.gather(data, 0, indices)

    class Gather1(Module):
        def forward(self, data, indices):
            return torch.gather(data, 1, indices)

    class Gather2(Module):
        def forward(self, data, indices):
            return torch.gather(data, -1, indices)

    class Gather3(Module):
        def forward(self, data, indices):
            return torch.gather(data, -2, indices)

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
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="int64"),
        ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=1)
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

    @tvm.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="int64"),
        ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=-2)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randint(0, 3, (2, 3), dtype=torch.int64),
    )

    verify_model(Gather0(), example_args, {}, Expected0)
    verify_model(Gather1(), example_args, {}, Expected1)
    verify_model(Gather2(), example_args, {}, Expected2)
    verify_model(Gather3(), example_args, {}, Expected3)


def test_index_put():
    # Test case 1: 1D input
    class IndexPut1D(Module):
        def forward(self, data, indices_0, values):
            indices_tuple = (indices_0,)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_1d = (
        torch.randn(64, dtype=torch.float32),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected1D:
        @R.function
        def main(
            data: R.Tensor((64,), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((64,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64,), dtype="float32") = R.index_put(
                    data, R.tuple(indices_0), values, accumulate=False
                )
                gv: R.Tuple(R.Tensor((64,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

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
                    data, R.tuple(indices_0, indices_1), values, accumulate=False
                )
                gv: R.Tuple(R.Tensor((32, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # Test case 3: 3D input
    class IndexPut3D(Module):
        def forward(self, data, indices_0, indices_1, indices_2, values):
            indices_tuple = (indices_0, indices_1, indices_2)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_3d = (
        torch.randn(16, 32, 64, dtype=torch.float32),
        torch.randint(0, 16, (128,), dtype=torch.int64),
        torch.randint(0, 32, (128,), dtype=torch.int64),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected3D:
        @R.function
        def main(
            data: R.Tensor((16, 32, 64), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            indices_1: R.Tensor((128,), dtype="int64"),
            indices_2: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((16, 32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((16, 32, 64), dtype="float32") = R.index_put(
                    data, R.tuple(indices_0, indices_1, indices_2), values, accumulate=False
                )
                gv: R.Tuple(R.Tensor((16, 32, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # Test case 4: 4D input
    class IndexPut4D(Module):
        def forward(self, data, indices_0, indices_1, indices_2, indices_3, values):
            indices_tuple = (indices_0, indices_1, indices_2, indices_3)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_4d = (
        torch.randn(8, 16, 32, 64, dtype=torch.float32),
        torch.randint(0, 8, (128,), dtype=torch.int64),
        torch.randint(0, 16, (128,), dtype=torch.int64),
        torch.randint(0, 32, (128,), dtype=torch.int64),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected4D:
        @R.function
        def main(
            data: R.Tensor((8, 16, 32, 64), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            indices_1: R.Tensor((128,), dtype="int64"),
            indices_2: R.Tensor((128,), dtype="int64"),
            indices_3: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((8, 16, 32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((8, 16, 32, 64), dtype="float32") = R.index_put(
                    data,
                    R.tuple(indices_0, indices_1, indices_2, indices_3),
                    values,
                    accumulate=False,
                )
                gv: R.Tuple(R.Tensor((8, 16, 32, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # Test case 5: 5D input
    class IndexPut5D(Module):
        def forward(self, data, indices_0, indices_1, indices_2, indices_3, indices_4, values):
            indices_tuple = (indices_0, indices_1, indices_2, indices_3, indices_4)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_5d = (
        torch.randn(4, 8, 16, 32, 64, dtype=torch.float32),
        torch.randint(0, 4, (128,), dtype=torch.int64),
        torch.randint(0, 8, (128,), dtype=torch.int64),
        torch.randint(0, 16, (128,), dtype=torch.int64),
        torch.randint(0, 32, (128,), dtype=torch.int64),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected5D:
        @R.function
        def main(
            data: R.Tensor((4, 8, 16, 32, 64), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            indices_1: R.Tensor((128,), dtype="int64"),
            indices_2: R.Tensor((128,), dtype="int64"),
            indices_3: R.Tensor((128,), dtype="int64"),
            indices_4: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 8, 16, 32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 8, 16, 32, 64), dtype="float32") = R.index_put(
                    data,
                    R.tuple(indices_0, indices_1, indices_2, indices_3, indices_4),
                    values,
                    accumulate=False,
                )
                gv: R.Tuple(R.Tensor((4, 8, 16, 32, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # Run verification for each case
    verify_model(IndexPut1D(), example_args_1d, {}, Expected1D)
    verify_model(IndexPut2D(), example_args_2d, {}, Expected2D)
    verify_model(IndexPut3D(), example_args_3d, {}, Expected3D)
    verify_model(IndexPut4D(), example_args_4d, {}, Expected4D)
    verify_model(IndexPut5D(), example_args_5d, {}, Expected5D)


def test_flip():
    class Flip0(Module):
        def forward(self, data):
            return torch.flip(data, [0])

    class Flip1(Module):
        def forward(self, data):
            return torch.flip(data, [1])

    @tvm.script.ir_module
    class Expected0:
        @R.function
        def main(
            inp_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.flip(inp_0, axis=0)
                gv: R.Tuple(R.Tensor((2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.flip(inp_0, axis=1)
                gv: R.Tuple(R.Tensor((2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 2, dtype=torch.float32),)

    verify_model(Flip0(), example_args, {}, Expected0)
    verify_model(Flip1(), example_args, {}, Expected1)


def test_take():
    class Take(Module):
        def forward(self, data, indices):
            return torch.take(data, indices)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"),
            inp_1: R.Tensor((3,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((3,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int32") = R.astype(inp_1, dtype="int32")
                lv1: R.Tensor((3,), dtype="float32") = R.take(inp_0, lv, axis=None)
                gv: R.Tuple(R.Tensor((3,), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(5, dtype=torch.float32),
        torch.randint(0, 5, (3,), dtype=torch.int64),
    )

    verify_model(Take(), example_args, {}, Expected)


def test_std():
    class Std(Module):
        def forward(self, x):
            return torch.std(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.std(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Std(), example_args, {}, Expected)


def test_var():
    class Var(Module):
        def forward(self, x):
            return torch.var(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.variance(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Var(), example_args, {}, Expected)


def test_prod():
    class Prod(Module):
        def forward(self, x):
            return torch.prod(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.prod(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
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

    condition = torch.randint(0, 2, (5, 3), dtype=torch.bool)
    x = torch.randn(5, 3, dtype=torch.float32)
    y = torch.randn(5, 3, dtype=torch.float32)

    verify_model(Where(), (condition, x, y), {}, Expected)


def test_argsort():
    class Argsort(Module):
        def forward(self, x):
            return torch.argsort(x, dim=1, descending=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(R.Tensor((5, 3), dtype="int32")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int32") = R.argsort(
                    x, axis=1, descending=True, dtype="int32"
                )
                gv: R.Tuple(R.Tensor((5, 3), dtype="int32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Argsort(), example_args, {}, Expected)


def test_topk():
    class Topk(Module):
        def forward(self, x):
            return torch.topk(x, k=2, dim=1, largest=True, sorted=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")
                ) = R.topk(x, k=2, axis=1, ret_type="both", largest=True, dtype="int64")
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

    B = tvm.tir.SizeVar("BatchSize", dtype="int64")

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            lhs: R.Tensor((B, 4), dtype="float32"),
            rhs: R.Tensor((B, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((B, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((B, 4), dtype="float32") = R.add(lhs, rhs)
                gv: R.Tuple(R.Tensor((B, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 4), torch.randn(2, 4))
    batch = torch.export.Dim("batch")
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    verify_model(DynamicModel(), example_args, {}, Expected, dynamic_shapes=dynamic_shapes)


def test_broadcast_to():
    class BroadcastTo(Module):
        def forward(self, x):
            return torch.broadcast_to(x, (5, 3))

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 1), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.broadcast_to(x, R.shape([5, 3]))
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32")) = (lv,)
                R.output(gv)

            return gv

    example_args = (torch.randn(5, 1, dtype=torch.float32),)
    verify_model(BroadcastTo(), example_args, {}, Expected)


def test_narrow():
    class Narrow(Module):
        def forward(self, x):
            return torch.narrow(x, 1, 0, 2)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 2), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(1),),
                    (R.prim_value(0),),
                    (R.prim_value(2),),
                    assume_inbound=False,
                )
                gv: R.Tuple(R.Tensor((5, 2), dtype="float32")) = (lv,)
                R.output(gv)

            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Narrow(), example_args, {}, Expected)


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


def test_norm():
    class Norm(Module):
        def __init__(self, p, dim=None, keepdim=False):
            super().__init__()
            self.p = p
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.max(R.abs(inp_0), axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.min(R.abs(inp_0), axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(2, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(0.5, "float32"))
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected4:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(1.0, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(1.0, "float32"))
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected5:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(-4.0, "float32"))
                lv2: R.Tensor((1, 1, 1, 1), dtype="float32") = R.sum(lv1, axis=None, keepdims=True)
                lv3: R.Tensor((1, 1, 1, 1), dtype="float32") = R.power(
                    lv2, R.const(-0.25, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected6:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((1, 1, 1, 1), dtype="float32") = R.sum(lv1, axis=None, keepdims=True)
                lv3: R.Tensor((1, 1, 1, 1), dtype="float32") = R.power(lv2, R.const(2.0, "float32"))
                gv: R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    norms = [
        ((float("inf"), None, False), Expected1),
        ((float("-inf"), None, False), Expected2),
        ((float(2), None, False), Expected3),
        ((float(1.0), None, False), Expected4),
        ((float(-4), None, True), Expected5),
        ((float(0.5), None, True), Expected6),
    ]

    example_args = (torch.randn(1, 3, 5, 3, dtype=torch.float32),)

    for (p, dim, keepdim), expected in norms:
        verify_model(Norm(p, dim=dim, keepdim=keepdim), example_args, {}, expected)


def test_eye():
    class Eye1(Module):
        def forward(self, input):
            return torch.eye(3, 5, dtype=torch.float32)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            input: R.Tensor((3, 5), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 5), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 5), dtype="float32") = R.eye(3, 5, dtype="float32")
                gv: R.Tuple(R.Tensor((3, 5), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Eye2(Module):
        def forward(self, input):
            return torch.eye(5, dtype=torch.float32)

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            input: R.Tensor((5,), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 5), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 5), dtype="float32") = R.eye(5, dtype="float32")
                gv: R.Tuple(R.Tensor((5, 5), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args1 = (torch.randn(3, 5, dtype=torch.float32),)
    verify_model(Eye1(), example_args1, {}, Expected1)

    example_args2 = (torch.randn(5, dtype=torch.float32),)
    verify_model(Eye2(), example_args2, {}, Expected2)


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
                lv: R.Tensor((4, 3), dtype="float32") = R.nn.log_softmax(x, axis=-1)
                lv1: R.Tensor((), dtype="float32") = R.nn.nll_loss(
                    lv,
                    targets=R.const([0, 1, 2, 1], dtype="int64"),
                    reduction="mean",
                    ignore_index=-100,
                )
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv1,)
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
        def main(
            input: R.Tensor((9, 9), dtype="float32")
        ) -> R.Tuple(R.Tensor((9,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((9,), dtype="float32") = R.arange(0, 1.0625, 0.125, dtype="float32")
                gv: R.Tuple(R.Tensor((9,), dtype="float32")) = (lv,)
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
        torch.randint(0, 10, (10, 10)).to(torch_dtype),
        torch.randint(0, 10, (10, 10)).to(torch_dtype),
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


if __name__ == "__main__":
    tvm.testing.main()
1
