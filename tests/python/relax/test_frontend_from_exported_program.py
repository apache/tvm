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
import torch
from torch.nn import Module
from torch.export import export

import tvm
from tvm import relax
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend.torch import from_exported_program


def verify_model(torch_model, example_args, binding, expected):
    exported_program = export(torch_model, args=example_args)
    mod = from_exported_program(exported_program)

    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm.ir.assert_structural_equal(mod, expected)


def test_unary():
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    # acos
    class Acos(Module):
        def forward(self, input):
            return torch.acos(input)

    @tvm.script.ir_module
    class expected_acos:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.acos(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Acos(), example_args, {}, expected_acos)

    # acosh
    class Acosh(Module):
        def forward(self, input):
            return torch.acosh(input)

    @tvm.script.ir_module
    class expected_acosh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.acosh(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Acosh(), example_args, {}, expected_acosh)

    # asin
    class Asin(Module):
        def forward(self, input):
            return torch.asin(input)

    @tvm.script.ir_module
    class expected_asin:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.asin(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Asin(), example_args, {}, expected_asin)

    # asinh
    class Asinh(Module):
        def forward(self, input):
            return torch.asinh(input)

    @tvm.script.ir_module
    class expected_asinh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.asinh(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Asinh(), example_args, {}, expected_asinh)

    # atan
    class Atan(Module):
        def forward(self, input):
            return torch.atan(input)

    @tvm.script.ir_module
    class expected_atan:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.atan(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Atan(), example_args, {}, expected_atan)

    # atanh
    class Atanh(Module):
        def forward(self, input):
            return torch.atanh(input)

    @tvm.script.ir_module
    class expected_atanh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.atanh(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Atanh(), example_args, {}, expected_atanh)

    # cos
    class Cos(Module):
        def forward(self, input):
            return torch.cos(input)

    @tvm.script.ir_module
    class expected_cos:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.cos(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Cos(), example_args, {}, expected_cos)

    # cosh
    class Cosh(Module):
        def forward(self, input):
            return torch.cosh(input)

    @tvm.script.ir_module
    class expected_cosh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.cosh(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Cosh(), example_args, {}, expected_cosh)

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

    # exp
    class Exp(Module):
        def forward(self, input):
            return torch.exp(input)

    @tvm.script.ir_module
    class expected_exp:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Exp(), example_args, {}, expected_exp)

    # neg
    class Neg(Module):
        def forward(self, input):
            return -input

    @I.ir_module
    class expected_neg:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.negative(inp_0)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Neg(), example_args, {}, expected_neg)

    # relu
    class ReLU0(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input):
            return self.relu(input)

    class ReLU1(Module):
        def forward(self, input):
            return torch.nn.functional.relu(input)

    @tvm.script.ir_module
    class expected_relu:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(ReLU0(), example_args, {}, expected_relu)
    verify_model(ReLU1(), example_args, {}, expected_relu)

    # rsqrt
    class Rsqrt(Module):
        def forward(self, input):
            return torch.rsqrt(input)

    @I.ir_module
    class expected_rsqrt:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.rsqrt(inp_0)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Rsqrt(), example_args, {}, expected_rsqrt)

    # sigmoid
    class Sigmoid(Module):
        def __init__(self):
            super().__init__()
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, input):
            return self.sigmoid(input)

    class Sigmoid2(Module):
        def forward(self, input):
            return torch.sigmoid(input)

    @tvm.script.ir_module
    class expected_sigmoid:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sigmoid(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Sigmoid(), example_args, {}, expected_sigmoid)
    verify_model(Sigmoid2(), example_args, {}, expected_sigmoid)

    # silu
    class SiLU(Module):
        def __init__(self):
            super().__init__()
            self.silu = torch.nn.SiLU()

        def forward(self, input):
            return self.silu(input)

    class SiLU2(Module):
        def forward(self, input):
            return torch.nn.functional.silu(input)

    @tvm.script.ir_module
    class expected_silu:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.silu(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(SiLU(), example_args, {}, expected_silu)
    verify_model(SiLU2(), example_args, {}, expected_silu)

    # sin
    class Sin(Module):
        def forward(self, input: torch.Tensor):
            return torch.sin(input)

    @tvm.script.ir_module
    class expected_sin:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sin(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Sin(), example_args, {}, expected_sin)

    # sinh
    class Sinh(Module):
        def forward(self, input):
            return torch.sinh(input)

    @tvm.script.ir_module
    class expected_sinh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sinh(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Sinh(), example_args, {}, expected_sinh)

    # sqrt
    class Sqrt(Module):
        def forward(self, input):
            return torch.sqrt(input)

    @tvm.script.ir_module
    class expected_sqrt:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sqrt(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Sqrt(), example_args, {}, expected_sqrt)

    # tan
    class Tan(Module):
        def forward(self, input):
            return torch.tan(input)

    @tvm.script.ir_module
    class expected_tan:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.tan(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Tan(), example_args, {}, expected_tan)

    # tanh
    class Tanh(Module):
        def forward(self, input):
            return torch.tanh(input)

    @tvm.script.ir_module
    class expected_tanh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.tanh(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Tanh(), example_args, {}, expected_tanh)


def test_clamp():
    class Clamp(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.1, max=0.5)

    @tvm.script.ir_module
    class expected_clamp:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(input_1, 0.1, 0.5)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Clamp(), example_args, {}, expected_clamp)


def test_gelu():
    class Gelu(Module):
        def __init__(self):
            super().__init__()
            self.gelu = torch.nn.GELU()

        def forward(self, input):
            return self.gelu(input)

    class Gelu2(Module):
        def forward(self, input):
            return torch.nn.functional.gelu(input)

    @tvm.script.ir_module
    class expected_gelu:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.gelu(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Gelu(), example_args, {}, expected_gelu)
    verify_model(Gelu2(), example_args, {}, expected_gelu)


def test_hardsigmoid():
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

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Hardsigmoid(), example_args, {}, expected_hardsigmoid)
    verify_model(Hardsigmoid2(), example_args, {}, expected_hardsigmoid)


def test_hardswish():
    class Hardswish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hs = torch.nn.Hardswish()

        def forward(self, input):
            return self.hs(input)

    class Hardswish2(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardswish(input)

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

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Hardswish(), example_args, {}, expected1)
    verify_model(Hardswish2(), example_args, {}, expected1)


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


def test_round():
    class Round(Module):
        def forward(self, input):
            return torch.round(input)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.round(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Round(), example_args, {}, expected)


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


def test_binary():
    example_args1 = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )
    example_args2 = (torch.randn(10, 10, dtype=torch.float32),)

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    @tvm.script.ir_module
    class expected_add1:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype="float32"),
            rhs: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.add(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    @tvm.script.ir_module
    class expected_add2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.add(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Add1(), example_args1, {}, expected_add1)
    verify_model(Add2(), example_args2, {}, expected_add2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    @tvm.script.ir_module
    class expected_truediv1:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
            rhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.divide(lhs_1, rhs_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    @tvm.script.ir_module
    class expected_truediv2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.divide(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(TrueDiv1(), example_args1, {}, expected_truediv1)
    verify_model(TrueDiv2(), example_args2, {}, expected_truediv2)

    # EQ
    class EQ1(Module):
        def forward(self, lhs, rhs):
            return lhs == rhs

    @tvm.script.ir_module
    class expected_eq1:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
            rhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = R.equal(lhs_1, rhs_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    class EQ2(Module):
        def forward(self, lhs):
            return lhs == 1.0

    @tvm.script.ir_module
    class expected_eq2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = R.equal(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    verify_model(EQ1(), example_args1, {}, expected_eq1)
    verify_model(EQ2(), example_args2, {}, expected_eq2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    @tvm.script.ir_module
    class expected_floordiv1:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
            rhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.floor_divide(lhs_1, rhs_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    @tvm.script.ir_module
    class expected_floordiv2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.floor_divide(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(FloorDiv1(), example_args1, {}, expected_floordiv1)
    verify_model(FloorDiv2(), example_args2, {}, expected_floordiv2)

    # LT
    class LT1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    @tvm.script.ir_module
    class expected_lt1:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
            rhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = R.less(lhs_1, rhs_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    class LT2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    @tvm.script.ir_module
    class expected_lt2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = R.less(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    verify_model(LT1(), example_args1, {}, expected_lt1)
    verify_model(LT2(), example_args2, {}, expected_lt2)

    # MatMul
    class MatMul1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    @tvm.script.ir_module
    class expected_matmul1:
        @R.function
        def main(
            input_1: R.Tensor((10, 10), dtype="float32"),
            input_2: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(
                    input_1, input_2, out_dtype="float32"
                )
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(MatMul1(), example_args1, {}, expected_matmul1)

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

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    @tvm.script.ir_module
    class expected_mul1:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
            rhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.multiply(lhs_1, rhs_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    @tvm.script.ir_module
    class expected_mul2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.multiply(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Mul1(), example_args1, {}, expected_mul1)
    verify_model(Mul2(), example_args2, {}, expected_mul2)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    @tvm.script.ir_module
    class expected_power1:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
            rhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.power(lhs_1, rhs_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    @tvm.script.ir_module
    class expected_power2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.power(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Power1(), example_args1, {}, expected_power1)
    verify_model(Power2(), example_args2, {}, expected_power2)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    @tvm.script.ir_module
    class expected_sub1:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
            rhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(lhs_1, rhs_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    @tvm.script.ir_module
    class expected_sub2:
        @R.function
        def main(
            lhs_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(lhs_1, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Sub1(), example_args1, {}, expected_sub1)
    verify_model(Sub2(), example_args2, {}, expected_sub2)


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
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="OIW",
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
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="OIW",
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
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
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
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
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
                    R.Tensor((0, 3, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=[1, 2, 3], axis=0)
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
                    R.Tensor((3, 0, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=[1, 2, 3], axis=1)
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
                    cubic_alpha=-0.5,
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
                    cubic_alpha=-0.5,
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
                    R.Tensor((0, 3, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=[1, 2, 3], axis=0)
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
                    R.Tensor((3, 0, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=[1, 2, 3], axis=1)
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
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float32")) = (x,)
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
