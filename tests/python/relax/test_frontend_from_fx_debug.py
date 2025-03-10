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

# TODO remove 
import sys
sys.path.append("/ssd1/htalendr/tvm/python")  # Refer to local TVM build

import torch
import torch.nn.functional as F
from torch import fx
from torch.nn import Module

import tvm
import math 
from tvm import relax
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.torch import from_fx


def verify_model(torch_model, input_info, binding, expected):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        mod = from_fx(graph_model, input_info)
    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm.ir.assert_structural_equal(mod, expected)



def test_extended_unary_ops():
    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv_celu
                R.output(gv)
            return gv

    verify_model(Celu1(), input_info, {}, expected_celu)
    verify_model(Celu2(), input_info, {}, expected_celu)

    # clamp
    class Clamp(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.1, max=0.5)

    @tvm.script.ir_module
    class expected_clamp:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(input_1, 0.1, 0.5)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Clamp(), input_info, {}, expected_clamp)

    class ClampMinOnly(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.5, max=None)

    @tvm.script.ir_module
    class expected_clamp_min_only:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(input_1, 0.5, math.inf)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(ClampMinOnly(), input_info, {}, expected_clamp_min_only)

    class ClampTensors(Module):
        def forward(self, input):
            return torch.clamp(input, min=input, max=input)

    @tvm.script.ir_module
    class expected_clamp_tensors:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.broadcast_to(
                    inp_0, R.shape([1, 3, 10, 10])
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.maximum(inp_0, lv)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.broadcast_to(
                    inp_0, R.shape([1, 3, 10, 10])
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.minimum(lv1, lv2)
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    lv3, R.prim_value(T.float64("-inf")), R.prim_value(T.float64("inf"))
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv4
                R.output(gv)
            return gv

    verify_model(ClampTensors(), input_info, {}, expected_clamp_tensors)

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = input_1
                R.output(gv)
            return gv

    verify_model(Dropout1(), input_info, {}, expected_dropout)
    verify_model(Dropout2(), input_info, {}, expected_dropout)

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv_elu
                R.output(gv)
            return gv

    verify_model(Elu(), input_info, {}, expected_elu)
    verify_model(Elu2(), input_info, {}, expected_elu)

    # gelu
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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.gelu(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Gelu(), input_info, {}, expected_gelu)
    verify_model(Gelu2(), input_info, {}, expected_gelu)


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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(inp_0, R.const(3, "float32"))
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(lv, 0, 6)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv1, R.const(6, "float32")
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv2
                R.output(gv)
            return gv

    verify_model(Hardsigmoid(), input_info, {}, expected_hardsigmoid)
    verify_model(Hardsigmoid2(), input_info, {}, expected_hardsigmoid)

    # hardswish
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
    class expected_hardswish:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(inp_0, R.const(3, "float32"))
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(lv, 0, 6)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv1, R.const(6, "float32")
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(inp_0, lv2)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv3
                R.output(gv)
            return gv

    verify_model(Hardswish(), input_info, {}, expected_hardswish)
    verify_model(Hardswish2(), input_info, {}, expected_hardswish)

    # hardtanh
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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    inp_0, R.prim_value(T.float64(-1.0)), R.prim_value(T.float64(1.0))
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Hardtanh(), input_info, {}, expected1)
    verify_model(Hardtanh2(), input_info, {}, expected1)

    # logical_not
    class LogicalNot(Module):
        def forward(self, input):
            return torch.logical_not(input)

    @tvm.script.ir_module
    class expected_logical_not:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.logical_not(inp_0)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(LogicalNot(), input_info, {}, expected_logical_not)

    # log_softmax
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
    class expected_log_softmax:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.log_softmax(input_1, axis=1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(LogSoftmax(), input_info, {}, expected_log_softmax)
    verify_model(LogSoftmax2(), input_info, {}, expected_log_softmax)

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(ReLU0(), input_info, {}, expected_relu)
    verify_model(ReLU1(), input_info, {}, expected_relu)

    # relu6
    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, input):
            return self.relu6(input)

    @tvm.script.ir_module
    class expected_relu6:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(input_1, 0, 6)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(ReLU6(), input_info, {}, expected_relu6)

    # selu
    class Selu1(Module):
        def __init__(self):
            super().__init__()
            self.selu = torch.nn.SELU()

        def forward(self, input):
            return self.selu(input)

    class Selu2(Module):
        def forward(self, input):
            return torch.nn.functional.selu(input)

    @tvm.script.ir_module
    class expected_selu:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv_relu: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input_1)
                lv_exp: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input_1)
                lv_sub: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(
                    lv_exp, R.const(1.0, "float32")
                )
                lv_scaled: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    R.const(1.6732631921768188, "float32"), lv_sub
                )
                lv_add: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv_relu, lv_scaled)
                lv_selu: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    R.const(1.0507009873554805, "float32"), lv_add
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv_selu
                R.output(gv)
            return gv

    verify_model(Selu1(), input_info, {}, expected_selu)
    verify_model(Selu2(), input_info, {}, expected_selu)

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sigmoid(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Sigmoid(), input_info, {}, expected_sigmoid)
    verify_model(Sigmoid2(), input_info, {}, expected_sigmoid)

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.silu(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(SiLU(), input_info, {}, expected_silu)
    verify_model(SiLU2(), input_info, {}, expected_silu)

    # softmax
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
    class expected_softmax:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.softmax(input_1, axis=1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Softmax(), input_info, {}, expected_softmax)
    verify_model(Softmax2(), input_info, {}, expected_softmax)

    # tanh
    class Tanh(Module):
        def __init__(self):
            super().__init__()
            self.tanh = torch.nn.Tanh()

        def forward(self, input):
            return self.tanh(input)

    class Tanh2(Module):
        def forward(self, input):
            return torch.tanh(input)

    @tvm.script.ir_module
    class expected_tanh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.tanh(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Tanh(), input_info, {}, expected_tanh)
    verify_model(Tanh2(), input_info, {}, expected_tanh)

    # tril
    class Tril(Module):
        def forward(self, input):
            return torch.tril(input, 1)

    class InplaceTril(Module):
        def forward(self, input):
            input.tril_(1)
            return input

    @tvm.script.ir_module
    class expected_tril:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.tril(input_1, 1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Tril(), input_info, {}, expected_tril)
    verify_model(InplaceTril(), input_info, {}, expected_tril)

    # triu
    class Triu(Module):
        def forward(self, input):
            return torch.triu(input, 1)

    class InplaceTriu(Module):
        def forward(self, input):
            input.triu_(1)
            return input

    @tvm.script.ir_module
    class expected_triu:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.triu(input_1, 1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Triu(), input_info, {}, expected_triu)
    verify_model(InplaceTriu(), input_info, {}, expected_triu)


if __name__ == "__main__":
    tvm.testing.main()
