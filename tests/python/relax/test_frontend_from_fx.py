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
import torch.nn.functional as F
from torch import fx
from torch.nn import Module
import torchvision
import math

import tvm
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
            input_1: R.Tensor((1, 3, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
        ) -> R.Tensor((1, 6, 4), dtype="float32"):
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
                gv: R.Tensor((1, 6, 4), dtype="float32") = lv3
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
            input_1: R.Tensor((1, 3, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7), dtype="float32"),
        ) -> R.Tensor((1, 6, 4), dtype="float32"):
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
                gv: R.Tensor((1, 6, 4), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = [([1, 3, 10], "float32")]

    model = Conv1D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Conv1D1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Conv1D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)


def test_conv1d_transpose():
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
        ) -> R.Tensor((1, 6, 6), dtype="float32"):
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
                lv2: R.Tensor((1, 6, 1), dtype="float32") = R.reshape(w2, [1, 6, 1])
                lv3: R.Tensor((1, 6, 6), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tensor((1, 6, 6), dtype="float32") = lv3
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
        ) -> R.Tensor((1, 6, 6), dtype="float32"):
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
                gv: R.Tensor((1, 6, 6), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = [([1, 6, 4], "float32")]

    model = ConvTranspose1d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = ConvTranspose1d1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = ConvTranspose1d2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)


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
        ) -> R.Tensor((1, 6, 4, 4), dtype="float32"):
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
                lv2: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(w2, [1, 6, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tensor((1, 6, 4, 4), dtype="float32") = lv3
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
        ) -> R.Tensor((1, 6, 4, 4), dtype="float32"):
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
                gv: R.Tensor((1, 6, 4, 4), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = [([1, 3, 10, 10], "float32")]

    model = Conv2D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Conv2D1Func()
    binding = {"w1": model.weight.numpy(), "w2": model.bias.numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Conv2D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)


def test_conv2d_transpose():
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
        ) -> R.Tensor((1, 3, 16, 16), dtype="float32"):
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
                lv2: R.Tensor((1, 3, 1, 1), dtype="float32") = R.reshape(w2, [1, 3, 1, 1])
                lv3: R.Tensor((1, 3, 16, 16), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tensor((1, 3, 16, 16), dtype="float32") = lv3
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
        ) -> R.Tensor((1, 3, 16, 16), dtype="float32"):
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
                gv: R.Tensor((1, 3, 16, 16), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = [([1, 3, 10, 10], "float32")]

    model = ConvTranspose2d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = ConvTranspose2d1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = ConvTranspose2d2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)


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
        ) -> R.Tensor((1, 6, 4, 4, 4), dtype="float32"):
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
                lv2: R.Tensor((1, 6, 1, 1, 1), dtype="float32") = R.reshape(w2, [1, 6, 1, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = lv3
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
        ) -> R.Tensor((1, 6, 4, 4, 4), dtype="float32"):
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
                gv: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = [([1, 3, 10, 10, 10], "float32")]

    model = Conv3D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Conv3D1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Conv3D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)


def test_linear():
    # nn.Linear
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
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((7, 10), dtype="float32"),
            w2: R.Tensor((7,), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 7), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=None)
                lv1: R.Tensor((1, 3, 10, 7), dtype="float32") = R.matmul(
                    input_1, lv, out_dtype="float32"
                )
                lv2: R.Tensor((1, 3, 10, 7), dtype="float32") = R.add(lv1, w2)
                gv: R.Tensor((1, 3, 10, 7), dtype="float32") = lv2
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
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((7, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 7), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=None)
                lv1: R.Tensor((1, 3, 10, 7), dtype="float32") = R.matmul(
                    input_1, lv, out_dtype="float32"
                )
                gv: R.Tensor((1, 3, 10, 7), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = [([1, 3, 10, 10], "float32")]

    model = Dense1()
    binding = {"w1": model.linear.weight.detach().numpy(), "w2": model.linear.bias.detach().numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Dense1Func()
    binding = {"w1": model.weight.numpy(), "w2": model.bias.numpy()}
    verify_model(model, input_info, binding, expected1)

    model = Dense2()
    binding = {"w1": model.linear.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)

    # matmul
    class MatMul1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((10, 10), dtype="float32"),
            input_2: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(
                    input_1, input_2, out_dtype="float32"
                )
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(
        MatMul1(),
        [([10, 10], "float32"), ([10, 10], "float32")],
        {},
        expected3,
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
        ) -> R.Tensor((4, 128, 512), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(
                    input_1, input_2, out_dtype="float32"
                )
                gv: R.Tensor((4, 128, 512), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(
        BMM(),
        [((4, 128, 256), "float32"), ((4, 256, 512), "float32")],
        {},
        Expected,
    )


def test_baddbmm():
    class BAddBMM1(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    class BAddBMM2(Module):
        def __init__(self):
            super().__init__()

        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tensor((4, 128, 512), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(inp_1, inp_2)
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.add(lv, inp_0)
                gv: R.Tensor((4, 128, 512), dtype="float32") = lv1
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tensor((4, 128, 512), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(inp_1, inp_2)
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.multiply(
                    lv, R.const(2, "float32")
                )
                gv: R.Tensor((4, 128, 512), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(
        BAddBMM1(),
        [((4, 128, 512), "float32"), ((4, 128, 256), "float32"), ((4, 256, 512), "float32")],
        {},
        Expected1,
    )

    verify_model(
        BAddBMM2(),
        [((4, 128, 512), "float32"), ((4, 128, 256), "float32"), ((4, 256, 512), "float32")],
        {},
        Expected2,
    )


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
        def main(inp_0: R.Tensor((4, 4), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.einsum((inp_0,), subscripts="ii")
                gv: R.Tensor((), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"), inp_1: R.Tensor((4,), dtype="float32")
        ) -> R.Tensor((5, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5, 4), dtype="float32") = R.einsum(
                    (inp_0, inp_1), subscripts="i,j->ij"
                )
                gv: R.Tensor((5, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Einsum1(), [([4, 4], "float32")], {}, Expected1)
    verify_model(Einsum2(), [([5], "float32"), ([4], "float32")], {}, Expected2)


@tvm.testing.requires_gpu
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
        def main(inp_0: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.nn.softplus(
                    inp_0, beta=1.0, threshold=20.0
                )
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    input_info = [([10, 10], "float32")]
    verify_model(Softplus0(), input_info, {}, expected)
    verify_model(Softplus1(), input_info, {}, expected)


@tvm.testing.requires_gpu
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
            input_1: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.nn.leakyrelu(input_1, 0.02)
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    input_info = [([10, 10], "float32")]
    verify_model(LeakyReLU0(), input_info, {}, expected)
    verify_model(LeakyReLU1(), input_info, {}, expected)


def test_maxpool2d():
    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
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
        ) -> R.Tensor((1, 3, 4, 4), dtype="float32"):
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
                gv: R.Tensor((1, 3, 4, 4), dtype="float32") = lv
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
        ) -> R.Tensor((1, 3, 6, 6), dtype="float32"):
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
                gv: R.Tensor((1, 3, 6, 6), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(MaxPool2d(), input_info, {}, expected1)
    verify_model(MaxPool2d_functional(), input_info, {}, expected1)
    verify_model(MaxPool2d2(), input_info, {}, expected2)
    verify_model(MaxPool2d3(), input_info, {}, expected3)


def test_avgpool2d():
    input_info = [([1, 3, 10, 10], "float32")]

    class AvgPool2d(Module):
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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
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
                gv = lv
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
                gv = lv
                R.output(gv)
            return gv

    verify_model(AvgPool2d(), input_info, {}, expected1)
    verify_model(AvgPool2d2(), input_info, {}, expected2)
    verify_model(AvgPool2d3(), input_info, {}, expected2)
    verify_model(AvgPool2d4(), input_info, {}, expected3)


def test_adaptive_avgpool2d():
    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    input_1, output_size=[10, 10], layout="NCHW", out_layout="NCHW"
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(AdaptiveAvgPool2d0(), input_info, {}, expected1)
    verify_model(AdaptiveAvgPool2d1(), input_info, {}, expected1)


def test_flatten():
    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 100), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 100), dtype="float32") = R.reshape(input_1, (1, 3, 100))
                gv: R.Tensor((1, 3, 100), dtype="float32") = lv
                R.output(gv)
            return gv

    # call_module
    verify_model(Flatten(), input_info, {}, expected1)
    # call_method
    verify_model(torch.nn.Flatten(2, -1), input_info, {}, expected1)


def test_batchnorm2d():
    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv1
                R.output(gv)
            return gv

    model = BatchNorm2d()
    binding = {
        "w1": model.bn.weight.detach().numpy(),
        "w2": model.bn.bias.detach().numpy(),
        "w3": model.bn.running_mean.detach().numpy(),
        "w4": model.bn.running_var.detach().numpy(),
    }
    verify_model(BatchNorm2d(), input_info, binding, expected1)


def test_embedding():
    input_info = [([4], "int64")]

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
        ) -> R.Tensor((4, 3), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int32") = R.astype(input_1, dtype="int32")
                lv1: R.Tensor((4, 3), dtype="float32") = R.take(w1, lv, axis=0)
                gv: R.Tensor((4, 3), dtype="float32") = lv1
                R.output(gv)
            return gv

    model = Embedding()
    binding = {"w1": model.embedding.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected1)


def test_stochastic_depth():
    input_info = [([1, 3, 10, 10], "float32")]

    class StochasticDepth1(Module):
        def __init__(self):
            super().__init__()
            self.stochastic_depth = torchvision.ops.StochasticDepth(0.5, mode="row")

        def forward(self, x):
            return self.stochastic_depth(x)

    class StochasticDepth2(Module):
        def forward(self, x):
            return torchvision.ops.stochastic_depth(x, 0.5, mode="row", training=False)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = input_1
                R.output(gv)
            return gv

    verify_model(StochasticDepth1(), input_info, {}, expected1)
    verify_model(StochasticDepth2(), input_info, {}, expected1)


def test_layernorm():
    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    model = LayerNorm()
    binding = {
        "w1": model.ln.weight.detach().numpy(),
        "w2": model.ln.bias.detach().numpy(),
    }
    verify_model(LayerNorm(), input_info, binding, expected1)


def test_functional_layernorm():
    import numpy as np

    input_info = [([1, 3, 10, 10], "float32")]

    class LayerNorm(Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, input):
            return torch.nn.functional.layer_norm(
                input, self.weight.shape, self.weight, self.bias, 1e-5
            )

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((10, 10), dtype="float32"),
            w2: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    model = LayerNorm((10, 10))
    binding = {
        "w1": model.weight.detach().numpy(),
        "w2": model.bias.detach().numpy(),
    }
    verify_model(model, input_info, binding, expected1)

    class LayerNorm2(Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.weight = None
            self.bias = None

        def forward(self, input):
            return torch.nn.functional.layer_norm(input, self.shape, self.weight, self.bias, 1e-5)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.layer_norm(
                    input_1,
                    gamma=relax.const(np.ones((10, 10)), dtype="float32"),
                    beta=relax.const(np.zeros((10, 10)), dtype="float32"),
                    axes=[-2, -1],
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    model = LayerNorm2((10, 10))
    binding = {}
    verify_model(model, input_info, binding, expected2)

    class LayerNorm3(Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, input):
            return torch.nn.functional.layer_norm(input, self.shape, self.weight, self.bias, 1e-5)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor([10, 10], dtype="float32"),
            w2: R.Tensor([10, 10], dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    model = LayerNorm3([10, 10])
    binding = {
        "w1": model.weight.detach().numpy(),
        "w2": model.bias.detach().numpy(),
    }
    verify_model(model, input_info, binding, expected3)


def test_cross_entropy():
    input_info = [([3, 2], "float32"), ([3], "int32")]

    class CrossEntropy1(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((3, 2), dtype="float32"), inp_1: R.Tensor((3,), dtype="int32")
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.nn.log_softmax(inp_0, axis=-1)
                lv1: R.Tensor((), dtype="float32") = R.nn.nll_loss(
                    lv, inp_1, reduction="mean", ignore_index=-100
                )
                gv: R.Tensor((), dtype="float32") = lv1
                R.output(gv)
            return gv

    class CrossEntropy2(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2,)))
            self.loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            inp_0: R.Tensor((3, 2), dtype="float32"),
            inp_1: R.Tensor((3,), dtype="int32"),
            w1: R.Tensor((2,), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.nn.log_softmax(inp_0, axis=-1)
                lv1: R.Tensor((), dtype="float32") = R.nn.nll_loss(
                    lv,
                    inp_1,
                    w1,
                    reduction="mean",
                    ignore_index=-100,
                )
                gv: R.Tensor((), dtype="float32") = lv1
                R.output(gv)
            return gv

    class CrossEntropy3(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=1, reduction="sum")

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            inp_0: R.Tensor((3, 2), dtype="float32"), inp_1: R.Tensor((3,), dtype="int32")
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.nn.log_softmax(inp_0, axis=-1)
                lv1: R.Tensor((), dtype="float32") = R.nn.nll_loss(
                    lv, inp_1, reduction="sum", ignore_index=1
                )
                gv: R.Tensor((), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(CrossEntropy1(), input_info, {}, expected1)
    model = CrossEntropy2()
    binding = {"w1": model.loss.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)
    verify_model(CrossEntropy3(), input_info, {}, expected3)


def test_functional_cross_entropy():
    input_info = [([3, 10], "float32"), ([3], "int32")]

    class CrossEntropy(Module):
        def forward(self, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((3, 10), dtype="float32"), inp_1: R.Tensor((3,), dtype="int32")
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 10), dtype="float32") = R.nn.log_softmax(inp_0, axis=-1)
                lv1: R.Tensor((), dtype="float32") = R.nn.nll_loss(
                    lv, inp_1, reduction="mean", ignore_index=-100
                )
                gv: R.Tensor((), dtype="float32") = lv1
                R.output(gv)
            return gv

    model = CrossEntropy()
    verify_model(model, input_info, {}, expected1)


def test_groupnorm():
    import torch
    from torch.nn import Module

    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
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
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    model = GroupNorm()
    binding = {
        "w1": model.gn.weight.detach().numpy(),
        "w2": model.gn.bias.detach().numpy(),
    }
    verify_model(model, input_info, binding, expected1)


operator_binary_1 = [
    (operator.add, R.add),
    (operator.sub, R.subtract),
    (operator.mul, R.multiply),
    (operator.truediv, R.divide),
    (operator.floordiv, R.floor_divide),
    (operator.pow, R.power),
    (operator.mod, R.mod),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_1)
def test_binary1(op, relax_op):
    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

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
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = relax_op(lhs, rhs)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
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
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = relax_op(lhs, R.const(1.0))
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Binary1(op), input_info1, {}, expected_binary1)
    verify_model(Binary2(op), input_info2, {}, expected_binary2)


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
    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

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
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="bool"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(lhs, rhs)
                gv: R.Tensor((1, 3, 10, 10), dtype="bool") = lv
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
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="bool"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(lhs, R.const(1.0))
                gv: R.Tensor((1, 3, 10, 10), dtype="bool") = lv
                R.output(gv)
            return gv

    verify_model(Binary1(op), input_info1, {}, expected_binary1)
    verify_model(Binary2(op), input_info2, {}, expected_binary2)


operator_binary_3 = [
    (operator.lshift, R.left_shift),
    (operator.rshift, R.right_shift),
    (operator.and_, R.bitwise_and),
    (operator.or_, R.bitwise_or),
    (operator.xor, R.bitwise_xor),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_3)
def test_binary3(op, relax_op):
    input_info1 = [([1, 3, 10, 10], "int32"), ([1, 3, 10, 10], "int32")]
    input_info2 = [([1, 3, 10, 10], "int32")]

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
            lhs: R.Tensor((1, 3, 10, 10), dtype="int32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="int32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="int32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="int32") = relax_op(lhs, rhs)
                gv: R.Tensor((1, 3, 10, 10), dtype="int32") = lv
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
            lhs: R.Tensor((1, 3, 10, 10), dtype="int32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="int32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="int32") = relax_op(lhs, R.const(1))
                gv: R.Tensor((1, 3, 10, 10), dtype="int32") = lv
                R.output(gv)
            return gv

    verify_model(Binary1(op), input_info1, {}, expected_binary1)
    verify_model(Binary2(op), input_info2, {}, expected_binary2)


def test_size():
    input_info = [([1, 3, 10, 10], "float32")]

    class Size(Module):
        def forward(self, input):
            return input.size()

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Shape([1, 3, 10, 10]):
            # block 0
            with R.dataflow():
                gv: R.Shape([1, 3, 10, 10]) = R.shape([1, 3, 10, 10])
                R.output(gv)
            return gv

    verify_model(Size(), input_info, {}, expected1)


def test_squeeze():
    input_info = [([3, 1, 4, 1], "float32")]

    class Squeeze1(Module):
        def forward(self, input):
            return input.squeeze(1)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")
        ) -> R.Tensor((3, 4, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 4, 1), dtype="float32") = R.squeeze(inp_0, axis=[1])
                gv: R.Tensor((3, 4, 1), dtype="float32") = lv
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
        ) -> R.Tensor((3, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.squeeze(inp_0, axis=None)
                gv: R.Tensor((3, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Squeeze1(), input_info, {}, Expected1)
    verify_model(Squeeze2(), input_info, {}, Expected2)


def test_unsqueeze():
    input_info = [([1, 3, 10, 10], "float32")]

    class Unsqueeze1(Module):
        def forward(self, input):
            return input.unsqueeze(1)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 1, 3, 10, 10), dtype="float32") = R.expand_dims(input_1, 1)
                gv: R.Tensor((1, 1, 3, 10, 10), dtype="float32") = lv
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
        ) -> R.Tensor((1, 3, 10, 10, 1), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10, 1), dtype="float32") = R.expand_dims(input_1, -1)
                gv: R.Tensor((1, 3, 10, 10, 1), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Unsqueeze1(), input_info, {}, expected1)
    verify_model(Unsqueeze2(), input_info, {}, expected2)


def test_getattr():
    input_info = [([1, 3, 10, 10], "float32")]

    class GetAttr1(Module):
        def forward(self, input):
            return input.shape

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Shape([1, 3, 10, 10]):
            # block 0
            with R.dataflow():
                gv: R.Shape([1, 3, 10, 10]) = R.shape([1, 3, 10, 10])
                R.output(gv)
            return gv

    verify_model(GetAttr1(), input_info, {}, expected1)


def test_getitem():
    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 1, 10, 3), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 1, 10, 3), dtype="float32") = R.strided_slice(
                    x,
                    axes=[0, 1, 2, 3],
                    begin=[0, 1, 0, 0],
                    end=[1, T.int64(3), T.int64(10), 3],
                    strides=[1, 2, 1, 1],
                )
                lv1: R.Tensor((1, 1, 10, 3), dtype="float32") = R.reshape(lv, (1, 1, 10, 3))
                gv: R.Tensor((1, 1, 10, 3), dtype="float32") = lv1
                R.output(gv)
            return gv

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    @I.ir_module
    class expected2:
        @R.function
        def main(
            inp_0: R.Tensor((8, 16), dtype="float32")
        ) -> R.Tensor((8, 1, 1, 16, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((8, 16), dtype="float32") = R.strided_slice(
                    inp_0, axes=[0, 1], begin=[0, 0], end=[8, 16], strides=[1, 1]
                )
                lv1: R.Tensor((8, 1, 1, 16, 1), dtype="float32") = R.reshape(
                    lv, R.shape([8, 1, 1, 16, 1])
                )
                gv: R.Tensor((8, 1, 1, 16, 1), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(Slice1(), [([1, 3, 10, 10], "float32")], {}, expected1)
    verify_model(Slice2(), [([8, 16], "float32")], {}, expected2)


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
    (torch.log, R.log),
    (torch.neg, R.negative),
    (torch.round, R.round),
    (torch.rsqrt, R.rsqrt),
    (torch.sin, R.sin),
    (torch.sinh, R.sinh),
    (torch.sign, R.sign),
    (torch.sqrt, R.sqrt),
    (torch.square, R.square),
    (torch.tan, R.tan),
]


@pytest.mark.parametrize("pytorch_op, relax_op", operator_basic_unary)
def test_basic_unary_ops(pytorch_op, relax_op):
    input_info = [([1, 3, 10, 10], "float32")]

    class Unary(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm.script.ir_module
    class expected_unary:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = relax_op(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Unary(), input_info, {}, expected_unary)


operator_bool_unary = [
    (torch.isnan, R.isnan),
    (torch.isinf, R.isinf),
    (torch.isfinite, R.isfinite),
]


@pytest.mark.parametrize("pytorch_op, relax_op", operator_bool_unary)
def test_bool_unary_ops(pytorch_op, relax_op):
    input_info = [([1, 3, 10, 10], "float32")]

    class Unary(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm.script.ir_module
    class expected_unary:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="bool"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="bool") = lv
                R.output(gv)
            return gv

    verify_model(Unary(), input_info, {}, expected_unary)


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

    # leaky_relu
    test_leakyrelu()

    # softplus
    test_softplus()

    # log2
    class Log2(Module):
        def forward(self, x):
            return torch.log2(x)

    @tvm.script.ir_module
    class Expected_log2:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(inp_0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(0.6931471805599453, "float32")
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(Log2(), input_info, {}, Expected_log2)

    # log10
    class Log10(Module):
        def forward(self, x):
            return torch.log10(x)

    @tvm.script.ir_module
    class Expected_log10:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(inp_0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(2.302585092994046, "float32")
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(Log10(), input_info, {}, Expected_log10)

    # log1p
    class Log1p(Module):
        def forward(self, x):
            return torch.log1p(x)

    @tvm.script.ir_module
    class Expected_log1p:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(
                    R.add(inp_0, R.const(1, "float32"))
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Log1p(), input_info, {}, Expected_log1p)

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

    # reciprocal
    class Reciprocal(Module):
        def forward(self, input):
            return torch.reciprocal(input)

    @tvm.script.ir_module
    class expected_reciprocal:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    R.const(1.0, "float32"), input_1
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Reciprocal(), input_info, {}, expected_reciprocal)

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
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.selu(inp_0)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
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


def test_interpolate():
    input_info = [([1, 3, 10, 10], "float32")]

    class Interpolate(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (5, 5))

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 5, 5), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 5), dtype="float32") = R.image.resize2d(
                    input_1,
                    (5, 5),
                    roi=[0.000000, 0.000000, 0.000000, 0.000000],
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode="asymmetric",
                    rounding_method="round",
                    cubic_alpha=-0.5,
                    cubic_exclude=0,
                    extrapolation_value=0,
                    out_dtype="",
                )
                gv: R.Tensor((1, 3, 5, 5), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Interpolate(), input_info, {}, expected1)

    class Interpolate2(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(
                input,
                size=None,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 20, 20), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 20, 20), dtype="float32") = R.image.resize2d(
                    input_1,
                    (20, 20),
                    roi=[0.000000, 0.000000, 0.000000, 0.000000],
                    layout="NCHW",
                    method="linear",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.5,
                    cubic_exclude=0,
                    extrapolation_value=0,
                    out_dtype="",
                )
                gv: R.Tensor((1, 3, 20, 20), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Interpolate2(), input_info, {}, expected2)

    class Interpolate3(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(
                input,
                size=None,
                scale_factor=(2.0, 1.0),
                mode="bilinear",
                align_corners=False,
            )

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 20, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 20, 10), dtype="float32") = R.image.resize2d(
                    input_1,
                    (20, 10),
                    roi=[0.000000, 0.000000, 0.000000, 0.000000],
                    layout="NCHW",
                    method="linear",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.5,
                    cubic_exclude=0,
                    extrapolation_value=0,
                    out_dtype="",
                )
                gv: R.Tensor((1, 3, 20, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Interpolate3(), input_info, {}, expected3)


def test_addmm():
    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]

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
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.add(x1, lv)
                gv: R.Tensor((10, 10), dtype="float32") = lv1
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.multiply(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((10, 10), dtype="float32") = R.multiply(x1, R.const(0.8, "float32"))
                lv3: R.Tensor((10, 10), dtype="float32") = R.add(lv2, lv1)
                gv: R.Tensor((10, 10), dtype="float32") = lv3
                R.output(gv)
            return gv

    verify_model(Addmm1(), input_info, {}, expected1)
    verify_model(Addmm2(), input_info, {}, expected2)


def test_split():
    input_info = [([1, 3, 10, 10], "float32")]

    class Split1(Module):
        def forward(self, input):
            return torch.split(input, 1, dim=1)

    class Split2(Module):
        def forward(self, input):
            return torch.split(input, [1, 2], dim=1)

    @tvm.script.ir_module
    class expected1:
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
                gv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                ) = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((1, 1, 10, 10), dtype="float32"), R.Tensor((1, 2, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 2, 10, 10), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=[1], axis=1)
                gv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 2, 10, 10), dtype="float32"),
                ) = lv
                R.output(gv)
            return gv

    verify_model(Split1(), input_info, {}, expected1)
    verify_model(Split2(), input_info, {}, expected2)


def test_unbind():
    input_info = [([3, 3, 10, 10], "float32")]

    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

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
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = lv7
                R.output(gv)
            return gv

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
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = lv7
                R.output(gv)
            return gv

    verify_model(Unbind1(), input_info, {}, expected1)
    verify_model(Unbind2(), input_info, {}, expected2)


def test_cumsum():
    input_info = [([1, 2, 3, 4], "float32")]

    class Cumsum(Module):
        def forward(self, input):
            return torch.cumsum(input, dim=1, dtype=torch.int32)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tensor((1, 2, 3, 4), dtype="int32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="int32") = R.cumsum(input_1, axis=1, dtype="int32")
                gv: R.Tensor((1, 2, 3, 4), dtype="int32") = lv
                R.output(gv)
            return gv

    verify_model(Cumsum(), input_info, {}, expected1)


def test_chunk():
    input_info = [([1, 3, 10, 10], "float32")]

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
                gv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                ) = lv
                R.output(gv)
            return gv

    verify_model(Chunk(), input_info, {}, Expected)


def test_inplace_fill():
    class InplaceFill(Module):
        def forward(self, input):
            input.fill_(1.5)
            return input

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.full(
                    R.shape([10, 10]), R.const(1.5, "float32"), dtype="float32"
                )
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(InplaceFill(), [([10, 10], "float32")], {}, Expected)


def test_arange():
    import numpy as np

    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    class Arange(Module):
        def forward(self, input):
            return torch.arange(0, 20, dtype=torch.int32)

    graph_model = fx.symbolic_trace(Arange())
    mod = from_fx(graph_model, [([10, 10], "float32")])
    assert len(mod["main"].body.blocks) == 1
    assert len(mod["main"].body.blocks[0].bindings) == 1
    assert isinstance(mod["main"].body.blocks[0].bindings[0].value, relax.Constant)
    tvm.testing.assert_allclose(
        mod["main"].body.blocks[0].bindings[0].value.data.numpy(),
        np.arange(0, 20, dtype="int32"),
    )


def test_empty():
    class Empty(Module):
        def forward(self, input):
            return torch.empty((10, 10), dtype=torch.float32)

    graph_model = fx.symbolic_trace(Empty())
    mod = from_fx(graph_model, [([10, 10], "float32")])
    assert len(mod["main"].body.blocks) == 1
    assert len(mod["main"].body.blocks[0].bindings) == 1
    assert isinstance(mod["main"].body.blocks[0].bindings[0].value, relax.Constant)
    assert mod["main"].body.blocks[0].bindings[0].value.data.shape == (10, 10)
    assert mod["main"].body.blocks[0].bindings[0].value.data.dtype == "float32"


def test_tensor():
    class Empty1(Module):
        def forward(self, input):
            return torch.tensor(3, dtype=torch.float32)

    class Empty2(Module):
        def forward(self, input):
            return torch.tensor(3)

    graph_model1 = fx.symbolic_trace(Empty1())
    mod1 = from_fx(graph_model1, [([10, 10], "float32")])
    assert len(mod1["main"].body.blocks) == 1
    assert len(mod1["main"].body.blocks[0].bindings) == 1
    assert isinstance(mod1["main"].body.blocks[0].bindings[0].value, relax.Constant)
    assert mod1["main"].body.blocks[0].bindings[0].value.data.shape == ()
    assert mod1["main"].body.blocks[0].bindings[0].value.data.dtype == "float32"

    graph_model2 = fx.symbolic_trace(Empty2())
    mod2 = from_fx(graph_model2, [([10, 10], "float32")])
    assert len(mod2["main"].body.blocks) == 1
    assert len(mod2["main"].body.blocks[0].bindings) == 1
    assert isinstance(mod2["main"].body.blocks[0].bindings[0].value, relax.Constant)
    assert mod2["main"].body.blocks[0].bindings[0].value.data.shape == ()
    assert mod2["main"].body.blocks[0].bindings[0].value.data.dtype == "int64"


def test_new_ones():
    input_info = [([1, 2, 3], "float32")]

    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3), dtype="float32")) -> R.Tensor((1, 2, 3), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3), dtype="float32") = R.full(
                    (1, 2, 3), R.const(1, "float32"), dtype="float32"
                )
                gv: R.Tensor((1, 2, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(NewOnes(), input_info, {}, expected1)


def test_expand():
    input_info = [([1, 2, 3, 4], "float32")]

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
        ) -> R.Tensor((4, 2, 3, 4), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 2, 3, 4), dtype="float32") = R.broadcast_to(x, (4, 2, 3, 4))
                gv: R.Tensor((4, 2, 3, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Expand1(), input_info, {}, expected1)
    verify_model(Expand2(), input_info, {}, expected1)


def test_reduce():
    input_info = [([1, 2, 3, 4], "float32")]

    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tensor((1, 4), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.sum(inp_0, axis=[2, 1], keepdims=False)
                gv: R.Tensor((1, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Sum(), input_info, {}, expected1)


def test_datatype():
    input_info = [([1, 2, 3, 4], "float32")]

    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tensor((1, 2, 3, 4), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float32") = R.astype(x, dtype="float32")
                gv: R.Tensor((1, 2, 3, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(ToFloat(), input_info, {}, expected1)

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tensor((1, 2, 3, 4), dtype="float16"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float16") = R.astype(x, dtype="float16")
                gv: R.Tensor((1, 2, 3, 4), dtype="float16") = lv
                R.output(gv)
            return gv

    verify_model(ToHalf(), input_info, {}, expected2)

    # type
    class Type(Module):
        def forward(self, x):
            return x.type(torch.float32)

    # type
    class TypeFromAttr(Module):
        def forward(self, x):
            return x.type(x.getattr("dtype"))

    # astype
    class AsType(Module):
        def forward(self, x):
            return x.astype(torch.float32)

    verify_model(Type(), input_info, {}, expected1)
    verify_model(TypeFromAttr(), input_info, {}, expected1)
    verify_model(AsType(), input_info, {}, expected1)


def test_permute():
    input_info = [([1, 2, 3, 4], "float32")]

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
        ) -> R.Tensor((1, 4, 3, 2), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tensor((1, 4, 3, 2), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Permute1(), input_info, {}, expected1)
    verify_model(Permute2(), input_info, {}, expected1)


def test_reshape():
    input_info = [([1, 2, 3, 4], "float32")]

    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((2, 12), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tensor((2, 12), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Reshape(), input_info, {}, expected1)


def test_tile():
    input_info = [([1, 3], "float32")]

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
        def main(x: R.Tensor((1, 3), dtype="float32")) -> R.Tensor((1, 6), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 6), dtype="float32") = R.tile(x, [2])
                gv: R.Tensor((1, 6), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(x: R.Tensor((1, 3), dtype="float32")) -> R.Tensor((4, 6), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 6), dtype="float32") = R.tile(x, [4, 2])
                gv: R.Tensor((4, 6), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Tile1(), input_info, {}, expected1)
    verify_model(Tile2(), input_info, {}, expected2)
    verify_model(Tile3(), input_info, {}, expected2)


def test_transpose():
    input_info = [([1, 2, 3, 4], "float32")]

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32")
        ) -> R.Tensor((1, 4, 3, 2), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tensor((1, 4, 3, 2), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Transpose(), input_info, {}, expected1)


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
        def main(x: R.Tensor((3,), dtype="float32")) -> R.Tensor((6,), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((6,), dtype="float32") = R.tile(x, 2)
                gv: R.Tensor((6,), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(x: R.Tensor((1, 3), dtype="float32")) -> R.Tensor((4, 6), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 6), dtype="float32") = R.tile(x, [4, 2])
                gv: R.Tensor((4, 6), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Tile1(), [([3], "float32")], {}, expected1)
    verify_model(Tile2(), [([1, 3], "float32")], {}, expected2)
    verify_model(Tile2(), [(torch.Size([1, 3]), "float32")], {}, expected2)


def test_view():
    input_info = [([1, 2, 3, 4], "float32")]

    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((2, 12), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tensor((2, 12), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(View(), input_info, {}, expected1)


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
            w1: R.Tensor((6,), dtype="float32"),
            w2: R.Tensor((6, 3, 7, 7), dtype="float32"),
        ) -> R.Tensor((1, 6, 4, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    w2,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(w1, [1, 6, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tensor((1, 6, 4, 4), dtype="float32") = lv3
                R.output(gv)
            return gv

    model = Conv2D1()
    graph_model = fx.symbolic_trace(model)
    mod = from_fx(graph_model, [([1, 3, 10, 10], "float32")], keep_params_as_input=True)
    mod, params = detach_params(mod)
    tvm.ir.assert_structural_equal(mod, expected1)
    func = mod["main"]
    params = params["main"]

    assert len(params) == len(func.params) - 1
    for param_var, param_ndarray in zip(func.params[1:], params):
        assert tuple(x.value for x in param_var.struct_info.shape.values) == param_ndarray.shape
        assert param_var.struct_info.dtype == param_ndarray.dtype

    tvm.testing.assert_allclose(params[0].numpy(), model.conv.bias.detach().detach().numpy())
    tvm.testing.assert_allclose(params[1].numpy(), model.conv.weight.detach().detach().numpy())


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

    graph_model = fx.symbolic_trace(Identity())
    mod = from_fx(graph_model, [([256, 256], "float32")], unwrap_unit_return_tuple=True)
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

    graph_model = fx.symbolic_trace(Identity())
    mod = from_fx(
        graph_model, [([256, 256], "float32"), ([256, 256], "float32")], no_bind_return_tuple=True
    )
    tvm.ir.assert_structural_equal(mod, Expected)


def test_argmax():
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
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor((256,), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((256,), dtype="int64") = R.argmax(inp_0, axis=-1, keepdims=False)
                gv: R.Tensor((256,), dtype="int64") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor((256, 1), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((256, 1), dtype="int64") = R.argmax(inp_0, axis=-1, keepdims=True)
                gv: R.Tensor((256, 1), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_model(Argmax1(), [([256, 256], "float32")], {}, Expected1)
    verify_model(Argmax2(), [([256, 256], "float32")], {}, Expected2)


def test_argmin():
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
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor((), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((), dtype="int64") = R.argmin(inp_0, axis=None, keepdims=False)
                gv: R.Tensor((), dtype="int64") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor((1, 1), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((1, 1), dtype="int64") = R.argmin(inp_0, axis=None, keepdims=True)
                gv: R.Tensor((1, 1), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_model(Argmin1(), [([256, 256], "float32")], {}, Expected1)
    verify_model(Argmin2(), [([256, 256], "float32")], {}, Expected2)


def test_to():
    class To1(Module):
        def forward(self, input):
            return input.to(torch.float16)

    class To2(Module):
        def forward(self, input):
            return input.to("cpu")

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float16"):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float16") = R.astype(inp_0, dtype="float16")
                gv: R.Tensor((256, 256), dtype="float16") = lv
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((256, 256), dtype="float32") = inp_0
                R.output(gv)
            return gv

    verify_model(To1(), [([256, 256], "float32")], {}, Expected1)
    verify_model(To2(), [([256, 256], "float32")], {}, Expected2)


def test_mean():
    class Mean(Module):
        def forward(self, input):
            return input.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, input):
            return input.mean(-1, keepdim=True)

    @I.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor((256,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256,), dtype="float32") = R.mean(inp_0, axis=[-1], keepdims=False)
                gv: R.Tensor((256,), dtype="float32") = lv
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256, 1), dtype="float32") = R.mean(inp_0, axis=[-1], keepdims=True)
                gv: R.Tensor((256, 1), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Mean(), [([256, 256], "float32")], {}, Expected1)
    verify_model(MeanKeepDim(), [([256, 256], "float32")], {}, Expected2)


def test_cat():
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
        ) -> R.Tensor((4, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((4, 3), dtype="float32") = R.concat((inp_0, inp_1), axis=0)
                gv: R.Tensor((4, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tensor((2, 6), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 6), dtype="float32") = R.concat((inp_0, inp_1), axis=1)
                gv: R.Tensor((2, 6), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Cat0(), [([2, 3], "float32"), ([2, 3], "float32")], {}, Expected1)
    verify_model(Cat1(), [([2, 3], "float32"), ([2, 3], "float32")], {}, Expected2)
    verify_model(Cat2(), [([2, 3], "float32"), ([2, 3], "float32")], {}, Expected2)
    verify_model(Cat3(), [([2, 3], "float32"), ([2, 3], "float32")], {}, Expected1)


def test_max():
    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32"),
            inp_1: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = R.maximum(inp_0, inp_1)
                gv: R.Tensor((256, 256), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Max(), [([256, 256], "float32"), ([256, 256], "float32")], {}, Expected1)


def test_min():
    class Min(Module):
        def forward(self, x, y):
            return torch.min(x, y)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32"),
            inp_1: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = R.minimum(inp_0, inp_1)
                gv: R.Tensor((256, 256), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Min(), [([256, 256], "float32"), ([256, 256], "float32")], {}, Expected1)


def test_attention():
    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_1: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_2: R.Tensor((32, 8, 128, 64), dtype="float32"),
        ) -> R.Tensor((32, 8, 128, 64), dtype="float32"):
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
                gv: R.Tensor((32, 8, 128, 64), dtype="float32") = lv4
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_1: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_2: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_3: R.Tensor((32, 8, 128, 128), dtype="float32"),
        ) -> R.Tensor((32, 8, 128, 64), dtype="float32"):
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
                gv: R.Tensor((32, 8, 128, 64), dtype="float32") = lv4
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_1: R.Tensor((32, 8, 128, 64), dtype="float32"),
            inp_2: R.Tensor((32, 8, 128, 64), dtype="float32"),
        ) -> R.Tensor((32, 8, 128, 64), dtype="float32"):
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
                    lv, lv1, lv2, scale=None, causal_mask="TopLeft"
                )
                lv4: R.Tensor((32, 8, 128, 64), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 2, 1, 3]
                )
                gv: R.Tensor((32, 8, 128, 64), dtype="float32") = lv4
                R.output(gv)
            return gv

    verify_model(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
        [
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
        ],
        {},
        Expected1,
    )

    verify_model(
        lambda q, k, v, mask: F.scaled_dot_product_attention(q, k, v, mask),
        [
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 128], "float32"),
        ],
        {},
        Expected2,
    )

    verify_model(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        [
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
        ],
        {},
        Expected3,
    )


def test_sym_size_int():
    class SymSizeInt1(Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.ops.aten.sym_size.int(x, self.dim)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 4), dtype="float32"),
        ) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="int32") = R.const(3, "int32")
                gv: R.Tensor((), dtype="int32") = lv
                R.output(gv)
            return gv

    verify_model(SymSizeInt1(dim=1), [([1, 3, 4], "float32")], {}, Expected1)
    verify_model(SymSizeInt1(dim=-2), [([1, 3, 4], "float32")], {}, Expected1)


def test_stack():
    input_info = [
        ([1, 3, 10, 10], "float32"),
        ([1, 3, 10, 10], "float32"),
        ([1, 3, 10, 10], "float32"),
    ]

    class Stack(Module):
        def forward(self, data, data1, data2):
            return torch.stack((data, data1, data2), dim=0)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
            inp_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            inp_2: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((3, 1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 3, 10, 10), dtype="float32") = R.concat(
                    (inp_0, inp_1, inp_2), axis=0
                )
                lv1: R.Tensor((3, 1, 3, 10, 10), dtype="float32") = R.reshape(
                    lv, R.shape([3, 1, 3, 10, 10])
                )
                gv: R.Tensor((3, 1, 3, 10, 10), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(Stack(), input_info, {}, expected)


def test_scatter():
    input_info = [([20, 20], "float32"), ([2, 5], "int64"), ([2, 5], "float32")]

    class Scatter(Module):
        def forward(self, data, index, src):
            return data.scatter(dim=0, index=index, src=src)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            inp_0: R.Tensor((20, 20), dtype="float32"),
            inp_1: R.Tensor((2, 5), dtype="int64"),
            inp_2: R.Tensor((2, 5), dtype="float32"),
        ) -> R.Tensor((20, 20), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((20, 20), dtype="float32") = R.scatter_elements(
                    inp_0, inp_1, inp_2, axis=0, reduction="update"
                )
                gv: R.Tensor((20, 20), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Scatter(), input_info, {}, expected)


def test_masked_scatter():
    class MaskedScatter1(Module):
        def forward(self, data, mask, src):
            return data.masked_scatter(mask, src)

    class MaskedScatter2(Module):
        def forward(self, data, mask, src):
            return data.masked_scatter(mask, src)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"),
            inp_1: R.Tensor((5,), dtype="bool"),
            inp_2: R.Tensor((10,), dtype="float32"),
        ) -> R.Tensor((5,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="int32") = R.cumsum(
                    inp_1, axis=0, dtype="int32", exclusive=False
                )
                lv1: R.Tensor((5,), dtype="int32") = R.subtract(lv, R.const(1, "int32"))
                lv2: R.Tensor((5,), dtype="float32") = R.take(inp_2, lv1, axis=0)
                lv3: R.Tensor((5,), dtype="float32") = R.where(inp_1, lv2, inp_0)
                gv: R.Tensor((5,), dtype="float32") = lv3
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 5), dtype="float32"),
            inp_1: R.Tensor((2, 5), dtype="bool"),
            inp_2: R.Tensor((3, 5), dtype="float32"),
        ) -> R.Tensor((2, 5), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10,), dtype="bool") = R.reshape(inp_1, R.shape([10]))
                lv1: R.Tensor((10,), dtype="int32") = R.cumsum(
                    lv, axis=0, dtype="int32", exclusive=False
                )
                lv2: R.Tensor((10,), dtype="int32") = R.subtract(lv1, R.const(1, "int32"))
                lv3: R.Tensor((15,), dtype="float32") = R.reshape(inp_2, R.shape([15]))
                lv4: R.Tensor((10,), dtype="float32") = R.take(lv3, lv2, axis=0)
                lv5: R.Tensor((2, 5), dtype="float32") = R.reshape(lv4, R.shape([2, 5]))
                lv6: R.Tensor((2, 5), dtype="float32") = R.where(inp_1, lv5, inp_0)
                gv: R.Tensor((2, 5), dtype="float32") = lv6
                R.output(gv)
            return gv

    verify_model(
        MaskedScatter1(), [([5], "float32"), ([5], "bool"), ([10], "float32")], {}, expected1
    )
    verify_model(
        MaskedScatter2(),
        [([2, 5], "float32"), ([2, 5], "bool"), ([3, 5], "float32")],
        {},
        expected2,
    )


def test_is_floating_point():
    class IsFloatingPoint(Module):
        def forward(self, x):
            return torch.is_floating_point(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((), dtype="bool"):
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.const(True, "bool")
                R.output(gv)
            return gv

    verify_model(IsFloatingPoint(), [([2, 3], "float32")], {}, Expected)


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
            inp_1: R.Tensor((2, 3), dtype="int32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=0)
                gv: R.Tensor((2, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="int32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=1)
                gv: R.Tensor((2, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="int32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=-1)
                gv: R.Tensor((2, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="int32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(inp_0, inp_1, axis=-2)
                gv: R.Tensor((2, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Gather0(), [([2, 3], "float32"), ([2, 3], "int32")], {}, Expected0)
    verify_model(Gather1(), [([2, 3], "float32"), ([2, 3], "int32")], {}, Expected1)
    verify_model(Gather2(), [([2, 3], "float32"), ([2, 3], "int32")], {}, Expected2)
    verify_model(Gather3(), [([2, 3], "float32"), ([2, 3], "int32")], {}, Expected3)


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
        ) -> R.Tensor((2, 2), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.flip(inp_0, axis=0)
                gv: R.Tensor((2, 2), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((2, 2), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.flip(inp_0, axis=1)
                gv: R.Tensor((2, 2), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Flip0(), [([2, 2], "float32")], {}, Expected0)
    verify_model(Flip1(), [([2, 2], "float32")], {}, Expected1)


def test_take():
    class Take(Module):
        def forward(self, data, indices):
            return torch.take(data, indices)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"),
            inp_1: R.Tensor((3,), dtype="int32"),
        ) -> R.Tensor((3,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int32") = R.astype(inp_1, "int32")
                lv1: R.Tensor((3,), dtype="float32") = R.take(inp_0, lv)
                gv: R.Tensor((3,), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(Take(), [([5], "float32"), ([3], "int32")], {}, Expected)


def test_one_hot():
    class OneHot(Module):
        def forward(self, indices):
            return torch.nn.functional.one_hot(indices, num_classes=10)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="int32"),
        ) -> R.Tensor((5, 10), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((5, 10), dtype="int64") = R.one_hot(
                    inp_0, R.prim_value(1), R.prim_value(0), depth=10, axis=-1
                )
                gv: R.Tensor((5, 10), dtype="int64") = lv
                R.output(gv)

            return gv

    verify_model(OneHot(), [([5], "int32")], {}, Expected)


def test_empty_like():
    class EmptyLike(Module):
        def forward(self, data):
            return torch.empty_like(data)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"),
        ) -> R.Tensor((5,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.zeros_like(inp_0)
                gv: R.Tensor((5,), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(EmptyLike(), [([5], "float32")], {}, Expected)


def test_numel():
    class Numel(Module):
        def forward(self, data):
            return torch.numel(data)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                gv: R.Tensor((), dtype="int32") = R.const(15, "int32")
                R.output(gv)
            return gv

    verify_model(Numel(), [([5, 3], "float32")], {}, Expected)


def test_select():
    class Select(Module):
        def forward(self, data):
            return torch.select(data, 0, 1)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.take(inp_0, R.const(1, "int64"), axis=0)
                gv: R.Tensor((3,), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Select(), [([5, 3], "float32")], {}, Expected)


def test_clone():
    class Clone(Module):
        def forward(self, x):
            return x.clone()

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((5, 3), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((5, 3), dtype="float32") = inp_0
                R.output(gv)
            return gv

    verify_model(Clone(), [([5, 3], "float32")], {}, Expected)


def test_lerp():
    class Lerp(Module):
        def forward(self, start, end, weight):
            return torch.lerp(start, end, weight)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
            inp_1: R.Tensor((5, 3), dtype="float32"),
            inp_2: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((5, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.add(
                    inp_0, R.multiply(inp_2, R.subtract(inp_1, inp_0))
                )
                gv: R.Tensor((5, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(
        Lerp(), [([5, 3], "float32"), ([5, 3], "float32"), ([5, 3], "float32")], {}, Expected
    )


def test_std():
    class Std(Module):
        def forward(self, x):
            return torch.std(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.std(inp_0, axis=None, keepdims=False)
                gv: R.Tensor((), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Std(), [([5, 3], "float32")], {}, Expected)


def test_var():
    class Var(Module):
        def forward(self, x):
            return torch.var(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.variance(inp_0, axis=None, keepdims=False)
                gv: R.Tensor((), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Var(), [([5, 3], "float32")], {}, Expected)


def test_prod():
    class Prod(Module):
        def forward(self, x):
            return torch.prod(x)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.prod(inp_0, axis=None, keepdims=False)
                gv: R.Tensor((), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Prod(), [([5, 3], "float32")], {}, Expected)


def test_cumprod():
    class Cumprod(Module):
        def forward(self, x):
            return torch.cumprod(x, 0)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((5, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.cumprod(inp_0, axis=0, exclusive=False)
                gv: R.Tensor((5, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Cumprod(), [([5, 3], "float32")], {}, Expected)


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
        ) -> R.Tensor((5, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.where(inp_0, inp_1, inp_2)
                gv: R.Tensor((5, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(
        Where(), [([5, 3], "bool"), ([5, 3], "float32"), ([5, 3], "float32")], {}, Expected
    )


def test_argsort():
    class Argsort(Module):
        def forward(self, x):
            return torch.argsort(x, dim=1, descending=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((5, 3), dtype="int32"):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int32") = R.argsort(inp_0, axis=1, descending=True)
                gv: R.Tensor((5, 3), dtype="int32") = lv
                R.output(gv)
            return gv

    verify_model(Argsort(), [([5, 3], "float32")], {}, Expected)


def test_sort():
    class Sort(Module):
        def forward(self, x):
            return torch.sort(x, dim=1, descending=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int32")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int32") = R.argsort(
                    inp_0, axis=1, descending=True, dtype="int32"
                )
                lv1: R.Tensor((5, 3), dtype="float32") = R.gather_elements(inp_0, lv, axis=1)
                lv2: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int32")) = (
                    lv1,
                    lv,
                )
                gv: R.Tuple(
                    R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int32")
                ) = lv2
                R.output(gv)
            return gv

    verify_model(Sort(), [([5, 3], "float32")], {}, Expected)


def test_topk():
    class Topk(Module):
        def forward(self, x):
            return torch.topk(x, k=2, dim=1, largest=True, sorted=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")
                ) = R.topk(inp_0, k=2, axis=1, ret_type="both", largest=True, dtype="int64")
                gv: R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")) = lv
                R.output(gv)
            return gv

    verify_model(Topk(), [([5, 3], "float32")], {}, Expected)


def test_broadcast_to():
    class BroadcastTo(Module):
        def forward(self, x):
            return torch.broadcast_to(x, (5, 3))

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 1), dtype="float32"),
        ) -> R.Tensor((5, 3), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.broadcast_to(inp_0, (5, 3))
                gv: R.Tensor((5, 3), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(BroadcastTo(), [([5, 1], "float32")], {}, Expected)


def test_narrow():
    class Narrow(Module):
        def forward(self, x):
            return torch.narrow(x, 1, 0, 2)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((5, 2), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5, 2), dtype="float32") = R.strided_slice(
                    inp_0, axes=[1], begin=[0], end=[2]
                )
                gv: R.Tensor((5, 2), dtype="float32") = lv
                R.output(gv)

            return gv

    verify_model(Narrow(), [([5, 3], "float32")], {}, Expected)


def test_norm():

    input_info = [([1, 3, 5, 3], "float32")]

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
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.max(R.abs(inp_0), axis=None, keepdims=False)
                gv: R.Tensor((), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.min(R.abs(inp_0), axis=None, keepdims=False)
                gv: R.Tensor((), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(2, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(0.5, "float32"))
                gv: R.Tensor((), dtype="float32") = lv3
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected4:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(1.0, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(1.0, "float32"))
                gv: R.Tensor((), dtype="float32") = lv3
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected5:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(-4, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(-0.25, "float32"))
                gv: R.Tensor((), dtype="float32") = lv3
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected6:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(2, "float32"))
                gv: R.Tensor((), dtype="float32") = lv3
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected7:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.multiply(inp_0, inp_0)
                lv1: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                lv2: R.Tensor((), dtype="float32") = R.sqrt(lv1)
                gv: R.Tensor((), dtype="float32") = lv2
                R.output(gv)
            return gv

    norms = [
        (float("inf"), None, False),
        (float("-inf"), None, False),
        (float(2), None, False),
        (float(1.0), None, False),
        (float(-4), None, True),
        (float(0.5), None, True),
        ("fro", None, False),
    ]

    for norm, expected in zip(
        norms, [Expected1, Expected2, Expected3, Expected4, Expected5, Expected6, Expected7]
    ):
        p, dim, keepdim = norm
        verify_model(Norm(p, dim=dim, keepdim=keepdim), input_info, {}, expected)


if __name__ == "__main__":
    tvm.testing.main()
