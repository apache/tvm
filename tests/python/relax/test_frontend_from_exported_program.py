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
import torch
import torch.nn.functional as F
from torch import fx
from torch.nn import Module

import tvm
from tvm import relax
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.torch import from_exported_program


def verify_model(torch_model, input_info, binding, expected):
    dummy_inputs = []
    for i in range(len(input_info)):
        input_shape, dtype = input_info[i]
        if dtype=='int32':
            dtype = torch.int32
        else:
            dtype = torch.float32
        dummy_input = torch.zeros(*tuple(input_shape), dtype=dtype)
        dummy_inputs.append(dummy_input)
    exported_program = torch.export.export(torch_model, tuple(dummy_inputs))
    with torch.no_grad():
        mod = from_exported_program(exported_program, input_info)
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
                lv2: R.Tensor((1, 6, 1)) = R.reshape(w2, [1, 6, 1])
                gv: R.Tensor((1, 6, 4), dtype="float32") = R.add(lv1, lv2)
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
                gv: R.Tensor((1, 6, 4), dtype="float32") = R.nn.conv1d(
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
                R.output(gv)
            return gv

    input_info = [([1, 3, 10], "float32")]

    model = Conv1D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
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
                    kernel_layout="IOW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1)) = R.reshape(w2, [1, 6, 1])
                gv: R.Tensor((1, 6, 6), dtype="float32") = R.add(lv1, lv2)
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
                gv: R.Tensor((1, 6, 6), dtype="float32") = R.nn.conv1d_transpose(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="IOW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    input_info = [([1, 6, 4], "float32")]

    model = ConvTranspose1d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
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
                lv2: R.Tensor((1, 6, 1, 1)) = R.reshape(w2, [1, 6, 1, 1])
                gv: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, lv2)
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
                gv: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
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
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 3, 1, 1)) = R.reshape(w2, [1, 3, 1, 1])
                gv: R.Tensor((1, 3, 16, 16), dtype="float32") = R.add(lv1, lv2)
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
                gv: R.Tensor((1, 3, 16, 16), dtype="float32") = R.nn.conv2d_transpose(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    input_info = [([1, 3, 10, 10], "float32")]

    model = ConvTranspose2d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
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
                lv2: R.Tensor((1, 6, 1, 1, 1)) = R.reshape(w2, [1, 6, 1, 1, 1])
                gv: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.add(lv1, lv2)
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
                gv: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.nn.conv3d(
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
                R.output(gv)
            return gv

    input_info = [([1, 3, 10, 10, 10], "float32")]

    model = Conv3D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
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
                lv: R.Tensor((30, 10), dtype="float32") = R.reshape(input_1, R.shape([30, 10]))
                lv1: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                lv2: R.Tensor((30, 7), dtype="float32") = R.matmul(lv, lv1, out_dtype="float32")
                lv3: R.Tensor((30, 7), dtype="float32") = R.add(w2, lv2)
                lv4: R.Tensor((1, 3, 10, 7), dtype="float32") = R.reshape(lv3, R.shape([1, 3, 10, 7]))
                gv: R.Tensor((1, 3, 10, 7), dtype="float32") = lv4
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
                lv: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                lv1: R.Tensor((30, 10), dtype="float32") = R.reshape(input_1, R.shape([30, 10]))
                lv2: R.Tensor((30, 7), dtype="float32") = R.matmul(lv1, lv, out_dtype="float32")
                lv3: R.Tensor((1, 3, 10, 7), dtype="float32") = R.reshape(lv2, R.shape([1, 3, 10, 7]))
                gv: R.Tensor((1, 3, 10, 7), dtype="float32") = lv3
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
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(inp_1, inp_2, out_dtype="float32")
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.add(inp_0, lv)
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
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(inp_1, inp_2, out_dtype="float32")
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


def test_relu():
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
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.nn.relu(input_1)
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    input_info = [([10, 10], "float32")]
    verify_model(ReLU0(), input_info, {}, expected)
    verify_model(ReLU1(), input_info, {}, expected)


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
                lv1: R.Tensor((1, 3), dtype="float32") = R.mean(input_1, axis=[-2, -1], keepdims=False)
                lv2: R.Tensor((), dtype="float32") = R.variance(input_1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.add(lv2, R.const(1e-5, "float32"))
                lv4: R.Tensor((), dtype="float32") = R.rsqrt(lv3)
                lv5: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32"), R.Tensor((1, 3), dtype="float32"), R.Tensor((), dtype="float32")) = lv, lv1, lv4
                lv6: R.Tensor((1, 3, 10, 10), dtype="float32") = lv5[0]
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv6
                R.output(gv)
            return gv

    model = LayerNorm()
    binding = {
        "w1": model.ln.weight.detach().numpy(),
        "w2": model.ln.bias.detach().numpy(),
    }
    verify_model(LayerNorm(), input_info, binding, expected1)


def test_unary():
    input_info = [([1, 3, 10, 10], "float32")]

    # sin
    class Sin(Module):
        def forward(self, input):
            return torch.sin(input)

    @tvm.script.ir_module
    class expected_sin:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sin(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Sin(), input_info, {}, expected_sin)

    # cos
    class Cos(Module):
        def forward(self, input):
            return torch.cos(input)

    @tvm.script.ir_module
    class expected_cos:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.cos(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Cos(), input_info, {}, expected_cos)

    # tan
    class Tan(Module):
        def forward(self, input):
            return torch.tan(input)

    @tvm.script.ir_module
    class expected_tan:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.tan(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Tan(), input_info, {}, expected_tan)

    # asin
    class Asin(Module):
        def forward(self, input):
            return torch.asin(input)

    @tvm.script.ir_module
    class expected_asin:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.asin(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Asin(), input_info, {}, expected_asin)

    # acos
    class Acos(Module):
        def forward(self, input):
            return torch.acos(input)

    @tvm.script.ir_module
    class expected_acos:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.acos(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Acos(), input_info, {}, expected_acos)

    # atan
    class Atan(Module):
        def forward(self, input):
            return torch.atan(input)

    @tvm.script.ir_module
    class expected_atan:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.atan(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Atan(), input_info, {}, expected_atan)

    # sinh
    class Sinh(Module):
        def forward(self, input):
            return torch.sinh(input)

    @tvm.script.ir_module
    class expected_sinh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sinh(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Sinh(), input_info, {}, expected_sinh)

    # cosh
    class Cosh(Module):
        def forward(self, input):
            return torch.cosh(input)

    @tvm.script.ir_module
    class expected_cosh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.cosh(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Cosh(), input_info, {}, expected_cosh)

    # tanh
    class Tanh(Module):
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

    # asinh
    class Asinh(Module):
        def forward(self, input):
            return torch.asinh(input)

    @tvm.script.ir_module
    class expected_asinh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.asinh(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Asinh(), input_info, {}, expected_asinh)

    # acosh
    class Acosh(Module):
        def forward(self, input):
            return torch.acosh(input)

    @tvm.script.ir_module
    class expected_acosh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.acosh(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Acosh(), input_info, {}, expected_acosh)

    # atanh
    class Atanh(Module):
        def forward(self, input):
            return torch.atanh(input)

    @tvm.script.ir_module
    class expected_atanh:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.atanh(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Atanh(), input_info, {}, expected_atanh)

    # exp
    class Exp(Module):
        def forward(self, input):
            return torch.exp(input)

    @tvm.script.ir_module
    class expected_exp:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Exp(), input_info, {}, expected_exp)

    # sqrt
    class Sqrt(Module):
        def forward(self, input):
            return torch.sqrt(input)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sqrt(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Sqrt(), input_info, {}, expected3)

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
    class expected4:
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

    verify_model(Sigmoid(), input_info, {}, expected4)
    verify_model(Sigmoid2(), input_info, {}, expected4)

    # round
    class Round(Module):
        def forward(self, input):
            return torch.round(input)

    @tvm.script.ir_module
    class expected5:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.round(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Round(), input_info, {}, expected5)


def test_expand():
    input_info = [([1, 2, 3, 4], "float32")]

    class Expand(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

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

    verify_model(Expand(), input_info, {}, expected1)


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


def test_permute():
    input_info = [([1, 2, 3, 4], "float32")]

    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

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

    verify_model(Permute(), input_info, {}, expected1)


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


def test_rsqrt():
    class Rsqrt(Module):
        def forward(self, input):
            return torch.rsqrt(input)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = R.rsqrt(inp_0)
                gv: R.Tensor((256, 256), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Rsqrt(), [([256, 256], "float32")], {}, Expected1)


def test_neg():
    class Neg(Module):
        def forward(self, input):
            return -input

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = R.negative(inp_0)
                gv: R.Tensor((256, 256), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Neg(), [([256, 256], "float32")], {}, Expected1)


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

if __name__ == "__main__":
    tvm.testing.main()
