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
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (input_1,)
                R.output(gv)
            return gv

    verify_model(Dropout1(), example_args, {}, expected1)
    verify_model(Dropout2(), example_args, {}, expected1)

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
    class expected:
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

    verify_model(ReLU0(), example_args, {}, expected)
    verify_model(ReLU1(), example_args, {}, expected)


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
            conv_weight: R.Tensor((6, 3, 7, 7), dtype="float32"),
            conv_bias: R.Tensor((6,), dtype="float32"),
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
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
    for param_var, param_ndarray in zip(func.params[:-1], params):
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
