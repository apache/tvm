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
import math
import operator

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from frontend_torch_utils import (
    AntialiasedResizeModel,
    ExponentialModel,
    PoolDivisorModel,
    UnaryModule,
    activation_cases,
    constants,
    make_expected,
    verify_exponential,
    verify_numerically,
)
from torch import fx
from torch.nn import Module

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.torch import from_fx
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


def verify_model(torch_model, input_info, binding, expected, **import_options):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        mod = from_fx(graph_model, input_info, **import_options)
    binding = {k: tvm.runtime.tensor(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm.ir.assert_structural_equal(mod, expected)


@pytest.mark.parametrize(
    "ndim,transpose", [(1, False), (2, False), (3, False), (1, True), (2, True)]
)
@pytest.mark.parametrize("as_module", [False, True], ids=["function-bias", "module-no-bias"])
def test_convolution(ndim, transpose, as_module):
    layer_type = getattr(torch.nn, f"Conv{'Transpose' if transpose else ''}{ndim}d")
    layer = layer_type(2, 3, 3, bias=not as_module)
    torch_op = getattr(F, f"conv_transpose{ndim}d" if transpose else f"conv{ndim}d")
    relax_op = getattr(relax.op.nn, f"conv{ndim}d" + ("_transpose" if transpose else ""))

    class Functional(Module):
        def __init__(self):
            super().__init__()
            self.weight, self.bias = layer.weight, layer.bias

        def forward(self, x):
            return torch_op(x, self.weight, self.bias)

    def expected(x):
        spatial = {1: "W", 2: "HW", 3: "DHW"}[ndim]
        result = relax_op(
            x,
            relax.const(layer.weight.detach().numpy()),
            data_layout="NC" + spatial,
            kernel_layout=("IO" if transpose else "OI") + spatial,
            out_layout="NC" + spatial,
            out_dtype="float32",
        )
        if layer.bias is not None:
            result = relax.op.add(
                result,
                relax.op.reshape(relax.const(layer.bias.detach().numpy()), [1, 3] + [1] * ndim),
            )
        return result

    info = [((1, 2) + (5,) * ndim, "float32")]
    model = UnaryModule(layer) if as_module else Functional()
    verify_model(model, info, {}, make_expected(info, expected))


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
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 14, 12), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="constant",
                    pad_value=0.0,
                )
                gv: R.Tensor((1, 3, 14, 12), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_reflect:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 14, 12), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="reflect",
                    pad_value=0.0,
                )
                gv: R.Tensor((1, 3, 14, 12), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_replicate:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 14, 12), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="replicate",
                    pad_value=0.0,
                )
                gv: R.Tensor((1, 3, 14, 12), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_circular:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 14, 12), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="circular",
                    pad_value=0.0,
                )
                gv: R.Tensor((1, 3, 14, 12), dtype="float32") = lv
                R.output(gv)
            return gv

    input_infos = [([1, 3, 10, 10], "float32")]
    verify_model(PadModel(pad=[1, 1, 2, 2]), input_infos, {}, expected_constant)
    verify_model(PadModel(pad=[1, 1, 2, 2], mode="reflect"), input_infos, {}, expected_reflect)
    verify_model(PadModel(pad=[1, 1, 2, 2], mode="replicate"), input_infos, {}, expected_replicate)
    verify_model(PadModel(pad=[1, 1, 2, 2], mode="circular"), input_infos, {}, expected_circular)


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
        def main(inp_0: R.Tensor((1, 8, 10, 15), dtype="float32")) -> R.Tensor(
            (1, 2, 20, 30), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 2, 20, 30), dtype="float32") = R.nn.pixel_shuffle(
                    inp_0, upscale_factor=2
                )
                gv: R.Tensor((1, 2, 20, 30), dtype="float32") = lv
                R.output(gv)
            return gv

    input_infos = [([1, 8, 10, 15], "float32")]
    verify_model(PixelShuffle1(2), input_infos, {}, expected)
    verify_model(PixelShuffle2(2), input_infos, {}, expected)


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
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    class BAddBMM2(Module):
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
        def forward(self, x):
            return torch.einsum("ii", x)

    class Einsum2(Module):
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


def test_outer():
    class Outer(torch.nn.Module):
        def forward(self, x, y):
            return torch.outer(x, y)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            a: R.Tensor((3,), dtype="float32"), b: R.Tensor((4,), dtype="float32")
        ) -> R.Tensor((3, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.outer(a, b)
                gv: R.Tensor((3, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    input_infos = [([3], "float32"), ([4], "float32")]
    verify_model(Outer(), input_infos, {}, expected)


def test_softplus():
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


def test_leakyrelu():
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
        def main(input_1: R.Tensor((10, 10), dtype="float32")) -> R.Tensor(
            (10, 10), dtype="float32"
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.nn.leakyrelu(input_1, 0.02)
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    input_info = [([10, 10], "float32")]
    verify_model(LeakyReLU0(), input_info, {}, expected)
    verify_model(LeakyReLU1(), input_info, {}, expected)


def test_prelu():
    class Prelu1(Module):
        def __init__(self, num_parameters=1, alpha=0.25):
            super().__init__()
            self.prelu = torch.nn.PReLU(num_parameters=num_parameters, init=alpha)

        def forward(self, x):
            return self.prelu(x)

    class Prelu2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.tensor([0.25]))

        def forward(self, x):
            return torch.nn.functional.prelu(x, self.alpha)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 10, 10), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.prelu(
                    x, R.const([0.25], dtype="float32"), axis=1
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Prelu1(), input_info, {}, expected)
    verify_model(Prelu2(), input_info, {}, expected)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("kind", ["max", "avg"])
@pytest.mark.parametrize("as_module", [True, False])
def test_pool(rank, kind, as_module):
    shape = (1, 2) + (9,) * rank
    layout = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[rank]
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
    expected = make_expected(info, lambda x: getattr(relax.op.nn, op_name)(x, **attrs))
    verify_model(UnaryModule(op), info, {}, expected)


def test_adaptive_avgpool1d():
    input_info = [([1, 3, 16], "float32")]

    class AdaptiveAvgPool1d0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool1d(8)

        def forward(self, input):
            return self.pool(input)

    class AdaptiveAvgPool1d1(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool1d(input, 8)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 16), dtype="float32")) -> R.Tensor(
            (1, 3, 8), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 8), dtype="float32") = R.nn.adaptive_avg_pool1d(
                    input_1, output_size=[8], layout="NCW", out_layout="NCW"
                )
                gv: R.Tensor((1, 3, 8), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(AdaptiveAvgPool1d0(), input_info, {}, expected1)
    verify_model(AdaptiveAvgPool1d1(), input_info, {}, expected1)


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
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 10, 10), dtype="float32"
        ):
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


def test_adaptive_avgpool3d():
    input_info = [([1, 3, 16, 16, 16], "float32")]

    class AdaptiveAvgPool3d0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool3d((8, 8, 8))

        def forward(self, input):
            return self.pool(input)

    class AdaptiveAvgPool3d1(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool3d(input, (8, 8, 8))

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 16, 16, 16), dtype="float32")) -> R.Tensor(
            (1, 3, 8, 8, 8), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 8, 8, 8), dtype="float32") = R.nn.adaptive_avg_pool3d(
                    input_1, output_size=[8, 8, 8], layout="NCDHW", out_layout="NCDHW"
                )
                gv: R.Tensor((1, 3, 8, 8, 8), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(AdaptiveAvgPool3d0(), input_info, {}, expected1)
    verify_model(AdaptiveAvgPool3d1(), input_info, {}, expected1)


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
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 100), dtype="float32"
        ):
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


@pytest.mark.parametrize(
    "op,message",
    [
        (torch.nn.Flatten(2, 1), "start_dim cannot come after end_dim"),
        (lambda x: torch.flatten(x, 0, 3), "flatten end_dim 3 is out of range"),
    ],
)
def test_flatten_invalid_dims(op, message):
    # FX traces metadata without running the invalid operation.
    with pytest.raises(ValueError, match=message):
        from_fx(fx.symbolic_trace(UnaryModule(op)), [((2, 3, 4), "float32")])


def test_flatten_scalar_input():
    input_info = [([], "float32")]

    class Flatten(Module):
        def forward(self, input):
            return torch.flatten(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1,), dtype="float32") = R.reshape(input_1, (1,))
                gv: R.Tensor((1,), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Flatten(), input_info, {}, expected1)


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
    torchvision = pytest.importorskip("torchvision")

    input_info = [([1, 3, 10, 10], "float32")]

    class StochasticDepth1(Module):
        def __init__(self):
            super().__init__()
            self.stochastic_depth = torchvision.ops.StochasticDepth(0.5, mode="row")

        def forward(self, x):
            return self.stochastic_depth(x)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 10, 10), dtype="float32"
        ):
            # block 0
            with R.dataflow():
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = input_1
                R.output(gv)
            return gv

    # The PyTorch frontend imports models with inference semantics, so stochastic
    # depth is lowered to an identity even when the traced module is in training mode.
    verify_model(StochasticDepth1(), input_info, {}, expected1)


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


def test_cross_entropy():
    input_info = [([3, 2], "float32"), ([3], "int64")]

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
            inp_0: R.Tensor((3, 2), dtype="float32"), inp_1: R.Tensor((3,), dtype="int64")
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
            self.weight = torch.nn.Parameter(torch.ones((2,)), requires_grad=False)
            self.loss = torch.nn.CrossEntropyLoss(
                weight=self.weight, reduction="sum", ignore_index=1
            )

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            inp_0: R.Tensor((3, 2), dtype="float32"),
            inp_1: R.Tensor((3,), dtype="int64"),
            w1: R.Tensor((2,), dtype="float32"),
        ) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 2), dtype="float32") = R.nn.log_softmax(inp_0, axis=-1)
                lv1: R.Tensor((), dtype="float32") = R.nn.nll_loss(
                    lv,
                    inp_1,
                    w1,
                    reduction="sum",
                    ignore_index=1,
                )
                gv: R.Tensor((), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(CrossEntropy1(), input_info, {}, expected1)
    model = CrossEntropy2()
    binding = {"w1": model.loss.weight.detach().numpy()}
    verify_model(model, input_info, binding, expected2)


def test_functional_cross_entropy():
    input_info = [([3, 10], "float32"), ([3], "int64")]

    class CrossEntropy(Module):
        def forward(self, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((3, 10), dtype="float32"), inp_1: R.Tensor((3,), dtype="int64")
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


def test_instancenorm2d():
    input_info = [([1, 3, 10, 10], "float32")]

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
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.instance_norm(
                    input_1,
                    w1,
                    w2,
                    channel_axis=1,
                    axes=[2, 3],
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    model = InstanceNorm2d()
    binding = {
        "w1": torch.ones(3).detach().numpy(),
        "w2": torch.zeros(3).detach().numpy(),
    }
    verify_model(model, input_info, binding, expected1)


operator_binary_1 = [
    (operator.add, R.add),
    (operator.sub, R.subtract),
    (operator.mul, R.multiply),
    (operator.truediv, R.divide),
    (operator.floordiv, R.floor_divide),
    (torch.ops.aten.fmod, R.mod),
    (operator.pow, R.power),
    (operator.mod, R.floor_mod),
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
    if op is operator.sub:
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
    if op is operator.lt:
        verify_model(Binary2(op), input_info2, {}, expected_binary2)


operator_binary_3 = [
    (torch.ops.aten.bitwise_or_, R.bitwise_or),
    (torch.ops.aten.bitwise_or, R.bitwise_or),
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
    if op is operator.and_:
        verify_model(Binary2(op), input_info2, {}, expected_binary2)


# RSub
def test_rsub():
    input_info1 = [([10, 10], "float32"), ([10, 10], "float32")]
    input_info2 = [([10, 10], "float32")]

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
        ) -> R.Tensor((10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(y, x)
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected_rsub2:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(R.const(5.0, "float32"), x)
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(RSub1(), input_info1, {}, expected_rsub1)
    verify_model(RSub2(), input_info2, {}, expected_rsub2)


# IsIn


def test_isin():
    input_info = [([10, 10], "float32"), ([8], "float32")]

    class IsInModel(torch.nn.Module):
        def forward(self, x, test_elements):
            return torch.isin(x, test_elements)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            inp_0: R.Tensor((10, 10), dtype="float32"), inp_1: R.Tensor((8,), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="bool"):
            with R.dataflow():
                lv: R.Tensor((10, 10, 1), dtype="float32") = R.expand_dims(inp_0, axis=[-1])
                lv1: R.Tensor((8,), dtype="float32") = R.reshape(inp_1, R.shape([8]))
                lv2: R.Tensor((10, 10, 8), dtype="bool") = R.equal(lv, lv1)
                lv3: R.Tensor((10, 10), dtype="bool") = R.sum(lv2, axis=[-1], keepdims=False)
                lv4: R.Tensor((10, 10), dtype="bool") = R.greater(lv3, R.const(0.0, "float32"))
                gv: R.Tensor((10, 10), dtype="bool") = lv4
                R.output(gv)
            return gv

    verify_model(IsInModel(), input_info, {}, expected)


def test_div_mode():
    input_info = [([64, 64], "float32"), ([64, 64], "float32")]

    # Case 1: Basic division (no rounding mode)
    class DivModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.div(x, y)

    @tvm.script.ir_module
    class expected_div:
        @R.function
        def main(
            inp_0: R.Tensor((64, 64), dtype="float32"), inp_1: R.Tensor((64, 64), dtype="float32")
        ) -> R.Tensor((64, 64), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.divide(inp_0, inp_1)
                gv: R.Tensor((64, 64), dtype="float32") = lv
                R.output(gv)
            return gv

    # Case 2: Division with trunc rounding
    class DivTruncModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.div(x, y, rounding_mode="trunc")

    @tvm.script.ir_module
    class expected_div_trunc:
        @R.function
        def main(
            inp_0: R.Tensor((64, 64), dtype="float32"), inp_1: R.Tensor((64, 64), dtype="float32")
        ) -> R.Tensor((64, 64), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.divide(inp_0, inp_1)
                lv1: R.Tensor((64, 64), dtype="float32") = R.trunc(lv)
                gv: R.Tensor((64, 64), dtype="float32") = lv1
                R.output(gv)
            return gv

    # Case 3: Division with floor rounding
    class DivFloorModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.div(x, y, rounding_mode="floor")

    @tvm.script.ir_module
    class expected_div_floor:
        @R.function
        def main(
            inp_0: R.Tensor((64, 64), dtype="float32"), inp_1: R.Tensor((64, 64), dtype="float32")
        ) -> R.Tensor((64, 64), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.floor_divide(inp_0, inp_1)
                gv: R.Tensor((64, 64), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(DivModel(), input_info, {}, expected_div)
    verify_model(DivTruncModel(), input_info, {}, expected_div_trunc)
    verify_model(DivFloorModel(), input_info, {}, expected_div_floor)


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
        def main(inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")) -> R.Tensor(
            (3, 4, 1), dtype="float32"
        ):
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
        def main(inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")) -> R.Tensor(
            (3, 4), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.squeeze(inp_0, axis=None)
                gv: R.Tensor((3, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Squeeze1(), input_info, {}, Expected1)
    verify_model(Squeeze2(), input_info, {}, Expected2)


def test_unsqueeze():
    shape = (2, 3)
    model = UnaryModule(lambda x: x.unsqueeze(-1))
    info = [(shape, "float32")]
    expected = make_expected(info, lambda x: relax.op.expand_dims(x, axis=-1))
    verify_model(model, info, {}, expected)


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


@pytest.mark.parametrize(
    "index,shape,compute",
    [
        (
            (0, slice(1, None, 2), slice(None), slice(None, 3)),
            (1, 3, 10, 10),
            lambda x: relax.op.reshape(
                relax.op.strided_slice(x, [0, 1, 2, 3], [0, 1, 0, 0], [1, 3, 10, 3], [1, 2, 1, 1]),
                (1, 10, 3),
            ),
        ),
        (
            (None, -1, Ellipsis, None),
            (2, 3, 4),
            lambda x: relax.op.reshape(
                relax.op.strided_slice(x, [0, 1, 2], [1, 0, 0], [2, 3, 4], [1, 1, 1]), (1, 3, 4, 1)
            ),
        ),
    ],
)
def test_getitem(index, shape, compute):
    info = [(shape, "float32")]
    verify_model(UnaryModule(lambda x: x[index]), info, {}, make_expected(info, compute))


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
    dtype = "int32" if pytorch_op is torch.bitwise_not else "float32"
    input_info = [([1, 3, 10, 10], dtype)]

    class Unary(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm.script.ir_module
    class expected_unary:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype=dtype)) -> R.Tensor(
            (1, 3, 10, 10), dtype=dtype
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype=dtype) = relax_op(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype=dtype) = lv
                R.output(gv)
            return gv

    verify_model(Unary(), input_info, {}, expected_unary)


def test_sqrt_integer_input_fx():
    input_info = [([1, 4], "int64")]

    class SqrtIntModel(Module):
        def forward(self, input):
            return torch.sqrt(input)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(input_1: R.Tensor((1, 4), dtype="int64")) -> R.Tensor((1, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.astype(input_1, dtype="float32")
                lv1: R.Tensor((1, 4), dtype="float32") = R.sqrt(lv)
                gv: R.Tensor((1, 4), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(SqrtIntModel(), input_info, {}, expected)


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
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tensor(
            (1, 3, 10, 10), dtype="bool"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="bool") = lv
                R.output(gv)
            return gv

    verify_model(Unary(), input_info, {}, expected_unary)


@pytest.mark.parametrize("torch_op,expected", activation_cases())
def test_extended_unary_ops(torch_op, expected):
    info = [((2, 3), "float32")]
    verify_model(UnaryModule(torch_op), info, {}, make_expected(info, expected))


def test_clamp():
    input_info = [((1, 3, 10, 10), "float32")]

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


@pytest.mark.parametrize(
    "torch_op, relax_op",
    [
        (torch.logical_and, R.logical_and),
        (torch.logical_or, R.logical_or),
        (torch.logical_xor, R.logical_xor),
    ],
)
def test_logical_binary(torch_op, relax_op):
    input_info = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]

    class LogicalBinary(Module):
        def forward(self, lhs, rhs):
            return torch_op(lhs, rhs)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="bool"):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(lhs, dtype="bool")
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(rhs, dtype="bool")
                lv2: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(lv, lv1)
                gv: R.Tensor((1, 3, 10, 10), dtype="bool") = lv2
                R.output(gv)
            return gv

    verify_model(LogicalBinary(), input_info, {}, expected)


def test_pow_integer():
    input_info = [([4], "int64")]

    class Pow(Module):
        def forward(self, input):
            return input.pow(4)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(inp_0: R.Tensor((4,), dtype="int64")) -> R.Tensor((4,), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int64") = R.multiply(inp_0, inp_0)
                lv1: R.Tensor((4,), dtype="int64") = R.multiply(lv, inp_0)
                lv2: R.Tensor((4,), dtype="int64") = R.multiply(lv1, inp_0)
                gv: R.Tensor((4,), dtype="int64") = lv2
                R.output(gv)
            return gv

    verify_model(Pow(), input_info, {}, expected)


@pytest.mark.parametrize(
    "shape, layout, kwargs, size, method, coordinate_mode",
    [
        ((1, 3, 10, 10), "NCHW", {"size": (5, 5)}, (5, 5), "nearest_neighbor", "asymmetric"),
        (
            (1, 3, 10, 10),
            "NCHW",
            {"scale_factor": 2.0, "mode": "bilinear", "align_corners": False},
            (20, 20),
            "linear",
            "half_pixel",
        ),
        (
            (1, 3, 10, 10),
            "NCHW",
            {"scale_factor": (2.0, 1.0), "mode": "bicubic", "align_corners": False},
            (20, 10),
            "cubic",
            "half_pixel",
        ),
        (
            (1, 3, 4, 10, 10),
            "NCDHW",
            {"scale_factor": (2.0, 4.0, 4.0), "mode": "trilinear", "align_corners": False},
            (8, 40, 40),
            "linear",
            "half_pixel",
        ),
        (
            (1, 3, 4, 10, 10),
            "NCDHW",
            {"size": (8, 40, 40), "mode": "trilinear", "align_corners": True},
            (8, 40, 40),
            "linear",
            "align_corners",
        ),
        ((1, 10, 10, 3), "NHWC", {"size": (5, 5)}, (5, 5), "nearest_neighbor", "asymmetric"),
        (
            (1, 10, 10, 3),
            "NHWC",
            {"scale_factor": 2.0, "mode": "bilinear", "align_corners": False},
            (20, 20),
            "linear",
            "half_pixel",
        ),
        (
            (1, 4, 10, 10, 3),
            "NDHWC",
            {"scale_factor": (2.0, 4.0, 4.0), "mode": "trilinear", "align_corners": True},
            (8, 40, 40),
            "linear",
            "align_corners",
        ),
    ],
    ids=[
        "nearest",
        "scalar-scale",
        "tuple-scale-cubic",
        "trilinear-scale",
        "trilinear-size-aligned",
        "nhwc-size",
        "nhwc-scale",
        "ndhwc-scale-aligned",
    ],
)
def test_interpolate(shape, layout, kwargs, size, method, coordinate_mode):
    class Interpolate(Module):
        def forward(self, x):
            return F.interpolate(x, **kwargs)

    resize = R.image.resize3d if len(shape) == 5 else R.image.resize2d
    rounding_method = "" if len(shape) == 5 else "round"

    x = relax.Var("x", relax.TensorType(shape, "float32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [x]):
        with builder.dataflow():
            resized = builder.emit(
                resize(
                    x,
                    size,
                    layout=layout,
                    method=method,
                    coordinate_transformation_mode=coordinate_mode,
                    rounding_method=rounding_method,
                    cubic_alpha=-0.75,
                )
            )
            output = builder.emit_output(resized)
        builder.emit_func_output(output)

    verify_model(
        Interpolate(), [(shape, "float32")], {}, builder.get(), default_image_layout=layout
    )


def test_addmm():
    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]

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
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
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
                ) = R.split(input_1, indices_or_sections=[1, 2], axis=1)
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
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
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
    input_info = [([2, 3, 4, 5], "float32")]

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 4, 5), dtype="float32"),
            R.Tensor((2, 4, 5), dtype="float32"),
            R.Tensor((2, 4, 5), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((2, 1, 4, 5), dtype="float32"),
                    R.Tensor((2, 1, 4, 5), dtype="float32"),
                    R.Tensor((2, 1, 4, 5), dtype="float32"),
                ) = R.split(input_1, indices_or_sections=3, axis=1)
                lv1: R.Tensor((2, 1, 4, 5), dtype="float32") = lv[0]
                lv2: R.Tensor((2, 4, 5), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv3: R.Tensor((2, 1, 4, 5), dtype="float32") = lv[1]
                lv4: R.Tensor((2, 4, 5), dtype="float32") = R.squeeze(lv3, axis=[1])
                lv5: R.Tensor((2, 1, 4, 5), dtype="float32") = lv[2]
                lv6: R.Tensor((2, 4, 5), dtype="float32") = R.squeeze(lv5, axis=[1])
                lv7: R.Tuple(
                    R.Tensor((2, 4, 5), dtype="float32"),
                    R.Tensor((2, 4, 5), dtype="float32"),
                    R.Tensor((2, 4, 5), dtype="float32"),
                ) = (lv2, lv4, lv6)
                gv: R.Tuple(
                    R.Tensor((2, 4, 5), dtype="float32"),
                    R.Tensor((2, 4, 5), dtype="float32"),
                    R.Tensor((2, 4, 5), dtype="float32"),
                ) = lv7
                R.output(gv)
            return gv

    verify_model(Unbind2(), input_info, {}, expected2)


def test_cumsum():
    input_info = [([1, 2, 3, 4], "float32")]

    class Cumsum(Module):
        def forward(self, input):
            return torch.cumsum(input, dim=1, dtype=torch.int32)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor(
            (1, 2, 3, 4), dtype="int32"
        ):
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
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
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


def test_masked_fill_inplace():
    class Masked_Fill_Inplace(Module):
        def forward(self, input: torch.Tensor, mask: torch.Tensor):
            input.masked_fill_(mask, 1.5)
            return input

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((10, 10), dtype="float32"), mask: R.Tensor((10, 10), dtype="bool")
        ) -> R.Tensor((10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.full_like(
                    input, R.const(1.5, "float32")
                )
                lv1: R.Tensor((10, 10), dtype="float32") = R.where(mask, lv, input)
                gv: R.Tensor((10, 10), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = [((10, 10), "float32"), ((10, 10), "bool")]
    verify_model(Masked_Fill_Inplace(), input_info, {}, Expected)


@pytest.mark.parametrize(
    "torch_dtype, expected_dtype",
    [(torch.float32, "float32"), (None, "int64")],
    ids=["float32", "default-int64"],
)
def test_get_attr_scalar_tensor_constant(torch_dtype, expected_dtype):
    """Import rank-0 tensor constants folded to get_attr by FX tracing."""

    class ScalarTensor(Module):
        def forward(self, input):
            if torch_dtype is None:
                return torch.tensor(3)
            return torch.tensor(3, dtype=torch_dtype)

    graph_model = fx.symbolic_trace(ScalarTensor())
    mod = from_fx(graph_model, [([10, 10], "float32")])
    bindings = mod["main"].body.blocks[0].bindings
    assert len(bindings) == 1
    value = bindings[0].value
    assert isinstance(value, relax.Constant)
    assert value.data.shape == ()
    assert value.data.dtype == expected_dtype

    assert any(value.shape == () and value.item() == 3 for value in constants(mod))


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


def test_new_zeros():
    input_info = [([1, 128, 128], "float32")]

    class NewZeros(Module):
        def forward(self, x):
            return x.new_zeros(1, 128, 128)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 128, 128), dtype="float32")) -> R.Tensor(
            (1, 128, 128), dtype="float32"
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 128, 128), dtype="float32") = R.full(
                    (1, 128, 128), R.const(0.0, "float32"), dtype="float32"
                )
                gv: R.Tensor((1, 128, 128), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(NewZeros(), input_info, {}, expected)


def test_expand():
    input_info = [([1, 2, 3, 4], "float32")]

    class Expand2(Module):
        def forward(self, x):
            return x.expand(4, -1, -1, 4)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor(
            (4, 2, 3, 4), dtype="float32"
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 2, 3, 4), dtype="float32") = R.broadcast_to(x, (4, 2, 3, 4))
                gv: R.Tensor((4, 2, 3, 4), dtype="float32") = lv
                R.output(gv)
            return gv

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
        def main(inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor(
            (1, 4), dtype="float32"
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.sum(inp_0, axis=[2, 1], keepdims=False)
                gv: R.Tensor((1, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Sum(), input_info, {}, expected1)


@pytest.mark.parametrize(
    "op,dtype",
    [
        (lambda x: x.float(), "float32"),
        (lambda x: x.half(), "float16"),
        (lambda x: x.type(torch.float64), "float64"),
    ],
)
def test_datatype(op, dtype):
    info = [((2, 3), "int32")]
    expected = make_expected(info, lambda x: relax.op.astype(x, dtype))
    verify_model(UnaryModule(op), info, {}, expected)


def test_meshgrid():
    input_infos = [
        (
            [
                3,
            ],
            "float32",
        ),
        (
            [
                3,
            ],
            "float32",
        ),
    ]

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
            inp_0: R.Tensor((3,), dtype="float32"), inp_1: R.Tensor((3,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = R.meshgrid((inp_0, inp_1), indexing="ij")
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            inp_0: R.Tensor((3,), dtype="float32"), inp_1: R.Tensor((3,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = R.meshgrid((inp_0, inp_1), indexing="xy")
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv
                R.output(gv)
            return gv

    verify_model(Meshgrid1(), input_infos, {}, expected1)
    verify_model(Meshgrid2(), input_infos, {}, expected2)


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
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor(
            (1, 4, 3, 2), dtype="float32"
        ):
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


@pytest.mark.parametrize(
    "op,repeats",
    [
        (lambda x: x.tile((2,)), [2]),
        (lambda x: x.tile(4, 2), [4, 2]),
        (lambda x: torch.tile(x, (4, 2)), [4, 2]),
    ],
)
def test_tile(op, repeats):
    info = [((1, 3), "float32")]
    verify_model(
        UnaryModule(op), info, {}, make_expected(info, lambda x: relax.op.tile(x, repeats))
    )


def test_transpose():
    input_info = [([1, 2, 3, 4], "float32")]

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor(
            (1, 4, 3, 2), dtype="float32"
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tensor((1, 4, 3, 2), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Transpose(), input_info, {}, expected1)


@pytest.mark.parametrize("repeats", [(2,), (4, 2)])
def test_repeat(repeats):
    info = [((3,), "float32")]
    model = UnaryModule(lambda x: x.repeat(*repeats))
    verify_model(model, info, {}, make_expected(info, lambda x: relax.op.tile(x, repeats)))


@pytest.mark.parametrize("multi_axis", [False, True])
def test_roll(multi_axis):
    shape = (4, 3)
    model = UnaryModule(
        lambda x: torch.roll(x, (-1, 1), (0, 1)) if multi_axis else torch.roll(x, 1)
    )

    def expected(x):
        def take_roll(x, size, offset, axis):
            builder = relax.BlockBuilder.current()
            if not isinstance(x, relax.Var):
                x = builder.emit(x)
            head = builder.emit(relax.op.strided_slice(x, [axis], [0], [offset], [1]))
            tail = builder.emit(relax.op.strided_slice(x, [axis], [offset], [size], [1]))
            return relax.op.concat((tail, head), axis=axis)

        if multi_axis:
            return take_roll(take_roll(x, 4, 1, 0), 3, 2, 1)
        return relax.op.reshape(take_roll(relax.op.reshape(x, (12,)), 12, 11, 0), shape)

    info = [(shape, "float32")]
    expected = make_expected(info, expected)
    verify_model(model, info, {}, expected)


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
    for param_var, param_tensor in zip(func.params[1:], params):
        assert tuple(x.value for x in param_var.ty.shape.values) == param_tensor.shape
        assert param_var.ty.dtype == param_tensor.dtype

    tvm.testing.assert_allclose(params[0].numpy(), model.conv.bias.detach().detach().numpy())
    tvm.testing.assert_allclose(params[1].numpy(), model.conv.weight.detach().detach().numpy())


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

    graph_model = fx.symbolic_trace(Identity())
    mod = from_fx(graph_model, [([256, 256], "float32")], unwrap_unit_return_tuple=True)
    tvm.ir.assert_structural_equal(mod, Expected)


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

    graph_model = fx.symbolic_trace(Identity())
    mod = from_fx(
        graph_model, [([256, 256], "float32"), ([256, 256], "float32")], no_bind_return_tuple=True
    )
    tvm.ir.assert_structural_equal(mod, Expected)


@pytest.mark.parametrize("op,dim,keepdim", [(torch.argmax, 1, True), (torch.argmin, None, False)])
def test_arg_reduce(op, dim, keepdim):
    shape = (2, 3)
    model = UnaryModule(lambda x: op(x, dim=dim, keepdim=keepdim))
    info = [(shape, "float32")]
    relax_op = relax.op.argmax if op is torch.argmax else relax.op.argmin
    expected = make_expected(info, lambda x: relax_op(x, axis=dim, keepdims=keepdim))
    verify_model(model, info, {}, expected)


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
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor(
            (256, 256), dtype="float16"
        ):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float16") = R.astype(inp_0, dtype="float16")
                gv: R.Tensor((256, 256), dtype="float16") = lv
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor(
            (256, 256), dtype="float32"
        ):
            with R.dataflow():
                gv: R.Tensor((256, 256), dtype="float32") = inp_0
                R.output(gv)
            return gv

    verify_model(To1(), [([256, 256], "float32")], {}, Expected1)
    verify_model(To2(), [([256, 256], "float32")], {}, Expected2)


@pytest.mark.parametrize("dim", [None, 1])
def test_mean(dim):
    shape = (2, 3)
    dtype = torch.float64 if dim is not None else None
    model = UnaryModule(lambda x: torch.mean(x, dim=dim, keepdim=dim is not None, dtype=dtype))
    info = [(shape, "float32")]
    expected = make_expected(
        info,
        lambda x: relax.op.mean(
            relax.op.astype(x, "float64") if dtype is not None else x,
            axis=dim,
            keepdims=dim is not None,
        ),
    )
    verify_model(model, info, {}, expected)


def test_cat():
    class Cat0(Module):
        def forward(self, x, y):
            return torch.cat((x, y))

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


def test_atan2():
    class Atan2(Module):
        def forward(self, x, y):
            return torch.atan2(x, y)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32"),
            inp_1: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tensor((256, 256), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((256, 256), dtype="float32") = R.atan2(inp_0, inp_1)
                gv: R.Tensor((256, 256), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Atan2(), [([256, 256], "float32"), ([256, 256], "float32")], {}, Expected1)


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
                lv: R.Tensor((3, 1, 3, 10, 10), dtype="float32") = R.stack(
                    (inp_0, inp_1, inp_2), axis=0
                )
                gv: R.Tensor((3, 1, 3, 10, 10), dtype="float32") = lv
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


@pytest.mark.parametrize("start,end,step", [(1, 7, 2), (0, -2, 1)])
def test_slice_scatter(start, end, step):
    class Scatter(Module):
        def forward(self, x, src):
            return torch.slice_scatter(x, src, dim=1, start=start, end=end, step=step)

    stop = end if end > 0 else 8 + end
    info = [((2, 8), "float32"), ((2, len(range(start, stop, step))), "float32")]
    expected = make_expected(
        info, lambda x, src: relax.op.slice_scatter(x, src, start, stop, step, axis=1)
    )
    verify_model(Scatter(), info, {}, expected)


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

    verify_model(Gather0(), [([2, 3], "float32"), ([2, 3], "int32")], {}, Expected0)


def test_index_put():
    class IndexPut(Module):
        def forward(self, data, indices, values):
            return data.index_put_((indices,), values, accumulate=False)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((64,), dtype="float32"),
            indices: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tensor((64,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((64,), dtype="float32") = R.index_put(
                    data, R.tuple(indices), values, accumulate=False
                )
                gv: R.Tensor((64,), dtype="float32") = lv
                R.output(gv)
            return gv

    input_info = [((64,), "float32"), ((128,), "int64"), ((128,), "float32")]
    verify_model(IndexPut(), input_info, {}, Expected)


def test_flip():
    shape = (2, 3, 4)
    model = UnaryModule(lambda x: torch.flip(x, dims=(0, -1)))
    info = [(shape, "float32")]
    expected = make_expected(info, lambda x: relax.op.flip(relax.op.flip(x, axis=0), axis=-1))
    verify_model(model, info, {}, expected)


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
        ) -> R.Tensor((3,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int32") = R.astype(inp_1, "int32")
                lv1: R.Tensor((3,), dtype="float32") = R.take(inp_0, lv)
                gv: R.Tensor((3,), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_model(Take(), [([5], "float32"), ([3], "int64")], {}, Expected)


def test_one_hot():
    class OneHot(Module):
        def forward(self, indices):
            return torch.nn.functional.one_hot(indices, num_classes=10)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="int64"),
        ) -> R.Tensor((5, 10), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((5, 10), dtype="int64") = R.one_hot(
                    inp_0, R.prim_value(1), R.prim_value(0), depth=10, axis=-1
                )
                gv: R.Tensor((5, 10), dtype="int64") = lv
                R.output(gv)

            return gv

    verify_model(OneHot(), [([5], "int64")], {}, Expected)


def test_one_hot_invalid_num_classes():
    input_info = [([5], "int64")]

    class OneHot(Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, indices):
            return torch.nn.functional.one_hot(indices, num_classes=self.num_classes)

    # Zero is invalid; -1 requires data-dependent depth inference, unsupported here.
    for num_classes in (0, -1):
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            from_fx(fx.symbolic_trace(OneHot(num_classes)), input_info)


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


def test_ones_like():
    class OnesLike(Module):
        def forward(self, data):
            return torch.ones_like(data)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((128, 128), dtype="float32")) -> R.Tensor(
            (128, 128), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.ones_like(inp_0)
                gv: R.Tensor((128, 128), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(OnesLike(), [([128, 128], "float32")], {}, Expected)


def test_zero_inplace():
    class ZeroInplace(Module):
        def forward(self, data):
            return data.zero_()

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((128, 128), dtype="float32")) -> R.Tensor(
            (128, 128), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.zeros_like(inp_0)
                gv: R.Tensor((128, 128), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(ZeroInplace(), [([128, 128], "float32")], {}, Expected)


def test_zeros_like():
    class ZerosLike(Module):
        def forward(self, data):
            return torch.zeros_like(data)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((128, 128), dtype="float32")) -> R.Tensor(
            (128, 128), dtype="float32"
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.zeros_like(inp_0)
                gv: R.Tensor((128, 128), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(ZerosLike(), [([128, 128], "float32")], {}, Expected)


def test_type_as():
    class TypeAs(Module):
        def forward(self, data, other):
            return data.type_as(other)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((128, 128), dtype="float16"),
            inp_1: R.Tensor((128, 128), dtype="float32"),
        ) -> R.Tensor((128, 128), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.astype(inp_0, dtype="float32")
                gv: R.Tensor((128, 128), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(TypeAs(), [([128, 128], "float16"), ([128, 128], "float32")], {}, Expected)


def test_item():
    class Item(Module):
        def forward(self, data):
            return data.item()

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((1,), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.take(inp_0, R.const(0, "int64"), axis=0)
                gv: R.Tensor((), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(
        Item(),
        [
            (
                [1],
                "float32",
            )
        ],
        {},
        Expected,
    )


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


def test_inplace_copy():
    class CopyBroadcast(Module):
        def forward(self, x, src):
            x.copy_(src)
            return x

    info = [((2, 3), "float32"), ((), "int64")]
    expected = make_expected(
        info, lambda x, src: relax.op.broadcast_to(relax.op.astype(src, "float32"), (2, 3))
    )
    verify_model(CopyBroadcast(), info, {}, expected)


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


@pytest.mark.parametrize("op,correction", [(torch.std, 1), (torch.var, 0)])
def test_statistics(op, correction):
    info = [((2, 3), "float32")]

    def expected(x):
        var = relax.op.variance(x, axis=1, keepdims=True)
        if correction:
            var = relax.op.multiply(var, relax.const(1.5, "float32"))
        return relax.op.sqrt(var) if op is torch.std else var

    model = UnaryModule(lambda x: op(x, 1, bool(correction), True))
    verify_model(model, info, {}, make_expected(info, expected))


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
            inp_0: R.Tensor((5, 3), dtype=relax_dtype),
        ) -> R.Tensor((), dtype=relax_dtype):
            with R.dataflow():
                lv: R.Tensor((), dtype=relax_dtype) = R.prod(inp_0, axis=None, keepdims=False)
                gv: R.Tensor((), dtype=relax_dtype) = lv
                R.output(gv)
            return gv

    verify_model(Prod(), [([5, 3], relax_dtype)], {}, Expected)


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


def test_bucketize():
    class Bucketize(Module):
        def forward(self, input_tensor, boundaries):
            return torch.bucketize(input_tensor, boundaries)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((5, 3), dtype="float32"), boundaries: R.Tensor((10,), dtype="float32")
        ) -> R.Tensor((5, 3), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int64") = R.bucketize(
                    input, boundaries, out_int32=False, right=False
                )
                gv: R.Tensor((5, 3), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_model(Bucketize(), [([5, 3], "float32"), ([10], "float32")], {}, Expected)


def test_argsort():
    class Argsort(Module):
        def forward(self, x):
            return torch.argsort(x, dim=1, descending=True)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tensor((5, 3), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int64") = R.argsort(
                    inp_0, axis=1, descending=True, dtype="int64"
                )
                gv: R.Tensor((5, 3), dtype="int64") = lv
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
        def main(inp_0: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int64") = R.argsort(
                    inp_0, axis=1, descending=True, dtype="int64"
                )
                lv1: R.Tensor((5, 3), dtype="float32") = R.gather_elements(inp_0, lv, axis=1)
                lv2: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")) = (
                    lv1,
                    lv,
                )
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")) = (
                    lv2
                )
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
                lv: R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")) = (
                    R.topk(inp_0, k=2, axis=1, ret_type="both", largest=True, dtype="int64")
                )
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


@pytest.mark.parametrize("p", [float("inf"), float("-inf"), 0.5, "fro"])
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
    expected = make_expected(info, expected)
    verify_model(model, info, {}, expected)


@pytest.mark.parametrize(
    "torch_dtype, relax_dtype",
    [
        # Float types
        (torch.float16, "float16"),
        (torch.float32, "float32"),
        (torch.float64, "float64"),
        (torch.bfloat16, "bfloat16"),
        # Signed integer types
        (torch.int8, "int8"),
        (torch.int16, "int16"),
        (torch.int32, "int32"),
        (torch.int64, "int64"),
        # Unsigned integer types
        (torch.uint8, "uint8"),
        (torch.uint16, "uint16"),
        (torch.uint32, "uint32"),
        (torch.uint64, "uint64"),
        # Boolean
        (torch.bool, "bool"),
    ],
)
def test_dtypes(torch_dtype, relax_dtype):
    class Model(Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor):
            return torch.ops.aten.add(lhs, rhs)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype=relax_dtype),
            rhs: R.Tensor((10, 10), dtype=relax_dtype),
        ) -> R.Tensor((10, 10), dtype=relax_dtype):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype=relax_dtype) = relax.op.add(lhs, rhs)
                gv: R.Tensor((10, 10), dtype=relax_dtype) = lv
                R.output(gv)
            return gv

    verify_model(Model(), [([10, 10], torch_dtype), ([10, 10], torch_dtype)], {}, Expected)


def test_round():
    input_info = [([3, 4], "float32")]

    class Round(Module):
        def __init__(self, decimals=0):
            super().__init__()
            self.decimals = decimals

        def forward(self, x):
            if self.decimals == 0:
                return torch.round(x)
            else:
                return torch.round(x, decimals=self.decimals)

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.round(inp_0)
                gv: R.Tensor((3, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.multiply(inp_0, R.const(100.0, "float32"))
                lv1: R.Tensor((3, 4), dtype="float32") = R.round(lv)
                lv2: R.Tensor((3, 4), dtype="float32") = R.divide(lv1, R.const(100.0, "float32"))
                gv: R.Tensor((3, 4), dtype="float32") = lv2
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedNegative:
        @R.function
        def main(x: R.Tensor((3, 4), "float32")):
            with R.dataflow():
                scaled = R.divide(x, R.const(10.0, "float32"))
                rounded = R.round(scaled)
                result = R.multiply(rounded, R.const(10.0, "float32"))
                output = result
                R.output(output)
            return output

    rounds = [
        (0, Expected1),
        (2, Expected2),
        (-1, ExpectedNegative),
    ]

    for decimals, expected in rounds:
        verify_model(Round(decimals), input_info, {}, expected)


@pytest.mark.parametrize("ndim,dtype", [(2, "float32"), (3, "float64")])
def test_pool_divisor_override(ndim, dtype):
    model = PoolDivisorModel(ndim)
    shape = (2, 2) + (6,) * ndim
    args = (torch.linspace(-3, 4, int(np.prod(shape)), dtype=getattr(torch, dtype)).reshape(shape),)
    mod = from_fx(fx.symbolic_trace(model), [(shape, dtype)])
    verify_numerically(mod, model, args, rtol=1e-6, atol=1e-6)


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
                torch.norm(x, p=0.5, dim=1, keepdim=True),
                x[-1, None, :],
                torch.select(x, 0, -1),
            )

    model = Model()
    args = (torch.tensor([[-1.0, 1.0, 3.0], [2.0, -2.0, 4.0]]),)
    mod = from_fx(fx.symbolic_trace(model), [((2, 3), "float32")])
    verify_numerically(mod, model, args, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint8"])
def test_interpolate_antialiased(dtype):
    model = AntialiasedResizeModel()
    x = ((torch.arange(112).reshape(1, 2, 7, 8) * 53) % 251).to(getattr(torch, dtype))
    if dtype != "uint8":
        x /= 251
    tolerance = 1e-6 if dtype == "float32" else 1e-10 if dtype == "float64" else 0
    shape = (
        (1, 2, tvm.tirx.Var("height", "int64"), tvm.tirx.Var("width", "int64"))
        if dtype == "float32"
        else x.shape
    )
    inputs = [(x,), (torch.rand(1, 2, 11, 10),)] if dtype == "float32" else None
    mod = from_fx(fx.symbolic_trace(model), [(shape, dtype)])
    verify_numerically(mod, model, (x,), rtol=tolerance, atol=tolerance, input_sets=inputs)


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16"])
def test_exponential(dtype):
    model = ExponentialModel()
    mod = from_fx(fx.symbolic_trace(model), [((32768,), dtype)])
    verify_exponential(mod, dtype)


@pytest.mark.parametrize(
    "op,message",
    [
        (lambda x: F.avg_pool2d(x, 2, divisor_override=0), "divisor_override"),
        (lambda x: x.exponential_(0), "lambda > 0"),
    ],
)
def test_invalid_pool_divisor_and_exponential_rate(op, message):
    with pytest.raises(ValueError, match=message):
        from_fx(fx.symbolic_trace(UnaryModule(op)), [((1, 1, 4, 4), "float32")])


if __name__ == "__main__":
    tvm.testing.main()
