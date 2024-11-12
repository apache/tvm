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

""" Test translate from torch. """

import torch
from torch.nn import Module

import tvm.testing
from tvm.contrib.msc.framework.torch.frontend import translate
from tvm.contrib.msc.framework.torch import codegen
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


def verify_model(torch_model, input_info, via_relax=True):
    """Compare torch module results"""

    torch_datas = [msc_utils.random_data(i, MSCFramework.TORCH) for i in input_info]
    with torch.no_grad():
        golden = torch_model(*torch_datas)
    graph, weights = translate.from_torch(torch_model, input_info, via_relax=via_relax)
    model = codegen.to_torch(graph, weights)
    with torch.no_grad():
        if not graph.get_inputs():
            result = model()
        else:
            result = model(*torch_datas)
    if not isinstance(golden, (list, tuple)):
        golden = [golden]
    if not isinstance(result, (list, tuple)):
        result = [result]
    assert len(golden) == len(result), "golden {} mismatch with result {}".format(
        len(golden), len(result)
    )
    for gol_r, new_r in zip(golden, result):
        if isinstance(gol_r, torch.Tensor):
            tvm.testing.assert_allclose(
                gol_r.detach().numpy(), new_r.detach().numpy(), atol=1e-5, rtol=1e-5
            )
        else:
            assert gol_r == new_r


def test_conv1d():
    """test torch translator for conv1d"""

    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)

        def forward(self, data):
            return self.conv(data)

    input_info = [([1, 3, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Conv1D1(), input_info, via_relax)
        verify_model(Conv1D2(), input_info, via_relax)


def test_conv2d():
    """test torch translator for conv2d"""

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, data):
            return self.conv(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Conv2D1(), input_info, via_relax)
        verify_model(Conv2D2(), input_info, via_relax)


def test_linear():
    """test torch translator for linear"""

    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, data):
            return self.linear(data)

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, data):
            return self.linear(data)

    class MatMul1(Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Dense1(), input_info, via_relax)
        verify_model(Dense2(), input_info, via_relax)
        verify_model(MatMul1(), [([10, 10], "float32"), ([10, 10], "float32")], via_relax)


def test_bmm():
    """test torch translator for bmm"""

    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    input_info = [((4, 128, 256), "float32"), ((4, 256, 512), "float32")]
    for via_relax in [True, False]:
        verify_model(BMM(), input_info, via_relax)


def test_baddbmm():
    """test torch translator for baddbmm"""

    class BAddBMM1(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    class BAddBMM2(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    input_info = [
        ((4, 128, 512), "float32"),
        ((4, 128, 256), "float32"),
        ((4, 256, 512), "float32"),
    ]
    for via_relax in [True, False]:
        verify_model(BAddBMM1(), input_info, via_relax)
        verify_model(BAddBMM2(), input_info, via_relax)


def test_relu():
    """test torch translator for relu"""

    class ReLU(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, data):
            return self.relu(data)

    class ReLU1(Module):
        def forward(self, data):
            return torch.nn.functional.relu(data)

    input_info = [([10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(ReLU(), input_info, via_relax)
        verify_model(ReLU1(), input_info, via_relax)


def test_relu6():
    """test torch translator for relu6"""

    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, data):
            return self.relu6(data)

    input_info = [([10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(ReLU6(), input_info, via_relax)


def test_maxpool2d():
    """test torch translator for maxpool2d"""

    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, data):
            return self.pool(data)

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, data):
            return self.pool(data)

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, data):
            return self.pool(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(MaxPool2d(), input_info, via_relax)
        verify_model(MaxPool2d2(), input_info, via_relax)
        verify_model(MaxPool2d3(), input_info, via_relax)


def test_avgpool2d():
    """test torch translator for avgpool2d"""

    class AvgPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[1, 1])

        def forward(self, data):
            return self.pool(data)

    class AvgPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True)

        def forward(self, data):
            return self.pool(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(AvgPool2d(), input_info, via_relax)
        verify_model(AvgPool2d2(), input_info, via_relax)


def test_adaptive_avgpool2d():
    """test torch translator for adaptive_avgpool2d"""

    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, data):
            return self.pool(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(AdaptiveAvgPool2d0(), input_info, via_relax)


def test_flatten():
    """test torch translator for flatten"""

    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, data):
            return self.f(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Flatten(), input_info, via_relax)
        verify_model(torch.nn.Flatten(2, -1), input_info, via_relax)


def test_batchnorm2d():
    """test torch translator for batchnorm2d"""

    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = torch.nn.BatchNorm2d(3)

        def forward(self, data):
            return self.batchnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(BatchNorm2d(), input_info, via_relax)


def test_embedding():
    """test torch translator for embedding"""

    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, data):
            return self.embedding(data)

    for via_relax in [True, False]:
        verify_model(Embedding(), [([4], "int64")], via_relax)
        verify_model(Embedding(), [([4, 5], "int64")], via_relax)


def test_layernorm():
    """test torch translator for layernorm"""

    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((10, 10))

        def forward(self, data):
            return self.layernorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(LayerNorm(), input_info)


def test_cross_entropy():
    """test torch translator for cross_entropy"""

    class CrossEntropy1(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    class CrossEntropy2(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2,)))
            self.loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    class CrossEntropy3(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=1, reduction="sum")

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    input_info = [([3, 2], "float32"), ([3], "int64")]
    for via_relax in [True, False]:
        verify_model(CrossEntropy1(), input_info, via_relax)
        verify_model(CrossEntropy2(), input_info, via_relax)
        verify_model(CrossEntropy3(), input_info, via_relax)


def test_silu():
    """test torch translator for silu"""

    class SiLU(Module):
        def __init__(self):
            super().__init__()
            self.silu = torch.nn.SiLU()

        def forward(self, data):
            return self.silu(data)

    class SiLU2(Module):
        def forward(self, data):
            return torch.nn.functional.silu(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(SiLU(), input_info, via_relax)
        verify_model(SiLU2(), input_info, via_relax)


def test_groupnorm():
    """test torch translator for groupnorm"""

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(3, 3)

        def forward(self, data):
            return self.groupnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(GroupNorm(), input_info, via_relax)


def test_softmax():
    """test torch translator for softmax"""

    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, data):
            return self.softmax(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Softmax(), input_info, via_relax)


def test_binary():
    """test torch translator for binary"""

    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    for via_relax in [True, False]:
        verify_model(Add1(), input_info1, via_relax)
        verify_model(Add2(), input_info2, via_relax)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    for via_relax in [True, False]:
        verify_model(Sub1(), input_info1, via_relax)
        verify_model(Sub2(), input_info2, via_relax)

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    for via_relax in [True, False]:
        verify_model(Mul1(), input_info1, via_relax)
        verify_model(Mul2(), input_info2, via_relax)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    for via_relax in [True, False]:
        verify_model(TrueDiv1(), input_info1, via_relax)
        verify_model(TrueDiv2(), input_info2, via_relax)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    for via_relax in [True, False]:
        verify_model(FloorDiv1(), input_info1, via_relax)
        verify_model(FloorDiv2(), input_info2, via_relax)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    for via_relax in [True, False]:
        verify_model(Power1(), input_info1, via_relax)
        verify_model(Power2(), input_info2, via_relax)

    # LT
    class LT1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    class LT2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    for via_relax in [True, False]:
        verify_model(LT1(), input_info1, via_relax)
        verify_model(LT2(), input_info2, via_relax)


def test_size():
    """test torch translator for size"""

    class Size(Module):
        def forward(self, data):
            return data.size()

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Size(), input_info)


def test_squeeze():
    """test torch translator for squeeze"""

    class Squeeze1(Module):
        def forward(self, data):
            return data.squeeze(1)

    class Squeeze2(Module):
        def forward(self, data):
            return data.squeeze()

    input_info = [([3, 1, 4, 1], "float32")]
    for via_relax in [True, False]:
        verify_model(Squeeze1(), input_info, via_relax)
        verify_model(Squeeze2(), input_info, via_relax)


def test_unsqueeze():
    """test torch translator for unsqueeze"""

    class Unsqueeze1(Module):
        def forward(self, data):
            return data.unsqueeze(1)

    class Unsqueeze2(Module):
        def forward(self, data):
            return data.unsqueeze(-1)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Unsqueeze1(), input_info, via_relax)
        verify_model(Unsqueeze2(), input_info, via_relax)


def test_getattr():
    """test torch translator for getattr"""

    class GetAttr1(Module):
        def forward(self, data):
            return data.shape

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(GetAttr1(), input_info)


def test_getitem():
    """test torch translator for getitem"""

    # TODO(tong.meng): strided_slice reshape bug for x[0, 1::2, :, :3]
    class Slice1(Module):
        def forward(self, x):
            return x[0:1, 1::2, :, :3]

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    for via_relax in [True, False]:
        verify_model(Slice1(), [([1, 3, 10, 10], "float32")], via_relax)
        verify_model(Slice2(), [([8, 16], "float32")], via_relax)


def test_unary():
    """test torch translator for unary"""

    input_info = [([1, 3, 10, 10], "float32")]

    # sin
    class Sin(Module):
        def forward(self, data):
            return torch.sin(data)

    for via_relax in [True, False]:
        verify_model(Sin(), input_info, via_relax)

    # cos
    class Cos(Module):
        def forward(self, data):
            return torch.cos(data)

    for via_relax in [True, False]:
        verify_model(Cos(), input_info, via_relax)

    # exp
    class Exp(Module):
        def forward(self, data):
            return torch.exp(data)

    for via_relax in [True, False]:
        verify_model(Exp(), input_info, via_relax)

    # sqrt
    class Sqrt(Module):
        def forward(self, data):
            return torch.sqrt(data)

    for via_relax in [True, False]:
        verify_model(Sqrt(), input_info, via_relax)

    # sigmoid
    class Sigmoid(Module):
        def forward(self, data):
            return torch.sigmoid(data)

    for via_relax in [True, False]:
        verify_model(Sigmoid(), input_info, via_relax)

    # round
    class Round(Module):
        def forward(self, data):
            return torch.round(data)

    for via_relax in [True, False]:
        verify_model(Round(), input_info, via_relax)


def test_gelu():
    """test torch translator for gelu"""

    class Gelu(Module):
        def forward(self, data):
            return torch.nn.functional.gelu(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Gelu(), input_info, via_relax)


def test_tanh():
    """test torch translator for tanh"""

    class Tanh(Module):
        def forward(self, data):
            return torch.tanh(data)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Tanh(), input_info, via_relax)


def test_clamp():
    """test torch translator for clamp"""

    class Clamp(Module):
        def forward(self, data):
            return torch.clamp(data, min=0.1, max=0.5)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Clamp(), input_info, via_relax)


def test_interpolate():
    """test torch translator for interpolate"""

    class Interpolate(Module):
        def forward(self, data):
            return torch.nn.functional.interpolate(data, (5, 5))

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Interpolate(), input_info, via_relax)


def test_addmm():
    """test torch translator for addmm"""

    class Addmm(Module):
        def forward(self, x_1, x_2, x_3):
            return torch.addmm(x_1, x_2, x_3)

    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]
    for via_relax in [True, False]:
        verify_model(Addmm(), input_info, via_relax)


def test_split():
    """test torch translator for split"""

    class Split1(Module):
        def forward(self, data):
            return torch.split(data, 1, dim=1)

    class Split2(Module):
        def forward(self, data):
            return torch.split(data, [1, 2], dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Split1(), input_info, via_relax)
        verify_model(Split2(), input_info, via_relax)


def test_unbind():
    """test torch translator for unbind"""

    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    input_info = [([3, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Unbind1(), input_info, via_relax)
        verify_model(Unbind2(), input_info, via_relax)


def test_cumsum():
    """test torch translator for cumsum"""

    class Cumsum(Module):
        def forward(self, data):
            return torch.cumsum(data, dim=1, dtype=torch.int32)

    input_info = [([1, 2, 3, 4], "float32")]
    for via_relax in [True, False]:
        verify_model(Cumsum(), input_info, via_relax)


def test_chunk():
    """test torch translator for chunk"""

    class Chunk(Module):
        def forward(self, data):
            return torch.chunk(data, 3, dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Chunk(), input_info, via_relax)


def test_inplace_fill():
    """test torch translator for inplace_fill"""

    class InplaceFill(Module):
        def forward(self, data):
            data.fill_(1.5)
            return data

    for via_relax in [True, False]:
        verify_model(InplaceFill(), [([10, 10], "float32")], via_relax)


def test_arange():
    """test torch translator for arange"""

    # pylint: disable=unused-argument
    class Arange(Module):
        def forward(self, data):
            return torch.arange(0, 20, dtype=torch.int32)

    verify_model(Arange(), [([10, 10], "float32")])


def test_tril():
    """test torch translator for tril"""

    class Tril(Module):
        def forward(self, data):
            return torch.tril(data, 1)

    class InplaceTril(Module):
        def forward(self, data):
            data.tril_(1)
            return data

    input_info = [([10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Tril(), input_info, via_relax)
        verify_model(InplaceTril(), input_info, via_relax)


def test_triu():
    """test torch translator for triu"""

    class Triu(Module):
        def forward(self, data):
            return torch.triu(data, 1)

    class InplaceTriu(Module):
        def forward(self, data):
            data.triu_(1)
            return data

    input_info = [([10, 10], "float32")]
    for via_relax in [True, False]:
        verify_model(Triu(), input_info, via_relax)
        verify_model(InplaceTriu(), input_info, via_relax)


def test_new_ones():
    """test torch translator for new_ones"""

    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    input_info = [([1, 2, 3], "float32")]
    for via_relax in [True, False]:
        verify_model(NewOnes(), input_info, via_relax)


def test_expand():
    """test torch translator for expand"""

    class Expand1(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    class Expand2(Module):
        def forward(self, x):
            return x.expand(4, -1, -1, 4)

    input_info = [([1, 2, 3, 4], "float32")]
    for via_relax in [True, False]:
        verify_model(Expand1(), input_info, via_relax)
        verify_model(Expand2(), input_info, via_relax)


def test_reduce():
    """test torch translator for reduce"""

    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    # max
    class Max(Module):
        def forward(self, x):
            return torch.max(x)

    # min
    class Min(Module):
        def forward(self, x):
            return torch.min(x)

    input_info = [([1, 2, 3, 4], "float32")]
    for via_relax in [True, False]:
        verify_model(Sum(), input_info, via_relax)
    verify_model(Max(), input_info, False)
    verify_model(Min(), input_info, False)


def test_datatype():
    """test torch translator for datatype"""

    input_info = [([1, 2, 3, 4], "float32")]

    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    for via_relax in [True, False]:
        verify_model(ToFloat(), input_info, via_relax)

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    for via_relax in [True, False]:
        verify_model(ToHalf(), input_info, via_relax)

    # type
    class Type(Module):
        def forward(self, x):
            return x.type(torch.float32)

    for via_relax in [True, False]:
        verify_model(Type(), input_info, via_relax)


def test_permute():
    """test torch translator for permute"""

    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    input_info = [([1, 2, 3, 4], "float32")]
    for via_relax in [True, False]:
        verify_model(Permute(), input_info, via_relax)


def test_reshape():
    """test torch translator for reshape"""

    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    for via_relax in [True, False]:
        verify_model(Reshape(), input_info, via_relax)


def test_transpose():
    """test torch translator for transpose"""

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    input_info = [([1, 2, 3, 4], "float32")]
    for via_relax in [True, False]:
        verify_model(Transpose(), input_info, via_relax)


def test_view():
    """test torch translator for view"""

    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    for via_relax in [True, False]:
        verify_model(View(), input_info, via_relax)


def test_keep_params():
    """test torch translator for keep_params"""

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    for via_relax in [True, False]:
        verify_model(Conv2D1(), [([1, 3, 10, 10], "float32")], via_relax)


def test_unwrap_unit_return_tuple():
    """test torch translator for unwrap_unit_return_tuple"""

    class Identity(Module):
        def forward(self, x):
            return (x,)

    for via_relax in [True, False]:
        verify_model(Identity(), [([256, 256], "float32")], via_relax)


def test_no_bind_return_tuple():
    """test torch translator for no_bind_return_tuple"""

    class Identity(Module):
        def forward(self, x, y):
            return (x, y)

    input_info = [([256, 256], "float32"), ([256, 256], "float32")]
    for via_relax in [True, False]:
        verify_model(Identity(), input_info, via_relax)


def test_argmax():
    """test torch translator for argmax"""

    class Argmax1(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1)

    class Argmax2(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1, keepdim=True)

    for via_relax in [True, False]:
        verify_model(Argmax1(), [([256, 256], "float32")], via_relax)
        verify_model(Argmax2(), [([256, 256], "float32")], via_relax)


def test_argmin():
    """test torch translator for argmin"""

    class Argmin1(Module):
        def forward(self, data):
            return torch.argmin(data)

    class Argmin2(Module):
        def forward(self, data):
            return torch.argmin(data, keepdim=True)

    verify_model(Argmin1(), [([256, 256], "float32")])
    verify_model(Argmin2(), [([256, 256], "float32")])


def test_to():
    """test torch translator for to"""

    class To1(Module):
        def forward(self, data):
            return data.to(torch.float16)

    class To2(Module):
        def forward(self, data):
            return data.to("cpu")

    for via_relax in [True, False]:
        verify_model(To1(), [([256, 256], "float32")], via_relax)
        verify_model(To2(), [([256, 256], "float32")], via_relax)


def test_mean():
    """test torch translator for mean"""

    class Mean(Module):
        def forward(self, data):
            return data.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, data):
            return data.mean(-1, keepdim=True)

    for via_relax in [True, False]:
        verify_model(Mean(), [([256, 256], "float32")], via_relax)
        verify_model(MeanKeepDim(), [([256, 256], "float32")], via_relax)


def test_rsqrt():
    """test torch translator for rsqrt"""

    class Rsqrt(Module):
        def forward(self, data):
            return torch.rsqrt(data)

    for via_relax in [True, False]:
        verify_model(Rsqrt(), [([256, 256], "float32")], via_relax)


def test_neg():
    """test torch translator for neg"""

    class Neg(Module):
        def forward(self, data):
            return -data

    for via_relax in [True, False]:
        verify_model(Neg(), [([256, 256], "float32")], via_relax)


def test_max():
    """test torch translator for max"""

    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    for via_relax in [True, False]:
        verify_model(Max(), [([256, 256], "float32"), ([256, 256], "float32")], via_relax)


def test_cat():
    """test torch translator for cat"""

    class Cat1(Module):
        def forward(self, data, data1, data2):
            return torch.cat((data, data1, data2), dim=1)

    class Cat2(Module):
        def forward(self, data):
            const1 = torch.ones((1, 3, 10, 10), dtype=torch.float32)
            const2 = torch.ones((1, 3, 10, 10), dtype=torch.float32)
            return torch.cat((data, const1, const2), dim=1)

    input_info = [
        ([1, 3, 10, 10], "float32"),
        ([1, 3, 10, 10], "float32"),
        ([1, 3, 10, 10], "float32"),
    ]
    for via_relax in [True, False]:
        verify_model(Cat1(), input_info, via_relax)
        verify_model(Cat2(), [([1, 3, 10, 10], "float32")], via_relax)


def test_stack():
    """test torch translator for stack"""

    class Stack1(Module):
        def forward(self, data, data1, data2):
            return torch.stack((data, data1, data2), dim=0)

    class Stack2(Module):
        def forward(self, data):
            const1 = torch.ones((1, 3, 10, 10), dtype=torch.float32)
            const2 = torch.ones((1, 3, 10, 10), dtype=torch.float32)
            return torch.stack((data, const1, const2), dim=1)

    input_info = [
        ([1, 3, 10, 10], "float32"),
        ([1, 3, 10, 10], "float32"),
        ([1, 3, 10, 10], "float32"),
    ]
    for via_relax in [True, False]:
        verify_model(Stack1(), input_info, via_relax)
        verify_model(Stack2(), [([1, 3, 10, 10], "float32")], via_relax)


def test_scatter():
    """test torch translator for scatter"""

    class Scatter1(Module):
        def __init__(self):
            super().__init__()
            self.index = msc_utils.random_data([(2, 5), "int64"], MSCFramework.TORCH, max_val=5)

        def forward(self, data, src):
            return data.scatter(dim=0, index=self.index, src=src)

    class Scatter2(Module):
        def forward(self, data, index, src):
            return data.scatter(0, index, src)

    for via_relax in [True, False]:
        verify_model(Scatter1(), [([20, 20], "float32"), ([2, 5], "float32")], via_relax)
        verify_model(
            Scatter2(), [([20, 20], "float32"), ([2, 5], "int64"), ([2, 5], "float32")], via_relax
        )


def test_masked_scatter():
    """test torch translator for masked_scatter"""

    class MaskedScatter1(Module):
        def __init__(self):
            super().__init__()
            self.mask = msc_utils.random_data([(5,), "bool"], MSCFramework.TORCH)

        def forward(self, data, src):
            return data.masked_scatter(self.mask, src)

    class MaskedScatter2(Module):
        def __init__(self):
            super().__init__()
            self.mask = msc_utils.random_data([(2, 5), "bool"], MSCFramework.TORCH)

        def forward(self, data, src):
            return data.masked_scatter(self.mask, src)

    verify_model(MaskedScatter1(), [([5], "float32"), ([10], "float32")], True)
    verify_model(MaskedScatter2(), [([2, 5], "float32"), ([3, 5], "float32")], True)


def test_put():
    """test torch translator for index_put"""

    class IndexPut(Module):
        def __init__(self):
            super().__init__()
            self.index = msc_utils.random_data([(5), "int64"], MSCFramework.TORCH, max_val=5)

        def forward(self, data, src):
            data[self.index] = src
            return data

    input_info = [([10, 20], "float32"), ([5, 20], "float32")]
    verify_model(IndexPut(), input_info, False)


def test_attention():
    """test torch translator for attention"""

    # pylint: disable=import-outside-toplevel
    import torch.nn.functional as F

    class Attention1(Module):
        def forward(self, q_data, k_data, v_data):
            return F.scaled_dot_product_attention(q_data, k_data, v_data)

    class Attention2(Module):
        def forward(self, q_data, k_data, v_data):
            return F.scaled_dot_product_attention(q_data, k_data, v_data, is_causal=True)

    input_info = [
        ([32, 8, 128, 64], "float32"),
        ([32, 8, 128, 64], "float32"),
        ([32, 8, 128, 64], "float32"),
    ]
    verify_model(Attention1(), input_info)
    verify_model(Attention2(), input_info)

    class Attention3(Module):
        def forward(self, q_data, k_data, v_data, mask):
            return F.scaled_dot_product_attention(q_data, k_data, v_data, mask)

    verify_model(
        Attention3(),
        [
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 128], "float32"),
        ],
    )


if __name__ == "__main__":
    tvm.testing.main()
