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

""" Test translate from relax. """

import torch
from torch import fx
from torch.nn import Module

import numpy as np

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.core.frontend import translate
from tvm.contrib.msc.framework.tvm import codegen as tvm_codegen


def _verify_model(torch_model, input_info, opt_config=None):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        orig_mod = from_fx(graph_model, input_info)

    target = "llvm"
    dev = tvm.cpu()
    args = [tvm.nd.array(np.random.random(size=shape).astype(dtype)) for shape, dtype in input_info]

    def _tvm_runtime_to_np(obj):
        if isinstance(obj, tvm.runtime.NDArray):
            return obj.numpy()
        elif isinstance(obj, tvm.runtime.ShapeTuple):
            return np.array(obj, dtype="int64")
        elif isinstance(obj, (list, tvm.ir.container.Array)):
            return [_tvm_runtime_to_np(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_tvm_runtime_to_np(item) for item in obj)
        else:
            return obj

    def _run_relax(relax_mod):
        relax_mod = tvm.relax.transform.LegalizeOps()(relax_mod)
        relax_exec = tvm.relax.build(relax_mod, target)
        vm_runner = tvm.relax.VirtualMachine(relax_exec, dev)
        res = vm_runner["main"](*args)

        return _tvm_runtime_to_np(res)

    rt_mod = tvm_codegen.to_relax(
        *translate.from_relax(orig_mod, opt_config=opt_config),
        codegen_config={"explicit_name": False},
    )

    orig_output = _run_relax(orig_mod)
    rt_output = _run_relax(rt_mod)
    tvm.testing.assert_allclose(orig_output, rt_output)


def test_conv1d():
    """test relax translator for conv1d"""

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
    _verify_model(Conv1D1(), input_info)
    _verify_model(Conv1D2(), input_info)


def test_conv2d():
    """test relax translator for conv2d"""

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
    _verify_model(Conv2D1(), input_info)
    _verify_model(Conv2D2(), input_info)


def test_linear():
    """test relax translator for linear"""

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
    _verify_model(Dense1(), input_info)
    _verify_model(Dense2(), input_info)
    _verify_model(MatMul1(), [([10, 10], "float32"), ([10, 10], "float32")])


def test_bmm():
    """test relax translator for bmm"""

    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    input_info = [((4, 128, 256), "float32"), ((4, 256, 512), "float32")]
    _verify_model(BMM(), input_info)


def test_baddbmm():
    """test relax translator for baddbmm"""

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
    _verify_model(BAddBMM1(), input_info)
    _verify_model(BAddBMM2(), input_info)


def test_relu():
    """test relax translator for relu"""

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
    _verify_model(ReLU(), input_info)
    _verify_model(ReLU1(), input_info)


def test_relu6():
    """test relax translator for relu6"""

    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, data):
            return self.relu6(data)

    input_info = [([10, 10], "float32")]
    _verify_model(ReLU6(), input_info)


def test_maxpool2d():
    """test relax translator for maxpool2d"""

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
    _verify_model(MaxPool2d(), input_info)
    _verify_model(MaxPool2d2(), input_info)
    _verify_model(MaxPool2d3(), input_info)


def test_avgpool2d():
    """test relax translator for avgpool2d"""

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
    _verify_model(AvgPool2d(), input_info)
    _verify_model(AvgPool2d2(), input_info)


def test_adaptive_avgpool2d():
    """test relax translator for adaptive_avgpool2d"""

    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, data):
            return self.pool(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(AdaptiveAvgPool2d0(), input_info)


def test_flatten():
    """test relax translator for flatten"""

    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, data):
            return self.f(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Flatten(), input_info)
    _verify_model(torch.nn.Flatten(2, -1), input_info)


def test_batchnorm2d():
    """test relax translator for batchnorm2d"""

    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = torch.nn.BatchNorm2d(3)

        def forward(self, data):
            return self.batchnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(BatchNorm2d(), input_info)


def test_embedding():
    """test relax translator for embedding"""

    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, data):
            return self.embedding(data)

    _verify_model(Embedding(), [([4], "int64")])
    _verify_model(Embedding(), [([4, 5], "int64")])


def test_dropout():
    """test relax translator for dropout"""

    class Dropout1(Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, data):
            return self.dropout(data)

    class Dropout2(Module):
        def forward(self, data):
            return torch.dropout(data, 0.5, train=True)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Dropout1(), input_info)
    _verify_model(Dropout2(), input_info)


def test_layernorm():
    """test relax translator for layernorm"""

    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((10, 10))

        def forward(self, data):
            return self.layernorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(LayerNorm(), input_info)


def test_functional_layernorm():
    """test relax translator for functional_layernorm"""

    class LayerNorm(Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, data):
            return torch.nn.functional.layer_norm(
                data, self.weight.shape, self.weight, self.bias, 1e-5
            )

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(LayerNorm((10, 10)), input_info)


def test_cross_entropy():
    """test relax translator for cross_entropy"""

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

    input_info = [([3, 2], "float32"), ([3], "int32")]
    _verify_model(CrossEntropy1(), input_info)
    _verify_model(CrossEntropy2(), input_info)
    _verify_model(CrossEntropy3(), input_info)


def test_functional_cross_entropy():
    """test relax translator for functional_cross_entropy"""

    class CrossEntropy(Module):
        def forward(self, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

    input_info = [([3, 10], "float32"), ([3], "int32")]
    _verify_model(CrossEntropy(), input_info)


def test_silu():
    """test relax translator for silu"""

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
    _verify_model(SiLU(), input_info)
    _verify_model(SiLU2(), input_info)


def test_groupnorm():
    """test relax translator for groupnorm"""

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(3, 3)

        def forward(self, data):
            return self.groupnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(GroupNorm(), input_info)


def test_softmax():
    """test relax translator for softmax"""

    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, data):
            return self.softmax(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Softmax(), input_info)


def test_binary():
    """test relax translator for binary"""

    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    _verify_model(Add1(), input_info1)
    _verify_model(Add2(), input_info2)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    _verify_model(Sub1(), input_info1)
    _verify_model(Sub2(), input_info2)

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    _verify_model(Mul1(), input_info1)
    _verify_model(Mul2(), input_info2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    _verify_model(TrueDiv1(), input_info1)
    _verify_model(TrueDiv2(), input_info2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    _verify_model(FloorDiv1(), input_info1)
    _verify_model(FloorDiv2(), input_info2)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    _verify_model(Power1(), input_info1)
    _verify_model(Power2(), input_info2)

    # LT
    class LT1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    class LT2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    _verify_model(LT1(), input_info1)
    _verify_model(LT2(), input_info2)


def test_size():
    """test relax translator for size"""

    class Size(Module):
        def forward(self, data):
            return data.size()

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Size(), input_info)


def test_squeeze():
    """test relax translator for squeeze"""

    class Squeeze1(Module):
        def forward(self, data):
            return data.squeeze(1)

    class Squeeze2(Module):
        def forward(self, data):
            return data.squeeze()

    input_info = [([3, 1, 4, 1], "float32")]
    _verify_model(Squeeze1(), input_info)
    _verify_model(Squeeze2(), input_info)


def test_unsqueeze():
    """test relax translator for unsqueeze"""

    class Unsqueeze1(Module):
        def forward(self, data):
            return data.unsqueeze(1)

    class Unsqueeze2(Module):
        def forward(self, data):
            return data.unsqueeze(-1)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Unsqueeze1(), input_info)
    _verify_model(Unsqueeze2(), input_info)


def test_getattr():
    """test relax translator for getattr"""

    class GetAttr1(Module):
        def forward(self, data):
            return data.shape

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(GetAttr1(), input_info)


def test_getitem():
    """test relax translator for getitem"""

    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    _verify_model(Slice1(), [([1, 3, 10, 10], "float32")])
    _verify_model(Slice2(), [([8, 16], "float32")])


def test_unary():
    """test relax translator for unary"""

    input_info = [([1, 3, 10, 10], "float32")]

    # sin
    class Sin(Module):
        def forward(self, data):
            return torch.sin(data)

    _verify_model(Sin(), input_info)

    # cos
    class Cos(Module):
        def forward(self, data):
            return torch.cos(data)

    _verify_model(Cos(), input_info)

    # exp
    class Exp(Module):
        def forward(self, data):
            return torch.exp(data)

    _verify_model(Exp(), input_info)

    # sqrt
    class Sqrt(Module):
        def forward(self, data):
            return torch.sqrt(data)

    _verify_model(Sqrt(), input_info)

    # sigmoid
    class Sigmoid(Module):
        def forward(self, data):
            return torch.sigmoid(data)

    _verify_model(Sigmoid(), input_info)

    # round
    class Round(Module):
        def forward(self, data):
            return torch.round(data)

    _verify_model(Round(), input_info)


def test_gelu():
    """test relax translator for gelu"""

    class Gelu(Module):
        def forward(self, data):
            return torch.nn.functional.gelu(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Gelu(), input_info)


def test_tanh():
    """test relax translator for tanh"""

    class Tanh(Module):
        def forward(self, data):
            return torch.tanh(data)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Tanh(), input_info)


def test_clamp():
    """test relax translator for clamp"""

    class Clamp(Module):
        def forward(self, data):
            return torch.clamp(data, min=0.1, max=0.5)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Clamp(), input_info)


def test_interpolate():
    """test relax translator for interpolate"""

    class Interpolate(Module):
        def forward(self, data):
            return torch.nn.functional.interpolate(data, (5, 5))

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Interpolate(), input_info)


def test_addmm():
    """test relax translator for addmm"""

    class Addmm(Module):
        def forward(self, x_1, x_2, x_3):
            return torch.addmm(x_1, x_2, x_3)

    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]
    _verify_model(Addmm(), input_info)


def test_split():
    """test relax translator for split"""

    class Split(Module):
        def forward(self, data):
            return torch.split(data, 1, dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Split(), input_info)


def test_cumsum():
    """test relax translator for cumsum"""

    class Cumsum(Module):
        def forward(self, data):
            return torch.cumsum(data, dim=1, dtype=torch.int32)

    input_info = [([1, 2, 3, 4], "float32")]
    _verify_model(Cumsum(), input_info)


def test_chunk():
    """test relax translator for chunk"""

    class Chunk(Module):
        def forward(self, data):
            return torch.chunk(data, 3, dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    _verify_model(Chunk(), input_info)


def test_inplace_fill():
    """test relax translator for inplace_fill"""

    class InplaceFill(Module):
        def forward(self, data):
            data.fill_(1.5)
            return data

    _verify_model(InplaceFill(), [([10, 10], "float32")], opt_config={"opt_level": 0})


def test_arange():
    """test relax translator for arange"""

    class Arange(Module):
        def forward(self):
            return torch.arange(0, 20, dtype=torch.int32)

    _verify_model(Arange(), [([10, 10], "float32")])


def test_empty():
    """test relax translator for empty"""

    class Empty(Module):
        def forward(self):
            return torch.empty((10, 10), dtype=torch.float32)

    _verify_model(Empty(), [([10, 10], "float32")])


def test_tensor():
    """test relax translator for tensor"""

    class Empty1(Module):
        def forward(self):
            return torch.tensor(3, dtype=torch.float32)

    class Empty2(Module):
        def forward(self):
            return torch.tensor(3)

    _verify_model(Empty1(), [([10, 10], "float32")])
    _verify_model(Empty2(), [([10, 10], "float32")])


def test_tril():
    """test relax translator for tril"""

    class Tril(Module):
        def forward(self, data):
            return torch.tril(data, 1)

    class InplaceTril(Module):
        def forward(self, data):
            data.tril_(1)
            return data

    input_info = [([10, 10], "float32")]
    _verify_model(Tril(), input_info)
    _verify_model(InplaceTril(), input_info)


def test_triu():
    """test relax translator for triu"""

    class Triu(Module):
        def forward(self, data):
            return torch.triu(data, 1)

    class InplaceTriu(Module):
        def forward(self, data):
            data.triu_(1)
            return data

    input_info = [([10, 10], "float32")]
    _verify_model(Triu(), input_info)
    _verify_model(InplaceTriu(), input_info)


def test_new_ones():
    """test relax translator for new_ones"""

    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    input_info = [([1, 2, 3], "float32")]
    _verify_model(NewOnes(), input_info, opt_config={"opt_level": 0})


def test_expand():
    """test relax translator for expand"""

    class Expand(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    input_info = [([1, 2, 3, 4], "float32")]
    _verify_model(Expand(), input_info)


def test_reduce():
    """test relax translator for reduce"""

    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    input_info = [([1, 2, 3, 4], "float32")]
    _verify_model(Sum(), input_info)


def test_datatype():
    """test relax translator for datatype"""

    input_info = [([1, 2, 3, 4], "float32")]

    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    _verify_model(ToFloat(), input_info)

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    _verify_model(ToHalf(), input_info)

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

    _verify_model(Type(), input_info)
    _verify_model(TypeFromAttr(), input_info)
    _verify_model(AsType(), input_info)


def test_permute():
    """test relax translator for permute"""

    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    input_info = [([1, 2, 3, 4], "float32")]
    _verify_model(Permute(), input_info)


def test_reshape():
    """test relax translator for reshape"""

    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    _verify_model(Reshape(), input_info)


def test_transpose():
    """test relax translator for transpose"""

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    input_info = [([1, 2, 3, 4], "float32")]
    _verify_model(Transpose(), input_info)


def test_view():
    """test relax translator for view"""

    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    _verify_model(View(), input_info)


def test_keep_params():
    """test relax translator for keep_params"""

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    _verify_model(Conv2D1(), [([1, 3, 10, 10], "float32")])


def test_unwrap_unit_return_tuple():
    """test relax translator for unwrap_unit_return_tuple"""

    class Identity(Module):
        def forward(self, x):
            return (x,)

    _verify_model(Identity(), [([256, 256], "float32")])


def test_no_bind_return_tuple():
    """test relax translator for no_bind_return_tuple"""

    class Identity(Module):
        def forward(self, x, y):
            return (x, y)

    input_info = [([256, 256], "float32"), ([256, 256], "float32")]
    _verify_model(Identity(), input_info)


def test_argmax():
    """test relax translator for argmax"""

    class Argmax1(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1)

    class Argmax2(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1, keepdim=True)

    _verify_model(Argmax1(), [([256, 256], "float32")])
    _verify_model(Argmax2(), [([256, 256], "float32")])


def test_argmin():
    """test relax translator for argmin"""

    class Argmin1(Module):
        def forward(self, data):
            return torch.argmin(data)

    class Argmin2(Module):
        def forward(self, data):
            return torch.argmin(data, keepdim=True)

    _verify_model(Argmin1(), [([256, 256], "float32")])
    _verify_model(Argmin2(), [([256, 256], "float32")])


def test_to():
    """test relax translator for to"""

    class To1(Module):
        def forward(self, data):
            return data.to(torch.float16)

    class To2(Module):
        def forward(self, data):
            return data.to("cpu")

    _verify_model(To1(), [([256, 256], "float32")])
    _verify_model(To2(), [([256, 256], "float32")])


def test_mean():
    """test relax translator for mean"""

    class Mean(Module):
        def forward(self, data):
            return data.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, data):
            return data.mean(-1, keepdim=True)

    _verify_model(Mean(), [([256, 256], "float32")])
    _verify_model(MeanKeepDim(), [([256, 256], "float32")])


def test_rsqrt():
    """test relax translator for rsqrt"""

    class Rsqrt(Module):
        def forward(self, data):
            return torch.rsqrt(data)

    _verify_model(Rsqrt(), [([256, 256], "float32")])


def test_neg():
    """test relax translator for neg"""

    class Neg(Module):
        def forward(self, data):
            return -data

    _verify_model(Neg(), [([256, 256], "float32")])


def test_max():
    """test relax translator for max"""

    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    _verify_model(Max(), [([256, 256], "float32"), ([256, 256], "float32")])


def test_attention():
    """test relax translator for attention"""

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
    _verify_model(Attention1(), input_info)
    _verify_model(Attention2(), input_info)

    class Attention3(Module):
        def forward(self, q_data, k_data, v_data, mask):
            return F.scaled_dot_product_attention(q_data, k_data, v_data, mask)

    _verify_model(
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
