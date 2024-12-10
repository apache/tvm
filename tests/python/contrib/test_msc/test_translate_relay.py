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
# pylint: disable=unused-argument

""" Test translate from relay. """

import torch
from torch import fx
from torch.nn import Module

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.relay.frontend import from_pytorch
from tvm import relay
from tvm.ir.module import IRModule
from tvm.contrib.msc.core.frontend import translate
from tvm.contrib.msc.framework.tvm import codegen as tvm_codegen
from tvm.contrib.msc.core import utils as msc_utils


def _valid_target(target):
    if not target:
        return target
    if target == "ignore":
        return None
    if target == "cuda" and not tvm.cuda().exist:
        return None
    if isinstance(target, str):
        target = tvm.target.Target(target)
    return target


def _run_relax(relax_mod, target, datas):
    relax_mod = tvm.relax.transform.LegalizeOps()(relax_mod)
    with tvm.transform.PassContext(opt_level=3):
        relax_exec = tvm.relax.build(relax_mod, target)
        runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cpu())
    res = runnable["main"](*datas)
    if isinstance(res, tvm.runtime.NDArray):
        return [res.asnumpy()]
    return [e.asnumpy() for e in res]


def verify_model(torch_model, input_info, opt_config=None, codegen_config=None, build_target=None):
    """Compare relax with relay"""

    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        expected = from_fx(graph_model, input_info)
    expected = tvm.relax.transform.CanonicalizeBindings()(expected)

    # graph from relay
    datas = [msc_utils.random_data(i) for i in input_info]
    torch_datas = [torch.from_numpy(i) for i in datas]
    with torch.no_grad():
        scripted_model = torch.jit.trace(torch_model, tuple(torch_datas)).eval()  # type: ignore
    shape_list = [("input" + str(idx), i) for idx, i in enumerate(input_info)]
    relay_mod, params = from_pytorch(scripted_model, shape_list)
    graph, weights = translate.from_relay(relay_mod, params, opt_config=opt_config)
    # to relax
    codegen_config = codegen_config or {}
    codegen_config.update({"explicit_name": False, "from_relay": True})
    mod = tvm_codegen.to_relax(graph, weights, codegen_config)
    if build_target:
        build_target = _valid_target(build_target)
        if not build_target:
            return
        tvm_datas = [tvm.nd.array(i) for i in datas]
        expected_res = _run_relax(expected, build_target, tvm_datas)
        if not graph.get_inputs():
            tvm_datas = []
        res = _run_relax(mod, build_target, tvm_datas)
        for exp_r, new_r in zip(expected_res, res):
            tvm.testing.assert_allclose(exp_r, new_r, atol=1e-5, rtol=1e-5)
    else:
        tvm.ir.assert_structural_equal(mod, expected)


def test_conv1d():
    """test relay to relax for conv1d"""

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
    verify_model(Conv1D1(), input_info)
    verify_model(Conv1D2(), input_info)


def test_conv2d():
    """test relay to relax for conv2d"""

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
    verify_model(Conv2D1(), input_info)
    verify_model(Conv2D2(), input_info)


def test_linear():
    """test relay to relax for linear"""

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
    verify_model(Dense1(), input_info, build_target="llvm")
    verify_model(Dense2(), input_info, build_target="llvm")
    verify_model(MatMul1(), [([10, 10], "float32"), ([10, 10], "float32")], build_target="llvm")


def test_bmm():
    """test relay to relax for bmm"""

    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    input_info = [((4, 128, 256), "float32"), ((4, 256, 512), "float32")]
    verify_model(BMM(), input_info, opt_config={"opt_level": 3})


def test_baddbmm():
    """test relay to relax for baddbmm"""

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
    verify_model(BAddBMM1(), input_info, opt_config={"opt_level": 3}, build_target="llvm")
    verify_model(BAddBMM2(), input_info, opt_config={"opt_level": 3}, build_target="llvm")


def test_relu():
    """test relay to relax for relu"""

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
    verify_model(ReLU(), input_info)
    verify_model(ReLU1(), input_info)


def test_relu6():
    """test relay to relax for relu6"""

    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, data):
            return self.relu6(data)

    input_info = [([10, 10], "float32")]
    verify_model(ReLU6(), input_info)


def test_maxpool2d():
    """test relay to relax for maxpool2d"""

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
    verify_model(MaxPool2d(), input_info)
    verify_model(MaxPool2d2(), input_info)
    verify_model(MaxPool2d3(), input_info)


def test_avgpool2d():
    """test relay to relax for avgpool2d"""

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
    verify_model(AvgPool2d(), input_info)
    verify_model(AvgPool2d2(), input_info)


def test_adaptive_avgpool2d():
    """test relay to relax for adaptive_avgpool2d"""

    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, data):
            return self.pool(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AdaptiveAvgPool2d0(), input_info)


def test_flatten():
    """test relay to relax for flatten"""

    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, data):
            return self.f(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Flatten(), input_info, opt_config={"opt_level": 3}, build_target="llvm")
    verify_model(
        torch.nn.Flatten(2, -1), input_info, opt_config={"opt_level": 3}, build_target="llvm"
    )


def test_batchnorm2d():
    """test relay to relax for batchnorm2d"""

    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = torch.nn.BatchNorm2d(3)

        def forward(self, data):
            return self.batchnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(BatchNorm2d(), input_info, build_target="llvm")


def test_embedding():
    """test relay to relax for embedding"""

    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, data):
            return self.embedding(data)

    verify_model(Embedding(), [([4], "int64")])
    verify_model(Embedding(), [([4, 5], "int64")])


def test_layernorm():
    """test relay to relax for layernorm"""

    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm(10)

        def forward(self, data):
            return self.layernorm(data)

    input_info = [([1, 10, 10], "float32")]
    verify_model(LayerNorm(), input_info)


def test_functional_layernorm():
    """test relay to relax for functional_layernorm"""

    class LayerNorm(Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, data):
            return torch.nn.functional.layer_norm(
                data, self.weight.shape, self.weight, self.bias, 1e-5
            )

    input_info = [([1, 10, 10], "float32")]
    verify_model(LayerNorm((10)), input_info)


def test_cross_entropy():
    """test relay to relax for cross_entropy"""

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
    verify_model(CrossEntropy1(), input_info, opt_config={"opt_level": 3}, build_target="llvm")
    verify_model(CrossEntropy2(), input_info, opt_config={"opt_level": 3}, build_target="llvm")
    verify_model(CrossEntropy3(), input_info, opt_config={"opt_level": 3}, build_target="llvm")


def test_functional_cross_entropy():
    """test relay to relax for functional_cross_entropy"""

    class CrossEntropy(Module):
        def forward(self, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

    input_info = [([3, 10], "float32"), ([3], "int64")]
    verify_model(CrossEntropy(), input_info, opt_config={"opt_level": 3}, build_target="llvm")


def test_silu():
    """test relay to relax for silu"""

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
    verify_model(SiLU(), input_info, build_target="llvm")
    verify_model(SiLU2(), input_info, build_target="llvm")


def test_groupnorm():
    """test relay to relax for groupnorm"""

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(3, 3)

        def forward(self, data):
            return self.groupnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(GroupNorm(), input_info)


def test_softmax():
    """test relay to relax for softmax"""

    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, data):
            return self.softmax(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Softmax(), input_info)


def test_binary():
    """test relay to relax for binary"""

    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    verify_model(Add1(), input_info1, opt_config={"opt_level": 3})
    verify_model(Add2(), input_info2, opt_config={"opt_level": 3})

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    verify_model(Sub1(), input_info1, opt_config={"opt_level": 3})
    verify_model(Sub2(), input_info2, opt_config={"opt_level": 3})

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    verify_model(Mul1(), input_info1, opt_config={"opt_level": 3})
    verify_model(Mul2(), input_info2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    verify_model(TrueDiv1(), input_info1, opt_config={"opt_level": 3})
    verify_model(TrueDiv2(), input_info2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    verify_model(FloorDiv1(), input_info1, opt_config={"opt_level": 3})
    verify_model(FloorDiv2(), input_info2, opt_config={"opt_level": 3})

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    verify_model(Power1(), input_info1, opt_config={"opt_level": 3})
    verify_model(Power2(), input_info2, opt_config={"opt_level": 3})

    # LT
    class LT1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    class LT2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    verify_model(LT1(), input_info1, opt_config={"opt_level": 3})
    verify_model(LT2(), input_info2, opt_config={"opt_level": 3})


def test_squeeze():
    """test relay to relax for squeeze"""

    class Squeeze1(Module):
        def forward(self, data):
            return data.squeeze(1)

    class Squeeze2(Module):
        def forward(self, data):
            return data.squeeze()

    input_info = [([3, 1, 4, 1], "float32")]
    verify_model(Squeeze1(), input_info)
    verify_model(Squeeze2(), input_info)


def test_unsqueeze():
    """test relay to relax for unsqueeze"""

    class Unsqueeze1(Module):
        def forward(self, data):
            return data.unsqueeze(1)

    class Unsqueeze2(Module):
        def forward(self, data):
            return data.unsqueeze(-1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Unsqueeze1(), input_info)
    verify_model(Unsqueeze2(), input_info)


def test_getitem():
    """test relay to relax for getitem"""

    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    verify_model(Slice1(), [([1, 3, 10, 10], "float32")], build_target="ignore")
    verify_model(Slice2(), [([8, 16], "float32")], build_target="llvm")


def test_unary():
    """test relay to relax for unary"""

    input_info = [([1, 3, 10, 10], "float32")]

    # sin
    class Sin(Module):
        def forward(self, data):
            return torch.sin(data)

    verify_model(Sin(), input_info)

    # cos
    class Cos(Module):
        def forward(self, data):
            return torch.cos(data)

    verify_model(Cos(), input_info)

    # exp
    class Exp(Module):
        def forward(self, data):
            return torch.exp(data)

    verify_model(Exp(), input_info)

    # sqrt
    class Sqrt(Module):
        def forward(self, data):
            return torch.sqrt(data)

    verify_model(Sqrt(), input_info)

    # sigmoid
    class Sigmoid(Module):
        def forward(self, data):
            return torch.sigmoid(data)

    verify_model(Sigmoid(), input_info)

    # round
    class Round(Module):
        def forward(self, data):
            return torch.round(data)

    verify_model(Round(), input_info)


def test_gelu():
    """test relay to relax for gelu"""

    class Gelu(Module):
        def forward(self, data):
            return torch.nn.functional.gelu(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Gelu(), input_info)


def test_tanh():
    """test relay to relax for tanh"""

    class Tanh(Module):
        def forward(self, data):
            return torch.tanh(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Tanh(), input_info)


def test_clamp():
    """test relay to relax for clamp"""

    class Clamp(Module):
        def forward(self, data):
            return torch.clamp(data, min=0.1, max=0.5)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Clamp(), input_info)


def test_interpolate():
    """test relay to relax for interpolate"""

    class Interpolate(Module):
        def forward(self, data):
            return torch.nn.functional.interpolate(data, (5, 5))

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Interpolate(), input_info, build_target="llvm")


def test_addmm():
    """test relay to relax for addmm"""

    class Addmm(Module):
        def forward(self, x_1, x_2, x_3):
            return torch.addmm(x_1, x_2, x_3)

    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]
    verify_model(Addmm(), input_info, build_target="llvm")


def test_split():
    """test relay to relax for split"""

    class Split1(Module):
        def forward(self, data):
            return torch.split(data, 1, dim=1)

    class Split2(Module):
        def forward(self, data):
            return torch.split(data, [1, 2], dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Split1(), input_info, build_target="llvm")
    verify_model(Split2(), input_info, build_target="llvm")


def test_unbind():
    """test relay to relax for unbind"""

    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    input_info = [([3, 3, 10, 10], "float32")]
    verify_model(Unbind1(), input_info, build_target="llvm")
    verify_model(Unbind2(), input_info, build_target="llvm")


def test_cumsum():
    """test relay to relax for cumsum"""

    class Cumsum(Module):
        def forward(self, data):
            return torch.cumsum(data, dim=1, dtype=torch.int32)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Cumsum(), input_info)


def test_chunk():
    """test relay to relax for chunk"""

    class Chunk(Module):
        def forward(self, data):
            return torch.chunk(data, 3, dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Chunk(), input_info, build_target="llvm")


def test_inplace_fill():
    """test relay to relax for inplace_fill"""

    class InplaceFill(Module):
        def forward(self, data):
            data.fill_(1.5)
            return data

    verify_model(InplaceFill(), [([10, 10], "float32")], build_target="llvm")


def test_arange():
    """test relay to relax for arange"""

    class Arange(Module):
        def forward(self, data):
            return torch.arange(0, 20, dtype=torch.int32)

    verify_model(
        Arange(), [([10, 10], "float32")], opt_config={"opt_level": 3}, build_target="llvm"
    )


def test_empty():
    """test relay to relax for empty"""

    class Empty(Module):
        def forward(self, data):
            return torch.empty((10, 10), dtype=torch.float32)

    verify_model(
        Empty(), [([10, 10], "float32")], opt_config={"opt_level": 3}, build_target="ignore"
    )


def test_tensor():
    """test relay to relax for tensor"""

    class Empty1(Module):
        def forward(self, data):
            return torch.tensor(3, dtype=torch.float32)

    class Empty2(Module):
        def forward(self, data):
            return torch.tensor(3)

    verify_model(Empty1(), [([10, 10], "float32")], build_target="llvm")
    verify_model(Empty2(), [([10, 10], "float32")], build_target="llvm")


def test_tril():
    """test relay to relax for tril"""

    class Tril(Module):
        def forward(self, data):
            return torch.tril(data, 1)

    class InplaceTril(Module):
        def forward(self, data):
            data.tril_(1)
            return data

    input_info = [([10, 10], "float32")]
    verify_model(Tril(), input_info)
    verify_model(InplaceTril(), input_info)


def test_triu():
    """test relay to relax for triu"""

    class Triu(Module):
        def forward(self, data):
            return torch.triu(data, 1)

    class InplaceTriu(Module):
        def forward(self, data):
            data.triu_(1)
            return data

    input_info = [([10, 10], "float32")]
    verify_model(Triu(), input_info)
    verify_model(InplaceTriu(), input_info)


def test_new_ones():
    """test relay to relax for new_ones"""

    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    input_info = [([1, 2, 3], "float32")]
    verify_model(NewOnes(), input_info, build_target="llvm")


def test_expand():
    """test relay to relax for expand"""

    class Expand1(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    class Expand2(Module):
        def forward(self, x):
            return x.expand(4, -1, -1, 4)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Expand1(), input_info, build_target="llvm")
    verify_model(Expand2(), input_info, build_target="llvm")


def test_reduce():
    """test relay to relax for reduce"""

    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Sum(), input_info)


def test_datatype():
    """test relay to relax for datatype"""

    input_info = [([1, 2, 3, 4], "float32")]

    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    verify_model(ToFloat(), input_info, build_target="llvm")

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    verify_model(ToHalf(), input_info)

    # type
    class Type(Module):
        def forward(self, x):
            return x.type(torch.float32)

    verify_model(Type(), input_info, build_target="llvm")


def test_permute():
    """test relay to relax for permute"""

    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Permute(), input_info)


def test_reshape():
    """test relay to relax for reshape"""

    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Reshape(), input_info)


def test_transpose():
    """test relay to relax for transpose"""

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Transpose(), input_info)


def test_view():
    """test relay to relax for view"""

    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(View(), input_info)


def test_keep_params():
    """test relay to relax for keep_params"""

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    verify_model(Conv2D1(), [([1, 3, 10, 10], "float32")])


def test_unwrap_unit_return_tuple():
    """test relay to relax for unwrap_unit_return_tuple"""

    class Identity(Module):
        def forward(self, x):
            return (x,)

    verify_model(Identity(), [([256, 256], "float32")], build_target="llvm")


def test_no_bind_return_tuple():
    """test relay to relax for no_bind_return_tuple"""

    class Identity(Module):
        def forward(self, x, y):
            return (x, y)

    input_info = [([256, 256], "float32"), ([256, 256], "float32")]
    verify_model(Identity(), input_info)


def test_argmax():
    """test relay to relax for argmax"""

    class Argmax1(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1)

    class Argmax2(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1, keepdim=True)

    verify_model(Argmax1(), [([256, 256], "float32")])
    verify_model(Argmax2(), [([256, 256], "float32")])


def test_to():
    """test relay to relax for to"""

    class To1(Module):
        def forward(self, data):
            return data.to(torch.float16)

    class To2(Module):
        def forward(self, data):
            return data.to("cpu")

    verify_model(To1(), [([256, 256], "float32")])
    verify_model(To2(), [([256, 256], "float32")])


def test_mean():
    """test relay to relax for mean"""

    class Mean(Module):
        def forward(self, data):
            return data.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, data):
            return data.mean(-1, keepdim=True)

    verify_model(Mean(), [([256, 256], "float32")])
    verify_model(MeanKeepDim(), [([256, 256], "float32")])


def test_rsqrt():
    """test relay to relax for rsqrt"""

    class Rsqrt(Module):
        def forward(self, data):
            return torch.rsqrt(data)

    verify_model(Rsqrt(), [([256, 256], "float32")])


def test_neg():
    """test relay to relax for neg"""

    class Neg(Module):
        def forward(self, data):
            return -data

    verify_model(Neg(), [([256, 256], "float32")])


def test_max():
    """test relay to relax for max"""

    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    verify_model(Max(), [([256, 256], "float32"), ([256, 256], "float32")])


def test_cat():
    """test relay to relax for cat"""

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
    verify_model(Cat1(), input_info, build_target="llvm")
    verify_model(Cat2(), [([1, 3, 10, 10], "float32")], build_target="llvm")


def test_name_string_with_colon():
    """test name string with colons,
    e.g., TFLite default input name 'serving_default_input:0'
    """

    dtype = "float32"
    x_var = relay.var("input_0:0", shape=(3, 5), dtype=dtype)
    y_var = relay.var("input_1:0", shape=(3, 5), dtype=dtype)
    z_add = relay.add(x_var, y_var)
    func = relay.Function([x_var, y_var], z_add)
    mod = IRModule()
    mod["main"] = func

    try:
        graph, _ = translate.from_relay(mod)
    except Exception as err:
        raise RuntimeError(f"Translation from relay to graph failed: {err}")
    inspect = graph.inspect()

    expected = {
        "inputs": [
            {"name": "input_0:0", "shape": [3, 5], "dtype": dtype, "layout": ""},
            {"name": "input_1:0", "shape": [3, 5], "dtype": dtype, "layout": ""},
        ],
        "outputs": [{"name": "add", "shape": [3, 5], "dtype": dtype, "layout": ""}],
        "nodes": {"total": 3, "input": 2, "add": 1},
    }

    assert msc_utils.dict_equal(inspect, expected), "Inspect {} mismatch with expected {}".format(
        inspect, expected
    )


if __name__ == "__main__":
    tvm.testing.main()
