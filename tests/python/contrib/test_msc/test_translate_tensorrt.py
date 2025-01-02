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

""" Test translate for TensorrRT. """

import pytest

import torch
from torch import fx
from torch.nn import Module

import tvm.testing
from tvm.relax import PyExprVisitor
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.framework.tensorrt.frontend import translate
from tvm.contrib.msc.framework.tensorrt import codegen
from tvm.contrib.msc.core import utils as msc_utils

requires_tensorrt = pytest.mark.skipif(
    tvm.get_global_func("relax.ext.tensorrt", True) is None,
    reason="TENSORRT is not enabled",
)


def build_and_run(mod, inputs):
    """Build and run the virtual machine"""

    target = tvm.target.Target("cuda")
    mod = tvm.relax.transform.LegalizeOps()(mod)
    with target:
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    with tvm.transform.PassContext(opt_level=3):
        rt_mod = tvm.relax.build(mod, target)
        runnable = tvm.relax.VirtualMachine(rt_mod, tvm.cuda())
    res = runnable["main"](*inputs)
    if isinstance(res, tvm.runtime.NDArray):
        return [res.asnumpy()]
    return [e.asnumpy() for e in res]


def check_names(mod):
    """Check the byoc name and unique_name"""

    @tvm.relax.expr_functor.visitor
    class NameChecker(PyExprVisitor):
        """Checker to check if any non-target ops exist"""

        def check(self, expr):
            self._recorded_names = set()
            if isinstance(expr, tvm.relax.Expr):
                self.visit_expr(expr)
            elif isinstance(expr, tvm.relax.BindingBlock):
                self.visit_binding_block(expr)

        def visit_function_(self, op: tvm.relax.Function) -> None:
            if "Composite" in op.attrs:
                assert "Unique" in op.attrs, "Can not find unique_name for func " + str(op)
                name = str(op.attrs["Unique"])
                assert name not in self._recorded_names, "Name {} is already in use".format(name)
                self._recorded_names.add(name)
            super().visit_function_(op)

    def _is_target_func(func):
        if "Codegen" not in func.attrs:
            return False
        return func.attrs["Codegen"] == "msc_tensorrt"

    for _, func in mod.functions.items():
        if not _is_target_func(func):
            continue
        assert "Unique" in func.attrs, "Can not find Unique from function attributes"
        NameChecker().check(func)


def verify_model(torch_model, input_info, **trans_config):
    """Build model and verify results"""

    graph_model = fx.symbolic_trace(torch_model)
    datas = [msc_utils.random_data(i) for i in input_info]
    torch_datas = [torch.from_numpy(i) for i in datas]
    with torch.no_grad():
        golden = torch_model(*torch_datas)
        mod = from_fx(graph_model, input_info)
    if not isinstance(golden, (list, tuple)):
        golden = [golden]
    golden = [g.detach().cpu().numpy() for g in golden]
    # partition module for tensorrt
    mod, graphs, weights = translate.partition_for_tensorrt(mod, trans_config=trans_config)
    check_names(mod)
    output_folder = msc_utils.msc_dir()
    # tranalte to tensorrt
    mod = codegen.to_tensorrt(mod, graphs, weights, output_folder=output_folder)
    tvm_datas = [tvm.nd.array(i, device=tvm.cuda()) for i in datas]
    results = build_and_run(mod, tvm_datas)
    for gol, res in zip(golden, results):
        tvm.testing.assert_allclose(gol, res, atol=1e-3, rtol=1e-3)
    output_folder.destory()


@requires_tensorrt
def test_conv1d():
    """test tensorrt translator for conv1d"""

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


@requires_tensorrt
def test_conv2d():
    """test tensorrt translator for conv2d"""

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


@requires_tensorrt
def test_linear():
    """test tensorrt translator for linear"""

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
    verify_model(Dense1(), input_info)
    verify_model(Dense2(), input_info)
    verify_model(Dense1(), input_info, linear_to_conv=True)
    verify_model(Dense2(), input_info, linear_to_conv=True)
    verify_model(MatMul1(), [([10, 10], "float32"), ([10, 10], "float32")])


@requires_tensorrt
def test_bmm():
    """test tensorrt translator for bmm"""

    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    input_info = [((4, 128, 256), "float32"), ((4, 256, 512), "float32")]
    verify_model(BMM(), input_info)


@requires_tensorrt
def test_baddbmm():
    """test tensorrt translator for baddbmm"""

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
    verify_model(BAddBMM1(), input_info)
    verify_model(BAddBMM2(), input_info)


@requires_tensorrt
def test_relu():
    """test tensorrt translator for relu"""

    class ReLU(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, data):
            return self.relu(data)

    input_info = [([10, 10], "float32")]
    verify_model(ReLU(), input_info)


@requires_tensorrt
def test_relu6():
    """test tensorrt translator for relu6"""

    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, data):
            return self.relu6(data)

    input_info = [([10, 10], "float32")]
    verify_model(ReLU6(), input_info)


@requires_tensorrt
def test_maxpool2d():
    """test tensorrt translator for maxpool2d"""

    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, data):
            return self.pool(data)

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, data):
            return self.pool(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(MaxPool2d(), input_info)
    verify_model(MaxPool2d2(), input_info)


@requires_tensorrt
def test_avgpool2d():
    """test tensorrt translator for avgpool2d"""

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


@requires_tensorrt
def test_adaptive_avgpool2d():
    """test tensorrt translator for adaptive_avgpool2d"""

    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, data):
            return self.pool(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AdaptiveAvgPool2d0(), input_info)


@requires_tensorrt
def test_flatten():
    """test tensorrt translator for flatten"""

    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, data):
            return self.f(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Flatten(), input_info)
    verify_model(torch.nn.Flatten(2, -1), input_info)


@requires_tensorrt
def test_batchnorm2d():
    """test tensorrt translator for batchnorm2d"""

    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = torch.nn.BatchNorm2d(3)

        def forward(self, data):
            return self.batchnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(BatchNorm2d().eval(), input_info)


@requires_tensorrt
def test_embedding():
    """test tensorrt translator for embedding"""

    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, data):
            return self.embedding(data.to(torch.int64))

    verify_model(Embedding(), [([4], "int32")])
    verify_model(Embedding(), [([4, 5], "int32")])


@requires_tensorrt
def test_layernorm():
    """test tensorrt translator for layernorm"""

    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((10, 10))

        def forward(self, data):
            return self.layernorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(LayerNorm(), input_info)


@requires_tensorrt
def test_silu():
    """test tensorrt translator for silu"""

    class SiLU(Module):
        def __init__(self):
            super().__init__()
            self.silu = torch.nn.SiLU()

        def forward(self, data):
            return self.silu(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(SiLU(), input_info)


@requires_tensorrt
def test_groupnorm():
    """test tensorrt translator for groupnorm"""

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(3, 3)

        def forward(self, data):
            return self.groupnorm(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(GroupNorm(), input_info)


@requires_tensorrt
def test_softmax():
    """test tensorrt translator for softmax"""

    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, data):
            return self.softmax(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Softmax(), input_info)


@requires_tensorrt
def test_binary():
    """test tensorrt translator for binary"""

    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    verify_model(Add1(), input_info1)
    verify_model(Add2(), input_info2)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    verify_model(Sub1(), input_info1)
    verify_model(Sub2(), input_info2)

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    verify_model(Mul1(), input_info1)
    verify_model(Mul2(), input_info2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    verify_model(TrueDiv1(), input_info1)
    verify_model(TrueDiv2(), input_info2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    verify_model(FloorDiv1(), input_info1)
    verify_model(FloorDiv2(), input_info2)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    verify_model(Power1(), input_info1)
    verify_model(Power2(), input_info2)


@requires_tensorrt
def test_squeeze():
    """test tensorrt translator for squeeze"""

    class Squeeze1(Module):
        def forward(self, data):
            return data.squeeze(1)

    class Squeeze2(Module):
        def forward(self, data):
            return data.squeeze()

    input_info = [([3, 1, 4, 1], "float32")]
    verify_model(Squeeze1(), input_info)
    verify_model(Squeeze2(), input_info)


@requires_tensorrt
def test_unsqueeze():
    """test tensorrt translator for unsqueeze"""

    class Unsqueeze1(Module):
        def forward(self, data):
            return data.unsqueeze(1)

    class Unsqueeze2(Module):
        def forward(self, data):
            return data.unsqueeze(-1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Unsqueeze1(), input_info)
    verify_model(Unsqueeze2(), input_info)


@requires_tensorrt
def test_getitem():
    """test tensorrt translator for getitem"""

    class Slice1(Module):
        def forward(self, x):
            return x[0:1, 1::2, :, :3]

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    verify_model(Slice1(), [([1, 3, 10, 10], "float32")])
    verify_model(Slice2(), [([8, 16], "float32")])


@requires_tensorrt
def test_unary():
    """test tensorrt translator for unary"""

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


@requires_tensorrt
def test_tanh():
    """test tensorrt translator for tanh"""

    class Tanh(Module):
        def forward(self, data):
            return torch.tanh(data)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Tanh(), input_info)


@requires_tensorrt
def test_clamp():
    """test tensorrt translator for clamp"""

    class Clamp(Module):
        def forward(self, data):
            return torch.clamp(data, min=0.1, max=0.5)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Clamp(), input_info)


@requires_tensorrt
def test_interpolate():
    """test tensorrt translator for interpolate"""

    class Interpolate(Module):
        def forward(self, data):
            return torch.nn.functional.interpolate(data, (5, 5))

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Interpolate(), input_info)


@requires_tensorrt
def test_addmm():
    """test tensorrt translator for addmm"""

    class Addmm(Module):
        def forward(self, x_1, x_2, x_3):
            return torch.addmm(x_1, x_2, x_3)

    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]
    verify_model(Addmm(), input_info)


@requires_tensorrt
def test_split():
    """test tensorrt translator for split"""

    class Split1(Module):
        def forward(self, data):
            return torch.split(data, 1, dim=1)

    class Split2(Module):
        def forward(self, data):
            return torch.split(data, [1, 2], dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Split1(), input_info)
    verify_model(Split2(), input_info)


@requires_tensorrt
def test_unbind():
    """test tensorrt to relax for unbind"""

    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    input_info = [([3, 3, 10, 10], "float32")]
    verify_model(Unbind1(), input_info)
    verify_model(Unbind2(), input_info)


@requires_tensorrt
def test_chunk():
    """test tensorrt translator for chunk"""

    class Chunk(Module):
        def forward(self, data):
            return torch.chunk(data, 3, dim=1)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Chunk(), input_info)


@requires_tensorrt
def test_expand():
    """test tensorrt translator for expand"""

    class Expand1(Module):
        def forward(self, x):
            x = x + 1.0
            return x.expand(4, 2, 3, 4)

    class Expand2(Module):
        def forward(self, x):
            x = x + 1.0
            return x.expand(4, -1, -1, 4)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Expand1(), input_info)
    verify_model(Expand2(), input_info)


@requires_tensorrt
def test_reduce():
    """test tensorrt translator for reduce"""

    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Sum(), input_info)


@requires_tensorrt
def test_permute():
    """test tensorrt translator for permute"""

    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Permute(), input_info)


@requires_tensorrt
def test_reshape():
    """test tensorrt translator for reshape"""

    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Reshape(), input_info)


@requires_tensorrt
def test_transpose():
    """test tensorrt translator for transpose"""

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Transpose(), input_info)


@requires_tensorrt
def test_view():
    """test tensorrt translator for view"""

    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(View(), input_info)


@requires_tensorrt
def test_argmax():
    """test tensorrt translator for argmax"""

    class Argmax1(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1).to(torch.int32)

    class Argmax2(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1, keepdim=True).to(torch.int32)

    verify_model(Argmax1(), [([256, 256], "float32")])
    verify_model(Argmax2(), [([256, 256], "float32")])


@requires_tensorrt
def test_argmin():
    """test tensorrt translator for argmin"""

    class Argmin1(Module):
        def forward(self, data):
            return torch.argmin(data, dim=-1).to(torch.int32)

    class Argmin2(Module):
        def forward(self, data):
            return torch.argmin(data, dim=-1, keepdim=True).to(torch.int32)

    verify_model(Argmin1(), [([256, 256], "float32")])
    verify_model(Argmin2(), [([256, 256], "float32")])


@requires_tensorrt
def test_mean():
    """test tensorrt translator for mean"""

    class Mean(Module):
        def forward(self, data):
            return data.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, data):
            return data.mean(-1, keepdim=True)

    verify_model(Mean(), [([256, 256], "float32")])
    verify_model(MeanKeepDim(), [([256, 256], "float32")])


@requires_tensorrt
def test_rsqrt():
    """test tensorrt translator for rsqrt"""

    class Rsqrt(Module):
        def forward(self, data):
            return torch.rsqrt(data)

    verify_model(Rsqrt(), [([256, 256], "float32")])


@requires_tensorrt
def test_neg():
    """test tensorrt translator for neg"""

    class Neg(Module):
        def forward(self, data):
            return -data

    verify_model(Neg(), [([256, 256], "float32")])


@requires_tensorrt
def test_max():
    """test tensorrt translator for max"""

    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    verify_model(Max(), [([256, 256], "float32"), ([256, 256], "float32")])


@requires_tensorrt
def test_gelu():
    """test tensorrt translator for gelu"""

    class Gelu1(Module):
        def forward(self, data):
            return torch.nn.functional.gelu(data)

    class Gelu2(Module):
        def forward(self, data):
            return torch.nn.functional.gelu(data, approximate="tanh")

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Gelu1(), input_info)
    verify_model(Gelu2(), input_info)


@requires_tensorrt
def test_cat():
    """test tensorrt translator for cat"""

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
    verify_model(Cat1(), input_info)
    verify_model(Cat2(), [([1, 3, 10, 10], "float32")])


if __name__ == "__main__":
    tvm.testing.main()
