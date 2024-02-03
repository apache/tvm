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

""" Test graph builder && graph. """

import torch
from torch import fx
from torch.nn import Module

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.core.frontend import translate
from tvm.contrib.msc.core import utils as msc_utils


def verify_model(torch_model, input_info, expected):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        mod = from_fx(graph_model, input_info)
    graph, _ = translate.from_relax(mod)
    inspect = graph.inspect()
    assert msc_utils.dict_equal(inspect, expected), "Inspect {} mismatch with expected {}".format(
        inspect, expected
    )


def test_conv1d():
    """test graph builder for conv1d"""

    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10], "dtype": "float32", "layout": "NCW"}],
        "outputs": [{"name": "conv1d", "shape": [1, 6, 4], "dtype": "float32", "layout": "NCW"}],
        "nodes": {"total": 2, "input": 1, "msc.conv1d_bias": 1},
    }

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)

        def forward(self, data):
            return self.conv(data)

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10], "dtype": "float32", "layout": "NCW"}],
        "outputs": [{"name": "conv1d", "shape": [1, 6, 4], "dtype": "float32", "layout": "NCW"}],
        "nodes": {"total": 2, "input": 1, "nn.conv1d": 1},
    }

    input_info = [([1, 3, 10], "float32")]
    verify_model(Conv1D1(), input_info, expected1)
    verify_model(Conv1D2(), input_info, expected2)


def test_conv2d():
    """test graph builder for conv2d"""

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "conv2d",
                "shape": [1, 6, 4, 4],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "msc.conv2d_bias": 1},
    }

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, data):
            return self.conv(data)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "conv2d", "shape": [1, 6, 4, 4], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.conv2d": 1},
    }
    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Conv2D1(), input_info, expected1)
    verify_model(Conv2D2(), input_info, expected2)


def test_linear():
    """test graph builder for linear"""

    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, data):
            return self.linear(data)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "matmul",
                "shape": [1, 3, 10, 7],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "msc.linear_bias": 1},
    }

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, data):
            return self.linear(data)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "matmul", "shape": [1, 3, 10, 7], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "msc.linear": 1},
    }

    class MatMul1(Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    expected3 = {
        "inputs": [
            {"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": "NC"},
            {"name": "inp_1", "shape": [10, 10], "dtype": "float32", "layout": "IO"},
        ],
        "outputs": [{"name": "matmul", "shape": [10, 10], "dtype": "float32", "layout": "NC"}],
        "nodes": {"total": 3, "input": 2, "matmul": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Dense1(), input_info, expected1)
    verify_model(Dense2(), input_info, expected2)
    verify_model(MatMul1(), [([10, 10], "float32"), ([10, 10], "float32")], expected3)


def test_bmm():
    """test graph builder for bmm"""

    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [4, 128, 256], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_1", "shape": [4, 256, 512], "dtype": "float32", "layout": "NIO"},
        ],
        "outputs": [
            {"name": "matmul", "shape": [4, 128, 512], "dtype": "float32", "layout": "NCD"}
        ],
        "nodes": {"total": 3, "input": 2, "matmul": 1},
    }

    input_info = [((4, 128, 256), "float32"), ((4, 256, 512), "float32")]
    verify_model(BMM(), input_info, expected)


def test_baddbmm():
    """test graph builder for baddbmm"""

    class BAddBMM1(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [4, 128, 512], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_1", "shape": [4, 128, 256], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_2", "shape": [4, 256, 512], "dtype": "float32", "layout": "NIO"},
        ],
        "outputs": [{"name": "add", "shape": [4, 128, 512], "dtype": "float32", "layout": "NCD"}],
        "nodes": {"total": 5, "input": 3, "matmul": 1, "add": 1},
    }

    class BAddBMM2(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [4, 128, 512], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [4, 128, 256], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_2", "shape": [4, 256, 512], "dtype": "float32", "layout": "NIO"},
        ],
        "outputs": [
            {"name": "multiply", "shape": [4, 128, 512], "dtype": "float32", "layout": "NCD"}
        ],
        "nodes": {"total": 6, "input": 3, "matmul": 1, "constant": 1, "multiply": 1},
    }

    input_info = [
        ((4, 128, 512), "float32"),
        ((4, 128, 256), "float32"),
        ((4, 256, 512), "float32"),
    ]
    verify_model(BAddBMM1(), input_info, expected1)
    verify_model(BAddBMM2(), input_info, expected2)


def test_relu():
    """test graph builder for relu"""

    class ReLU(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, data):
            return self.relu(data)

    class ReLU1(Module):
        def forward(self, data):
            return torch.nn.functional.relu(data)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "relu", "shape": [10, 10], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "nn.relu": 1},
    }

    input_info = [([10, 10], "float32")]
    verify_model(ReLU(), input_info, expected)
    verify_model(ReLU1(), input_info, expected)


def test_relu6():
    """test graph builder for relu6"""

    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, data):
            return self.relu6(data)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "clip", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "clip": 1},
    }
    input_info = [([10, 10], "float32")]
    verify_model(ReLU6(), input_info, expected)


def test_maxpool2d():
    """test graph builder for maxpool2d"""

    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, data):
            return self.pool(data)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "max_pool2d", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.max_pool2d": 1},
    }

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, data):
            return self.pool(data)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "max_pool2d", "shape": [1, 3, 4, 4], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.max_pool2d": 1},
    }

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, data):
            return self.pool(data)

    expected3 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "max_pool2d", "shape": [1, 3, 6, 6], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.max_pool2d": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(MaxPool2d(), input_info, expected1)
    verify_model(MaxPool2d2(), input_info, expected2)
    verify_model(MaxPool2d3(), input_info, expected3)


def test_avgpool2d():
    """test graph builder for avgpool2d"""

    class AvgPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[1, 1])

        def forward(self, data):
            return self.pool(data)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "avg_pool2d", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.avg_pool2d": 1},
    }

    class AvgPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True)

        def forward(self, data):
            return self.pool(data)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "avg_pool2d", "shape": [1, 3, 6, 6], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.avg_pool2d": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AvgPool2d(), input_info, expected1)
    verify_model(AvgPool2d2(), input_info, expected2)


def test_adaptive_avgpool2d():
    """test graph builder for adaptive_avgpool2d"""

    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, data):
            return self.pool(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "adaptive_avg_pool2d",
                "shape": [1, 3, 10, 10],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "nn.adaptive_avg_pool2d": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AdaptiveAvgPool2d0(), input_info, expected)


def test_flatten():
    """test graph builder for flatten"""

    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, data):
            return self.f(data)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "reshape", "shape": [1, 3, 100], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "reshape": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Flatten(), input_info, expected)
    verify_model(torch.nn.Flatten(2, -1), input_info, expected)


def test_batchnorm2d():
    """test graph builder for batchnorm2d"""

    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = torch.nn.BatchNorm2d(3)

        def forward(self, data):
            return self.batchnorm(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "batch_norm.0",
                "shape": [1, 3, 10, 10],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 3, "input": 1, "nn.batch_norm": 1, "get_item": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(BatchNorm2d(), input_info, expected)


def test_embedding():
    """test graph builder for embedding"""

    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, data):
            return self.embedding(data)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [4], "dtype": "int64", "layout": "A"}],
        "outputs": [{"name": "take", "shape": [4, 3], "dtype": "float32", "layout": "NA"}],
        "nodes": {"total": 2, "input": 1, "msc.embedding": 1},
    }

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [4, 5], "dtype": "int64", "layout": "AB"}],
        "outputs": [{"name": "take", "shape": [4, 5, 3], "dtype": "float32", "layout": "CNB"}],
        "nodes": {"total": 2, "input": 1, "msc.embedding": 1},
    }

    verify_model(Embedding(), [([4], "int64")], expected1)
    verify_model(Embedding(), [([4, 5], "int64")], expected2)


def test_dropout():
    """test graph builder for dropout"""

    class Dropout1(Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, data):
            return self.dropout(data)

    class Dropout2(Module):
        def forward(self, data):
            return torch.dropout(data, 0.5, train=True)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 1, "input": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Dropout1(), input_info, expected)
    verify_model(Dropout2(), input_info, expected)


def test_layernorm():
    """test graph builder for layernorm"""

    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((10, 10))

        def forward(self, data):
            return self.layernorm(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "layer_norm", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.layer_norm": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(LayerNorm(), input_info, expected)


def test_functional_layernorm():
    """test graph builder for functional_layernorm"""

    class LayerNorm(Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(shape))
            self.bias = torch.nn.Parameter(torch.zeros(shape))

        def forward(self, data):
            return torch.nn.functional.layer_norm(
                data, self.weight.shape, self.weight, self.bias, 1e-5
            )

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "layer_norm", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.layer_norm": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(LayerNorm((10, 10)), input_info, expected)


def test_cross_entropy():
    """test graph builder for cross_entropy"""

    class CrossEntropy1(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [3, 2], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [3], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 4, "input": 2, "nn.log_softmax": 1, "nn.nll_loss": 1},
    }

    class CrossEntropy2(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2,)))
            self.loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [3, 2], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [3], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 5, "input": 2, "nn.log_softmax": 1, "constant": 1, "nn.nll_loss": 1},
    }

    class CrossEntropy3(Module):
        def __init__(self):
            super().__init__()
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=1, reduction="sum")

        def forward(self, logits, targets):
            return self.loss(logits, targets)

    expected3 = {
        "inputs": [
            {"name": "inp_0", "shape": [3, 2], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [3], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 4, "input": 2, "nn.log_softmax": 1, "nn.nll_loss": 1},
    }

    input_info = [([3, 2], "float32"), ([3], "int32")]
    verify_model(CrossEntropy1(), input_info, expected1)
    verify_model(CrossEntropy2(), input_info, expected2)
    verify_model(CrossEntropy3(), input_info, expected3)


def test_functional_cross_entropy():
    """test graph builder for functional_cross_entropy"""

    class CrossEntropy(Module):
        def forward(self, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [3, 10], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [3], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 4, "input": 2, "nn.log_softmax": 1, "nn.nll_loss": 1},
    }

    input_info = [([3, 10], "float32"), ([3], "int32")]
    verify_model(CrossEntropy(), input_info, expected)


def test_silu():
    """test graph builder for silu"""

    class SiLU(Module):
        def __init__(self):
            super().__init__()
            self.silu = torch.nn.SiLU()

        def forward(self, data):
            return self.silu(data)

    class SiLU2(Module):
        def forward(self, data):
            return torch.nn.functional.silu(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "silu", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.silu": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(SiLU(), input_info, expected)
    verify_model(SiLU2(), input_info, expected)


def test_groupnorm():
    """test graph builder for groupnorm"""

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(3, 3)

        def forward(self, data):
            return self.groupnorm(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "group_norm", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.group_norm": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(GroupNorm(), input_info, expected)


def test_softmax():
    """test graph builder for softmax"""

    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, data):
            return self.softmax(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "softmax", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.softmax": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Softmax(), input_info, expected)


def test_binary():
    """test graph builder for binary"""

    input_info1 = [([1, 3, 10, 10], "float32"), ([1, 3, 10, 10], "float32")]
    input_info2 = [([1, 3, 10, 10], "float32")]

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    expected_add1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [{"name": "add", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}],
        "nodes": {"total": 3, "input": 2, "add": 1},
    }

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    expected_add2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [{"name": "add", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}],
        "nodes": {"total": 3, "input": 1, "constant": 1, "add": 1},
    }

    verify_model(Add1(), input_info1, expected_add1)
    verify_model(Add2(), input_info2, expected_add2)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    expected_sub1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "subtract", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "subtract": 1},
    }

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    expected_sub2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "subtract", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "subtract": 1},
    }

    verify_model(Sub1(), input_info1, expected_sub1)
    verify_model(Sub2(), input_info2, expected_sub2)

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    expected_mul1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "multiply", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "multiply": 1},
    }

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    expected_mul2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "multiply", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "multiply": 1},
    }

    verify_model(Mul1(), input_info1, expected_mul1)
    verify_model(Mul2(), input_info2, expected_mul2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    expected_div1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "divide", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "divide": 1},
    }

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    expected_div2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "divide", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "divide": 1},
    }

    verify_model(TrueDiv1(), input_info1, expected_div1)
    verify_model(TrueDiv2(), input_info2, expected_div2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    expected_floordiv1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {
                "name": "floor_divide",
                "shape": [1, 3, 10, 10],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 3, "input": 2, "floor_divide": 1},
    }

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    expected_floordiv2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {
                "name": "floor_divide",
                "shape": [1, 3, 10, 10],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "floor_divide": 1},
    }

    verify_model(FloorDiv1(), input_info1, expected_floordiv1)
    verify_model(FloorDiv2(), input_info2, expected_floordiv2)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    expected_power1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "power", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "power": 1},
    }

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    expected_power2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "power", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "power": 1},
    }

    verify_model(Power1(), input_info1, expected_power1)
    verify_model(Power2(), input_info2, expected_power2)

    # LT
    class LT1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    expected_lt1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [{"name": "less", "shape": [1, 3, 10, 10], "dtype": "bool", "layout": "ABCD"}],
        "nodes": {"total": 3, "input": 2, "less": 1},
    }

    class LT2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    expected_lt2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [{"name": "less", "shape": [1, 3, 10, 10], "dtype": "bool", "layout": "ABCD"}],
        "nodes": {"total": 3, "input": 1, "constant": 1, "less": 1},
    }

    verify_model(LT1(), input_info1, expected_lt1)
    verify_model(LT2(), input_info2, expected_lt2)


def test_size():
    """test graph builder for size"""

    class Size(Module):
        def forward(self, data):
            return data.size()

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "shape", "shape": [4], "dtype": "int32", "layout": "O"}],
        "nodes": {"total": 2, "input": 1, "shape": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Size(), input_info, expected)


def test_squeeze():
    """test graph builder for squeeze"""

    class Squeeze1(Module):
        def forward(self, data):
            return data.squeeze(1)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [3, 1, 4, 1], "dtype": "float32", "layout": "ANBC"}],
        "outputs": [{"name": "squeeze", "shape": [3, 4, 1], "dtype": "float32", "layout": "ABC"}],
        "nodes": {"total": 2, "input": 1, "squeeze": 1},
    }

    class Squeeze2(Module):
        def forward(self, data):
            return data.squeeze()

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [3, 1, 4, 1], "dtype": "float32", "layout": "ANBC"}],
        "outputs": [{"name": "squeeze", "shape": [3, 4], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "squeeze": 1},
    }

    input_info = [([3, 1, 4, 1], "float32")]
    verify_model(Squeeze1(), input_info, expected1)
    verify_model(Squeeze2(), input_info, expected2)


def test_unsqueeze():
    """test graph builder for unsqueeze"""

    class Unsqueeze1(Module):
        def forward(self, data):
            return data.unsqueeze(1)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ACDE"}
        ],
        "outputs": [
            {
                "name": "expand_dims",
                "shape": [1, 1, 3, 10, 10],
                "dtype": "float32",
                "layout": "ABCDE",
            }
        ],
        "nodes": {"total": 2, "input": 1, "expand_dims": 1},
    }

    class Unsqueeze2(Module):
        def forward(self, data):
            return data.unsqueeze(-1)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCE"}
        ],
        "outputs": [
            {
                "name": "expand_dims",
                "shape": [1, 3, 10, 10, 1],
                "dtype": "float32",
                "layout": "ABCDE",
            }
        ],
        "nodes": {"total": 2, "input": 1, "expand_dims": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Unsqueeze1(), input_info, expected1)
    verify_model(Unsqueeze2(), input_info, expected2)


def test_getattr():
    """test graph builder for getattr"""

    class GetAttr1(Module):
        def forward(self, data):
            return data.shape

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "shape", "shape": [4], "dtype": "int32", "layout": "O"}],
        "nodes": {"total": 2, "input": 1, "shape": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(GetAttr1(), input_info, expected)


def test_getitem():
    """test graph builder for getitem"""

    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "reshape", "shape": [1, 1, 10, 3], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "strided_slice": 1, "reshape": 1},
    }

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [8, 16], "dtype": "float32", "layout": "AB"}],
        "outputs": [
            {"name": "reshape", "shape": [8, 1, 1, 16, 1], "dtype": "float32", "layout": "ANCHB"}
        ],
        "nodes": {"total": 3, "input": 1, "strided_slice": 1, "reshape": 1},
    }

    verify_model(Slice1(), [([1, 3, 10, 10], "float32")], expected1)
    verify_model(Slice2(), [([8, 16], "float32")], expected2)


def test_unary():
    """test graph builder for unary"""

    input_info = [([1, 3, 10, 10], "float32")]

    # sin
    class Sin(Module):
        def forward(self, data):
            return torch.sin(data)

    expected_sin = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [{"name": "sin", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}],
        "nodes": {"total": 2, "input": 1, "sin": 1},
    }

    verify_model(Sin(), input_info, expected_sin)

    # cos
    class Cos(Module):
        def forward(self, data):
            return torch.cos(data)

    expected_cos = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [{"name": "cos", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}],
        "nodes": {"total": 2, "input": 1, "cos": 1},
    }

    verify_model(Cos(), input_info, expected_cos)

    # exp
    class Exp(Module):
        def forward(self, data):
            return torch.exp(data)

    expected_exp = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [{"name": "exp", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}],
        "nodes": {"total": 2, "input": 1, "exp": 1},
    }

    verify_model(Exp(), input_info, expected_exp)

    # sqrt
    class Sqrt(Module):
        def forward(self, data):
            return torch.sqrt(data)

    expected_sqrt = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "sqrt", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "sqrt": 1},
    }

    verify_model(Sqrt(), input_info, expected_sqrt)

    # sigmoid
    class Sigmoid(Module):
        def forward(self, data):
            return torch.sigmoid(data)

    expected_sigmoid = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "sigmoid", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "sigmoid": 1},
    }

    verify_model(Sigmoid(), input_info, expected_sigmoid)

    # round
    class Round(Module):
        def forward(self, data):
            return torch.round(data)

    expected_round = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "round", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "round": 1},
    }

    verify_model(Round(), input_info, expected_round)


def test_gelu():
    """test graph builder for gelu"""

    class Gelu(Module):
        def forward(self, data):
            return torch.nn.functional.gelu(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "gelu", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.gelu": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Gelu(), input_info, expected)


def test_tanh():
    """test graph builder for tanh"""

    class Tanh(Module):
        def forward(self, data):
            return torch.tanh(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "tanh", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "tanh": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Tanh(), input_info, expected)


def test_clamp():
    """test graph builder for clamp"""

    class Clamp(Module):
        def forward(self, data):
            return torch.clamp(data, min=0.1, max=0.5)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "clip", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "clip": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Clamp(), input_info, expected)


def test_interpolate():
    """test graph builder for interpolate"""

    class Interpolate(Module):
        def forward(self, data):
            return torch.nn.functional.interpolate(data, (5, 5))

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "resize2d", "shape": [1, 3, 5, 5], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "image.resize2d": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Interpolate(), input_info, expected)


def test_addmm():
    """test graph builder for addmm"""

    class Addmm(Module):
        def forward(self, x_1, x_2, x_3):
            return torch.addmm(x_1, x_2, x_3)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": "NC"},
            {"name": "inp_1", "shape": [10, 10], "dtype": "float32", "layout": "NC"},
            {"name": "inp_2", "shape": [10, 10], "dtype": "float32", "layout": "IO"},
        ],
        "outputs": [{"name": "add", "shape": [10, 10], "dtype": "float32", "layout": "NC"}],
        "nodes": {"total": 5, "input": 3, "matmul": 1, "add": 1},
    }

    input_info = [
        ([10, 10], "float32"),
        ([10, 10], "float32"),
        ([10, 10], "float32"),
    ]
    verify_model(Addmm(), input_info, expected)


def test_split():
    """test graph builder for split"""

    class Split(Module):
        def forward(self, data):
            return torch.split(data, 1, dim=1)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "split_0", "shape": [1, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_1", "shape": [1, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_2", "shape": [1, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "nodes": {"total": 2, "input": 1, "split": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Split(), input_info, expected)


def test_cumsum():
    """test graph builder for cumsum"""

    class Cumsum(Module):
        def forward(self, data):
            return torch.cumsum(data, dim=1, dtype=torch.int32)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "cumsum", "shape": [1, 2, 3, 4], "dtype": "int32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "cumsum": 1},
    }

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Cumsum(), input_info, expected)


def test_chunk():
    """test graph builder for chunk"""

    class Chunk(Module):
        def forward(self, data):
            return torch.chunk(data, 3, dim=1)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "split_0", "shape": [1, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_1", "shape": [1, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_2", "shape": [1, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "nodes": {"total": 2, "input": 1, "split": 1},
    }

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Chunk(), input_info, expected)


def test_inplace_fill():
    """test graph builder for inplace_fill"""

    class InplaceFill(Module):
        def forward(self, data):
            data.fill_(1.5)
            return data

    expected = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "const", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "constant": 1},
    }

    verify_model(InplaceFill(), [([10, 10], "float32")], expected)


def test_arange():
    """test graph builder for arange"""

    class Arange(Module):
        def forward(self):
            return torch.arange(0, 20, dtype=torch.int32)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "const", "shape": [20], "dtype": "int32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "constant": 1},
    }

    verify_model(Arange(), [([10, 10], "float32")], expected)


def test_empty():
    """test graph builder for empty"""

    class Empty(Module):
        def forward(self):
            return torch.empty((10, 10), dtype=torch.float32)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "const", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "constant": 1},
    }

    verify_model(Empty(), [([10, 10], "float32")], expected)


def test_tensor():
    """test graph builder for tensor"""

    class Empty1(Module):
        def forward(self):
            return torch.tensor(3, dtype=torch.float32)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "const", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "constant": 1},
    }

    class Empty2(Module):
        def forward(self):
            return torch.tensor(3)

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "const", "shape": [], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "constant": 1},
    }

    verify_model(Empty1(), [([10, 10], "float32")], expected1)
    verify_model(Empty2(), [([10, 10], "float32")], expected2)


def test_tril():
    """test graph builder for tril"""

    class Tril(Module):
        def forward(self, data):
            return torch.tril(data, 1)

    class InplaceTril(Module):
        def forward(self, data):
            data.tril_(1)
            return data

    expected = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "tril", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "tril": 1},
    }

    input_info = [([10, 10], "float32")]
    verify_model(Tril(), input_info, expected)
    verify_model(InplaceTril(), input_info, expected)


def test_triu():
    """test graph builder for triu"""

    class Triu(Module):
        def forward(self, data):
            return torch.triu(data, 1)

    class InplaceTriu(Module):
        def forward(self, data):
            data.triu_(1)
            return data

    expected = {
        "inputs": [{"name": "inp_0", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "triu", "shape": [10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "triu": 1},
    }

    input_info = [([10, 10], "float32")]
    verify_model(Triu(), input_info, expected)
    verify_model(InplaceTriu(), input_info, expected)


def test_new_ones():
    """test graph builder for new_ones"""

    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "const", "shape": [1, 2, 3], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "constant": 1},
    }

    input_info = [([1, 2, 3], "float32")]
    verify_model(NewOnes(), input_info, expected)


def test_expand():
    """test graph builder for expand"""

    class Expand(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [
            {"name": "broadcast_to", "shape": [4, 2, 3, 4], "dtype": "float32", "layout": ""}
        ],
        "nodes": {"total": 2, "input": 1, "broadcast_to": 1},
    }

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Expand(), input_info, expected)


def test_reduce():
    """test graph builder for reduce"""

    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ANCB"}],
        "outputs": [{"name": "sum", "shape": [1, 4], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "sum": 1},
    }

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Sum(), input_info, expected)


def test_datatype():
    """test graph builder for datatype"""

    input_info = [([1, 2, 3, 4], "float32")]

    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }

    verify_model(ToFloat(), input_info, expected1)

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [1, 2, 3, 4], "dtype": "float16", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }

    verify_model(ToHalf(), input_info, expected2)

    # type
    class Type(Module):
        def forward(self, x):
            return x.type(torch.float32)

    expected3 = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }

    # type
    class TypeFromAttr(Module):
        def forward(self, x):
            return x.type(x.getattr("dtype"))

    expected4 = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }

    # astype
    class AsType(Module):
        def forward(self, x):
            return x.astype(torch.float32)

    expected5 = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }

    verify_model(Type(), input_info, expected3)
    verify_model(TypeFromAttr(), input_info, expected4)
    verify_model(AsType(), input_info, expected5)


def test_permute():
    """test graph builder for permute"""

    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ADCB"}],
        "outputs": [
            {"name": "permute_dims", "shape": [1, 4, 3, 2], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "permute_dims": 1},
    }

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Permute(), input_info, expected)


def test_reshape():
    """test graph builder for reshape"""

    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "reshape", "shape": [2, 12], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "reshape": 1},
    }

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Reshape(), input_info, expected)


def test_transpose():
    """test graph builder for transpose"""

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": "ADCB"}],
        "outputs": [
            {"name": "permute_dims", "shape": [1, 4, 3, 2], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "permute_dims": 1},
    }

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(Transpose(), input_info, expected)


def test_view():
    """test graph builder for view"""

    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [1, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "reshape", "shape": [2, 12], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "reshape": 1},
    }

    input_info = [([1, 2, 3, 4], "float32")]
    verify_model(View(), input_info, expected)


def test_keep_params():
    """test graph builder for keep_params"""

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "conv2d",
                "shape": [1, 6, 4, 4],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "msc.conv2d_bias": 1},
    }

    verify_model(Conv2D1(), [([1, 3, 10, 10], "float32")], expected)


def test_unwrap_unit_return_tuple():
    """test graph builder for unwrap_unit_return_tuple"""

    class Identity(Module):
        def forward(self, x):
            return (x,)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "tuple", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "tuple": 1},
    }

    verify_model(Identity(), [([256, 256], "float32")], expected)


def test_no_bind_return_tuple():
    """test graph builder for no_bind_return_tuple"""

    class Identity(Module):
        def forward(self, x, y):
            return (x, y)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [256, 256], "dtype": "float32", "layout": ""},
        ],
        "outputs": [
            {"name": "tuple_0", "shape": [256, 256], "dtype": "float32", "layout": ""},
            {"name": "tuple_1", "shape": [256, 256], "dtype": "float32", "layout": ""},
        ],
        "nodes": {"total": 3, "input": 2, "tuple": 1},
    }

    input_info = [([256, 256], "float32"), ([256, 256], "float32")]
    verify_model(Identity(), input_info, expected)


def test_argmax():
    """test graph builder for argmax"""

    class Argmax1(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmax", "shape": [256], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmax": 1},
    }

    class Argmax2(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1, keepdim=True)

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmax", "shape": [256, 1], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmax": 1},
    }

    verify_model(Argmax1(), [([256, 256], "float32")], expected1)
    verify_model(Argmax2(), [([256, 256], "float32")], expected2)


def test_argmin():
    """test graph builder for argmin"""

    class Argmin1(Module):
        def forward(self, data):
            return torch.argmin(data)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmin", "shape": [], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmin": 1},
    }

    class Argmin2(Module):
        def forward(self, data):
            return torch.argmin(data, keepdim=True)

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmin", "shape": [1, 1], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmin": 1},
    }

    verify_model(Argmin1(), [([256, 256], "float32")], expected1)
    verify_model(Argmin2(), [([256, 256], "float32")], expected2)


def test_to():
    """test graph builder for to"""

    class To1(Module):
        def forward(self, data):
            return data.to(torch.float16)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "astype", "shape": [256, 256], "dtype": "float16", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }

    class To2(Module):
        def forward(self, data):
            return data.to("cpu")

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 1, "input": 1},
    }

    verify_model(To1(), [([256, 256], "float32")], expected1)
    verify_model(To2(), [([256, 256], "float32")], expected2)


def test_mean():
    """test graph builder for mean"""

    class Mean(Module):
        def forward(self, data):
            return data.mean(-1)

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": "AN"}],
        "outputs": [{"name": "mean", "shape": [256], "dtype": "float32", "layout": "A"}],
        "nodes": {"total": 2, "input": 1, "mean": 1},
    }

    class MeanKeepDim(Module):
        def forward(self, data):
            return data.mean(-1, keepdim=True)

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "mean", "shape": [256, 1], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "mean": 1},
    }

    verify_model(Mean(), [([256, 256], "float32")], expected1)
    verify_model(MeanKeepDim(), [([256, 256], "float32")], expected2)


def test_rsqrt():
    """test graph builder for rsqrt"""

    class Rsqrt(Module):
        def forward(self, data):
            return torch.rsqrt(data)

    expected = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "rsqrt", "shape": [256, 256], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "rsqrt": 1},
    }

    verify_model(Rsqrt(), [([256, 256], "float32")], expected)


def test_neg():
    """test graph builder for neg"""

    class Neg(Module):
        def forward(self, data):
            return -data

    expected = {
        "inputs": [{"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "negative", "shape": [256, 256], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "negative": 1},
    }

    verify_model(Neg(), [([256, 256], "float32")], expected)


def test_max():
    """test graph builder for max"""

    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [256, 256], "dtype": "float32", "layout": "AB"},
            {"name": "inp_1", "shape": [256, 256], "dtype": "float32", "layout": "AB"},
        ],
        "outputs": [{"name": "maximum", "shape": [256, 256], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 3, "input": 2, "maximum": 1},
    }

    verify_model(Max(), [([256, 256], "float32"), ([256, 256], "float32")], expected)


def test_attention():
    """test graph builder for attention"""

    # pylint: disable=import-outside-toplevel
    import torch.nn.functional as F

    class Attention1(Module):
        def forward(self, q_data, k_data, v_data):
            return F.scaled_dot_product_attention(q_data, k_data, v_data)

    class Attention2(Module):
        def forward(self, q_data, k_data, v_data):
            return F.scaled_dot_product_attention(q_data, k_data, v_data, is_causal=True)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [32, 8, 128, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_1", "shape": [32, 8, 128, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_2", "shape": [32, 8, 128, 64], "dtype": "float32", "layout": "ACBD"},
        ],
        "outputs": [
            {
                "name": "attention",
                "shape": [32, 128, 8, 64],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 4, "input": 3, "msc.attention": 1},
    }

    input_info = [
        ([32, 8, 128, 64], "float32"),
        ([32, 8, 128, 64], "float32"),
        ([32, 8, 128, 64], "float32"),
    ]
    verify_model(Attention1(), input_info, expected1)
    verify_model(Attention2(), input_info, expected1)

    class Attention3(Module):
        def forward(self, q_data, k_data, v_data, mask):
            return F.scaled_dot_product_attention(q_data, k_data, v_data, mask)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [32, 8, 128, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_1", "shape": [32, 8, 128, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_2", "shape": [32, 8, 128, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_3", "shape": [32, 8, 128, 128], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {
                "name": "attention_bias",
                "shape": [32, 128, 8, 64],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 5, "input": 4, "msc.attention": 1},
    }
    verify_model(
        Attention3(),
        [
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 64], "float32"),
            ([32, 8, 128, 128], "float32"),
        ],
        expected2,
    )


if __name__ == "__main__":
    tvm.testing.main()
