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
# pylint: disable=invalid-name

""" Test graph builder && graph. """

import pytest
import torch
from torch.nn import Module

import tvm.testing
from tvm.contrib.msc.framework.torch.frontend import translate
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


def verify_model(torch_model, input_info, expected):
    graph, _ = translate.from_torch(torch_model, input_info)
    inspect = graph.inspect()
    assert msc_utils.dict_equal(inspect, expected), "Inspect {} mismatch with expected {}".format(
        inspect, expected
    )


@pytest.mark.parametrize("dynamic", [True, False])
def test_conv1d(dynamic):
    """test graph builder for conv1d"""

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

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 3, 10], "dtype": "float32", "layout": "NCW"}],
        "outputs": [{"name": "conv1d", "shape": [bz, 6, 4], "dtype": "float32", "layout": "NCW"}],
        "nodes": {"total": 2, "input": 1, "msc.conv1d_bias": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 3, 10], "dtype": "float32", "layout": "NCW"}],
        "outputs": [{"name": "conv1d", "shape": [bz, 6, 4], "dtype": "float32", "layout": "NCW"}],
        "nodes": {"total": 2, "input": 1, "nn.conv1d": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10], "float32")]
    verify_model(Conv1D1(), input_info, expected1)
    verify_model(Conv1D2(), input_info, expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_conv2d(dynamic):
    """test graph builder for conv2d"""

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

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "conv2d",
                "shape": [bz, 6, 4, 4],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "msc.conv2d_bias": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "conv2d", "shape": [bz, 6, 4, 4], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.conv2d": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Conv2D1(), input_info, expected1)
    verify_model(Conv2D2(), input_info, expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_linear(dynamic):
    """test graph builder for linear"""

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

    bz = "bz" if dynamic else 1
    mdim = "mdim" if dynamic else 10
    ndim = "ndim" if dynamic else 20
    kdim = "kdim" if dynamic else 30

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "matmul",
                "shape": [bz, 3, 10, 7],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "msc.linear_bias": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "matmul", "shape": [bz, 3, 10, 7], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "msc.linear": 1},
    }
    expected3 = {
        "inputs": [
            {"name": "inp_0", "shape": [mdim, kdim], "dtype": "float32", "layout": "NC"},
            {"name": "inp_1", "shape": [kdim, ndim], "dtype": "float32", "layout": "IO"},
        ],
        "outputs": [{"name": "matmul", "shape": [mdim, ndim], "dtype": "float32", "layout": "NC"}],
        "nodes": {"total": 3, "input": 2, "matmul": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}
        expected3["prims"] = {"total": 3, "shape": 3}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Dense1(), input_info, expected1)
    verify_model(Dense2(), input_info, expected2)
    verify_model(MatMul1(), [([mdim, kdim], "float32"), ([kdim, ndim], "float32")], expected3)


@pytest.mark.parametrize("dynamic", [True, False])
def test_bmm(dynamic):
    """test graph builder for bmm"""

    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 128, 256], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_1", "shape": [bz, 256, 512], "dtype": "float32", "layout": "NIO"},
        ],
        "outputs": [
            {"name": "matmul", "shape": [bz, 128, 512], "dtype": "float32", "layout": "NCD"}
        ],
        "nodes": {"total": 3, "input": 2, "matmul": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [((bz, 128, 256), "float32"), ((bz, 256, 512), "float32")]
    verify_model(BMM(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_baddbmm(dynamic):
    """test graph builder for baddbmm"""

    class BAddBMM1(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    class BAddBMM2(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 128, 512], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_1", "shape": [bz, 128, 256], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_2", "shape": [bz, 256, 512], "dtype": "float32", "layout": "NIO"},
        ],
        "outputs": [{"name": "add", "shape": [bz, 128, 512], "dtype": "float32", "layout": "NCD"}],
        "nodes": {"total": 5, "input": 3, "matmul": 1, "add": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 128, 512], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz, 128, 256], "dtype": "float32", "layout": "NCD"},
            {"name": "inp_2", "shape": [bz, 256, 512], "dtype": "float32", "layout": "NIO"},
        ],
        "outputs": [
            {"name": "multiply", "shape": [bz, 128, 512], "dtype": "float32", "layout": "NCD"}
        ],
        "nodes": {"total": 6, "input": 3, "matmul": 1, "constant": 1, "multiply": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    input_info = [
        ((bz, 128, 512), "float32"),
        ((bz, 128, 256), "float32"),
        ((bz, 256, 512), "float32"),
    ]
    verify_model(BAddBMM1(), input_info, expected1)
    verify_model(BAddBMM2(), input_info, expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_relu(dynamic):
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

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 10], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "relu", "shape": [bz, 10], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "nn.relu": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 10], "float32")]
    verify_model(ReLU(), input_info, expected)
    verify_model(ReLU1(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_relu6(dynamic):
    """test graph builder for relu6"""

    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, data):
            return self.relu6(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "clip", "shape": [bz, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "clip": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 10], "float32")]
    verify_model(ReLU6(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_maxpool2d(dynamic):
    """test graph builder for maxpool2d"""

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

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "max_pool2d", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.max_pool2d": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "max_pool2d", "shape": [bz, 3, 4, 4], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.max_pool2d": 1},
    }
    expected3 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "max_pool2d", "shape": [bz, 3, 6, 6], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.max_pool2d": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}
        expected3["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(MaxPool2d(), input_info, expected1)
    verify_model(MaxPool2d2(), input_info, expected2)
    verify_model(MaxPool2d3(), input_info, expected3)


@pytest.mark.parametrize("dynamic", [True, False])
def test_avgpool2d(dynamic):
    """test graph builder for avgpool2d"""

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

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "avg_pool2d", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.avg_pool2d": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "avg_pool2d", "shape": [bz, 3, 6, 6], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.avg_pool2d": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(AvgPool2d(), input_info, expected1)
    verify_model(AvgPool2d2(), input_info, expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_adaptive_avgpool2d(dynamic):
    """test graph builder for adaptive_avgpool2d"""

    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, data):
            return self.pool(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "adaptive_avg_pool2d",
                "shape": [bz, 3, 10, 10],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "nn.adaptive_avg_pool2d": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(AdaptiveAvgPool2d0(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_flatten(dynamic):
    """test graph builder for flatten"""

    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, data):
            return self.f(data)

    bz = "bz" if dynamic else 1
    dim = "dim" if dynamic else 10
    out_dim = "MUL_3" if dynamic else 100
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 3, 10, dim], "dtype": "float32", "layout": ""}],
        "outputs": [
            {"name": "reshape", "shape": [bz, 3, out_dim], "dtype": "float32", "layout": ""}
        ],
        "nodes": {"total": 2, "input": 1, "reshape": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 4, "shape": 2, "Int": 1, "Mul": 1}

    input_info = [([bz, 3, 10, dim], "float32")]
    verify_model(Flatten(), input_info, expected)
    verify_model(torch.nn.Flatten(2, -1), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_batchnorm2d(dynamic):
    """test graph builder for batchnorm2d"""

    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = torch.nn.BatchNorm2d(3)

        def forward(self, data):
            return self.batchnorm(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "batch_norm.0",
                "shape": [bz, 3, 10, 10],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 3, "input": 1, "nn.batch_norm": 1, "get_item": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(BatchNorm2d(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_embedding(dynamic):
    """test graph builder for embedding"""

    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, data):
            return self.embedding(data)

    vocab = "vocab" if dynamic else 4
    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [vocab], "dtype": "int64", "layout": "A"}],
        "outputs": [{"name": "take", "shape": [vocab, 3], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "msc.embedding": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [vocab, 5], "dtype": "int64", "layout": "AB"}],
        "outputs": [
            {
                "name": "take",
                "shape": [vocab, 5, 3],
                "dtype": "float32",
                "layout": "" if dynamic else "CBA",
            }
        ],
        "nodes": {"total": 2, "input": 1, "msc.embedding": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 3, "shape": 1, "Int": 1, "Mul": 1}

    verify_model(Embedding(), [([vocab], "int64")], expected1)
    verify_model(Embedding(), [([vocab, 5], "int64")], expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_dropout(dynamic):
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

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 1, "input": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Dropout1(), input_info, expected)
    verify_model(Dropout2(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_layernorm(dynamic):
    """test graph builder for layernorm"""

    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm((10, 10))

        def forward(self, data):
            return self.layernorm(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "layer_norm", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.layer_norm": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(LayerNorm(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_functional_layernorm(dynamic):
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

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "layer_norm", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.layer_norm": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(LayerNorm((10, 10)), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_cross_entropy(dynamic):
    """test graph builder for cross_entropy"""

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

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 2], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 4, "input": 2, "nn.log_softmax": 1, "nn.nll_loss": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 2], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 5, "input": 2, "nn.log_softmax": 1, "constant": 1, "nn.nll_loss": 1},
    }
    expected3 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 2], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 4, "input": 2, "nn.log_softmax": 1, "nn.nll_loss": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}
        expected3["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 2], "float32"), ([bz], "int32")]
    verify_model(CrossEntropy1(), input_info, expected1)
    verify_model(CrossEntropy2(), input_info, expected2)
    verify_model(CrossEntropy3(), input_info, expected3)


@pytest.mark.parametrize("dynamic", [True, False])
def test_functional_cross_entropy(dynamic):
    """test graph builder for functional_cross_entropy"""

    class CrossEntropy(Module):
        def forward(self, logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 10], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz], "dtype": "int32", "layout": ""},
        ],
        "outputs": [{"name": "nll_loss", "shape": [], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 4, "input": 2, "nn.log_softmax": 1, "nn.nll_loss": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 10], "float32"), ([bz], "int32")]
    verify_model(CrossEntropy(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_silu(dynamic):
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

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "silu", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.silu": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(SiLU(), input_info, expected)
    verify_model(SiLU2(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_groupnorm(dynamic):
    """test graph builder for groupnorm"""

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.groupnorm = torch.nn.GroupNorm(3, 3)

        def forward(self, data):
            return self.groupnorm(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "group_norm", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.group_norm": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(GroupNorm(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_softmax(dynamic):
    """test graph builder for softmax"""

    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, data):
            return self.softmax(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "softmax", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.softmax": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Softmax(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_binary(dynamic):
    """test graph builder for binary"""

    bz = "bz" if dynamic else 1
    input_info1 = [([bz, 3, 10, 10], "float32"), ([bz, 3, 10, 10], "float32")]
    input_info2 = [([bz, 3, 10, 10], "float32")]

    # Add
    class Add1(Module):
        def forward(self, lhs, rhs):
            return lhs + rhs

    class Add2(Module):
        def forward(self, lhs):
            return lhs + 1.0

    expected_add1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "add", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "add": 1},
    }
    expected_add2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "add", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "add": 1},
    }
    if dynamic:
        expected_add1["prims"] = {"total": 1, "shape": 1}
        expected_add2["prims"] = {"total": 1, "shape": 1}

    verify_model(Add1(), input_info1, expected_add1)
    verify_model(Add2(), input_info2, expected_add2)

    # Sub
    class Sub1(Module):
        def forward(self, lhs, rhs):
            return lhs - rhs

    class Sub2(Module):
        def forward(self, lhs):
            return lhs - 1.0

    expected_sub1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "subtract", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "subtract": 1},
    }
    expected_sub2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "subtract", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "subtract": 1},
    }
    if dynamic:
        expected_sub1["prims"] = {"total": 1, "shape": 1}
        expected_sub2["prims"] = {"total": 1, "shape": 1}

    verify_model(Sub1(), input_info1, expected_sub1)
    verify_model(Sub2(), input_info2, expected_sub2)

    # Mul
    class Mul1(Module):
        def forward(self, lhs, rhs):
            return lhs * rhs

    class Mul2(Module):
        def forward(self, lhs):
            return lhs * 1.0

    expected_mul1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "multiply", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "multiply": 1},
    }
    expected_mul2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "multiply", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "multiply": 1},
    }
    if dynamic:
        expected_mul1["prims"] = {"total": 1, "shape": 1}
        expected_mul2["prims"] = {"total": 1, "shape": 1}

    verify_model(Mul1(), input_info1, expected_mul1)
    verify_model(Mul2(), input_info2, expected_mul2)

    # True div
    class TrueDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs / rhs

    class TrueDiv2(Module):
        def forward(self, lhs):
            return lhs / 1.0

    expected_div1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "divide", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "divide": 1},
    }
    expected_div2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "divide", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "divide": 1},
    }
    if dynamic:
        expected_div1["prims"] = {"total": 1, "shape": 1}
        expected_div2["prims"] = {"total": 1, "shape": 1}

    verify_model(TrueDiv1(), input_info1, expected_div1)
    verify_model(TrueDiv2(), input_info2, expected_div2)

    # Floor div
    class FloorDiv1(Module):
        def forward(self, lhs, rhs):
            return lhs // rhs

    class FloorDiv2(Module):
        def forward(self, lhs):
            return lhs // 1.0

    expected_floordiv1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {
                "name": "floor_divide",
                "shape": [bz, 3, 10, 10],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 3, "input": 2, "floor_divide": 1},
    }
    expected_floordiv2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {
                "name": "floor_divide",
                "shape": [bz, 3, 10, 10],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "floor_divide": 1},
    }
    if dynamic:
        expected_floordiv1["prims"] = {"total": 1, "shape": 1}
        expected_floordiv2["prims"] = {"total": 1, "shape": 1}

    verify_model(FloorDiv1(), input_info1, expected_floordiv1)
    verify_model(FloorDiv2(), input_info2, expected_floordiv2)

    # Power
    class Power1(Module):
        def forward(self, lhs, rhs):
            return lhs**rhs

    class Power2(Module):
        def forward(self, lhs):
            return lhs**1.0

    expected_power1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {"name": "power", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 2, "power": 1},
    }
    expected_power2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "power", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 3, "input": 1, "constant": 1, "power": 1},
    }
    if dynamic:
        expected_power1["prims"] = {"total": 1, "shape": 1}
        expected_power2["prims"] = {"total": 1, "shape": 1}

    verify_model(Power1(), input_info1, expected_power1)
    verify_model(Power2(), input_info2, expected_power2)

    # LT
    class LT1(Module):
        def forward(self, lhs, rhs):
            return lhs < rhs

    class LT2(Module):
        def forward(self, lhs):
            return lhs < 1.0

    expected_lt1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [{"name": "less", "shape": [bz, 3, 10, 10], "dtype": "bool", "layout": "ABCD"}],
        "nodes": {"total": 3, "input": 2, "less": 1},
    }
    expected_lt2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [{"name": "less", "shape": [bz, 3, 10, 10], "dtype": "bool", "layout": "ABCD"}],
        "nodes": {"total": 3, "input": 1, "constant": 1, "less": 1},
    }
    if dynamic:
        expected_lt1["prims"] = {"total": 1, "shape": 1}
        expected_lt2["prims"] = {"total": 1, "shape": 1}

    verify_model(LT1(), input_info1, expected_lt1)
    verify_model(LT2(), input_info2, expected_lt2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_size(dynamic):
    """test graph builder for size"""

    class Size(Module):
        def forward(self, data):
            return data.size()

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "shape", "shape": [4], "dtype": "int32", "layout": "O"}],
        "nodes": {"total": 2, "input": 1, "shape": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Size(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_squeeze(dynamic):
    """test graph builder for squeeze"""

    class Squeeze1(Module):
        def forward(self, data):
            return data.squeeze(1)

    class Squeeze2(Module):
        def forward(self, data):
            return data.squeeze()

    bz = "bz" if dynamic else 10
    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 1, 4, 1], "dtype": "float32", "layout": "ADBC"}],
        "outputs": [{"name": "squeeze", "shape": [bz, 4, 1], "dtype": "float32", "layout": "ABC"}],
        "nodes": {"total": 2, "input": 1, "squeeze": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2 = {
            "inputs": [
                {"name": "inp_0", "shape": [bz, 1, 4, 1], "dtype": "float32", "layout": "ACBD"}
            ],
            "outputs": [{"name": "squeeze", "shape": [], "dtype": "float32", "layout": "AB"}],
            "nodes": {"total": 2, "input": 1, "squeeze": 1},
            "prims": {"total": 1, "shape": 1},
        }
    else:
        expected2 = {
            "inputs": [
                {"name": "inp_0", "shape": [bz, 1, 4, 1], "dtype": "float32", "layout": "ACBD"}
            ],
            "outputs": [{"name": "squeeze", "shape": [bz, 4], "dtype": "float32", "layout": "AB"}],
            "nodes": {"total": 2, "input": 1, "squeeze": 1},
        }
    input_info = [([bz, 1, 4, 1], "float32")]
    verify_model(Squeeze1(), input_info, expected1)
    verify_model(Squeeze2(), input_info, expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_unsqueeze(dynamic):
    """test graph builder for unsqueeze"""

    class Unsqueeze1(Module):
        def forward(self, data):
            return data.unsqueeze(1)

    class Unsqueeze2(Module):
        def forward(self, data):
            return data.unsqueeze(-1)

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ACDE"}
        ],
        "outputs": [
            {
                "name": "expand_dims",
                "shape": [bz, 1, 3, 10, 10],
                "dtype": "float32",
                "layout": "ABCDE",
            }
        ],
        "nodes": {"total": 2, "input": 1, "expand_dims": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCE"}
        ],
        "outputs": [
            {
                "name": "expand_dims",
                "shape": [bz, 3, 10, 10, 1],
                "dtype": "float32",
                "layout": "ABCDE",
            }
        ],
        "nodes": {"total": 2, "input": 1, "expand_dims": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Unsqueeze1(), input_info, expected1)
    verify_model(Unsqueeze2(), input_info, expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_getattr(dynamic):
    """test graph builder for getattr"""

    class GetAttr1(Module):
        def forward(self, data):
            return data.shape

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "shape", "shape": [4], "dtype": "int32", "layout": "O"}],
        "nodes": {"total": 2, "input": 1, "shape": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(GetAttr1(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_getitem(dynamic):
    """test graph builder for getitem"""

    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {
                "name": "reshape",
                "shape": ["MIN_2" if dynamic else 1, 1, 10, 3],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 3, "input": 1, "strided_slice": 1, "reshape": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 16], "dtype": "float32", "layout": "AB"}],
        "outputs": [
            {"name": "reshape", "shape": [bz, 1, 1, 16, 1], "dtype": "float32", "layout": "CDAEB"}
        ],
        "nodes": {"total": 3, "input": 1, "strided_slice": 1, "reshape": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 3, "shape": 1, "Int": 1, "Min": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(Slice1(), [([bz, 3, 10, 10], "float32")], expected1)
    verify_model(Slice2(), [([bz, 16], "float32")], expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_unary(dynamic):
    """test graph builder for unary"""

    bz = "bz" if dynamic else 1
    input_info = [([bz, 3, 10, 10], "float32")]

    # sin
    class Sin(Module):
        def forward(self, data):
            return torch.sin(data)

    expected_sin = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "sin", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "sin": 1},
    }
    if dynamic:
        expected_sin["prims"] = {"total": 1, "shape": 1}

    verify_model(Sin(), input_info, expected_sin)

    # cos
    class Cos(Module):
        def forward(self, data):
            return torch.cos(data)

    expected_cos = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "cos", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "cos": 1},
    }
    if dynamic:
        expected_cos["prims"] = {"total": 1, "shape": 1}

    verify_model(Cos(), input_info, expected_cos)

    # exp
    class Exp(Module):
        def forward(self, data):
            return torch.exp(data)

    expected_exp = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "exp", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "exp": 1},
    }
    if dynamic:
        expected_exp["prims"] = {"total": 1, "shape": 1}

    verify_model(Exp(), input_info, expected_exp)

    # sqrt
    class Sqrt(Module):
        def forward(self, data):
            return torch.sqrt(data)

    expected_sqrt = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "sqrt", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "sqrt": 1},
    }
    if dynamic:
        expected_sqrt["prims"] = {"total": 1, "shape": 1}

    verify_model(Sqrt(), input_info, expected_sqrt)

    # sigmoid
    class Sigmoid(Module):
        def forward(self, data):
            return torch.sigmoid(data)

    expected_sigmoid = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "sigmoid", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "sigmoid": 1},
    }
    if dynamic:
        expected_sigmoid["prims"] = {"total": 1, "shape": 1}

    verify_model(Sigmoid(), input_info, expected_sigmoid)

    # round
    class Round(Module):
        def forward(self, data):
            return torch.round(data)

    expected_round = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "round", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "round": 1},
    }
    if dynamic:
        expected_round["prims"] = {"total": 1, "shape": 1}

    verify_model(Round(), input_info, expected_round)


@pytest.mark.parametrize("dynamic", [True, False])
def test_gelu(dynamic):
    """test graph builder for gelu"""

    class Gelu(Module):
        def forward(self, data):
            return torch.nn.functional.gelu(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "gelu", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "nn.gelu": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Gelu(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_tanh(dynamic):
    """test graph builder for tanh"""

    class Tanh(Module):
        def forward(self, data):
            return torch.tanh(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "tanh", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "tanh": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Tanh(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_clamp(dynamic):
    """test graph builder for clamp"""

    class Clamp(Module):
        def forward(self, data):
            return torch.clamp(data, min=0.1, max=0.5)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "clip", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "clip": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Clamp(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_interpolate(dynamic):
    """test graph builder for interpolate"""

    class Interpolate(Module):
        def forward(self, data):
            return torch.nn.functional.interpolate(data, (5, 5))

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {"name": "resize2d", "shape": [bz, 3, 5, 5], "dtype": "float32", "layout": "NCHW"}
        ],
        "nodes": {"total": 2, "input": 1, "image.resize2d": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Interpolate(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_addmm(dynamic):
    """test graph builder for addmm"""

    class Addmm(Module):
        def forward(self, x_1, x_2, x_3):
            return torch.addmm(x_1, x_2, x_3)

    mdim = "mdim" if dynamic else 10
    ndim = "ndim" if dynamic else 20
    kdim = "kdim" if dynamic else 30
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [mdim, ndim], "dtype": "float32", "layout": "NC"},
            {"name": "inp_1", "shape": [mdim, kdim], "dtype": "float32", "layout": "NC"},
            {"name": "inp_2", "shape": [kdim, ndim], "dtype": "float32", "layout": "IO"},
        ],
        "outputs": [{"name": "add", "shape": [mdim, ndim], "dtype": "float32", "layout": "NC"}],
        "nodes": {"total": 5, "input": 3, "matmul": 1, "add": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 3, "shape": 3}

    input_info = [([mdim, ndim], "float32"), ([mdim, kdim], "float32"), ([kdim, ndim], "float32")]
    verify_model(Addmm(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_split(dynamic):
    """test graph builder for split"""

    class Split1(Module):
        def forward(self, data):
            return torch.split(data, 1, dim=1)

    class Split2(Module):
        def forward(self, data):
            return torch.split(data, [1, 2], dim=1)

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "split_0", "shape": [bz, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_1", "shape": [bz, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_2", "shape": [bz, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "nodes": {"total": 2, "input": 1, "split": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "split_0", "shape": [bz, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_1", "shape": [bz, 2, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "nodes": {"total": 2, "input": 1, "split": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Split1(), input_info, expected1)
    verify_model(Split2(), input_info, expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_unbind(dynamic):
    """test graph builder for unbind"""

    class Unbind(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "tuple_0", "shape": [bz, 10, 10], "dtype": "float32", "layout": "ACD"},
            {"name": "tuple_1", "shape": [bz, 10, 10], "dtype": "float32", "layout": "ACD"},
            {"name": "tuple_2", "shape": [bz, 10, 10], "dtype": "float32", "layout": "ACD"},
        ],
        "nodes": {"total": 9, "input": 1, "split": 1, "get_item": 3, "squeeze": 3, "tuple": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Unbind(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_cumsum(dynamic):
    """test graph builder for cumsum"""

    class Cumsum(Module):
        def forward(self, data):
            return torch.cumsum(data, dim=1, dtype=torch.int32)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "cumsum", "shape": [bz, 2, 3, 4], "dtype": "int32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "cumsum": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 2, 3, 4], "float32")]
    verify_model(Cumsum(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_chunk(dynamic):
    """test graph builder for chunk"""

    class Chunk(Module):
        def forward(self, data):
            return torch.chunk(data, 3, dim=1)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "outputs": [
            {"name": "split_0", "shape": [bz, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_1", "shape": [bz, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
            {"name": "split_2", "shape": [bz, 1, 10, 10], "dtype": "float32", "layout": "ABCD"},
        ],
        "nodes": {"total": 2, "input": 1, "split": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 3, 10, 10], "float32")]
    verify_model(Chunk(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_inplace_fill(dynamic):
    """test graph builder for inplace_fill"""

    class InplaceFill(Module):
        def forward(self, data):
            data.fill_(1.5)
            return data

    bz = "bz" if dynamic else 1
    if dynamic:
        expected = {
            "inputs": [{"name": "inp_0", "shape": [bz, 10], "dtype": "float32", "layout": ""}],
            "outputs": [{"name": "full", "shape": [bz, 10], "dtype": "float32", "layout": ""}],
            "nodes": {"total": 3, "input": 1, "constant": 1, "full": 1},
            "prims": {"total": 1, "shape": 1},
        }
    else:
        expected = {
            "inputs": [{"name": "inp_0", "shape": [bz, 10], "dtype": "float32", "layout": ""}],
            "outputs": [{"name": "const", "shape": [bz, 10], "dtype": "float32", "layout": ""}],
            "nodes": {"total": 2, "input": 1, "constant": 1},
        }
    verify_model(InplaceFill(), [([bz, 10], "float32")], expected)


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


@pytest.mark.parametrize("dynamic", [True, False])
def test_tril(dynamic):
    """test graph builder for tril"""

    class Tril(Module):
        def forward(self, data):
            return torch.tril(data, 1)

    class InplaceTril(Module):
        def forward(self, data):
            data.tril_(1)
            return data

    row = "row" if dynamic else 10
    col = "col" if dynamic else 10
    expected = {
        "inputs": [{"name": "inp_0", "shape": [row, col], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "tril", "shape": [row, col], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "tril": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 2, "shape": 2}

    input_info = [([row, col], "float32")]
    verify_model(Tril(), input_info, expected)
    verify_model(InplaceTril(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_triu(dynamic):
    """test graph builder for triu"""

    class Triu(Module):
        def forward(self, data):
            return torch.triu(data, 1)

    class InplaceTriu(Module):
        def forward(self, data):
            data.triu_(1)
            return data

    row = "row" if dynamic else 10
    col = "col" if dynamic else 10
    expected = {
        "inputs": [{"name": "inp_0", "shape": [row, col], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "triu", "shape": [row, col], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "triu": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 2, "shape": 2}

    input_info = [([row, col], "float32")]
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


@pytest.mark.parametrize("dynamic", [True, False])
def test_expand(dynamic):
    """test graph builder for expand"""

    class Expand1(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    class Expand2(Module):
        def forward(self, x):
            return x.expand(4, -1, -1, 4)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [
            {"name": "broadcast_to", "shape": [4, 2, 3, 4], "dtype": "float32", "layout": ""}
        ],
        "nodes": {"total": 2, "input": 1, "broadcast_to": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 2, 3, 4], "float32")]
    verify_model(Expand1(), input_info, expected)
    verify_model(Expand2(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_reduce(dynamic):
    """test graph builder for reduce"""

    # sum
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ACDB"}],
        "outputs": [{"name": "sum", "shape": [bz, 4], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "sum": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz, 2, 3, 4], "float32")]
    verify_model(Sum(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_datatype(dynamic):
    """test graph builder for datatype"""

    bz = "bz" if dynamic else 1
    input_info = [([bz, 2, 3, 4], "float32")]

    # float
    class ToFloat(Module):
        def forward(self, x):
            return x.float()

    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}

    verify_model(ToFloat(), input_info, expected1)

    # half
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [bz, 2, 3, 4], "dtype": "float16", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }
    if dynamic:
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(ToHalf(), input_info, expected2)

    # type
    class Type(Module):
        def forward(self, x):
            return x.type(torch.float32)

    expected3 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }
    if dynamic:
        expected3["prims"] = {"total": 1, "shape": 1}

    # type
    class TypeFromAttr(Module):
        def forward(self, x):
            return x.type(x.getattr("dtype"))

    expected4 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }
    if dynamic:
        expected4["prims"] = {"total": 1, "shape": 1}

    # astype
    class AsType(Module):
        def forward(self, x):
            return x.astype(torch.float32)

    expected5 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}],
        "outputs": [
            {"name": "astype", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }
    if dynamic:
        expected5["prims"] = {"total": 1, "shape": 1}

    verify_model(Type(), input_info, expected3)
    verify_model(TypeFromAttr(), input_info, expected4)
    verify_model(AsType(), input_info, expected5)


@pytest.mark.parametrize("dynamic", [True, False])
def test_permute(dynamic):
    """test graph builder for permute"""

    class Permute(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    bz = "bz" if dynamic else 1
    channel = "channel" if dynamic else 2
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, channel, 3, 4], "dtype": "float32", "layout": "ADCB"}
        ],
        "outputs": [
            {
                "name": "permute_dims",
                "shape": [bz, 4, 3, channel],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 2, "input": 1, "permute_dims": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 2, "shape": 2}

    input_info = [([bz, channel, 3, 4], "float32")]
    verify_model(Permute(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_reshape(dynamic):
    """test graph builder for reshape"""

    class Reshape(Module):
        def forward(self, x):
            return x.reshape(-1, 12)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [
            {
                "name": "reshape",
                "shape": ["MUL_2" if dynamic else 2, 12],
                "dtype": "float32",
                "layout": "",
            }
        ],
        "nodes": {"total": 2, "input": 1, "reshape": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 3, "shape": 1, "Int": 1, "Mul": 1}

    input_info = [([bz, 2, 3, 4], "float32")]
    verify_model(Reshape(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_transpose(dynamic):
    """test graph builder for transpose"""

    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    bz = "bz" if dynamic else 1
    hidden = "hidden" if dynamic else 4
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 2, 3, hidden], "dtype": "float32", "layout": "ADCB"}
        ],
        "outputs": [
            {
                "name": "permute_dims",
                "shape": [bz, hidden, 3, 2],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 2, "input": 1, "permute_dims": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 2, "shape": 2}

    input_info = [([bz, 2, 3, hidden], "float32")]
    verify_model(Transpose(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_view(dynamic):
    """test graph builder for view"""

    class View(Module):
        def forward(self, x):
            return x.view(-1, 12)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 2, 3, 4], "dtype": "float32", "layout": ""}],
        "outputs": [
            {
                "name": "reshape",
                "shape": ["MUL_2" if dynamic else 2, 12],
                "dtype": "float32",
                "layout": "",
            }
        ],
        "nodes": {"total": 2, "input": 1, "reshape": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 3, "shape": 1, "Int": 1, "Mul": 1}

    input_info = [([bz, 2, 3, 4], "float32")]
    verify_model(View(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_keep_params(dynamic):
    """test graph builder for keep_params"""

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, data):
            return self.conv(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": "NCHW"}
        ],
        "outputs": [
            {
                "name": "conv2d",
                "shape": [bz, 6, 4, 4],
                "dtype": "float32",
                "layout": "NCHW",
            }
        ],
        "nodes": {"total": 2, "input": 1, "msc.conv2d_bias": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    verify_model(Conv2D1(), [([bz, 3, 10, 10], "float32")], expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_unwrap_unit_return_tuple(dynamic):
    """test graph builder for unwrap_unit_return_tuple"""

    class Identity(Module):
        def forward(self, x):
            return (x,)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "tuple", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "tuple": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    verify_model(Identity(), [([bz, 256], "float32")], expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_no_bind_return_tuple(dynamic):
    """test graph builder for no_bind_return_tuple"""

    class Identity(Module):
        def forward(self, x, y):
            return (x, y)

    bz_x = "bz" if dynamic else 1
    bz_y = "bz" if dynamic else 2
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz_x, 256], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz_y, 256], "dtype": "float32", "layout": ""},
        ],
        "outputs": [
            {"name": "tuple_0", "shape": [bz_x, 256], "dtype": "float32", "layout": ""},
            {"name": "tuple_1", "shape": [bz_y, 256], "dtype": "float32", "layout": ""},
        ],
        "nodes": {"total": 3, "input": 2, "tuple": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    input_info = [([bz_x, 256], "float32"), ([bz_y, 256], "float32")]
    verify_model(Identity(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_argmax(dynamic):
    """test graph builder for argmax"""

    class Argmax1(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1)

    class Argmax2(Module):
        def forward(self, data):
            return torch.argmax(data, dim=-1, keepdim=True)

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmax", "shape": [bz], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmax": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmax", "shape": [bz, 1], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmax": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(Argmax1(), [([bz, 256], "float32")], expected1)
    verify_model(Argmax2(), [([bz, 256], "float32")], expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_argmin(dynamic):
    """test graph builder for argmin"""

    class Argmin1(Module):
        def forward(self, data):
            return torch.argmin(data)

    class Argmin2(Module):
        def forward(self, data):
            return torch.argmin(data, keepdim=True)

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmin", "shape": [], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmin": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "argmin", "shape": [1, 1], "dtype": "int64", "layout": ""}],
        "nodes": {"total": 2, "input": 1, "argmin": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(Argmin1(), [([bz, 256], "float32")], expected1)
    verify_model(Argmin2(), [([bz, 256], "float32")], expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_to(dynamic):
    """test graph builder for to"""

    class To1(Module):
        def forward(self, data):
            return data.to(torch.float16)

    class To2(Module):
        def forward(self, data):
            return data.to("cpu")

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "astype", "shape": [bz, 256], "dtype": "float16", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "astype": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "outputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 1, "input": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(To1(), [([bz, 256], "float32")], expected1)
    verify_model(To2(), [([bz, 256], "float32")], expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_mean(dynamic):
    """test graph builder for mean"""

    class Mean(Module):
        def forward(self, data):
            return data.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, data):
            return data.mean(-1, keepdim=True)

    bz = "bz" if dynamic else 1
    expected1 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "mean", "shape": [bz], "dtype": "float32", "layout": "A"}],
        "nodes": {"total": 2, "input": 1, "mean": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "mean", "shape": [bz, 1], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "mean": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(Mean(), [([bz, 256], "float32")], expected1)
    verify_model(MeanKeepDim(), [([bz, 256], "float32")], expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_rsqrt(dynamic):
    """test graph builder for rsqrt"""

    class Rsqrt(Module):
        def forward(self, data):
            return torch.rsqrt(data)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "rsqrt", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "rsqrt": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    verify_model(Rsqrt(), [([bz, 256], "float32")], expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_neg(dynamic):
    """test graph builder for neg"""

    class Neg(Module):
        def forward(self, data):
            return -data

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [{"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "outputs": [{"name": "negative", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 2, "input": 1, "negative": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    verify_model(Neg(), [([bz, 256], "float32")], expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_max(dynamic):
    """test graph builder for max"""

    class Max(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    bz = "bz" if dynamic else 1
    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 256], "dtype": "float32", "layout": "AB"},
            {"name": "inp_1", "shape": [bz, 256], "dtype": "float32", "layout": "AB"},
        ],
        "outputs": [{"name": "maximum", "shape": [bz, 256], "dtype": "float32", "layout": "AB"}],
        "nodes": {"total": 3, "input": 2, "maximum": 1},
    }
    if dynamic:
        expected["prims"] = {"total": 1, "shape": 1}

    verify_model(Max(), [([bz, 256], "float32"), ([bz, 256], "float32")], expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_cat(dynamic):
    """test graph builder for cat"""

    class Cat1(Module):
        def forward(self, data, data1, data2):
            return torch.cat((data, data1, data2), dim=1)

    class Cat2(Module):
        def forward(self, data):
            const1 = torch.ones((1, 3, 10, 10), dtype=torch.float32)
            const2 = torch.ones((1, 3, 10, 10), dtype=torch.float32)
            return torch.cat((data, const1, const2), dim=1)

    bz = "bz" if dynamic else 1
    dim = "dim" if dynamic else 3
    input_info = [
        ([bz, dim, 10, 10], "float32"),
        ([bz, dim, 10, 10], "float32"),
        ([bz, dim, 10, 10], "float32"),
    ]
    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, dim, 10, 10], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz, dim, 10, 10], "dtype": "float32", "layout": ""},
            {"name": "inp_2", "shape": [bz, dim, 10, 10], "dtype": "float32", "layout": ""},
        ],
        "outputs": [
            {
                "name": "concat",
                "shape": [bz, "MUL_3" if dynamic else 9, 10, 10],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 4, "input": 3, "concat": 1},
    }
    expected2 = {
        "inputs": [{"name": "inp_0", "shape": [1, 3, 10, 10], "dtype": "float32", "layout": ""}],
        "outputs": [
            {"name": "concat", "shape": [1, 9, 10, 10], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 4, "input": 1, "constant": 2, "concat": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 4, "shape": 2, "Int": 1, "Mul": 1}

    verify_model(Cat1(), input_info, expected1)
    verify_model(Cat2(), [([1, 3, 10, 10], "float32")], expected2)


@pytest.mark.parametrize("dynamic", [True, False])
def test_stack(dynamic):
    """test graph builder for stack"""

    bz = "bz" if dynamic else 1

    class Stack(Module):
        def forward(self, data, data1, data2):
            return torch.stack((data, data1, data2), dim=0)

    input_info = [
        ([bz, 3, 10, 10], "float32"),
        ([bz, 3, 10, 10], "float32"),
        ([bz, 3, 10, 10], "float32"),
    ]

    expected = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""},
            {"name": "inp_2", "shape": [bz, 3, 10, 10], "dtype": "float32", "layout": ""},
        ],
        "outputs": [
            {
                "name": "reshape",
                "shape": [3, bz, 3, 10, 10],
                "dtype": "float32",
                "layout": "" if dynamic else "EABCD",
            }
        ],
        "nodes": {"total": 5, "input": 3, "concat": 1, "reshape": 1},
    }

    if dynamic:
        expected["prims"] = {"total": 3, "shape": 1, "Int": 1, "Mul": 1}

    verify_model(Stack(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_scatter(dynamic):
    """test graph builder for scatter"""

    bz = "bz" if dynamic else 20

    class Scatter1(Module):
        def __init__(self):
            super().__init__()
            self.index = msc_utils.random_data([(2, 5), "int64"], MSCFramework.TORCH, max_val=5)

        def forward(self, data, src):
            return data.scatter(dim=0, index=self.index, src=src)

    class Scatter2(Module):
        def forward(self, data, index, src):
            return data.scatter(0, index, src)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 20], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [2, 5], "dtype": "float32", "layout": ""},
        ],
        "outputs": [
            {"name": "scatter_elements", "shape": [bz, 20], "dtype": "float32", "layout": ""}
        ],
        "nodes": {"total": 4, "input": 2, "constant": 1, "scatter_elements": 1},
    }
    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [bz, 20], "dtype": "float32", "layout": ""},
            {"name": "inp_1", "shape": [2, 5], "dtype": "int64", "layout": ""},
            {"name": "inp_2", "shape": [2, 5], "dtype": "float32", "layout": ""},
        ],
        "outputs": [
            {"name": "scatter_elements", "shape": [bz, 20], "dtype": "float32", "layout": ""}
        ],
        "nodes": {"total": 4, "input": 3, "scatter_elements": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(Scatter1(), [([bz, 20], "float32"), ([2, 5], "float32")], expected1)
    verify_model(
        Scatter2(), [([bz, 20], "float32"), ([2, 5], "int64"), ([2, 5], "float32")], expected2
    )


@pytest.mark.parametrize("dynamic", [True, False])
def test_masked_scatter(dynamic):
    """test graph builder for masked_scatter"""

    dim = "dim" if dynamic else 5

    class MaskedScatter1(Module):
        def forward(self, data, mask, src):
            return data.masked_scatter(mask, src)

    class MaskedScatter2(Module):
        def forward(self, data, mask, src):
            return data.masked_scatter(mask, src)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [dim], "dtype": "float32", "layout": "A"},
            {"name": "inp_1", "shape": [dim], "dtype": "bool", "layout": "A"},
            {"name": "inp_2", "shape": [10], "dtype": "float32", "layout": "A"},
        ],
        "outputs": [{"name": "where", "shape": [dim], "dtype": "float32", "layout": "A"}],
        "nodes": {
            "total": 8,
            "input": 3,
            "cumsum": 1,
            "constant": 1,
            "subtract": 1,
            "take": 1,
            "where": 1,
        },
    }
    expected2 = {
        "inputs": [
            {
                "name": "inp_0",
                "shape": [2, dim],
                "dtype": "float32",
                "layout": "" if dynamic else "BA",
            },
            {
                "name": "inp_1",
                "shape": [2, dim],
                "dtype": "bool",
                "layout": "" if dynamic else "BA",
            },
            {
                "name": "inp_2",
                "shape": [3, dim],
                "dtype": "float32",
                "layout": "" if dynamic else "BA",
            },
        ],
        "outputs": [
            {
                "name": "where",
                "shape": [2, dim],
                "dtype": "float32",
                "layout": "" if dynamic else "BA",
            }
        ],
        "nodes": {
            "total": 11,
            "input": 3,
            "reshape": 3,
            "cumsum": 1,
            "constant": 1,
            "subtract": 1,
            "take": 1,
            "where": 1,
        },
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}
        expected2["prims"] = {"total": 5, "shape": 1, "Int": 2, "Mul": 2}

    verify_model(
        MaskedScatter1(), [([dim], "float32"), ([dim], "bool"), ([10], "float32")], expected1
    )
    verify_model(
        MaskedScatter2(),
        [([2, dim], "float32"), ([2, dim], "bool"), ([3, dim], "float32")],
        expected2,
    )


def test_put():
    """test graph builder for index_put"""

    class IndexPut(Module):
        def __init__(self):
            super().__init__()
            self.index = msc_utils.random_data([(5), "int64"], MSCFramework.TORCH, max_val=5)

        def forward(self, data, src):
            data[self.index] = src
            return data

    expected = {
        "inputs": [
            {"name": "input0", "shape": [10, 20], "dtype": "float32", "layout": ""},
            {"name": "input1", "shape": [5, 20], "dtype": "float32", "layout": ""},
        ],
        "outputs": [{"name": "scatter_nd", "shape": [10, 20], "dtype": "float32", "layout": ""}],
        "nodes": {"total": 4, "input": 2, "constant": 1, "scatter_nd": 1},
    }

    input_info = [([10, 20], "float32"), ([5, 20], "float32")]
    verify_model(IndexPut(), input_info, expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_attention(dynamic):
    """test graph builder for attention"""

    # pylint: disable=import-outside-toplevel
    import torch.nn.functional as F

    seq = "seq" if dynamic else 128

    class Attention1(Module):
        def forward(self, q_data, k_data, v_data):
            return F.scaled_dot_product_attention(q_data, k_data, v_data)

    class Attention2(Module):
        def forward(self, q_data, k_data, v_data):
            return F.scaled_dot_product_attention(q_data, k_data, v_data, is_causal=True)

    expected1 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 8, seq, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_1", "shape": [1, 8, seq, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_2", "shape": [1, 8, seq, 64], "dtype": "float32", "layout": "ACBD"},
        ],
        "outputs": [
            {"name": "attention", "shape": [1, 8, seq, 64], "dtype": "float32", "layout": "ABCD"}
        ],
        "nodes": {"total": 4, "input": 3, "msc.attention": 1},
    }
    if dynamic:
        expected1["prims"] = {"total": 1, "shape": 1}

    input_info = [
        ([1, 8, seq, 64], "float32"),
        ([1, 8, seq, 64], "float32"),
        ([1, 8, seq, 64], "float32"),
    ]
    verify_model(Attention1(), input_info, expected1)
    verify_model(Attention2(), input_info, expected1)

    class Attention3(Module):
        def forward(self, q_data, k_data, v_data, mask):
            return F.scaled_dot_product_attention(q_data, k_data, v_data, mask)

    expected2 = {
        "inputs": [
            {"name": "inp_0", "shape": [1, 8, seq, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_1", "shape": [1, 8, seq, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_2", "shape": [1, 8, seq, 64], "dtype": "float32", "layout": "ACBD"},
            {"name": "inp_3", "shape": [1, 8, seq, seq], "dtype": "float32", "layout": "ABCD"},
        ],
        "outputs": [
            {
                "name": "attention_bias",
                "shape": [1, 8, seq, 64],
                "dtype": "float32",
                "layout": "ABCD",
            }
        ],
        "nodes": {"total": 5, "input": 4, "msc.attention": 1},
    }
    if dynamic:
        expected2["prims"] = {"total": 1, "shape": 1}

    verify_model(
        Attention3(),
        [
            ([1, 8, seq, 64], "float32"),
            ([1, 8, seq, 64], "float32"),
            ([1, 8, seq, 64], "float32"),
            ([1, 8, seq, seq], "float32"),
        ],
        expected2,
    )


if __name__ == "__main__":
    tvm.testing.main()
