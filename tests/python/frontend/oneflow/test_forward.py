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
# pylint: disable=arguments-differ, unused-argument
"""Unit tests for various models and operators"""
import os

import numpy as np
import oneflow as flow
from packaging import version as package_version
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay

MODEL_HOME = "test_model"


def mkdir(path):
    # init
    path = path.strip()
    path = path.rstrip("\\")

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"{path} is already here")


def rmdir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.removedirs(path)


def assert_shape(out1, out2):
    if out1.shape != out2.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(out1.shape, out2.shape))


class OneFlowGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


class OneFlowGraphV2(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, input_1, input_2, input_3):
        out = self.m(input_1, input_2, input_3)
        return out


class OneFlowGraphV3(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, input_1, input_2):
        out = self.m(input_1, input_2)
        return out


def get_oneflow_output(model, inputs):
    flow_output = model(inputs)
    return flow_output.numpy()


def get_oneflow_concat_output(model, input1, input2, input3):
    flow_output = model(input1, input2, input3).numpy()
    return flow_output


def get_oneflow_elementwise_output(model, input1, input2):
    return model(input1, input2).numpy()


def get_tvm_output(graph, model_path, inputs: flow.tensor, target="llvm", dtype="float32"):
    """Generic function to execute and get tvm output"""
    inputs_numpy = inputs.numpy()
    if target == "llvm":
        device = tvm.cpu(0)
    elif target == "cuda":
        device = tvm.cuda(0)

    mod, params = relay.frontend.from_oneflow(graph, model_path)
    with tvm.transform.PassContext(opt_level=10):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    tvm_output = intrp.evaluate()(tvm.nd.array(inputs_numpy.astype(dtype)), **params).numpy()
    return tvm_output


def get_tvm_concat_output(
    graph,
    model_path,
    input1: flow.tensor,
    input2: flow.tensor,
    input3: flow.tensor,
    target="llvm",
    dtype="float32",
):
    """Generic function to execute and get tvm concat output"""
    input1_numpy = input1.numpy()
    input2_numpy = input2.numpy()
    input3_numpy = input3.numpy()
    if target == "llvm":
        device = tvm.cpu(0)
    elif target == "cuda":
        device = tvm.cuda(0)

    mod, params = relay.frontend.from_oneflow(graph, model_path)
    with tvm.transform.PassContext(opt_level=10):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    tvm_output = intrp.evaluate()(
        tvm.nd.array(input1_numpy.astype(dtype)),
        tvm.nd.array(input2_numpy.astype(dtype)),
        tvm.nd.array(input3_numpy.astype(dtype)),
        **params,
    ).numpy()
    return tvm_output


def get_tvm_elementwise_output(
    graph,
    model_path,
    input1: flow.tensor,
    input2: flow.tensor,
    target="llvm",
    dtype="float32",
):
    """Generic function to execute and get tvm elementwise output"""
    input1_numpy = input1.numpy()
    input2_numpy = input2.numpy()
    if target == "llvm":
        device = tvm.cpu(0)
    elif target == "cuda":
        device = tvm.cuda(0)

    mod, params = relay.frontend.from_oneflow(graph, model_path)
    with tvm.transform.PassContext(opt_level=10):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    tvm_output = intrp.evaluate()(
        tvm.nd.array(input1_numpy.astype(dtype)),
        tvm.nd.array(input2_numpy.astype(dtype)),
        **params,
    ).numpy()
    return tvm_output


def verify_conv(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 224, 224),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """verify_conv"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_pool(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 224, 224),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """verify_pool"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_normalization(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 224, 224),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """verify_normalization"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    # write params
    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_upsample(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 50, 50),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """verify_upsample"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_convtran(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(1, 3, 50, 50),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """verify_convtran"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_activation(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(10, 10),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """verify_activation"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_math(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs=flow.tensor(
        np.random.rand(100, 1),
        dtype=flow.float32,
    ),
    device="llvm",
):
    """verify_math"""
    if device == "cuda":
        model.to(device)
        inputs = inputs.to(device)

    graph = OneFlowGraph(model)
    graph._compile(inputs)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_matmul(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs1=flow.tensor(np.random.randn(2, 5), dtype=flow.float32),
    inputs2=flow.tensor(np.random.randn(5, 2), dtype=flow.float32),
    device="llvm",
):
    """verify_matmul"""
    if device == "cuda":
        model.to(device)
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)

    graph = OneFlowGraphV3(model)
    graph._compile(inputs1, inputs2)
    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_elementwise_output(graph, inputs1, inputs2)
    out_tvm = get_tvm_elementwise_output(graph, MODEL_HOME, inputs1, inputs2, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


def verify_concat(
    model,
    name="",
    rtol=1e-5,
    atol=1e-5,
    inputs1=flow.tensor(np.random.randn(2, 5, 5, 4), dtype=flow.float32),
    inputs2=flow.tensor(np.random.randn(2, 5, 5, 2), dtype=flow.float32),
    inputs3=flow.tensor(np.random.randn(2, 5, 5, 3), dtype=flow.float32),
    device="llvm",
):
    """verify_concat"""
    if device == "cuda":
        model.to(device)
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)

    graph = OneFlowGraphV2(model)
    graph._compile(inputs1, inputs2, inputs3)

    mkdir(MODEL_HOME)
    flow.save(model.state_dict(), MODEL_HOME)

    out_flow = get_oneflow_concat_output(graph, inputs1, inputs2, inputs3)
    out_tvm = get_tvm_concat_output(graph, MODEL_HOME, inputs1, inputs2, inputs3, target=device)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


# defs/nn
@tvm.testing.uses_gpu
def test_conv2d():
    """Conv2d"""

    class Conv2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = flow.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model = Conv2dModel()
    model.eval()

    for device in ["llvm"]:
        verify_conv(model, device=device)


@tvm.testing.uses_gpu
def test_pool2d():
    """Pool2d"""

    class MaxPool2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = flow.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.pool(x)
            return x

    class AvgPool2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = flow.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.pool(x)
            return x

    class AdaptiveAvgPool2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = flow.nn.AdaptiveAvgPool2d((None, 7))

        def forward(self, x):
            x = self.pool(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model1 = MaxPool2dModel().eval()
    model2 = AvgPool2dModel().eval()
    model3 = AdaptiveAvgPool2dModel().eval()

    for device in ["llvm"]:
        verify_pool(model1, device=device)
        verify_pool(model2, device=device)
        verify_pool(model3, device=device)


@tvm.testing.uses_gpu
def test_normalization():
    """Normalization"""

    class BatchNorm2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.normalization = flow.nn.BatchNorm2d(3)

        def forward(self, x):
            x = self.normalization(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model = BatchNorm2dModel().eval()

    for device in ["llvm"]:
        verify_normalization(model, device=device)


@tvm.testing.uses_gpu
def test_upsample():
    """Upsample"""

    class UpsampleModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = flow.nn.Upsample(scale_factor=2.0, mode="nearest")

        def forward(self, x):
            x = self.upsample(x)
            return x

    class UpsampleBiliModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = flow.nn.UpsamplingBilinear2d(scale_factor=2.0)

        def forward(self, x):
            x = self.upsample(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model1 = UpsampleModel().eval()
    model2 = UpsampleBiliModel().eval()

    for device in ["llvm"]:
        verify_upsample(model1, device=device)
        verify_upsample(model2, device=device)


@tvm.testing.uses_gpu
def test_convtran():
    """ConvTran"""

    class ConvTranModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.convtran = flow.nn.ConvTranspose2d(3, 4, (3, 5), stride=(2, 1), padding=(4, 2))

        def forward(self, x):
            x = self.convtran(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model = ConvTranModel().eval()

    for device in ["llvm"]:
        verify_convtran(model, device=device)


@tvm.testing.uses_gpu
def test_activation():
    """Activation"""

    class Softmax(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Softmax()

        def forward(self, x):
            x = self.active(x)
            return x

    class Softplus(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Softplus()

        def forward(self, x):
            x = self.active(x)
            return x

    class Softsign(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Softsign()

        def forward(self, x):
            x = self.active(x)
            return x

    class Tanh(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Tanh()

        def forward(self, x):
            x = self.active(x)
            return x

    class ReLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.ReLU()

        def forward(self, x):
            x = self.active(x)
            return x

    class ReLU6(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.ReLU6()

        def forward(self, x):
            x = self.active(x)
            return x

    class PReLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.PReLU()

        def forward(self, x):
            x = self.active(x)
            return x

    class SELU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.SELU()

        def forward(self, x):
            x = self.active(x)
            return x

    class SiLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.SiLU()

        def forward(self, x):
            x = self.active(x)
            return x

    class LeakyReLU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.LeakyReLU(0.1)

        def forward(self, x):
            x = self.active(x)
            return x

    class GELU(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.GELU()

        def forward(self, x):
            x = self.active(x)
            return x

    class HardTanh(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Hardtanh()

        def forward(self, x):
            x = self.active(x)
            return x

    class TensorSoftmax(flow.nn.Module):
        def forward(self, x):
            x = x.softmax(dim=-1)
            return x

    class Threshold(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = flow.nn.Threshold(0.5, 0.2)

        def forward(self, x):
            x = self.active(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model1 = Softmax().eval()
    model2 = Softplus().eval()  # pylint: disable=unused-variable
    model3 = Softsign().eval()
    model4 = Tanh().eval()
    model5 = ReLU().eval()
    model6 = ReLU6().eval()
    model7 = PReLU().eval()
    model8 = SELU().eval()
    model9 = SiLU().eval()
    model10 = LeakyReLU().eval()
    model11 = GELU().eval()
    model12 = HardTanh().eval()
    model13 = TensorSoftmax().eval()

    for device in ["llvm"]:
        verify_activation(model1, device=device)
        verify_activation(model2, device=device)
        verify_activation(model3, device=device)
        verify_activation(model4, device=device)
        verify_activation(model5, device=device)
        verify_activation(model6, device=device)
        verify_activation(model7, device=device)
        verify_activation(model8, device=device)
        verify_activation(model9, device=device)
        verify_activation(model10, device=device)
        verify_activation(model11, device=device)
        verify_activation(model12, device=device)
        verify_activation(
            model13,
            device=device,
            inputs=flow.tensor(np.random.rand(1, 12, 197, 197).astype(np.float32)),
        )

    # Threshold was introduced in the version 0.8.0 of oneflow
    if package_version.parse(flow.__version__) >= package_version.parse("0.8.0"):
        model14 = Threshold().eval()
        verify_activation(model14, device="llvm")


@tvm.testing.uses_gpu
def test_math():
    """Math"""

    class Sigmoid(flow.nn.Module):
        def forward(self, x):
            return flow.sigmoid(x)

    class Sign(flow.nn.Module):
        def forward(self, x):
            return flow.sign(x)

    class Reciprocal(flow.nn.Module):
        def forward(self, x):
            return flow.reciprocal(x)

    class Pow(flow.nn.Module):
        def forward(self, x):
            return flow.pow(x, 2.0)

    class Log(flow.nn.Module):
        def forward(self, x):
            return flow.log(x)

    class Log2(flow.nn.Module):
        def forward(self, x):
            return flow.log1p(x)

    class Exp(flow.nn.Module):
        def forward(self, x):
            return flow.exp(x)

    class Exp2(flow.nn.Module):
        def forward(self, x):
            return flow.expm1(x)

    class Variance(flow.nn.Module):
        def forward(self, x):
            return flow.var(x, 1, unbiased=False, keepdim=True)

    model1 = Sigmoid().eval()
    model2 = Sign().eval()
    model3 = Log().eval()
    model4 = Log2().eval()
    model5 = Exp().eval()
    model6 = Exp2().eval()
    model7 = Reciprocal().eval()
    model8 = Pow().eval()
    model9 = Variance().eval()

    for device in ["llvm"]:
        verify_math(model1, device=device)
        verify_math(model2, device=device)
        verify_math(model3, device=device)
        verify_math(model4, device=device)
        verify_math(model5, device=device)
        verify_math(model6, device=device)
        verify_math(model7, device=device)
        verify_math(model8, device=device)
        verify_math(model9, device=device)


@tvm.testing.uses_gpu
def test_slice():
    """Slice"""

    class Slice(flow.nn.Module):
        def forward(self, x):
            tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
            out = flow.slice(x, slice_tup_list=tup_list)
            return out

    model = Slice().eval()

    for device in ["llvm"]:
        verify_math(
            model, device=device, inputs=flow.tensor(np.random.randn(3, 6, 9).astype(np.float32))
        )


@tvm.testing.uses_gpu
def test_concat():
    """Concat"""

    class Concat(flow.nn.Module):
        def forward(self, input_1, input_2, input_3):
            out = flow.cat([input_1, input_2, input_3], dim=-1)
            return out

    model = Concat().eval()

    for device in ["llvm"]:
        verify_concat(model, device=device)


@tvm.testing.uses_gpu
def test_add_constant():
    """ConstantAdd"""

    class ConstantAdd(flow.nn.Module):
        def forward(self, x):
            out = flow.add(1.0, x)
            return out

    model = ConstantAdd().eval()

    for device in ["llvm"]:
        verify_math(
            model, device=device, inputs=flow.tensor(np.random.randn(3, 6, 9).astype(np.float32))
        )


@tvm.testing.uses_gpu
def test_logical():
    class LogicalGreater(flow.nn.Module):
        def forward(self, x):
            return x > 1.0

    model1 = LogicalGreater().eval()

    for device in ["llvm"]:
        verify_math(
            model1, device=device, inputs=flow.tensor(np.random.randn(3, 6, 9).astype(np.float32))
        )


@tvm.testing.uses_gpu
def test_expand():
    class Expand(flow.nn.Module):
        def forward(self, x):
            return x.expand(2, -1, -1)

    model1 = Expand().eval()

    for device in ["llvm"]:
        verify_math(
            model1, device=device, inputs=flow.tensor(np.random.randn(1, 6, 9).astype(np.float32))
        )


@tvm.testing.uses_gpu
def test_matmul():
    """MatMul"""

    class MatMul(flow.nn.Module):
        def forward(self, x, y):
            return flow._C.matmul(x, y)

    class MatMulTranspose(flow.nn.Module):
        def forward(self, x, y):
            return flow._C.matmul(x, y, transpose_b=True)

    class BatchMatMul(flow.nn.Module):
        def forward(self, x, y):
            return flow._C.batch_matmul(x, y)

    class BroadCastMatMul(flow.nn.Module):
        def forward(self, x, y):
            return flow._C.matmul(x, y)

    model1 = MatMul().eval()
    model2 = MatMulTranspose().eval()
    model3 = BatchMatMul().eval()
    model4 = BroadCastMatMul().eval()

    for device in ["llvm"]:
        verify_matmul(
            model1,
            device=device,
            inputs1=flow.tensor(np.random.randn(2, 3).astype(np.float32)),
            inputs2=flow.tensor(np.random.randn(3, 3).astype(np.float32)),
        )
        verify_matmul(
            model2,
            device=device,
            inputs1=flow.tensor(np.random.randn(1, 2).astype(np.float32)),
            inputs2=flow.tensor(np.random.randn(3, 2).astype(np.float32)),
        )
        verify_matmul(
            model3,
            device=device,
            inputs1=flow.tensor(np.random.randn(2, 1, 2).astype(np.float32)),
            inputs2=flow.tensor(np.random.randn(2, 2, 3).astype(np.float32)),
        )
        verify_matmul(
            model4,
            device=device,
            inputs1=flow.tensor(np.random.randn(3, 8, 8, 16).astype(np.float32)),
            inputs2=flow.tensor(np.random.randn(16, 8).astype(np.float32)),
        )


if __name__ == "__main__":
    test_conv2d()
    test_pool2d()
    test_normalization()
    test_upsample()
    test_convtran()
    test_activation()
    test_math()
    test_slice()
    test_concat()
    test_add_constant()
    test_logical()
    test_expand()
    test_matmul()
    rmdir("log")
