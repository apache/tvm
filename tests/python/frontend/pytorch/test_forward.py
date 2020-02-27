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
# pylint: disable=import-self, invalid-name, unused-argument
"""Unit tests for various models and operators"""
from time import time
import os
import sys
from tempfile import TemporaryDirectory
from scipy.stats import t as tdistr
import numpy as np
import torch
from torch.nn import Module
import tvm
from tvm import te
import torchvision

from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list

sys.setrecursionlimit(10000)

def _vectorize(ten):
    return ten.reshape(-1)

def atol(tru, est):
    def _atol_elt(tru, est):
        return abs(tru - est)
    tru = _vectorize(tru)
    est = _vectorize(est)
    return max([_atol_elt(x, y) for x, y in zip(tru, est)])

def rtol(tru, est):
    def _rtol_elt(tru, est):
        return abs(tru - est) / min(abs(tru), abs(est))
    tru = _vectorize(tru)
    est = _vectorize(est)
    return max([_rtol_elt(x, y) for x, y in zip(tru, est)])

def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))

def load_torchvision(model_name):
    """Given a model name, returns a Torchvision model in eval mode as well
    as an example input."""
    with torch.no_grad():
        if model_name.startswith("inception"):
            height = width = 299
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            height = width = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        input_shape = [1, 3, height, width]
        input_data = torch.randn(input_shape).float()
        for channel in range(3):
            input_data[:, channel] -= mean[channel]
            input_data[:, channel] /= std[channel]
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.float().eval()
        return model, input_data

def load_pretrainedmodels(model_name):
    """Given a model name, returns a pretrainedmodels.pytorch model in eval
    mode as well as an example input."""
    import pretrainedmodels # https://github.com/Cadene/pretrained-models.pytorch
    model = getattr(pretrainedmodels, model_name)().float().eval()
    input_shape = [1, *model.input_size]
    input_data = torch.rand(input_shape).float() * 256
    for channel in range(3):
        input_data[:, channel] -= model.mean[channel]
        input_data[:, channel] /= model.std[channel]
    return model, input_data

def load_model(model_name):
    """Given a model name, returns a model as well as an example input."""
    if hasattr(torchvision.models, model_name):
        return load_torchvision(model_name)
    try:
        if hasattr(pretrainedmodels, model_name):
            return load_pretrainedmodels(model_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install pretrainedmodels.pytorch")
    raise RuntimeError("Model not supported")


def confidence_interval(mean, stdev, count, alpha=.01):
    """Returns the lower and upper bounds of the confidence interval of a random
    variable. Confidence is 1 - alpha (default confidence is 99%)."""
    stdval = tdistr.ppf(1 - alpha / 2, count - 1)
    lower, upper = mean + np.array([-1, 1]) * stdval * stdev / np.sqrt(count)
    return lower, upper

def measure_latency(model, input_shapes, output_shapes, thresh, dryruns=40):
    """Compute the latency of the given model"""
    latencies = []
    count = 0
    while True:
        if isinstance(model, torch.nn.Module):
            input_data = [torch.rand(shape).float() for shape in input_shapes]
            if torch.cuda.is_available():
                input_data = list(map(lambda x: x.cuda(), input_data))
                model = model.cuda()
            t_start = time()
            with torch.no_grad():
                model(*input_data)
            t_end = time()
            latencies.append(t_end - t_start)
        else:
            input_data = {}
            for i, shape in enumerate(input_shapes):
                name = "input" + str(i)
                arr = np.random.random(shape).astype("float32")
                input_data[name] = tvm.nd.array(arr)
            t_start = time()
            model.set_input(**input_data)
            model.run()
            for i, shape in enumerate(output_shapes):
                arr = np.zeros(shape).astype("float32")
                model.get_output(i, tvm.nd.array(arr))
            t_end = time()
        count += 1
        if count < dryruns:
            continue
        latencies.append(t_end - t_start)
        mean = np.mean(latencies)
        stdev = np.std(latencies)
        sample_size = len(latencies)
        if sample_size > dryruns:
            lower, upper = confidence_interval(mean, stdev, sample_size)
            est = (upper + lower) / 2
            err = (upper - lower) / 2
            if err < thresh:
                return est

def verify_model(model_name, input_data=[]):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    if len(input_data) == 0:
        baseline_model, baseline_input = load_model(model_name)
    else:
        baseline_model = model_name
        baseline_input = input_data
    if torch.cuda.is_available():
        baseline_model = baseline_model.cuda()
        baseline_input = baseline_input.cuda()
    with torch.no_grad():
        baseline_outputs = baseline_model(baseline_input)
    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.float().cpu().numpy(),)
    output_shapes = [out.shape for out in baseline_outputs]
    dtype = "float32"
    input_name = "input0"
    input_shapes = {input_name: list(baseline_input.shape)}
    trace = torch.jit.trace(baseline_model, baseline_input).float().eval()
    if torch.cuda.is_available():
        trace = trace.cuda()
    else:
        trace = trace.cpu()

    mod, params = relay.frontend.from_pytorch(trace, input_shapes)
    compiled_input = {input_name: tvm.nd.array(baseline_input.cpu().numpy())}

    with relay.build_config(opt_level=3):
        for target, ctx in ctx_list():
            relay_graph, relay_lib, relay_params = relay.build(mod, target=target, params=params)
            relay_model = graph_runtime.create(relay_graph, relay_lib, ctx)
            relay_model.set_input(**relay_params)
            relay_model.set_input(**compiled_input)
            relay_model.run()

            for i, baseline_output in enumerate(baseline_outputs):
                compiled_output = relay_model.get_output(i).asnumpy()

                assert_shapes_match(baseline_output, compiled_output)
                tvm.testing.assert_allclose(baseline_output, compiled_output,
                                            rtol=1e-3, atol=1e-3)

    del model_name
    del baseline_model
    torch.cuda.empty_cache()

# Single operator tests
def test_forward_add():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Add1(Module):
        def forward(self, *args):
            return args[0] + args[0]

    class Add2(Module):
        def forward(self, *args):
            return args[0] + 1

    class Add3(Module):
        def forward(self, *args):
            ones = torch.ones(input_shape, dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    class Add4(Module):
        def forward(self, *args):
            ones = torch.ones([], dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Add1().float().eval(), input_data=input_data)
        verify_model(Add2().float().eval(), input_data=input_data)
        verify_model(Add3().float().eval(), input_data=input_data)
        verify_model(Add4().float().eval(), input_data=input_data)

def test_forward_subtract():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Subtract1(Module):
        def forward(self, *args):
            return args[0] - args[0]

    class Subtract2(Module):
        def forward(self, *args):
            return args[0] - 1

    class Subtract3(Module):
        def forward(self, *args):
            ones = torch.ones(input_shape)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] - ones

    class Subtract4(Module):
        def forward(self, *args):
            ones = torch.ones([])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] - ones

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Subtract1().float().eval(), input_data=input_data)
        verify_model(Subtract2().float().eval(), input_data=input_data)
        verify_model(Subtract3().float().eval(), input_data=input_data)
        verify_model(Subtract4().float().eval(), input_data=input_data)

def test_forward_multiply():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Multiply1(Module):
        def forward(self, *args):
            return args[0] * args[0]

    class Multiply2(Module):
        def forward(self, *args):
            return args[0] * 1

    class Multiply3(Module):
        def forward(self, *args):
            ones = torch.ones(input_shape)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] * ones

    class Multiply4(Module):
        def forward(self, *args):
            ones = torch.ones([])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] * ones

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Multiply1().float().eval(), input_data=input_data)
        verify_model(Multiply2().float().eval(), input_data=input_data)
        verify_model(Multiply3().float().eval(), input_data=input_data)
        verify_model(Multiply4().float().eval(), input_data=input_data)

def test_forward_unsqueeze():
    torch.set_grad_enabled(False)
    input_shape = [10, 10]

    class Unsqueeze1(Module):
        def forward(self, *args):
            return args[0].unsqueeze(2)

    input_data = torch.rand(input_shape).float()
    verify_model(Unsqueeze1().float().eval(), input_data=input_data)

def test_forward_concatenate():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Concatenate1(Module):
        def forward(self, *args):
            return torch.cat([args[0][:, 0].unsqueeze(1), args[0][:, 1].unsqueeze(1)], 1)

    class Concatenate2(Module):
        def forward(self, *args):
            a = (args[0][:, :, 0] + 2) * 7
            b = (args[0][:, :, 1] + 3) * 11
            c = (args[0][:, :, 2] + 5) * 13
            return torch.cat([t.unsqueeze(2) for t in [a, b, c]], 2)

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Concatenate1().float().eval(), input_data=input_data)
        verify_model(Concatenate2().float().eval(), input_data=input_data)

def test_forward_relu():
    torch.set_grad_enabled(False)
    input_shape = [10, 10]

    class ReLU1(Module):
        def forward(self, *args):
            return torch.nn.ReLU()(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(ReLU1().float().eval(), input_data=input_data)

def test_forward_adaptiveavgpool():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class AdaptiveAvgPool2D1(Module):
        def forward(self, *args):
            return torch.nn.AdaptiveAvgPool2d([1, 1])(args[0])

    class AdaptiveAvgPool2D2(Module):
        def forward(self, *args):
            return torch.nn.AdaptiveAvgPool2d([10, 10])(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(AdaptiveAvgPool2D1().float().eval(), input_data=input_data)
        verify_model(AdaptiveAvgPool2D2().float().eval(), input_data=input_data)

def test_forward_maxpool():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class MaxPool2D1(Module):
        def forward(self, *args):
            return torch.nn.MaxPool2d(kernel_size=[1, 1])(args[0])

    class MaxPool2D2(Module):
        def forward(self, *args):
            return torch.nn.MaxPool2d(kernel_size=[10, 10])(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(MaxPool2D1().float().eval(), input_data=input_data)
        verify_model(MaxPool2D2().float().eval(), input_data=input_data)

def test_forward_avgpool():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class AvgPool2D1(Module):
        def forward(self, *args):
            return torch.nn.AvgPool2d(kernel_size=[10, 10])(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(AvgPool2D1().float().eval(), input_data=input_data)

def test_forward_hardtanh():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class HardTanh1(Module):
        def forward(self, *args):
            return torch.nn.Hardtanh()(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(HardTanh1().float().eval(), input_data=input_data)

def test_forward_conv():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Conv2D1(Module):
        def __init__(self):
            super(Conv2D1, self).__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv2D2(Module):
        def __init__(self):
            super(Conv2D2, self).__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv2D3(Module):
        def __init__(self):
            super(Conv2D3, self).__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, groups=3, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Conv2D1().float().eval(), input_data=input_data)
        verify_model(Conv2D2().float().eval(), input_data=input_data)
        verify_model(Conv2D3().float().eval(), input_data=input_data)

def test_forward_threshold():
    torch.set_grad_enabled(False)
    input_shape = [1, 3]

    class Threshold1(Module):
        def forward(self, *args):
            return torch.nn.Threshold(0, 0)(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Threshold1().float().eval(), input_data=input_data)

def test_forward_contiguous():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Contiguous1(Module):
        def forward(self, *args):
            return args[0].contiguous()

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Contiguous1().float().eval(), input_data=input_data)

def test_forward_batchnorm():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class BatchNorm1(Module):
        def __init__(self):
            super(BatchNorm1, self).__init__()
            self.batch_norm = torch.nn.BatchNorm2d(3, affine=True)
        def forward(self, *args):
            return self.batch_norm(args[0])

    class BatchNorm2(Module):
        def __init__(self):
            super(BatchNorm2, self).__init__()
            self.batch_norm = torch.nn.BatchNorm2d(3, affine=False)
        def forward(self, *args):
            return self.batch_norm(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(BatchNorm1().float().eval(), input_data=input_data)
        verify_model(BatchNorm2().float().eval(), input_data=input_data)

def test_forward_transpose():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Transpose1(Module):
        def forward(self, *args):
            return args[0].transpose(2, 3)

    class Transpose2(Module):
        def forward(self, *args):
            return args[0].transpose(-2, -1)

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Transpose1().float().eval(), input_data=input_data)
        verify_model(Transpose2().float().eval(), input_data=input_data)

def test_forward_size():
    torch.set_grad_enabled(False)
    input_shape = [1, 3]

    class Size1(Module):
        def forward(self, *args):
            return args[0].size(0) * args[0]

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Size1().float().eval(), input_data=input_data)

def test_forward_view():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class View1(Module):
        def forward(self, *args):
            return args[0].view((1, 3 * 10 * 10))

    class View2(Module):
        def forward(self, *args):
            return args[0].view(args[0].shape[0], -1)

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(View1().float().eval(), input_data=input_data)
        verify_model(View2().float().eval(), input_data=input_data)

def test_forward_select():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Select1(Module):
        def forward(self, *args):
            return args[0].select(1, 1)

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Select1().float().eval(), input_data=input_data)

def test_forward_clone():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Clone1(Module):
        def forward(self, *args):
            return args[0].clone()

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Clone1().float().eval(), input_data=input_data)

def test_forward_logsoftmax():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class LogSoftmax1(Module):
        def forward(self, *args):
            return torch.nn.LogSoftmax(dim=1)(args[0][0, 0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(LogSoftmax1().float().eval(), input_data=input_data)

def test_forward_sigmoid():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Sigmoid1(Module):
        def forward(self, *args):
            return torch.nn.Sigmoid()(args[0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Sigmoid1().float().eval(), input_data=input_data)

def test_forward_dense():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Dense1(Module):
        def __init__(self):
            super(Dense1, self).__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)
        def forward(self, *args):
            return self.linear(args[0][0, 0])

    class Dense2(Module):
        def __init__(self):
            super(Dense2, self).__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)
        def forward(self, *args):
            return self.linear(args[0][0, 0])

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Dense1().float().eval(), input_data=input_data)
        verify_model(Dense2().float().eval(), input_data=input_data)

def test_forward_dropout():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Dropout1(Module):
        def forward(self, *args):
            return torch.nn.functional.dropout(args[0][0, 0], 0.5, False)

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Dropout1().float().eval(), input_data=input_data)

def test_forward_slice():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Slice1(Module):
        def forward(self, *args):
            return args[0][:, :, :, :3]

    class Slice2(Module):
        def forward(self, *args):
            return args[0][0, :, :, :]

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Slice1().float().eval(), input_data=input_data)
        verify_model(Slice2().float().eval(), input_data=input_data)

def test_forward_mean():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Mean1(Module):
        def forward(self, *args):
            return args[0].mean(2)

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Mean1().float().eval(), input_data=input_data)

def test_forward_expand():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Expand1(Module):
        def forward(self, *args):
            return args[0].expand((3, -1, -1, -1))

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Expand1().float().eval(), input_data=input_data)

def test_forward_pow():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Pow1(Module):
        def forward(self, *args):
            return args[0] ** 2

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Pow1().float().eval(), input_data=input_data)

def test_forward_chunk():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 14, 14]

    class Chunk1(Module):
        def forward(self, *args):
            chunks = args[0].chunk(7, 2)
            return torch.cat(chunks, 2)

    with torch.no_grad():
        input_data = torch.rand(input_shape).float()
        verify_model(Chunk1().float().eval(), input_data=input_data)

# Model tests
def test_resnet18():
    torch.set_grad_enabled(False)
    verify_model("resnet18")

def test_squeezenet1_0():
    torch.set_grad_enabled(False)
    verify_model("squeezenet1_0")

def test_squeezenet1_1():
    torch.set_grad_enabled(False)
    verify_model("squeezenet1_1")

def test_densenet121():
    torch.set_grad_enabled(False)
    verify_model("densenet121")

def test_inception_v3():
    torch.set_grad_enabled(False)
    verify_model("inception_v3")

def test_googlenet():
    torch.set_grad_enabled(False)
    verify_model("googlenet")

def test_mnasnet0_5():
    torch.set_grad_enabled(False)
    verify_model("mnasnet0_5")

"""
#TODO: Fix VGG and AlexNet issues (probably due to pooling)
def test_alexnet():
    torch.set_grad_enabled(False)
    verify_model("alexnet")

def test_vgg11():
    torch.set_grad_enabled(False)
    verify_model("vgg11")

def test_vgg11_bn():
    torch.set_grad_enabled(False)
    verify_model("vgg11_bn")

#TODO: Need to update schedule in tophub file after PR #4787 updated workloads
def test_mobilenet_v2():
    torch.set_grad_enabled(False)
    verify_model("mobilenet_v2")
"""

if __name__ == "__main__":
    # Single operator tests
    test_forward_add()
    test_forward_subtract()
    test_forward_multiply()
    test_forward_unsqueeze()
    test_forward_concatenate()
    test_forward_relu()
    test_forward_adaptiveavgpool()
    test_forward_maxpool()
    test_forward_hardtanh()
    test_forward_conv()
    test_forward_threshold()
    test_forward_contiguous()
    test_forward_batchnorm()
    test_forward_transpose()
    test_forward_size()
    test_forward_view()
    test_forward_select()
    test_forward_clone()
    test_forward_logsoftmax()
    test_forward_sigmoid()
    test_forward_dense()
    test_forward_avgpool()
    test_forward_dropout()
    test_forward_slice()
    test_forward_mean()
    test_forward_expand()
    test_forward_pow()
    test_forward_chunk()

    # Model tests
    test_resnet18()
    test_squeezenet1_0()
    test_squeezenet1_1()
    test_densenet121()
    test_inception_v3()
    test_googlenet()
    test_mnasnet0_5()
