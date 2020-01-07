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
    if model_name.startswith('inception'):
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
        raise ModuleNotFoundError('Please install pretrainedmodels.pytorch')
    raise RuntimeError('Model not supported')


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
            model(*input_data)
            t_end = time()
            latencies.append(t_end - t_start)
        else:
            input_data = {}
            for i, shape in enumerate(input_shapes):
                name = 'input' + str(i)
                arr = np.random.random(shape).astype('float32')
                input_data[name] = tvm.nd.array(arr)
            t_start = time()
            model.set_input(**input_data)
            model.run()
            for i, shape in enumerate(output_shapes):
                arr = np.zeros(shape).astype('float32')
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

    print(model_name)

    if len(input_data) == 0:
        baseline_model, baseline_input = load_model(model_name)
    else:
        baseline_model = model_name
        baseline_input = input_data
    if torch.cuda.is_available():
        baseline_model = baseline_model.cuda()
        baseline_input = baseline_input.cuda()
    baseline_outputs = baseline_model(baseline_input)
    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.detach().cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.detach().float().cpu().numpy(),)
    output_shapes = [out.shape for out in baseline_outputs]
    dtype = 'float32'
    input_name = 'input0'
    input_shapes = {input_name: list(baseline_input.shape)}
    baseline_model(baseline_input)
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
                output_shape = baseline_output.shape
                compiled_output = relay_model.get_output(
                    i, tvm.nd.array(np.zeros(output_shape).astype(dtype), ctx)).asnumpy()

                compiled_relay_output = relay_model.get_output(
                    i, tvm.nd.array(np.zeros(output_shape).astype(dtype), ctx)).asnumpy()

                assert_shapes_match(baseline_output, compiled_output)
                tvm.testing.assert_allclose(baseline_output, compiled_output,
                                            rtol=1e-3, atol=1e-3)

                assert_shapes_match(baseline_output, compiled_relay_output)
                tvm.testing.assert_allclose(baseline_output, compiled_relay_output,
                                            rtol=1e-3, atol=1e-3)

    from subprocess import call
    call('rm -rf ~/.torch/models/*', shell=True)

# Single operator tests
def test_forward_add():
    input_shape = [1, 3, 224, 224]

    class Add1(Module):
        def forward(self, *args):
            return args[0] + args[0]

    class Add2(Module):
        def forward(self, *args):
            return args[0] + 1

    class Add3(Module):
        def forward(self, *args):
            ones = torch.ones([1, 3, 224, 224], dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    class Add4(Module):
        def forward(self, *args):
            ones = torch.ones([1, 1, 224, 224], dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    class Add5(Module):
        def forward(self, *args):
            ones = torch.ones([], dtype=torch.float)
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] + ones

    input_data = torch.rand(input_shape).float()

    verify_model(Add1().float().eval(), input_data=input_data)
    verify_model(Add2().float().eval(), input_data=input_data)
    verify_model(Add3().float().eval(), input_data=input_data)
    verify_model(Add4().float().eval(), input_data=input_data)
    verify_model(Add5().float().eval(), input_data=input_data)

def test_forward_subtract():
    input_shape = [1, 3, 224, 224]

    class Subtract1(Module):
        def forward(self, *args):
            return args[0] - args[0]

    class Subtract2(Module):
        def forward(self, *args):
            return args[0] - 1

    class Subtract3(Module):
        def forward(self, *args):
            ones = torch.ones([1, 3, 224, 224])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] - ones

    class Subtract4(Module):
        def forward(self, *args):
            ones = torch.ones([1, 1, 224, 224])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] - ones

    class Subtract5(Module):
        def forward(self, *args):
            ones = torch.ones([])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] - ones

    input_data = torch.rand(input_shape).float()

    verify_model(Subtract1().float().eval(), input_data=input_data)
    verify_model(Subtract2().float().eval(), input_data=input_data)
    verify_model(Subtract3().float().eval(), input_data=input_data)
    verify_model(Subtract4().float().eval(), input_data=input_data)
    verify_model(Subtract5().float().eval(), input_data=input_data)

def test_forward_multiply():
    input_shape = [1, 3, 224, 224]

    class Multiply1(Module):
        def forward(self, *args):
            return args[0] * args[0]

    class Multiply2(Module):
        def forward(self, *args):
            return args[0] * 1

    class Multiply3(Module):
        def forward(self, *args):
            ones = torch.ones([1, 3, 224, 224])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] * ones

    class Multiply4(Module):
        def forward(self, *args):
            ones = torch.ones([1, 1, 224, 224])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] * ones

    class Multiply5(Module):
        def forward(self, *args):
            ones = torch.ones([])
            if torch.cuda.is_available():
                ones = ones.cuda()
            return args[0] * ones

    input_data = torch.rand(input_shape).float()
    verify_model(Multiply1().float().eval(), input_data=input_data)
    verify_model(Multiply2().float().eval(), input_data=input_data)
    verify_model(Multiply3().float().eval(), input_data=input_data)
    verify_model(Multiply4().float().eval(), input_data=input_data)
    verify_model(Multiply5().float().eval(), input_data=input_data)

def test_forward_unsqueeze():
    input_shape = [1, 3, 224, 224]

    class Unsqueeze1(Module):
        def forward(self, *args):
            return args[0].unsqueeze(2)

    input_data = torch.rand(input_shape).float()
    verify_model(Unsqueeze1().float().eval(), input_data=input_data)

def test_forward_concatenate():
    input_shape = [1, 3, 224, 224]

    class Concatenate1(Module):
        def forward(self, *args):
            return torch.cat([args[0][:, 0].unsqueeze(1), args[0][:, 1].unsqueeze(1)], 1)

    class Concatenate2(Module):
        def forward(self, *args):
            a = (args[0][:, :, 0] + 2) * 7
            b = (args[0][:, :, 1] + 3) * 11
            c = (args[0][:, :, 2] + 5) * 13
            return torch.cat([t.unsqueeze(2) for t in [a, b, c]], 2)

    input_data = torch.rand(input_shape).float()
    verify_model(Concatenate1().float().eval(), input_data=input_data)
    verify_model(Concatenate2().float().eval(), input_data=input_data)

def test_forward_relu():
    input_shape = [1, 3, 224, 224]

    class ReLU1(Module):
        def forward(self, *args):
            return torch.nn.ReLU()(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(ReLU1().float().eval(), input_data=input_data)

def test_forward_adaptiveavgpool1():
    input_shape = [1, 3, 224, 224]

    class AdaptiveAvgPool2D1(Module):
        def forward(self, *args):
            return torch.nn.AdaptiveAvgPool2d([1, 1])(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(AdaptiveAvgPool2D1().float().eval(), input_data=input_data)

def test_forward_adaptiveavgpool2():
    input_shape = [1, 3, 224, 224]

    class AdaptiveAvgPool2D2(Module):
        def forward(self, *args):
            return torch.nn.AdaptiveAvgPool2d([100, 100])(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(AdaptiveAvgPool2D2().float().eval(), input_data=input_data)

def test_forward_adaptiveavgpool3():
    input_shape = [1, 3, 224, 224]

    class AdaptiveAvgPool2D3(Module):
        def forward(self, *args):
            return torch.nn.AdaptiveAvgPool2d([224, 224])(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(AdaptiveAvgPool2D3().float().eval(), input_data=input_data)

def test_forward_maxpool1():
    input_shape = [1, 3, 224, 224]

    class MaxPool2D1(Module):
        def forward(self, *args):
            return torch.nn.MaxPool2d(kernel_size=[1, 1])(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(MaxPool2D1().float().eval(), input_data=input_data)

def test_forward_maxpool2():
    input_shape = [1, 3, 224, 224]

    class MaxPool2D2(Module):
        def forward(self, *args):
            return torch.nn.MaxPool2d(kernel_size=[100, 100])(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(MaxPool2D2().float().eval(), input_data=input_data)

def test_forward_maxpool3():
    input_shape = [1, 3, 224, 224]

    class MaxPool2D3(Module):
        def forward(self, *args):
            return torch.nn.MaxPool2d(kernel_size=[224, 224])(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(MaxPool2D3().float().eval(), input_data=input_data)

def test_forward_avgpool():
    input_shape = [1, 3, 224, 224]

    class AvgPool2D1(Module):
        def forward(self, *args):
            return torch.nn.AvgPool2d(kernel_size=[100, 100])(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(AvgPool2D1().float().eval(), input_data=input_data)

def test_forward_hardtanh():
    input_shape = [1, 3, 224, 224]

    class HardTanh1(Module):
        def forward(self, *args):
            return torch.nn.Hardtanh()(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(HardTanh1().float().eval(), input_data=input_data)

def test_forward_conv():
    input_shape = [1, 3, 224, 224]

    class Conv2D1(Module):
        def __init__(self):
            super(Conv2D1, self).__init__()
            self.conv = torch.nn.Conv2d(3, 64, 7, bias=True)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    class Conv2D2(Module):
        def __init__(self):
            super(Conv2D2, self).__init__()
            self.conv = torch.nn.Conv2d(3, 64, 7, bias=False)
            self.softmax = torch.nn.Softmax()

        def forward(self, *args):
            return self.softmax(self.conv(args[0]))

    input_data = torch.rand(input_shape).float()
    verify_model(Conv2D1().float().eval(), input_data=input_data)
    verify_model(Conv2D2().float().eval(), input_data=input_data)

def test_forward_threshold():
    input_shape = [1, 3, 224, 224]

    class Threshold1(Module):
        def forward(self, *args):
            return torch.nn.Threshold(0, 0)(args[0])

    input_data = torch.rand(input_shape).float()
    verify_model(Threshold1().float().eval(), input_data=input_data)

def test_forward_contiguous():
    input_shape = [1, 3, 224, 224]

    class Contiguous1(Module):
        def forward(self, *args):
            return args[0].contiguous()

    input_data = torch.rand(input_shape).float()
    verify_model(Contiguous1().float().eval(), input_data=input_data)

def test_forward_batchnorm():
    input_shape = [1, 3, 224, 224]

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

    input_data = torch.rand(input_shape).float()
    verify_model(BatchNorm1().float().eval(), input_data=input_data)
    verify_model(BatchNorm2().float().eval(), input_data=input_data)

def test_forward_transpose():
    input_shape = [1, 3, 224, 224]

    class Transpose1(Module):
        def forward(self, *args):
            return args[0].transpose(2, 3)

    class Transpose2(Module):
        def forward(self, *args):
            return args[0].transpose(-2, -1)

    input_data = torch.rand(input_shape).float()
    verify_model(Transpose1().float().eval(), input_data=input_data)
    verify_model(Transpose2().float().eval(), input_data=input_data)

def test_forward_size():
    input_shape = [1, 3, 224, 224]

    class Size1(Module):
        def forward(self, *args):
            return args[0].size(0) * args[0]

    input_data = torch.rand(input_shape).float()
    verify_model(Size1().float().eval(), input_data=input_data)

def test_forward_view():
    input_shape = [1, 3, 224, 224]

    class View1(Module):
        def forward(self, *args):
            return args[0].view((1, 3 * 224 * 224))

    class View2(Module):
        def forward(self, *args):
            return args[0].view(args[0].shape[0], -1)

    input_data = torch.rand(input_shape).float()
    verify_model(View1().float().eval(), input_data=input_data)
    verify_model(View2().float().eval(), input_data=input_data)

def test_forward_select():
    input_shape = [1, 3, 224, 224]

    class Select1(Module):
        def forward(self, *args):
            return args[0].select(1, 1)

    input_data = torch.rand(input_shape).float()
    verify_model(Select1().float().eval(), input_data=input_data)

def test_forward_clone():
    input_shape = [1, 3, 224, 224]

    class Clone1(Module):
        def forward(self, *args):
            return args[0].clone()

    input_data = torch.rand(input_shape).float()
    verify_model(Clone1().float().eval(), input_data=input_data)

def test_forward_logsoftmax():
    input_shape = [1, 3, 224, 224]

    class LogSoftmax1(Module):
        def forward(self, *args):
            return torch.nn.LogSoftmax(dim=1)(args[0][0, 0])

    input_data = torch.rand(input_shape).float()
    verify_model(LogSoftmax1().float().eval(), input_data=input_data)

def test_forward_sigmoid():
    input_shape = [1, 3, 224, 224]

    class Sigmoid1(Module):
        def forward(self, *args):
            return torch.nn.Sigmoid()(args[0])
    input_data = torch.rand(input_shape).float()
    verify_model(Sigmoid1().float().eval(), input_data=input_data)

def test_forward_dense():
    input_shape = [1, 3, 224, 224]

    class Dense1(Module):
        def __init__(self):
            super(Dense1, self).__init__()
            self.linear = torch.nn.Linear(224, 7, bias=True)
        def forward(self, *args):
            return self.linear(args[0][0, 0])

    class Dense2(Module):
        def __init__(self):
            super(Dense2, self).__init__()
            self.linear = torch.nn.Linear(224, 7, bias=False)
        def forward(self, *args):
            return self.linear(args[0][0, 0])

    input_data = torch.rand(input_shape).float()
    verify_model(Dense1().float().eval(), input_data=input_data)
    verify_model(Dense2().float().eval(), input_data=input_data)

def test_forward_dropout():
    input_shape = [1, 3, 224, 224]

    class Dropout1(Module):
        def forward(self, *args):
            return torch.nn.functional.dropout(args[0][0, 0], 0.5, False)

    input_data = torch.rand(input_shape).float()
    verify_model(Dropout1().float().eval(), input_data=input_data)

def test_forward_slice():
    input_shape = [1, 3, 224, 224]

    class Slice1(Module):
        def forward(self, *args):
            return args[0][:, :, :, :3]

    class Slice2(Module):
        def forward(self, *args):
            return args[0][0, :, :, :]

    input_data = torch.rand(input_shape).float()
    verify_model(Slice1().float().eval(), input_data=input_data)
    verify_model(Slice2().float().eval(), input_data=input_data)

def test_forward_mean():
    input_shape = [1, 3, 224, 224]

    class Mean1(Module):
        def forward(self, *args):
            return args[0].mean(2)

    input_data = torch.rand(input_shape).float()
    verify_model(Mean1().float().eval(), input_data=input_data)

def test_forward_expand():
    input_shape = [1, 3, 224, 224]

    class Expand1(Module):
        def forward(self, *args):
            return args[0].expand((3, -1, -1, -1))

    input_data = torch.rand(input_shape).float()
    verify_model(Expand1().float().eval(), input_data=input_data)

def test_forward_pow():
    input_shape = [1, 3, 224, 224]

    class Pow1(Module):
        def forward(self, *args):
            return args[0] ** 2

    input_data = torch.rand(input_shape).float()
    verify_model(Pow1().float().eval(), input_data=input_data)

def test_forward_chunk():
    input_shape = [1, 3, 224, 224]

    class Chunk1(Module):
        def forward(self, *args):
            chunks = args[0].chunk(7, 2)
            return torch.cat(chunks, 2)

    input_data = torch.rand(input_shape).float()
    verify_model(Chunk1().float().eval(), input_data=input_data)

# Model tests
def test_resnet18():
    verify_model('resnet18')

def test_resnet34():
    verify_model('resnet34')

def test_resnet50():
    verify_model('resnet50')

def test_resnet101():
    verify_model('resnet101')

def test_resnet152():
    verify_model('resnet152')

def test_squeezenet1_0():
    verify_model('squeezenet1_0')

def test_squeezenet1_1():
    verify_model('squeezenet1_1')

def test_vgg11():
    verify_model('vgg11')

def test_vgg13():
    verify_model('vgg13')

def test_vgg16():
    verify_model('vgg16')

def test_vgg19():
    verify_model('vgg19')

def test_vgg11_bn():
    verify_model('vgg11_bn')

def test_vgg13_bn():
    verify_model('vgg13_bn')

def test_vgg19_bn():
    verify_model('vgg19_bn')

def test_mobilenet_v2():
    verify_model('mobilenet_v2')

def test_densenet121():
    verify_model('densenet121')

def test_densenet161():
    verify_model('densenet161')

def test_densenet169():
    verify_model('densenet169')

def test_densenet201():
    verify_model('densenet201')

def test_inception_v3():
    verify_model('inception_v3')

def test_alexnet():
    verify_model('alexnet')

def test_googlenet():
    verify_model('googlenet')

def test_mnasnet0_5():
    verify_model('mnasnet0_5')

def test_mnasnet1_0():
    verify_model('mnasnet1_0')

if __name__ == '__main__':

    # Single operator tests
    test_forward_add()
    test_forward_subtract()
    test_forward_multiply()
    test_forward_unsqueeze()
    test_forward_concatenate()
    test_forward_relu()
    test_forward_adaptiveavgpool1()
    test_forward_maxpool1()
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

    """
    # Model tests
    test_resnet18()
    test_resnet34()
    test_resnet50()
    test_resnet101()
    test_resnet152()
    test_squeezenet1_0()
    test_squeezenet1_1()
    test_vgg11()
    test_vgg13()
    test_vgg16()
    test_vgg19()
    test_vgg11_bn()
    test_vgg13_bn()
    test_vgg19_bn()
    test_mobilenet_v2()
    test_densenet121()
    test_densenet161()
    test_densenet169()
    test_densenet201()
    test_inception_v3()
    test_alexnet()
    test_googlenet()
    test_mnasnet0_5()
    test_mnasnet1_0()
    """