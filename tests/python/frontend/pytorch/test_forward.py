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
import sys
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
        return model, [input_data]

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
    return model, [input_data]

def load_model(model_name):
    """Given a model name, returns a model as well as an example input."""
    if hasattr(torchvision.models, model_name):
        return load_torchvision(model_name)
    try:
        import pretrainedmodels
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
        if isinstance(model, Module):
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

def verify_model(model_name, input_data=[],
                 custom_convert_map={},
                 ctx_list=ctx_list()):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    if isinstance(model_name, str):
        baseline_model, baseline_input = load_model(model_name)
    elif isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or len(input_data.shape) == 0:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"

    if torch.cuda.is_available():
        baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*baseline_input)

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.float().cpu().numpy(),)

    trace = torch.jit.trace(baseline_model, baseline_input).float().eval()

    if torch.cuda.is_available():
        trace = trace.cuda()
    else:
        trace = trace.cpu()

    input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
    input_shapes = list(zip(input_names,
                            [inp.shape for inp in baseline_input]))
    mod, params = relay.frontend.from_pytorch(trace, input_shapes,
                                              custom_convert_map)
    compiled_input = dict(zip(input_names,
                              [inp.cpu().numpy() for inp in baseline_input]))

    with relay.build_config(opt_level=3):
        for target, ctx in ctx_list:
            relay_graph, relay_lib, relay_params = relay.build(mod, target=target, params=params)
            relay_model = graph_runtime.create(relay_graph, relay_lib, ctx)
            relay_model.set_input(**relay_params)
            for name, inp in compiled_input.items():
                relay_model.set_input(name, inp)
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
            return args[0] * 1.0

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

    input_data = torch.rand(input_shape).float()
    verify_model(Multiply1().float().eval(), input_data=input_data)
    verify_model(Multiply2().float().eval(), input_data=input_data)
    verify_model(Multiply3().float().eval(), input_data=input_data)
    verify_model(Multiply4().float().eval(), input_data=input_data)

def test_forward_reciprocal():
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]
    class Reciprocal1(Module):
        def forward(self, *args):
            return args[0].reciprocal()

    input_data = torch.rand(input_shape).float()
    verify_model(Reciprocal1().float().eval(), input_data=input_data)

def test_forward_repeat():
    torch.set_grad_enabled(False)
    input_shape = [1, 3]
    class Repeat1(Module):
        def forward(self, *args):
            return args[0].repeat(1, 1)

    class Repeat2(Module):
        def forward(self, *args):
            return args[0].repeat(4, 2)

    class Repeat3(Module):
        def forward(self, *args):
            return args[0].repeat(4, 2, 1)

    input_data = torch.rand(input_shape).float()
    verify_model(Repeat1().float().eval(), input_data=input_data)
    verify_model(Repeat2().float().eval(), input_data=input_data)
    verify_model(Repeat3().float().eval(), input_data=input_data)

def test_forward_repeat_interleave():
    torch.set_grad_enabled(False)
    input_shape = [2, 2, 3]
    class RepeatInterleave1(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(2)

    class RepeatInterleave2(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(3, dim=0)

    class RepeatInterleave3(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(2, dim=1)

    class RepeatInterleave4(Module):
        def forward(self, *args):
            return args[0].repeat_interleave(4, dim=2)

    input_data = torch.rand(input_shape).float()
    verify_model(RepeatInterleave1().float().eval(), input_data=input_data)
    verify_model(RepeatInterleave2().float().eval(), input_data=input_data)
    verify_model(RepeatInterleave3().float().eval(), input_data=input_data)
    verify_model(RepeatInterleave4().float().eval(), input_data=input_data)

def test_forward_unsqueeze():
    torch.set_grad_enabled(False)
    input_shape = [10, 10]

    class Unsqueeze1(Module):
        def forward(self, *args):
            return args[0].unsqueeze(2)

    input_data = torch.rand(input_shape).float()
    verify_model(Unsqueeze1().float().eval(), input_data=input_data)

def test_forward_squeeze():
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]

    class Squeeze1(Module):
        def forward(self, *args):
            return args[0].squeeze()

    class Squeeze2(Module):
        def forward(self, *args):
            return args[0].squeeze(1)

    input_data = torch.rand(input_shape).float()
    verify_model(Squeeze1().float().eval(), input_data=input_data)
    verify_model(Squeeze2().float().eval(), input_data=input_data)

def test_forward_arange():
    torch.set_grad_enabled(False)

    class Arange1(Module):
        def forward(self, *args):
            return torch.arange(5)
    class Arange2(Module):
        def forward(self, *args):
            return torch.arange(2.5)
    class Arange3(Module):
        def forward(self, *args):
            return torch.arange(1, 4)
    class Arange4(Module):
        def forward(self, *args):
            return torch.arange(1, 2.5, 0.5)
    class Arange5(Module):
        def forward(self, *args):
            return torch.arange(1, 2, 1, dtype=torch.int32)
    class Arange6(Module):
        def forward(self, *args):
            return torch.arange(start=1, end=6, step=2)
    class Arange7(Module):
        def forward(self, *args):
            return torch.arange(1, 4, dtype=torch.float32)
    class Arange8(Module):
        def forward(self, *args):
            return torch.arange(1, 2, 1, dtype=torch.int16)

    verify_model(Arange1().float().eval())
    verify_model(Arange2().float().eval())
    verify_model(Arange3().float().eval())
    verify_model(Arange4().float().eval())
    verify_model(Arange5().float().eval())
    verify_model(Arange6().float().eval())
    verify_model(Arange7().float().eval())
    verify_model(Arange8().float().eval())

def test_forward_abs():
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]

    class Abs1(Module):
        def forward(self, *args):
            return args[0].abs()

    input_data = torch.rand(input_shape).float()
    verify_model(Abs1().float().eval(), input_data=input_data)

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

    input_data = torch.rand(input_shape).float()
    verify_model(Concatenate1().float().eval(), input_data=input_data)
    verify_model(Concatenate2().float().eval(), input_data=input_data)

def test_forward_relu():
    torch.set_grad_enabled(False)
    input_shape = [10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.ReLU().eval(), input_data=input_data)

def test_forward_prelu():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.PReLU(num_parameters=3).eval(), input_data=input_data)

def test_forward_leakyrelu():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.LeakyReLU().eval(), input_data=input_data)
    verify_model(torch.nn.LeakyReLU(negative_slope=0.05).eval(), input_data=input_data)
    verify_model(torch.nn.LeakyReLU(negative_slope=1.0).eval(), input_data=input_data)
    verify_model(torch.nn.LeakyReLU(negative_slope=1.25).eval(), input_data=input_data)

def test_forward_elu():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.ELU().eval(), input_data=input_data)
    verify_model(torch.nn.ELU(alpha=0.3).eval(), input_data=input_data)
    verify_model(torch.nn.ELU(alpha=1.0).eval(), input_data=input_data)
    verify_model(torch.nn.ELU(alpha=1.3).eval(), input_data=input_data)

def test_forward_celu():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.CELU().eval(), input_data=input_data)
    verify_model(torch.nn.CELU(alpha=0.3).eval(), input_data=input_data)
    verify_model(torch.nn.CELU(alpha=1.0).eval(), input_data=input_data)
    verify_model(torch.nn.CELU(alpha=1.3).eval(), input_data=input_data)

def test_forward_gelu():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.GELU().eval(), input_data=input_data)

def test_forward_selu():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.SELU().eval(), input_data=input_data)

def test_forward_softplus():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Softplus().eval(), input_data=input_data)
    verify_model(torch.nn.Softplus(beta=1.5, threshold=20).eval(), input_data=input_data)
    verify_model(torch.nn.Softplus(beta=5, threshold=10).eval(), input_data=input_data)

def test_forward_softsign():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Softsign().eval(), input_data=input_data)

def test_forward_log_sigmoid():
    torch.set_grad_enabled(False)
    input_shape = [10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.LogSigmoid().eval(), input_data=input_data)

def test_forward_adaptiveavgpool():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AdaptiveAvgPool2d([1, 1]).eval(), input_data=input_data)
    verify_model(torch.nn.AdaptiveAvgPool2d([10, 10]).eval(), input_data=input_data)

def test_forward_maxpool2d():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()

    verify_model(torch.nn.MaxPool2d(kernel_size=[1, 1]).eval(),
                input_data)
    verify_model(torch.nn.MaxPool2d(kernel_size=[10, 10]).eval(),
                input_data)
    verify_model(torch.nn.MaxPool2d(kernel_size=[4, 4],
                                    padding=2,
                                    stride=2).eval(),
                input_data)

def test_forward_maxpool1d():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10]
    input_data = torch.rand(input_shape).float()

    verify_model(torch.nn.MaxPool1d(kernel_size=1).eval(),
                input_data)
    verify_model(torch.nn.MaxPool1d(kernel_size=10).eval(),
                input_data)
    verify_model( torch.nn.MaxPool1d(kernel_size=4,
                                    padding=2,
                                    stride=2).eval(),
                input_data)

def test_forward_maxpool3d():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10, 10]
    input_data = torch.rand(input_shape).float()

    verify_model(torch.nn.MaxPool3d(kernel_size=[1, 1, 1]).eval(),
                input_data)
    verify_model(torch.nn.MaxPool3d(kernel_size=[10, 10, 10]).eval(),
                input_data)
    verify_model(torch.nn.MaxPool3d(kernel_size=[4, 4, 4],
                                    padding=2,
                                    stride=2).eval(),
                input_data)

def test_forward_split():
    torch.set_grad_enabled(False)
    input_shape = [4, 10]

    class Split(Module):
        def __init__(self, split_size_or_sections, dim):
            super(Split, self).__init__()
            self.split_size_or_sections = split_size_or_sections
            self.dim = dim

        def forward(self, *args):
            return torch.split(args[0], self.split_size_or_sections, self.dim)

    input_data = torch.rand(input_shape).float()
    verify_model(Split(2, 0).float().eval(),
                input_data=input_data)
    verify_model(Split(3, 1).float().eval(),
                input_data=input_data)
    verify_model(Split(4, 1).float().eval(),
                input_data=input_data)
    verify_model(Split([2, 3, 5], 1).float().eval(),
                input_data=input_data)

def test_forward_avgpool():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class AvgPool2D2(Module):
        def forward(self, *args):
            return torch.nn.functional.avg_pool2d(args[0], kernel_size=[10, 10])

    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AvgPool2d(kernel_size=[10, 10]).eval(), input_data=input_data)
    verify_model(AvgPool2D2().float().eval(), input_data=input_data)

def test_forward_avgpool3d():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10, 10]

    class AvgPool3D1(Module):
        def forward(self, *args):
            return torch.nn.functional.avg_pool3d(args[0], kernel_size=[10, 10, 10])

    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.AvgPool3d(kernel_size=[10, 10, 10]).eval(), input_data=input_data)
    verify_model(AvgPool3D1().float().eval(), input_data=input_data)

def test_forward_hardtanh():
    torch.set_grad_enabled(False)
    input_shape = [10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Hardtanh().eval(), input_data=input_data)

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

    input_data = torch.rand(input_shape).float()
    verify_model(Conv2D1().float().eval(), input_data=input_data)
    verify_model(Conv2D2().float().eval(), input_data=input_data)
    # depth wise conv with channel mult 2
    verify_model(Conv2D3().float().eval(), input_data=input_data)
    # group conv
    verify_model(torch.nn.Conv2d(8, 8, kernel_size=(3, 3),
                                 stride=(1, 1), groups=2).eval(),
                 input_data=torch.randn((1, 8, 16, 16)))


def test_forward_conv_transpose():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.ConvTranspose2d(3, 6, 7, bias=True), input_data=input_data)
    verify_model(torch.nn.ConvTranspose2d(3, 12, 3, bias=False), input_data=input_data)


def test_forward_threshold():
    torch.set_grad_enabled(False)
    input_shape = [1, 3]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Threshold(0, 0).float().eval(), input_data=input_data)

def test_forward_contiguous():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Contiguous1(Module):
        def forward(self, *args):
            return args[0].contiguous()

    input_data = torch.rand(input_shape).float()
    verify_model(Contiguous1().float().eval(), input_data=input_data)


def test_forward_batchnorm():
    def init_weight(m):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.normal_(m.bias)

    inp_2d = torch.rand((1, 16, 10, 10))
    inp_3d = torch.rand((1, 16, 10, 10, 10))

    for bn, inp in [(torch.nn.BatchNorm2d(16), inp_2d),
                    (torch.nn.BatchNorm3d(16), inp_3d)]:
        init_weight(bn.eval())
        verify_model(bn.eval(), input_data=inp)


def test_forward_instancenorm():
    inp_2d = torch.rand((1, 16, 10, 10))
    inp_3d = torch.rand((1, 16, 10, 10, 10))

    for ins_norm, inp in [(torch.nn.InstanceNorm2d(16), inp_2d),
                          (torch.nn.InstanceNorm3d(16), inp_3d)]:
        verify_model(ins_norm.eval(), input_data=inp)

def test_forward_layernorm():
    def init_weight(m):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.normal_(m.bias, 0.02)

    inp_2d = torch.rand((1, 16, 10, 10))
    inp_3d = torch.rand((1, 16, 10, 10, 10))
    for ln, inp in [(torch.nn.LayerNorm(10), inp_2d),
                    (torch.nn.LayerNorm(10), inp_3d)]:
        init_weight(ln.eval())
        verify_model(ln.eval(), input_data=inp)

def test_forward_reshape():
    torch.set_grad_enabled(False)
    input_shape = [2, 1, 10, 1, 10]
    new_shape = [2, 1, 10, 10]
    class Reshape1(Module):
        def forward(self, *args):
            return args[0].reshape(new_shape)

    class Reshape2(Module):
        def forward(self, *args):
            return args[0].reshape([-1])

    input_data = torch.rand(input_shape).float()
    verify_model(Reshape1().float().eval(), input_data=input_data)
    verify_model(Reshape2().float().eval(), input_data=input_data)

def test_forward_transpose():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

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
    torch.set_grad_enabled(False)
    input_shape = [1, 3]

    class Size1(Module):
        def forward(self, *args):
            return float(args[0].size(0)) * args[0]

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

    input_data = torch.rand(input_shape).float()
    verify_model(View1().float().eval(), input_data=input_data)
    verify_model(View2().float().eval(), input_data=input_data)

def test_forward_select():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Select1(Module):
        def forward(self, *args):
            return args[0].select(1, 1)

    input_data = torch.rand(input_shape).float()
    verify_model(Select1().float().eval(), input_data=input_data)

def test_forward_clone():
    torch.set_grad_enabled(False)
    input_shape = [10]

    class Clone1(Module):
        def forward(self, *args):
            return args[0].clone()

    input_data = torch.rand(input_shape).float()
    verify_model(Clone1().float().eval(), input_data=input_data)

def test_forward_logsoftmax():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class LogSoftmax1(Module):
        def forward(self, *args):
            return torch.nn.LogSoftmax(dim=1)(args[0][0, 0])

    input_data = torch.rand(input_shape).float()
    verify_model(LogSoftmax1().float().eval(), input_data=input_data)

def test_forward_sigmoid():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Sigmoid().eval(), input_data=input_data)

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

    input_data = torch.rand(input_shape).float()
    verify_model(Dense1().float().eval(), input_data=input_data)
    verify_model(Dense2().float().eval(), input_data=input_data)

def test_forward_dropout():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]
    input_data = torch.rand(input_shape).float()
    verify_model(torch.nn.Dropout(p=0.5).eval(), input_data=input_data[0, 0])
    verify_model(torch.nn.Dropout2d(p=0.5).eval(), input_data=input_data[0])
    verify_model(torch.nn.Dropout3d(p=0.5).eval(), input_data=input_data)
    verify_model(torch.nn.AlphaDropout(p=0.5).eval(), input_data=input_data[0, 0])

def test_forward_slice():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

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
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Mean1(Module):
        def forward(self, *args):
            return args[0].mean(2)

    input_data = torch.rand(input_shape).float()
    verify_model(Mean1().float().eval(), input_data=input_data)

def test_forward_expand():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Expand1(Module):
        def forward(self, *args):
            return args[0].expand((3, -1, -1, -1))

    input_data = torch.rand(input_shape).float()
    verify_model(Expand1().float().eval(), input_data=input_data)

def test_forward_pow():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 10, 10]

    class Pow1(Module):
        def forward(self, *args):
            return args[0] ** 2

    input_data = torch.rand(input_shape).float()
    verify_model(Pow1().float().eval(), input_data=input_data)

def test_forward_chunk():
    torch.set_grad_enabled(False)
    input_shape = [1, 3, 14, 14]

    class Chunk1(Module):
        def forward(self, *args):
            chunks = args[0].chunk(7, 2)
            return torch.cat(chunks, 2)

    input_data = torch.rand(input_shape).float()
    verify_model(Chunk1().float().eval(), input_data=input_data)

def test_upsample():
    class Upsample(Module):
        def __init__(self, size=None, scale=None,
                     mode="nearest", align_corners=None):
            super().__init__()
            self.size = size
            self.scale = scale
            self.mode = mode
            self.align_corners = align_corners

        def forward(self, x):
            return torch.nn.functional.interpolate(x, size=self.size,
                                                   scale_factor=self.scale,
                                                   mode=self.mode,
                                                   align_corners=self.align_corners)
    inp = torch.rand((1, 3, 32, 32))
    verify_model(Upsample(size=(64, 64), mode="nearest"), inp)
    verify_model(Upsample(scale=2, mode="nearest"), inp)
    verify_model(Upsample(size=(50, 50), mode="nearest"), inp)
    verify_model(Upsample(size=(64, 64), mode="bilinear", align_corners=True), inp)
    verify_model(Upsample(scale=2, mode="bilinear", align_corners=True), inp)
    verify_model(Upsample(size=(50, 50), mode="bilinear", align_corners=True), inp)

def test_to():
    """ test for aten::to(...) """
    class ToCPU(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.to("cpu")

    class ToFloat(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.float()

    class ToInt(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.int()

    verify_model(ToCPU().eval(), torch.rand((1, 3, 32, 32)))
    verify_model(ToFloat().eval(), torch.zeros((1, 3, 32, 32), dtype=torch.int))
    verify_model(ToFloat().eval(), torch.tensor(2, dtype=torch.int))
    verify_model(ToInt().eval(), torch.zeros((1, 3, 32, 32)))
    verify_model(ToInt().eval(), torch.tensor(2.0))


def test_adaptive_pool3d():
    for ishape in [(1, 32, 16, 16, 16),
                   (1, 32, 9, 15, 15),
                   (1, 32, 13, 7, 7)]:
        inp = torch.rand(ishape)
        verify_model(torch.nn.AdaptiveMaxPool3d((1, 1, 1)).eval(), inp)
        verify_model(torch.nn.AdaptiveMaxPool3d((2, 2, 2)).eval(), inp)
        verify_model(torch.nn.AdaptiveAvgPool3d((1, 1, 1)).eval(), inp)
        verify_model(torch.nn.AdaptiveAvgPool3d((2, 2, 2)).eval(), inp)
        verify_model(torch.nn.AdaptiveAvgPool3d((4, 8, 8)).eval(), inp)
        verify_model(torch.nn.AdaptiveMaxPool3d((7, 8, 9)).eval(), inp)


def test_conv3d():
    for ishape in [(1, 32, 16, 16, 16),
                   (1, 32, 9, 15, 15),
                   (1, 32, 13, 7, 7)]:
        inp = torch.rand(ishape)
        verify_model(torch.nn.Conv3d(32, 16, (3, 3, 3),
                                     padding=(1, 1, 1)).eval(),
                     inp),
        verify_model(torch.nn.Conv3d(32, 16, (5, 5, 5),
                                     padding=(2, 2, 2)).eval(),
                     inp),
        verify_model(torch.nn.Conv3d(32, 16, kernel_size=1).eval(),
                     inp)
        # downsample
        verify_model(torch.nn.Conv3d(32, 16, kernel_size=1, stride=2).eval(),
                     inp)


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

def test_mobilenet_v2():
    torch.set_grad_enabled(False)
    verify_model("mobilenet_v2")

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
"""

def test_custom_conversion_map():
    def get_roi_align():
        pool_size = 5
        n_channels = 2 * (pool_size ** 2)
        x = torch.rand(2, n_channels, 10, 10)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]], dtype=torch.float)
        roi_align = torchvision.ops.RoIAlign(pool_size, spatial_scale=1,
                                             sampling_ratio=-1)
        return roi_align.eval(), [x, rois]

    def convert_roi_align():
        def _impl(inputs, input_types):
            spatial_scale = inputs[2]
            pooled_size = (inputs[3], inputs[4])
            sampling_ratio = inputs[5]
            return relay.op.vision.roi_align(inputs[0], inputs[1],
                                             pooled_size, spatial_scale,
                                             sampling_ratio)
        return _impl

    custom_map = {'torchvision::roi_align': convert_roi_align()}
    model, inputs = get_roi_align()

    verify_model(model, inputs, custom_map)


def test_segmentaton_models():
    class SegmentationModelWrapper(Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            out = self.model(inp)
            return out["out"]

    fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
    deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    inp = [torch.rand((1, 3, 300, 300), dtype=torch.float)]

    verify_model(SegmentationModelWrapper(fcn.eval()), inp)

    # depthwise + dilated covolution not supported on x86
    # see https://github.com/apache/incubator-tvm/issues/4962
    cuda_ctx = ("cuda", tvm.gpu(0))
    if cuda_ctx[1].exist:
        verify_model(SegmentationModelWrapper(deeplab.eval()), inp, [cuda_ctx])


def test_3d_models():
    input_shape = (1, 3, 4, 56, 56)
    resnet3d = torchvision.models.video.r3d_18(pretrained=True).eval()
    verify_model(resnet3d, [torch.rand(input_shape)])


def verify_script_model(pt_model, ishapes):
    script_module = torch.jit.script(pt_model)

    input_names = ["i{}".format(idx) for idx, ish in enumerate(ishapes)]
    input_shapes = list(zip(input_names, ishapes))

    inputs = [torch.randn(shape, dtype=torch.float)
              for shape in ishapes]

    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    executor = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0),
                                     target="llvm")
    evaluator = executor.evaluate()

    for name, inp in zip(input_names, inputs):
        params[name] = inp.numpy()

    op_res = evaluator(**params)

    with torch.no_grad():
        pt_result = pt_model(*inputs)

    if not isinstance(pt_result, torch.Tensor):
        tvm_res = op_res.asnumpy().item()
        assert pt_result == tvm_res
    else:
        tvm.testing.assert_allclose(op_res.asnumpy(), pt_result.numpy(),
                                    rtol=1e-5, atol=1e-5)


def test_control_flow():
    class SimpleIf(torch.nn.Module):
        def __init__(self, N, M):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(N, M))

        def forward(self, inp):
            if inp.sum() > 0.:
                output = self.weight + inp
            else:
                output = self.weight - inp
            return output

    class NestedIf(torch.nn.Module):
        def __init__(self, N, M):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(N, M))

        def forward(self, inp):
            if inp.sum() > 0.:
                if inp.mean() > 0.:
                    output = self.weight + inp
                else:
                    output = self.weight - inp
            else:
                if inp.mean() >= 0.:
                    output = self.weight * inp
                else:
                    output = self.weight / inp

            return output

    class ScalarLoop(torch.nn.Module):
        def forward(self, inp):
            a = 0
            for i in range(inp.size(0)):
                b = i * i
                b = b + 1
                a += b
            if a != 0:
                a += 1
            else:
                a += 2
            return a

    class SimpleLoop(torch.nn.Module):
        def forward(self, inp):
            a = inp
            for i in range(inp.size(0)):
                b = a * 2.
                c = a + b
                a += c
            return a

    class LoopWithIf(torch.nn.Module):
        def forward(self, inp):
            a = inp
            for i in range(inp.size(0)):
                b = a * 2.
                b = a + b
                if b.sum() > 0.0:
                    a += b
                else:
                    a -= b
            return a

    class NestedLoop(torch.nn.Module):
        def forward(self, inp):
            a = inp
            for i in range(inp.size(0)):
                b = a * float(i)
                for j in range(inp.size(1)):
                    a += b * float(j)
            return a

    class SimpleScalarWhileLoop(torch.nn.Module):
        def forward(self, inp):
            a = 1
            i = 0
            while i <= inp.size(0):
                a += i
                i += 2
            i = 0
            # also test constant init cond
            while i < 10:
                a += i
                i += 3
            return a

    class SimpleWhileLoop(torch.nn.Module):
        def forward(self, inp):
            a = inp
            i = 0
            while i < inp.size(0):
                a += a * float(i) * 2.0
                i += 1
            return a

    models = [
        SimpleIf(10, 20),
        NestedIf(10, 20),
        ScalarLoop(),
        SimpleLoop(),
        LoopWithIf(),
        SimpleScalarWhileLoop(),
        SimpleWhileLoop(),
        NestedLoop(),
    ]

    for pt_model in models:
        verify_script_model(pt_model.eval(), [(10, 20)])


def test_simple_rnn():
    # The mixed tracing and scripting example from
    # https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#mixing-scripting-and-tracing
    class DecisionGate(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x
            else:
                return -x

    class Cell(torch.nn.Module):
        def __init__(self, dg):
            super(Cell, self).__init__()
            self.dg = dg
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.dg(self.linear(x)) + h)
            return new_h, new_h

    class RNNLoop(torch.nn.Module):
        def __init__(self):
            super().__init__()
            x = torch.rand(10, 4, dtype=torch.float)
            h = torch.rand(10, 4, dtype=torch.float)
            self.cell = torch.jit.trace(Cell(DecisionGate()), (x, h))

        def forward(self, xs):
            h = torch.zeros(10, 4, dtype=torch.float)
            y = torch.zeros(10, 4, dtype=torch.float)
            for i in range(xs.size(0)):
                y, h = self.cell(xs[i], h)
            return y

    verify_script_model(RNNLoop().eval(), [(10, 10, 4)])


if __name__ == "__main__":
    # Single operator tests
    test_forward_add()
    test_forward_subtract()
    test_forward_multiply()
    test_forward_reshape()
    test_forward_reciprocal()
    test_forward_repeat()
    test_forward_repeat_interleave()
    test_forward_squeeze()
    test_forward_unsqueeze()
    test_forward_concatenate()
    test_forward_relu()
    test_forward_prelu()
    test_forward_leakyrelu()
    test_forward_elu()
    test_forward_celu()
    test_forward_gelu()
    test_forward_selu()
    test_forward_log_sigmoid()
    test_forward_adaptiveavgpool()
    test_forward_maxpool2d()
    test_forward_maxpool1d()
    test_forward_maxpool3d()
    test_forward_hardtanh()
    test_forward_conv()
    test_forward_conv_transpose()
    test_forward_threshold()
    test_forward_contiguous()
    test_forward_batchnorm()
    test_forward_instancenorm()
    test_forward_layernorm()
    test_forward_transpose()
    test_forward_size()
    test_forward_view()
    test_forward_select()
    test_forward_clone()
    test_forward_softplus()
    test_forward_softsign()
    test_forward_logsoftmax()
    test_forward_sigmoid()
    test_forward_dense()
    test_forward_avgpool()
    test_forward_avgpool3d()
    test_forward_dropout()
    test_forward_slice()
    test_forward_mean()
    test_forward_expand()
    test_forward_pow()
    test_forward_abs()
    test_forward_arange()
    test_forward_chunk()
    test_forward_split()
    test_upsample()
    test_to()
    test_adaptive_pool3d()
    test_conv3d()

    # Model tests
    test_resnet18()
    test_squeezenet1_0()
    test_squeezenet1_1()
    test_densenet121()
    # disable inception test for now, since loading it takes ~5min on torchvision-0.5 due to scipy bug
    # See https://discuss.pytorch.org/t/torchvisions-inception-v3-takes-much-longer-to-load-than-other-models/68756
    # test_inception_v3()
    test_googlenet()
    test_mnasnet0_5()
    test_mobilenet_v2()

    test_custom_conversion_map()

    test_segmentaton_models()
    test_3d_models()

    # Quantization test
    from qnn_test import test_quantized_imagenet, test_quantized_modules

    test_quantized_modules()
    test_quantized_imagenet()

    # Test simple conditionals and loop
    test_control_flow()
    test_simple_rnn()
