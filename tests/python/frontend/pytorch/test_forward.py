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
import tvm
import torchvision
import single_op


from tvm import relay
from tvm.contrib import graph_runtime
#from tvm.relay.testing.config import ctx_list

sys.setrecursionlimit(10000)

TARGET = 'llvm'
CTX = tvm.cpu()
EXT_ACCEL = None

model_names = []
baseline_latencies_map = {}
compiled_latencies_map = {}
speedups_map = {}

test_repeats = 1

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

def load_single_op(model_name):
    """Given a model name, returns a single-operator model in eval
    mode as well as an example input."""
    model = getattr(single_op, model_name)().float().eval()
    input_shape = [1, 3, 224, 224]
    input_data = torch.rand(input_shape).float()
    return model, input_data

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
    if hasattr(single_op, model_name):
        return load_single_op(model_name)
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

def verify_model(model_name):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    baseline_model, baseline_input = load_model(model_name)
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
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.pth')
        torch.jit.save(trace, path)
        mod, params = relay.frontend.from_pytorch(trace, input_shapes)

    compiled_input = {input_name: tvm.nd.array(baseline_input.cpu().numpy())}

    with relay.build_config(opt_level=3):
        relay_graph, relay_lib, relay_params = relay.build(mod, target=TARGET, params=params)
        relay_model = graph_runtime.create(relay_graph, relay_lib, CTX)
        relay_model.set_input(**relay_params)
        relay_model.set_input(**compiled_input)
        relay_model.run()

    for i, baseline_output in enumerate(baseline_outputs):
        output_shape = baseline_output.shape
        compiled_output = relay_model.get_output(
            i, tvm.nd.array(np.zeros(output_shape).astype(dtype), CTX)).asnumpy()

        compiled_relay_output = relay_model.get_output(
            i, tvm.nd.array(np.zeros(output_shape).astype(dtype), CTX)).asnumpy()

        assert_shapes_match(baseline_output, compiled_output)
        tvm.testing.assert_allclose(baseline_output, compiled_output,
                                    rtol=1e-5, atol=1e-5)

        assert_shapes_match(baseline_output, compiled_relay_output)
        tvm.testing.assert_allclose(baseline_output, compiled_relay_output,
                                    rtol=1e-5, atol=1e-5)

    if(test_repeats > 0):
        thresh = 1e-2
        units = 1e3
        thresh = int(thresh * units)
        input_shapes = list(input_shapes.values())

        compiled_latencies = []
        baseline_latencies = []
        speedups = []

        for i in range(0, test_repeats):
            print("computing compiled latency")
            compiled_latency = measure_latency(relay_model, input_shapes,
                                               output_shapes, thresh) * units
            print(f'Compiled latency is {compiled_latency:.3f} +/- {thresh:d} ms.')
            print("computing baseline latency")
            baseline_latency = measure_latency(baseline_model, input_shapes,
                                               output_shapes, thresh) * units

            print(f'Baseline latency is {baseline_latency:.3f} +/- {thresh:d} ms.')

            speedup = baseline_latency/compiled_latency
            print(f'Relative speedup is {speedup:.3f}')

            compiled_latencies.append(compiled_latency)
            baseline_latencies.append(baseline_latency)
            speedups.append(speedup)

        baseline_latencies_map[model_name] = baseline_latencies
        compiled_latencies_map[model_name] = compiled_latencies
        speedups_map[model_name] = speedups
        model_names.append(model_name)

        print_results()

    from subprocess import call
    call('rm -rf ~/.torch/models/*', shell=True)

def print_results():
    print(baseline_latencies_map)
    print(compiled_latencies_map)
    print(speedups_map)

    thresh = 1e-2
    units = 1e3
    thresh = int(thresh * units)

    for model_name in model_names:

        compiled_sum = 0.0
        baseline_sum = 0.0
        speedup_sum = 0.0

        print("For model name "+model_name)
        for i in range(0, test_repeats):
            print(f'Compiled latency is {compiled_latencies_map[model_name][i]:.3f} +/- {thresh:d} ms.')
            print(f'Baseline latency is {baseline_latencies_map[model_name][i]:.3f} +/- {thresh:d} ms.')
            print(f'Relative speedup is {speedups_map[model_name][i]:.3f}')

            compiled_sum = compiled_sum + compiled_latencies_map[model_name][i]
            baseline_sum = baseline_sum + baseline_latencies_map[model_name][i]
            speedup_sum = speedup_sum + speedups_map[model_name][i]

        print(f'Average compiled latency is {compiled_sum/test_repeats:.3f} +/- {thresh:d} ms.')
        print(f'Average baseline latency is {baseline_sum/test_repeats:.3f} +/- {thresh:d} ms.')
        print(f'Average relative speedup is {speedup_sum/test_repeats:.3f}')

# Test Functions
def test_add1():
    verify_model('Add1')

def test_add2():
    verify_model('Add2')

def test_add3():
    verify_model('Add3')

def test_add4():
    verify_model('Add4')

def test_add5():
    verify_model('Add5')

def test_subtract1():
    verify_model('Subtract1')

def test_subtract2():
    verify_model('Subtract2')

def test_subtract3():
    verify_model('Subtract3')

def test_subtract4():
    verify_model('Subtract4')

def test_subtract5():
    verify_model('Subtract5')

def test_multiply1():
    verify_model('Multiply1')

def test_multiply2():
    verify_model('Multiply2')

def test_multiply3():
    verify_model('Multiply3')

def test_multiply4():
    verify_model('Multiply4')

def test_multiply5():
    verify_model('Multiply5')

def test_unsqueeze1():
    verify_model('Unsqueeze1')

def test_concatenate1():
    verify_model('Concatenate1')

def test_concatenate2():
    verify_model('Concatenate2')

def test_relu1():
    verify_model('ReLU1')

def test_adaptiveavgpool2d1():
    verify_model('AdaptiveAvgPool2D1')

def test_adaptiveavgpool2d2():
    verify_model('AdaptiveAvgPool2D2')

def test_adaptiveavgpool2d3():
    verify_model('AdaptiveAvgPool2D3')

def test_maxpool2d1():
    verify_model('MaxPool2D1')

def test_maxpool2d2():
    verify_model('MaxPool2D2')

def test_maxpool2d3():
    verify_model('MaxPool2D3')

def test_hardtanh1():
    verify_model('HardTanh1')

def test_conv2d1():
    verify_model('Conv2D1')

def test_conv2d2():
    verify_model('Conv2D2')

def test_threshold1():
    verify_model('Threshold1')

def test_contiguous1():
    verify_model('Contiguous1')

def test_batchnorm1():
    verify_model('BatchNorm1')

def test_batchnorm2():
    verify_model('BatchNorm2')

def test_transpose1():
    verify_model('Transpose1')

def test_transpose2():
    verify_model('Transpose2')

def test_size1():
    verify_model('Size1')

def test_view1():
    verify_model('View1')

def test_view2():
    verify_model('View2')

def test_select1():
    verify_model('Select1')

def test_clone1():
    verify_model('Clone1')

def test_logsoftmax1():
    verify_model('LogSoftmax1')

def test_sigmoid1():
    verify_model('Sigmoid1')

def test_dense1():
    verify_model('Dense1')

def test_dense2():
    verify_model('Dense2')

def test_avgpool2d1():
    verify_model('AvgPool2D1')

def test_dropout1():
    verify_model('Dropout1')

def test_slice1():
    verify_model('Slice1')

def test_slice2():
    verify_model('Slice2')

def test_mean1():
    verify_model('Mean1')

def test_expand1():
    verify_model('Expand1')

def test_pow1():
    verify_model('Pow1')

def test_chunk1():
    verify_model('Chunk1')

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

"""
def test_shufflenet_v2_x0_5():
    verify_model('shufflenet_v2_x0_5')

def test_shufflenet_v2_x1_0():
    verify_model('shufflenet_v2_x1_0')
"""

if __name__ == '__main__':

    # Single operator tests
    test_add1()
    test_add2()
    test_add3()
    test_add4()
    test_add5()
    test_subtract1()
    test_subtract2()
    test_subtract3()
    test_subtract4()
    test_subtract5()
    test_multiply1()
    test_multiply2()
    test_multiply3()
    test_multiply4()
    test_multiply5()
    test_unsqueeze1()
    test_concatenate1()
    test_concatenate2()
    test_relu1()
    test_adaptiveavgpool2d1()
    test_adaptiveavgpool2d2()
    test_adaptiveavgpool2d3()
    test_maxpool2d1()
    test_maxpool2d2()
    test_maxpool2d3()
    test_hardtanh1()
    test_conv2d1()
    test_conv2d2()
    test_threshold1()
    test_contiguous1()
    test_batchnorm1()
    test_batchnorm2()
    test_transpose1()
    test_transpose2()
    test_size1()
    test_view1()
    test_view2()
    test_select1()
    test_clone1()
    test_logsoftmax1()
    test_sigmoid1()
    test_dense1()
    test_dense2()
    test_avgpool2d1()
    test_dropout1()
    test_slice1()
    test_slice2()
    test_mean1()
    test_expand1()
    test_pow1()
    test_chunk1()

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

    #test_shufflenet_v2_x0_5()
    #test_shufflenet_v2_x1_0()
