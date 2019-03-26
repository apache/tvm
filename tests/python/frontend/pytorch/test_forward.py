import os
import sys
import numpy as np
import torch
import torchvision
from tempfile import TemporaryDirectory
import tvm
import nnvm
import logging
from subprocess import call

sys.setrecursionlimit(10000)
logging.basicConfig(filename='torchvision.log', level=logging.INFO)

def verify_torchvision_model(model_name, input_shape):
    dtype = 'float32'
    input_name = 'input0'
    input_shapes = {input_name: list(input_shape)}
    output_shape = (1, 1000)
    ctx = tvm.cpu()
    model_input = np.random.uniform(-128, 128, input_shape).astype(dtype)
    baseline_input = torch.tensor(model_input, requires_grad=False).float()
    baseline_model = getattr(torchvision.models, model_name)(pretrained=True).float().eval()
    baseline_output = baseline_model(baseline_input).detach().numpy()
    trace = torch.jit.trace(baseline_model, baseline_input)
    print(trace.graph)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.pth')
        torch.jit.save(trace, path)
#        sym, params, dtype_dict = nnvm.frontend.from_pytorch(path, input_shapes)
        sym, params = nnvm.frontend.from_pytorch(path, input_shapes)
    for k, v in params.items():
        print(f'{k} = {v.dtype}{v.shape}')
    compiled_input = {input_name: tvm.nd.array(baseline_input)}
    graph, lib, params = nnvm.compiler.build(sym, 'llvm', input_shapes,
#                                             dtype=dtype_dict,
                                             dtype='float32',
                                             params=params)
#    nnvm.compiler.build_module._remove_noref_params(params, graph)
    compiled_model = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    compiled_model.set_input(**params)
    compiled_model.set_input(**compiled_input)
    compiled_model.run()
    compiled_output = compiled_model.get_output(
        0, tvm.nd.array(np.zeros(output_shape).astype(dtype), ctx)).asnumpy()
    tvm.testing.assert_allclose(baseline_output, compiled_output, rtol=1e-4, atol=1e-4)


model_names = [
#    'alexnet',
#    'vgg11',
#    'vgg13',
#    'vgg16',
#    'vgg19',
#    'vgg11_bn',
#    'vgg13_bn',
#    'vgg16_bn',
#    'vgg19_bn',
#    'resnet18',
#    'resnet34',
#    'resnet50',
#    'resnet101',
#    'resnet152',
#    'squeezenet1_0',
#    'squeezenet1_1',
#    'densenet121',
#    'densenet169',
#    'densenet201',
#    'densenet161',
    'inception_v3',
]


if __name__ == '__main__':

    for model_name in model_names:
        height = width = 299 if model_name == 'inception_v3' else 224
        input_shape = (1, 3, height, width)
        verify_torchvision_model(model_name, input_shape)
#        try:
#            verify_torchvision_model(model_name, input_shape)
#            print(f'{model_name} passed')
#            logging.info(model_name + ' passed')
#        except:
#            print(f'{model_name} failed')
#            logging.error(model_name + ' failed')
        call('rm -rf /home/ubuntu/.torch/models/*', shell=True)
