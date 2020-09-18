
import tvm
import tvm.relay.testing
from tvm import relay
import torch

from torchvision.models import resnet
from tvm.data import RandomDatasetManager
from tvm.relay.transform.quantize import Quantizer, QuantizationCalibrator, AverageMaxCalibrationCallback, AverageMaxPerChannelConv2DBiasAddPattern, AverageMaxPerChannelConv2DPattern, AverageMaxPerChannelDenseBiasAddPattern, AverageMaxPerChannelConv2DPattern, AverageMaxPerChannelDensePattern, AddPattern, MultiplyPattern, Requantizer

import numpy as np


pytorch_model = resnet.resnet18(pretrained=True)
input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shape = (3, 3, 224, 224)
named_input_shape = [(input_name, input_shape)]
input_data = torch.randn(input_shape)
script_module = torch.jit.trace(pytorch_model, input_data)

input_shapes = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, named_input_shape)
print(mod['main'])
cc = AverageMaxCalibrationCallback()
# Conv2d bias does does not work
# Dense works

#quantizer = Quantizer(mod['main'], params, [AverageMaxPerChannelDenseBiasAddPattern(cc), AverageMaxPerChannelDensePattern(cc)])
quantizer = Quantizer(mod['main'], params, [AverageMaxPerChannelConv2DBiasAddPattern(cc), AverageMaxPerChannelConv2DPattern(cc), AverageMaxPerChannelDenseBiasAddPattern(cc), AverageMaxPerChannelDensePattern(cc), AddPattern(cc), MultiplyPattern(cc)])
random_dataset_manager = RandomDatasetManager(input_shape, 'float32', 3, 20)

calibrator = QuantizationCalibrator(quantizer, target='llvm', ctx=tvm.cpu(), dataset_manager=random_dataset_manager)
calibrated_func = calibrator.calibrate()
print(calibrated_func)
requantized_func = Requantizer().requantize(calibrated_func)
requantized_mod = tvm.ir.IRModule.from_expr(requantized_func)
print(requantized_mod)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    #q_lib = relay.build(requantized_mod, target='llvm')

from tvm.contrib import graph_runtime
input_np = np.random.randn(*input_shape).astype('float32')

gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
gmod.set_input(**params)
gmod.set_input(input_name, input_np)
gmod.run()
out = gmod.get_output(0).asnumpy()
print("Unquantized Output:")
print(out)


print(" ___________ ")
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
q_gmod.set_input(input_name, input_np)
q_gmod.set_input(**params)
q_gmod.run()
q_out = q_gmod.get_output(0).asnumpy()
print("Quantized output:")
print(q_out)
