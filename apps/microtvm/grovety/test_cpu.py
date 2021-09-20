
import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime
import test_models.synth as synth
import test_models.cifar10_2 as cifar10
import numpy as np


# relay_mod, params, input, output = synth.avgpool_1d_relay_mod()
# dataset = synth.get_data()

relay_mod, params, input, output = cifar10.open_model('llvm')
dataset = cifar10.get_data()

print(relay_mod)
# exit(0)

# desired_layouts = {'nn.conv2d': ['NHWC', 'HWIO']}
# seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])
# with tvm.transform.PassContext(opt_level=3):
#     relay_mod = seq(relay_mod)

with tvm.transform.PassContext(opt_level=3, config={}):
    graph, lib, params = relay.build(relay_mod, 'llvm', params=params)

_, data = dataset[0]
print(data.shape)

m = graph_runtime.create(graph, lib, tvm.cpu(0))
m.set_input('data', [data])
m.set_input(**params)

m.run()

tvm_output = m.get_output(0).asnumpy()
print(tvm_output.shape)

print(np.reshape(tvm_output, -1))
