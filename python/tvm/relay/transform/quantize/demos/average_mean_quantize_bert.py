# Demo based on code from https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.data import DatasetManager
from tvm.relay.transform.quantize import Quantizer, GlobalCalibrator, Requantizer

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import onnx
import numpy as np

class HardcodedBertInputs(DatasetManager):
    # Assumes numpy_data is in form [num_inputs, c, h, w] and labels is [num_inputs]
    def __init__(self, n_batches=100):
        self.idx = 0
        self.num_batches = 100
    
    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        
        unique_ids_raw_output = np.random.randn([1])
        segment_ids =  np.random.randn([1, 256])
        input_mask = np.random.randn([1, 256])
        input_ids = np.random.randn([1, 256]) 
        self.idx += 1
        return [unique_ids_raw_output, segment_ids, input_mask, input_ids], None

    def num_batches(self):
        return self.num_batches

    def is_empty(self):
        return self.idx >= self.num_batches

    def reset(self):
        self.idx = 0

batch_size = 1
onnx_model = onnx.load('/home/lorthsmith/tvm/python/tvm/relay/new_quantize/demos/bertsquad-10.onnx')
input_dict = {'unique_ids_raw_output___9:0': [1], 'segment_ids:0': [1, 256], 'input_mask:0': [1, 256], 'input_ids:0': [1, 256]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)

quantized_mod, calibration_map = Quantizer().quantize(mod, params, skip_layers=[0])

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    #lib = relay.build(mod, target='llvm')
    q_lib = relay.build(quantized_mod, target='llvm')

# Calibrate
global_calibrator = GlobalCalibrator(0.05, 0)
calibrated_mod = global_calibrator.calibrate(quantized_mod, calibration_map)
print("Calibrated mod: \n", calibrated_mod.astext(False))

# Requantize
requantized_mod = Requantizer().requantize(calibrated_mod)
print("Requantized mod: \n", requantized_mod.astext(False))

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    #lib = relay.build(mod, params=params, target='llvm')
    q_lib = relay.build(requantized_mod, target='llvm')

from tvm.contrib import graph_runtime
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
q_gmod.set_input(**{'unique_ids_raw_output___9:0': np.random.randn(*[1]), 'segment_ids:0': np.random.randn(*[1, 256]), 'input_mask:0': np.random.randn(*[1, 256]), 'input_ids:0': np.random.randn(*[1, 256])})
q_gmod.run()
q_out = q_gmod.get_output(0).asnumpy()
print(q_out)
exit()

from tvm.contrib import graph_runtime
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
q_correct = 0
correct = 0
total = 0

while not test_dataset_manager.is_empty():
    image_list, label = test_dataset_manager.get_next_batch()
    q_gmod.set_input(**{'conv2d_input:0': image_list[0]})
    q_gmod.run()
    q_out = q_gmod.get_output(0).asnumpy()

    gmod.set_input(**{'conv2d_input:0': image_list[0]})
    gmod.run()
    out = gmod.get_output(0).asnumpy()

    q_predicted_labels = np.argmax(q_out, axis=1)
    predicted_labels = np.argmax(out, axis=1)

    #print("Int8 labels: ", q_predicted_labels)
    #print("Float32 labels: ", predicted_labels)
    #print("Actual labels: ", label)

    q_correct += np.sum(q_predicted_labels == label)
    correct += np.sum(predicted_labels == label)
    total += batch_size

print("Int8 percent correct: ", (q_correct / total) * 100)
print("Float32 percent correct: ", (correct / total) * 100)
print("Difference: ", (((correct / total) * 100) - ((q_correct / total) * 100)))
