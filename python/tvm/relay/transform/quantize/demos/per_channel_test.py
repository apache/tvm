# Demo based on code from https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf
import tvm
from tvm import relay
from tvm.relay.transform.quantize import Quantizer, AverageMeanCalibrator, DatasetManager, Requantizer, Calibrator

from tensorflow.keras import datasets, layers, models
import onnx
import numpy as np

class NumpyDatasetManager(DatasetManager):
    # Assumes numpy_data is in form [num_inputs, c, h, w] and labels is [num_inputs]
    def __init__(self, numpy_data, numpy_labels, batch_size=1, n_batches=None):
        self.idx = 0
        self.numpy_data = numpy_data
        self.numpy_labels = numpy_labels
        assert self.numpy_data.shape[0] == self.numpy_labels.shape[0], "First dimension of data and label arrays must match."
        assert self.numpy_data.shape[0] >= batch_size, "Batch size too large. You must provide enough data points for at least one batch."
        self.batch_size = batch_size
        if n_batches is None:
            self.n_batches = numpy_data.shape[0] // self.batch_size
        else:
            assert n_batches * batch_size <= numpy_data.shape[0]
            self.n_batches = n_batches

    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        batched_data = self.numpy_data[self.idx * self.batch_size : (self.idx + 1) * self.batch_size]
        batched_label = self.numpy_labels[self.idx * self.batch_size : (self.idx + 1) * self.batch_size]
        self.idx += 1
        return [batched_data], batched_label

    def num_batches(self):
        return self.n_batches

    def is_empty(self):
        return self.idx >= self.n_batches

    def reset(self):
        self.idx = 0

class PerChannelTestCalibrator(Calibrator):

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def _calibration_callback(self, variable_pairs):
        value_dict = {}
        op = self._get_layer_op()
        attrs = self._get_layer_attributes()
        # How will dequantize work? I don't know
        
        if (op == relay.op.get('qnn.dense')):
            units = attrs['units']
            scales = np.random.randn(units).astype('float32')
            ((data_scale, data_zp), (weight_scale, weight_zp)) = variable_pairs
            value_dict[data_scale.name_hint] = np.array(2.0).astype('float32')
            value_dict[data_zp.name_hint] = np.array(0).astype('int32')
            value_dict[weight_zp.name_hint] = np.array(0).astype('int32')
            #value_dict[weight_scale.name_hint] = np.array(2.0).astype('float32')
            value_dict[weight_scale.name_hint] = scales
        
        elif op == relay.op.get('qnn.conv2d'):
            channels = attrs['channels']
            scales = np.random.randn(channels).astype('float32')
            ((data_scale, data_zp), (weight_scale, weight_zp)) = variable_pairs
            value_dict[data_scale.name_hint] = np.array(2.0).astype('float32')
            value_dict[data_zp.name_hint] = np.array(0).astype('int32')
            value_dict[weight_zp.name_hint] = np.array(0).astype('int32')
            #value_dict[weight_scale.name_hint] = np.array(2.0).astype('float32')
            value_dict[weight_scale.name_hint] = scales
            print(scales.shape)
        else:
            for (scale_var, zp_var) in variable_pairs:
                value_dict[scale_var.name_hint] = np.array(2.0).astype('float32')
                value_dict[zp_var.name_hint] = np.array(0).astype('int32')
        return value_dict

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create dataset manager
batch_size = 1
train_dataset_manager = NumpyDatasetManager(train_images, np.ndarray.flatten(train_labels), batch_size, n_batches=100)
test_dataset_manager = NumpyDatasetManager(test_images, np.ndarray.flatten(test_labels), batch_size, n_batches=100)

# Load onnx model (model obtained from https://www.tensorflow.org/tutorials/images/cnn), exported to onnx
onnx_model = onnx.load('/home/lorthsmith/tvm/python/tvm/relay/new_quantize/demos/cifar-model.onnx')
input_dict = {'conv2d_input:0': [batch_size, 32, 32, 3]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)

# Quantize
quantized_mod, calibration_map = Quantizer().quantize(mod, params, skip_layers=[0])
#print("Quantized mod: \n", quantized_mod.astext(False))

# Calibrate
average_mean_calibrator = PerChannelTestCalibrator([batch_size, 32, 32, 3])
calibrated_mod = average_mean_calibrator.calibrate(quantized_mod, calibration_map)
#print("Calibrated mod: \n", calibrated_mod.astext(False))

# Requantize
requantized_mod = Requantizer().requantize(calibrated_mod)
#print("Requantized mod: \n", requantized_mod.astext(False))

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(requantized_mod, target='llvm')

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
