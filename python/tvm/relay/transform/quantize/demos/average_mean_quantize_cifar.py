# Demo based on code from https://www.tensorflow.org/tutorials/images/cnn
import onnx
import tensorflow as tf
import tvm
from tvm import relay
from tvm.data import DatasetManager
from tvm.relay.transform.quantize import (
    Quantizer,
    QuantizationCalibrator,
    AverageMaxCalibrationCallback,
    GlobalCalibrationCallback,
    Requantizer,
    AverageMaxPerChannelConv2DBiasAddPattern,
    AverageMaxPerChannelConv2DPattern,
    Conv2DBiasAddPattern,
    Conv2DPattern,
    DensePattern,
    AddPattern,
    MultiplyPattern,
    AverageMaxPerChannelConv2DBiasAddPattern,
    AverageMaxPerChannelConv2DPattern,
    AverageMaxPerChannelDensePattern,
)

from tensorflow.keras import datasets

import numpy as np

# tf and onnx use different versions of protobuf??
# Versions that work: pip installed protobuf version 3.12.2
# Need libprotobuf version 3.0.0 (libprotobuf is also protoc)


class NumpyDatasetManager(DatasetManager):
    # Assumes numpy_data is in form [num_inputs, c, h, w] and labels is [num_inputs]
    def __init__(self, numpy_data, numpy_labels, batch_size=1, n_batches=None):
        self.idx = 0
        self.numpy_data = numpy_data
        self.numpy_labels = numpy_labels
        assert (
            self.numpy_data.shape[0] == self.numpy_labels.shape[0]
        ), "First dimension of data and label arrays must match."
        assert (
            self.numpy_data.shape[0] >= batch_size
        ), "Batch size too large. You must provide enough data points for at least one batch."
        self.batch_sz = batch_size
        if n_batches is None:
            self.n_batches = numpy_data.shape[0] // self.batch_size
        else:
            assert n_batches * batch_size <= numpy_data.shape[0]
            self.n_batches = n_batches

    def get_next_batch(self):
        if self.is_empty():
            raise IndexError
        batched_data = self.numpy_data[self.idx * self.batch_sz : (self.idx + 1) * self.batch_sz]
        batched_label = self.numpy_labels[self.idx * self.batch_sz : (self.idx + 1) * self.batch_sz]
        self.idx += 1
        return [batched_data], batched_label

    def batch_size(self):
        return self.batch_sz

    def num_batches(self):
        return self.n_batches

    def is_empty(self):
        return self.idx >= self.n_batches

    def reset(self):
        self.idx = 0


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create dataset manager
# For "training", it seems like batch size 10 and n batches = 5000 works pretty well
batch_size = 10
train_dataset_manager = NumpyDatasetManager(
    train_images, np.ndarray.flatten(train_labels), batch_size, n_batches=5000
)
test_dataset_manager = NumpyDatasetManager(
    test_images, np.ndarray.flatten(test_labels), batch_size, n_batches=1000
)

# Load onnx model (model obtained from https://www.tensorflow.org/tutorials/images/cnn), exported to onnx
onnx_model = onnx.load(
    "/home/lorthsmith/tvm/python/tvm/relay/transform/quantize/demos/cifar-model.onnx"
)
input_dict = {"conv2d_input:0": [batch_size, 32, 32, 3]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)
print("main: ", mod["main"])
cc = AverageMaxCalibrationCallback()
quantizer = Quantizer(
    mod["main"],
    params,
    [
        AverageMaxPerChannelConv2DBiasAddPattern(cc),
        AverageMaxPerChannelConv2DPattern(cc),
        AverageMaxPerChannelDensePattern(cc),
        AddPattern(cc),
        MultiplyPattern(cc),
    ],
    skip_last=False,
)  # , AddPattern(cc), MultiplyPattern(cc)], skip_last=False)
# quantizer = Quantizer(mod['main'], params, [Conv2DBiasAddPattern(cc), Conv2DPattern(cc), DensePattern(cc), AddPattern(cc), MultiplyPattern(cc)], skip_last=True, skip_first=True)#, AddPattern(cc), MultiplyPattern(cc)], skip_last=False)

# cc = GlobalCalibrationCallback(2.0, 0)
# quantizer = Quantizer(mod['main'], params, [Conv2DBiasAddPattern(cc), Conv2DPattern(cc), DensePattern(cc), AddPattern(cc), MultiplyPattern(cc)], skip_last=False)#, AddPattern(cc), MultiplyPattern(cc)], skip_last=False)


calibrator = QuantizationCalibrator(
    quantizer,
    target="llvm",
    ctx=tvm.cpu(),
    dataset_manager=train_dataset_manager,
    show_scale_zps=True,
)
calibrated_func = calibrator.calibrate()
calibrated_mod = tvm.ir.IRModule.from_expr(calibrated_func)
print("Calibrated func: ", calibrated_func)
print("Requantizing...")
requantized_func = Requantizer().requantize(calibrated_func)
print("Requantized func: ", requantized_func)
requantized_mod = tvm.ir.IRModule.from_expr(requantized_func)

print("Calculating accuracy...")
with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, params=params, target="llvm")
    c_lib = relay.build(calibrated_mod, params=params, target="llvm")
    q_lib = relay.build(requantized_mod, params=params, target="llvm")


from tvm.contrib import graph_runtime

q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
c_gmod = graph_runtime.GraphModule(c_lib["default"](tvm.cpu()))
gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
q_correct = 0
c_correct = 0
correct = 0
total = 0

while not test_dataset_manager.is_empty():
    image_list, label = test_dataset_manager.get_next_batch()
    q_gmod.set_input(**{"conv2d_input:0": image_list[0]})
    q_gmod.run()
    q_out = q_gmod.get_output(0).asnumpy()

    c_gmod.set_input(**{"conv2d_input:0": image_list[0]})
    c_gmod.run()
    c_out = q_gmod.get_output(0).asnumpy()

    gmod.set_input(**{"conv2d_input:0": image_list[0]})
    gmod.run()
    out = gmod.get_output(0).asnumpy()

    q_predicted_labels = np.argmax(q_out, axis=1)
    c_predicted_labels = np.argmax(c_out, axis=1)
    predicted_labels = np.argmax(out, axis=1)

    print("Int8 labels: ", q_predicted_labels)
    print("Calibrated int8 labels: ", c_predicted_labels)
    print("Float32 labels: ", predicted_labels)
    print("Actual labels: ", label)
    print()

    q_correct += np.sum(q_predicted_labels == label)
    c_correct += np.sum(c_predicted_labels == label)
    correct += np.sum(predicted_labels == label)
    total += batch_size

print("Int8 percent correct: ", (q_correct / total) * 100)
print("Calibrated Int8 percent correct: ", (c_correct / total) * 100)
print("Float32 percent correct: ", (correct / total) * 100)
print("Difference: ", (((correct / total) * 100) - ((q_correct / total) * 100)))
