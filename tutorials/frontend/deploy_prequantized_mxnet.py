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
"""
Deploy a Framework-prequantized Model with TVM - Part 2 (MXNet)
==============================================
**Author**: `Animesh Jain <https://github.com/anijain2305>`_

Welcome to Part 2 of Deploy Framework-Prequantized Model with TVM. In this tutorial, we will start
with a FP32 MXNet graph, quantize it using MXNet, and then compile and execute it via TVM.

For more details on quantizing the model using MXNet, readers are encouraged to go through `Model
Quantization with Calibration Examples
<https://github.com/apache/incubator-mxnet/tree/master/example/quantization>`_.

Pre-requisites
    pip3 install mxnet-mkl --user
    pip3 install gluoncv --user
"""


###############################################################################
# Necessary imports
# -----------------
import os

import mxnet as mx
from gluoncv.model_zoo import get_model
from mxnet.contrib.quantization import *

import tvm
from tvm import relay


###############################################################################
# Helper functions
# ----------------
def download_calib_dataset(dataset_url, calib_dataset):
    """ Download calibration dataset. """
    print('Downloading calibration dataset from %s to %s' % (dataset_url, calib_dataset))
    mx.test_utils.download(dataset_url, calib_dataset)


def prepare_calib_dataset(data_shape, label_name):
    """ Preprocess the dataset and set up the data iterator. """
    mean_args = {'mean_r': 123.68, 'mean_g': 116.779, 'mean_b': 103.939}
    std_args = {'std_r': 58.393, 'std_g': 57.12, 'std_b': 57.375}
    combine_mean_std = {}
    combine_mean_std.update(mean_args)
    combine_mean_std.update(std_args)
    data = mx.io.ImageRecordIter(path_imgrec='data/val_256_q90.rec',
                                 label_width=1,
                                 preprocess_threads=60,
                                 batch_size=1,
                                 data_shape=data_shape,
                                 label_name=label_name,
                                 rand_crop=False,
                                 rand_mirror=False,
                                 shuffle=True,
                                 **combine_mean_std)
    return data


def get_mxnet_fp32_model():
    """ Read the MXNet symbol. """
    model_name = 'resnet50_v1'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    block = get_model(name=model_name, pretrained=True)

    # Convert the model to symbol format.
    block.hybridize()
    data = mx.sym.Variable('data')
    sym = block(data)
    sym = mx.sym.SoftmaxOutput(data=sym, name='softmax')
    params = block.collect_params()
    args = {}
    auxs = {}
    for param in params.values():
        v = param._reduce()
        k = param.name
        if 'running' in k:
            auxs[k] = v
        else:
            args[k] = v
    return sym, args, auxs


def quantize_model(sym, arg_params, aux_params, data, ctx, label_name):
    """ Quantize the model using MXNet. """
    return quantize_model_mkldnn(sym=sym,
                                 arg_params=arg_params,
                                 aux_params=aux_params,
                                 ctx=ctx,
                                 calib_mode='naive', calib_data=data,
                                 num_calib_examples=5,
                                 quantized_dtype='auto',
                                 label_names=(label_name,))


def run_mxnet(qsym, data, batch, ctx, label_name):
    """ Run MXNet pre-quantized model inference. """
    mod = mx.mod.Module(symbol=qsym, context=[ctx], label_names=[label_name, ])
    mod.bind(for_training=False, data_shapes=data.provide_data, label_shapes=data.provide_label)
    mod.set_params(qarg_params, qaux_params)
    mod.forward(batch, is_train=False)
    mxnet_res = mod.get_outputs()[0].asnumpy()
    mxnet_pred = np.squeeze(mxnet_res).argsort()[-5:][::-1]
    return mxnet_pred


def run_tvm(graph, lib, params, batch):
    """ Run TVM compiler model inference. """
    from tvm.contrib import graph_runtime
    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.set_input(**params)
    rt_mod.set_input('data', batch.data[0].asnumpy())
    rt_mod.run()
    tvm_res = rt_mod.get_output(0).asnumpy()
    tvm_pred = np.squeeze(tvm_res).argsort()[-5:][::-1]
    return tvm_pred, rt_mod


# Initialize variables.
label_name = 'softmax_label'
data_shape = (3, 224, 224)
ctx = mx.cpu(0)


###############################################################################
# MXNet quantization and inference.
# ---------------------------------

# Download and prepare calibrarion dataset.
download_calib_dataset('http://data.mxnet.io/data/val_256_q90.rec', 'data/val_256_q90.rec')
data = prepare_calib_dataset(data_shape, label_name)

# Get a FP32 Resnet 50 MXNet model.
sym, arg_params, aux_params = get_mxnet_fp32_model()

# Quantize the MXNet model using MXNet quantizer.
qsym, qarg_params, qaux_params = quantize_model(sym, arg_params, aux_params,
                                                data, ctx, label_name)

# Get the testing image from the MXNet data iterator.
batch = data.next()

# Run MXNet inference on the quantized model.
mxnet_pred = run_mxnet(qsym, data, batch, ctx, label_name)


###############################################################################
# TVM compilation of pre-quantized model and inference.
# ---------------------------------

# Use MXNet-Relay parser. Note that the frontend parser call is exactly same as frontend parser call
# for a FP32 model.
input_shape = [1] + list(data_shape)
input_dict = {'data': input_shape}
mod, params = relay.frontend.from_mxnet(qsym,
                                        dtype={},
                                        shape=input_dict,
                                        arg_params=qarg_params,
                                        aux_params=qaux_params)

# Please inspect the module. You will have QNN operators like requantize, quantize, conv2d.
# print(mod)

# Compile Relay module. Set the target platform. Replace the target with the your target type.
target = 'llvm -mcpu=cascadelake'
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(mod, target=target, params=params)

# Call inference on the compiled module.
tvm_pred, rt_mod = run_tvm(graph, lib, params, batch)


###############################################################################
# Accuracy comparison.
# --------------------

# Print the top-5 labels for MXNet and TVM inference. Note that final tensors can slightly differ
# between MXNet and TVM quantized inference, but the classification accuracy is not significantly
# affected. Output of the following code is as follows
#
# TVM Top-5 labels: [236 211 178 165 168]
# MXNet Top-5 labels: [236 211 178 165 168]
print("TVM Top-5 labels:", tvm_pred)
print("MXNet Top-5 labels:", mxnet_pred)


##########################################################################
# Measure performance.
# --------------------
# Here we give an example of how to measure performance of TVM compiled models.
n_repeat = 100  # should be bigger to make the measurement more accurate
ctx = tvm.cpu(0)
ftimer = rt_mod.module.time_evaluator("run", ctx, number=1, repeat=n_repeat)
prof_res = np.array(ftimer().results) * 1e3
print("Elapsed average ms:", np.mean(prof_res))

##########################################################################
# Notes
# -----
# 1) On Intel Cascadelake server, the performance is 2.01 ms.
# 2) Auto-tuning can potentially improve this performance. Please follow the tutorial at
# `Auto-tuning a convolution network for x86 CPU
# <https://tvm.apache.org/docs/tutorials/autotvm/tune_relay_x86.html>`_.
