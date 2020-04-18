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
===============================================================
**Author**: `Animesh Jain <https://github.com/anijain2305>`_

Welcome to part 2 of the Deploy Framework-Prequantized Model with TVM tutorial.
In this part, we will start with a FP32 MXNet graph, quantize it using
MXNet, and then compile and execute it via TVM.

For more details on quantizing the model using MXNet, readers are encouraged to
go through `Model Quantization with Calibration Examples
<https://github.com/apache/incubator-mxnet/tree/master/example/quantization>`_.

To get started, we need mxnet-mkl and gluoncv package. They can be installed as follows.

.. code-block:: bash

    pip3 install mxnet-mkl --user
    pip3 install gluoncv --user
"""


###############################################################################
# Necessary imports
# -----------------
import os

import mxnet as mx
from gluoncv.model_zoo import get_model
from mxnet.contrib.quantization import quantize_model_mkldnn

import numpy as np

import tvm
from tvm import relay


###############################################################################
# Helper functions
# ----------------

###############################################################################
# We need to download the calibration dataset. This dataset is used to find minimum and maximum
# values of intermediate tensors while post-training MXNet quantization. MXNet quantizer, using
# these min/max values, finds outs a suitable scale for the quantized tensors.
def download_calib_dataset(dataset_url, calib_dataset_fname):
    print('Downloading calibration dataset from %s to %s' % \
            (dataset_url, calib_dataset_fname))
    mx.test_utils.download(dataset_url, calib_dataset_fname)


###############################################################################
# Lets preprare the calibration dataset by pre-processing. In this tutorial, we follow the
# pre-processing used the MXNet quantization `tutorial
# <https://github.com/apache/incubator-mxnet/tree/master/example/quantization>`_. Please replace it
# with your pre-processing if needed.
def prepare_calib_dataset(data_shape, label_name, calib_dataset_fname):
    """ Preprocess the dataset and set up the data iterator. """
    mean_args = {'mean_r': 123.68, 'mean_g': 116.779, 'mean_b': 103.939}
    std_args = {'std_r': 58.393, 'std_g': 57.12, 'std_b': 57.375}
    combine_mean_std = {}
    combine_mean_std.update(mean_args)
    combine_mean_std.update(std_args)
    data = mx.io.ImageRecordIter(path_imgrec=calib_dataset_fname,
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


###############################################################################
# The following function reads the FP32 MXNet model. In this example, we use resnet50-v1 model. The
# readers are encouraged to go through MXNet quantization tutorial to get more models. We convert
# the MXNet model to its symbol format.
def get_mxnet_fp32_model():
    """ Read the MXNet symbol. """
    model_name = 'resnet50_v1'
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


###############################################################################
# Lets now quantize the model using MXNet. MXNet works in concert with MKLDNN to quantize the
# model. Note that MKLDNN is used only for quantizing. Once we get a quantized model, we can compile
# and execute it on any supported hardware platform in TVM.
def quantize_model(sym, arg_params, aux_params, data, ctx, label_name):
    return quantize_model_mkldnn(sym=sym,
                                 arg_params=arg_params,
                                 aux_params=aux_params,
                                 ctx=ctx,
                                 calib_mode='naive', calib_data=data,
                                 num_calib_examples=5,
                                 quantized_dtype='auto',
                                 label_names=(label_name,))


###############################################################################
# Lets run MXNet pre-quantized model inference and get the MXNet prediction.
def run_mxnet(qsym, data, batch, ctx, label_name):
    mod = mx.mod.Module(symbol=qsym, context=[ctx], label_names=[label_name, ])
    mod.bind(for_training=False, data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(qarg_params, qaux_params)
    mod.forward(batch, is_train=False)
    mxnet_res = mod.get_outputs()[0].asnumpy()
    mxnet_pred = np.squeeze(mxnet_res).argsort()[-5:][::-1]
    return mxnet_pred


###############################################################################
# Lets run TVM compiled pre-quantized model inference and get the TVM prediction.
def run_tvm(graph, lib, params, batch):
    from tvm.contrib import graph_runtime
    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.set_input(**params)
    rt_mod.set_input('data', batch.data[0].asnumpy())
    rt_mod.run()
    tvm_res = rt_mod.get_output(0).asnumpy()
    tvm_pred = np.squeeze(tvm_res).argsort()[-5:][::-1]
    return tvm_pred, rt_mod


###############################################################################
# Initialize variables.
label_name = 'softmax_label'
data_shape = (3, 224, 224)
ctx = mx.cpu(0)

###############################################################################
# MXNet quantization and inference
# --------------------------------

###############################################################################
# Download and prepare calibrarion dataset.
calib_dataset_fname = '/tmp/val_256_q90.rec'
download_calib_dataset(dataset_url='http://data.mxnet.io/data/val_256_q90.rec',
                       calib_dataset_fname=calib_dataset_fname )
data = prepare_calib_dataset(data_shape, label_name, calib_dataset_fname)

###############################################################################
# Get a FP32 Resnet 50 MXNet model.
sym, arg_params, aux_params = get_mxnet_fp32_model()

###############################################################################
# Quantize the MXNet model using MXNet quantizer.
qsym, qarg_params, qaux_params = quantize_model(sym, arg_params, aux_params,
                                                data, ctx, label_name)

###############################################################################
# Get the testing image from the MXNet data iterator.
batch = data.next()

###############################################################################
# Run MXNet inference on the quantized model.
mxnet_pred = run_mxnet(qsym, data, batch, ctx, label_name)

###############################################################################
# TVM compilation and inference
# ----------------------------------------------------

###############################################################################
# We use the MXNet-Relay parser to conver the MXNet pre-quantized graph into Relay IR. Note that the
# frontend parser call for a pre-quantized model is exactly same as frontend parser call for a FP32
# model. We encourage you to remove the comment from print(mod) and inspect the Relay module. You
# will see many QNN operators, like, Requantize, Quantize and QNN Conv2D.
input_shape = [1] + list(data_shape)
input_dict = {'data': input_shape}
mod, params = relay.frontend.from_mxnet(qsym,
                                        dtype={},
                                        shape=input_dict,
                                        arg_params=qarg_params,
                                        aux_params=qaux_params)
# print(mod)

###############################################################################
# Lets now the compile the Relay module. We use the "llvm" target here. Please replace it with the
# target platform that you are interested in.
target = 'llvm'
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(mod, target=target,
                                                  params=params)

###############################################################################
# Finally, lets call inference on the TVM compiled module.
tvm_pred, rt_mod = run_tvm(graph, lib, params, batch)

###############################################################################
# Accuracy comparison
# -------------------

###############################################################################
# Print the top-5 labels for MXNet and TVM inference. Note that final tensors can slightly differ
# between MXNet and TVM quantized inference, but the classification accuracy is not significantly
# affected.
print("TVM Top-5 labels:", tvm_pred)
print("MXNet Top-5 labels:", mxnet_pred)


##########################################################################
# Measure performance
# -------------------
# Here we give an example of how to measure performance of TVM compiled models.
n_repeat = 100  # should be bigger to make the measurement more accurate
ctx = tvm.cpu(0)
ftimer = rt_mod.module.time_evaluator("run", ctx, number=1, repeat=n_repeat)
prof_res = np.array(ftimer().results) * 1e3
print("Elapsed average ms:", np.mean(prof_res))

######################################################################
# .. note::
#
#   Unless the hardware has special support for fast 8 bit instructions, quantized models are
#   not expected to be any faster than FP32 models. Without fast 8 bit instructions, TVM does
#   quantized convolution in 16 bit, even if the model itself is 8 bit.
#
#   For x86, the best performance can be achieved on CPUs with AVX512 instructions set.
#   In this case, TVM utilizes the fastest available 8 bit instructions for the given target.
#   This includes support for the VNNI 8 bit dot product instruction (CascadeLake or newer).
#   For EC2 C5.12x large instance, TVM latency for this tutorial is ~2 ms.
#
#   Moreover, the following general tips for CPU performance equally applies:
#
#    * Set the environment variable TVM_NUM_THREADS to the number of physical cores
#    * Choose the best target for your hardware, such as "llvm -mcpu=skylake-avx512" or
#      "llvm -mcpu=cascadelake" (more CPUs with AVX512 would come in the future)
#    * Perform autotuning - `Auto-tuning a convolution network for x86 CPU
#      <https://tvm.apache.org/docs/tutorials/autotvm/tune_relay_x86.html>`_.
