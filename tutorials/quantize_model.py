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
.. _tutorial-quantize-model:

Speed Up Inference and Compress Model with Quantization
===========================================
**Author**: `Ziheng Jiang <https://ziheng.org/>`_

This is an example to speed up and compress
a ResNet model with quantization.
"""

import tvm
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import util, graph_runtime as runtime
from tvm.contrib.download import download_testdata

from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import numpy as np
# get model

# one line to get the model
block = get_model('resnet18_v1', pretrained=True)

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
img_path = download_testdata(img_url, img_name, module='data')
image = Image.open(img_path).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)


shape_dict = {'data': x.shape}
mod, params = relay.frontend.from_mxnet(block, shape_dict)


local_demo = True

target = tvm.target.create('llvm')

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

def evaluate_inference_speed(graph, lib, params):
    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    lib.export_library(lib_fname)

    if local_demo:
        remote = rpc.LocalSession()
    else:
        # The following is my environment, change this to the IP address of your target device
        host = '10.77.1.162'
        port = 9090
        remote = rpc.connect(host, port)

    # upload the library to remote device and load it
    remote.upload(lib_fname)
    rlib = remote.load_module('net.tar')

    # create the remote runtime module
    ctx = remote.cpu(0)
    module = runtime.create(graph, rlib, ctx)
    # set parameter (upload params to the remote device. This may take a while)
    module.set_input(**params)
    # set input data
    module.set_input('data', tvm.nd.array(x.astype('float32')))
    # run

    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))


evaluate_inference_speed(graph, lib, params)

import tvm.relay.quantize as qtz

qconfig_i8_i32 = qtz.qconfig(skip_conv_layers=[0],
                             nbit_input=8,
                             nbit_weight=8,
                             global_scale=4.0,
                             dtype_input="int8",
                             dtype_weight="int8",
                             dtype_activation="int32",
                             do_simulation=False)

# explain configures


with qconfig_i8_i32:
    mod = qtz.quantize(mod, params)


# compare origin size and quantized size

# compare origin speed and quantized speed

def profile_speed_and_size():
    pass

# compare origin speed and i16 speed

qconfig_i8_i16


# How do we get those model

# search configure on Machine
