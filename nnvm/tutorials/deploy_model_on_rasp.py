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
.. _tutorial-deploy-model-on-rasp:

Deploy the Pretrained Model on Raspberry Pi
===========================================
**Author**: `Ziheng Jiang <https://ziheng.org/>`_

This is an example of using NNVM to compile a ResNet model and deploy
it on Raspberry Pi.
"""

import tvm
import nnvm.compiler
import nnvm.testing
from tvm import rpc
from tvm.contrib import util, graph_runtime as runtime
from tvm.contrib.download import download_testdata

######################################################################
# .. _build-tvm-runtime-on-device:
#
# Build TVM Runtime on Device
# ---------------------------
#
# The first step is to build tvm runtime on the remote device.
#
# .. note::
#
#   All instructions in both this section and next section should be
#   executed on the target device, e.g. Raspberry Pi. And we assume it
#   has Linux running.
#
# Since we do compilation on local machine, the remote device is only used
# for running the generated code. We only need to build tvm runtime on
# the remote device.
#
# .. code-block:: bash
#
#   git clone --recursive https://github.com/dmlc/tvm
#   cd tvm
#   make runtime -j4
#
# After building runtime successfully, we need to set environment varibles
# in :code:`~/.bashrc` file. We can edit :code:`~/.bashrc`
# using :code:`vi ~/.bashrc` and add the line below (Assuming your TVM
# directory is in :code:`~/tvm`):
#
# .. code-block:: bash
#
#   export PYTHONPATH=$PYTHONPATH:~/tvm/python
#
# To update the environment variables, execute :code:`source ~/.bashrc`.

######################################################################
# Set Up RPC Server on Device
# ---------------------------
# To start an RPC server, run the following command on your remote device
# (Which is Raspberry Pi in our example).
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
#
# If you see the line below, it means the RPC server started
# successfully on your device.
#
#    .. code-block:: bash
#
#      INFO:root:RPCServer: bind to 0.0.0.0:9090
#

######################################################################
# Prepare the Pre-trained Model
# -----------------------------
# Back to the host machine, which should have a full TVM installed (with LLVM).
#
# We will use pre-trained model from
# `MXNet Gluon model zoo <https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_.
# You can found more details about this part at tutorial :ref:`tutorial-from-mxnet`.

from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import numpy as np

# one line to get the model
block = get_model('resnet18_v1', pretrained=True)

######################################################################
# In order to test our model, here we download an image of cat and
# transform its format.
img_name = 'cat.png'
img_path = download_testdata('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true',
                             img_name, module='data')
image = Image.open(img_path).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)

######################################################################
# synset is used to transform the label from number of ImageNet class to
# the word human can understand.
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())

######################################################################
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.

# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
net, params = nnvm.frontend.from_mxnet(block)
# we want a probability so add a softmax operator
net = nnvm.sym.softmax(net)

######################################################################
# Here are some basic data workload configurations.
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

######################################################################
# Compile The Graph
# -----------------
# To compile the graph, we call the :any:`nnvm.compiler.build` function
# with the graph configuration and parameters. However, You cannot to
# deploy a x86 program on a device with ARM instruction set. It means
# NNVM also needs to know the compilation option of target device,
# apart from arguments :code:`net` and :code:`params` to specify the
# deep learning workload. Actually, the option matters, different option
# will lead to very different performance.

######################################################################
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the Raspberry Pi, we need to
# specify its instruction set. Set :code:`local_demo` to False if you want
# to run this tutorial with a real device.

local_demo = True

if local_demo:
    target = tvm.target.create('llvm')
else:
    target = tvm.target.arm_cpu('rasp3b')
    # The above line is a simple form of
    # target = tvm.target.create('llvm -device=arm_cpu -model=bcm2837 -target=armv7l-linux-gnueabihf -mattr=+neon')

with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)

# After `nnvm.compiler.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = util.tempdir()
lib_fname = tmp.relpath('net.tar')
lib.export_library(lib_fname)

######################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# With RPC, you can deploy the model remotely from your host machine
# to the remote device.

# obtain an RPC session from remote device.
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
module.run()
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.asnumpy())
print('TVM prediction top-1: {}'.format(synset[top1]))
