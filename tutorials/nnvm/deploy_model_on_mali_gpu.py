"""
.. _tutorial-deploy-model-on-mali-gpu:

Deploy the Pretrained Model on ARM Mali GPU
===========================================
**Author**: `Lianmin Zheng <https://lmzheng.net/>`_, `Ziheng Jiang <https://ziheng.org/>`_

This is an example of using NNVM to compile a ResNet model and
deploy it on Firefly-RK3399 with ARM Mali GPU. We will use the
Mali-T860 MP4 GPU on this board to accelerate the inference.
"""

import tvm
import nnvm.compiler
import nnvm.testing
from tvm import rpc
from tvm.contrib import util, graph_runtime as runtime

######################################################################
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
# Since we do compilaton on local machine, the remote device is only used
# for runing the generated code. We only need to build tvm runtime on
# the remote device. Make sure you have opencl driver in your board.
# You can refer to `tutorial <https://gist.github.com/mli/585aed2cec0b5178b1a510f9f236afa2>`_
# to setup OS and opencl driver for rk3399.
#
# .. code-block:: bash
#
#   git clone --recursive https://github.com/dmlc/tvm
#   cd tvm
#   cp cmake/config.cmake .
#   sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake 
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
# In our webpage building server (the machine that built this tutorial webpage),
# we do not have access to RK3399 board.
#
# So for local demonstration, we simply start a "fake" RPC server on the same machine.
#
# .. note::
#
#  If you have real remote device, you should change :code:`local_demo` to False, and 
#  set the host and port correctly.

local_demo = True

if local_demo:
    # start a "fake" RPC server locally
    host = 'localhost'
    port = 9091
    server = rpc.Server(host=host, port=port, use_popen=True)
else:
    # The following is my environment, change this to your target device IP
    host = '10.77.1.145'
    port = 9090

######################################################################
# Prepare the Pretrained Model
# ----------------------------
# Back to the host machine, which should have a full TVM installed (with LLVM).
# 
# We will use pre-trained model from
# `MXNet Gluon model zoo <https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_.
# You can found more details about this part at tutorial :ref:`tutorial-from-mxnet`

from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
import numpy as np

# only one line to get the model
block = get_model('resnet18_v1', pretrained=True)

######################################################################
# In order to test our model, here we download an image of cat and
# transform its format.
img_name = 'cat.jpg'
download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
image = Image.open(img_name).resize((224, 224))

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

synset_name = 'synset.txt'
download(synset_url, synset_name)
with open(synset_name) as f:
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
out_shape = (batch_size, num_classes)

######################################################################
# Compile The Graph
# -----------------
# To compile the graph, we call the :any:`nnvm.compiler.build` function
# with the graph configuration and parameters. As we use OpenCL for
# GPU computing, the tvm will generate both OpenCL kernel code and ARM
# CPU host code. The CPU host code is used for calling OpenCL kernels.
# In order to generate correct CPU code, we need to specify the target
# triplet for host ARM device by setting the parameter :code:`target_host`.

######################################################################
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the RK3399, we need to
# specify its instruction set. 

if local_demo:
    target_host = "llvm"
    target = "llvm"
else:
    # Here is the setting for my rk3399 board
    # If you don't use rk3399, you can query your target triple by 
    # execute `gcc -v` on your board.
    target_host = "llvm -target=aarch64-linux-gnu"

    # set target as  `tvm.target.mali` instead of 'opencl' to enable
    # optimization for mali
    target = tvm.target.mali()

with nnvm.compiler.build_config(opt_level=2):
    graph, lib, params = nnvm.compiler.build(net, target=target,
            shape={"data": data_shape}, params=params, target_host=target_host)

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

# connect the server
remote = rpc.connect(host, port)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module('net.tar')

ctx = remote.cpu(0) if local_demo else remote.cl(0)
# upload the parameter
rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

# create the remote runtime module
module = runtime.create(graph, rlib, ctx)
# set parameter
module.set_input(**rparams)
# set input data
module.set_input('data', tvm.nd.array(x.astype('float32')))
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx))
# get top1 result
top1 = np.argmax(out.asnumpy())
print('TVM prediction top-1: {}'.format(synset[top1]))

if local_demo:
    # terminate the local server
    server.terminate()

