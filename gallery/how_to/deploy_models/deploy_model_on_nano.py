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
.. _tutorial-deploy-model-on-nano:

Deploy the Pretrained Model on Jetson Nano
===========================================
**Author**: `BBuf <https://github.com/BBuf>`_

This is an example of using Relay to compile a ResNet model and deploy
it on Jetson Nano.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_requires_cuda = True
# sphinx_gallery_end_ignore
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata

######################################################################
# .. _build-tvm-runtime-on-jetson-nano:
#
# Build TVM Runtime on Jetson Nano
# --------------------------------
#
# The first step is to build the TVM runtime on the remote device.
#
# .. note::
#
#   All instructions in both this section and next section should be
#   executed on the target device, e.g. Jetson Nano. And we assume it
#   has Linux running.
#
# Since we do compilation on local machine, the remote device is only used
# for running the generated code. We only need to build tvm runtime on
# the remote device.
#
# .. code-block:: bash
#
#   git clone --recursive https://github.com/apache/tvm tvm
#   cd tvm
#   mkdir build
#   cp cmake/config.cmake build
#   cd build
#   cmake ..
#   make runtime -j4
# .. note::
#
#   If we want to use Jetson Nano's GPU for inference,
#   we need to enable the CUDA option in `config.cmake`,
#   that is, `set(USE_CUDA ON)`
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
# (Which is Jetson Nano in our example).
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9091
#
# If you see the line below, it means the RPC server started
# successfully on your device.
#
#    .. code-block:: bash
#
#      INFO:RPCServer:bind to 0.0.0.0:9091
#

######################################################################
# Prepare the Pre-trained Model
# -----------------------------
# Back to the host machine, which should have a full TVM installed (with LLVM).
#
# We will use pre-trained model from
# `MXNet Gluon model zoo <https://mxnet.apache.org/api/python/gluon/model_zoo.html>`_.
# You can found more details about this part at tutorial :ref:`tutorial-from-mxnet`.

from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import numpy as np

# one line to get the model
block = get_model("resnet18_v1", pretrained=True)

######################################################################
# In order to test our model, here we download an image of cat and
# transform its format.
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
img_path = download_testdata(img_url, img_name, module="data")
image = Image.open(img_path).resize((224, 224))


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


x = transform_image(image)

######################################################################
# synset is used to transform the label from number of ImageNet class to
# the word human can understand.
synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())

######################################################################
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.

# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
shape_dict = {"data": x.shape}
mod, params = relay.frontend.from_mxnet(block, shape_dict)
# we want a probability so add a softmax operator
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

######################################################################
# Here are some basic data workload configurations.
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

######################################################################
# Compile The Graph
# -----------------
# To compile the graph, we call the :py:func:`relay.build` function
# with the graph configuration and parameters. However, You cannot to
# deploy a x86 program on a device with ARM instruction set. It means
# Relay also needs to know the compilation option of target device,
# apart from arguments :code:`net` and :code:`params` to specify the
# deep learning workload. Actually, the option matters, different option
# will lead to very different performance.

######################################################################
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the Jetson Nano, we need to
# set it as :code:`nvidia/jetson-nano`. Set :code:`local_demo` to False
# if you want to run this tutorial with a real device.

local_demo = True

if local_demo:
    target = tvm.target.Target("llvm")
else:
    target = tvm.target.Target("nvidia/jetson-nano")
    assert target.kind.name == "cuda"
    assert target.attrs["arch"] == "sm_53"
    assert target.attrs["shared_memory_per_block"] == 49152
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["thread_warp_size"] == 32
    assert target.attrs["registers_per_block"] == 32768

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)

# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
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
    host = "192.168.1.11"
    port = 9091
    remote = rpc.connect(host, port)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("net.tar")

# create the remote runtime module
if local_demo:
    dev = remote.cpu(0)
else:
    dev = remote.cuda(0)

module = runtime.GraphModule(rlib["default"](dev))
# set input data
module.set_input("data", tvm.nd.array(x.astype("float32")))
# run
module.run()
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}".format(synset[top1]))
