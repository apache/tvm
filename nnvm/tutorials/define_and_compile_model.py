"""
Quick Start - End-to-End Tutorial for NNVM/TVM Pipeline
=======================================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_

This example shows how to build a neural network with NNVM python frontend and
generate runtime library for Nvidia GPU and Raspberry Pi with TVM. (Thanks to
Tianqi's `tutorial for cuda <http://nnvm.tvmlang.org/tutorials/get_started.html>`_ and
Ziheng's `tutorial for Raspberry Pi <http://nnvm.tvmlang.org/tutorials/deploy_model_on_rasp.html>`_)
To run this notebook, you need to install tvm and nnvm following
`these instructions <https://github.com/dmlc/nnvm/blob/master/docs/how_to/install.md>`_.
Notice that you need to build tvm with cuda and llvm.
"""

######################################################################
# Overview for Supported Hardware Backend of TVM
# -----------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/tvm_support_list.png
#      :align: center
#      :scale: 100%
#
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import NNVM and TVM.
import tvm
import nnvm.compiler
import nnvm.testing


######################################################################
# Define Neural Network in NNVM
# -----------------------------
# First, let's define a neural network with nnvm python frontend.
# For simplicity, we'll use pre-defined resnet-18 network in NNVM.
# Parameters are initialized with Xavier initializer.
# NNVM also supports other model formats such as MXNet, CoreML and ONNX.
#
# In this tutorial, we assume we will do inference on our device
# and the batch size is set to be 1. Input images are RGB color
# images of size 224 * 224. We can call the :any:`nnvm.symbol.debug_str`
# to show the network structure.

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

net, params = nnvm.testing.resnet.get_workload(batch_size=batch_size, image_shape=image_shape)
print(net.debug_str())

######################################################################
# Compilation
# ----------------------------
# Next step is to compile the model using the NNVM/TVM pipeline.
# Users can specify the optimization level of the compilation.
# Currently this value can be 0 to 2, which corresponds to
# "SimplifyInference", "OpFusion" and "PrecomputePrune" respectively.
# In this example we set optimization level to be 0
# and use Raspberry Pi as compile target.
#
# :any:`nnvm.compiler.build` returns three components: the execution graph in
# json format, the TVM module library of compiled functions specifically
# for this graph on the target hardware, and the parameter blobs of
# the model. During the compilation, NNVM does the graph-level
# optimization while TVM does the tensor-level optimization, resulting
# in an optimized runtime module for model serving.
#
# We'll first compile for Nvidia GPU. Behind the scene, `nnvm.compiler.build`
# first does a number of graph-level optimizations, e.g. pruning, fusing, etc.,
# then registers the operators (i.e. the nodes of the optmized graphs) to 
# TVM implementations to generate a `tvm.module`.
# To generate the module library, TVM will first transfer the HLO IR into the lower
# intrinsic IR of the specified target backend, which is CUDA in this example.
# Then the machine code will be generated as the module library.

opt_level = 0
target = tvm.target.cuda()
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)

######################################################################
# Save Compiled Module
# ----------------------------
# After compilation, we can save the graph, lib and params into separate files
# and deploy them to Nvidia GPU.

from tvm.contrib import util

temp = util.tempdir()
path_lib = temp.relpath("deploy_lib.so")
lib.export_library(path_lib)
with open(temp.relpath("deploy_graph.json"), "w") as fo:
    fo.write(graph.json())
with open(temp.relpath("deploy_param.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print(temp.listdir())

######################################################################
# Deploy locally to Nvidia GPU
# ------------------------------
# Now we can load the module back.

import numpy as np
from tvm.contrib import graph_runtime

loaded_lib = tvm.module.load(path_lib)
loaded_json = open(temp.relpath("deploy_graph.json")).read()
loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
module = graph_runtime.create(loaded_json, loaded_lib, tvm.gpu(0))
module.load_params(loaded_params)

input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
module.run(data=input_data)
out = module.get_output(0, out=tvm.nd.empty(out_shape))
# Print first 10 elements of output
print(out.asnumpy()[0][0:10])

######################################################################
# Compile and Deploy the Model to Raspberry Pi Remotely with RPC
# ------------------------------
# Following the steps above, we can also compile the model for Raspberry Pi.
# TVM provides rpc module to help with remote deploying.
#
# For demonstration, we simply start an RPC server on the same machine,
# if :code:`use_rasp` is False. If you have set up the remote
# environment, please change the three lines below: change the
# :code:`use_rasp` to True, also change the host and port with your
# device's host address and port number.

# If we run the example locally for demonstration, we can simply set the 
# compilation target as `llvm`. 
# To run it on the Raspberry Pi, you need to specify its instruction set. 
# `llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon`
# is the recommended compilation configuration, thanks to Ziheng's work.

from tvm.contrib import rpc

use_rasp = False
host = 'rasp0'
port = 9090

if not use_rasp:
    # run server locally
    host = 'localhost'
    port = 9090
    server = rpc.Server(host=host, port=port, use_popen=True)

# compile and save model library
if use_rasp:
    target = "llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"
else:
    target = "llvm"
# use `with tvm.target.rasp` for some target-specified optimization
with tvm.target.rasp():
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)

temp = util.tempdir()
path_lib = temp.relpath("deploy_lib_rasp.o")
lib.save(path_lib)

# connect the server
remote = rpc.connect(host, port)

# upload the library to remote device and load it
remote.upload(path_lib)
rlib = remote.load_module('deploy_lib_rasp.o')

ctx = remote.cpu(0)
# upload the parameter
rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

# create the remote runtime module
module = graph_runtime.create(graph, rlib, ctx)
# set parameter
module.set_input(**rparams)
# set input data
input_data = np.random.uniform(size=data_shape)
module.set_input('data', tvm.nd.array(input_data.astype('float32')))
# run
module.run()

out = module.get_output(0, out=tvm.nd.empty(out_shape, ctx=ctx))
# Print first 10 elements of output
print(out.asnumpy()[0][0:10])

if not use_rasp:
    # terminate the local server
    server.terminate()

