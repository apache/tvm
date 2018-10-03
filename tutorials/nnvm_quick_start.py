"""
.. _tutorial-nnvm-quick-start:

Quick Start Tutorial for Compiling Deep Learning Models
=======================================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_

This example shows how to build a neural network with NNVM python frontend and
generate runtime library for Nvidia GPU with TVM.
Notice that you need to build TVM with cuda and llvm enabled.
"""

######################################################################
# Overview for Supported Hardware Backend of TVM
# ----------------------------------------------
# The image below shows hardware backend currently supported by TVM:
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/tvm_support_list.png
#      :align: center
#      :scale: 100%
#
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import NNVM and TVM.

import numpy as np

import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime

######################################################################
# Define Neural Network in NNVM
# -----------------------------
# First, let's define a neural network with nnvm python frontend.
# For simplicity, we'll use pre-defined resnet-18 network in NNVM.
# Parameters are initialized with Xavier initializer.
# NNVM also supports other model formats such as MXNet, CoreML, ONNX and 
# Tensorflow.
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

net, params = nnvm.testing.resnet.get_workload(
    layers=18, batch_size=batch_size, image_shape=image_shape)
print(net.debug_str())

######################################################################
# Compilation
# -----------
# Next step is to compile the model using the NNVM/TVM pipeline.
# Users can specify the optimization level of the compilation.
# Currently this value can be 0 to 3. The optimization passes include
# operator fusion, pre-computation, layout transformation and so on.
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
# then registers the operators (i.e. the nodes of the optimized graphs) to
# TVM implementations to generate a `tvm.module`.
# To generate the module library, TVM will first transfer the High level IR
# into the lower intrinsic IR of the specified target backend, which is CUDA
# in this example. Then the machine code will be generated as the module library.

opt_level = 3
target = tvm.target.cuda()
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)

#####################################################################
# Run the generate library
# ------------------------
# Now we can create graph runtime and run the module on Nvidia GPU.

# create random input
ctx = tvm.gpu()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_runtime.create(graph, lib, ctx)
# set input and parameters
module.set_input("data", data)
module.set_input(**params)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape))
# convert to numpy
out.asnumpy()

# Print first 10 elements of output
print(out.asnumpy().flatten()[0:10])

######################################################################
# Save and Load Compiled Module
# -----------------------------
# We can also save the graph, lib and parameters into files and load them
# back in deploy environment.

####################################################

# save the graph, lib and params into separate files
from tvm.contrib import util

temp = util.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
with open(temp.relpath("deploy_graph.json"), "w") as fo:
    fo.write(graph.json())
with open(temp.relpath("deploy_param.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
print(temp.listdir())

####################################################

# load the module back.
loaded_json = open(temp.relpath("deploy_graph.json")).read()
loaded_lib = tvm.module.load(path_lib)
loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.run(data=input_data)
out = module.get_output(0).asnumpy()
