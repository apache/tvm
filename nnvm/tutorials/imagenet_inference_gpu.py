"""
Compile GPU Inference
=====================
**Author**: `Yuwei Hu <https://huyuwei.github.io/>`_

This is an example of using NNVM to compile MobileNet/ResNet model and deploy its inference on GPU.

To begin with, we import nnvm(for compilation) and TVM(for deployment).
"""
import tvm
import numpy as np
from tvm.contrib import nvcc, graph_runtime
import nnvm.compiler
import nnvm.testing

######################################################################
# Register the NVCC Compiler Option
# ---------------------------------
# NNVM optimizes the graph and relies on TVM to generate fast GPU code.
# To get the maximum performance, we need to enable nvcc's compiler hook.
# This usually gives better performance than nvrtc mode.

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

######################################################################
# Prepare the Benchmark
# ---------------------
# We construct a standard imagenet inference benchmark.
# NNVM needs two things to compile a deep learning model:
#
# - net: the graph representation of the computation
# - params: a dictionary of str to parameters
#
# We use nnvm's testing utility to produce the model description and random parameters
# so that the example does not depend on a specific front-end framework.
#
# .. note::
#
#   In a typical workflow, we can get this pair from :any:`nnvm.frontend`
#
target = "cuda"
ctx = tvm.gpu(0)
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_classes)
# To use ResNet to do inference, run the following instead
#net, params = nnvm.testing.resnet.get_workload(
#    batch_size=1, image_shape=image_shape)
net, params = nnvm.testing.mobilenet.get_workload(
    batch_size=1, image_shape=image_shape)

######################################################################
# Compile the Graph
# -----------------
# To compile the graph, we call the build function with the graph
# configuration and parameters.
# When parameters are provided, NNVM will pre-compute certain part of the graph if possible (e.g. simplify batch normalization to scale shift),
# and return the updated parameters.

graph, lib, params = nnvm.compiler.build(
    net, target, shape={"data": data_shape}, params=params)


######################################################################
# Run the Compiled Module
# -----------------------
#
# To deploy the module, we call :any:`tvm.contrib.graph_runtime.create` passing in the graph, the lib, and context.
# Thanks to TVM, we can deploy the compiled module to many platforms and languages.
# The deployment module is designed to contain minimum dependencies.
# This example runs on the same machine.
#
# Note that the code below no longer depends on NNVM, and only relies TVM's runtime to run(deploy).
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = graph_runtime.create(graph, lib, ctx)
# set input
module.set_input(**params)
module.set_input("data", data)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape))
# convert to numpy
out.asnumpy()
