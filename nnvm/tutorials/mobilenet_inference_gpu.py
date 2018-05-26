"""
Compile MobileNet Inference on GPU
==================================
**Author**: `Yuwei Hu <https://huyuwei.github.io/>`_

This is an example of using NNVM to compile MobileNet model and deploy its inference on GPU.

To begin with, we import nnvm(for compilation) and TVM(for deployment).
"""
import tvm
import nnvm.compiler
import nnvm.runtime
import nnvm.testing
from tvm.contrib import nvcc

######################################################################
# Register the NVCC Compiler Option
# ---------------------------------
# NNVM optimizes the graph and relies on TVM to generate fast
# GPU code, to get the maximum performance, we need to enable
# nvcc's compiler hook. This gives better performance than nvrtc mode.

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", options=["-arch=sm_52"])
    return ptx

######################################################################
# Prepare the Benchmark
# ---------------------
# We construct a standard imagenet inference benchmark.
# We use nnvm's testing utility to produce the model description and random parameters that so the example does not
# depend on a specific front-end framework.
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
net, params = nnvm.testing.mobilenet.get_workload(
    batch_size=1, image_shape=image_shape)

######################################################################
# Compile The Graph
# -----------------
# NNVM needs two things to compile a deep learning model:
#
# - net which is the graph representation of the computation
# - params a dictionary of str to parameters.
#
# To compile the graph, we call the build function with the graph
# configuration and parameters.
# When parameters are provided, NNVM will pre-compute certain part of the graph if possible, 
# the new parameter set returned as the third return value.

graph, lib, params = nnvm.compiler.build(
    net, target, shape={"data": data_shape}, params=params)

######################################################################
# Run the Compiled Module
# -----------------------
#
# To deploy the module, we call :any:`nnvm.runtime.create` passing in the graph the lib and context.
# Thanks to TVM, we can deploy the compiled module to many platforms and languages.
# The deployment module is designed to contain minimum dependencies.
# This example runs on the same machine.

module = nnvm.runtime.create(graph, lib, ctx)
# set input
module.set_input(**params)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape))
# Convert to numpy
out.asnumpy()
