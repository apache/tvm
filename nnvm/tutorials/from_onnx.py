"""
Compiling ONNX Models with NNVM
================================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy ONNX models with NNVM.

For us to begin with, onnx module is required to be installed.

A quick solution is to install protobuf compiler, and
```bash
pip install onnx --user
```
or please refer to offical site.
https://github.com/onnx/onnx
"""
import nnvm
import tvm
import onnx
import numpy as np

######################################################################
# Load pretrained ONNX model
# ---------------------------------------------
# The example super resolution model used here is exactly the same model in onnx tutorial
# http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
# we skip the pytorch model construction part, and download the saved onnx model
import urllib2
model_url = ''.join(['https://gist.github.com/zhreshold/',
                     'bcda4716699ac97ea44f791c24310193/raw/',
                     '41b443bf2b6cf795892d98edd28bacecd8eb0d8d/',
                     'super_resolution.onnx'])
with open('super_resolution.onnx', 'w') as f:
    f.write(urllib2.urlopen(model_url).read())
# now you have super_resolution.onnx on disk
onnx_graph = onnx.load('super_resolution.onnx')
# we can load the graph as NNVM compatible model
sym, params = nnvm.frontend.from_onnx(onnx_graph)

######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples!
import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
with open('cat.jpg', 'w') as f:
    f.write(urllib2.urlopen(img_url).read())
img = Image.open('cat.jpg').convert("L")  # convert to greyscale
x = np.array(img.resize((224, 224)))[np.newaxis, np.newaxis, :, :]

######################################################################
# Compile the model on NNVM
# ---------------------------------------------
# We should be familiar with the process right now.
import nnvm.compiler
target = 'cuda'
shape_dict = {'input_0': x.shape}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

######################################################################
# Execute on TVM
# ---------------------------------------------
# The process is no different from other example
from tvm.contrib import graph_runtime
ctx = tvm.gpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('input_0', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
output_shape = (1, 1, 672, 672)
tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()
out_img = tvm_output.reshape((672, 672))

######################################################################
# Display results
# ---------------------------------------------
# We put input and output image neck to neck
from matplotlib import pyplot as plt
canvas = np.full((672, 672*2), 255)
canvas[0:224, 0:224] = x[0, 0, :, :]
canvas[:, 672:] = out_img
plt.imshow(canvas, cmap='gray')
plt.show()
