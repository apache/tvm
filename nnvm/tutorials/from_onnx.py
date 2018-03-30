"""
Compile ONNX Models
===================
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

def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)

######################################################################
# Load pretrained ONNX model
# ---------------------------------------------
# The example super resolution model used here is exactly the same model in onnx tutorial
# http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
# we skip the pytorch model construction part, and download the saved onnx model
model_url = ''.join(['https://gist.github.com/zhreshold/',
                     'bcda4716699ac97ea44f791c24310193/raw/',
                     '93672b029103648953c4e5ad3ac3aadf346a4cdc/',
                     'super_resolution_0.2.onnx'])
download(model_url, 'super_resolution.onnx', True)
# now you have super_resolution.onnx on disk
onnx_model = onnx.load('super_resolution.onnx')
# we can load the graph as NNVM compatible model
sym, params = nnvm.frontend.from_onnx(onnx_model)

######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples!
from PIL import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
download(img_url, 'cat.png')
img = Image.open('cat.png').resize((224, 224))
img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

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

######################################################################
# Display results
# ---------------------------------------------
# We put input and output image neck to neck
from matplotlib import pyplot as plt
out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode='L')
out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
result = Image.merge('YCbCr', [out_y, out_cb, out_cr]).convert('RGB')
canvas = np.full((672, 672*2, 3), 255)
canvas[0:224, 0:224, :] = np.asarray(img)
canvas[:, 672:, :] = np.asarray(result)
plt.imshow(canvas.astype(np.uint8))
plt.show()
